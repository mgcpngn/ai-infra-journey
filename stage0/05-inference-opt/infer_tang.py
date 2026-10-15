"""第 5 步：在第 4 步训练好的小 GPT 上做推理优化（KV cache / prefill-decode / 显存 / batching / int8）。

用法：
    python infer_tang.py            # 跑全部 6 个实验，结果写到 results.md
    python infer_tang.py --exp 1 3  # 只跑指定实验
结构参数（层数、block_size 等）全部从 checkpoints/gpt_tang.pt 里读。
全程 float32、不开 autocast：有/无 cache 的矩阵形状不同，bf16 下舍入差异可能让 argmax 翻转，干扰正确性检查。
"""
import argparse
import math
import os
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7])
parser.add_argument("--ckpt", default=os.path.join("checkpoints", "gpt_tang.pt"))
cli = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False  # 保证 GPU 上是真 fp32
torch.backends.cudnn.allow_tf32 = False
# WSL 下显存超了不会报错，而是悄悄借用系统内存，速度掉几十倍；限个额，让 PyTorch 在到顶前先回收缓存
torch.cuda.set_per_process_memory_fraction(0.8)

ckpt = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
cfg = ckpt["args"]
chars = ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}
V, L, H, C, BS = len(chars), cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["block_size"]
print(f"checkpoint: {L} 层 {H} 头 {C} 维，block_size={BS}，词表 {V}")


# ---------- 模型：和 gpt_tang.py 结构一致，注意力多了一个可选的 KV cache ----------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(C, 3 * C)
        self.proj = nn.Linear(C, C)
        self.register_buffer("mask", torch.tril(torch.ones(BS, BS, dtype=torch.bool)))

    def forward(self, x, cache=None, pos=0):
        """cache=(K, V) 是预先分配好的 (B, 头, BS, hs) 缓冲区；pos 是这次输入的第一个字在序列里的位置。"""
        B, T, _ = x.shape
        hs = C // H
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, H, hs).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            K, Vc = cache
            K[:, :, pos:pos + T] = k  # 新字的 K、V 写进缓存，以后不用再算
            Vc[:, :, pos:pos + T] = v
            k, v = K[:, :, :pos + T], Vc[:, :, :pos + T]  # 注意力看「之前所有字 + 新字」
        att = q @ k.transpose(-2, -1) / math.sqrt(hs)
        att = att.masked_fill(~self.mask[pos:pos + T, :pos + T], float("-inf"))  # 无 cache 时 pos=0，和原版一样
        y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.mlp = nn.LayerNorm(C), CausalSelfAttention(), nn.LayerNorm(C), MLP()

    def forward(self, x, cache=None, pos=0):
        x = x + self.attn(self.ln1(x), cache, pos)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, C)
        self.pos_emb = nn.Embedding(BS, C)
        self.blocks = nn.ModuleList(Block() for _ in range(L))
        self.ln_f = nn.LayerNorm(C)
        self.head = nn.Linear(C, V, bias=False)

    def forward(self, idx, caches=None, pos=0, last=False):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(pos, pos + T, device=idx.device))
        for i, b in enumerate(self.blocks):
            x = b(x, None if caches is None else caches[i], pos)
        if last:  # 生成时只需要最后一个位置的分数，省掉 (B, T, 词表) 这个大张量
            x = x[:, -1:]
        return self.head(self.ln_f(x))


def load_model(device):
    m = GPT()
    m.load_state_dict(ckpt["model"])
    return m.to(device).eval()


def new_cache(B, device):
    """每层一对 (K, V)，形状 (B, 头, block_size, hs)，一次分配到最大长度。"""
    return [(torch.empty(B, H, BS, C // H, device=device), torch.empty(B, H, BS, C // H, device=device))
            for _ in range(L)]


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def generate(model, idx, use_cache, step_times=None):
    """贪心生成（temperature=0，每步取 argmax），一直生成到总长度 = block_size。
    返回 (完整序列, prefill 耗时, 每步 decode 耗时列表)。
    prefill = 第一次前向（处理整个开头），之后每次前向算一个新字。step_times 非 None 时逐步同步计时。"""
    dev = idx.device.type
    B, T0 = idx.shape
    caches = new_cache(B, idx.device) if use_cache else None
    times = []
    t = time.perf_counter()
    while idx.shape[1] < BS:
        if not use_cache:
            logits = model(idx, last=True)  # 没 cache：整段重算
        elif idx.shape[1] == T0:
            logits = model(idx, caches, 0, True)  # 有 cache 的第一步：prefill，整个开头一次过
        else:
            logits = model(idx[:, -1:], caches, idx.shape[1] - 1, True)  # 之后只喂最新那个字
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        if step_times is not None:
            sync(dev)
            now = time.perf_counter()
            times.append(now - t)
            t = now
    if step_times is not None:
        step_times.extend(times)
    return idx


def timed(fn, device):
    fn()  # 预热：cuBLAS 初始化、首次 kernel 编译等不算进去
    sync(device)
    t = time.perf_counter()
    out = fn()
    sync(device)
    return time.perf_counter() - t, out


# ---------- 输出：每个表打印到终端，同时收集到 results.md ----------
report = [f"# 小 GPT 推理优化实验结果\n\n模型：{L} 层 {H} 头 {C} 维，block_size={BS}，词表 {V}，"
          f"GPU：{torch.cuda.get_device_name(0)}，全程 float32，贪心解码。\n"]


def table(title, header, rows, note=""):
    md = [f"## {title}\n", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    if note:
        md += ["", note]
    s = "\n".join(md) + "\n"
    print("\n" + s)
    report.append(s)


def decode(t):
    return "".join(chars[i] for i in t.tolist()).replace("\n", "⏎")  # 换行会撑破表格


# 不同的开头字：取语料里最常见的字（去掉换行和标点），保证每条请求开头不同
with open(os.path.join("data", "tang.txt"), encoding="utf-8") as f:
    text = f.read()
cnt = Counter(text)
common = [c for c, _ in cnt.most_common() if c not in "\n，。、？！；：“”《》"]


def starts(B, device):
    return torch.tensor([[stoi[c]] for c in common[:B]], device=device)


gpu = load_model("cuda")
ms = lambda s: f"{s * 1000:.2f}"

# ---------- 实验 1：KV cache 正确性 ----------
if 1 in cli.exp:
    rows = []
    for s in "春月山风花":
        idx = torch.tensor([[stoi[s]]], device="cuda")
        a = generate(gpu, idx, False)[0, :201]  # 开头 1 字 + 生成 200 字
        b = generate(gpu, idx, True)[0, :201]
        diff = (a != b).nonzero()
        rows.append([s, decode(a[1:41]) + "…", decode(b[1:41]) + "…", "一字不差" if len(diff) == 0
                     else f"第 {diff[0].item()} 字起不同"])
    table("实验 1：KV cache 正确性（temperature=0，生成 200 字）",
          ["开头", "无 cache（前 40 字）", "有 cache（前 40 字）", "200 字对比"], rows)

# ---------- 实验 2：有 cache vs 没 cache 速度 ----------
if 2 in cli.exp:
    rows = []
    cpu = load_model("cpu")
    for dev, model, batches in (("cuda", gpu, (1, 8, 32, 128)), ("cpu", cpu, (1, 8))):
        for B in batches:
            res = {}
            for use_cache in (False, True):
                idx = starts(B, dev)
                total, _ = timed(lambda: generate(model, idx, use_cache), dev)
                st = []
                generate(model, idx, use_cache, st)  # 单独再跑一遍，逐步同步，拿每一步的耗时
                n = len(st)
                res[use_cache] = total
                print(f"  {dev} batch={B} cache={use_cache}: {total:.2f}s", flush=True)
                rows.append([dev.upper(), B, "有" if use_cache else "无", ms(total), ms(total / n),
                             f"{B * n / total:,.0f}", ms(st[9]), ms(st[99]), ms(st[249]),
                             f"{res[False] / total:.1f}x" if use_cache else ""])
    table(f"实验 2：有/无 KV cache 速度（每条从 1 个字生成到 {BS} 字，共 {BS - 1} 步）",
          ["设备", "batch", "cache", "总耗时 ms", "平均每步 ms", "字/秒", "第10步 ms", "第100步 ms", "第250步 ms", "加速"],
          rows, "每步耗时那三列来自单独一次逐步同步计时的运行。没 cache 时第 t 步要重算前 t 个字，"
                "所以越往后越慢；有 cache 时每步只算 1 个字，基本是平的。")

# ---------- 实验 3：prefill vs decode ----------
if 3 in cli.exp:
    val_text = text[int(0.9 * len(text)):]
    poem = next(line for line in val_text.split("\n") if len(line) >= 200)[:200]
    print(f"prompt（验证集里的唐诗，前 200 字）：{poem}")
    idx = torch.tensor([[stoi[c] for c in poem]], device="cuda")
    runs = []
    for _ in range(6):  # 第 1 次当预热丢掉，后 5 次取中位数
        st = []
        out = generate(gpu, idx, True, st)
        runs.append(st)
    runs = runs[1:]
    prefill = sorted(r[0] for r in runs)[2]
    dec = sorted(sum(r[1:]) / len(r[1:]) for r in runs)[2]
    n_dec = len(runs[0]) - 1
    table("实验 3：prefill vs decode（batch=1，有 cache，200 字唐诗当 prompt）",
          ["指标", "数值"],
          [["prefill：一次前向处理 200 字（≈ TTFT）", f"{ms(prefill)} ms"],
           [f"decode：之后每生成 1 字（≈ TPOT，平均 {n_dec} 步）", f"{ms(dec)} ms"],
           ["prefill 平均每字", f"{prefill / 200 * 1000:.3f} ms"],
           ["decode 每字 / prefill 每字", f"{dec / (prefill / 200):.0f} 倍"]],
          f"prompt 200 字 + 生成到 block_size={BS}，所以只有 {n_dec} 步 decode。"
          f"续写：{decode(out[0, 200:])}")

# ---------- 实验 4：KV cache 显存 ----------
if 4 in cli.exp:
    n_param = sum(p.numel() for p in gpu.parameters())
    param_mb = n_param * 4 / 1024**2
    rows = []
    for B in (1, 32, 128):
        formula = 2 * L * B * BS * C * 4
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        cache = new_cache(B, "cuda")
        measured = torch.cuda.memory_allocated() - before
        del cache
        rows.append([B, BS, f"{formula / 1024**2:.1f}", f"{measured / 1024**2:.1f}",
                     f"{measured / (n_param * 4):.2f}x"])
    table("实验 4：KV cache 显存（float32，每个数 4 字节）",
          ["batch", "长度", "公式 MB", "实测 MB", "相对模型参数"], rows,
          f"公式 = 2(K和V) × {L}层 × batch × 长度 × {C}(每字的数) × 4字节。"
          f"模型参数 {n_param:,} 个 = {param_mb:.1f} MB。")

# ---------- 实验 5：batching 的吞吐和延迟 ----------
if 5 in cli.exp:
    rows = []
    for B in (1, 4, 16, 64, 256):
        idx = starts(B, "cuda")
        total, _ = timed(lambda: generate(gpu, idx, True), "cuda")
        rows.append([B, f"{B * (BS - 1) / total:,.0f}", ms(total), ms(total / (BS - 1))])
    table(f"实验 5：batching 的代价（有 cache，每条生成 {BS - 1} 字）",
          ["batch", "吞吐 字/秒", "每条请求延迟 ms", "每步 ms（TPOT）"], rows,
          "同一个 batch 里所有请求一起开始、一起结束，所以每条请求的延迟 = 整个 batch 的耗时。")


# ---------- 实验 6：weight-only int8 ----------
class QLinear(nn.Module):
    """int8 权重 + 每个输出通道一个 fp32 缩放系数；前向时先还原成 fp32 再做普通矩阵乘。"""

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.data
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127
        self.register_buffer("wq", torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale)
        self.bias = lin.bias

    def forward(self, x):
        return F.linear(x, self.wq.float() * self.scale[:, None], self.bias)  # 还原这一步是额外开销


def quantize(model, cls=None):
    cls = cls or QLinear
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, child_name, cls(child))
    return model


@torch.no_grad()
def val_loss(model):
    """验证集（后 10%）按 block_size 切成不重叠的段，全部算一遍，两个模型看到的数据完全一样。"""
    data = torch.tensor([stoi[c] for c in text[int(0.9 * len(text)):]], device="cuda")
    n = (len(data) - 1) // BS
    x, y = data[:n * BS].view(n, BS), data[1:n * BS + 1].view(n, BS)
    tot = 0.0
    for i in range(0, n, 64):
        tot += F.cross_entropy(model(x[i:i + 64]).view(-1, V), y[i:i + 64].reshape(-1), reduction="sum").item()
    return tot / (n * BS)


if 6 in cli.exp:
    q = quantize(load_model("cuda"))
    q_path = os.path.join("checkpoints", "gpt_tang_int8.pt")
    torch.save({"model": q.state_dict(), "args": cfg, "chars": chars}, q_path)
    size = lambda p: f"{os.path.getsize(p) / 1024**2:.1f}"
    lf, lq = val_loss(gpu), val_loss(q)
    rows = [["文件大小 MB", size(cli.ckpt), size(q_path), ""],
            ["验证集 loss", f"{lf:.4f}", f"{lq:.4f}", f"{lq - lf:+.4f}"]]
    for B in (1, 64):
        idx = starts(B, "cuda")
        tf, _ = timed(lambda: generate(gpu, idx, True), "cuda")
        tq, _ = timed(lambda: generate(q, idx, True), "cuda")
        rows.append([f"batch={B} 生成 {BS - 1} 字 ms", ms(tf), ms(tq), f"{tq / tf:.2f}x"])
    table("实验 6：weight-only int8 量化", ["指标", "fp32", "int8", "差别"], rows,
          "所有 nn.Linear（qkv、attn.proj、mlp.fc、mlp.proj、head）都量化；embedding 和 LayerNorm 保持 fp32，"
          "所以文件不是正好 1/4。")
    rows = []
    for s in ("床前明月光", "白日依山尽", "春眠不觉晓", "千山鸟飞绝", "红豆生南国"):
        idx = torch.tensor([[stoi[c] for c in s]], device="cuda")
        a, b = (decode(generate(m, idx, True)[0]).split("⏎")[0] for m in (gpu, q))
        rows.append([s, a, b, "相同" if a == b else "不同"])
    table("实验 6：生成对比（temperature=0，到换行为止）", ["开头", "fp32", "int8", "是否相同"], rows)



# ---------- 实验 7：真正用 int8 / fp8 矩阵乘（权重和输入都量化） ----------
class W8A8Int8(nn.Module):
    """权重按输出通道、输入按行（每个字）各算缩放系数，都量化成 int8，用 torch._int_mm 做 int8×int8→int32。"""

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.data
        N = w.shape[0]
        Np = (N + 7) // 8 * 8  # _int_mm 要求输出维是 8 的倍数（head 的 8182 要补到 8184）
        w = F.pad(w, (0, 0, 0, Np - N))
        sw = w.abs().amax(dim=1).clamp(min=1e-8) / 127
        self.register_buffer("wq_t", torch.round(w / sw[:, None]).to(torch.int8).t())  # (K, Np)
        self.register_buffer("sw", sw)
        self.N, self.bias = N, lin.bias

    def forward(self, x):
        shape, M = x.shape, x.shape[:-1].numel()
        x = x.reshape(M, -1)
        sx = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
        xq = torch.round(x / sx).to(torch.int8)
        if M <= 16:  # _int_mm 要求行数 > 16，小 batch decode 时补零行
            xq = F.pad(xq, (0, 0, 0, 17 - M))
        y = torch._int_mm(xq, self.wq_t)[:M, :self.N].float() * sx * self.sw[:self.N]
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape[:-1], self.N)


class W8A8Fp8(nn.Module):
    """权重、输入都转成 fp8 e4m3（最大 448），按行缩放，用 torch._scaled_mm 在 fp8 tensor core 上算，输出 bf16。"""

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.data
        N = w.shape[0]
        Np = (N + 15) // 16 * 16  # fp8 矩阵乘要求维度是 16 的倍数
        w = F.pad(w, (0, 0, 0, Np - N))
        sw = w.abs().amax(dim=1).clamp(min=1e-8) / 448
        self.register_buffer("wq_t", (w / sw[:, None]).to(torch.float8_e4m3fn).t())
        self.register_buffer("sw", sw[None, :].contiguous())  # (1, Np)
        self.N, self.bias = N, lin.bias

    def forward(self, x):
        shape, M = x.shape, x.shape[:-1].numel()
        x = x.reshape(M, -1)
        sx = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 448
        xq = (x / sx).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(xq, self.wq_t, sx, self.sw, out_dtype=torch.bfloat16)[:, :self.N].float()
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape[:-1], self.N)


if 7 in cli.exp:
    models = {"fp32": gpu,
              "int8 只量化权重（实验 6）": quantize(load_model("cuda")),
              "int8×int8（_int_mm）": quantize(load_model("cuda"), W8A8Int8),
              "fp8×fp8（_scaled_mm）": quantize(load_model("cuda"), W8A8Fp8)}
    rows, base = [], {}
    for name, m in models.items():
        t_val, lv = timed(lambda: val_loss(m), "cuda")
        mc = torch.compile(m)  # 把量化/反量化那些逐元素小算子融合进少数几个 kernel
        t_comp, _ = timed(lambda: val_loss(mc), "cuda")
        base.setdefault("loss", lv), base.setdefault("val", t_val), base.setdefault("comp", t_comp)
        r = [name, f"{lv:.4f}", f"{lv - base['loss']:+.4f}", f"{t_val:.2f}（{base['val'] / t_val:.2f}x）",
             f"{t_comp:.2f}（{base['comp'] / t_comp:.2f}x）"]
        for B in (1, 64, 256):
            idx = starts(B, "cuda")
            t, _ = timed(lambda: generate(m, idx, True), "cuda")
            base.setdefault(B, t)
            r.append(f"{ms(t)}（{base[B] / t:.2f}x）")
        rows.append(r)
        print(f"  {name} 完成", flush=True)
    table(f"实验 7：int8 / fp8 矩阵乘（有 cache，生成 {BS - 1} 字；括号里是相对 fp32 的加速）",
          ["方法", "验证集 loss", "比 fp32", "验证集整遍前向 s", "同上 + torch.compile s", "batch=1 ms", "batch=64 ms", "batch=256 ms"], rows,
          "「验证集整遍前向」每次喂 64×256 个字，是计算量最大、最能体现 tensor core 的场景；"
          "生成时每步只喂 batch 个字，小 batch 主要卡在 kernel 发射开销。"
          "int8×int8 / fp8×fp8 每层要先量化输入（多几个小 kernel），再矩阵乘、再乘回缩放系数；"
          "不融合时这些逐元素操作的耗时超过矩阵乘省下的时间，torch.compile 融合后才显出 tensor core 的优势。"
          "生成那几列没有 compile（每步 cache 长度在变，会反复重新编译）。")
    rows = []
    for s in ("床前明月光", "白日依山尽", "春眠不觉晓", "千山鸟飞绝", "红豆生南国"):
        idx = torch.tensor([[stoi[c] for c in s]], device="cuda")
        outs = [decode(generate(m, idx, True)[0]).split("⏎")[0] for m in models.values()]
        rows.append(outs + ["全部相同" if len(set(outs)) == 1 else "有不同"])
    table("实验 7：生成对比（temperature=0，到换行为止）", list(models) + ["是否相同"], rows)

full_run = set(cli.exp) == {1, 2, 3, 4, 5, 6, 7}
with open("results.md", "w" if full_run else "a", encoding="utf-8") as f:  # 只跑部分实验时追加，不覆盖
    f.write("\n".join(report if full_run else [""] + report[1:]))
print("已写入 results.md")
