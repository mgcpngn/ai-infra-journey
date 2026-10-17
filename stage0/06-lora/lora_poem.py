"""用 LoRA 微调 Qwen3-0.6B-Base 写唐诗绝句。

实验 0：check_format 检查格式
实验 1：基线（不微调）生成 + 验证 loss
实验 2：全量微调 5 步，看显存够不够
实验 3：LoRA r=8 训练 1000 步，前后对比，存小表
实验 4：r=2 / r=32 各训练 300 步，和 r=8 的前 300 步比较
所有表汇总到 results_lora.md。

下载慢时：HF_ENDPOINT=https://hf-mirror.com python lora_poem.py
"""
import gc
import glob
import json
import math
import os
import random
import re
import time

import torch
import zhconv
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B-Base"
L = 64  # 序列长度
dev = "cuda"
GB = 1024 ** 3
md = []  # 汇总到 results_lora.md 的内容


def save_md():
    """每加一张表就重写一次，中途崩了前面的结果也在。"""
    with open("results_lora.md", "w", encoding="utf-8") as f:
        f.write("# Qwen3-0.6B-Base LoRA 写绝句实验结果\n\n" + "\n".join(md))


def table(title, header, rows):
    """打印一张 markdown 表，同时记进 md。"""
    lines = [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    s = "\n".join(lines) + "\n"
    print("\n" + s)
    md.append(s)
    save_md()


def note(s):
    print(s)
    md.append(s + "\n")
    save_md()


# ---------- 0. 格式检查 ----------
def check_format(poem, genre):
    """4 句、每句字数对（五言 5 字、七言 7 字，只算汉字）才算格式正确。句子按标点或换行切开。"""
    n = 5 if genre == "五言绝句" else 7
    s = [x for x in re.split(r"[，。！？；、,.!?;\s]+", poem) if x]
    return len(s) == 4 and all(len(x) == n and re.fullmatch(r"[一-鿿]+", x) for x in s)


assert check_format("床前明月光，疑是地上霜。\n举头望明月，低头思故乡。", "五言绝句")
assert not check_format("床前明月光，疑是地上霜。\n举头望明月，低头思故乡。", "七言绝句")
assert not check_format("床前明月光，疑是地上霜。", "五言绝句")
assert not check_format("床前明月光，疑是地上霜。举头望明月，低头思故乡。日暮苍山远，天寒白屋贫。", "五言绝句")
assert not check_format("床前明月光，疑是地上霜。举头望明月，低头abc乡。", "五言绝句")


# ---------- 数据 ----------
def load_quatrains():
    """全唐诗原始 json 里挑绝句：正好 4 句、每句都 5 字或都 7 字。缺字过滤、繁转简和 bigram_tang.py 一致。"""
    out = []
    for f in sorted(glob.glob(os.path.join("data", "tang_raw", "poet.tang.*.json"))):
        for p in json.load(open(f, encoding="utf-8")):
            t = "".join(p["paragraphs"]).strip()
            if not t or any(c in t for c in "□{}[]（）()"):
                continue
            s = re.findall(r"[^，。！？；]+[，。！？；]", t)
            if "".join(s) != t or len(s) != 4 or len({len(x) for x in s}) != 1 or len(s[0]) - 1 not in (5, 7):
                continue
            t = zhconv.convert(t, "zh-cn")
            s = re.findall(r"[^，。！？；]+[，。！？；]", t)
            genre = "五言绝句" if len(s[0]) - 1 == 5 else "七言绝句"
            out.append((zhconv.convert(p["title"], "zh-cn").strip(), genre, s[0] + s[1] + "\n" + s[2] + s[3]))
    return out


def prompt_of(title, genre):
    return f"题目：{title}\n体裁：{genre}\n诗："


note(f"GPU：{torch.cuda.get_device_name(0)}，模型 {MODEL}，bf16。")
tok = AutoTokenizer.from_pretrained(MODEL)
poems = load_quatrains()
random.Random(0).shuffle(poems)
train_poems, val_poems = poems[:-500], poems[-500:]
n5 = sum(g == "五言绝句" for _, g, _ in poems)
note(f"挑出绝句 {len(poems):,} 首（五言 {n5:,}，七言 {len(poems) - n5:,}），训练 {len(train_poems):,}，验证 {len(val_poems)}")
print("样本示例：\n" + prompt_of(*poems[0][:2]) + poems[0][2])


def encode(ps):
    """每首诗变成长度 L 的一行：input_ids、attention_mask、labels（题目/体裁部分和 padding 的 label 是 -100，不算 loss）。"""
    X = torch.full((len(ps), L), tok.pad_token_id)
    M = torch.zeros(len(ps), L, dtype=torch.long)
    Y = torch.full((len(ps), L), -100)
    cut = 0
    for i, (title, genre, poem) in enumerate(ps):
        a = tok(prompt_of(title, genre)).input_ids
        b = tok(poem).input_ids + [tok.eos_token_id]  # 结尾加 eos，模型学会写完就停
        ids, lab = (a + b)[:L], ([-100] * len(a) + b)[:L]
        cut += len(a) + len(b) > L
        X[i, :len(ids)], M[i, :len(ids)], Y[i, :len(ids)] = torch.tensor(ids), 1, torch.tensor(lab)
    return X.to(dev), M.to(dev), Y.to(dev), cut


Xtr, Mtr, Ytr, cut_tr = encode(train_poems)
Xva, Mva, Yva, cut_va = encode(val_poems)
note(f"超过 {L} token 被截断：训练 {cut_tr} 首，验证 {cut_va} 首")
# 验证集里固定 10 个题目：5 首五言 + 5 首七言
eval_items = [p for p in val_poems if p[1] == "五言绝句"][:5] + [p for p in val_poems if p[1] == "七言绝句"][:5]


def load_model():
    return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)


def free():
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def val_loss(model, bs=16):
    """验证集 500 首，只算「诗：」后面那部分 token 的平均 loss。"""
    model.eval()
    tot = cnt = 0
    for i in range(0, len(Xva), bs):
        y = Yva[i:i + bs]
        n = (y[:, 1:] != -100).sum().item()  # HF 内部会错一位：用第 t 个 token 预测第 t+1 个
        tot += model(input_ids=Xva[i:i + bs], attention_mask=Mva[i:i + bs], labels=y).loss.item() * n
        cnt += n
    model.train()
    return tot / cnt


@torch.no_grad()
def generate_all(model):
    """10 个题目各生成 1 首：temperature=0.8，遇到 eos、空行或 64 个新 token 就停。每首都用同一个随机种子，前后可比。"""
    model.eval()
    out = []
    for title, genre, _ in eval_items:
        ids = tok(prompt_of(title, genre), return_tensors="pt").input_ids.to(dev)
        torch.manual_seed(1234)
        g = model.generate(ids, do_sample=True, temperature=0.8, top_k=0, top_p=1.0, max_new_tokens=64,
                           stop_strings=["\n\n"], tokenizer=tok,
                           eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        poem = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).split("\n\n")[0].strip()
        out.append((poem, check_format(poem, genre)))
    model.train()
    return out


def show(poem, n=10 ** 9):
    s = poem.replace("\n", " / ").replace("|", "｜")
    return s if len(s) <= n else s[:n] + "…"


def acc(gens):
    return sum(ok for _, ok in gens) / len(gens)


# ---------- 1. 基线 ----------
print("\n========== 实验 1：基线 ==========")
model = load_model()
n_params = sum(p.numel() for p in model.parameters())
base_gens = generate_all(model)
base_val = val_loss(model)
table("实验 1：基线（不微调）", ["题目", "体裁", "生成的诗", "格式"],
      [(t, g, show(p), "✓" if ok else "✗") for (t, g, _), (p, ok) in zip(eval_items, base_gens)])
note(f"基线格式正确率 {acc(base_gens):.0%}，验证 loss {base_val:.4f}")
del model
free()


# ---------- 2. 全量微调 ----------
def full_ft(steps=5, bs=4):
    model = load_model()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(device=dev).manual_seed(0)
    status, ms = "成功", []
    try:
        for step in range(steps):
            torch.cuda.synchronize()
            t0 = time.time()
            ix = torch.randint(len(Xtr), (bs,), device=dev, generator=g)
            loss = model(input_ids=Xtr[ix], attention_mask=Mtr[ix], labels=Ytr[ix]).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            ms.append((time.time() - t0) * 1000)
            print(f"全量微调 step {step + 1}  loss={loss.item():.4f}  {ms[-1]:.0f} ms  显存峰值 {torch.cuda.max_memory_allocated() / GB:.2f} GB")
    except torch.OutOfMemoryError:
        status = "全量微调 OOM"
        print(f"全量微调 OOM，报错前显存用了 {torch.cuda.max_memory_allocated() / GB:.2f} GB")
    return status, len(ms), (sum(ms[1:]) / len(ms[1:]) if len(ms) > 1 else float("nan")), torch.cuda.max_memory_allocated() / GB


print("\n========== 实验 2：全量微调 ==========")
status, done, step_ms, peak = full_ft()
free()
est = n_params * (2 + 2 + 4 + 4 + 4) / GB
table("实验 2：全量微调 5 步（batch=4，长度 64，AdamW）", ["项目", "数值"], [
    ("结果", status),
    ("完成步数", f"{done}/5"),
    ("每步耗时（不算第 1 步）", f"{step_ms:.0f} ms"),
    ("实测显存峰值（成功时）/ 报错前显存（OOM 时）", f"{peak:.2f} GB"),
    ("模型参数", f"{n_params / 1e6:.0f} M"),
    ("估算：参数 2 + 梯度 2 + AdamW 两状态 4+4 + fp32 主参数 4 = 16 字节/参数", f"{est:.2f} GB（还不含激活值）"),
    ("显卡总显存", f"{torch.cuda.get_device_properties(0).total_memory / GB:.2f} GB"),
])
note("注：这里实测用的是纯 bf16 训练，PyTorch 的 AdamW 状态跟着参数也是 bf16，没有 fp32 主参数，"
     "实际约 2+2+2+2=8 字节/参数，所以实测会比 16 字节的估算小一半左右；正规的混合精度训练才是 16 字节。")


# ---------- 3 / 4. LoRA ----------
def lr_at(step, total=1000, warmup=50, lr=2e-4):
    if step <= warmup:
        return lr * step / warmup
    return lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def train_lora(r, steps, gen_at=()):
    """训练 steps 步。学习率曲线总是按 1000 步算，所以 r=2/32 跑 300 步和 r=8 的前 300 步完全同一条曲线。"""
    torch.manual_seed(0)
    model = get_peft_model(load_model(), LoraConfig(
        r=r, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.config.use_cache = False
    trainable, total = model.get_nb_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    g = torch.Generator(device=dev).manual_seed(0)  # 取 batch 用独立的随机数，三个 r 看到的数据顺序一样
    torch.cuda.reset_peak_memory_stats()
    rows, gens, vals, losses, t_train = [], {}, {}, [], 0.0
    model.train()
    for step in range(1, steps + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        ix = torch.randint(len(Xtr), (16,), device=dev, generator=g)
        loss = model(input_ids=Xtr[ix], attention_mask=Mtr[ix], labels=Ytr[ix]).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())  # .item() 会等 GPU 算完
        t_train += time.time() - t0
        if step % 100 == 0:
            vals[step] = val_loss(model)
            rows.append((step, f"{sum(losses[-100:]) / 100:.4f}", f"{vals[step]:.4f}",
                         f"{t_train * 10:.0f}", f"{torch.cuda.max_memory_allocated() / GB:.2f}"))
            print(f"r={r} step {step}  训练 loss={rows[-1][1]}  验证 loss={rows[-1][2]}  {rows[-1][3]} ms/步  显存峰值 {rows[-1][4]} GB")
            t_train = 0.0
        if step in gen_at:
            gens[step] = generate_all(model)
    return model, trainable, total, rows, vals, gens


print("\n========== 实验 3：LoRA r=8 ==========")
model, trainable, total, rows, vals8, gens8 = train_lora(8, 1000, gen_at=(300, 1000))
table("实验 3a：LoRA 参数量（r=8，q/k/v/o_proj）", ["要训练的参数", "模型总参数", "占比"],
      [(f"{trainable:,}", f"{total:,}", f"{trainable / total:.3%}")])
table("实验 3b：LoRA r=8 训练过程（batch=16，长度 64，lr=2e-4，预热 50 步 + 余弦）",
      ["step", "训练 loss（近 100 步平均）", "验证 loss", "每步耗时 ms", "显存峰值 GB"], rows)
table("实验 3c：同样 10 个题目，基线 vs LoRA 1000 步", ["题目", "体裁", "基线", "格式", "LoRA", "格式"],
      [(t, g, show(bp, 30), "✓" if bok else "✗", show(lp, 40), "✓" if lok else "✗")
       for (t, g, _), (bp, bok), (lp, lok) in zip(eval_items, base_gens, gens8[1000])])
note(f"格式正确率：基线 {acc(base_gens):.0%} → LoRA {acc(gens8[1000]):.0%}；验证 loss：基线 {base_val:.4f} → LoRA {vals8[1000]:.4f}")

out_dir = os.path.join("checkpoints", "qwen3_0.6b_lora_r8")
model.save_pretrained(out_dir)
lora_size = os.path.getsize(os.path.join(out_dir, "adapter_model.safetensors"))
base_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(snapshot_download(MODEL, local_files_only=True), "*.safetensors")))
table("实验 3d：文件大小", ["文件", "大小", "倍数"], [
    (f"LoRA 小表 {out_dir}/adapter_model.safetensors", f"{lora_size / 1024 ** 2:.2f} MB", "1×"),
    ("原模型 model.safetensors（bf16）", f"{base_size / 1024 ** 2:.0f} MB", f"{base_size / lora_size:.0f}×"),
])
del model
free()

print("\n========== 实验 4：换 r ==========")
cmp = {8: (trainable, vals8[300], acc(gens8[300]))}
for r in (2, 32):
    model, tr, _, _, vals, gens = train_lora(r, 300, gen_at=(300,))
    cmp[r] = (tr, vals[300], acc(gens[300]))
    del model
    free()
table("实验 4：不同 r 训练 300 步对比（lora_alpha 都是 16，学习率曲线相同）",
      ["r", "要训练的参数", "验证 loss（300 步）", "格式正确率（300 步）"],
      [(r, f"{cmp[r][0]:,}", f"{cmp[r][1]:.4f}", f"{cmp[r][2]:.0%}") for r in (2, 8, 32)])

print("已写入 results_lora.md")
