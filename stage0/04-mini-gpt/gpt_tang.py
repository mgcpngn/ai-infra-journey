"""从零写的小 GPT（参考 nanoGPT 结构），用唐诗训练，和 bigram 对比。

用法：
    python gpt_tang.py                          # 默认 6 层 6 头 384 维，看 128 个字，训练 5000 步
    python gpt_tang.py --n-layer 4 --n-embd 256 # 改小模型
    python gpt_tang.py --temperature 1.0        # 生成更随机
需要先跑过 bigram_tang.py 生成 data/tang.txt。
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 1. 命令行参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--n-layer", type=int, default=6, help="Transformer 块的个数（层数）")
parser.add_argument("--n-head", type=int, default=6, help="每层注意力的头数")
parser.add_argument("--n-embd", type=int, default=384, help="每个字用多少个数表示（必须能被头数整除）")
parser.add_argument("--block-size", type=int, default=128, help="一次最多看多少个字（上下文长度）")
parser.add_argument("--dropout", type=float, default=0.1, help="训练时随机丢弃的比例，防止死记硬背")
parser.add_argument("--batch-size", type=int, default=64, help="每步同时看多少段文本")
parser.add_argument("--steps", type=int, default=5000, help="训练步数")
parser.add_argument("--lr", type=float, default=1e-3, help="最大学习率")
parser.add_argument("--temperature", type=float, default=0.8, help="采样温度：<1 更保守，>1 更随机")
args = parser.parse_args()
assert args.n_embd % args.n_head == 0, "n_embd 必须能被 n_head 整除"

torch.manual_seed(0)
device = "cuda"
BIGRAM_VAL = 6.44  # 第 3 步 bigram 的验证 loss


# ---------- 2. 数据和词表：和 bigram 完全一样 ----------
with open(os.path.join("data", "tang.txt"), encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
V = len(chars)
data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"词表大小: {V:,}，训练集 {split:,} 字，验证集 {len(data) - split:,} 字")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block_size - 1, (args.batch_size,), device=device)
    offs = torch.arange(args.block_size, device=device)
    return d[ix[:, None] + offs], d[ix[:, None] + offs + 1]


# ---------- 3. 模型 ----------
class CausalSelfAttention(nn.Module):
    """多头因果自注意力：每个字去「看」它前面所有字，按相关程度加权汇总信息。"""

    def __init__(self):
        super().__init__()
        self.n_head = args.n_head
        self.qkv = nn.Linear(args.n_embd, 3 * args.n_embd)  # 一次算出 Q、K、V
        self.proj = nn.Linear(args.n_embd, args.n_embd)     # 各头结果拼起来后再混合一下
        self.drop = nn.Dropout(args.dropout)
        # 下三角掩码：第 t 个字只能看第 0..t 个字，不能偷看后面的答案
        self.register_buffer("mask", torch.tril(torch.ones(args.block_size, args.block_size, dtype=torch.bool)))

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head
        # 拆成多个头：每个头在自己的 hs 维小空间里独立算注意力，可以各自关注不同的关系（押韵、对仗、语法…）
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, hs).transpose(1, 2) for t in (q, k, v))  # (B, 头, T, hs)
        att = q @ k.transpose(-2, -1) / math.sqrt(hs)          # (B, 头, T, T)：每对字的相关分数
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))  # 未来位置设成 -inf，softmax 后概率为 0
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).reshape(B, T, C)          # 按分数加权求和，再把各头拼回去
        return self.drop(self.proj(y))


class MLP(nn.Module):
    """每个字各自过一个两层小网络（先放大 4 倍再缩回），对注意力收集来的信息做加工。"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(args.n_embd, 4 * args.n_embd)
        self.proj = nn.Linear(4 * args.n_embd, args.n_embd)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.mlp = MLP()

    def forward(self, x):
        # 残差连接：x + f(x)，每层只在原有表示上「加点修改」，深层网络才好训练
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, args.n_embd)               # 字是什么
        self.pos_emb = nn.Embedding(args.block_size, args.n_embd)  # 字在第几个位置
        self.drop = nn.Dropout(args.dropout)
        self.blocks = nn.ModuleList(Block() for _ in range(args.n_layer))
        self.ln_f = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, V, bias=False)          # 384 个数 -> 词表里每个字的分数
        self.apply(self._init)

    @staticmethod
    def _init(m):  # nanoGPT 的初始化：权重小一点的正态分布，训练初期更稳
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, V).float(), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0, stop=None):
        """没有 KV cache：每生成一个字，都把（最多 block_size 个）前文整个重新算一遍。"""
        for _ in range(n):
            logits, _ = self(idx[:, -args.block_size:])  # 超过上下文长度就只保留最后 block_size 个字
            probs = F.softmax(logits[:, -1, :].float() / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            if stop is not None and nxt.item() == stop:
                break
            idx = torch.cat([idx, nxt], dim=1)
        return idx


model = GPT().to(device)


# ---------- 4. 参数量 ----------
# LayerNorm 也有少量参数（每个 2*n_embd），归到它后面紧跟的那一部分
groups = {"embedding（字+位置）": 0, "注意力": 0, "MLP": 0, "输出层（ln_f+head）": 0}
for name, p in model.named_parameters():
    if "emb" in name:
        groups["embedding（字+位置）"] += p.numel()
    elif "attn" in name or "ln1" in name:
        groups["注意力"] += p.numel()
    elif "mlp" in name or "ln2" in name:
        groups["MLP"] += p.numel()
    else:
        groups["输出层（ln_f+head）"] += p.numel()
total = sum(groups.values())
print(f"\n参数总数: {total:,}（{total * 4 / 1024**2:.1f} MB float32）")
for k, n in groups.items():
    print(f"  {k:<16}{n:>12,}  {n / total:6.1%}")


# ---------- 5. 训练 ----------
@torch.no_grad()
def estimate_loss(n=50):
    model.eval()  # 关掉 dropout
    out = {}
    for s in ("train", "val"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[s] = sum(model(*get_batch(s))[1].item() for _ in range(n)) / n
    model.train()
    return out


def lr_at(step):
    """先用 100 步从 0 升到最大学习率（预热），再按余弦曲线降到 1/10。"""
    warmup = 100
    if step < warmup:
        return args.lr * step / warmup
    r = (step - warmup) / max(1, args.steps - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * r)))


optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1)
print(f"\n未训练 loss: {estimate_loss(10)['val']:.4f}（瞎猜 ln{V} = {math.log(V):.4f}）")
every = max(1, args.steps // 10)
tokens_per_step = args.batch_size * args.block_size
t_last, step_last = time.time(), 0
for step in range(1, args.steps + 1):
    for g in optimizer.param_groups:
        g["lr"] = lr_at(step)
    # bf16 混合精度：矩阵乘法用 16 位算（快、省显存），参数和梯度仍是 32 位（精度不丢）
    # bf16 指数位和 float32 一样多，不会溢出，所以不需要 GradScaler
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(*get_batch("train"))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度太大时按比例缩小，防止偶发的一步把模型带崩
    optimizer.step()
    if step % every == 0 or step == args.steps:
        torch.cuda.synchronize()
        tps = (step - step_last) * tokens_per_step / (time.time() - t_last)  # 只算训练时间，不含下面的评估
        l = estimate_loss()
        print(f"step {step:5d}  训练 loss={l['train']:.4f}  验证 loss={l['val']:.4f}  "
              f"(bigram {BIGRAM_VAL})  {tps:,.0f} token/s")
        t_last, step_last = time.time(), step

model.eval()


# ---------- 6. 生成 ----------
def encode(s):
    return torch.tensor([[stoi[c] for c in s]], device=device)


print(f"\n生成（温度 {args.temperature}）:")
for s in "春月山风花":
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(encode(s), 100, args.temperature, stop=stoi["\n"])
    print("  " + "".join(itos[i] for i in out[0].tolist()))


# ---------- 7. 生成速度（无 KV cache） ----------
# 没有 KV cache 时，第 t 个字要把前面 min(t, block_size) 个字整个重算，所以越往后每个字越慢，
# 超过 block_size 后因为截断，每个字的耗时会稳定下来
print("\n生成速度（无 KV cache）:")
for n in (50, 200, 500):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.generate(encode("春"), n, args.temperature)  # 不在换行处停，保证正好生成 n 个字
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  {n:4d} 个字：总耗时 {dt * 1000:8.1f} ms，平均每字 {dt / n * 1000:.2f} ms")


# ---------- 8. 保存 ----------
# 存权重 + 结构参数 + 词表，下一步加载时不用重新训练、也不用重新建词表
os.makedirs("checkpoints", exist_ok=True)
path = os.path.join("checkpoints", "gpt_tang.pt")
torch.save({"model": model.state_dict(), "args": vars(args), "chars": chars}, path)
print(f"\n模型已保存到 {path}")
