"""最简单的语言模型（bigram）：只看前一个字，猜下一个字。用唐诗训练。

用法：
    python bigram_tang.py                      # 默认训练 5000 步，温度 1.0
    python bigram_tang.py --steps 20000        # 多训练一些
    python bigram_tang.py --temperature 0.5    # 采样更保守（更爱挑高概率的字）
    python bigram_tang.py --temperature 1.5    # 采样更随机（更敢挑冷门字）
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 1. 命令行参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=5000, help="训练多少步（每步更新一次参数）")
parser.add_argument("--temperature", type=float, default=1.0, help="采样温度：<1 更保守，>1 更随机")
parser.add_argument("--batch-size", type=int, default=64, help="每步同时看多少段文本")
parser.add_argument("--block-size", type=int, default=32, help="每段文本多少个字")
parser.add_argument("--lr", type=float, default=0.1, help="学习率")
args = parser.parse_args()

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {device}")


# ---------- 2. 准备数据 ----------
# chinese-poetry 项目把全唐诗拆成 58 个 json 文件（poet.tang.0.json ... poet.tang.57000.json），繁体。
# 每个文件是一个列表，每首诗形如 {"title": ..., "author": ..., "paragraphs": ["句1，句2。", ...]}
DATA = os.path.join("data", "tang.txt")
RAW = os.path.join("data", "tang_raw")  # 原始 json 缓存
URL = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/" + urllib.parse.quote("全唐诗") + "/poet.tang.{}.json"


def download():
    import zhconv  # 繁转简，只在第一次下载时用到：pip install zhconv
    lines = []
    os.makedirs(RAW, exist_ok=True)
    for k in range(0, 58000, 1000):
        path = os.path.join(RAW, f"poet.tang.{k}.json")
        if not os.path.exists(path):  # 已下载的跳过，断了重跑能接着下
            print(f"\r下载 poet.tang.{k}.json ...", end="", flush=True)
            for attempt in range(5):  # 代理偶尔中途断开，重试几次
                try:
                    urllib.request.urlretrieve(URL.format(k), path + ".part")
                    os.replace(path + ".part", path)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(3)
        with open(path, encoding="utf-8") as f:
            poems = json.load(f)
        for p in poems:
            text = "".join(p["paragraphs"]).strip()
            # 原始数据里有些缺字用 □ 或 {…} 标注，这种诗直接丢掉，免得污染词表
            if text and not any(c in text for c in "□{}[]（）()"):
                lines.append(zhconv.convert(text, "zh-cn"))
    print()
    os.makedirs("data", exist_ok=True)
    with open(DATA, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if not os.path.exists(DATA):
    try:
        download()
    except Exception as e:
        sys.exit(f"下载失败：{e}\n请手动把唐诗放到 {DATA}（UTF-8，简体，每首诗一行）后重新运行。")

with open(DATA, encoding="utf-8") as f:
    text = f.read()
# 换行符也当成一个「字」：它表示「一首诗结束了」，模型也要学会什么时候收尾
print(f"诗的数量: {text.count(chr(10)) + 1:,}，总字数: {len(text) - text.count(chr(10)):,}")


# ---------- 3. 词表 ----------
# 神经网络只认数字，所以给每个出现过的字编个号：字→编号 用来把文本变成数字，编号→字 用来把输出变回文字
counts = Counter(text)
chars = sorted(counts)
stoi = {c: i for i, c in enumerate(chars)}  # 字 -> 编号
itos = {i: c for i, c in enumerate(chars)}  # 编号 -> 字
V = len(chars)
print(f"词表大小: {V:,}（含换行符）")
print("出现最多的 20 个字:", " ".join(f"{'换行' if c == chr(10) else c}{n}" for c, n in counts.most_common(20)))

data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
# 前 90% 训练，后 10% 验证：验证集模型从没见过，用来判断它是学到了规律还是死记硬背
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


# ---------- 4. 模型 ----------
# 整个模型就是一张 V x V 的表：第 i 行 = 「前一个字是 i 时，下一个字分别是 0..V-1 的分数（logits）」
# 分数经过 softmax 变成概率。训练就是在调这张表里的数。
model = nn.Embedding(V, V).to(device)
n_params = model.weight.numel()
print(f"参数量: {V:,} x {V:,} = {n_params:,}，占 {n_params * 4 / 1024**2:.1f} MB（float32 每个数 4 字节）")


def get_batch(split):
    """随机抽 batch_size 段连续文本。输入 x 是每个字，标签 y 是它后面那个字（整体右移一位）。"""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block_size - 1, (args.batch_size,), device=device)
    offs = torch.arange(args.block_size, device=device)
    x = d[ix[:, None] + offs]
    y = d[ix[:, None] + offs + 1]
    return x, y


def loss_of(x, y):
    logits = model(x)  # (batch, block, V)：查表，每个位置拿到「下一个字」的 V 个分数
    return F.cross_entropy(logits.view(-1, V), y.view(-1))


@torch.no_grad()
def estimate_loss(n=50):
    """抽 n 个 batch 取平均，比单个 batch 的 loss 稳定。"""
    return {s: sum(loss_of(*get_batch(s)).item() for _ in range(n)) / n for s in ("train", "val")}


@torch.no_grad()
def generate(start, max_len=64):
    """从开头的字出发，按概率一个字一个字往下抽，抽到换行（诗结束）或够长就停。"""
    idx, out = stoi[start], start
    for _ in range(max_len):
        # 温度：分数先除以 T 再 softmax。T<1 把差距拉大（高概率字更占优），T>1 把差距抹平（更随机）
        probs = F.softmax(model.weight[idx] / args.temperature, dim=0)
        idx = torch.multinomial(probs, 1).item()
        if itos[idx] == "\n":
            break
        out += itos[idx]
    return out


STARTS = "春月山风花"


def show_poems(title):
    print(f"\n{title}（温度 {args.temperature}）:")
    for s in STARTS:
        print("  " + generate(s))


# ---------- 5. 训练前 ----------
# 完全瞎猜 = 每个字概率都是 1/V，交叉熵 = -ln(1/V) = ln V
print(f"\n完全瞎猜的 loss: ln({V}) = {math.log(V):.4f}")
# 未训练的表是随机数（标准正态），分数有高有低但毫无道理，所以通常比均匀瞎猜还差一点
print(f"未训练模型的 loss: {estimate_loss()['val']:.4f}")
show_poems("训练前生成")


# ---------- 6. 训练 ----------
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
every = max(1, args.steps // 10)
print()
t0 = time.time()
for step in range(1, args.steps + 1):
    loss = loss_of(*get_batch("train"))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % every == 0 or step == args.steps:
        l = estimate_loss()
        print(f"step {step:6d}  训练 loss={l['train']:.4f}  验证 loss={l['val']:.4f}  ({time.time() - t0:.0f}s)")

show_poems("训练后生成")
