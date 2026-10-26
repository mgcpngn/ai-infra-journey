"""第 8 步：读懂一个训练程序。

和第 6 步 lora_poem.py 实验 3 是同一种训练：LoRA r=8 训练 Qwen3-0.6B-Base 写绝句。
不同的是：每一步分成 5 个环节，每个环节都量时间和显存，并打印出来。
为了短，这里不分训练集和验证集，也不生成诗。

要用到 poems.csv（peek_data.py 生成的，这个文件夹里也放了一份）。
这个文件给 M1 Mac 用。NVIDIA 显卡用 train_loop.py，两个文件只有第 8～52 行不一样。
运行：cd 到放了 poems.csv 的文件夹 && python3 train_loop_mac.py
"""
import re
import time

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== 可以改的参数 =====================
BATCH = 16     # 每一步读几首诗（lora_poem.py 第 256 行也是 16）
L = 64         # 每首诗变成几个 token（lora_poem.py 第 28 行）
LR = 2e-4      # 学习率（lora_poem.py 的最大学习率；这里不预热，一直不变）
STEPS = 20     # 训练几步
R = 8          # LoRA 的 r
LORA = True    # False = 全量微调：所有参数都改，8GB 显存可能不够
# 每个数用 fp32：4 字节。M1 对 bf16 支持不好（推测），所以不用第 6 步的 bf16
DTYPE = torch.float32
# ======================================================

MODEL = "Qwen/Qwen3-0.6B-Base"
dev = "mps"  # Mac 的 GPU。Mac 上显存和内存是同一块（这台 16 GB）
MB = 1024 ** 2
_seen = []  # mem() 每次看到的数都记在这里


def sync():
    """GPU 是自己排队干活的。先等它把手上的活做完，再看时间和显存。"""
    torch.mps.synchronize()


def mem():
    """现在显存里放了多少 MB（只算 PyTorch 放进去的东西）。"""
    sync()
    _seen.append(torch.mps.current_allocated_memory() / MB)
    return _seen[-1]


def peak_mem():
    """Mac 记不下真正的最高点，只能用 mem() 看过的几个时刻里最大的那个。
    所以这个数会比真正的最高点小一些。"""
    return max(_seen)


def now():
    sync()
    return time.time()


# ---------- 准备 A：读诗，变成 3 张数字表（和 peek_data.py 环节 3、4 一样） ----------
tok = AutoTokenizer.from_pretrained(MODEL)
df = pd.read_csv("poems.csv", dtype=str, encoding="utf-8-sig")
X = torch.full((len(df), L), tok.pad_token_id)  # token 编号
Y = torch.full((len(df), L), -100)               # 要猜的答案；-100 = 这个位置不算 loss（题目、体裁、补位）
M = torch.zeros(len(df), L, dtype=torch.long)    # 1 = 真 token，0 = 补位
for i, r in enumerate(df.itertuples()):
    s = re.findall(r"[^，。！？；]+[，。！？；]", r.诗句)
    a = tok(f"题目：{r.题目}\n体裁：{r.体裁}\n诗：").input_ids
    b = tok(s[0] + s[1] + "\n" + s[2] + s[3]).input_ids + [tok.eos_token_id]
    ids, lab = (a + b)[:L], ([-100] * len(a) + b)[:L]
    X[i, :len(ids)], Y[i, :len(ids)], M[i, :len(ids)] = torch.tensor(ids), torch.tensor(lab), 1
X, Y, M = X.to(dev), Y.to(dev), M.to(dev)
m_data = mem()
print(f"设备 {dev}，每个数 {DTYPE}")
print(f"[准备 A] 3 张数字表，每张 {tuple(X.shape)}，放进 GPU：{m_data:.0f} MB")

# ---------- 准备 B：把模型放进 GPU，加 LoRA，建 AdamW ----------
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(dev)
m_model = mem()
n_all = sum(p.numel() for p in model.parameters())
print(f"[准备 B] 模型 {n_all:,} 个参数 × {torch.finfo(DTYPE).bits // 8} 字节，放进 GPU：+{m_model - m_data:.0f} MB")
if LORA:
    model = get_peft_model(model, LoraConfig(r=R, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.config.use_cache = False
params = [p for p in model.parameters() if p.requires_grad]  # 要训练的参数：LoRA 只有小矩阵
opt = torch.optim.AdamW(params, lr=LR)
m_ready = mem()
n_train = sum(p.numel() for p in params)
print(f"[准备 B] 要训练的参数 {n_train:,} 个（占 {n_train / n_all:.3%}），加上以后：+{m_ready - m_model:.0f} MB")

# ---------- 训练：每一步 5 个环节 ----------
print("\nstep   loss    |  时间 ms：1挑数据 2猜下一个字 3算loss 4算梯度 5改参数  合计  |  显存 MB：猜完 算完loss 算完梯度 改完")
g = torch.Generator().manual_seed(0)  # 挑行号的随机数在 CPU 上，两台电脑挑到的行一样
torch.manual_seed(0)
model.train()
ms = []
for step in range(1, STEPS + 1):
    t0 = now()
    # 环节 1：挑数据。随机挑 BATCH 个行号，从 3 张表里取出这几行
    ix = torch.randint(len(X), (BATCH,), generator=g).to(dev)
    x, y, m = X[ix], Y[ix], M[ix]
    t1 = now()

    # 环节 2：猜下一个字。每个位置给出词表里每个 token 的分数（logits）
    logits = model(input_ids=x, attention_mask=m).logits
    t2 = now()
    mem2 = mem()

    # 环节 3：算 loss。第 t 个位置的分数去猜第 t+1 个 token，所以错开一位
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), y[:, 1:].flatten(), ignore_index=-100)
    t3 = now()
    mem3 = mem()

    # 环节 4：算梯度。每个要训练的参数得到一个梯度
    loss.backward()
    t4 = now()
    mem4 = mem()

    # 环节 5：改参数。AdamW 按学习率和梯度改每个要训练的参数，然后清空梯度
    opt.step()
    opt.zero_grad(set_to_none=True)
    loss_val, shape, logits_mb = loss.item(), tuple(logits.shape), logits.numel() * logits.element_size() / MB
    del logits, loss  # 这一步用完了，扔掉。不扔的话，下面量到的「改完」会多算 logits
    t5 = now()
    mem5 = mem()

    d = [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3, (t5 - t4) * 1e3]
    ms.append(d)
    print(f"{step:>4}  {loss_val:.4f}  |  {d[0]:6.1f} {d[1]:9.1f} {d[2]:8.1f} {d[3]:7.1f} {d[4]:7.1f}  {sum(d):6.1f}"
          f"  |  {mem2:6.0f} {mem3:7.0f} {mem4:7.0f} {mem5:7.0f}")
    if step == 1:
        print(f"      （logits 的大小 {shape} = {BATCH}首 × {L}个位置 × 词表里的 token 数，占 {logits_mb:.0f} MB）")

# ---------- 汇总 ----------
rest = ms[1:] if len(ms) > 1 else ms  # 第 1 步要做很多第一次的准备，比较慢，不算
avg = [sum(c) / len(rest) for c in zip(*rest)]
names = ["1 挑数据", "2 猜下一个字", "3 算 loss", "4 算梯度", "5 改参数"]
print(f"\n每个环节平均时间（不算第 1 步，{len(rest)} 步）：")
for n, a in zip(names, avg):
    print(f"  {n}：{a:7.1f} ms  {a / sum(avg):5.1%}")
print(f"  合计：{sum(avg):7.1f} ms（第 6 步 lora_poem.py 是约 370 ms）")
peak = peak_mem()
print(f"\n显存峰值 {peak:.0f} MB（第 6 步在 NVIDIA 上是 5.08 GB = {5.08 * 1024:.0f} MB）")
print(f"  其中：数字表 {m_data:.0f}、模型 {m_model - m_data:.0f}、LoRA 和 AdamW {mem5 - m_model:.0f}，"
      f"剩下的 {peak - mem5:.0f} MB 是训练一步中间临时用的")
