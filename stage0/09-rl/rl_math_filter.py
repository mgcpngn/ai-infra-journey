"""第 9f 步：先筛题，再训练。题目和打分和 9b（rl_math.py）一样，只多了「筛题」。

9b 的问题：100 步里 79 步全对或全错，什么也学不到，写答案的时间白花了。
9f 的做法：
- 训练前先筛题：出 POOL 道题，每道让模型写 G 次（和训练时一样抽签），数对了几次。
  全对（太容易）和全错（太难）的扔掉，只留有对有错的（第 118～140 行）。
- 训练时只从留下的题里挑（加法、乘法轮流）。
- 筛题也要写答案，所以单独计时，看筛题多花了多少时间。
其余（6 个环节、loss、考试）和 9b 一样。考试的题和筛题、训练的题都不一样。

不用读任何文件。
运行：cd ~/ai-infra-journey-git/stage0/09-rl && python rl_math_filter.py
"""
import random
import re
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== 可以改的参数 =====================
G = 16          # 每道题写几次答案（一组）
NEW = 8         # 每个答案最多写几个 token。两位数相乘最多 4 位数，一个数字一个 token（推测）
TEMP = 1.0      # 写答案时的随机程度
LR = 2e-4       # 学习率（和第 9 步一样）
STEPS = 100     # 训练几步
R = 8           # LoRA 的 r
SHOW_EVERY = 10  # 每几步打印这一组的答案
OPS = ["+", "×"]  # 出哪几种题。只要乘法：OPS = ["×"]。每步轮流出（第 9 步 BALANCE=True 的做法）
TEST_N = 50     # 训练前后，每种题考几道
POOL = 100      # 筛题时一共出几道（加法、乘法各一半）
# ======================================================

MODEL = "Qwen/Qwen3-0.6B-Base"
dev = "cuda"
MB = 1024 ** 2


def now():
    torch.cuda.synchronize()
    return time.time()


def mem():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / MB


# ---------- 出题和打分：检查程序 ----------
def make(rng, op):
    """出一道题：两个 10～99 的数。返回题目文字和正确答案。"""
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    return f"{a}{op}{b}=", (a + b if op == "+" else a * b)


def reward_of(text, answer):
    """模型写出来的部分，开头的第一个数等于正确答案，得 1 分；否则 0 分。"""
    m = re.match(r"\s*(-?\d+)", text)
    return 1.0 if m and int(m.group(1)) == answer else 0.0


assert reward_of("2183", 2183) == 1.0
assert reward_of(" 2183\n", 2183) == 1.0
assert reward_of("2184", 2183) == 0.0
assert reward_of("答案是 2183", 2183) == 0.0   # 开头不是数，不给分
assert reward_of("21830", 2183) == 0.0

# ---------- 准备：模型、LoRA、AdamW ----------
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"   # 考试时 100 道题一起写，短的题在左边补位，这样所有题都从同一个位置开始写
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
model = get_peft_model(model, LoraConfig(r=R, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
print(f"模型和 LoRA 放进 GPU：{mem():.0f} MB")
print(f"一道题变成几个 token：{tok.convert_ids_to_tokens(tok('37×46=').input_ids)}")


def write(prompts, n, sample, seed=0):
    """每道题写 n 个答案。sample=False：不抽签，每次选分数最高的 token（考试用）。
    返回：整段 token（题目 + 写出的部分）、题目部分有几个 token、写出来的文字。"""
    x = tok(prompts, return_tensors="pt", padding=True).to(dev)
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        kw = dict(do_sample=True, temperature=TEMP, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
        out = model.generate(x.input_ids.repeat(n, 1), attention_mask=x.attention_mask.repeat(n, 1), max_new_tokens=NEW,
                             stop_strings=["\n"], tokenizer=tok, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, **kw)
    model.train()
    P = x.input_ids.shape[1]
    texts = [tok.decode(o[P:], skip_special_tokens=True).split("\n")[0] for o in out]
    return out, P, texts


# 训练前后考同样 100 道题（加法 50 道、乘法 50 道）
test_rng = random.Random(1234)
test = {op: [make(test_rng, op) for _ in range(TEST_N)] for op in OPS}


EXAM = {}   # 记下每次考试对了几道，最后放在一起打印


def exam(when):
    print(f"\n{when}考试（每种题 {TEST_N} 道，不抽签）：")
    for op, qs in test.items():
        _, _, texts = write([q for q, _ in qs], 1, sample=False)
        right = [reward_of(t, a) for t, (_, a) in zip(texts, qs)]
        EXAM.setdefault(op, {})[when] = int(sum(right))
        print(f"  {op}：对 {int(sum(right))} / {len(qs)}。前 3 道：", end="")
        print("；".join(f"{q}{t!r}（应为 {a}）" for (q, a), t in zip(qs[:3], texts[:3])))


exam("训练前")

# ---------- 筛题：每道题写 G 次，只留有对有错的 ----------
print(f"\n筛题：出 {POOL} 道，每道写 {G} 次")
pool_rng = random.Random(99)
kept = {op: [] for op in OPS}
n_all = {op: [0, 0, 0] for op in OPS}            # 每种题：全错、有对有错、全对 各几道
t_f = now()
for k in range(POOL):
    op = OPS[k % len(OPS)]
    q, answer = make(pool_rng, op)
    _, _, texts = write([q], G, sample=True, seed=10_000 + k)
    right = int(sum(reward_of(t, answer) for t in texts))
    n_all[op][0 if right == 0 else 2 if right == G else 1] += 1
    if 0 < right < G:
        kept[op].append((q, answer))
t_filter = now() - t_f
for op in OPS:
    a, b, c = n_all[op]
    print(f"  {op}：全错 {a} 道（太难，扔掉），有对有错 {b} 道（留下），全对 {c} 道（太容易，扔掉）")
print(f"  筛题用了 {t_filter:.0f} 秒")
ops_left = [op for op in OPS if kept[op]]
if not ops_left:
    raise SystemExit("没有留下任何题：全是太难或太容易的。可以把 POOL 改大再试。")

# ---------- 训练：每一步 6 个环节 ----------
print("\nstep 题目        对的比例 |  时间 ms：1出题 2写答案 3打分 4猜下一个字 5算梯度 6改参数  合计 | 显存 MB")
rng = random.Random(0)
ms, scores, ops, still = [], [], [], 0
torch.cuda.reset_peak_memory_stats()
for step in range(1, STEPS + 1):
    t0 = now()
    # 环节 1：出题。从筛下来的题里挑，加法、乘法轮流（某一种一道也没留下，就只出另一种）
    op = ops_left[(step - 1) % len(ops_left)]
    q, answer = rng.choice(kept[op])
    t1 = now()

    # 环节 2：写答案。同一道题写 G 次（这就是推理）
    out, P, texts = write([q], G, sample=True, seed=step)
    t2 = now()

    # 环节 3：打分。每个答案对了 1 分、错了 0 分，减去这一组的平均分
    r = torch.tensor([reward_of(t, answer) for t in texts], device=dev)
    adv = r - r.mean()
    if r.std() == 0:
        still += 1                                      # 全对或全错：每个都减成 0，这一步什么也学不到
    t3 = now()

    # 环节 4：猜下一个字（和 rl_poem.py 第 120～127 行一样）
    ans = out[:, P:]
    pad = ans == tok.pad_token_id
    c = pad.cumsum(1)
    keep = (c == 0) | (pad & (c == 1))
    logits = model(input_ids=out, attention_mask=torch.ones_like(out)).logits[:, P - 1:-1].float()
    logp = logits.log_softmax(-1).gather(-1, ans.unsqueeze(-1)).squeeze(-1)
    loss = -(adv.unsqueeze(1) * logp * keep).sum() / keep.sum()
    t4 = now()

    # 环节 5：算梯度
    loss.backward()
    t5 = now()

    # 环节 6：改参数
    opt.step()
    opt.zero_grad(set_to_none=True)
    del logits, logp, loss
    t6 = now()

    d = [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3, (t5 - t4) * 1e3, (t6 - t5) * 1e3]
    ms.append(d)
    scores.append(r.mean().item())
    ops.append(op)
    print(f"{step:>4}  {q + str(answer):<12} {scores[-1]:5.0%} | {d[0]:6.1f} {d[1]:7.0f} {d[2]:5.1f} {d[3]:8.1f} {d[4]:7.1f} {d[5]:6.1f}"
          f"  {sum(d):6.0f} | {torch.cuda.max_memory_allocated() / MB:6.0f}")
    if step % SHOW_EVERY == 0:
        print(f"      这一组 {G} 个答案：{texts}")

# ---------- 汇总 ----------
half = STEPS // 2
print(f"\n训练时对的比例，按题目种类分开看（前一半 {half} 步 → 后一半 {STEPS - half} 步）：")
for o in OPS:
    a = [x for x, y in zip(scores[:half], ops[:half]) if y == o]
    b = [x for x, y in zip(scores[half:], ops[half:]) if y == o]
    if a and b:
        print(f"  {o}：{sum(a) / len(a):.0%} → {sum(b) / len(b):.0%}")
print(f"全对或全错、什么也学不到的步数：{still} / {STEPS}")
rest = ms[1:] if len(ms) > 1 else ms
avg = [sum(c) / len(rest) for c in zip(*rest)]
names = ["1 出题", "2 写答案", "3 打分", "4 猜下一个字", "5 算梯度", "6 改参数"]
print(f"\n每个环节平均时间（不算第 1 步）：")
for n, a in zip(names, avg):
    print(f"  {n}：{a:7.1f} ms  {a / sum(avg):5.1%}")
print(f"  合计：{sum(avg):7.1f} ms")
print(f"训练 {STEPS} 步一共约 {sum(map(sum, ms)) / 1000:.0f} 秒；筛题多花了 {t_filter:.0f} 秒")
print(f"显存峰值 {torch.cuda.max_memory_allocated() / MB:.0f} MB")

exam("训练后")

# 训练前、训练后的考试放在一起，不用往上翻
print("\n考试对的道数：训练前 → 训练后")
for k, v in EXAM.items():
    print(f"  {k}：{v['训练前']} → {v['训练后']}")
