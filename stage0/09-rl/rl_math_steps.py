"""第 9c 步：RL 后训练，算术题，要求模型先写过程，再写答案。

和第 9b 步 rl_math.py 的区别只有 3 处，其余（6 个环节、loss）一样：
- 题目前面先给 2 个写过程的例子（第 35～36 行），模型照着写：37×46=37×40+37×6=1480+222=1702
- 打分：用 = 分开，每一段 Python 都算一遍，必须都等于正确答案（过程每一步都对），
  最后一段必须是一个数。都对得 1 分，否则 0 分。CHECK_STEPS=False 时只看最后的答案（和 9b 一样）。
- 每个答案最多 48 个 token（9b 是 8 个），放得下过程。
考试时分开数：「答案对」几道，「过程和答案都对」几道。

不用读任何文件。
运行：cd ~/ai-infra-journey-git/stage0/09-rl && python rl_math_steps.py
"""
import math
import random
import re
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== 可以改的参数 =====================
G = 16          # 每道题写几次答案（一组）
NEW = 48        # 每个答案最多写几个 token。要写过程，所以比 9b（8 个）多
TEMP = 1.0      # 写答案时的随机程度
LR = 2e-4       # 学习率（和第 9 步一样）
STEPS = 100     # 训练几步
R = 8           # LoRA 的 r
SHOW_EVERY = 10  # 每几步打印这一组的答案
OPS = ["+", "×"]  # 出哪几种题。只要乘法：OPS = ["×"]。每步轮流出（第 9 步 BALANCE=True 的做法）
TEST_N = 50     # 训练前后，每种题考几道
CHECK_STEPS = True  # True = 过程每一步都要对；False = 只看最后的答案
# ======================================================

# 写过程的例子：放在每道题前面，模型照着这个样子写
EXAMPLES = ("25×38=25×30+25×8=750+200=950\n"
            "57+86=50+80+7+6=130+13=143\n")

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


def value(expr):
    """算一段式子，比如 "37×40+37×6" → 1702。只认数字、+、×，别的写法都算错（返回 None）。"""
    expr = expr.strip()
    if not re.fullmatch(r"\d+([+×]\d+)*", expr) or len(expr) > 30:
        return None
    return sum(math.prod(int(x) for x in term.split("×")) for term in expr.split("+"))


def reward_of(text, answer):
    """把写出来的部分用 = 分开。
    CHECK_STEPS=True：至少写了一步过程；每一段都等于正确答案；最后一段是一个数。都对得 1 分。
    CHECK_STEPS=False：只看最后一段是不是正确答案（写不写过程都行）。"""
    parts = text.strip().split("=")
    last = parts[-1].strip()
    if not (last.isdigit() and int(last) == answer):
        return 0.0
    if not CHECK_STEPS:
        return 1.0
    return 1.0 if len(parts) >= 2 and all(value(p) == answer for p in parts) else 0.0


assert value("37×40+37×6") == 1702
assert reward_of("37×40+37×6=1480+222=1702", 1702) == 1.0
assert reward_of("37×40+37×6=1480+232=1702", 1702) == 0.0   # 答案对，但是有一步错了
assert reward_of("1702", 1702) == 0.0                         # 没写过程
assert reward_of("37×40+37×6=1480+222=1712", 1702) == 0.0
assert reward_of("答案是=1702", 1702) == 0.0


def answer_right(text, answer):
    """只看最后的答案对不对（考试时另外数一下）。"""
    last = text.strip().split("=")[-1].strip()
    return last.isdigit() and int(last) == answer


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
print(f"一道题前面放的例子：\n{EXAMPLES}")


def write(prompts, n, sample, seed=0):
    """每道题写 n 个答案。sample=False：不抽签，每次选分数最高的 token（考试用）。
    返回：整段 token（题目 + 写出的部分）、题目部分有几个 token、写出来的文字。"""
    x = tok([EXAMPLES + p for p in prompts], return_tensors="pt", padding=True).to(dev)
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


def exam(when):
    print(f"\n{when}考试（每种题 {TEST_N} 道，不抽签）：")
    for op, qs in test.items():
        _, _, texts = write([q for q, _ in qs], 1, sample=False)
        ans = sum(answer_right(t, a) for t, (_, a) in zip(texts, qs))
        both = sum(reward_of(t, a) == 1 for t, (_, a) in zip(texts, qs)) if CHECK_STEPS else "—"
        print(f"  {op}：答案对 {ans} / {len(qs)}，过程和答案都对 {both} / {len(qs)}")
        for (q, a), t in zip(qs[:3], texts[:3]):
            print(f"      {q}{t}   （应为 {a}）")


exam("训练前")

# ---------- 训练：每一步 6 个环节 ----------
print("\nstep 题目        得 1 分的比例 |  时间 ms：1出题 2写答案 3打分 4猜下一个字 5算梯度 6改参数  合计 | 显存 MB")
rng = random.Random(0)
ms, scores, ops, still = [], [], [], 0
torch.cuda.reset_peak_memory_stats()
for step in range(1, STEPS + 1):
    t0 = now()
    # 环节 1：出题。加法、乘法轮流出
    op = OPS[(step - 1) % len(OPS)]
    q, answer = make(rng, op)
    t1 = now()

    # 环节 2：写过程和答案。同一道题写 G 次（这就是推理）
    out, P, texts = write([q], G, sample=True, seed=step)
    t2 = now()

    # 环节 3：打分。过程和答案都对 1 分，否则 0 分，减去这一组的平均分
    r = torch.tensor([reward_of(t, answer) for t in texts], device=dev)
    adv = r - r.mean()
    if r.std() == 0:
        still += 1                                      # 全对或全错：每个都减成 0，这一步什么也学不到
    t3 = now()

    # 环节 4：猜下一个字（和 rl_math.py 一样）
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
        print(f"      这一组的前 4 个答案：{texts[:4]}")

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
print(f"显存峰值 {torch.cuda.max_memory_allocated() / MB:.0f} MB")

exam("训练后")
