"""第 9d 步：RL 后训练，让模型调用计算器。用户 10-29 提出：模型只需要知道调用什么工具完成计算。

和第 9c 步 rl_math_steps.py 的区别：
- 模型不自己算。它只写一个算式，放在 <算> 和 </算> 中间，比如 <算>37×46</算>。
  程序看到 </算> 就停，用 Python 把算式算出来，这个数就是模型的答案（第 67～80 行）。
- 打分：Python 算出来的数等于正确答案，得 1 分，否则 0 分。所以模型只要把算式「列对」。
- 题目有两种：算式（37×46=）和应用题（每盒有 37 个苹果，有 46 盒，一共有几个？）。
  应用题要先读懂，才列得出对的算式。
- 考试另加两种训练时没有的：三位数×两位数；换了说法的应用题。
起点：Qwen3-0.6B-Base 加一个新的 LoRA。

不用读任何文件。
运行：cd ~/ai-infra-journey-git/stage0/09-rl && python rl_math_tool.py
"""
import math
import random
import re
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== 可以改的参数 =====================
G = 16          # 每道题写几次（一组）
NEW = 24        # 每次最多写几个 token。只写一个算式，比 9c（48 个）短
TEMP = 1.0      # 写的时候的随机程度
LR = 2e-4       # 学习率
STEPS = 100     # 训练几步
R = 8           # LoRA 的 r
SHOW_EVERY = 10  # 每几步打印这一组写的
TEST_N = 50     # 训练前后，每种题考几道
# ======================================================

# 调用计算器的例子：放在每道题前面，模型照着这个样子写
EXAMPLES = ("题：25×38=\n答：<算>25×38</算>\n"
            "题：班里有57个人，又来了86个人，现在有几个人？\n答：<算>57+86</算>\n")

# 训练时用的题目说法。{a}、{b} 换成两个数
TRAIN_WORDS = {
    "+": ["小明有{a}元，妈妈又给他{b}元，他现在有几元？", "仓库里有{a}箱水，又运来{b}箱，现在有几箱？"],
    "×": ["每盒有{a}个苹果，有{b}盒，一共有几个苹果？", "一本书有{a}页，{b}本这样的书一共有几页？"],
}
# 考试才用的说法，训练时一次也没见过
NEW_WORDS = {
    "+": ["树上有{a}只鸟，又飞来{b}只，现在有几只？"],
    "×": ["一辆车能坐{a}个人，{b}辆车一共能坐几个人？"],
}
KINDS = ["+ 算式", "× 算式", "+ 应用题", "× 应用题"]   # 训练时 4 种轮流出

MODEL = "Qwen/Qwen3-0.6B-Base"
dev = "cuda"
MB = 1024 ** 2


def now():
    torch.cuda.synchronize()
    return time.time()


def mem():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / MB


# ---------- 计算器和打分 ----------
def calc(text):
    """计算器：找到 <算>…</算> 中间的算式，用 Python 算出来。只认数字、+、×。找不到或看不懂，返回 None。"""
    m = re.search(r"<算>(.*?)</算>", text)
    if not m:
        return None
    expr = m.group(1).replace(" ", "")
    if not re.fullmatch(r"\d+([+×]\d+)*", expr) or len(expr) > 30:
        return None
    return sum(math.prod(int(x) for x in term.split("×")) for term in expr.split("+"))


def reward_of(text, answer):
    """计算器算出来的数等于正确答案，得 1 分；否则 0 分。"""
    return 1.0 if calc(text) == answer else 0.0


assert calc("<算>37×46</算>") == 1702
assert reward_of("<算>37×46</算>", 1702) == 1.0
assert reward_of("<算>37×64</算>", 1702) == 0.0     # 列错了，计算器算得再准也不对
assert reward_of("<算>37+46</算>", 1702) == 0.0     # 应该乘，写成了加
assert reward_of("1702", 1702) == 0.0               # 没有调用计算器
assert reward_of("<算>37×46", 1702) == 0.0          # 没写 </算>


# ---------- 出题 ----------
def make(rng, kind, big=False, words=TRAIN_WORDS):
    """出一道题。kind 是 KINDS 里的一种。返回：给模型看的题目、正确答案。"""
    op = kind[0]
    a, b = rng.randint(100, 999) if big else rng.randint(10, 99), rng.randint(10, 99)
    q = f"{a}{op}{b}=" if "算式" in kind else rng.choice(words[op]).format(a=a, b=b)
    return f"题：{q}\n答：", (a + b if op == "+" else a * b)


# ---------- 准备：模型、LoRA、AdamW ----------
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
model = get_peft_model(model, LoraConfig(r=R, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
print(f"模型和 LoRA 放进 GPU：{mem():.0f} MB")
print(f"一道题前面放的例子：\n{EXAMPLES}")


def write(prompts, n, sample, seed=0):
    """每道题写 n 次。写到 </算> 或换行就停。返回：整段 token、题目部分有几个 token、写出来的文字。"""
    x = tok([EXAMPLES + p for p in prompts], return_tensors="pt", padding=True).to(dev)
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        kw = dict(do_sample=True, temperature=TEMP, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
        out = model.generate(x.input_ids.repeat(n, 1), attention_mask=x.attention_mask.repeat(n, 1), max_new_tokens=NEW,
                             stop_strings=["</算>", "\n"], tokenizer=tok, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, **kw)
    model.train()
    P = x.input_ids.shape[1]
    texts = [tok.decode(o[P:], skip_special_tokens=True).split("\n")[0] for o in out]
    return out, P, texts


def show(prompt, text, answer):
    q = prompt.replace("题：", "").replace("\n答：", "")
    return f"{q}  写了 {text!r} → 计算器 {calc(text)}（应为 {answer}）"


# 训练前后考同样的题。最后两种训练时没有：三位数×两位数；换了说法的应用题
test_rng = random.Random(1234)
test = {k: [make(test_rng, k) for _ in range(TEST_N)] for k in KINDS}
test["三位数×两位数（没练过）"] = [make(test_rng, "× 算式", big=True) for _ in range(TEST_N)]
test["换了说法的应用题（没见过）"] = [make(test_rng, k, words=NEW_WORDS) for k in ("+ 应用题", "× 应用题") for _ in range(TEST_N // 2)]


def exam(when):
    print(f"\n{when}考试（每种题 {TEST_N} 道，不抽签）：")
    for kind, qs in test.items():
        _, _, texts = write([q for q, _ in qs], 1, sample=False)
        right = sum(reward_of(t, a) for t, (_, a) in zip(texts, qs))
        print(f"  {kind}：对 {int(right)} / {len(qs)}")
        for (q, a), t in zip(qs[:2], texts[:2]):
            print(f"      {show(q, t, a)}")


exam("训练前")

# ---------- 训练：每一步 6 个环节 ----------
print("\nstep 题目种类   得 1 分的比例 |  时间 ms：1出题 2写算式 3算+打分 4猜下一个字 5算梯度 6改参数  合计 | 显存 MB")
rng = random.Random(0)
ms, scores, kinds, still = [], [], [], 0
torch.cuda.reset_peak_memory_stats()
for step in range(1, STEPS + 1):
    t0 = now()
    # 环节 1：出题。4 种轮流出
    kind = KINDS[(step - 1) % len(KINDS)]
    q, answer = make(rng, kind)
    t1 = now()

    # 环节 2：写算式。同一道题写 G 次（这就是推理）
    out, P, texts = write([q], G, sample=True, seed=step)
    t2 = now()

    # 环节 3：计算器算，打分，减去这一组的平均分
    r = torch.tensor([reward_of(t, answer) for t in texts], device=dev)
    adv = r - r.mean()
    if r.std() == 0:
        still += 1
    t3 = now()

    # 环节 4：猜下一个字（和 rl_math.py 一样）。只算模型自己写的 token，计算器的结果不放进来
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
    kinds.append(kind)
    print(f"{step:>4}  {kind:<8} {scores[-1]:5.0%} | {d[0]:6.1f} {d[1]:7.0f} {d[2]:5.1f} {d[3]:8.1f} {d[4]:7.1f} {d[5]:6.1f}"
          f"  {sum(d):6.0f} | {torch.cuda.max_memory_allocated() / MB:6.0f}")
    if step % SHOW_EVERY == 0:
        print(f"      {show(q, texts[0], answer)}")

# ---------- 汇总 ----------
half = STEPS // 2
print(f"\n训练时得 1 分的比例，按题目种类分开看（前一半 {half} 步 → 后一半 {STEPS - half} 步）：")
for k in KINDS:
    a = [x for x, y in zip(scores[:half], kinds[:half]) if y == k]
    b = [x for x, y in zip(scores[half:], kinds[half:]) if y == k]
    if a and b:
        print(f"  {k}：{sum(a) / len(a):.0%} → {sum(b) / len(b):.0%}")
print(f"全对或全错、什么也学不到的步数：{still} / {STEPS}")
rest = ms[1:] if len(ms) > 1 else ms
avg = [sum(c) / len(rest) for c in zip(*rest)]
names = ["1 出题", "2 写算式", "3 算 + 打分", "4 猜下一个字", "5 算梯度", "6 改参数"]
print(f"\n每个环节平均时间（不算第 1 步）：")
for n, a in zip(names, avg):
    print(f"  {n}：{a:7.1f} ms  {a / sum(avg):5.1%}")
print(f"  合计：{sum(avg):7.1f} ms")
print(f"显存峰值 {torch.cuda.max_memory_allocated() / MB:.0f} MB")

exam("训练后")
