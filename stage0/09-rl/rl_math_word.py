"""第 9e 步：难一点的应用题，还是调用计算器。看 RL 训练到底有没有用。

9d 的题太简单：数字照抄，看到「又」就加。训练一开始就 93～97% 对，100 步里 90 步全对、学不到。
9e 和 9d 一样，模型只写 <算>算式</算>，Python 算。不同的是：
- 每道应用题要两步才算得出来，比如「每盒 37 个，有 46 盒，卖掉 25 个，还剩几个」= 37×46-25。
  要用到减号和括号，计算器也认这两样（第 77～89 行）。
- 有一半的题多给一个用不上的数，比如「老师今年 38 岁。」，模型要知道不用它。
- 题前面的 2 个例子还是一步的题（和 9d 一样），两步的题没有例子，模型要自己学。
- 考试 3 种：练过的说法；练过的说法 + 用不上的数；没见过的说法。
起点：Qwen3-0.6B-Base 加一个新的 LoRA。

不用读任何文件。
运行：cd ~/ai-infra-journey-git/stage0/09-rl && python rl_math_word.py
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
NEW = 32        # 每次最多写几个 token。算式比 9d 长一点
TEMP = 1.0      # 写的时候的随机程度
LR = 2e-4       # 学习率
STEPS = 100     # 训练几步
R = 8           # LoRA 的 r
SHOW_EVERY = 10  # 每几步打印这一组写的
TEST_N = 48     # 训练前后，每种题考几道（4 种说法各 12 道）
# ======================================================

# 调用计算器的例子：放在每道题前面，模型照着这个样子写
EXAMPLES = ("题：25×38=\n答：<算>25×38</算>\n"
            "题：班里有57个人，又来了86个人，现在有几个人？\n答：<算>57+86</算>\n")

# 两步的应用题。每种：题目的说法、怎么出 3 个数、正确答案怎么算
WORDS = {
    "卖掉": ("每盒有{a}个苹果，有{b}盒，卖掉了{c}个，还剩几个？",
             lambda r: (r.randint(10, 99), r.randint(10, 99), r.randint(10, 99)), lambda a, b, c: a * b - c),
    "买笔": ("小明有{a}元，买了{b}支笔，每支{c}元，还剩几元？",
             lambda r: (r.randint(200, 999), r.randint(2, 9), r.randint(10, 20)), lambda a, b, c: a - b * c),
    "看书": ("一本书有{a}页，小红每天看{b}页，看了{c}天，还剩几页？",
             lambda r: (r.randint(300, 999), r.randint(10, 30), r.randint(2, 9)), lambda a, b, c: a - b * c),
    "发本子": ("班里有{a}个男生，{b}个女生，每人发{c}本本子，一共发几本？",
               lambda r: (r.randint(10, 40), r.randint(10, 40), r.randint(2, 9)), lambda a, b, c: (a + b) * c),
}
# 考试才用的说法，训练时一次也没见过
NEW_WORDS = {
    "停车": ("停车场有{a}辆车，开走了{b}辆，又开来了{c}辆，现在有几辆？",
             lambda r: (r.randint(50, 99), r.randint(10, 40), r.randint(10, 99)), lambda a, b, c: a - b + c),
    "买水": ("每箱有{a}瓶水，有{b}箱，又买来{c}瓶，一共有几瓶？",
             lambda r: (r.randint(10, 30), r.randint(10, 99), r.randint(10, 99)), lambda a, b, c: a * b + c),
}
# 用不上的数：有一半的题，在最前面加一句这个
NOISE = ["老师今年{d}岁。", "今天气温{d}度。", "教室在{d}号楼。"]
KINDS = list(WORDS)   # 训练时 4 种轮流出

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
    """计算器：找到 <算>…</算> 中间的算式，用 Python 算出来。认数字、+、-、×、括号。找不到或看不懂，返回 None。"""
    m = re.search(r"<算>(.*?)</算>", text)
    if not m:
        return None
    expr = m.group(1).replace(" ", "").replace("×", "*")
    if not re.fullmatch(r"[\d+\-*()]+", expr) or "**" in expr or len(expr) > 40:
        return None
    try:
        return eval(expr, {"__builtins__": {}})   # 上面已经检查过，只剩数字和 + - * ( )
    except Exception:
        return None


def reward_of(text, answer):
    """计算器算出来的数等于正确答案，得 1 分；否则 0 分。"""
    return 1.0 if calc(text) == answer else 0.0


assert calc("<算>37×46</算>") == 1702
assert calc("<算>(12+30)×5</算>") == 210
assert calc("<算>37×46-25</算>") == 1677
assert reward_of("<算>37×46</算>", 1702) == 1.0
assert reward_of("<算>37×64</算>", 1702) == 0.0     # 列错了，计算器算得再准也不对
assert reward_of("<算>37+46</算>", 1702) == 0.0     # 应该乘，写成了加
assert reward_of("1702", 1702) == 0.0               # 没有调用计算器
assert reward_of("<算>37×46", 1702) == 0.0          # 没写 </算>


# ---------- 出题 ----------
def make(rng, kind, words=WORDS, noise=None):
    """出一道题。noise=None 时一半的题加一句用不上的数。返回：给模型看的题目、正确答案。"""
    text, nums, ans = words[kind]
    a, b, c = nums(rng)
    q = text.format(a=a, b=b, c=c)
    if noise or (noise is None and rng.random() < 0.5):
        q = rng.choice(NOISE).format(d=rng.randint(10, 60)) + q
    return f"题：{q}\n答：", ans(a, b, c)


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


# 训练前后考同样的题
test_rng = random.Random(1234)
test = {
    "练过的说法": [make(test_rng, k, noise=False) for k in KINDS for _ in range(TEST_N // len(KINDS))],
    "练过的说法 + 用不上的数": [make(test_rng, k, noise=True) for k in KINDS for _ in range(TEST_N // len(KINDS))],
    "没见过的说法": [make(test_rng, k, words=NEW_WORDS, noise=False) for k in NEW_WORDS for _ in range(TEST_N // len(NEW_WORDS))],
}


EXAM = {}   # 记下每次考试对了几道，最后放在一起打印


def exam(when):
    print(f"\n{when}考试（每种题 {TEST_N} 道，不抽签）：")
    for kind, qs in test.items():
        _, _, texts = write([q for q, _ in qs], 1, sample=False)
        right = sum(reward_of(t, a) for t, (_, a) in zip(texts, qs))
        EXAM.setdefault(kind, {})[when] = int(right)
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
    print(f"{step:>4}  {kind:<6} {scores[-1]:5.0%} | {d[0]:6.1f} {d[1]:7.0f} {d[2]:5.1f} {d[3]:8.1f} {d[4]:7.1f} {d[5]:6.1f}"
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

# 训练前、训练后的考试放在一起，不用往上翻
print("\n考试对的道数：训练前 → 训练后")
for k, v in EXAM.items():
    print(f"  {k}：{v['训练前']} → {v['训练后']}")
