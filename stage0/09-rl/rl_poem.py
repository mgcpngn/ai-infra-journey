"""第 9 步：RL 后训练。模型自己写诗，检查程序打分，写得好的方向被加强。

和第 8 步 train_loop.py 的区别：
- 第 8 步：每一步读 16 首「标准答案」（真唐诗），让模型学着猜下一个字。
- 第 9 步：没有标准答案。每一步给 1 个题目，让模型自己写 16 首，
  检查程序给每首打分（奖励），比这 16 首平均分高的，加强；低的，减弱。
起点：Qwen3-0.6B-Base 加一个新的 LoRA（没有经过第 6 步的训练），所以一开始几乎写不对格式。
没有 DPO 里那个「不动的旧模型」，为了短。

要用到 ../08-train-loop/poems.csv（只用里面的题目和体裁，不用诗句）。
运行：cd ~/ai-infra-journey-git/stage0/09-rl && python rl_poem.py
"""
import re
import time

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== 可以改的参数 =====================
G = 16          # 每个题目写几首（一组）。第 8 步每步读 16 首，这里每步写 16 首
NEW = 48        # 每首最多写几个 token
TEMP = 1.0      # 写诗时的随机程度。越大，16 首之间越不一样
LR = 2e-4       # 学习率（和第 6 步一样）
STEPS = 100     # 训练几步
R = 8           # LoRA 的 r
SHOW_EVERY = 10  # 每几步打印一首诗看看
BALANCE = False  # True = 五言、七言题目轮流出；False = 随机挑（七言多，10,540 首对 3,935 首）
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


# ---------- 打分：检查程序（规则和 lora_poem.py 第 57～61 行的 check_format 一样，但给部分分） ----------
def reward_of(poem, genre):
    """4 句、每句 5 字（五言）或 7 字（七言）才算全对，得 1 分。
    部分分：前 4 句里每句字数对，得 0.25；每多写一句，扣 0.25；最低 0 分。"""
    n = 5 if genre == "五言绝句" else 7
    s = [x for x in re.split(r"[，。！？；、,.!?;\s]+", poem) if x]
    right = sum(len(x) == n and re.fullmatch(r"[一-鿿]+", x) is not None for x in s[:4])
    extra = max(0, len(s) - 4)
    return max(0.0, (right - extra) / 4)


assert reward_of("床前明月光，疑是地上霜。\n举头望明月，低头思故乡。", "五言绝句") == 1.0
assert reward_of("床前明月光，疑是地上霜。", "五言绝句") == 0.5
assert reward_of("床前明月光，疑是地上霜。举头望明月，低头思故乡。日暮苍山远。", "五言绝句") == 0.75

# ---------- 准备：题目、模型、LoRA、AdamW ----------
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
df = pd.read_csv("../08-train-loop/poems.csv", dtype=str, encoding="utf-8-sig")
items = list(zip(df["题目"], df["体裁"]))
by_genre = {g: [it for it in items if it[1] == g] for g in ("五言绝句", "七言绝句")}
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
model = get_peft_model(model, LoraConfig(r=R, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
print(f"模型和 LoRA 放进 GPU：{mem():.0f} MB")


def write(title, genre, n, seed):
    """给一个题目写 n 首。返回：整段 token（题目 + 写出的部分）、题目部分有几个 token、n 首诗的文字。"""
    x = tok(f"题目：{title}\n体裁：{genre}\n诗：", return_tensors="pt").input_ids.to(dev)
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(x.repeat(n, 1), attention_mask=torch.ones_like(x).repeat(n, 1), do_sample=True, temperature=TEMP, top_k=0, top_p=1.0,
                             max_new_tokens=NEW, stop_strings=["\n\n"], tokenizer=tok,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model.train()
    P = x.shape[1]
    poems = [tok.decode(o[P:], skip_special_tokens=True).split("\n\n")[0].strip() for o in out]
    return out, P, poems


# 训练前后都用这 6 个题目各写 1 首，好对比：3 个五言、3 个七言
fixed = by_genre["五言绝句"][:3] + by_genre["七言绝句"][:3]
print("\n训练前：")
for t, g in fixed:
    _, _, p = write(t, g, 1, 1234)
    print(f"  {t}（{g}）：{p[0][:40]!r}  得分 {reward_of(p[0], g):.2f}")

# ---------- 训练：每一步 6 个环节 ----------
print("\nstep 体裁 平均分 全对 |  时间 ms：1挑题目 2写诗 3打分 4猜下一个字 5算梯度 6改参数  合计 | 显存 MB")
gen = torch.Generator().manual_seed(0)
ms, scores, full, genres = [], [], [], []
torch.cuda.reset_peak_memory_stats()
for step in range(1, STEPS + 1):
    t0 = now()
    # 环节 1：挑题目。随机挑 1 个（BALANCE=True 时，单数步挑五言，双数步挑七言）
    pool = by_genre["五言绝句" if step % 2 else "七言绝句"] if BALANCE else items
    title, genre = pool[torch.randint(len(pool), (1,), generator=gen).item()]
    t1 = now()

    # 环节 2：写诗。同一个题目写 G 首（这就是推理：一次写一个 token，不算梯度）
    out, P, poems = write(title, genre, G, step)
    t2 = now()

    # 环节 3：打分。每首一个奖励，减去这一组的平均分：比平均好的是正数，比平均差的是负数
    r = torch.tensor([reward_of(p, genre) for p in poems], device=dev)
    adv = r - r.mean()
    t3 = now()

    # 环节 4：猜下一个字。模型重新读一遍自己写的 G 首，算出每个写出来的 token 的 log 概率
    ans = out[:, P:]                                    # 写出来的部分
    pad = ans == tok.pad_token_id
    c = pad.cumsum(1)
    keep = (c == 0) | (pad & (c == 1))                  # 写完以后补的空位不算；只留第一个，让模型学会停
    logits = model(input_ids=out, attention_mask=torch.ones_like(out)).logits[:, P - 1:-1].float()
    logp = logits.log_softmax(-1).gather(-1, ans.unsqueeze(-1)).squeeze(-1)
    # loss：分高的那几首，让它的 token 更可能被写出来；分低的，更不可能
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
    full.append((r == 1).float().mean().item())
    genres.append(genre)
    print(f"{step:>4}  {genre[:2]}  {scores[-1]:5.2f} {full[-1]:4.0%} | {d[0]:6.1f} {d[1]:7.0f} {d[2]:5.1f} {d[3]:8.1f} {d[4]:7.1f} {d[5]:6.1f}"
          f"  {sum(d):6.0f} | {torch.cuda.max_memory_allocated() / MB:6.0f}")
    if step % SHOW_EVERY == 0:
        best = int(r.argmax())
        print(f"      题目：{title}（{genre}），这一组得分最高的一首（{r[best]:.2f} 分）：{poems[best][:40]!r}")

# ---------- 汇总 ----------
k = min(10, STEPS)
print(f"\n平均分：前 {k} 步 {sum(scores[:k]) / k:.2f} → 最后 {k} 步 {sum(scores[-k:]) / k:.2f}")
print(f"全对的比例：前 {k} 步 {sum(full[:k]) / k:.0%} → 最后 {k} 步 {sum(full[-k:]) / k:.0%}")
half = STEPS // 2
print(f"后一半 {STEPS - half} 步，按体裁分开看平均分（上面两行只看最后 10 步，那 10 步可能碰巧全是七言题目，所以这里分开看）：")
for g in ("五言绝句", "七言绝句"):
    sc = [x for x, y in zip(scores[half:], genres[half:]) if y == g]
    print(f"  {g}：{len(sc)} 步，平均分 {sum(sc) / len(sc):.2f}" if sc else f"  {g}：0 步")
rest = ms[1:] if len(ms) > 1 else ms
avg = [sum(c) / len(rest) for c in zip(*rest)]
names = ["1 挑题目", "2 写诗", "3 打分", "4 猜下一个字", "5 算梯度", "6 改参数"]
print(f"\n每个环节平均时间（不算第 1 步）：")
for n, a in zip(names, avg):
    print(f"  {n}：{a:7.1f} ms  {a / sum(avg):5.1%}")
print(f"  合计：{sum(avg):7.1f} ms")
print(f"显存峰值 {torch.cuda.max_memory_allocated() / MB:.0f} MB")

print("\n训练后（同样 6 个题目、同样的随机种子）：")
for t, g in fixed:
    _, _, p = write(t, g, 1, 1234)
    print(f"  {t}（{g}）：{p[0][:40]!r}  得分 {reward_of(p[0], g):.2f}")
