"""DPO：从 LoRA r=8（checkpoints/qwen3_0.6b_lora_r8）继续训练，让模型更偏向「扣题」的诗。

数据：chosen = 这首诗本身，rejected = 随机另一首同体裁、不同题目的诗，prompt 相同。
参考模型 = 同一个模型关掉 LoRA（disable_adapter），即基础模型。
所有表汇总到 results_dpo.md。
"""
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import PeftModel

# lora_poem.py 没有 __main__ 保护，import 会把实验全跑一遍，所以只执行到「实验 1」之前
# （数据读取、prompt_of、check_format、encode、generate_all、table/note 等），并把输出文件改成 results_dpo.md。
src = open("lora_poem.py", encoding="utf-8").read().split("# ---------- 1. 基线")[0]
exec(src.replace('"results_lora.md"', '"results_dpo.md"').replace("LoRA 写绝句实验结果", "DPO 写绝句实验结果"))

BETA, BS, LR, STEPS = 0.1, 8, 5e-6, 300
LORA_IN, LORA_OUT = "checkpoints/qwen3_0.6b_lora_r8", "checkpoints/qwen3_0.6b_lora_r8_dpo"


# ---------- 0. 成对数据 ----------
def make_pairs(ps, seed):
    """每首诗配一首同体裁、不同题目的诗作 rejected。"""
    rng = random.Random(seed)
    by_genre = {}
    for p in ps:
        by_genre.setdefault(p[1], []).append(p)
    pairs = []
    for title, genre, poem in ps:
        while (r := rng.choice(by_genre[genre]))[0] == title:
            pass
        pairs.append((title, genre, poem, r[2]))
    return pairs


def encode_pairs(pairs):
    """chosen / rejected 各自用 lora_poem 的 encode：prompt 部分 label=-100，只算「诗：」后面的 token。"""
    return encode([(t, g, c) for t, g, c, _ in pairs])[:3], encode([(t, g, r) for t, g, _, r in pairs])[:3]


train_pairs, val_pairs = make_pairs(train_poems, 1), make_pairs(val_poems, 2)
Ctr, Rtr = encode_pairs(train_pairs)
Cva, Rva = encode_pairs(val_pairs)
for t, g, c, r in train_pairs[:2]:
    print(f"\n--- 样例 ---\nprompt：{prompt_of(t, g)!r}\nchosen：{c!r}\nrejected：{r!r}")
note(f"成对数据：训练 {len(train_pairs):,} 对，验证 {len(val_pairs)} 对")
table("实验 0：成对数据样例（训练集前 2 对）", ["题目", "体裁", "chosen（原诗）", "rejected（同体裁别的诗）"],
      [(t, g, show(c), show(r)) for t, g, c, r in train_pairs[:2]])


# ---------- 工具 ----------
def seq_logp(model, X, M, Y):
    """每条序列「诗：」后面 token 的 log 概率之和。第 t 个位置的 logits 预测第 t+1 个 token，所以错开一位。"""
    logits = model(input_ids=X, attention_mask=M).logits[:, :-1].float()
    y = Y[:, 1:]
    mask = y != -100
    lp = logits.log_softmax(-1).gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)


@torch.no_grad()
def pick_acc(model, n=None, bs=16):
    """选对率：模型给 chosen 的 log 概率和 > 给 rejected 的，就算选对。"""
    model.eval()
    n = n or len(Cva[0])
    right = 0
    for i in range(0, n, bs):
        j = min(i + bs, n)
        right += (seq_logp(model, *(x[i:j] for x in Cva)) > seq_logp(model, *(x[i:j] for x in Rva))).sum().item()
    model.train()
    return right / n


def timed_gen(model):
    """generate_all 生成 10 首（种子 1234，和第 6 步一致），顺便计时，返回 (结果, 每字平均 ms)。"""
    torch.cuda.synchronize()
    t0 = time.time()
    gens = generate_all(model)
    torch.cuda.synchronize()
    chars = sum(len(re.findall(r"[一-鿿]", p)) for p, _ in gens)
    return gens, (time.time() - t0) * 1000 / max(chars, 1), chars


# ---------- 1. 训练前 ----------
print("\n========== 训练前 ==========")
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, LORA_IN, is_trainable=True).to(dev)  # is_trainable=True：LoRA 继续训练
acc_before = pick_acc(model)
gens_before, ms_before, chars_before = timed_gen(model)
note(f"训练前：验证集 500 对选对率 {acc_before:.1%}，10 首格式正确率 {acc(gens_before):.0%}")
table("实验 1：训练前生成（LoRA r=8，种子 1234）", ["题目", "体裁", "生成的诗", "格式"],
      [(t, g, show(p), "✓" if ok else "✗") for (t, g, _), (p, ok) in zip(eval_items, gens_before)])

# ---------- 2. DPO 训练 ----------
print("\n========== DPO 训练 ==========")
model.config.use_cache = False
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
g = torch.Generator(device=dev).manual_seed(0)
torch.manual_seed(0)  # LoRA dropout 可复现
torch.cuda.reset_peak_memory_stats()
rows, losses, t_train = [], [], 0.0
model.train()
for step in range(1, STEPS + 1):
    torch.cuda.synchronize()
    t0 = time.time()
    ix = torch.randint(len(Ctr[0]), (BS,), device=dev, generator=g)
    # chosen 和 rejected 拼成一个 batch（前 BS 条 chosen，后 BS 条 rejected），一次前向
    X, M, Y = (torch.cat([c[ix], r[ix]]) for c, r in zip(Ctr, Rtr))

    # 旧模型（参考模型）：同一个模型临时关掉 LoRA = 基础模型，不需要梯度
    with torch.no_grad(), model.disable_adapter():
        ref = seq_logp(model, X, M, Y)
    # 新模型（策略模型）：带 LoRA，要梯度
    pol = seq_logp(model, X, M, Y)

    # 新模型相对旧模型，在 chosen / rejected 上各把 log 概率抬高了多少
    d_chosen = pol[:BS] - ref[:BS]
    d_rejected = pol[BS:] - ref[BS:]
    # DPO：希望 chosen 抬得比 rejected 多；beta 控制偏离旧模型的力度
    loss = -F.logsigmoid(BETA * (d_chosen - d_rejected)).mean()

    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    losses.append(loss.item())
    t_train += time.time() - t0
    if step % 50 == 0:
        rows.append((step, f"{sum(losses[-50:]) / 50:.4f}", f"{pick_acc(model, 200):.1%}",
                     f"{t_train * 1000 / 50:.0f}", f"{torch.cuda.max_memory_allocated() / GB:.2f}"))
        print(f"step {step}  loss={rows[-1][1]}  选对率(200 对)={rows[-1][2]}  {rows[-1][3]} ms/步  显存峰值 {rows[-1][4]} GB")
        t_train = 0.0
model.config.use_cache = True
note(f"第 1 步 DPO loss = {losses[0]:.4f}（ln2 = 0.6931；参考模型是基础模型而不是 LoRA 起点，所以第 1 步就不等于 ln2）")
table(f"实验 2：DPO 训练过程（batch={BS} 对，长度 {L}，lr={LR}，beta={BETA}，AdamW）",
      ["step", "loss（近 50 步平均）", "验证选对率（前 200 对）", "每步耗时 ms", "显存峰值 GB"], rows)

# ---------- 3. 训练后 ----------
print("\n========== 训练后 ==========")
acc_after = pick_acc(model)
gens_after, ms_after, chars_after = timed_gen(model)
table("实验 3a：选对率（验证集 500 对）", ["", "训练前", "训练后"], [("选对率", f"{acc_before:.1%}", f"{acc_after:.1%}"),
      ("10 首格式正确率", f"{acc(gens_before):.0%}", f"{acc(gens_after):.0%}")])
table("实验 3b：同样 10 个题目，训练前 vs DPO 后（种子 1234）", ["题目", "体裁", "训练前", "格式", "DPO 后", "格式"],
      [(t, g, show(bp, 40), "✓" if bok else "✗", show(ap, 40), "✓" if aok else "✗")
       for (t, g, _), (bp, bok), (ap, aok) in zip(eval_items, gens_before, gens_after)])

# ---------- 4. 推理对比 + 文件大小 ----------
model.save_pretrained(LORA_OUT)
size = os.path.getsize(os.path.join(LORA_OUT, "adapter_model.safetensors"))
table("实验 4：推理速度与文件大小", ["项目", "训练前", "训练后"], [
    ("10 首共生成汉字", chars_before, chars_after),
    ("每字平均生成时间", f"{ms_before:.1f} ms", f"{ms_after:.1f} ms"),
    ("LoRA 文件", f"{os.path.getsize(os.path.join(LORA_IN, 'adapter_model.safetensors')) / 1024 ** 2:.2f} MB",
     f"{size / 1024 ** 2:.2f} MB（{LORA_OUT}）"),
])
print("已写入 results_dpo.md")
