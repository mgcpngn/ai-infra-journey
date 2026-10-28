"""看懂 RL 的环节 3：为什么减去这一组的平均分。只用 CPU，几秒跑完，不用模型。

一个最小的「模型」：题目 37+46=，只能从 4 个答案里选一个。
模型只有 4 个参数（每个答案一个分数），分数变成概率，一开始每个答案 25%。
下面做 rl_poem.py / rl_math.py 里同样的一步：写 16 个答案 → 打分 → 减去平均分 → loss → 算梯度 → 改参数，
然后看 4 个答案的概率怎么变。
为了看得清楚，16 个答案是写好的，不抽签；改参数用最简单的办法：参数 = 参数 − 学习率 × 梯度。

运行：cd ~/ai-infra-journey-git/stage0/09-rl && python adv_demo.py
"""
import torch

ANSWERS = ["83", "84", "73", "93"]   # 只有 83 是对的
LR = 1.0                              # 学习率

# 3 种情况，每种都是同一个模型、同样的起点
CASES = {
    "A：16 个里对了 4 个": ["83"] * 4 + ["84"] * 5 + ["73"] * 4 + ["93"] * 3,
    "B：16 个全对": ["83"] * 16,
    "C：16 个全错": ["84"] * 6 + ["73"] * 6 + ["93"] * 4,
}

for name, written in CASES.items():
    w = torch.zeros(4, requires_grad=True)             # 4 个参数，都是 0 → 4 个答案概率一样，各 25%
    before = w.softmax(0).tolist()

    # 环节 3：打分，减去平均分
    idx = torch.tensor([ANSWERS.index(a) for a in written])
    r = torch.tensor([1.0 if a == "83" else 0.0 for a in written])
    adv = r - r.mean()

    # 环节 4：每个写出来的答案，模型给它的 log 概率；loss 和 rl_math.py 一样
    logp = w.log_softmax(0)[idx]
    loss = -(adv * logp).mean()

    # 环节 5：算梯度；环节 6：改参数
    loss.backward()
    with torch.no_grad():
        w -= LR * w.grad
    after = w.softmax(0).tolist()

    print(f"\n{name}：平均分 {r.mean():.2f}，loss {loss.item():.3f}")
    print("  答案  写了几次  得分  减去平均分以后  参数的梯度   概率：改之前 → 改之后")
    for i, a in enumerate(ANSWERS):
        n = int((idx == i).sum())
        s = 1.0 if a == "83" else 0.0
        d = f"{s - r.mean():+.2f}" if n else "  — "      # 没写过的答案，没有这一项
        print(f"  {a}     {n:>4}     {s:.0f}       {d}         {w.grad[i]:+.3f}       {before[i]:.0%} → {after[i]:.0%}")
