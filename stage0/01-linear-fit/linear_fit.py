"""用梯度下降拟合 y = 3x + 2，对比「手动求梯度」和「PyTorch 自动求导」。

用法：
    python linear_fit.py --lr 0.1
    python linear_fit.py --lr 0.001
    python linear_fit.py --lr 1.5
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # 不弹窗口，直接把图存成文件（WSL/服务器上没有显示器也能用）
import matplotlib.pyplot as plt
import torch

# ---------- 1. 读取命令行参数 ----------
# 学习率（lr）决定每一步沿梯度方向走多远：太小走得慢，太大会冲过头甚至发散。
# 做成参数，就能不改代码地对比不同学习率的效果。
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1, help="学习率")
parser.add_argument("--steps", type=int, default=200, help="训练步数")
args = parser.parse_args()

# ---------- 2. 生成数据 ----------
# 固定随机种子，保证每次运行生成的数据一样，结果才可对比、可复现。
torch.manual_seed(0)
# 用 float64（双精度）：让两种求梯度方式的结果能精确比较；
# 另外 lr=1.5 时数值会爆炸到很大，float32 会溢出成 inf。
N = 100
x = torch.rand(N, dtype=torch.float64) * 2 - 1           # 100 个点，均匀分布在 [-1, 1]
noise = torch.randn(N, dtype=torch.float64) * 0.1        # 小噪声，模拟真实测量误差
y = 3 * x + 2 + noise                                    # 真实规律：w=3, b=2


# ---------- 3. 版本一：手动计算梯度 ----------
def train_manual(lr, steps):
    # 参数从 0 开始：模型一开始什么都不知道，预测全是 0。
    w, b = 0.0, 0.0
    losses = []
    for step in range(steps):
        # 前向：用当前参数做预测
        y_pred = w * x + b
        err = y_pred - y
        # 损失：均方误差 L = mean((wx + b - y)^2)，衡量预测离真实值有多远
        loss = (err ** 2).mean()
        losses.append(loss.item())
        # 梯度：对 L 分别求 w、b 的偏导（链式法则）
        #   dL/dw = mean(2 * (wx + b - y) * x)
        #   dL/db = mean(2 * (wx + b - y))
        # 梯度指向损失「上升最快」的方向
        grad_w = (2 * err * x).mean().item()
        grad_b = (2 * err).mean().item()
        # 更新：沿梯度的反方向走一小步（步长 = lr），损失就会下降
        w -= lr * grad_w
        b -= lr * grad_b
        if step % 20 == 0 or step == steps - 1:
            print(f"[手动] step {step:3d}  w={w:12.6g}  b={b:12.6g}  loss={loss.item():.6g}")
    return w, b, losses


# ---------- 4. 版本二：PyTorch 自动求导 ----------
def train_autograd(lr, steps):
    # requires_grad=True 告诉 PyTorch：记录对这两个张量的所有运算，之后要对它们求导。
    w = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    losses = []
    for step in range(steps):
        y_pred = w * x + b
        loss = ((y_pred - y) ** 2).mean()
        losses.append(loss.item())
        # backward()：PyTorch 沿记录下的计算图自动应用链式法则，
        # 把 dL/dw、dL/db 存进 w.grad、b.grad —— 不用再手写导数公式。
        loss.backward()
        # 更新参数本身不该被当成模型运算记录进计算图，所以放在 no_grad 里。
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        # PyTorch 的梯度默认是「累加」的，每步用完必须清零，否则会和下一步的梯度叠在一起。
        w.grad.zero_()
        b.grad.zero_()
        if step % 20 == 0 or step == steps - 1:
            print(f"[自动] step {step:3d}  w={w.item():12.6g}  b={b.item():12.6g}  loss={loss.item():.6g}")
    return w.item(), b.item(), losses


print(f"学习率 lr = {args.lr}\n")
w1, b1, losses1 = train_manual(args.lr, args.steps)
print()
w2, b2, losses2 = train_autograd(args.lr, args.steps)

# ---------- 5. 证明两种方法结果一致 ----------
# 数学上两者算的是同一个梯度，所以每一步的参数和损失都应相同（只差浮点舍入误差）。
# 用相对误差比较，这样即使 lr=1.5 发散到很大的数也能正确判断。
same = (
    torch.allclose(torch.tensor([w1, b1]), torch.tensor([w2, b2]), rtol=1e-9)
    and torch.allclose(torch.tensor(losses1), torch.tensor(losses2), rtol=1e-9)
)
print(f"\n手动: w={w1:.6g}, b={b1:.6g}")
print(f"自动: w={w2:.6g}, b={b2:.6g}")
print("两种方法结果一致" if same else "两种方法结果不一致！")
assert same

# ---------- 6. 画 loss 曲线 ----------
# 纵轴用对数刻度：loss 可能从几降到 0.01，或在发散时涨到 1e100，线性刻度看不清。
plt.plot(losses1, label="manual gradient", linewidth=3)
plt.plot(losses2, "--", label="autograd")
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("MSE loss (log scale)")
plt.title(f"Loss curve, lr={args.lr}")
plt.legend()
out = f"loss_lr{args.lr}.png"
plt.savefig(out)
print(f"loss 曲线已保存到 {out}")
