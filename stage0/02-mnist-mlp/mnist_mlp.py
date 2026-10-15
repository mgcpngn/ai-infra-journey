"""两层神经网络（784 -> 128 -> 10）识别 MNIST 手写数字。

用法：
    python mnist_mlp.py              # 有 ReLU 激活
    python mnist_mlp.py --no-act     # 去掉 ReLU，对比效果
    python mnist_mlp.py --hidden 16  # 中间层改成 16 个神经元（可试 16 / 128 / 512）
"""
import argparse
import gzip
import os
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")  # 不弹窗口，直接存图
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------- 1. 命令行参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--no-act", action="store_true", help="去掉中间层的 ReLU 激活函数")
parser.add_argument("--hidden", type=int, default=128, help="中间层神经元个数")
parser.add_argument("--epochs", type=int, default=10, help="把整个训练集过几遍")
parser.add_argument("--batch-size", type=int, default=128, help="每次更新参数用多少张图")
parser.add_argument("--lr", type=float, default=0.1, help="学习率")
args = parser.parse_args()

torch.manual_seed(0)  # 固定随机种子，结果可复现
# 有 GPU 就用 GPU：神经网络大量矩阵乘法，GPU 能并行算得快很多
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {device}")


# ---------- 2. 准备数据 ----------
# MNIST 原始文件是 gzip 压缩的二进制：前面一小段文件头，后面是逐字节的像素/标签。
# 自己下载解析，省得为这一件事安装 torchvision。
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def load(name, header_bytes):
    path = os.path.join("data", name)
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        print(f"下载 {name} ...")
        urllib.request.urlretrieve(MIRROR + name, path)
    with gzip.open(path, "rb") as f:
        raw = f.read()[header_bytes:]  # 跳过文件头
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8)


# 图片：每张 28x28=784 个像素，展平成一行 784 个数，这就是网络的输入层为什么是 784
images = load("train-images-idx3-ubyte.gz", 16).reshape(-1, 784).float() / 255  # 像素缩放到 0~1
labels = load("train-labels-idx1-ubyte.gz", 8).long()                           # 0~9 的数字标签
# 标准化：减均值除标准差，让输入大致在 0 附近，梯度下降更稳
images = (images - 0.1307) / 0.3081

# 拆出验证集：训练用前 5 万张，最后 1 万张模型从没学过，用来检验它是不是真学会了（而不是死记硬背）
# 整个数据集只有约 200MB，一次性放进显存，省掉逐批搬运数据的开销
x_train, y_train = images[:50000].to(device), labels[:50000].to(device)
x_val, y_val = images[50000:].to(device), labels[50000:].to(device)


# ---------- 3. 定义模型 ----------
# 第一层 Linear(784,128)：每个隐藏神经元 = 784 个像素的加权和 + 偏置，相当于 128 个「特征探测器」
# ReLU：把负数变 0。这个非线性是关键——没有它，两层线性变换叠起来仍是一个线性变换，
#       784->128->10 就退化成 784->10 的单层模型，表达能力不会因为加层而变强
# 第二层 Linear(128,10)：把 128 个特征组合成 10 个分数，分别对应数字 0~9，分数最高的就是预测
# --hidden 改的就是这个「128」：神经元越多，能探测的特征越多，但参数和计算量也越大
layers = [nn.Linear(784, args.hidden)]
if not args.no_act:
    layers.append(nn.ReLU())
layers.append(nn.Linear(args.hidden, 10))
model = nn.Sequential(*layers).to(device)

# 逐层打印参数：Linear 层的权重形状是 (输出, 输入)，偏置形状是 (输出,)；
# 该层参数量 = 输出*输入 + 输出。ReLU 只是逐元素变换，没有参数。
print("层" + " " * 10 + "权重形状" + " " * 12 + "偏置形状" + " " * 6 + " " * 4 + "参数量")  # 中文字符占两格宽，手动补空格对齐
for name, layer in model.named_children():
    label = f"{name} {type(layer).__name__}"
    if isinstance(layer, nn.Linear):
        count = layer.weight.numel() + layer.bias.numel()
        print(f"{label:<12}{str(tuple(layer.weight.shape)):<20}{str(tuple(layer.bias.shape)):<14}{count:>10,}")
    else:
        print(f"{label:<12}{'-':<20}{'-':<14}{0:>10}")

# 参数总数：每层 权重(输入x输出) + 偏置(输出)，即 784*hidden+hidden + hidden*10+10
n_params = sum(p.numel() for p in model.parameters())
n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())  # float32 每个数 4 字节
print(f"参数总数: {n_params:,}，占 {n_bytes:,} 字节 ({n_bytes / 1024:.1f} KB)")

# 交叉熵损失：分类任务的标准损失。正确类别的预测概率越接近 1，损失越接近 0
loss_fn = nn.CrossEntropyLoss()
# SGD 就是上一个实验里的梯度下降：参数 -= lr * 梯度，只不过现在由优化器帮我们对所有参数做
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


def train_step(xb, yb):
    """训练一个 batch：前向算损失 -> 反向求梯度 -> 更新参数。"""
    loss = loss_fn(model(xb), yb)
    optimizer.zero_grad()  # 梯度会累加，先清零
    loss.backward()        # 自动求导，算出所有参数的梯度
    optimizer.step()       # 沿梯度反方向更新参数
    return loss


@torch.no_grad()  # 评估时不需要梯度，关掉能省内存、更快
def evaluate(x, y):
    """在整个数据集上算平均 loss 和准确率。"""
    logits = model(x)
    loss = loss_fn(logits, y).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()  # 分数最高的类别 == 真实标签 的比例
    return loss, acc


# ---------- 4. 训练 ----------
train_losses, val_losses = [], []
n = len(x_train)
for epoch in range(1, args.epochs + 1):
    # 每个 epoch 打乱顺序：避免模型受数据排列顺序影响
    perm = torch.randperm(n, device=device)
    model.train()
    for i in range(0, n, args.batch_size):
        idx = perm[i:i + args.batch_size]
        train_step(x_train[idx], y_train[idx])

    model.eval()
    tr_loss, tr_acc = evaluate(x_train, y_train)
    va_loss, va_acc = evaluate(x_val, y_val)
    train_losses.append(tr_loss)
    val_losses.append(va_loss)
    print(f"epoch {epoch:2d}  训练 loss={tr_loss:.4f} acc={tr_acc:.2%}  |  验证 loss={va_loss:.4f} acc={va_acc:.2%}")


# ---------- 5. 计时 ----------
# GPU 是「异步」执行的：Python 发出指令就立刻返回，GPU 在后台慢慢算。
# 所以计时前后必须 synchronize()，等 GPU 真正算完，否则测到的只是「发指令」的时间。
# 先预热几次（首次调用有初始化开销），再重复多次取平均，结果才稳定。
def time_it(fn, repeat=100):
    for _ in range(10):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000  # 毫秒


xb, yb = x_train[:args.batch_size], y_train[:args.batch_size]
model.train()
t_train = time_it(lambda: train_step(xb, yb))  # 注意：这会让模型多训练 110 步，计时放在最后就不影响上面的结果
model.eval()
with torch.no_grad():  # 推理不需要梯度，不记录计算图
    t_infer = time_it(lambda: model(xb))
# 比值 = 训练比推理慢几倍：训练多了反向传播（算梯度）和参数更新
print(f"\n训练一个 batch：{t_train:.3f} ms，推理一个 batch：{t_infer:.3f} ms，比值：{t_train / t_infer:.1f}")


# ---------- 6. 画 loss 曲线 ----------
# 两条线对比：训练 loss 持续下降而验证 loss 回升，就说明开始「过拟合」（死记训练集）
tag = ("no_act" if args.no_act else "relu") + f"_h{args.hidden}"  # 文件名带上配置，不同实验的图不会互相覆盖
epochs = range(1, args.epochs + 1)
plt.plot(epochs, train_losses, "o-", label="train")
plt.plot(epochs, val_losses, "s-", label="validation")
plt.xlabel("epoch")
plt.ylabel("cross-entropy loss")
plt.title(f"MNIST MLP ({tag})")
plt.legend()
out = f"loss_{tag}.png"
plt.savefig(out)
print(f"loss 曲线已保存到 {out}")
