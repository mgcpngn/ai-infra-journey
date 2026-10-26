# 第 8 步：读懂一个训练程序

两个文件，做的事一样，只有第 8～52 行不一样（用哪块 GPU、每个数几个字节、怎么量显存）：

| 文件 | 给哪台电脑 | 每个数 |
|---|---|---|
| [train_loop.py](train_loop.py) | RTX 4060 笔记本（WSL） | bf16，2 字节（和第 6 步一样） |
| [train_loop_mac.py](train_loop_mac.py) | M1 MacBook Air 16GB | fp32，4 字节 |

两个文件都要读同一个文件夹里的 [poems.csv](poems.csv)（14,475 首绝句，和 peek_data.py 生成的一样）。

## 1. 运行

**4060 笔记本**
```bash
cd /home/wangli/step1
python train_loop.py          # poems.csv 已经在这里（昨天 peek_data.py 生成的）
```

**M1 Mac**（第一次要装库、下载模型）
```bash
git clone https://github.com/mgcpngn/ai-infra-journey.git   # 已经有就 git pull
cd ai-infra-journey/stage0/08-train-loop
python3 -m venv ~/venv-ai           # 建一个只给这个项目用的 Python 环境，只做一次
source ~/venv-ai/bin/activate       # 进入环境；每次开新终端都要运行这一行，命令行前面会出现 (venv-ai)
python -m pip install torch transformers peft pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
HF_ENDPOINT=https://hf-mirror.com python train_loop_mac.py   # 第一次会下载模型，约 1.2 GB
```
为什么要 venv：Homebrew 的 Python 不让直接装库（报错 `externally-managed-environment`）。
如果装 torch 时报找不到合适的版本：`brew install python@3.12 && python3.12 -m venv ~/venv-ai`，再从 `source` 那一行开始（推测：Python 太新，torch 还没有出安装包）。

## 2. 程序的结构

| 行 | 做什么 | 每一步都做吗 |
|---|---|---|
| 21～28 | 可以改的参数 | |
| 60～75 | 准备 A：读 poems.csv，变成 3 张数字表，放进 GPU | 只做一次 |
| 77～90 | 准备 B：模型放进 GPU，加 LoRA，建 AdamW | 只做一次 |
| 100～103 | 环节 1：挑数据。随机挑 16 行 | 每一步 |
| 105～108 | 环节 2：猜下一个字。模型读 16 首诗，每个位置给出每个 token 的分数 | 每一步 |
| 110～113 | 环节 3：算 loss。分数和正确答案比，得到一个数 | 每一步 |
| 115～118 | 环节 4：算梯度。`loss.backward()` | 每一步 |
| 120～126 | 环节 5：改参数。`opt.step()` | 每一步 |
| 135～ | 汇总：每个环节平均多少 ms，显存用在哪里 | 最后一次 |

## 3. 屏幕上看哪一行

- `[准备 A]`、`[准备 B]` 三行：每样东西放进 GPU 后，显存多了多少 MB。
- 每一步一行：`step  loss | 5 个环节各自的 ms  合计 | 猜完、算完 loss、算完梯度、改完这 4 个时刻的显存 MB`。
- 第 1 步下面多一行：logits 有多大。
- 最后的汇总：5 个环节各占百分之几；显存峰值，以及分成哪几块。

第 1 步会比后面慢很多，因为很多东西是第一次准备。汇总里不算第 1 步。

## 4. 运行之前，请先预测（不用算）

1. 5 个环节里，哪个花的时间最多？哪个几乎不花时间？
2. 显存里，「模型本身」和「训练一步中间临时用的」，哪个更大？
3. Mac 和 4060，哪个每一步更快？

## 5. 跑完以后，一次只改一个参数（第 21～26 行）

| 改什么 | 改成 | 先预测：哪个环节的时间、哪个显存数会变 |
|---|---|---|
| `BATCH` | 32 | |
| `L` | 32 | |
| `R` | 32 | |
| `LORA` | False（全量微调） | 4060 上可能显存不够（报 out of memory），这本身就是一个结果 |

每次改完，把汇总那几行贴回来。
