# 怎么手动运行 bench.py（vLLM 并发测试）

代码：[bench.py](bench.py)（用户本地 /home/wangli/step1/bench.py，10-20 上传）。下面的行号都指这个文件。

## 1. 先开 vLLM 服务（终端 1）
```bash
cd /home/wangli/step1
HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve Qwen/Qwen3-0.6B-Base --max-model-len 2048 --gpu-memory-utilization 0.8 --port 8000 2>&1 | tee server.log
```
等到日志里出现服务已经启动的字样，再做下一步。

**启动日志里看这 4 行**（数在 results.md 第 1 节）：
| 你要找的数 | 日志原文 |
|---|---|
| 模型占 1.12 GiB | `Model loading took 1.12 GiB memory ...` |
| KV cache 留了 4.61 GiB | `Available KV cache memory: 4.61 GiB` |
| KV cache 能放 43,120 个 token | `GPU KV cache size: 43,120 tokens` |
| 最大并发 21 | `Maximum concurrency for 2,048 tokens per request: 21.05x` |

21.05 = 43,120 ÷ 2,048。2,048 是启动命令里的 `--max-model-len`。改这个数，这一行就会变。

## 2. 跑测试（终端 2）
```bash
cd /home/wangli/step1
python bench.py
```
第 55 行先打印 `prompt tokens = 63`。第 57 行打印表头，第 66 行每个 N 打印一行：
```
N   tok/s   TTFT_ms  TPOT_ms  E2E_s  GPU_MiB
1   160.7   30.7     6.03     0.80   6681
```
结果同时存进 `bench.json`（第 67 行）。

## 3. 每一列是怎么算的
| 列 | 意思 | 行 |
|---|---|---|
| tok/s | 所有请求一共写出的 token ÷ 总时间 | 62 |
| TTFT_ms | 发出请求到收到第一个字 | 32（`first - t0`） |
| TPOT_ms | 第一个字以后，平均每个字的时间。**6.03 ms 就是这一列 N=1 那一行** | 32（`(t1 - first) / (ntok - 1)`） |
| E2E_s | 一个请求从发出到结束 | 33 |
| GPU_MiB | nvidia-smi 每 0.2 秒看一次，取最大值。包括 vLLM 启动时预留的 KV cache | 35～45 |

## 4. 可以改的参数
| 想改什么 | 行 | 现在 |
|---|---|---|
| 同时发多少个请求 | 58 `for n in (1, 4, 16, 64)` | 1、4、16、64。可以改成 `(1, 64, 256, 1000)` 看排队 |
| 每个请求写多少个字 | 15 `max_tokens` | 128 |
| 输入多长 | 11 `[:64]` | 64 个 token |
| 先热身一次 | 59 | 每个 N 先跑一遍不记录 |

## 5. 「天花板 5.73 ms」从哪里来
不是 bench.py 算的。是另一个小程序 ceiling.py（服务关掉以后跑）：
- 在 GPU 上把 1 GiB 的数复制 20 次，算出每秒能搬 208.2 GB（results.md 第 3 节）。
- 模型参数 596,049,920 个 × 2 字节 = 1,192,099,840 字节。
- 每写一个字要把全部参数读一遍：1.192 GB ÷ 208.2 GB/s = 5.73 ms。
- 实测 6.03 ms，是天花板的 1.05 倍。

我之前说的 4.7 ms 是用显卡标称带宽 256 GB/s 算的（凭记忆，未核实），不是实测，以 5.73 ms 为准。
