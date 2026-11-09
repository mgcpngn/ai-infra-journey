# vLLM 基线结果：Qwen3-0.6B-Base，RTX 4060 Laptop 8GB（WSL2），用户本地跑出，10-20

vllm 0.32.0 / torch 2.13.0+cu130。启动命令按 prompt.md；另加环境变量 HF_HUB_OFFLINE=1、VLLM_USE_FLASHINFER_SAMPLER=0（系统 nvcc 太旧，FlashInfer 采样算子 JIT 编译失败）。

## 1. 启动日志（server.log）

| 项目 | 值 | 日志原文 |
|---|---|---|
| 模型权重显存 | 1.12 GiB | `Model loading took 1.12 GiB memory and 5.003635 seconds` |
| KV cache 显存 | 4.61 GiB | `Available KV cache memory: 4.61 GiB` |
| KV cache 容量 | 43,120 tokens | `GPU KV cache size: 43,120 tokens` |
| 最大并发 | 21.05x（每请求 2048 token） | `Maximum concurrency for 2,048 tokens per request: 21.05x` |

KV cache 大小随启动时空闲显存浮动（两次启动分别为 4.39 GiB/41,136 与 4.61 GiB/43,120 tokens）。

## 2. 并发测试（bench.py）

输入 63 token 中文，输出 128 token（ignore_eos，实测均为 128），temperature=0，每个 N 先热身再正式跑。显存为 nvidia-smi 峰值（含 vLLM 预分配）。

| N | 总吞吐 tok/s | 平均首字 TTFT ms | 平均每字 TPOT ms | 平均请求时间 s | 显存 MiB |
|---|---|---|---|---|---|
| 1 | 160.7 | 30.7 | 6.03 | 0.80 | 6681 |
| 4 | 602.6 | 36.8 | 6.39 | 0.85 | 6683 |
| 16 | 2162.5 | 45.8 | 7.08 | 0.95 | 6683 |
| 64 | 5379.3 | 133.2 | 10.86 | 1.51 | 6703 |

## 3. 带宽天花板（ceiling.py，服务已关）

1 GiB 张量 `copy_` 20 次。带宽按每次复制读 1 GiB + 写 1 GiB 计。

| 参数个数 | 参数字节 (bf16) | 实测带宽 GB/s | 天花板 ms/token | N=1 实测 TPOT ms | 实测/天花板 |
|---|---|---|---|---|---|
| 596,049,920 | 1,192,099,840 | 208.2 | 5.73 | 6.03 | 1.05x |

## 核对（Claude 算）
- KV cache：每个 token 112 KB（第 6 步按 config 算的）。4.61 GiB ÷ 112 KB ≈ 43,160 tokens，和日志的 43,120 对上。
- 每个请求 63 + 128 = 191 token。43,120 ÷ 191 ≈ 225 个请求能同时放进 KV cache。
- N 1 → 64：总吞吐 33 倍；每个字的时间 1.8 倍；第一个字的时间 4.3 倍。
- 同一个模型、同一张卡：第 7 步脚本每字 129.4 ms，vLLM N=1 每字 6.03 ms，快 21 倍。

## 11-09 新环境复跑（conda 环境 vllm：vllm 0.32.0、torch 2.13.0+cu130）

启动日志：`Model loading took 1.12 GiB`；`Available KV cache memory: 4.39 GiB`；`GPU KV cache size: 41,136 tokens`；最大并发 20.09x。
（和 10-20 第二次启动的 4.39 GiB / 41,136 一样，见第 1 节。）

| N | 总吞吐 tok/s | TTFT ms | TPOT ms | 请求时间 s | 显存 MiB |
|---|---|---|---|---|---|
| 1 | 158.6 | 19.1 | 6.20 | 0.81 | 6463 |
| 4 | 588.3 | 27.3 | 6.62 | 0.87 | 6465 |
| 16 | 2101.9 | 39.2 | 7.35 | 0.97 | 6465 |
| 64 | 5240.1 | 93.0 | 11.49 | 1.55 | 6499 |

和 10-20 比（Claude 算）：
- TPOT 差不多：N=1 6.20 vs 6.03（+3%），N=64 11.49 vs 10.86（+6%）。
- TTFT 变短：N=1 19.1 vs 30.7，N=64 93.0 vs 133.2。日志里有 `Prefix cache hit rate: 38.1%`：bench.py 每次发同样的 63 个 token，热身以后前面一段可以直接用存好的 KV cache，不用重算（推测；10-20 的日志没有记这一行）。
- 以后 fp8 和这一次比（同一个环境）。
