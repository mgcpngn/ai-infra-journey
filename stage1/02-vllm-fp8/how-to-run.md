# 1 月-2：vLLM 跑 fp8 的 Qwen（10-30）

10-29 用户问：「如果计算时还需要变回 fp32，量化有什么意义？」
这次用 vLLM 自带的 fp8：权重存成 1 个字节，乘的时候也用 fp8（4060 是 Ada 架构，有 fp8 的计算单元，凭记忆，未核实）。
不是第 5 步那种「先还原再乘」。

和 1 月-1 vLLM 基线只差一个地方：启动命令多了 `--quantization fp8`。模型、bench.py、输入输出长度全部一样。

## 1. 先开 vLLM 服务（终端 1）
```bash
cd ~/ai-infra-journey-git/stage1/02-vllm-fp8
HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve Qwen/Qwen3-0.6B-Base --quantization fp8 --max-model-len 2048 --gpu-memory-utilization 0.8 --port 8000 2>&1 | tee server.log
```
`--quantization fp8`：vLLM 读进 bf16 权重，启动时自己转成 fp8，不用另外下载模型。

启动日志里找和基线同样的 4 行，记下来：
| 数 | 日志原文 | 基线（bf16） | fp8 |
|---|---|---|---|
| 模型占多少 | `Model loading took ... GiB` | 1.12 GiB | |
| KV cache 留了多少 | `Available KV cache memory: ... GiB` | 4.61 GiB | |
| KV cache 能放几个 token | `GPU KV cache size: ... tokens` | 43,120 | |
| 最大并发 | `Maximum concurrency ...` | 21.05x | |

## 2. 跑测试（终端 2）
```bash
cd ~/ai-infra-journey-git/stage1/02-vllm-fp8
python ../01-vllm-baseline/bench.py
```
用的就是基线的 bench.py，一个字没改。结果存在这个文件夹的 `bench.json`。

## 3. 先预测，再跑（只说方向和原因，不用算）
1. 「模型占多少」：比 1.12 GiB 小、一样、还是大？
2. 「KV cache 能放几个 token」：比 43,120 多、一样、还是少？为什么？
3. batch 1 每个字（TPOT）：比 6.03 ms 快还是慢？
4. batch 64 每个字：比 10.86 ms 快还是慢？变化比 batch 1 大还是小？

## 4. 新的天花板（Claude 算，推测）
- 全部参数 596,049,920 个。其中词表那一层 151,936 × 1,024 = 155,582,464 个（Qwen3-0.6B 的 config，凭记忆，未核实）。
- 词表那一层 vLLM 一般不转 fp8，还是 2 个字节（凭记忆，未核实）。它同时是最后一层（输出用同一份），每写一个字都要读。
- 其余 440,467,456 个 × 1 字节 + 155,582,464 × 2 字节 ≈ 0.752 GB。
- ÷ 实测带宽 208.2 GB/s ≈ **3.61 ms**（基线天花板 5.73 ms）。
跑完对一下：「模型占多少」是不是约 0.70 GiB，就知道词表那一层有没有转。
