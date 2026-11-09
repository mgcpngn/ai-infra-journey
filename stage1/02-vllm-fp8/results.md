# 1 月-2 结果：vLLM fp8，Qwen3-0.6B-Base，RTX 4060 Laptop 8GB（WSL2），11-09 用户本地跑出

环境：conda 环境 vllm（vllm 0.32.0、torch 2.13.0+cu130）。和同一天、同一环境的基线比（01-vllm-baseline/results.md 「11-09 新环境复跑」）。

## 1. 启动日志：确认是 fp8
| 日志原文 | 意思 |
|---|---|
| `non-default args: {... 'quantization': 'fp8' ...}` | 服务确实用 fp8 开的 |
| `--quantization fp8 is deprecated for online quantization; use --quantization fp8_per_tensor instead.` | 只是提醒名字要换，这次照样用了 fp8 |
| `Selected CutlassFP8ScaledMMLinearKernel for Fp8PerTensorOnlineLinearMethod` | 乘法用 fp8 的算子直接算，不是先还原（第 5 步那种） |
| `Quantizing 112 layers of types: mlp.down_proj: 28; mlp.gate_up_proj: 28; self_attn.o_proj: 28; self_attn.qkv_proj: 28` | 28 层 × 4 种 = 112 个转成 fp8。词表那一层（embed_tokens / lm_head）不在里面，还是 bf16 |

模型占多少、KV cache：待补（`Model loading took`、`GPU KV cache size`）。

## 2. 并发测试（bench.py，没改）
| N | 总吞吐 tok/s | TTFT ms | TPOT ms | 请求时间 s | 显存 MiB |
|---|---|---|---|---|---|
| 1 | 210.1 | 23.4 | 4.61 | 0.61 | 6443 |
| 4 | 768.7 | 32.0 | 4.98 | 0.66 | 6445 |
| 16 | 2701.4 | 43.6 | 5.61 | 0.76 | 6445 |
| 64 | 5662.9 | 95.8 | 10.58 | 1.44 | 6465 |

## 3. 和 bf16 比（Claude 算）
| N | bf16 TPOT | fp8 TPOT | 快了 |
|---|---|---|---|
| 1 | 6.20 | 4.61 | 26% |
| 4 | 6.62 | 4.98 | 25% |
| 16 | 7.35 | 5.61 | 24% |
| 64 | 11.49 | 10.58 | 8% |

- 4.61 ms 比 bf16 的天花板 5.73 ms 还快：只有要搬的字节变少才做得到。
- fp8 天花板：440,467,456 × 1 字节 + 155,582,464 × 2 字节（词表那一层没转，日志证实）≈ 0.752 GB ÷ 208.2 GB/s ≈ 3.61 ms。实测 4.61，是天花板的 1.28 倍；bf16 是 6.20 ÷ 5.73 ≈ 1.08 倍。
- N=64 只快 8%：这时不只是搬权重（推测：还有 64 份 KV cache 和计算）。
- TTFT 差不多（19.1 → 23.4 ms），两次 Prefix cache hit rate 不一样（38.1% / 75.7%），不好比。

## 4. 结论
权重变小、而且乘的时候直接用 fp8，batch 小、卡在搬权重时就快（26%）；batch 大时快得少（8%）。
