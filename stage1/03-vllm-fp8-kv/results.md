# 1 月-3 结果：KV cache 也转成 fp8（11-09，用户本地，conda 环境 vllm）

## B：权重 fp8 + KV cache fp8 + TRITON_ATTN
日志：`non-default args: {... 'quantization': 'fp8', 'attention_backend': 'TRITON_ATTN', ... 'kv_cache_dtype': 'fp8'}`；
`Model loading took 0.71 GiB`；`GPU KV cache size: 89,920 tokens, Maximum concurrency for 2,048 tokens per request: 43.91x`。

| N | 总吞吐 tok/s | TTFT ms | TPOT ms | 请求时间 s | 显存 MiB |
|---|---|---|---|---|---|
| 1 | 217.9 | 32.6 | 4.37 | 0.59 | 6425 |
| 4 | 794.7 | 65.8 | 4.54 | 0.64 | 6427 |
| 16 | 2661.4 | 73.1 | 5.47 | 0.77 | 6427 |
| 64 | 6879.7 | 159.2 | 8.05 | 1.18 | 6447 |

和 1 月-2（权重 fp8、KV cache bf16、FLASH_ATTN）比（Claude 算）：
- KV cache 能放的 token：44,928 → 89,920，2.00 倍。每个 token 112 KiB → 56 KiB（每个数 2 字节 → 1 字节）。
- TPOT：N=1 4.61 → 4.37；N=64 10.58 → 8.05（快 24%）。
- TTFT 变慢：N=1 23.4 → 32.6；N=64 95.8 → 159.2。
- 这次同时换了 KV cache 和注意力算法，速度变化分不清是哪一个造成的。要跑 A（只换 TRITON_ATTN）。

## A：权重 fp8 + KV cache bf16 + TRITON_ATTN
（日志还没贴，按命令推测 KV cache 约 44,928 tokens。）

| N | 总吞吐 tok/s | TTFT ms | TPOT ms | 请求时间 s | 显存 MiB |
|---|---|---|---|---|---|
| 1 | 156.6 | 56.7 | 5.99 | 0.82 | 6425 |
| 4 | 799.1 | 44.7 | 4.68 | 0.64 | 6427 |
| 16 | 2569.7 | 59.3 | 5.79 | 0.79 | 6427 |
| 64 | 6301.2 | 116.2 | 9.26 | 1.29 | 6447 |

## 三次放在一起（batch 64 每个字）
| | 权重 | KV cache | 注意力 | TPOT ms |
|---|---|---|---|---|
| 1 月-2 | fp8 | bf16 | FLASH_ATTN | 10.58 |
| A | fp8 | bf16 | TRITON_ATTN | 9.26 |
| B | fp8 | fp8 | TRITON_ATTN | 8.05 |

- 一共快了 2.53 ms：换 Triton 占 1.32 ms，KV cache 换 fp8 占 1.21 ms，差不多一半一半。
- 用户预测 A 接近 10.6（全是 KV cache 的功劳）：一半对。
- A 的 N=1 不可信：5.99 ms 比 N=4 的 4.68 还慢，TTFT 56.7 也比 N=4 高。每种只跑了 1 次（推测：第一次请求时还有东西没准备好）。要下结论，每种要跑 3 次取中间的数。
