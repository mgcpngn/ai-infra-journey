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
待跑。
