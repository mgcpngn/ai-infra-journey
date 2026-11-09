# 1 月-3：vLLM 把 KV cache 也转成 fp8（11-09）

1 月-2 只把权重转成 fp8，日志 `kv_cache_dtype=auto`：KV cache 还是 bf16，每个 token 112 KiB。
这次加 `--kv-cache-dtype fp8`。

## 11-09 第一次：服务没起来
- `Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN']`：KV cache 是 fp8 时，FLASH_ATTN 不能用。
- `/usr/bin/nvcc ... nvcc fatal : Unknown option '--compress-mode=size'`：FlashInfer 要现场编译，系统的 nvcc 太旧。和 `VLLM_USE_FLASHINFER_SAMPLER=0` 同一个原因。
- 办法：指定 `TRITON_ATTN`。

## 要跑两次（一次只改一件事）
| 次 | 权重 | KV cache | 注意力算法 | 和谁比 |
|---|---|---|---|---|
| A | fp8 | bf16 | TRITON_ATTN | 1 月-2（FLASH_ATTN）：只看换算法的影响 |
| B | fp8 | fp8 | TRITON_ATTN | A：只看 KV cache 换 fp8 的影响 |

每次一个终端（A 去掉 `--kv-cache-dtype fp8`）：
```bash
conda activate vllm
cd ~/ai-infra-journey-git/stage1/03-vllm-fp8-kv
HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
nohup vllm serve Qwen/Qwen3-0.6B-Base --quantization fp8 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN \
  --max-model-len 2048 --gpu-memory-utilization 0.8 --port 8000 > server.log 2>&1 &
PID=$!
until curl -s localhost:8000/v1/models > /dev/null; do
  sleep 2
  if ! kill -0 $PID 2>/dev/null; then echo "服务没起来"; grep -E "fatal|Error" server.log | tail -5; break; fi
done
python ../01-vllm-baseline/bench.py
grep -E "attention backend|Model loading took|GPU KV cache size|Maximum concurrency" server.log
kill $PID
```
`--attention-backend` 不认识时：去掉它，最前面加 `VLLM_ATTENTION_BACKEND=TRITON_ATTN`（凭记忆，未核实）。

## 先预测
1. 每个 token 的 KV cache 从 112 KiB 变成多少？能放的 token（1 月-2 是 44,928）约几倍？
2. batch 64 每个字（1 月-2 是 10.58 ms）会变快吗？为什么？batch 1 呢？
