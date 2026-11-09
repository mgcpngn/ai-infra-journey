#!/usr/bin/env bash
# 一次跑完 4 种设置，每种跑 REPEAT 次（默认 3）。人不用守着，跑完看最后的表。
# 11-09 用户：「重启服务跑测试然后再重复，这个过程非常磨人没有价值」。
# 运行：conda activate vllm && cd ~/ai-infra-journey-git/stage1/03-vllm-fp8-kv && bash sweep.sh
# 只跑 1 次：REPEAT=1 bash sweep.sh
set -u
cd "$(dirname "$0")"
REPEAT=${REPEAT:-3}
BENCH=${BENCH:-../01-vllm-baseline/bench.py}
mkdir -p runs

# 名字|启动参数。一行一种设置
CONFIGS=(
  "bf16|"
  "fp8权重|--quantization fp8"
  "A_fp8权重_Triton|--quantization fp8 --attention-backend TRITON_ATTN"
  "B_fp8权重_fp8KV_Triton|--quantization fp8 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN"
)

for cfg in "${CONFIGS[@]}"; do
  name=${cfg%%|*}; args=${cfg#*|}
  for i in $(seq 1 "$REPEAT"); do
    echo "=== $name 第 $i 次（$(date +%H:%M:%S)）==="
    log=runs/${name}_$i.log
    # $args 不加引号：要拆成几个参数
    HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
      vllm serve Qwen/Qwen3-0.6B-Base $args --max-model-len 2048 --gpu-memory-utilization 0.8 --port 8000 > "$log" 2>&1 &
    PID=$!
    ok=1
    until curl -s localhost:8000/v1/models > /dev/null; do
      sleep 2
      if ! kill -0 $PID 2>/dev/null; then ok=0; break; fi
    done
    if [ $ok = 1 ]; then
      python "$BENCH" > /dev/null 2>&1 && mv bench.json "runs/${name}_$i.json" || echo "  bench.py 出错"
    else
      echo "  服务没起来，看 $log："; grep -E "fatal|Error" "$log" | tail -3
    fi
    kill $PID 2>/dev/null; wait $PID 2>/dev/null   # 等服务完全退出，显存放出来，再开下一个
  done
done

python summary.py
