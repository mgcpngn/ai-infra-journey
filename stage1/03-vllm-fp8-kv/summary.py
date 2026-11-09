"""读 runs/ 里每次的结果，每种设置取 3 次的中间数（中位数），打印一张表。sweep.sh 最后自动运行。"""
import glob
import json
import re
import statistics

runs = {}
for f in sorted(glob.glob("runs/*.json")):
    name = f[5:].rsplit("_", 1)[0]
    runs.setdefault(name, []).append(json.load(open(f))["rows"])

def kv_tokens(name):
    """从这种设置第 1 次的日志里找 GPU KV cache size。"""
    try:
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", open(f"runs/{name}_1.log").read())
        return m.group(1) if m else "?"
    except OSError:
        return "?"

print("\n每种设置跑了几次取中位数。TPOT = 每个字 ms，TTFT = 第一个字 ms。括号里是最小～最大。")
print(f"{'设置':<26}{'次数':>4}{'KV token':>10}  {'N':>3}  {'TPOT':>18}  {'TTFT':>22}")
for name, rs in runs.items():
    for k, n in enumerate(r["N"] for r in rs[0]):
        tpot = [r[k]["tpot"] for r in rs]
        ttft = [r[k]["ttft"] for r in rs]
        head = f"{name:<26}{len(rs):>4}{kv_tokens(name):>10}" if k == 0 else " " * 40
        print(f"{head}  {n:>3}  {statistics.median(tpot):6.2f} ({min(tpot):.2f}～{max(tpot):.2f})"
              f"  {statistics.median(ttft):7.1f} ({min(ttft):.1f}～{max(ttft):.1f})")
