import asyncio, json, subprocess, time, sys
import aiohttp
from transformers import AutoTokenizer

URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen3-0.6B-Base"
TEXT = ("人工智能正在深刻地改变我们的生活方式。从智能手机上的语音助手，到医院里辅助医生诊断的系统，"
        "再到工厂中自动运转的机器人，这些技术已经渗透到社会的各个角落。与此同时，人们也开始思考它带来的"
        "就业、隐私和伦理问题，以及如何让技术更好地服务于每一个人。")
tok = AutoTokenizer.from_pretrained(MODEL)
PROMPT = tok.decode(tok(TEXT)["input_ids"][:64])
PROMPT_LEN = len(tok(PROMPT)["input_ids"])

async def one(session):
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": 128, "ignore_eos": True,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; chunks = 0; ntok = None
    async with session.post(URL, json=body) as r:
        async for line in r.content:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                ntok = d["usage"]["completion_tokens"]
            if d.get("choices") and d["choices"][0]["text"]:
                chunks += 1
                if first is None:
                    first = time.perf_counter()
    t1 = time.perf_counter()
    ntok = ntok or chunks
    return {"ttft_ms": (first - t0) * 1e3, "tpot_ms": (t1 - first) * 1e3 / (ntok - 1),
            "e2e_s": t1 - t0, "ntok": ntok}

def gpu_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])

async def run(n):
    peak = 0; done = False
    async def sampler():
        nonlocal peak
        while not done:
            peak = max(peak, await asyncio.to_thread(gpu_mem)); await asyncio.sleep(0.2)
    s = asyncio.create_task(sampler())
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        t0 = time.perf_counter()
        res = await asyncio.gather(*[one(session) for _ in range(n)])
        wall = time.perf_counter() - t0
    done = True; await s
    return res, wall, peak

async def main():
    print(f"prompt tokens = {PROMPT_LEN}", file=sys.stderr)
    rows = []
    print("N\ttok/s\tTTFT_ms\tTPOT_ms\tE2E_s\tGPU_MiB")
    for n in (1, 4, 16, 64):
        await run(n)  # warmup
        res, wall, peak = await run(n)
        avg = lambda k: sum(r[k] for r in res) / n
        tput = sum(r["ntok"] for r in res) / wall
        row = dict(N=n, tput=tput, ttft=avg("ttft_ms"), tpot=avg("tpot_ms"), e2e=avg("e2e_s"),
                   mem=peak, ntok=sorted({r["ntok"] for r in res}))
        rows.append(row)
        print(f"{n}\t{tput:.1f}\t{row['ttft']:.1f}\t{row['tpot']:.2f}\t{row['e2e']:.2f}\t{peak}", flush=True)
    json.dump({"prompt_tokens": PROMPT_LEN, "rows": rows}, open("bench.json", "w"), indent=1)

asyncio.run(main())
