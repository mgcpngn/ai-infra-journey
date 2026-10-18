# 第 1 月第 1 步：用 vLLM 跑 Qwen3-0.6B，看同时来很多请求时会怎样

只用到已经学过的概念：KV cache、batch、prefill（第一个字出来之前）、decode（之后一个字一个字地写）、显存。
没有新词。

## 先预测（跑之前写下来）
1. 同时来的请求从 1 个变成 64 个，**所有请求加起来**每秒写出的字数会怎么变？
2. 同样的变化，**每一个请求**等第一个字的时间、写完的时间会怎么变？
3. vLLM 启动时会先占掉一大块显存。你猜它拿这块显存做什么？

## 提示词（给本机 Claude Code）

```
在 /home/wangli/step1 下新建目录 vllm_baseline，完成下面的事。总时间控制在 20 分钟内。

0. 安装 vLLM：pip install vllm。如果在 WSL 上装不上或启动报错，超过 15 分钟就停下，把完整报错存到 install_error.txt 交给我，不要换别的软件。

1. 启动服务：
   vllm serve Qwen/Qwen3-0.6B-Base --max-model-len 2048 --gpu-memory-utilization 0.8 --port 8000
   把启动日志存到 server.log。从日志里找出并打印：
   - 模型本身占了多少显存
   - 给 KV cache 留了多少显存、能放下多少个 token
   - 日志里写的最大并发数（Maximum concurrency 一类的字样）

2. 写 bench.py：用 asyncio + 流式请求（OpenAI 兼容接口 /v1/completions，stream=True），
   - 每个请求的输入固定为同一段约 64 个 token 的中文，要求输出正好 128 个 token（max_tokens=128，ignore_eos=True），temperature=0。
   - 同时发出 N 个请求，N 依次取 1、4、16、64。每个 N 先空跑一次热身，再正式跑一次。
   - 每个请求记录：从发出到收到第一个字的时间（毫秒）、之后平均每个字的时间（毫秒）、整个请求的时间（秒）。
   - 每个 N 打印一行：N、所有请求加起来每秒写出多少字、第一个字的平均时间、每个字的平均时间、整个请求的平均时间、跑的时候 nvidia-smi 看到的显存。

3. 把第 1、2 步的结果汇总成 results.md 的两张表。

4. 关掉服务。
```

## 跑完后用自己的话回答（不考计算）
1. N 从 1 到 64，总速度变了多少倍？每个请求自己慢了多少？这和第 5 步你自己的小 GPT 一样吗？
2. vLLM 启动时占的那一大块显存是拿来做什么的？
3. 如果 N 再变大很多（比如 1000），你觉得会先撞到什么上限？
