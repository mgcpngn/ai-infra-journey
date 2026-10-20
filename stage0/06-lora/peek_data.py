"""看训练数据：诗表有多大、数字表有多大。只用 CPU，不训练，不用 GPU。

读诗、挑绝句、变成数字的做法和 lora_poem.py 第 72～119 行一样。
运行：cd /home/wangli/step1 && python peek_data.py
没有 pandas 时先装：pip install pandas
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import zhconv
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B-Base"
L = 64      # 每首诗变成几个数，和 lora_poem.py 第 28 行一样
BATCH = 16  # 每一步挑几首，和 lora_poem.py 第 256 行一样
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)

# ---------- 环节 1：读所有 json 文件。一次读完一个文件，每首诗放进表里的一行 ----------
files = sorted(glob.glob(os.path.join("data", "tang_raw", "poet.tang.*.json")))
rows = []
for f in files:
    for p in json.load(open(f, encoding="utf-8")):
        rows.append({"文件": os.path.basename(f), "题目": p["title"], "作者": p.get("author", ""),
                     "诗句": "".join(p["paragraphs"]).strip()})
raw = pd.DataFrame(rows)
print(f"\n[环节 1] json 文件 {len(files)} 个，诗 {len(raw):,} 首。表的大小 {raw.shape}（行, 列）")
print(raw.head(3))
print("每个文件有几首（前 5 个文件）：")
print(raw["文件"].value_counts(sort=False).head(5))


# ---------- 环节 2：只留绝句（规则和 lora_poem.py 第 78～85 行一样） ----------
def genre_of(t):
    if not t or any(c in t for c in "□{}[]（）()"):
        return None
    s = re.findall(r"[^，。！？；]+[，。！？；]", t)
    if "".join(s) != t or len(s) != 4 or len({len(x) for x in s}) != 1 or len(s[0]) - 1 not in (5, 7):
        return None
    return "五言绝句" if len(s[0]) - 1 == 5 else "七言绝句"


raw["体裁"] = raw["诗句"].map(genre_of)
q = raw[raw["体裁"].notna()].copy()
q["题目"] = q["题目"].map(lambda t: zhconv.convert(t, "zh-cn").strip())
q["诗句"] = q["诗句"].map(lambda t: zhconv.convert(t, "zh-cn"))
q = q.reset_index(drop=True)
print(f"\n[环节 2] 绝句 {len(q):,} 首（lora_poem.py 第 100 行打印的是 14,475）。表的大小 {q.shape}")
print(q["体裁"].value_counts())

# ---------- 环节 3：每首诗变成 token 编号（做法和 lora_poem.py 第 111～112 行一样） ----------
tok = AutoTokenizer.from_pretrained(MODEL)


def ids_of(title, genre, text):
    s = re.findall(r"[^，。！？；]+[，。！？；]", text)
    a = tok(f"题目：{title}\n体裁：{genre}\n诗：").input_ids
    b = tok(s[0] + s[1] + "\n" + s[2] + s[3]).input_ids + [tok.eos_token_id]
    return a + b


q["ids"] = [ids_of(r.题目, r.体裁, r.诗句) for r in q.itertuples()]
q["token数"] = q["ids"].map(len)
print(f"\n[环节 3] 每首诗有几个 token：")
print(q["token数"].describe().round(1))
print(f"超过 {L} 个 token、会被截断的：{(q['token数'] > L).sum()} 首")

# ---------- 环节 4：排成一张数字表，每首诗一行，每行 L 个数，不够的用 pad 补上 ----------
X = np.full((len(q), L), tok.pad_token_id, dtype=np.int64)
for i, ids in enumerate(q["ids"]):
    X[i, :min(len(ids), L)] = ids[:L]
Xdf = pd.DataFrame(X)
print(f"\n[环节 4] 数字表的大小 {Xdf.shape}（行 = 诗，列 = token 位置）")
print(f"每个数 {X.itemsize} 字节，整张表 {X.nbytes / 1024 ** 2:.1f} MB。"
      f"lora_poem.py 有 3 张这样的表（X、M、Y，第 106～108 行），一共约 {3 * X.nbytes / 1024 ** 2:.1f} MB")
print(Xdf.iloc[:3, :12])
print(f"第 0 行前 12 个数各代表什么：{[tok.decode([t]) for t in X[0, :12]]}")

# ---------- 环节 5：模拟训练的一步：随机挑 BATCH 行 ----------
ix = np.random.default_rng(0).integers(len(X), size=BATCH)
print(f"\n[环节 5] 随机挑的行号：{ix.tolist()}")
print(f"挑出来的表大小 {X[ix].shape}。这就是一步交给模型的数据")
print(q.loc[ix, ["题目", "体裁", "诗句"]].head(5))

q.drop(columns="ids").to_csv("poems.csv", index=False, encoding="utf-8-sig")
print("\n所有绝句存进 poems.csv，可以用 Excel 或 pandas 打开")
