"""LoRA r=8：「山中桂」「落第」各用 5 个不同随机种子生成 5 首，统计第一句各不相同的有几首。生成参数和 lora_poem.py 一致。"""
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16),
                                  "checkpoints/qwen3_0.6b_lora_r8").to("cuda").eval()

all_first = []
for title, genre in (("山中桂", "五言绝句"), ("落第", "七言绝句")):
    ids = tok(f"题目：{title}\n体裁：{genre}\n诗：", return_tensors="pt").input_ids.to("cuda")
    firsts = []
    print(f"\n## {title}（{genre}）")
    for seed in range(5):
        torch.manual_seed(seed)
        g = model.generate(ids, do_sample=True, temperature=0.8, top_k=0, top_p=1.0, max_new_tokens=64,
                           stop_strings=["\n\n"], tokenizer=tok, attention_mask=torch.ones_like(ids),
                           eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        poem = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).split("\n\n")[0].strip()
        firsts.append(re.split(r"[，。！？；\s]", poem)[0])
        print(f"种子 {seed}：{poem.replace(chr(10), ' / ')}")
    print(f"第一句各不相同：{len(set(firsts))}/5")
    all_first += firsts
print(f"\n10 首合计第一句各不相同：{len(set(all_first))}/10")
