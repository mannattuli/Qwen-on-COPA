import json, re, string
from collections import defaultdict
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
THINKING = False
N = None

OUT = f"copa_text_{MODEL.split('/')[-1]}_{'think' if THINKING else 'nothink'}.jsonl"
print("CUDA available:", torch.cuda.is_available())

tok = AutoTokenizer.from_pretrained(MODEL)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="auto")
print("Model device:", model.device)

ds = load_dataset("aps/super_glue", "copa")
items = list(ds["train"]) + list(ds["validation"])
if N:
    items = items[:N]

def norm(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def make_prompt(it, swap):
    q = "CAUSE" if it["question"] == "cause" else "EFFECT"
    a, b = (it["choice2"], it["choice1"]) if swap else (it["choice1"], it["choice2"])
    return (f"{it['premise']} What was the {q} of this?\n"
            f"Option A: {a}\nOption B: {b}\n"
            "Reply with the exact text of the correct option and nothing else.")

def parse(text, it):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    t = norm(text)
    h1 = norm(it["choice1"]) in t
    h2 = norm(it["choice2"]) in t
    if h1 == h2:
        return None
    return 0 if h1 else 1

rows = []
for i, it in enumerate(items):
    for swap in (False, True):
        msgs = [{"role": "user", "content": make_prompt(it, swap)}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=THINKING)
        inp = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp,
                                 max_new_tokens=2048 if THINKING else 40,
                                 do_sample=False)
        raw = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = parse(raw, it)
        first_listed = None if pred is None else ((pred == 1) if swap else (pred == 0))
        rows.append({"idx": it["idx"], "question": it["question"], "swap": swap,
                     "gold": it["label"], "raw": raw, "pred": pred,
                     "first_listed": first_listed,
                     "correct": pred == it["label"]})
    print(i, flush=True)

with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print("\n=== RESULTS ===")
for swap in (False, True):
    for d in ("cause", "effect"):
        rs = [r for r in rows if r["question"] == d and r["swap"] == swap]
        c = sum(r["correct"] for r in rs)
        print(f"swap={swap} {d}: {c}/{len(rs)} = {c/len(rs):.1%}")
for d in ("cause", "effect"):
    by = defaultdict(dict)
    for r in rows:
        if r["question"] == d:
            by[r["idx"]][r["swap"]] = r["correct"]
    both = sum(v[False] and v[True] for v in by.values())
    print(f"{d} correct in BOTH orders: {both}/{len(by)} = {both/len(by):.1%}")
valid = [r for r in rows if r["pred"] is not None]
print("invalid answers:", len(rows) - len(valid), "of", len(rows))
print("chose first-listed option:", sum(r["first_listed"] for r in valid), "of", len(valid))
print("saved to", OUT)