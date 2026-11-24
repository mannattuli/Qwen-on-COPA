import json, random
from collections import Counter

FILE = "copa_text_Qwen3-0.6B_nothink.jsonl"
rows = [json.loads(l) for l in open(FILE, encoding="utf-8")]
assert len(rows) % 2 == 0

items = []
for k in range(0, len(rows), 2):
    a, b = rows[k], rows[k + 1]
    assert a["swap"] is False and b["swap"] is True
    assert a["idx"] == b["idx"] and a["question"] == b["question"]
    items.append({"dir": a["question"], "gold": a["gold"],
                  "c0": int(a["correct"]), "c1": int(b["correct"]),
                  "p0": a["pred"], "p1": b["pred"]})

def score(it):
    return (it["c0"] + it["c1"]) / 2

by = {d: [it for it in items if it["dir"] == d] for d in ("cause", "effect")}

for d, its in by.items():
    n = len(its)
    acc = sum(score(i) for i in its) / n
    both = sum(i["c0"] and i["c1"] for i in its)
    same = sum(i["p0"] is not None and i["p0"] == i["p1"] for i in its)
    gold = Counter(i["gold"] for i in its)
    print(f"{d}: n={n} acc(avg of orders)={acc:.1%} both-orders={both}/{n}={both/n:.1%} "
          f"same-answer-in-both-orders={same}/{n}={same/n:.1%} gold=choice1:{gold[0]} choice2:{gold[1]}")

random.seed(0)
B = 10000
gaps = []
for _ in range(B):
    s = {}
    for d, its in by.items():
        sample = random.choices(its, k=len(its))
        s[d] = sum(score(i) for i in sample) / len(sample)
    gaps.append(s["cause"] - s["effect"])
gaps.sort()
obs = (sum(score(i) for i in by["cause"]) / len(by["cause"])
       - sum(score(i) for i in by["effect"]) / len(by["effect"]))
print(f"cause minus effect = {obs*100:+.1f} points, 95% bootstrap CI "
      f"[{gaps[int(0.025*B)]*100:+.1f}, {gaps[int(0.975*B)]*100:+.1f}]")

allacc = [score(i) for i in items]
print(f"overall accuracy = {sum(allacc)/len(allacc):.1%}")