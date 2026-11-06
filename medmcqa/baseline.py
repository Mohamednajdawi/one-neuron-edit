"""Baseline zero-shot MedMCQA accuracy of the downloaded model (thinking off, letter scoring)."""
import json

import torch

import lab

OUT = lab.RESULTS

S = lab.splits()
tok, model = lab.load_model()
report = {}
for name in ("eval", "val_rest", "practice"):
    res = lab.score(model, tok, S[name])
    torch.save(res, OUT / f"baseline_{name}.pt")
    report[name] = lab.summary(res, S[name])
    print(name, f"accuracy {report[name]['accuracy']:.4f}", report[name]["predicted_letters"], flush=True)
report["eval"]["circular_accuracy"] = lab.circular_accuracy(model, tok, S["eval"])
report["eval"]["always_A_accuracy"] = report["eval"]["gold_letters"]["A"]
print("eval circular accuracy", report["eval"]["circular_accuracy"])
json.dump(report, open(OUT / "baseline.json", "w"), indent=1)
