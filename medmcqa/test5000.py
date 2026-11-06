"""Before/after test of Edit 1 on the mock-exam test: 5,000 fresh questions (never used for analysis or selection),
each in all 4 cyclic option orders.

    python medmcqa/test5000.py           # needs models/medmcqa-edit-qwen25 (edit.py or apply_edit.py)
"""
import json
import time
from collections import Counter

import torch

import lab
from edit import OUT_DIR, mcnemar

S = lab.splits(n_fresh=5000)
items = S["fresh"]
practice_q = {it["question"].strip().lower() for it in S["practice"]}
assert len(items) == 5000 and not practice_q & {it["question"].strip().lower() for it in items}
print("fresh questions:", len(items), "| gold letters:", sorted(Counter("ABCD"[it["answer"]] for it in items).items()), flush=True)

t0 = time.time()
report, results, rotations = {}, {}, {}
for name, path in [("before", lab.MODEL_DIR), ("after", OUT_DIR)]:
    tok, model = lab.load_model(path)
    rotations[name] = []
    for k, order in enumerate(lab.ROTATIONS):
        r = lab.score(model, tok, items, order=order, desc=f"{name} order {k + 1}/4")
        rotations[name].append(r)
        print(f"[{(time.time() - t0) / 60:.1f} min] {name}: pass {k + 1}/4 done (accuracy {lab.accuracy(r):.4f})", flush=True)
    results[name] = rotations[name][0]  # rotation 0 is the normal option order
    rot_acc = [lab.accuracy(r) for r in rotations[name]]
    report[name] = lab.summary(results[name], items)
    report[name]["circular_accuracy"] = sum(rot_acc) / 4
    report[name]["rotation_accuracies"] = rot_acc
    report[name]["consistency"] = lab.consistency(rotations[name], items)
    print(name, f"accuracy {report[name]['accuracy']:.4f} | circular {report[name]['circular_accuracy']:.4f} | pred",
          {k: round(v, 3) for k, v in report[name]["predicted_letters"].items()}, flush=True)
    print(name, "consistency:", {k: (round(v, 4) if isinstance(v, float) else {a: round(b, 4) for a, b in v.items()})
                                 for k, v in report[name]["consistency"].items()}, flush=True)
    del model

ok = {k: r["logits"].argmax(-1) == r["target"] for k, r in results.items()}
report["paired"] = mcnemar(ok["before"], ok["after"])
report["gold_letters"] = report["before"]["gold_letters"]
report["always_A_accuracy"] = report["gold_letters"]["A"]
print("paired:", report["paired"], "| always-A:", round(report["always_A_accuracy"], 4), flush=True)
torch.save(rotations, lab.RESULTS / "test5000.pt")
json.dump(report, open(lab.RESULTS / "test5000.json", "w"), indent=1)
print("DONE", flush=True)
