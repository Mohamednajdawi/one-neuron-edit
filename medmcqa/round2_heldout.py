"""Round 2 on all held-out data: rebuild the 2-neuron edit exactly as tested in round2.py, save it, and score
the 1,000 test, the other 3,183 validation and the 5,000 fresh questions (all 4 option orders) against round 1.

    python medmcqa/round2_heldout.py     # -> models/medmcqa-edit-qwen25-r2 and results/qwen25/round2_heldout.json
"""
import json

import torch
from safetensors import safe_open

import lab
from edit import OUT_DIR, WEIGHTS, changed_entries, mcnemar, mean_nll, save, side_effect_texts
from try_edits import calibrate_uniform, unletter_neurons

R2_DIR = lab.ROOT / "models/medmcqa-edit-qwen25-r2"
RES = lab.RESULTS

S = lab.splits(n_fresh=5000)
tok, model = lab.load_model(OUT_DIR)
with torch.no_grad():
    unletter_neurons(model, 21, [114])
    calibrate_uniform(model, tok, 23, S["practice"][:500])
edits = changed_entries(model, safe_open(lab.MODEL_DIR / WEIGHTS, "pt"))
total = sum(p.numel() for p in model.parameters())
n_changed = sum(edits.values())
print(f"round 2 edit: {n_changed:,} of {total:,} weights changed vs the original model", edits, flush=True)
save(model, edits, R2_DIR)
report = {"edit": {"tensors": edits, "changed_entries": n_changed, "total_weights": total,
                   "recipe": "round 1 + remove the letter-vote part of block 21 neuron 114, re-solve the counterweight"},
          "medical_prose_nll": mean_nll(model, tok, side_effect_texts())}
print("medical-prose loss:", round(report["medical_prose_nll"], 4), flush=True)


def ok(r):
    return r["logits"].argmax(-1) == r["target"]


for name in ("eval", "val_rest"):
    r = lab.score(model, tok, S[name], desc=f"round 2 / {name}")
    torch.save(r, RES / f"round2_{name}.pt")
    r1, r0 = torch.load(RES / f"edited_{name}.pt"), torch.load(RES / f"baseline_{name}.pt")
    report[name] = {"accuracy": lab.accuracy(r), "round1_accuracy": lab.accuracy(r1), "baseline_accuracy": lab.accuracy(r0),
                    "vs_round1": mcnemar(ok(r1), ok(r)), "vs_baseline": mcnemar(ok(r0), ok(r)),
                    "predicted_letters": lab.summary(r, S[name])["predicted_letters"]}
    print(name, {k: v for k, v in report[name].items() if k != "predicted_letters"}, flush=True)

rot = [lab.score(model, tok, S["fresh"], order=o, desc=f"round 2 / fresh order {k + 1}/4") for k, o in enumerate(lab.ROTATIONS)]
torch.save(rot, RES / "round2_fresh.pt")
prev = torch.load(RES / "test5000.pt")
report["fresh"] = {"accuracy": lab.accuracy(rot[0]), "rotation_accuracies": [lab.accuracy(r) for r in rot],
                   "circular_accuracy": sum(lab.accuracy(r) for r in rot) / 4,
                   "consistency": lab.consistency(rot, S["fresh"]),
                   "predicted_letters": lab.summary(rot[0], S["fresh"])["predicted_letters"],
                   "vs_round1": mcnemar(ok(prev["after"][0]), ok(rot[0])), "vs_baseline": mcnemar(ok(prev["before"][0]), ok(rot[0]))}
print("fresh", {k: v for k, v in report["fresh"].items() if k not in ("predicted_letters", "consistency")}, flush=True)
print("fresh consistency", report["fresh"]["consistency"], flush=True)
json.dump(report, open(RES / "round2_heldout.json", "w"), indent=1)
print("DONE", flush=True)
