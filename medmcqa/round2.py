"""Round 2: does editing ~1,000 more weights help? Starts from the saved edit (models/medmcqa-edit-qwen25).

1. Attribution on 1,500 practice questions: for each neuron in blocks 19-23, how much would removing its
   question-to-question letter vote (its average is re-absorbed by the counterweight) raise the right letter?
2. A "re-calibrate only" control and the 3 best neurons are applied for real (surgery = 896 weights, then the
   counterweight is re-solved without labels) and scored on the 1,000-question tryout set (practice 3000-3999).
3. If the best beats the current edit on dev, it is scored once on the 1,000 held-out test questions.

    python medmcqa/round2.py > medmcqa/results/qwen25/round2.log
"""
import json

import torch
from tqdm.auto import tqdm

import lab
from edit import OUT_DIR, mcnemar
from try_edits import calibrate_uniform, unletter_neurons

LAYERS = list(range(19, 24))
S = lab.splits()
dev, items = S["practice"][3000:4000], S["practice"][:1500]
tok, model = lab.load_model(OUT_DIR)
lm = lab.decoder(model)
eps = getattr(lm.norm, "variance_epsilon", 1e-6)
WUg = lm.embed_tokens.weight[lab.LETTER_IDS].float() * lm.norm.weight.float()
P = {L: WUg @ lm.layers[L].mlp.down_proj.weight.float() for L in LAYERS}

cap = {}
hooks = [lm.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: cap.__setitem__(L, a[0][0, -1].float()))
         for L in LAYERS]
hooks.append(lm.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("final", a[0][0, -1].float())))
N, K = len(items), lm.layers[LAYERS[0]].mlp.down_proj.weight.shape[1]
dla = torch.zeros(N, len(LAYERS), 4, K, dtype=torch.float16)
final = torch.zeros(N, 4)
with torch.no_grad():
    for n, it in enumerate(tqdm(items, desc="attribution", unit="q", mininterval=5)):
        model(**tok([lab.prompt(tok, it)], return_tensors="pt", add_special_tokens=False).to(lab.DEVICE), logits_to_keep=1)
        r = (cap["final"].pow(2).mean() + eps).sqrt()
        final[n] = (WUg @ cap["final"] / r).cpu()
        for k, L in enumerate(LAYERS):
            dla[n, k] = (P[L] * cap[L] / r).half().cpu()
for h in hooks:
    h.remove()

gold = torch.tensor([it["answer"] for it in items])
idx = torch.arange(N)
base_lp = final.log_softmax(-1)[idx, gold]
print(f"current edit, analysis accuracy {(final.argmax(-1) == gold).float().mean():.4f}", flush=True)
cands = []
for k, L in enumerate(LAYERS):
    d = dla[:, k].float()
    new = final[:, :, None] - (d - d.mean(0, keepdim=True))
    gain = (new.log_softmax(1)[idx, gold] - base_lp[:, None]).mean(0)
    acc = (new.argmax(1) == gold[:, None]).float().mean(0)
    cands += [(gain[j].item(), acc[j].item(), L, j) for j in range(K)]
cands.sort(reverse=True)
print("top neurons (gain in log-prob of right letter, est. accuracy):",
      [(f"L{L}.n{j}", round(g, 4), round(a, 4)) for g, a, L, j in cands[:8]], flush=True)
del dla

base = lab.score(model, tok, dev, desc="current edit / dev")
base_ok = base["logits"].argmax(-1) == base["target"]
print(f"current edit, dev accuracy {base_ok.float().mean():.4f}", flush=True)
bias = lm.layers[23].self_attn.v_proj.bias
tests = [("recalibrate only", None, None)] + [(f"L{L}.n{j}", L, j) for _, _, L, j in cands[:3]]
results = []
for name, L, j in tests:
    keep_b = bias.detach().clone()
    keep_W = lm.layers[L].mlp.down_proj.weight.detach().clone() if L is not None else None
    with torch.no_grad():
        if L is not None:
            unletter_neurons(model, L, [j])
        calibrate_uniform(model, tok, 23, S["practice"][:500])
    r = lab.score(model, tok, dev, desc=f"{name} / dev")
    ok = r["logits"].argmax(-1) == r["target"]
    results.append({"edit": name, "layer": L, "neuron": j, "dev_accuracy": ok.float().mean().item(), **mcnemar(base_ok, ok)})
    print(f"{name}: dev accuracy {results[-1]['dev_accuracy']:.4f} (fixed {results[-1]['fixed']}, broke {results[-1]['broke']})", flush=True)
    with torch.no_grad():
        bias.copy_(keep_b)
        if L is not None:
            lm.layers[L].mlp.down_proj.weight.copy_(keep_W)

report = {"current_edit_dev_accuracy": base_ok.float().mean().item(), "dev_tests": results}
best = max(results, key=lambda x: x["dev_accuracy"])
if best["dev_accuracy"] > report["current_edit_dev_accuracy"]:
    with torch.no_grad():
        if best["layer"] is not None:
            unletter_neurons(model, best["layer"], [best["neuron"]])
        calibrate_uniform(model, tok, 23, S["practice"][:500])
    r = lab.score(model, tok, S["eval"], desc=f"{best['edit']} / held-out test")
    prev = torch.load(lab.RESULTS / "edited_eval.pt")
    report["held_out_test"] = {"edit": best["edit"], "accuracy": lab.accuracy(r), "current_edit_accuracy": lab.accuracy(prev),
                               **mcnemar(prev["logits"].argmax(-1) == prev["target"], r["logits"].argmax(-1) == r["target"])}
    print("held-out 1,000 test:", report["held_out_test"], flush=True)
else:
    print("no candidate beat the current edit on dev; held-out test not run", flush=True)
json.dump(report, open(lab.RESULTS / "round2.json", "w"), indent=1)
print("DONE", flush=True)
