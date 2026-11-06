"""Apply Edit 1 (recipe.py), check the rules and the budget, evaluate on the real-exam test once, and save.

Rules the agent adopted: no training or gradients, no edits to the embedding / output layer / final norm, no
multiplying a whole tensor by a constant; changed entries <= 4% of all weights (the author's budget).

    python medmcqa/edit.py               # -> models/medmcqa-edit-qwen25 and results/qwen25/edited.json
"""
import json
import math
import random
import shutil

import torch
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import lab
import recipe

WEIGHTS = lab.WEIGHTS
OUT_DIR = lab.ROOT / "models/medmcqa-edit-qwen25"
RES = lab.RESULTS


def forbidden(model):
    final_norm = next(n for n, p in model.named_parameters() if p is lab.decoder(model).norm.weight)
    return lambda name: "embed_tokens" in name or name.startswith("lm_head") or name == final_norm


def changed_entries(model, original):
    is_forbidden = forbidden(model)
    out = {}
    for name, p in model.named_parameters():
        before, now = original.get_tensor(name).to(p.dtype), p.detach().cpu()
        if torch.equal(before, now):
            continue
        assert not is_forbidden(name), f"forbidden tensor edited: {name}"
        nz = before != 0
        ratio = now.float()[nz] / before.float()[nz]
        assert not torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-2), f"whole-tensor scaling: {name}"
        out[name] = int((before != now).sum())
    return out


def mcnemar(base_ok, new_ok):
    fixed, broke = int((new_ok & ~base_ok).sum()), int((~new_ok & base_ok).sum())
    n, k = fixed + broke, min(fixed, broke)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"fixed": fixed, "broke": broke, "p_two_sided": p}


def side_effect_texts(n=200):
    """Expert explanations of random train questions: ordinary medical prose, not multiple choice."""
    exp = [e for e in load_dataset("openlifescienceai/medmcqa")["train"]["exp"] if e and len(e) > 300]
    return random.Random(2).sample(exp, n)


@torch.no_grad()
def mean_nll(model, tok, texts):
    total, count = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=256).input_ids.to(lab.DEVICE)
        logits = model(input_ids=ids).logits[0, :-1].float()
        total += torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum").item()
        count += ids.shape[1] - 1
    return total / count


def evaluate(model, tok, S, base):
    out = {}
    for name in ("eval", "val_rest"):
        r = lab.score(model, tok, S[name])
        torch.save(r, RES / f"edited_{name}.pt")
        out[name] = lab.summary(r, S[name])
        b = torch.load(RES / f"baseline_{name}.pt")
        out[name]["vs_baseline"] = mcnemar(b["logits"].argmax(-1) == b["target"], r["logits"].argmax(-1) == r["target"])
        out[name]["baseline_accuracy"] = base[name]["accuracy"]
        print(name, f"{base[name]['accuracy']:.4f} -> {out[name]['accuracy']:.4f}", out[name]["vs_baseline"], flush=True)
    out["eval"]["circular_accuracy"] = lab.circular_accuracy(model, tok, S["eval"])
    out["eval"]["baseline_circular_accuracy"] = base["eval"]["circular_accuracy"]
    print("eval circular", f"{base['eval']['circular_accuracy']:.4f} -> {out['eval']['circular_accuracy']:.4f}")
    return out


def save(model, edits, out_dir=OUT_DIR):
    shutil.copytree(lab.MODEL_DIR, out_dir, ignore=shutil.ignore_patterns(".cache", "eval", WEIGHTS), dirs_exist_ok=True)
    tensors = load_file(lab.MODEL_DIR / WEIGHTS)
    for name in edits:
        new, old = model.get_parameter(name).detach().cpu(), tensors[name]
        changed = new != old.to(new.dtype)  # untouched entries keep the file's full precision (float32 models)
        old[changed] = new[changed].to(old.dtype)
    save_file(tensors, out_dir / WEIGHTS, metadata={"format": "pt"})


if __name__ == "__main__":
    S = lab.splits()
    tok, model = lab.load_model()
    original = safe_open(lab.MODEL_DIR / WEIGHTS, "pt")
    texts = side_effect_texts()
    nll_before = mean_nll(model, tok, texts)
    with torch.no_grad():
        notes = recipe.apply(model, tok, S["practice"])
    nll_after = mean_nll(model, tok, texts)
    print(f"medical-prose loss per token: {nll_before:.4f} -> {nll_after:.4f}")
    edits = changed_entries(model, original)
    total = sum(p.numel() for p in model.parameters())
    n_changed = sum(edits.values())
    print(f"changed {n_changed:,} of {total:,} weights ({n_changed / total:.4%})", edits)
    assert n_changed <= 0.04 * total
    report = {"edit": {"tensors": edits, "changed_entries": n_changed, "total_weights": total,
                       "fraction": n_changed / total, "notes": notes},
              "side_effects": {"medical_prose_nll_before": nll_before, "medical_prose_nll_after": nll_after}}
    report.update(evaluate(model, tok, S, json.load(open(RES / "baseline.json"))))
    json.dump(report, open(RES / "edited.json", "w"), indent=1)
    save(model, edits)
    print("saved", OUT_DIR)
