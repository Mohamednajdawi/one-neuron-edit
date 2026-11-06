"""Is the answer present inside the original model? Run after the edits as a check on the diagnosis.

1) Answer content at the first answer token: rotation-averaged accuracy (the 4 cyclic option orders cancel
   letter habits).
2) Linear probes: can the right displayed letter be decoded from the residual stream at several blocks?
   Fit on 800 practice questions, tested on 400 others, all in 4 orders (chance = 25%). Control: the same probe
   on option lengths only.

    python medmcqa/probes.py             # -> results/qwen25/probes.json
"""
import json

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import lab

BLOCKS = [6, 10, 14, 17, 20, 22, 23]
N, N_TRAIN, N_FIT = 1200, 800, 600

items = lab.splits()["practice"][:N]
gold = torch.tensor([it["answer"] for it in items])
target = torch.stack([torch.tensor([o.index(g) for g in gold.tolist()]) for o in lab.ROTATIONS])  # displayed gold [4, N]
tok, model = lab.load_model()
lm = lab.decoder(model)
cap = {}
for i in BLOCKS:
    lm.layers[i].register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, -1].float()))


@torch.no_grad()
def run():
    logits = torch.zeros(4, N, 4)
    res = torch.zeros(4, N, len(BLOCKS), lm.config.hidden_size, dtype=torch.float16)
    for k, order in enumerate(lab.ROTATIONS):
        for j, it in enumerate(tqdm(items, desc=f"order {k + 1}/4", unit="q", mininterval=10)):
            out = model(**tok([lab.prompt(tok, it, order)], return_tensors="pt", add_special_tokens=False).to(lab.DEVICE),
                        logits_to_keep=1)
            logits[k, j] = out.logits[0, -1, lab.LETTER_IDS].float().cpu()
            for b, i in enumerate(BLOCKS):
                res[k, j, b] = (cap[i] / cap[i].pow(2).mean().sqrt()).half().cpu()
    return logits, res


def content(logits):
    lp = logits.log_softmax(-1)
    by_option = torch.stack([torch.stack([lp[k, :, o.index(j)] for j in range(4)], -1)
                             for k, o in enumerate(lab.ROTATIONS)]).mean(0)
    return {"content": round((by_option.argmax(-1) == gold).float().mean().item() * 100, 1),
            "balanced": round((logits.argmax(-1) == target).float().mean().item() * 100, 1),
            "letter_preference": [round(v * 100, 1) for v in lp.exp().mean((0, 1)).tolist()]}


def probe_accuracy(X):
    """X [4, N, d] features per (order, question); multinomial logistic regression, L2 chosen on a validation split."""
    y = target

    def fit(tr, lam):
        Xt, yt = X[:, tr].reshape(-1, X.shape[-1]), y[:, tr].reshape(-1)
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        W = torch.zeros(X.shape[-1], 4, requires_grad=True)
        b = torch.zeros(4, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy((Xt - mu) / sd @ W + b, yt) + lam * W.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return lambda idx: (((X[:, idx] - mu) / sd @ W.detach() + b.detach()).argmax(-1) == y[:, idx]).float().mean().item()

    fit_idx, val_idx = torch.arange(N_FIT), torch.arange(N_FIT, N_TRAIN)
    lam = max([1e-4, 1e-3, 1e-2, 1e-1], key=lambda l: fit(fit_idx, l)(val_idx))
    return round(fit(torch.arange(N_TRAIN), lam)(torch.arange(N_TRAIN, N)) * 100, 1)


out = {}
lengths = torch.tensor([[[len(it["options"][o[p]]) for p in range(4)] for it in items] for o in lab.ROTATIONS]).float()
out["option_length_probe"] = probe_accuracy(torch.log1p(lengths))
print("probe on option lengths only:", out["option_length_probe"], flush=True)
logits, res = run()
out["first_token"] = content(logits)
out["first_token"]["probe_by_block"] = {b: probe_accuracy(res[:, :, j].float()) for j, b in enumerate(BLOCKS)}
print(out["first_token"], flush=True)
json.dump(out, open(lab.RESULTS / "probes.json", "w"), indent=1)
print("DONE", flush=True)
