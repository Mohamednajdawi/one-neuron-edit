"""Where does the model keep the answer, and where does the letter habit take over? Logit lens on 1,000 practice
questions in all 4 cyclic option orders. Per block: plain accuracy, rotation-averaged accuracy (the 4 orders put
each option once at A, B, C and D, so letter habits cancel and only content is left) and the content-free letter
preference. Run after the edits as a check on the diagnosis.

    python medmcqa/lens_rot.py           # -> results/qwen25/lens_rot.json
"""
import json

import torch
from tqdm.auto import tqdm

import lab

N = 1000
items = lab.splits()["practice"][:N]
gold = torch.tensor([it["answer"] for it in items])
tok, model = lab.load_model()
lm = lab.decoder(model)
eps = lm.norm.variance_epsilon
WUg = lm.embed_tokens.weight[lab.LETTER_IDS].float() * lm.norm.weight.float()
cap = {}
for i, l in enumerate(lm.layers):
    l.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, -1].float()))
lens = torch.zeros(4, N, len(lm.layers), 4)
with torch.no_grad():
    for k, order in enumerate(lab.ROTATIONS):
        for n, it in enumerate(tqdm(items, desc=f"order {k + 1}/4", unit="q", mininterval=5)):
            model(**tok([lab.prompt(tok, it, order)], return_tensors="pt", add_special_tokens=False).to(lab.DEVICE),
                  logits_to_keep=1)
            for i in range(len(lm.layers)):
                lens[k, n, i] = (WUg @ (cap[i] / (cap[i].pow(2).mean() + eps).sqrt())).cpu()
logp = lens.log_softmax(-1)
by_option = torch.stack([torch.stack([logp[k, :, :, o.index(j)] for j in range(4)], -1)
                         for k, o in enumerate(lab.ROTATIONS)]).mean(0)  # [N, layers, option]
pct = lambda x: [round(v * 100, 1) for v in x.tolist()]
out = {"original": {"accuracy": pct((lens[0].argmax(-1) == gold[:, None]).float().mean(0)),
                    "rotation_averaged_accuracy": pct((by_option.argmax(-1) == gold[:, None]).float().mean(0)),
                    "letter_preference": [pct(p) for p in logp.exp().mean((0, 1))]}}
print({k: v for k, v in out["original"].items() if k != "letter_preference"}, flush=True)
json.dump(out, open(lab.RESULTS / "lens_rot.json", "w"), indent=1)
print("DONE", flush=True)
