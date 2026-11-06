"""Logit lens for baseline, iteration 1 and iteration 2 on the same 1,500 practice questions: accuracy and share
of "D" answers if the answer is read out after each block.

    python medmcqa/lens3.py              # needs both edited models
"""
import json

import torch
from tqdm.auto import tqdm

import lab
from edit import OUT_DIR

MODELS = {"baseline": lab.MODEL_DIR, "iteration 1": OUT_DIR, "iteration 2": lab.ROOT / "models/medmcqa-edit-qwen25-r2"}
items = lab.splits()["practice"][:1500]
gold = torch.tensor([it["answer"] for it in items])
out = {}
for name, path in MODELS.items():
    tok, model = lab.load_model(path)
    lm = lab.decoder(model)
    eps = getattr(lm.norm, "variance_epsilon", 1e-6)
    WUg = lm.embed_tokens.weight[lab.LETTER_IDS].float() * lm.norm.weight.float()
    cap = {}
    hooks = [l.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, -1].float()))
             for i, l in enumerate(lm.layers)]
    lens = torch.zeros(len(items), len(lm.layers), 4)
    with torch.no_grad():
        for n, it in enumerate(tqdm(items, desc=f"lens / {name}", unit="q", mininterval=5)):
            model(**tok([lab.prompt(tok, it)], return_tensors="pt", add_special_tokens=False).to(lab.DEVICE), logits_to_keep=1)
            for i in range(len(lm.layers)):
                lens[n, i] = (WUg @ (cap[i] / (cap[i].pow(2).mean() + eps).sqrt())).cpu()
    for h in hooks:
        h.remove()
    pred = lens.argmax(-1)
    out[name] = {"accuracy": [round((pred[:, i] == gold).float().mean().item() * 100, 1) for i in range(len(lm.layers))],
                 "share_D": [round((pred[:, i] == 3).float().mean().item() * 100, 1) for i in range(len(lm.layers))]}
    print(name, out[name], flush=True)
    del model
json.dump(out, open(lab.RESULTS / "lens3.json", "w"), indent=1)
print("DONE", flush=True)
