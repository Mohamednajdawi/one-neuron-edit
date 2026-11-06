"""Ablation on the tryout set (practice[3000:4000]): what does the neuron surgery add beyond the counterweight?
One variant per process (they can run in parallel):
    python medmcqa/ablation.py original      (then: surgery, counterweight, both)
For each: accuracy in all 4 cyclic option orders (normal order, rotation-balanced, agreement), the size of the
value-bias change, the constant letter offset it injects, and the loss on 200 medical explanations.
Writes results/qwen25/ablation_{variant}.json. Held-out sets are not touched.
"""
import json
import sys

import torch

import lab
from edit import mean_nll, side_effect_texts
from try_edits import calibrate_uniform, unletter_neurons

VARIANT = sys.argv[1]
S = lab.splits()
tryout, calib = S["practice"][3000:4000], S["practice"][:500]
tok, model = lab.load_model()
lm = lab.decoder(model)
attn = lm.layers[23].self_attn
b0 = attn.v_proj.bias.detach().clone()
with torch.no_grad():
    if VARIANT in ("surgery", "both"):
        unletter_neurons(model, 21, [2921])
    if VARIANT in ("counterweight", "both"):
        calibrate_uniform(model, tok, 23, calib)

db = (attn.v_proj.bias.detach() - b0).float()
n_heads, hd = attn.config.num_attention_heads, attn.head_dim
group = n_heads // attn.config.num_key_value_heads
Wo = attn.o_proj.weight.float().view(-1, n_heads, hd)
M = torch.cat([Wo[:, k * group:(k + 1) * group].sum(1) for k in range(n_heads // group)], 1)
WUg = lm.embed_tokens.weight[lab.LETTER_IDS].float() * lm.norm.weight.float()
C = torch.eye(4, device=M.device) - 1 / 4
offset = C @ WUg @ (M @ db) / 1.54

rot = [lab.score(model, tok, tryout, order=o, desc=f"{VARIANT} / order {i}") for i, o in enumerate(lab.ROTATIONS)]
ok = torch.stack([(r["logits"].argmax(-1) == r["target"]).float() for r in rot])
chosen = torch.stack([torch.tensor(o)[r["logits"].argmax(-1).cpu()] for o, r in zip(lab.ROTATIONS, rot)], 1)
votes = torch.stack([(chosen == j).sum(1) for j in range(4)], 1).max(1).values
prior = torch.stack([r["logits"] for r in rot]).log_softmax(-1).exp().mean((0, 1))
out = {"variant": VARIANT, "accuracy": ok[0].mean().item(), "rotation_balanced": ok.mean().item(),
       "per_rotation": ok.mean(1).tolist(), "same_option_4of4": (votes == 4).float().mean().item(),
       "different_every_time": (votes == 1).float().mean().item(), "content_free_prior": prior.tolist(),
       "bias_change_norm": db.norm().item(), "bias_entries_changed": int((db != 0).sum()),
       "letter_offset": offset.tolist(), "letter_offset_norm": offset.norm().item(),
       "medical_text_loss": mean_nll(model, tok, side_effect_texts())}
json.dump(out, open(lab.RESULTS / f"ablation_{VARIANT}.json", "w"), indent=1)
print(json.dumps(out), flush=True)
