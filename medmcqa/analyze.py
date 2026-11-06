"""Direct logit attribution (DLA) of the answer letter on practice questions.

At the answer position the final residual is a sum of every block's attention/MLP output, so the letter
logits split exactly into per-head and per-neuron contributions (holding the final RMSNorm scale fixed).
Saves per-item head DLA, per-layer attention/MLP DLA, per-neuron summary statistics and a logit lens.

    python medmcqa/analyze.py            # 1,500 practice questions -> results/qwen25/dla_practice1500.pt
"""
import sys

import torch
from tqdm.auto import tqdm

import lab

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
OUT = lab.RESULTS

S = lab.splits()
items = S["practice"][:N]
tok, model = lab.load_model()
lm = lab.decoder(model)
cfg = model.config
D = lab.DEVICE
eps = lm.norm.variance_epsilon
WUg = lm.embed_tokens.weight[lab.LETTER_IDS].float() * lm.norm.weight.float()  # [4, hidden]

n_layers, n_neurons, n_heads = len(lm.layers), cfg.intermediate_size, cfg.num_attention_heads
P_mix = [WUg @ l.self_attn.o_proj.weight.float() for l in lm.layers]
P_down = [WUg @ l.mlp.down_proj.weight.float() for l in lm.layers]
heads = [n_heads] * n_layers
out = lambda o: o[0] if isinstance(o, tuple) else o

cap = {}
for i, layer in enumerate(lm.layers):
    layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(("mix", i), a[0][0, -1].float()))
    layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(("act", i), a[0][0, -1].float()))
    layer.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("res", i), out(o)[0, -1].float()))
lm.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("final", a[0][0, -1].float()))

head_dla = torch.zeros(N, n_layers, n_heads, 4)
block_dla = torch.zeros(N, n_layers, 2, 4)   # [attention, MLP]
lens = torch.zeros(N, n_layers, 4)
final = torch.zeros(N, 4)
rms_final = torch.zeros(N)
gold = torch.tensor([it["answer"] for it in items])
nsum = torch.zeros(n_layers, n_neurons, 4, device=D)   # sum of per-neuron letter DLA
ncor = torch.zeros(n_layers, n_neurons, device=D)      # sum of (DLA[gold] - mean DLA)
nact = torch.zeros(n_layers, n_neurons, device=D)      # sum of activations

with torch.no_grad():
    for n, it in enumerate(tqdm(items, desc="attribution", unit="q", mininterval=5)):
        batch = tok([lab.prompt(tok, it)], return_tensors="pt", add_special_tokens=False).to(D)
        logits = model(**batch, logits_to_keep=1).logits[0, -1, lab.LETTER_IDS].float()
        rms = lambda v: (v.pow(2).mean() + eps).sqrt()
        r = rms(cap["final"])
        rms_final[n] = r.item()
        final[n] = (WUg @ cap["final"] / r).cpu()
        assert (final[n] - logits.cpu()).abs().max() < 0.5, (final[n], logits)
        g = int(gold[n])
        for i in range(n_layers):
            hd = (P_mix[i] * cap[("mix", i)] / r).view(4, n_heads, -1).sum(-1)   # [4, heads]
            head_dla[n, i] = hd.T.cpu()
            dn = P_down[i] * cap[("act", i)] / r                                  # [4, neurons]
            block_dla[n, i, 0] = hd.sum(-1).cpu()
            block_dla[n, i, 1] = dn.sum(-1).cpu()
            nsum[i] += dn.T
            ncor[i] += dn[g] - dn.mean(0)
            nact[i] += cap[("act", i)]
            res = cap[("res", i)]
            lens[n, i] = (WUg @ res / rms(res)).cpu()

torch.save({"head_dla": head_dla, "block_dla": block_dla, "lens": lens, "final": final, "gold": gold,
            "rms_final": rms_final, "neuron_mean_dla": (nsum / N).cpu(), "neuron_mean_correct": (ncor / N).cpu(),
            "neuron_mean_act": (nact / N).cpu(), "heads": heads}, OUT / f"dla_practice{N}.pt")
print("saved")
