"""Edit tools, and forward-pass tests of candidate edits on the tryout set (practice[3000:4000], train split;
the test sets are never used here).

    python medmcqa/try_edits.py "unletter_neurons(model, 21, [2921])" "scale_neurons(model, 21, [2921], 0)"

Each argument is a Python expression that edits `model` in place; weights are restored after each test.
"""
import sys

import torch
from safetensors import safe_open

import lab

DEV = slice(3000, 4000)


def restore(model, original):
    for name, p in model.named_parameters():
        ref = original.get_tensor(name).to(p.dtype)
        if not torch.equal(p.detach().cpu(), ref):
            p.copy_(ref)


def scale_neurons(model, layer, neurons, s):
    lab.decoder(model).layers[layer].mlp.down_proj.weight[:, neurons] *= s


def calibrate(model, layer, target, rms=1.54):
    """Add a constant letter-logit offset via the value bias of `layer`: softmax weights sum to 1, so a
    value-bias change reaches the residual stream unchanged as o_proj @ bias at every position."""
    lm = lab.decoder(model)
    attn = lm.layers[layer].self_attn
    hd, n_heads = attn.head_dim, attn.config.num_attention_heads
    group = n_heads // attn.config.num_key_value_heads
    Wo = attn.o_proj.weight.float().view(-1, n_heads, hd)
    M = torch.cat([Wo[:, k * group:(k + 1) * group].sum(1) for k in range(n_heads // group)], 1)  # [hidden, kv*hd]
    WUg = lm.embed_tokens.weight[lab.LETTER_IDS].float() * lm.norm.weight.float()
    C = torch.eye(4, device=M.device) - 1 / 4
    A = C @ WUg @ M / rms
    delta_b = torch.linalg.pinv(A) @ (C @ target.to(M.device).float())
    attn.v_proj.bias += delta_b.to(attn.v_proj.bias.dtype)
    return delta_b


def calibrate_uniform(model, tok, layer, items, rounds=3, rms=1.54):
    """Repeat: measure the content-free letter preference on unlabeled practice questions, cancel it."""
    for _ in range(rounds):
        rot = torch.stack([lab.score(model, tok, items, order=o)["logits"] for o in lab.ROTATIONS])
        prior = rot.log_softmax(-1).exp().mean((0, 1))
        print("  letter preference before this round:", prior.cpu().numpy().round(3), flush=True)
        calibrate(model, layer, -prior.log(), rms)


@torch.no_grad()
def final_rms(model, tok, items):
    """Mean RMS of the final residual at the answer position: how far a residual change moves the letter logits."""
    vals = []
    hook = lab.decoder(model).norm.register_forward_pre_hook(
        lambda m, a: vals.append(a[0][0, -1].float().pow(2).mean().sqrt().item()))
    for it in items:
        model(**tok([lab.prompt(tok, it)], return_tensors="pt", add_special_tokens=False).to(lab.DEVICE), logits_to_keep=1)
    hook.remove()
    return sum(vals) / len(vals)


def letter_basis(model):
    """Orthonormal basis of the residual directions that change letter-logit differences (rank 3)."""
    lm = lab.decoder(model)
    WUg = lm.embed_tokens.weight[lab.LETTER_IDS].float() * lm.norm.weight.float()
    C = torch.eye(4, device=WUg.device) - 1 / 4
    return torch.linalg.svd((C @ WUg).T, full_matrices=False)[0][:, :3]


def unletter_neurons(model, layer, neurons):
    """Remove only the letter-voting part of these neurons' output vectors; everything else stays."""
    Q = letter_basis(model)
    W = lab.decoder(model).layers[layer].mlp.down_proj.weight
    w = W[:, neurons].float()
    W[:, neurons] = (w - Q @ (Q.T @ w)).to(W.dtype)


def unletter_blocks(model, layers, attention=True, mlp=True):
    """Stop whole blocks from moving letter-logit differences: project the letter subspace out of their
    output matrices, so the letter readout keeps what earlier blocks computed."""
    Q = letter_basis(model)
    for L in layers:
        l = lab.decoder(model).layers[L]
        for W in [l.self_attn.o_proj.weight] * attention + [l.mlp.down_proj.weight] * mlp:
            w = W.float()
            W.copy_((w - Q @ (Q.T @ w)).to(W.dtype))


if __name__ == "__main__":
    S = lab.splits()
    dev = S["practice"][DEV]
    tok, model = lab.load_model()
    original = safe_open(lab.MODEL_DIR / lab.WEIGHTS, "pt")
    base = torch.load(lab.RESULTS / "baseline_practice.pt")
    base_ok = base["logits"][DEV].argmax(-1) == base["target"][DEV]
    print(f"baseline dev acc {base_ok.float().mean():.4f}", flush=True)
    for expr in sys.argv[1:]:
        with torch.no_grad():
            exec(expr)
        r = lab.score(model, tok, dev)
        ok = r["logits"].argmax(-1) == r["target"]
        print(f"{expr}: dev acc {ok.float().mean():.4f} (fixed {int((ok & ~base_ok).sum())}, "
              f"broke {int((~ok & base_ok).sum())}) pred {torch.bincount(r['logits'].argmax(-1), minlength=4).tolist()}",
              flush=True)
        with torch.no_grad():
            restore(model, original)
