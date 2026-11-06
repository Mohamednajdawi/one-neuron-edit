"""Figures, tables and the numbers quoted in the paper, computed from the saved experiment results.

    python paper/make_figures.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "medmcqa"))
import lab  # noqa: E402

OUT, DATA = ROOT / "paper/figures", ROOT / "paper/data"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
R = ROOT / "medmcqa/results"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8.5,
                     "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.4})
COL = {"base": "#8c96a3", "it1": "#1f7a8c", "it2": "#0b3c49", "gold": "#d9a441", "D": "#c0392b"}
NAME = {"base": "Original", "it1": "Edit 1", "it2": "Edit 2"}
ld = lambda f: torch.load(R / f, weights_only=False)
cat = lambda a, b: {k: torch.cat([a[k], b[k]]) for k in ("logits", "target")}
ok = lambda r: r["logits"].argmax(-1) == r["target"]
pct = lambda x: 100 * float(x)


def wilson(k, n, z=1.96):
    """Observed accuracy and the half-width of the Wilson 95% interval (nearly symmetric at these n)."""
    p = k / n
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * h


def mcnemar(a, b):
    f, br = int((~a & b).sum()), int((a & ~b).sum())
    n, k = f + br, min(f, br)
    log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k + 1)]
    m = max(log_terms)
    log_p = math.log(2) + m + math.log(sum(math.exp(t - m) for t in log_terms)) - n * math.log(2)
    return f, br, min(0.0, log_p) / math.log(10)  # log10 of the two-sided exact p-value


def paired_delta(a, b):
    n = len(a)
    f, br = int((~a & b).sum()), int((a & ~b).sum())
    d = (f - br) / n
    se = math.sqrt(f + br - (f - br) ** 2 / n) / n
    return 100 * d, 196 * se


# ---------------------------------------------------------------- held-out predictions
t5, r2f = ld("qwen25/test5000.pt"), ld("qwen25/round2_fresh.pt")
r2f = r2f if isinstance(r2f, (list, tuple)) else r2f["after"]
real = {"base": cat(ld("qwen25/baseline_eval.pt"), ld("qwen25/baseline_val_rest.pt")),
        "it1": cat(ld("qwen25/edited_eval.pt"), ld("qwen25/edited_val_rest.pt")),
        "it2": cat(ld("qwen25/round2_eval.pt"), ld("qwen25/round2_val_rest.pt"))}
mock = {"base": t5["before"], "it1": t5["after"], "it2": list(r2f)}
both = {m: torch.cat([ok(real[m]), ok(mock[m][0])]) for m in real}
gold_all = torch.cat([real["base"]["target"], mock["base"][0]["target"]])
N = {"real": len(real["base"]["target"]), "mock": len(mock["base"][0]["target"]), "both": len(gold_all)}

num = {}
for m in real:
    rb = torch.stack([ok(r).float() for r in mock[m]]).mean(0)          # per-question mean over the 4 orders
    num[m] = {"real": wilson(int(ok(real[m]).sum()), N["real"]), "mock": wilson(int(ok(mock[m][0]).sum()), N["mock"]),
              "both": wilson(int(both[m].sum()), N["both"]),
              "balanced": (pct(rb.mean()), 196 * float(rb.std() / math.sqrt(len(rb)))),
              "rotations": [pct(ok(r).float().mean()) for r in mock[m]]}
    p = torch.cat([real[m]["logits"].argmax(-1), mock[m][0]["logits"].argmax(-1)])
    num[m]["letters"] = [pct((p == i).float().mean()) for i in range(4)]
    chosen = torch.stack([torch.tensor(o)[r["logits"].argmax(-1)] for o, r in zip(lab.ROTATIONS, mock[m])], 1)
    g = mock[m][0]["target"]
    votes = torch.stack([(chosen == j).sum(1) for j in range(4)], 1).max(1).values
    lp = torch.stack([torch.stack([r["logits"].log_softmax(-1)[:, o.index(j)] for j in range(4)], 1)
                      for o, r in zip(lab.ROTATIONS, mock[m])]).mean(0)
    num[m]["agreement"] = [pct((votes == k).float().mean()) for k in (4, 3, 2, 1)]
    num[m]["correct_all4"] = pct((chosen == g[:, None]).all(1).float().mean())
    num[m]["ensemble"] = pct((lp.argmax(-1) == g).float().mean())
    num[m]["prior"] = [pct(v) for v in torch.stack([r["logits"] for r in mock[m]]).log_softmax(-1).exp().mean((0, 1))]
num["gold_letters"] = [pct((gold_all == i).float().mean()) for i in range(4)]
num["always_A"] = {"real": pct((real["base"]["target"] == 0).float().mean()), "mock": pct((mock["base"][0]["target"] == 0).float().mean()),
                   "both": pct((gold_all == 0).float().mean())}
num["mcnemar"] = {f"{a}_{b}_{s}": mcnemar(x, y) for a, b in (("base", "it1"), ("base", "it2"), ("it1", "it2"))
                  for s, x, y in (("real", ok(real[a]), ok(real[b])), ("mock", ok(mock[a][0]), ok(mock[b][0])), ("both", both[a], both[b]))}
num["delta_it2_it1"] = {"real": paired_delta(ok(real["it1"]), ok(real["it2"])), "mock": paired_delta(ok(mock["it1"][0]), ok(mock["it2"][0])),
                        "both": paired_delta(both["it1"], both["it2"])}
r2 = json.load(open(R / "qwen25/round2.json"))
n114 = next(t for t in r2["dev_tests"] if t["edit"] == "L21.n114")
nd = 1000
d = (n114["fixed"] - n114["broke"]) / nd
num["delta_it2_it1"]["tryout"] = (100 * d, 196 * math.sqrt(n114["fixed"] + n114["broke"] - (n114["fixed"] - n114["broke"]) ** 2 / nd) / nd)
guess = [100 * x for x in (4 / 256, 48 / 256, 180 / 256, 24 / 256)]  # agreement pattern of a uniform random guesser
for m in real:  # accuracy by the letter of the right answer, both test sets, normal order
    num[m]["acc_by_gold"] = [pct(both[m][gold_all == i].float().mean()) for i in range(4)]
rb = {m: torch.stack([ok(r).float() for r in mock[m]]).mean(0) for m in real}
for a, b in (("base", "it1"), ("it1", "it2")):
    dlt = rb[b] - rb[a]
    num[f"balanced_delta_{a}_{b}"] = (pct(dlt.mean()), 196 * float(dlt.std() / math.sqrt(len(dlt))),
                                      float(dlt.mean() / (dlt.std() / math.sqrt(len(dlt)))))
gold_by_set = {"real": real["base"]["target"], "mock": mock["base"][0]["target"]}
num["broken_gold_D_share"] = pct((gold_all[both["base"] & ~both["it1"]] == 3).float().mean())

# ---------------------------------------------------------------- neuron 2921, per question (practice analysis set)
cache = DATA / "n2921.json"
if not cache.exists():
    items = lab.splits()["practice"][:1500]
    tok, model = lab.load_model()
    dec = lab.decoder(model)
    WUg = dec.embed_tokens.weight[lab.LETTER_IDS].float() * dec.norm.weight.float()
    w = WUg @ dec.layers[21].mlp.down_proj.weight[:, 2921].float()
    cap = {}
    dec.layers[21].mlp.down_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("act", a[0][0, -1, 2921].float()))
    dec.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("final", a[0][0, -1].float()))
    acts, pushes = [], []
    with torch.no_grad():
        for it in items:
            model(**tok([lab.prompt(tok, it)], return_tensors="pt", add_special_tokens=False).to(lab.DEVICE), logits_to_keep=1)
            r = cap["final"].pow(2).mean().sqrt()
            v = cap["act"] * w / r
            acts.append(cap["act"].item())
            pushes.append((v - v.mean()).tolist())
    json.dump({"act": acts, "push": pushes, "gold": [it["answer"] for it in items]}, open(cache, "w"))
n29 = json.load(open(cache))
act, push, g29 = torch.tensor(n29["act"]), torch.tensor(n29["push"]), torch.tensor(n29["gold"])
num["n2921"] = {"act_mean": float(act.mean()), "act_std": float(act.std()), "act_positive": pct((act > 0).float().mean()),
                "act_min": float(act.min()), "push_D_mean": float(push[:, 3].mean()), "push_D_std": float(push[:, 3].std()),
                "push_mean": push.mean(0).tolist(),
                "corr_D_push_with_gold_D": float(torch.corrcoef(torch.stack([push[:, 3], (g29 == 3).float()]))[0, 1])}

dla = ld("qwen25/dla_practice1500.pt")
md = dla["neuron_mean_dla"]
dpush = (md[..., 3] - md.mean(-1))                                     # [24, 4864] centered mean push on "D"
order = dpush.flatten().argsort(descending=True)
num["neurons"] = {"top": [(int(i // md.shape[1]), int(i % md.shape[1]), float(dpush.flatten()[i])) for i in order[:6]],
                  "n_total": int(dpush.numel()), "rms_final": float(dla["rms_final"].mean()), "rms_final_std": float(dla["rms_final"].std())}
f = dla["final"]
c = f - f.mean(-1, keepdim=True)
habit = c.mean(0)
num["final_habit"] = habit.tolist()
num["right_letter_boost"] = float((c - habit).gather(1, dla["gold"][:, None]).mean())
num["acc_if_habit_removed"] = pct(((c - habit).argmax(-1) == dla["gold"]).float().mean())
num["probe"] = json.load(open(R / "qwen25/probes.json"))
num["lens3"] = json.load(open(R / "qwen25/lens3.json"))
num["lens_rot"] = json.load(open(R / "qwen25/lens_rot.json"))["original"]
json.dump(num, open(DATA / "numbers.json", "w"), indent=1)

# ---------------------------------------------------------------- Figure 2: block-by-block readout
layers = list(range(24))
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.5))
fig.suptitle("Reading the model block by block (logit lens, practice questions)", fontweight="bold", fontsize=9.5)
lr = num["lens_rot"]
ax[0].plot(layers, lr["rotation_averaged_accuracy"], color=COL["it1"], marker="o", ms=2.5, label="answer content (4 orders)")
ax[0].plot(layers, num["lens3"]["baseline"]["accuracy"], color=COL["base"], ls="--", label="accuracy (normal order)")
ax[0].axhline(25, color="k", lw=0.6, ls=":")
ax[0].text(23.4, 25.5, "chance", fontsize=7, ha="right")
ax[0].set(xlabel="block", ylabel="accuracy of readout (%)", ylim=(20, 38), title="(a) The answer forms in blocks 14\u201316")
ax[0].legend(frameon=False, loc="upper left")
for m, key in (("base", "baseline"), ("it1", "iteration 1"), ("it2", "iteration 2")):
    ax[1].plot(layers, num["lens3"][key]["share_D"], color=COL[m], label=NAME[m], marker="o" if m == "base" else None, ms=2.5)
ax[1].axhline(num["gold_letters"][3], color=COL["gold"], lw=1, ls="--")
ax[1].text(0.3, num["gold_letters"][3] + 2, "'D' is correct", fontsize=7, color="#9a7220")
ax[1].axvspan(20.5, 23.5, color=COL["D"], alpha=0.07, lw=0)
ax[1].set(xlabel="block", ylabel="answers 'D' (%)", ylim=(0, 85), title="(b) The 'D' habit takes over in blocks 21\u201323")
ax[1].legend(frameon=False, loc="upper left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig_lens.pdf")

# ---------------------------------------------------------------- Figure 3: every neuron's push on "D"
fig, ax = plt.subplots(figsize=(6.5, 2.5))
ax.set_title("One neuron dominates the push toward 'D' (all 116,736 MLP neurons, original model)", fontweight="bold", fontsize=9.5)
gen = torch.Generator().manual_seed(0)
x = torch.arange(24)[:, None].expand_as(dpush) + (torch.rand(dpush.shape, generator=gen) - 0.5) * 0.7
ax.scatter(x.flatten(), dpush.flatten(), s=1.2, color=COL["base"], alpha=0.35, lw=0, rasterized=True)
ax.scatter([21], [dpush[21, 2921]], s=22, color=COL["D"], zorder=3)
ax.annotate(f"block 21, neuron 2921: +{dpush[21, 2921]:.2f}", (21, dpush[21, 2921]), xytext=(11.5, 1.75), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", lw=0.6, color="k"))
(l2, j2, v2) = num["neurons"]["top"][1]
ax.scatter([l2], [v2], s=14, color="k", zorder=3)
ax.annotate(f"runner-up (block {l2}, neuron {j2}): +{v2:.2f}", (l2, v2), xytext=(9.5, 0.95), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", lw=0.6, color="k"))
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="block", ylabel="mean push on 'D' (logits)", xticks=range(0, 24, 2), xlim=(-0.8, 23.8))
fig.tight_layout()
fig.savefig(OUT / "fig_neurons.pdf", dpi=300)

# ---------------------------------------------------------------- Figure 4: held-out accuracy and letter mix
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw={"width_ratios": [1.35, 1]})
fig.suptitle("Held-out results: the original model and the agent's two edits", fontweight="bold", fontsize=9.5)
groups = [("real", f"Real-exam test\n(n={N['real']:,})", num["always_A"]["real"]),
          ("mock", f"Mock-exam test\n(n={N['mock']:,})", num["always_A"]["mock"]),
          ("balanced", "Mock-exam,\nrotation-balanced", 25.0)]
for gi, (k, label, ref) in enumerate(groups):
    for mi, m in enumerate(("base", "it1", "it2")):
        v, h = num[m][k]
        ax[0].bar(gi + (mi - 1) * 0.26, v, 0.24, yerr=h, color=COL[m], capsize=1.5, error_kw={"lw": 0.6},
                  label=NAME[m] if gi == 0 else None)
    ax[0].plot([gi - 0.42, gi + 0.42], [ref, ref], color="k", lw=0.8, ls="--",
               label="always 'A' (balanced: chance)" if gi == 0 else None)
ax[0].set_xticks(range(3), [g[1] for g in groups])
ax[0].set(ylabel="accuracy (%)", ylim=(0, 50), yticks=range(0, 41, 10), title="(a) Held-out accuracy (95% CI)")
ax[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=6.8, columnspacing=1.0)
for mi, (m, color) in enumerate((("base", COL["base"]), ("it1", COL["it1"]), ("it2", COL["it2"]), ("gold", COL["gold"]))):
    vals = num["gold_letters"] if m == "gold" else num[m]["letters"]
    ax[1].bar([i + (mi - 1.5) * 0.2 for i in range(4)], vals, 0.19, color=color, label="correct answers" if m == "gold" else NAME[m])
ax[1].set_xticks(range(4), list("ABCD"))
ax[1].set(ylabel="share of answers (%)", ylim=(0, 82), title="(b) Letters chosen (both test sets)")
ax[1].legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig_results.pdf")

# ---------------------------------------------------------------- Figure 5: consistency under option rotation
fig, ax = plt.subplots(figsize=(6.5, 2.05))
ax.set_title("Same option when the options are reordered? (5,000 mock-exam questions, 4 orders each)", fontweight="bold", fontsize=9.5)
rows = [("Random guesser", guess), ("Original", num["base"]["agreement"]), ("Edit 1", num["it1"]["agreement"]), ("Edit 2", num["it2"]["agreement"])]
shades = ["#155e6b", "#4fa3b3", "#c9d6dd", "#e8a09a"]
labels = ["same option in 4/4 orders", "3/4", "2/4", "4 different options"]
for ri, (name, vals) in enumerate(rows[::-1]):
    left = 0
    for k, v in enumerate(vals):
        ax.barh(ri, v, left=left, color=shades[k], height=0.62, label=labels[k] if ri == 0 else None)
        if v > 6:
            ax.text(left + v / 2, ri, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if k < 2 else "k")
        left += v
ax.set_yticks(range(4), [r[0] for r in rows[::-1]])
ax.set(xlim=(0, 100), xlabel="share of the 5,000 mock-exam questions (%)")
ax.legend(frameon=False, ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.38))
fig.tight_layout()
fig.savefig(OUT / "fig_consistency.pdf")

# ---------------------------------------------------------------- Figure 6: Edit 2 vs Edit 1 (winner's curse)
fig, ax = plt.subplots(figsize=(3.4, 2.0))
fig.suptitle("Edit 2 vs. Edit 1: the tryout gain does not hold up", fontweight="bold", fontsize=8.5)
rows = [("Tryout (selection set)\nn=1,000", num["delta_it2_it1"]["tryout"]), ("Real-exam test\nn=4,183", num["delta_it2_it1"]["real"]),
        ("Mock-exam test\nn=5,000", num["delta_it2_it1"]["mock"]), ("Both tests\nn=9,183", num["delta_it2_it1"]["both"])]
for i, (name, (dv, h)) in enumerate(rows[::-1]):
    ax.errorbar(dv, i, xerr=h, fmt="o", color=COL["D"] if i == 3 else COL["it2"], ms=3.5, capsize=2, lw=0.9)
ax.axvline(0, color="k", lw=0.6)
ax.set_yticks(range(4), [r[0] for r in rows[::-1]], fontsize=6.8)
ax.set(xlabel="Edit 2 minus Edit 1 (accuracy points)", xlim=(-1.6, 3.6))
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "fig_round2.pdf")

# ---------------------------------------------------------------- LaTeX table with the main held-out results
def cell(m, k):
    v, h = num[m][k]
    return f"{v:.1f} {{\\scriptsize$\\pm${h:.1f}}}"


def pfmt(lp):
    e = math.floor(lp)
    return f"${10 ** (lp - e):.1f}\\times10^{{{e}}}$" if lp < -3 else f"{10 ** lp:.2f}"


mc = num["mcnemar"]
lines = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{\textbf{Held-out results.} Accuracy (\%) with the half-width of the 95\% interval. Edit 2 is described in \S\ref{sec:round2}. "
         r"``Always A'' scores 32.2 (real-exam), 29.9 (mock-exam) and 30.9 (both). Medical-text loss is the mean next-token loss on "
         r"200 MedMCQA expert explanations (ordinary prose, not MCQs).}", r"\label{tab:main}",
         r"\begin{tabular}{lccc}", r"\toprule", r" & Original & Edit 1 & Edit 2 \\", r"\midrule",
         r"Weights changed & 0 & 1{,}015 & 1{,}905 \\",
         f"Real-exam test ($n$={N['real']:,}) & {cell('base', 'real')} & {cell('it1', 'real')} & {cell('it2', 'real')} \\\\".replace(",", "{,}", 1),
         f"Mock-exam test ($n$={N['mock']:,}) & {cell('base', 'mock')} & {cell('it1', 'mock')} & {cell('it2', 'mock')} \\\\".replace(",", "{,}", 1),
         f"\\quad rotation-balanced & {cell('base', 'balanced')} & {cell('it1', 'balanced')} & {cell('it2', 'balanced')} \\\\",
         f"Both tests ($n$={N['both']:,}) & {cell('base', 'both')} & \\textbf{{{num['it1']['both'][0]:.1f}}} {{\\scriptsize$\\pm${num['it1']['both'][1]:.1f}}} & {cell('it2', 'both')} \\\\".replace(",", "{,}", 1),
         r"\midrule",
         f"Fixed / broken vs.\\ Original & -- & {mc['base_it1_both'][0]:,} / {mc['base_it1_both'][1]} & {mc['base_it2_both'][0]:,} / {mc['base_it2_both'][1]} \\\\".replace(",", "{,}"),
         f"McNemar $p$ vs.\\ Original & -- & {pfmt(mc['base_it1_both'][2])} & {pfmt(mc['base_it2_both'][2])} \\\\",
         f"Fixed / broken vs.\\ Edit 1 & -- & -- & {mc['it1_it2_both'][0]} / {mc['it1_it2_both'][1]} ($p$={10 ** mc['it1_it2_both'][2]:.2f}) \\\\",
         r"Medical-text loss (nats/token) & 3.045 & 3.050 & 3.058 \\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT.parent / "tab_main.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- LaTeX table: ablation on the tryout set
abl = {v: json.load(open(R / f"qwen25/ablation_{v}.json")) for v in ("original", "surgery", "counterweight", "both")}
loss0 = abl["original"]["medical_text_loss"]
nz = lambda v: "0" if v == 0 else f"{v:.2f}"
lines = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{\textbf{What each part of Edit 1 does.} Ablation on the tryout set (1{,}000 practice questions, four orders each), "
         r"run by the agent for this paper. ``Prior D'' is the content-free probability of ``D''; $\|\Delta\vect b\|$ is the norm of the "
         r"value-bias change; ``Offset'' is the norm of the constant centered letter-logit offset it injects; ``Loss'' is the change "
         r"in medical-text loss.}", r"\label{tab:ablation}",
         r"\begin{tabular}{lcccccccc}", r"\toprule",
         r"Variant & Acc. & Balanced & Same 4/4 & Diff.\ 4/4 & Prior D & $\|\Delta\vect b\|$ & Offset & Loss \\", r"\midrule"]
for key, label in (("original", "Original"), ("surgery", "Surgery only"), ("counterweight", "Counterweight only"), ("both", "Both (Edit 1)")):
    a = abl[key]
    loss = "--" if key == "original" else f"+{100 * (a['medical_text_loss'] / loss0 - 1):.2f}\\%"
    lines.append(f"{label} & {100 * a['accuracy']:.1f} & {100 * a['rotation_balanced']:.1f} & {100 * a['same_option_4of4']:.1f} & "
                 f"{100 * a['different_every_time']:.1f} & {100 * a['content_free_prior'][3]:.1f} & {nz(a['bias_change_norm'])} & "
                 f"{nz(a['letter_offset_norm'])} & {loss} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT.parent / "tab_ablation.tex").write_text("\n".join(lines) + "\n")

print(json.dumps({k: num[k] for k in ("n2921", "neurons", "final_habit", "right_letter_boost", "acc_if_habit_removed", "always_A",
                                       "gold_letters", "delta_it2_it1")}, indent=1))
for m in ("base", "it1", "it2"):
    print(m, {k: (round(v[0], 2), round(v[1], 2)) if isinstance(v, tuple) else v for k, v in num[m].items() if k not in ("letters", "prior")},
          "letters", [round(v, 1) for v in num[m]["letters"]], "prior", [round(v, 1) for v in num[m]["prior"]])
print({k: (v[0], v[1], round(v[2], 2)) for k, v in mc.items()})
print("probe", num["probe"]["first_token"]["probe_by_block"], "content", num["probe"]["first_token"]["content"], "length", num["probe"]["option_length_probe"])
print("acc by gold letter", {m: [round(v, 1) for v in num[m]["acc_by_gold"]] for m in ("base", "it1", "it2")})
print("balanced deltas (pts, 95% half-width, z)", num["balanced_delta_base_it1"], num["balanced_delta_it1_it2"])
print("share of answers broken by Edit 1 whose right answer is D", round(num["broken_gold_D_share"], 1))
