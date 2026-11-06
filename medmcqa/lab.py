"""MedMCQA zero-shot harness for Qwen2.5-0.5B-Instruct: data splits, prompt, and letter scoring.

The base model is downloaded on first use into models/Qwen2.5-0.5B-Instruct (pinned revision).
"""
import random
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parent.parent
BASE_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
WEIGHTS = "model.safetensors"
RESULTS = ROOT / "medmcqa/results/qwen25"
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
LETTERS = "ABCD"
LETTER_IDS = [32, 33, 34, 35]  # "A".."D" as the first answer token
INSTRUCTION = ("Answer the following multiple choice question. "
               "Reply with only the letter (A, B, C, or D) of the correct option.")


def base_model_dir():
    if not (MODEL_DIR / WEIGHTS).exists():
        snapshot_download(BASE_REPO, revision=BASE_REVISION, local_dir=MODEL_DIR)
    return MODEL_DIR


def load_model(path=MODEL_DIR):
    path = Path(path)
    if path == MODEL_DIR:
        base_model_dir()
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEVICE).eval()
    model.requires_grad_(False)
    return tok, model


def decoder(model):
    """The transformer stack (has .layers, .norm, .embed_tokens)."""
    return model.model


def _items(rows):
    return [{"id": r["id"], "question": r["question"], "options": [r["opa"], r["opb"], r["opc"], r["opd"]],
             "answer": r["cop"], "subject": r["subject_name"]} for r in rows]


def splits(seed=0, n_eval=1000, n_practice=4000, n_fresh=0):
    """eval: n_eval random validation questions (held out); val_rest: the other validation questions (held out);
    practice: random train questions (all analysis and selection); fresh: n_fresh more train questions never used
    for anything (question text unseen in practice/validation). eval + val_rest = the real-exam test (NEET PG);
    fresh = the mock-exam test."""
    ds = load_dataset("openlifescienceai/medmcqa")
    val, train = ds["validation"], ds["train"]
    eval_idx = random.Random(seed).sample(range(len(val)), n_eval)
    rest_idx = sorted(set(range(len(val))) - set(eval_idx))
    held_out_q = {q.strip().lower() for q in val["question"]}
    train_q = train["question"]
    practice_idx = [i for i in random.Random(seed + 1).sample(range(len(train)), n_practice + 200)
                    if train_q[i].strip().lower() not in held_out_q][:n_practice]
    out = {"eval": _items(val.select(eval_idx)), "val_rest": _items(val.select(rest_idx)),
           "practice": _items(train.select(practice_idx))}
    if n_fresh:
        seen = held_out_q | {train_q[i].strip().lower() for i in practice_idx}
        fresh_idx = []
        for i in random.Random(seed + 2).sample(range(len(train)), 3 * n_fresh):
            q = train_q[i].strip().lower()
            if q not in seen:
                seen.add(q)
                fresh_idx.append(i)
        out["fresh"] = _items(train.select(fresh_idx[:n_fresh]))
    return out


def prompt(tok, item, order=(0, 1, 2, 3), system=None):
    opts = "\n".join(f"{L}. {item['options'][j]}" for L, j in zip(LETTERS, order))
    msg = f"{INSTRUCTION}\n\nQuestion: {item['question']}\n{opts}"
    chat = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": msg}]
    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def score(model, tok, items, batch_size=1, order=(0, 1, 2, 3), hidden=False, desc="scoring", system=None):
    """Letter logits [N, 4] (in displayed order), the displayed index of the right answer, the share of
    next-token probability on the 4 letters, and optionally the last-position residual of every layer.
    batch_size=1 by default: bf16 logits shift with batch shape (enough to flip near-ties); size 1 is
    deterministic."""
    logits, mass, hs = [], [], []
    for i in tqdm(range(0, len(items), batch_size), desc=desc, unit="q", mininterval=5, dynamic_ncols=True):
        batch = tok([prompt(tok, it, order, system) for it in items[i:i + batch_size]], return_tensors="pt",
                    padding=True, add_special_tokens=False).to(DEVICE)
        out = model(**batch, output_hidden_states=hidden, logits_to_keep=1)
        last = out.logits[:, -1].float()
        logits.append(last[:, LETTER_IDS].cpu())
        mass.append(last.softmax(-1)[:, LETTER_IDS].sum(-1).cpu())
        if hidden:
            hs.append(torch.stack([h[:, -1].float().cpu() for h in out.hidden_states], 1))
    target = torch.tensor([order.index(it["answer"]) for it in items])
    res = {"logits": torch.cat(logits), "target": target, "letter_mass": torch.cat(mass)}
    if hidden:
        res["hidden"] = torch.cat(hs)
    return res


def accuracy(res):
    return (res["logits"].argmax(-1) == res["target"]).float().mean().item()


ROTATIONS = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]


def summary(res, items):
    pred = res["logits"].argmax(-1)
    by_subject = {}
    for it, ok in zip(items, (pred == res["target"]).tolist()):
        by_subject.setdefault(it["subject"], []).append(ok)
    return {"n": len(items), "accuracy": accuracy(res),
            "predicted_letters": {L: (pred == i).float().mean().item() for i, L in enumerate(LETTERS)},
            "gold_letters": {L: (res["target"] == i).float().mean().item() for i, L in enumerate(LETTERS)},
            "letter_mass": res["letter_mass"].mean().item(),
            "by_subject": {s: [round(sum(v) / len(v), 4), len(v)] for s, v in
                           sorted(by_subject.items(), key=lambda kv: -len(kv[1]))}}


def circular_accuracy(model, tok, items):
    """Mean accuracy over the 4 cyclic option orders: each answer sits once at A, B, C and D,
    so a preference for one letter cannot raise the score."""
    return sum(accuracy(score(model, tok, items, order=o, desc=f"rotation {k + 1}/4"))
               for k, o in enumerate(ROTATIONS)) / 4


def consistency(rot_results, items):
    """Does the model pick the same option (by content) when the 4 options are cyclically re-ordered?
    rot_results[k] is score(...) under ROTATIONS[k]."""
    chosen = torch.stack([torch.tensor(o)[r["logits"].argmax(-1)] for o, r in zip(ROTATIONS, rot_results)], 1)
    gold = torch.tensor([it["answer"] for it in items])
    votes = torch.stack([(chosen == j).sum(1) for j in range(4)], 1)
    top = votes.max(1).values
    same = top == 4
    by_option = torch.stack([torch.stack([r["logits"].log_softmax(-1)[:, o.index(j)] for j in range(4)], 1)
                             for o, r in zip(ROTATIONS, rot_results)]).mean(0)
    return {"same_answer_in_all_4": same.float().mean().item(),
            "agreement": {f"{k}/4": (top == k).float().mean().item() for k in (4, 3, 2, 1)},
            "consistent_and_correct": (same & (chosen[:, 0] == gold)).float().mean().item(),
            "consistent_and_wrong": (same & (chosen[:, 0] != gold)).float().mean().item(),
            "correct_in_all_4": (chosen == gold[:, None]).all(1).float().mean().item(),
            "mean_accuracy_over_4": (chosen == gold[:, None]).float().mean().item(),
            "averaged_over_4_orders_accuracy": (by_option.argmax(-1) == gold).float().mean().item()}
