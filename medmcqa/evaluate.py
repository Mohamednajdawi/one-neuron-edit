"""Score a model on the real-exam test (all 4,183 MedMCQA validation questions, normal option order).

    python medmcqa/evaluate.py                                 # original model
    python medmcqa/evaluate.py models/medmcqa-edit-qwen25      # Edit 1
"""
import sys
from pathlib import Path

import lab

path = Path(sys.argv[1]) if len(sys.argv) > 1 else lab.MODEL_DIR
S = lab.splits()
items = S["eval"] + S["val_rest"]
tok, model = lab.load_model(path)
s = lab.summary(lab.score(model, tok, items), items)
print(f"{path.name}: accuracy {s['accuracy']:.2%} on {s['n']:,} questions")
print("letters chosen:", {k: f"{v:.1%}" for k, v in s["predicted_letters"].items()})
