"""Write an edit file: every weight that differs between an edited model and the original Qwen2.5-0.5B-Instruct.
This is how edits/edit1.json and edits/edit2.json were made.

    python medmcqa/export_edit.py models/medmcqa-edit-qwen25 edits/edit1.json
"""
import json
import sys
from pathlib import Path

from safetensors.torch import load_file

import lab

edited_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
base = load_file(lab.base_model_dir() / lab.WEIGHTS)
edited = load_file(edited_dir / lab.WEIGHTS)
tensors = {}
for name, old in base.items():
    new = edited[name]
    idx = (new != old).flatten().nonzero().flatten()
    if len(idx):
        tensors[name] = {"shape": list(old.shape), "dtype": str(old.dtype).removeprefix("torch."),
                         "flat_indices": idx.tolist(), "values": new.flatten()[idx].float().tolist()}
n = sum(len(t["flat_indices"]) for t in tensors.values())
out = {"base_model": lab.BASE_REPO, "base_revision": lab.BASE_REVISION, "changed_weights": n,
       "total_weights": sum(t.numel() for t in base.values()), "tensors": tensors}
out_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(out_path, "w"))
print(f"{n:,} changed weights in {len(tensors)} tensors -> {out_path}")
