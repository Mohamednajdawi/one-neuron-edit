"""Apply an edit file to the original Qwen2.5-0.5B-Instruct (downloaded on first use) and save the edited model.

    python medmcqa/apply_edit.py edits/edit1.json models/medmcqa-edit-qwen25       # Edit 1 (1,015 weights)
    python medmcqa/apply_edit.py edits/edit2.json models/medmcqa-edit-qwen25-r2    # Edit 2 (1,905 weights)
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import lab

edit_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
edit = json.load(open(edit_path))
assert edit["base_revision"] == lab.BASE_REVISION, "the edit was made for a different revision of the base model"
base = lab.base_model_dir()
shutil.copytree(base, out_dir, ignore=shutil.ignore_patterns(".cache", lab.WEIGHTS), dirs_exist_ok=True)
tensors = load_file(base / lab.WEIGHTS)
for name, t in edit["tensors"].items():
    flat = tensors[name].view(-1)
    flat[torch.tensor(t["flat_indices"])] = torch.tensor(t["values"], dtype=torch.float32).to(flat.dtype)
save_file(tensors, out_dir / lab.WEIGHTS, metadata={"format": "pt"})
print(f"applied {edit['changed_weights']:,} changed weights -> {out_dir}")
