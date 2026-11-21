# -*- coding: utf-8 -*-
"""
Copies what the SE model needs into this folder. Cog can only see files inside the folder that holds
cog.yaml, so the model code and checkpoint are copied here instead of linked.

Run it again whenever the SE model code or checkpoint changes, before `cog push`:
    C:\\demo\\Scripts\\python.exe replicate\\se_model\\prepare.py        (Windows)
    python3 prepare.py                                                  (WSL, inside this folder)
"""

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parents[1]                  # the First_Demo folder

ITEMS = [
    "SE_Model",
    "Harmonic_Mask",
    "Configuration/config_1.json",
    "Checkpoints_v2_1_lossTest/epoch_50.pth",
    "services/common/__init__.py",
    "services/common/contract.py",
    "services/se_service/se_core.py",     # the same model code as the SE service
]

for item in ITEMS:
    src, dst = DEMO / item, HERE / item
    if not src.exists():
        raise SystemExit(f"[ERROR] not found: {src}")
    if dst.is_dir():
        shutil.rmtree(dst)
    elif dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)
    print(f"copied {item}")
print("[INFO] Ready for cog build / cog push.")
