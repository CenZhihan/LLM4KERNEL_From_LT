#!/usr/bin/env python3
"""
Diagnose deploy step: rebuild relu_custom and check file counts at each stage.
"""
import os, sys, glob

sys.path.insert(0, ".")
from config import op_engineer_dir
from utils.ascend_compile_pipeline import ascend_compile

op_name = "relu"
txt_path = f"output/ascendc/external165/0.0-1.0/external_lab/run0/{op_name}.txt"
with open(txt_path) as f:
    code = f.read()

deploy_path_abs = os.path.abspath(os.path.join(op_engineer_dir, f"opp_{op_name}_custom"))

def count_files(d):
    return len([f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True) if os.path.isfile(f)])

print(f"[DIAG] Before compile: {count_files(deploy_path_abs)} files in {deploy_path_abs}")

ctx = {}
try:
    ascend_compile(code, op_name, ctx)
    print("[DIAG] ascend_compile succeeded")
except Exception as e:
    print(f"[DIAG] ascend_compile FAILED: {e}")

print(f"[DIAG] After compile: {count_files(deploy_path_abs)} files in {deploy_path_abs}")

# Check kernel JSON
import json
for p in glob.glob(os.path.join(deploy_path_abs, "**", "*.json"), recursive=True):
    try:
        with open(p) as f:
            d = json.load(f)
        if "opParaSize" in d:
            print(f"[DIAG] {os.path.basename(p)}: opParaSize={d['opParaSize']}")
    except:
        pass
