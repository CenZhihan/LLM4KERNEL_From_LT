import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from dataset import dataset
from utils.ascend_compile_pipeline import ascend_compile
from utils.utils import underscore_to_pascalcase
from backends.ascendc_backend import AscendBackend

backend = AscendBackend()
op = 'sigmoid'
with open(f'output/ascendc/rag_four_ops_with_official_doc/0.0-1.0/gpt-5/run0/{op}.txt', 'r') as f:
    code = f.read()

print("Compiling", op)
backend.compile(code, op)
print("Done. Check directory:", backend.current_op)
