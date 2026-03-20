#!/usr/bin/env python3
"""
Diagnostic script: build & run a single AscendC custom operator
with full debug output. Does NOT cleanup build artifacts.
"""
import os, sys, json, subprocess, shutil, re

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from config import op_engineer_dir, deploy_path, ascendc_device, project_root_path
from utils.utils import underscore_to_pascalcase

OP_NAME = sys.argv[1] if len(sys.argv) > 1 else 'relu'
TXT_DIR = f'output/ascendc/external165/0.0-1.0/external_lab/run0'
txt_path = os.path.join(TXT_DIR, f'{OP_NAME}.txt')

print(f'=== Diagnosing op: {OP_NAME} ===')
print(f'txt_path: {txt_path}')

# 1. Read & exec generated code to populate context
with open(txt_path, 'r') as f:
    generated_code = f.read()
context = {}
exec(compile(generated_code, "<string>", "exec"), context)

op = OP_NAME + '_custom'
op_capital = underscore_to_pascalcase(op)
target_directory = os.path.join(op_engineer_dir, op_capital)

# 2. Create project
if os.path.isdir(target_directory):
    shutil.rmtree(target_directory)
json_path = os.path.join(op_engineer_dir, f'{op}.json')
with open(json_path, 'w') as f:
    f.write(context.get('project_json_src') or '')

os.chdir(op_engineer_dir)
subprocess.run(["msopgen", 'gen', '-i', f'{op}.json', '-c', ascendc_device, '-lan', 'cpp', '-out', op_capital],
               check=True, capture_output=True, text=True)
print('[OK] msopgen project created')

# 3. Write host sources
with open(os.path.join(target_directory, 'op_host', f'{op}_tiling.h'), 'w') as f:
    f.write(context.get('host_tiling_src') or '')
with open(os.path.join(target_directory, 'op_host', f'{op}.cpp'), 'w') as f:
    f.write(context.get('host_operator_src') or '')

# 4. Kernel injection (same as pipeline)
kernel_src = context.get('kernel_src') or ''
if kernel_src and 'GET_TILING_DATA' in kernel_src:
    tiling_include = f'#include "{op}_tiling_data.h"'
    has_npu_tiling = '__NPU_TILING__' in kernel_src
    need_include = tiling_include not in kernel_src
    if need_include or not has_npu_tiling:
        insertion = ''
        if not has_npu_tiling:
            insertion += '#define __NPU_TILING__\n'
        if need_include:
            insertion += tiling_include + '\n'
        include_pat = r'(#\s*include\s*["\']kernel_operator\.h["\']\s*\n)'
        if re.search(include_pat, kernel_src):
            kernel_src = re.sub(include_pat, r'\1' + insertion, kernel_src, count=1)
        else:
            kernel_src = insertion + kernel_src
kernel_cpp_path = os.path.join(target_directory, 'op_kernel', f'{op}.cpp')
with open(kernel_cpp_path, 'w') as f:
    f.write(kernel_src)

print('\n=== KERNEL SOURCE (first 25 lines) ===')
for i, line in enumerate(kernel_src.split('\n')[:25], 1):
    print(f'  {i:3d}| {line}')

# 5. Generate tiling_data.h
cmake_util = os.path.join(target_directory, 'cmake', 'util')
sys.path.insert(0, cmake_util)
from tiling_data_def_build import gen_tiling
tiling_h_path = os.path.join(target_directory, 'op_host', f'{op}_tiling.h')
tiling_data_h_path = os.path.join(target_directory, 'op_kernel', f'{op}_tiling_data.h')
gen_tiling(tiling_h_path, tiling_data_h_path)
sys.path.pop(0)

print('\n=== GENERATED tiling_data.h ===')
with open(tiling_data_h_path) as f:
    print(f.read())

# 6. Patch makeself
makeself_path = os.path.join(target_directory, 'cmake', 'makeself.cmake')
if os.path.isfile(makeself_path):
    with open(makeself_path) as f:
        s = f.read()
    s = s.replace(' --sha256', '').replace('--sha256', '')
    with open(makeself_path, 'w') as f:
        f.write(s)

# 7. Build
deploy_path_abs = os.path.abspath(os.path.join(op_engineer_dir, f'opp_{op}'))
if os.path.isdir(deploy_path_abs):
    shutil.rmtree(deploy_path_abs)
os.makedirs(deploy_path_abs, exist_ok=True)
build_env = os.environ.copy()
build_env["ASCEND_CUSTOM_OPP_PATH"] = deploy_path_abs
os.chdir(target_directory)
print('\n=== BUILDING ===')
r = subprocess.run(["./build.sh"], capture_output=True, text=True, env=build_env)
if r.returncode != 0:
    print('[FAIL] build failed')
    for line in (r.stdout + '\n' + r.stderr).split('\n'):
        if '[ERROR]' in line or 'error:' in line or 'Error' in line:
            print('  ', line)
    sys.exit(1)
print('[OK] build succeeded')

# 8. Deploy
os.chdir(os.path.join(target_directory, 'build_out'))
deploy_env = os.environ.copy()
deploy_env.pop("ASCEND_CUSTOM_OPP_PATH", None)
deploy_env.pop("ASCEND_OPP_PATH", None)
r = subprocess.run(["./custom_opp_ubuntu_aarch64.run", f"--install-path={deploy_path_abs}"],
                   capture_output=True, text=True, env=deploy_env)
if r.returncode != 0:
    print('[FAIL] deploy failed')
    print(r.stdout[-2000:])
    sys.exit(1)
print('[OK] deploy succeeded')

# 9. Check deployed files
custom_opp_path = f"{deploy_path_abs}/vendors/customize"
lib_path = os.path.join(custom_opp_path, "op_api", "lib", "libcust_opapi.so")
print(f'\n=== DEPLOY CHECK ===')
print(f'ASCEND_CUSTOM_OPP_PATH = {custom_opp_path}')
print(f'libcust_opapi.so exists: {os.path.exists(lib_path)}')
if os.path.exists(lib_path):
    print(f'libcust_opapi.so size: {os.path.getsize(lib_path)} bytes')
    # Check symbols
    nm_r = subprocess.run(['nm', '-D', lib_path], capture_output=True, text=True)
    symbols = [l for l in nm_r.stdout.split('\n') if op_capital.replace('Custom', '') in l.lower() or 'aclnn' in l.lower()]
    print(f'Relevant symbols ({len(symbols)}):')
    for s in symbols[:20]:
        print(f'  {s}')

# List all files in deploy path
print(f'\nFiles in deploy path:')
for root, dirs, files in os.walk(deploy_path_abs):
    for fn in files:
        fp = os.path.join(root, fn)
        print(f'  {os.path.relpath(fp, deploy_path_abs)} ({os.path.getsize(fp)} bytes)')

# 10. Build pybind
cpp_ext_dir = os.path.join(op_engineer_dir, f'CppExtension_{op}')
if os.path.isdir(cpp_ext_dir):
    shutil.rmtree(cpp_ext_dir)
os.makedirs(os.path.join(cpp_ext_dir, 'csrc'), exist_ok=True)
shutil.copy2(os.path.join(op_engineer_dir, 'CppExtension', 'build_and_run.sh'), cpp_ext_dir)
shutil.copy2(os.path.join(op_engineer_dir, 'CppExtension', 'setup.py'), cpp_ext_dir)
shutil.copy2(os.path.join(op_engineer_dir, 'CppExtension', 'csrc', 'pytorch_npu_helper.hpp'),
             os.path.join(cpp_ext_dir, 'csrc'))
if os.path.exists(os.path.join(op_engineer_dir, 'CppExtension', 'csrc', 'CMakeLists.txt')):
    shutil.copy2(os.path.join(op_engineer_dir, 'CppExtension', 'csrc', 'CMakeLists.txt'),
                 os.path.join(cpp_ext_dir, 'csrc'))

python_bind_src_patched = (context.get('python_bind_src') or '').replace('custom_ops_lib', f'custom_ops_lib_{op}')
with open(os.path.join(cpp_ext_dir, 'csrc', 'op.cpp'), 'w') as f:
    f.write(python_bind_src_patched)

os.chdir(cpp_ext_dir)
pybind_env = os.environ.copy()
pybind_env['CUSTOM_OP_NAME'] = op
print('\n=== BUILDING PYBIND ===')
r = subprocess.run(['bash', 'build_and_run.sh'], capture_output=True, text=True, env=pybind_env)
if r.returncode != 0:
    print('[FAIL] pybind build failed')
    print(r.stdout[-2000:])
    sys.exit(1)
print('[OK] pybind build succeeded')

# 11. Set env and run the operator
os.environ["ASCEND_CUSTOM_OPP_PATH"] = custom_opp_path
custom_lib_path = f"{deploy_path_abs}/vendors/customize/op_api/lib/"
if custom_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"{custom_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

print(f'\n=== RUNNING OPERATOR ===')
print(f'ASCEND_CUSTOM_OPP_PATH = {os.environ["ASCEND_CUSTOM_OPP_PATH"]}')
print(f'LD_LIBRARY_PATH (first 200 chars) = {os.environ["LD_LIBRARY_PATH"][:200]}')

import torch
import torch_npu

# Import pybind module
mod_name = f'custom_ops_lib_{op}'
print(f'Importing {mod_name}...')
import importlib
mod = importlib.import_module(mod_name)
print(f'Module loaded: {mod}')

# Create test input
device = torch.device('npu:6')
x = torch.randn(16, 16, dtype=torch.float32, device=device)
torch_npu.npu.synchronize(device=device)

print(f'Input (first 10): {x.flatten()[:10].tolist()}')

# Call operator
op_func = getattr(mod, f'{op}')
y = op_func(x)
torch_npu.npu.synchronize(device=device)

print(f'Output (first 10): {y.flatten()[:10].tolist()}')

# Compare with reference
ref = torch.relu(x) if OP_NAME == 'relu' else x  # fallback
if OP_NAME == 'relu':
    ref = torch.relu(x)
elif OP_NAME == 'sigmoid':
    ref = torch.sigmoid(x)
elif OP_NAME == 'tanh':
    ref = torch.tanh(x)
elif OP_NAME == 'softmax':
    ref = torch.softmax(x, dim=-1)
print(f'Expected (first 10): {ref.flatten()[:10].tolist()}')

# Check
has_nan = torch.isnan(y).any().item()
print(f'Output has NaN: {has_nan}')
if not has_nan:
    max_diff = torch.max(torch.abs(ref - y)).item()
    print(f'Max diff: {max_diff}')

os.chdir(project_root_path)
print('\n=== DONE (artifacts preserved) ===')
print(f'Project dir: {target_directory}')
print(f'Deploy dir:  {deploy_path_abs}')
