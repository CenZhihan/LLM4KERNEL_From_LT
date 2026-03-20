#!/usr/bin/env python3
"""
Diagnostic: rebuild & deploy relu to the SHARED opp/ directory (like MKB original)
then test if the operator produces correct output.
"""
import os, sys, subprocess, shutil, re, importlib

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from config import op_engineer_dir, deploy_path, ascendc_device, project_root_path
from utils.utils import underscore_to_pascalcase

OP_NAME = sys.argv[1] if len(sys.argv) > 1 else 'relu'
TXT_DIR = f'output/ascendc/external165/0.0-1.0/external_lab/run0'
txt_path = os.path.join(TXT_DIR, f'{OP_NAME}.txt')

print(f'=== Diagnosing op (shared deploy): {OP_NAME} ===')

# 1. Read & exec generated code
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
with open(os.path.join(op_engineer_dir, f'{op}.json'), 'w') as f:
    f.write(context.get('project_json_src') or '')
os.chdir(op_engineer_dir)
subprocess.run(["msopgen", 'gen', '-i', f'{op}.json', '-c', ascendc_device, '-lan', 'cpp', '-out', op_capital],
               check=True, capture_output=True, text=True)
print('[OK] project created')

# 3. Write sources
with open(os.path.join(target_directory, 'op_host', f'{op}_tiling.h'), 'w') as f:
    f.write(context.get('host_tiling_src') or '')
with open(os.path.join(target_directory, 'op_host', f'{op}.cpp'), 'w') as f:
    f.write(context.get('host_operator_src') or '')

# 4. Kernel injection
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
with open(os.path.join(target_directory, 'op_kernel', f'{op}.cpp'), 'w') as f:
    f.write(kernel_src)

# 5. Generate tiling_data.h
cmake_util = os.path.join(target_directory, 'cmake', 'util')
sys.path.insert(0, cmake_util)
from tiling_data_def_build import gen_tiling
gen_tiling(os.path.join(target_directory, 'op_host', f'{op}_tiling.h'),
           os.path.join(target_directory, 'op_kernel', f'{op}_tiling_data.h'))
sys.path.pop(0)

# 6. Patch makeself
makeself_path = os.path.join(target_directory, 'cmake', 'makeself.cmake')
if os.path.isfile(makeself_path):
    with open(makeself_path) as f:
        s = f.read()
    s = s.replace(' --sha256', '').replace('--sha256', '')
    with open(makeself_path, 'w') as f:
        f.write(s)

# 7. Build - MKB style: POP ASCEND_CUSTOM_OPP_PATH
os.chdir(target_directory)
build_env = os.environ.copy()
build_env.pop("ASCEND_CUSTOM_OPP_PATH", None)
print('[INFO] Building (MKB style: no ASCEND_CUSTOM_OPP_PATH)...')
r = subprocess.run(["./build.sh"], capture_output=True, text=True, env=build_env)
if r.returncode != 0:
    print('[FAIL] build failed')
    for line in (r.stdout + '\n' + r.stderr).split('\n'):
        if '[ERROR]' in line or 'error:' in line or 'Error' in line or 'failed' in line.lower():
            print('  ', line)
    # Try with ASCEND_CUSTOM_OPP_PATH set to shared opp
    shared_opp = os.path.join(op_engineer_dir, 'opp')
    os.makedirs(shared_opp, exist_ok=True)
    build_env["ASCEND_CUSTOM_OPP_PATH"] = shared_opp
    print(f'[INFO] Retrying build with ASCEND_CUSTOM_OPP_PATH={shared_opp}')
    r = subprocess.run(["./build.sh"], capture_output=True, text=True, env=build_env)
    if r.returncode != 0:
        print('[FAIL] build failed again')
        for line in (r.stdout + '\n' + r.stderr).split('\n'):
            if '[ERROR]' in line or 'error:' in line or 'failed' in line.lower():
                print('  ', line)
        sys.exit(1)
print('[OK] build succeeded')

# 8. Deploy to SHARED opp/ - MKB style: no --install-path
shared_opp = os.path.join(op_engineer_dir, 'opp')
os.makedirs(shared_opp, exist_ok=True)
os.chdir(os.path.join(target_directory, 'build_out'))
deploy_env = os.environ.copy()
deploy_env.pop("ASCEND_CUSTOM_OPP_PATH", None)
deploy_env.pop("ASCEND_OPP_PATH", None)
# MKB style: no --install-path argument
print('[INFO] Deploying (MKB style: no --install-path, using ASCEND_CUSTOM_OPP_PATH)...')
# Actually MKB deploys WITHOUT --install-path. The installer reads ASCEND_CUSTOM_OPP_PATH
# or falls back to /usr/local/Ascend/... which we don't have permission for.
# So set ASCEND_CUSTOM_OPP_PATH to shared opp.
deploy_env["ASCEND_CUSTOM_OPP_PATH"] = shared_opp
r = subprocess.run(["./custom_opp_ubuntu_aarch64.run"], capture_output=True, text=True, env=deploy_env)
if r.returncode != 0:
    print('[FAIL] MKB-style deploy failed, trying --install-path')
    r = subprocess.run(["./custom_opp_ubuntu_aarch64.run", f"--install-path={shared_opp}"],
                       capture_output=True, text=True, env=deploy_env)
    if r.returncode != 0:
        print('[FAIL] deploy failed')
        print(r.stdout[-2000:])
        sys.exit(1)
print('[OK] deploy succeeded')

# 9. Check shared deploy
custom_opp_path = f"{shared_opp}/vendors/customize"
lib_path = os.path.join(custom_opp_path, "op_api", "lib", "libcust_opapi.so")
print(f'\n=== DEPLOY CHECK (shared) ===')
print(f'ASCEND_CUSTOM_OPP_PATH = {custom_opp_path}')
print(f'libcust_opapi.so exists: {os.path.exists(lib_path)}')
if os.path.exists(lib_path):
    nm_r = subprocess.run(['nm', '-D', lib_path], capture_output=True, text=True)
    symbols = [l for l in nm_r.stdout.split('\n') if 'aclnn' in l.lower() and 'relu' in l.lower()]
    print(f'Relevant symbols: {symbols}')

# 10. Build pybind (same as before)
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
r = subprocess.run(['bash', 'build_and_run.sh'], capture_output=True, text=True, env=pybind_env)
if r.returncode != 0:
    print('[FAIL] pybind failed')
    print(r.stdout[-2000:])
    sys.exit(1)
print('[OK] pybind succeeded')

# 11. Set env and run
os.environ["ASCEND_CUSTOM_OPP_PATH"] = custom_opp_path
custom_lib_path = f"{shared_opp}/vendors/customize/op_api/lib/"
if custom_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"{custom_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

print(f'\n=== RUNNING OPERATOR (shared deploy) ===')
print(f'ASCEND_CUSTOM_OPP_PATH = {os.environ["ASCEND_CUSTOM_OPP_PATH"]}')

import torch
import torch_npu

mod_name = f'custom_ops_lib_{op}'
mod = importlib.import_module(mod_name)
print(f'Module loaded: {mod}')

device = torch.device('npu:6')
x = torch.randn(16, 16, dtype=torch.float32, device=device)
torch_npu.npu.synchronize(device=device)
print(f'Input (first 10): {x.flatten()[:10].tolist()}')

op_func = getattr(mod, f'{op}')
y = op_func(x)
torch_npu.npu.synchronize(device=device)
print(f'Output (first 10): {y.flatten()[:10].tolist()}')

ref = torch.relu(x)
print(f'Expected (first 10): {ref.flatten()[:10].tolist()}')

has_nan = torch.isnan(y).any().item()
print(f'Output has NaN: {has_nan}')
if not has_nan:
    max_diff = torch.max(torch.abs(ref - y)).item()
    print(f'Max diff: {max_diff}')
    if max_diff < 1e-4:
        print('*** CORRECTNESS CHECK PASSED! ***')
    else:
        print(f'*** MISMATCH (max diff = {max_diff}) ***')

os.chdir(project_root_path)
print('\n=== DONE ===')
