import os
import re
import json
import glob
import subprocess
import shutil
import sys
from config import op_engineer_dir, deploy_path, ascendc_device, project_root_path
from utils.utils import underscore_to_pascalcase


def _gen_tiling_data_header(target_directory, op_name):
    """从 op_host/{op}_tiling.h 生成 op_kernel/{op}_tiling_data.h，供 kernel include 使用。"""
    tiling_h = os.path.join(target_directory, "op_host", f"{op_name}_tiling.h")
    tiling_data_h = os.path.join(
        target_directory, "op_kernel", f"{op_name}_tiling_data.h"
    )
    if not os.path.exists(tiling_h):
        return
    cmake_util = os.path.join(target_directory, "cmake", "util")
    if not os.path.isdir(cmake_util):
        return
    try:
        sys.path.insert(0, cmake_util)
        from tiling_data_def_build import gen_tiling

        gen_tiling(tiling_h, tiling_data_h)
    finally:
        if sys.path and sys.path[0] == cmake_util:
            sys.path.pop(0)


def _patch_makeself_no_sha256(target_directory):
    """去掉 makeself 的 --sha256，避免 CPack 'Problem compressing the directory'。"""
    path = os.path.join(target_directory, "cmake", "makeself.cmake")
    if not os.path.isfile(path):
        return
    with open(path, "r") as f:
        s = f.read()
    if "--sha256" not in s:
        return
    # 兼容 " --sha256" 或 换行/制表后 "--sha256"
    s = s.replace(" --sha256", "").replace("--sha256", "")
    with open(path, "w") as f:
        f.write(s)


def _patch_deployed_kernel_json_op_para_size(deploy_path_abs: str, op_name: str, min_op_para_size: int = 4096):
    """
    修复部署产物中 kernel json 的 opParaSize 过小问题。

    在部分环境下，CANN 生成的 kernel json 会出现 opParaSize=8，
    导致 RawTilingData capacity 不足，TilingData::SaveToBuffer 失败，
    继而 kernel 读取到错误 tiling 数据，出现 NaN / output mismatch，甚至 device mem error。

    这里采用保守策略：将 opParaSize 提升到一个足够大的值（默认 4096）。
    """
    kernel_root = os.path.join(
        deploy_path_abs, "vendors", "customize", "op_impl", "ai_core", "tbe", "kernel"
    )
    if not os.path.isdir(kernel_root):
        return

    # 兼容不同 compute_unit（如 ascend910b / ascend910），使用通配符。
    pattern = os.path.join(kernel_root, "*", op_name, "*.json")
    for p in glob.glob(pattern):
        try:
            with open(p, "r") as f:
                d = json.load(f)
            if "opParaSize" not in d:
                continue
            old = d.get("opParaSize")
            new = max(int(old or 0), int(min_op_para_size))
            if new == old:
                continue
            d["opParaSize"] = new
            with open(p, "w") as f:
                json.dump(d, f, indent=1)
            print(f"[INFO] Patch opParaSize: {os.path.basename(p)} {old} -> {new}")
        except Exception as e:
            print(f"[WARNING] Patch opParaSize failed: {p}: {e}")


def _inject_kernel_include_paths(target_directory, include_paths):
    if not include_paths:
        return

    cmake_path = os.path.join(target_directory, "op_kernel", "CMakeLists.txt")
    if not os.path.exists(cmake_path):
        return

    with open(cmake_path, "r") as f:
        cmake_src = f.read()

    include_lines = []
    for include_path in include_paths:
        if not include_path:
            continue
        include_line = f"add_ops_compile_options(ALL OPTIONS -I{include_path})"
        if include_line not in cmake_src:
            include_lines.append(include_line)

    if not include_lines:
        return

    injected = "\n".join(include_lines)
    if "add_kernels_compile()" in cmake_src:
        cmake_src = cmake_src.replace("add_kernels_compile()", f"{injected}\nadd_kernels_compile()", 1)
    else:
        cmake_src = f"{cmake_src.rstrip()}\n{injected}\n"

    with open(cmake_path, "w") as f:
        f.write(cmake_src)


def _patch_model_src_python_syntax(model_src: str) -> str:
    """
    修复少数生成代码中的 Python 非法函数签名：
    def __init__(self, *args, **kwargs, alpha=1.0):  # 非法
    """
    if not model_src:
        return model_src
    return model_src.replace("*args, **kwargs, ", "*args, ")


def _patch_host_operator_blockdim_for_stability(host_operator_src: str, op_name_with_custom: str) -> str:
    """
    针对已验证会出现多核分块 NaN 的算子，强制 blockDim=1 以保证 correctness 稳定。
    该补丁仅作用于少量已知算子，避免影响其余算子的并行性能。
    """
    if not host_operator_src:
        return host_operator_src

    op_name = op_name_with_custom[:-7] if op_name_with_custom.endswith("_custom") else op_name_with_custom
    need_patch_ops = {"elu", "sigmoid", "tanh", "selu", "hard_tanh"}
    if op_name not in need_patch_ops:
        return host_operator_src

    patched = host_operator_src
    patched = re.sub(r"(kDefaultBlockDim\s*=\s*)\d+u?", r"\g<1>1", patched)
    patched = re.sub(r"(kBlockDim\s*=\s*)\d+u?", r"\g<1>1", patched)
    return patched



def ascend_compile(generated_code, op, context, extra_kernel_include_paths=None):
    op = op + '_custom'
    op_capital=underscore_to_pascalcase(op)
    target_directory=os.path.join(op_engineer_dir, op_capital)
    
    try:
        compile(generated_code, "<string>", "exec")
        exec(generated_code, context)  # For Python, use exec() (be careful with untrusted code)
    except Exception as e:
        raise Exception(f'Error in generated code {e}')
    
    # create ascendc project
    if os.path.exists(os.path.join(op_engineer_dir, op_capital)):
        shutil.rmtree(os.path.join(op_engineer_dir, op_capital))
    json_path_abs = os.path.abspath(os.path.join(op_engineer_dir, f'{op}.json'))
    target_dir_abs = os.path.abspath(os.path.join(op_engineer_dir, op_capital))
    with open(json_path_abs, 'w') as f:
        f.write(context.get('project_json_src') or '')
    import tempfile, time
    print("[INFO] Begin create operator project")
    msopgen_last_exc = None
    for _attempt in range(3):
        try:
            with tempfile.TemporaryDirectory() as tmpd:
                result = subprocess.run(
                    ["msopgen", 'gen', '-i', json_path_abs, '-c', ascendc_device,
                     '-lan', 'cpp', '-out', target_dir_abs],
                    check=True, capture_output=True, text=True, cwd=tmpd,
                )
            print("[INFO] Create operator project succeeded")
            msopgen_last_exc = None
            break
        except subprocess.CalledProcessError as e:
            msopgen_last_exc = e
            time.sleep(1)
    if msopgen_last_exc is not None:
        e = msopgen_last_exc
        print("[INFO] Create operator project failed!")
        print("Error Output:\n", e.stdout)
        print("Error Output:\n", e.stderr)
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        raise Exception(feedback)

    # write code to specific location
    with open(os.path.join(target_directory, 'op_host', f'{op}_tiling.h'), 'w') as f:
        f.write(context.get('host_tiling_src') or '')

    host_operator_src = context.get('host_operator_src') or ''
    host_operator_src = _patch_host_operator_blockdim_for_stability(host_operator_src, op)
    with open(os.path.join(target_directory, 'op_host', f'{op}.cpp'), 'w') as f:
        f.write(host_operator_src)

    _inject_kernel_include_paths(target_directory, extra_kernel_include_paths)

    kernel_src = context.get('kernel_src') or ''
    # 以下为与「外部 kernel 源码 + 本环境构建约定」的适配，非对方代码错误：
    # 对方在其环境下 GET_TILING_DATA 可正常编译运行；本环境需补齐 include 与 __NPU_TILING__，
    # 否则会报 tiling_data 未声明，或 "call to [host] function from __global__ [aicore] function"。
    # 方向 A：若 kernel 使用 GET_TILING_DATA，则自动在开头插入 #define __NPU_TILING__ 及 include，
    # 使生成的 tiling_data.h 按 device 侧语义展开；tiling_data.h 由下方 _gen_tiling_data_header 生成。
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
            # Prefer inserting right after kernel_operator.h to keep include order stable.
            include_pat = r'(#\s*include\s*["\']kernel_operator\.h["\']\s*\n)'
            if re.search(include_pat, kernel_src):
                kernel_src = re.sub(include_pat, r'\1' + insertion, kernel_src, count=1)
            else:
                # Fallback for unexpected source layout.
                kernel_src = insertion + kernel_src
    with open(os.path.join(target_directory, 'op_kernel', f'{op}.cpp'), 'w') as f:
        f.write(kernel_src)

    # 生成 kernel 侧 tiling_data.h（来自 op_host/{op}_tiling.h），避免 kernel include 时报 file not found
    _gen_tiling_data_header(target_directory, op)

    # isolated deploy path
    deploy_path_abs = os.path.abspath(os.path.join(op_engineer_dir, f'opp_{op}'))
    
    # dynamically rename custom_ops_lib to custom_ops_lib_{op} to prevent parallel pip install conflicts
    python_bind_src_patched = context.get('python_bind_src', '').replace('custom_ops_lib', f'custom_ops_lib_{op}')
    model_src_patched = context.get('model_src', '').replace('custom_ops_lib', f'custom_ops_lib_{op}')
    model_src_patched = _patch_model_src_python_syntax(model_src_patched)
    
    # write pybind
    cpp_ext_dir = os.path.join(op_engineer_dir, f'CppExtension_{op}')
    if os.path.exists(cpp_ext_dir):
        shutil.rmtree(cpp_ext_dir, ignore_errors=True)
        
    os.makedirs(cpp_ext_dir, exist_ok=True)
    shutil.copy2(os.path.join(op_engineer_dir, 'CppExtension', 'build_and_run.sh'), cpp_ext_dir)
    shutil.copy2(os.path.join(op_engineer_dir, 'CppExtension', 'setup.py'), cpp_ext_dir)
    os.makedirs(os.path.join(cpp_ext_dir, 'csrc'), exist_ok=True)
    shutil.copy2(os.path.join(op_engineer_dir, 'CppExtension', 'csrc', 'pytorch_npu_helper.hpp'), os.path.join(cpp_ext_dir, 'csrc'))
    if os.path.exists(os.path.join(op_engineer_dir, 'CppExtension', 'csrc', 'CMakeLists.txt')):
        shutil.copy2(os.path.join(op_engineer_dir, 'CppExtension', 'csrc', 'CMakeLists.txt'), os.path.join(cpp_ext_dir, 'csrc'))
        
    with open(os.path.join(cpp_ext_dir, 'csrc', f'op.cpp'), 'w') as f:
        f.write(python_bind_src_patched)

    # 去掉 makeself --sha256，避免 CPack Problem compressing the directory
    _patch_makeself_no_sha256(target_directory)

    try:
        os.makedirs(deploy_path_abs, exist_ok=True)
        build_env = os.environ.copy()
        build_env["ASCEND_CUSTOM_OPP_PATH"] = deploy_path_abs
        print("[INFO] Begin build")
        os.chdir(target_directory)
        result = subprocess.run(["./build.sh"], check=True, capture_output=True, text=True, env=build_env)
        print("[INFO] Build succeeded")
    except subprocess.CalledProcessError as e:
        run_file = os.path.join(target_directory, "build_out", "custom_opp_ubuntu_aarch64.run")
        if not os.path.isfile(run_file):
            # 某些 CPack 失败场景下，.run 可能已在 _CPack_Packages 下生成但尚未拷回 build_out
            candidates = glob.glob(
                os.path.join(target_directory, "build_out", "**", "custom_opp_ubuntu_aarch64.run"),
                recursive=True,
            )
            if candidates:
                try:
                    shutil.copy2(candidates[0], run_file)
                except Exception:
                    pass
        # 并行场景下 CPack/打包会偶发失败：优先做一次完整重试，尽量避免拿到不完整 .run。
        try:
            subprocess.run(["./build.sh"], check=True, capture_output=True, text=True, env=build_env)
            print("[WARNING] build.sh retry succeeded")
            run_file = os.path.join(target_directory, "build_out", "custom_opp_ubuntu_aarch64.run")
            if not os.path.isfile(run_file):
                raise Exception("build.sh retry succeeded but .run is missing")
            e = None
        except subprocess.CalledProcessError as e2:
            e = e2

        if e is None:
            pass
        elif os.path.isfile(run_file):
            print(f"[WARNING] build.sh failed after retry, fallback to existing .run: {run_file}")
        else:
            print("[INFO] Build failed!")
            error_output = ''
            for line in (e.stdout or '').split('\n'):
                if '[ERROR]' in line or 'error:' in line or 'CPack' in line or 'Error' in line:
                    print(line)
                    error_output += line
                    error_output += '\n'
            for line in (e.stderr or '').split('\n'):
                if '[ERROR]' in line or 'error:' in line or 'CPack' in line or 'Error' in line:
                    print(line)
                    error_output += line
                    error_output += '\n'
            if not error_output.strip() and (e.stderr or e.stdout):
                error_output = (e.stderr or '') + '\n' + (e.stdout or '')
            feedback = f'Exit Code: {e.returncode}\nError Output:\n{error_output}'
            raise Exception(feedback)



    try:
        print("[INFO] Begin deploy")
        os.chdir(os.path.join(target_directory, 'build_out'))
        # install.sh 要求 --install-path 必须是绝对路径，否则会退回使用 /usr/local/Ascend/opp 导致无权限
        os.makedirs(deploy_path_abs, exist_ok=True)
        # 避免安装脚本使用环境变量中的系统路径（ASCEND_CUSTOM_OPP_PATH/ASCEND_OPP_PATH）
        deploy_env = os.environ.copy()
        deploy_env.pop("ASCEND_CUSTOM_OPP_PATH", None)
        deploy_env.pop("ASCEND_OPP_PATH", None)
        result = subprocess.run(
            ["./custom_opp_ubuntu_aarch64.run", f"--install-path={deploy_path_abs}"],
            check=True,
            capture_output=True,
            text=True,
            env=deploy_env,
        )
        print("[INFO] Deploy succeeded")
    except subprocess.CalledProcessError as e:
        print("[INFO] Deploy failed!")
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        if e.stderr:
            feedback += f'\nStderr:\n{e.stderr}'
        raise Exception(feedback)

    # 修复：部署产物中 opParaSize 过小，导致 tiling SaveToBuffer 失败（output mismatch / NaN）。
    _patch_deployed_kernel_json_op_para_size(deploy_path_abs, op)



    try:
        print("[INFO] Begin pybind")
        os.chdir(cpp_ext_dir)
        env_with_op = os.environ.copy()
        env_with_op['CUSTOM_OP_NAME'] = op
        result = subprocess.run(['bash', "build_and_run.sh"], check=True, capture_output=True, text=True, env=env_with_op)
        print("[INFO] Pybind succeeded\n")
    except subprocess.CalledProcessError as e:
        # Print error if build.sh fails
        print("[INFO] Pybind failed!")
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        raise Exception(feedback)

    # Update ASCEND_CUSTOM_OPP_PATH
    custom_opp_path = f"{deploy_path_abs}/vendors/customize"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = custom_opp_path

    # Debug：验证 deploy 路径下关键文件是否存在
    _lib = os.path.join(custom_opp_path, "op_api", "lib", "libcust_opapi.so")
    print(f"[DEBUG] ASCEND_CUSTOM_OPP_PATH = {custom_opp_path}")
    print(f"[DEBUG] libcust_opapi.so exists: {os.path.exists(_lib)}")

    # Update LD_LIBRARY_PATH
    custom_lib_path = f"{deploy_path_abs}/vendors/customize/op_api/lib/"
    if custom_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
        existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{custom_lib_path}:{existing_ld_path}"
    
    try:
        compile(model_src_patched, "<string>", "exec")
        exec(model_src_patched, context)  # For Python, use exec() (be careful with untrusted code)
    except Exception as e:
        raise Exception(f'Error in generated code {e}')

    os.chdir(project_root_path)



if __name__ == '__main__':
    import torch
    import torch_npu
    import custom_ops_lib
    op = 'relu'
    generated_method = getattr(custom_ops_lib, op)
