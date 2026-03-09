import os
import re
import subprocess
import shutil
import sys
from config import op_engineer_dir, deploy_path, ascendc_device, project_root_path
from utils.utils import underscore_to_pascalcase


def _gen_tiling_data_header(target_directory, op_name):
    """从 op_host/{op}_tiling.h 生成 op_kernel/{op}_tiling_data.h，build 时 cp op_kernel/*.* 会一并拷到 binary/.../src。"""
    tiling_h = os.path.join(target_directory, "op_host", f"{op_name}_tiling.h")
    tiling_data_h = os.path.join(target_directory, "op_kernel", f"{op_name}_tiling_data.h")
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


def _ensure_kernel_tiling_boilerplate(kernel_src, op_name):
    """若 kernel 使用 GET_TILING_DATA 但未包含 tiling 宏与头文件，则在 #include \"kernel_operator.h\" 后自动插入，减少因 LLM 漏写导致的编译失败。"""
    if not kernel_src or "GET_TILING_DATA" not in kernel_src:
        return kernel_src
    if "__NPU_TILING__" in kernel_src and "tiling_data.h" in kernel_src:
        return kernel_src
    # 在首次 #include "kernel_operator.h" 后插入（允许前后有空格）
    pattern = r'(#\s*include\s*["\']kernel_operator\.h["\']\s*\n)'
    insertion = (
        f'#define __NPU_TILING__\n'
        f'#include "{op_name}_tiling_data.h"\n'
    )
    if re.search(pattern, kernel_src):
        kernel_src = re.sub(pattern, r'\1' + insertion, kernel_src, count=1)
    return kernel_src


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
        print("[INFO] Operator project already exists, deleted")
        shutil.rmtree(os.path.join(op_engineer_dir, op_capital))
    with open(os.path.join(op_engineer_dir, f'{op}.json'), 'w') as f:
        f.write(context.get('project_json_src'))
    try:
        print("[INFO] Begin create operator project")
        os.chdir(op_engineer_dir)
        result = subprocess.run(["msopgen", 'gen', '-i', f'{op}.json', '-c', ascendc_device, '-lan', 'cpp', '-out', op_capital], check=True, capture_output=True, text=True)
        print("[INFO] Create operator project succeeded")
    except subprocess.CalledProcessError as e:
        print("[INFO] Create operator project failed!")
        # print("Exit Code:", e.returncode)
        print("Error Output:\n", e.stdout)
        print("Error Output:\n", e.stderr)
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        raise Exception(feedback) 

    # write code to specific location
    with open(os.path.join(target_directory, 'op_host', f'{op}_tiling.h'), 'w') as f:
        f.write(context.get('host_tiling_src'))

    with open(os.path.join(target_directory, 'op_host', f'{op}.cpp'), 'w') as f:
        f.write(context.get('host_operator_src'))

    _inject_kernel_include_paths(target_directory, extra_kernel_include_paths)

    kernel_src = context.get('kernel_src') or ''
    kernel_src = _ensure_kernel_tiling_boilerplate(kernel_src, op)
    with open(os.path.join(target_directory, 'op_kernel', f'{op}.cpp'), 'w') as f:
        f.write(kernel_src)

    # isolated deploy path
    deploy_path_abs = os.path.abspath(os.path.join(op_engineer_dir, f'opp_{op}'))
    
    # dynamically rename custom_ops_lib to custom_ops_lib_{op} to prevent parallel pip install conflicts
    python_bind_src_patched = context.get('python_bind_src', '').replace('custom_ops_lib', f'custom_ops_lib_{op}')
    model_src_patched = context.get('model_src', '').replace('custom_ops_lib', f'custom_ops_lib_{op}')
    
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

    # 生成 kernel 用 tiling_data.h 到 op_kernel/，build 时 cp op_kernel/*.* 会拷到 binary/.../src，避免 'add_custom_tiling_data.h' file not found
    _gen_tiling_data_header(target_directory, op)

    # 去掉 makeself --sha256，避免 CPack Problem compressing the directory
    _patch_makeself_no_sha256(target_directory)

    try:
        # build.sh 在打包后会执行 ./cust*.run 且不传参，install.sh 会读 ASCEND_CUSTOM_OPP_PATH 或退回到 /usr/local
        # 必须让这次执行安装到项目内 opp，否则会报 create /usr/local/Ascend/.../framework failed
        os.makedirs(deploy_path_abs, exist_ok=True)
        build_env = os.environ.copy()
        build_env["ASCEND_CUSTOM_OPP_PATH"] = deploy_path_abs
        print("[INFO] Begin build")
        os.chdir(target_directory)
        result = subprocess.run(["./build.sh"], check=True, capture_output=True, text=True, env=build_env)
        print("[INFO] Build succeeded")
    except subprocess.CalledProcessError as e:
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
        # CPack/makeself 的详细错误常在 stderr，若过滤后为空则附上最后一段原始输出便于排查
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
