import torch_npu
import torch
import shutil
from backends.backend_registry import register_backend, Backend
from utils.ascend_compile_pipeline import ascend_compile
from utils.correctness import execute_template
from utils.performance import time_execution_event_template
from utils.utils import underscore_to_pascalcase
from config import project_root_path, ascendc_device, op_engineer_dir
import os

@register_backend('ascendc')
class AscendBackend(Backend):
    def __init__(self):
        self.context = {}
        self.device = self.get_device()
        self.current_op = None  # 当前评测的 op，用于用完后删除工程目录

    def get_device(self):
        return torch.device('npu:6')

    def get_hardware_name(self):
        return ascendc_device  # torch_npu.npu.get_device_name(device) causes crash

    def compile(self, generated_code, op):
        self.current_op = op
        try:
            ascend_compile(generated_code, op, self.context)
            return True, None
        except Exception as e:
            os.chdir(project_root_path)
            return False, str(e)

    def correctness_execution(self, ref_src):
        synchronize = torch_npu.npu.synchronize
        try:
            exec(ref_src, self.context)
        except Exception as e:
            raise RuntimeError(f"Failed to compile reference model: {str(e)}")
        return execute_template(synchronize, self.device, self.context)

    def time_execution(self, eval_target='ModelNew'):
        event_class = torch_npu.npu.Event
        synchronize = torch_npu.npu.synchronize
        return time_execution_event_template(self.context, self.device, synchronize, event_class, eval_target)

    def cleanup(self):
        del self.context
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize(device=self.device)

    def cleanup_project_if_any(self):
        """评测结束后删除当前 op 在 ascend_op_projects 下的工程目录和 JSON，避免堆积。"""
        if self.current_op is None:
            return
        op_custom = self.current_op + '_custom'
        op_capital = underscore_to_pascalcase(op_custom)
        project_dir = os.path.join(op_engineer_dir, op_capital)
        json_path = os.path.join(op_engineer_dir, op_custom + '.json')
        cpp_ext_dir = os.path.join(op_engineer_dir, f'CppExtension_{op_custom}')
        opp_dir = os.path.join(op_engineer_dir, f'opp_{op_custom}')
        
        try:
            if os.path.isdir(project_dir):
                shutil.rmtree(project_dir, ignore_errors=True)
            if os.path.isfile(json_path):
                os.remove(json_path)
            if os.path.isdir(cpp_ext_dir):
                shutil.rmtree(cpp_ext_dir, ignore_errors=True)
            if os.path.isdir(opp_dir):
                shutil.rmtree(opp_dir, ignore_errors=True)
        except Exception:
            pass
        self.current_op = None
