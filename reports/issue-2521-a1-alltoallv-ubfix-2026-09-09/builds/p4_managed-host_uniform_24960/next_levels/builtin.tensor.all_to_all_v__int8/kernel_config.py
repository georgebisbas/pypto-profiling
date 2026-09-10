# Kernel and Orchestration Configuration
from pathlib import Path

from simpler.task_interface import ArgDirection as _D

_ROOT_DIR = Path(__file__).parent

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 0,
}

ORCHESTRATION = {
    "source": str(_ROOT_DIR / "orchestration" / "builtin_tensor_all_to_all_v__int8.cpp"),
    "function_name": "aicpu_orchestration_entry",
    "signature": [_D.IN, _D.OUT, _D.OUT, _D.IN, _D.OUT],
}

KERNELS = [
    {
        "func_id": 0,
        "name": "builtin_tensor_all_to_all_v__int8_kernel",
        "source": str(_ROOT_DIR / "kernels" / "aiv" / "builtin_tensor_all_to_all_v__int8_kernel.cpp"),
        "core_type": "aiv",
        "signature": [_D.IN, _D.OUT, _D.OUT, _D.IN, _D.OUT],
    },
]
