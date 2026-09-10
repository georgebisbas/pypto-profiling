# Kernel and Orchestration Configuration

from pathlib import Path

from simpler.task_interface import ArgDirection as _D

_ROOT_DIR = Path(__file__).parent

# Runtime configuration for tensormap_and_ringbuffer.
# AICPU thread count 0 selects the runtime's architecture default (a2a3: 4; a5: 5).
RUNTIME_CONFIG = {
	"runtime": "tensormap_and_ringbuffer",
	"aicpu_thread_num": 0,
}

ORCHESTRATION = {
	"source": str(_ROOT_DIR / "orchestration" / "chip_pipeline.cpp"),
	"function_name": "aicpu_orchestration_entry",
	"signature": [_D.IN, _D.IN, _D.OUT, _D.OUT, _D.INOUT, _D.INOUT, _D.INOUT, _D.INOUT, _D.INOUT],
}

KERNELS = [
	{"func_id": 0, "name": "stage_step", "source": str(_ROOT_DIR / "kernels" / "aiv" / "stage_step.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.INOUT, _D.INOUT]},
	{"func_id": 1, "name": "__builtin_all_to_all_v__int8", "source": str(_ROOT_DIR / "kernels" / "aiv" / "__builtin_all_to_all_v__int8.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.INOUT]},
	{"func_id": 2, "name": "consume_step", "source": str(_ROOT_DIR / "kernels" / "aiv" / "consume_step.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT, _D.OUT]},
]
