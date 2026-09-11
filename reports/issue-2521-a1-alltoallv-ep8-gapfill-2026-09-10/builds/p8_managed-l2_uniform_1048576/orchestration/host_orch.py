import torch
from simpler.task_interface import CallConfig, CommBufferSpec, DataType, TaskArgs, TensorArgType
from pypto.runtime.tensor_arg import make_tensor_arg
from pypto.runtime.distributed_runner import _submit_chip

def _alloc_intermediates(tensors, world_size=1):
    tensors["__comm_d0_ord"] = [torch.zeros((1,), dtype=torch.int32).share_memory_() for _ in range(max(world_size, 1))]

def host_orch(orch, _args, config, *, tensors, callables, sub_ids, _keep, world_size, _domain_provider=None):
    with (_domain_provider or orch.allocate_domain)(
        name="comm_d0",
        workers=[*range(world_size)],
        window_size=((((8388608 + 31) // 32) * 32)) + ((((8388608 + 31) // 32) * 32)) + ((((32 + 31) // 32) * 32)) + ((((32 + 31) // 32) * 32)) + ((((32 + 31) // 32) * 32)),
        buffers=[
            CommBufferSpec(name="stage_buf__ssa_v0", dtype="opaque", count=8388608, nbytes=(((8388608 + 31) // 32) * 32)),
            CommBufferSpec(name="data_buf__ssa_v0", dtype="opaque", count=8388608, nbytes=(((8388608 + 31) // 32) * 32)),
            CommBufferSpec(name="signal_buf__ssa_v0", dtype="opaque", count=32, nbytes=(((32 + 31) // 32) * 32)),
            CommBufferSpec(name="counts_buf__ssa_v0", dtype="opaque", count=32, nbytes=(((32 + 31) // 32) * 32)),
            CommBufferSpec(name="recv_buf__ssa_v0", dtype="opaque", count=32, nbytes=(((32 + 31) // 32) * 32)),
        ],
    ) as __comm_d0:
        t__tmp_v0 = world_size
        for r__idx_v0 in range(0, t__tmp_v0, 1):
            tensors["t__tmp_v1"] = tensors["inputs__ssa_v0"][r__idx_v0, 0:8, 0:1048576]
            tensors["t__tmp_v2"] = tensors["send_counts__ssa_v0"][r__idx_v0, 0:8, 0:1]
            tensors["t__tmp_v3"] = tensors["outputs__ssa_v0"][r__idx_v0, 0:8, 0:1048576]
            tensors["t__tmp_v4"] = tensors["recv_outputs__ssa_v0"][r__idx_v0, 0:8, 0:1]
            _ta_0 = TaskArgs()
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v1"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v2"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v3"]), TensorArgType.OUTPUT_EXISTING)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v4"]), TensorArgType.OUTPUT_EXISTING)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["stage_buf__ssa_v0"].tensor(shapes=(8, 1048576), dtype=DataType.INT8), TensorArgType.INOUT)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["data_buf__ssa_v0"].tensor(shapes=(8, 1048576), dtype=DataType.INT8), TensorArgType.INOUT)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["signal_buf__ssa_v0"].tensor(shapes=(8, 1), dtype=DataType.INT32), TensorArgType.INOUT)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["counts_buf__ssa_v0"].tensor(shapes=(8, 1), dtype=DataType.INT32), TensorArgType.INOUT)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["recv_buf__ssa_v0"].tensor(shapes=(8, 1), dtype=DataType.INT32), TensorArgType.INOUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["__comm_d0_ord"][r__idx_v0]), TensorArgType.INOUT)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _keep.append(_ta_0)
            _submit_chip(orch, callables["chip_pipeline"], _ta_0, config, r__idx_v0)

host_orch._pypto_distributed_entry = True

