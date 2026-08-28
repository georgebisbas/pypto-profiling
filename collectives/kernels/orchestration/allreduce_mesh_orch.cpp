/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Mesh AllReduce orchestration shim — dynamic-count variant.
 *
 * Passes count as an extra scalar so the AIV kernel can use dynamic tile shapes.
 *
 *   tensor(0) input   INPUT           (plain device mem)
 *   tensor(1) output  OUTPUT_EXISTING (plain device mem)
 *   tensor(2) scratch INOUT           (HCCL window slot; cross-rank read/write)
 *   scalar(0) count   (number of float32 elements)
 *   scalar(1) nranks
 *   scalar(2) CommContext device pointer
 */

#include <stdint.h>

#include "orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig
allreduce_mesh_orch_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{
        .expected_arg_count = 6,  // 3 tensors + 3 scalars
    };
}

__attribute__((visibility("default"))) void allreduce_mesh_orch(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &input = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &output = orch_args.tensor(1).ref();
    const simpler::tmr::Tensor &scratch = orch_args.tensor(2).ref();

    CoreTaskArgs params;
    params.add_input(input);
    params.add_output(output);
    params.add_inout(scratch);
    params.add_scalar(orch_args.scalar(0));  // count
    params.add_scalar(orch_args.scalar(1));  // nranks
    params.add_scalar(orch_args.scalar(2));  // CommContext
    rt_submit_aiv_task(0, params);
}

}  // extern "C"
