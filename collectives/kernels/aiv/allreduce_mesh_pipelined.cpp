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
 * Mesh AllReduce kernel — PIPELINED variant (performance idea #1).
 *
 * Same 4-phase mesh algorithm + golden contract as allreduce_mesh.cpp, but Phase 3
 * (compute) is pipelined:
 *
 *   1. All (P-1) peer remote loads for a chunk are issued back-to-back into
 *      per-peer recv tiles (no per-peer wait), so the MTE2 queue / HCCS link stays
 *      busy instead of idling between load->add->load->add.
 *   2. Chunk width is reduced (kChunkCols=2048) so UB can hold one recv tile per
 *      peer (P<=16) plus acc + stage tiles.
 *
 * This targets the serial-per-chunk, serial-per-peer structure the generated pypto
 * composite kernel exhibits (and that our bandwidth model shows leaves ~2x on the
 * table in the 64-256KB latency regime).
 *
 * args layout: same as allreduce_mesh.cpp
 *   tensor(0) = input    tensor(1) = output    tensor(2) = scratch (HCCL window)
 *   scalar(0) = count    scalar(1) = nranks    scalar(2) = CommContext
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "platform_comm/comm_context.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr int kMaxSupportedRanks = 16;
// Pipelined chunk width: 2048 float32 = 8 KiB per tile. 16 tiles (1 stage + 1 acc +
// up to 14 peer-recv) = 128 KiB < UB.
static constexpr int kChunkCols = 2048;
// Byte offsets for UB tile slots (kChunkCols * 4 B each).
static constexpr uint64_t kStageAddr = 0x0;
static constexpr uint64_t kAccAddr = 0x2000;
static constexpr uint64_t kRecvBase = 0x4000;  // per-peer recv tiles: kRecvBase + peer*0x2000

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ TaskTensor *input_tensor = reinterpret_cast<__gm__ TaskTensor *>(args[0]);
    __gm__ TaskTensor *output_tensor = reinterpret_cast<__gm__ TaskTensor *>(args[1]);
    __gm__ TaskTensor *scratch_tensor = reinterpret_cast<__gm__ TaskTensor *>(args[2]);
    int count = static_cast<int>(args[3]);
    int nranks = static_cast<int>(args[4]);
    __gm__ CommContext *commCtx = reinterpret_cast<__gm__ CommContext *>(args[5]);

    __gm__ float *input = reinterpret_cast<__gm__ float *>(input_tensor->buffer.addr) + input_tensor->start_offset;
    __gm__ float *output = reinterpret_cast<__gm__ float *>(output_tensor->buffer.addr) + output_tensor->start_offset;
    __gm__ float *scratch =
        reinterpret_cast<__gm__ float *>(scratch_tensor->buffer.addr) + scratch_tensor->start_offset;
    __gm__ int32_t *signal_base = reinterpret_cast<__gm__ int32_t *>(scratch + count);

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, kChunkCols, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 0 || nranks > kMaxSupportedRanks || count <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    TileData stageTile(1, kChunkCols);
    TileData accTile(1, kChunkCols);
    TileData recvTiles[kMaxSupportedRanks];  // [peer] recv tile; [my_rank] unused
    TASSIGN(stageTile, kStageAddr);
    TASSIGN(accTile, kAccAddr);
    for (int peer = 0; peer < kMaxSupportedRanks; ++peer) {
        TASSIGN(recvTiles[peer], kRecvBase + (uint64_t)peer * (uint64_t)(kChunkCols * 4));
    }

    StrideDyn fullStride(count, count, count, count, 1);

    // ------------------------------------------------------------------
    // Phase 1: stage-in — copy local input into my scratch slot (chunked).
    // ------------------------------------------------------------------
    for (int offset = 0; offset < count; offset += kChunkCols) {
        int chunkCount = (offset + kChunkCols <= count) ? kChunkCols : (count - offset);

        stageTile.ColMaskInternal = chunkCount;

        ShapeDyn shape(1, 1, 1, 1, chunkCount);
        Global inputG(input + offset, shape, fullStride);
        Global scratchG(scratch + offset, shape, fullStride);

        TLOAD(stageTile, inputG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(scratchG, stageTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 2: device barrier (signal matrix).
    // ------------------------------------------------------------------
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_signal = CommRemotePtr(commCtx, signal_base + my_rank, peer);
        pto::comm::Signal sig(remote_signal);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        pto::comm::Signal sig(signal_base + peer);
        pto::comm::TWAIT(sig, (int32_t)1, pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 3 + 4: PIPELINED compute + stage-out.
    //
    // For each chunk: issue ALL (P-1) peer remote loads back-to-back (async,
    // no per-peer wait), wait once, then reduce. The MTE2 pipe stays busy with
    // the full set of peer reads instead of alternating load->wait->add.
    // ------------------------------------------------------------------
    for (int offset = 0; offset < count; offset += kChunkCols) {
        int chunkCount = (offset + kChunkCols <= count) ? kChunkCols : (count - offset);

        accTile.ColMaskInternal = chunkCount;
        for (int peer = 0; peer < nranks; ++peer) {
            recvTiles[peer].ColMaskInternal = chunkCount;
        }

        ShapeDyn shape(1, 1, 1, 1, chunkCount);
        Global scratchG(scratch + offset, shape, fullStride);
        Global outputG(output + offset, shape, fullStride);

        // Own scratch -> acc.
        TLOAD(accTile, scratchG);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Fire all peer remote loads back-to-back (no waits in between).
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == my_rank) continue;
            __gm__ float *remote_scratch = CommRemotePtr(commCtx, scratch, peer);
            Global remoteG(remote_scratch + offset, shape, fullStride);
            TLOAD(recvTiles[peer], remoteG);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        // Wait for acc + all peer loads, then reduce.
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == my_rank) continue;
            TADD(accTile, accTile, recvTiles[peer]);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(outputG, accTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
