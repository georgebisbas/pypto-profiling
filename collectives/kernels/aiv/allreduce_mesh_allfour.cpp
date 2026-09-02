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
 * Mesh AllReduce kernel — ALL-FOUR ABLATION (plan 107) — composition check.
 *
 * handicapped (per-peer pipe_barrier before TNOTIFY + whole-cache dcci after
 * TWAIT, already measured at +25% P=2 / +15% P=4) PLUS the three plan-107
 * constructs: chunkbarrier, chunkdcci, fillpad.
 *
 * If the four compose to the measured 6.7x composite gap the decomposition is
 * complete. If they fall short, a FIFTH cause exists and the campaign must say
 * so rather than report a partial decomposition as complete.
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
static constexpr int kChunkCols = 16384;

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
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, kChunkCols, pto::BLayout::RowMajor,
                               -1, -1, pto::SLayout::NoneBox, 512, pto::PadValue::Zero,
                               pto::CompactMode::Null>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 0 || nranks > kMaxSupportedRanks || count <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    TileData stageTile(1, kChunkCols);
    TileData accTile(1, kChunkCols);
    TileData recvTile(1, kChunkCols);

    TASSIGN(stageTile, 0x0);
    TASSIGN(accTile, 0x10000);
    TASSIGN(recvTile, 0x20000);

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
    // Phase 2: device barrier.
    // HANDICAP 1: per-peer full pipeline barrier before each TNOTIFY,
    // exactly as the generated composite kernel emits.
    // ------------------------------------------------------------------
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        pipe_barrier(PIPE_ALL);
        __gm__ int32_t *remote_signal = CommRemotePtr(commCtx, signal_base + my_rank, peer);
        pto::comm::Signal sig(remote_signal);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    // HANDICAP 2: whole-data-cache invalidate after every peer TWAIT,
    // exactly as the generated composite kernel emits.
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        pto::comm::Signal sig(signal_base + peer);
        pto::comm::TWAIT(sig, (int32_t)1, pto::comm::WaitCmp::GE);
        dcci((__gm__ void *)0, cache_line_t::ENTIRE_DATA_CACHE);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 3 + 4: compute and stage-out (chunked) — unchanged from baseline.
    // ------------------------------------------------------------------
    for (int offset = 0; offset < count; offset += kChunkCols) {
        int chunkCount = (offset + kChunkCols <= count) ? kChunkCols : (count - offset);

        accTile.ColMaskInternal = chunkCount;
        recvTile.ColMaskInternal = chunkCount;

        ShapeDyn shape(1, 1, 1, 1, chunkCount);
        Global scratchG(scratch + offset, shape, fullStride);
        Global outputG(output + offset, shape, fullStride);

        TLOAD(accTile, scratchG);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == my_rank) continue;
            __gm__ float *remote_scratch = CommRemotePtr(commCtx, scratch, peer);
            Global remoteG(remote_scratch + offset, shape, fullStride);
            TLOAD(recvTile, remoteG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            // fillpad
            TFILLPAD<pto::TFillPadMode::InPlace>(recvTile, recvTile);
            TADD(accTile, accTile, recvTile);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            // chunkdcci
            dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
        }

        // chunkbarrier
        pipe_barrier(PIPE_ALL);
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == my_rank) continue;
            __gm__ int32_t *rs = CommRemotePtr(commCtx, signal_base + my_rank, peer);
            pto::comm::Signal s2(rs);
            pto::comm::TNOTIFY(s2, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
        }
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == my_rank) continue;
            pto::comm::Signal s2(signal_base + peer);
            pto::comm::TWAIT(s2, (int32_t)(2 + offset / kChunkCols), pto::comm::WaitCmp::GE);
        }
        pipe_barrier(PIPE_ALL);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(outputG, accTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
