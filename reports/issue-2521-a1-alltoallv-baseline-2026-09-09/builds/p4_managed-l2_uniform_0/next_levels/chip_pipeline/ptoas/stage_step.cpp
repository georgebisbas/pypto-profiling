#include "pto/pto-inst.hpp"
using namespace pto;

template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}


enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static AICORE inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

AICORE void stage_step(__gm__ int8_t* v1, __gm__ int32_t* v2, __gm__ int8_t* v3, __gm__ int32_t* v4, __gm__ int64_t* v5, __gm__ int64_t* v6) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v7 = 0;
  // pto: %c1012_index
  const int64_t v8 = 1012;
  // pto: %c4160_index
  const int64_t v9 = 4160;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c4_index
  const int64_t v11 = 4;
  // pto: %c0_index
  const int64_t v12 = 0;
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v13 = v12; v13 < v8; v13 += v10) {
    // pto: %chunk__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 4160, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v14 = Tile<TileType::Vec, int8_t, 1, 4160, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v9);
    // pto: %chunk__ssa_v0
    ;
    uint64_t v15 = (uint64_t) v7;
    TASSIGN(v14, v15);
    // pto: %0
    ;
    int64_t v16 = v13 < v12 ? v12 : v13;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v17 = 0;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v18 = 4160;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 4160> v19 = pto::Shape<1, 1, 1, 1, 4160>();
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<4160, 4160, 4160, 4160, 1> v20 = pto::Stride<4160, 4160, 4160, 4160, 1>();
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4160>, pto::Stride<4160, 4160, 4160, 4160, 1>, pto::Layout::ND> v21 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4160>, pto::Stride<4160, 4160, 4160, 4160, 1>, pto::Layout::ND>((__gm__ int8_t*) v1 + (v17 + v16 * v18), v19, v20);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v14, v21);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v22 = 0;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v23 = 4160;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, 4160> v24 = pto::Shape<1, 1, 1, 1, 4160>();
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<4160, 4160, 4160, 4160, 1> v25 = pto::Stride<4160, 4160, 4160, 4160, 1>();
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4160>, pto::Stride<4160, 4160, 4160, 4160, 1>, pto::Layout::ND> v26 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4160>, pto::Stride<4160, 4160, 4160, 4160, 1>, pto::Layout::ND>((__gm__ int8_t*) v3 + (v22 + v16 * v23), v24, v25);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v26, v14);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 1012, 4160> v27 = pto::Shape<1, 1, 1, 1012, 4160>();
    // pto: %2
    ;
    pto::Stride<4209920, 4209920, 4209920, 4160, 1> v28 = pto::Stride<4209920, 4209920, 4209920, 4160, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1012, 4160>, pto::Stride<4209920, 4209920, 4209920, 4160, 1>, pto::Layout::ND> v29 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1012, 4160>, pto::Stride<4209920, 4209920, 4209920, 4160, 1>, pto::Layout::ND>((__gm__ int8_t*) v3, v27, v28);
    __gm__ int8_t* v30 = PTOAS__GLOBAL_TENSOR_DATA(v29);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v30);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  for (int64_t v31 = v12; v31 < v11; v31 += v10) {
    // pto: %v__tile
    ;
    int32_t v32 = (v2)[v31];
    (v4)[v31] = v32;
    // pto: %counts__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 4, 1> v33 = pto::Shape<1, 1, 1, 4, 1>();
    // pto: %counts__ssa_v0_pview
    ;
    pto::Stride<4, 4, 4, 1, 4> v34 = pto::Stride<4, 4, 4, 1, 4>();
    // pto: %counts__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN> v35 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN>((__gm__ int32_t*) v4, v33, v34);
    __gm__ int32_t* v36 = PTOAS__GLOBAL_TENSOR_DATA(v35);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v36);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  #endif // __DAV_VEC__

  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}