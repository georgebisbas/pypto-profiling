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
  // pto: %c2_index
  const int64_t v8 = 2;
  // pto: %c1024_index
  const int64_t v9 = 1024;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c0_index
  const int64_t v11 = 0;
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v12 = v11; v12 < v8; v12 += v10) {
    // pto: %chunk__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v13 = Tile<TileType::Vec, int8_t, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v9);
    // pto: %chunk__ssa_v0
    ;
    uint64_t v14 = (uint64_t) v7;
    TASSIGN(v13, v14);
    // pto: %0
    ;
    int64_t v15 = v12 < v11 ? v11 : v12;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v16 = 0;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v17 = 1024;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 1024> v18 = pto::Shape<1, 1, 1, 1, 1024>();
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<1024, 1024, 1024, 1024, 1> v19 = pto::Stride<1024, 1024, 1024, 1024, 1>();
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v20 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>((__gm__ int8_t*) v1 + (v16 + v15 * v17), v18, v19);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v13, v20);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v21 = 0;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v22 = 1024;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, 1024> v23 = pto::Shape<1, 1, 1, 1, 1024>();
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<1024, 1024, 1024, 1024, 1> v24 = pto::Stride<1024, 1024, 1024, 1024, 1>();
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v25 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>((__gm__ int8_t*) v3 + (v21 + v15 * v22), v23, v24);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v25, v13);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 2, 1024> v26 = pto::Shape<1, 1, 1, 2, 1024>();
    // pto: %2
    ;
    pto::Stride<2048, 2048, 2048, 1024, 1> v27 = pto::Stride<2048, 2048, 2048, 1024, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 2, 1024>, pto::Stride<2048, 2048, 2048, 1024, 1>, pto::Layout::ND> v28 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 2, 1024>, pto::Stride<2048, 2048, 2048, 1024, 1>, pto::Layout::ND>((__gm__ int8_t*) v3, v26, v27);
    __gm__ int8_t* v29 = PTOAS__GLOBAL_TENSOR_DATA(v28);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v29);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  for (int64_t v30 = v11; v30 < v8; v30 += v10) {
    // pto: %v__tile
    ;
    int32_t v31 = (v2)[v30];
    (v4)[v30] = v31;
    // pto: %counts__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 2, 1> v32 = pto::Shape<1, 1, 1, 2, 1>();
    // pto: %counts__ssa_v0_pview
    ;
    pto::Stride<2, 2, 2, 1, 2> v33 = pto::Stride<2, 2, 2, 1, 2>();
    // pto: %counts__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 2, 1>, pto::Stride<2, 2, 2, 1, 2>, pto::Layout::DN> v34 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 2, 1>, pto::Stride<2, 2, 2, 1, 2>, pto::Layout::DN>((__gm__ int32_t*) v4, v32, v33);
    __gm__ int32_t* v35 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v35);
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