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

AICORE void stage_step(__gm__ int8_t* v1, __gm__ int8_t* v2, __gm__ int64_t* v3) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v4 = 0;
  // pto: %c2_index
  const int64_t v5 = 2;
  // pto: %c8192_index
  const int64_t v6 = 8192;
  // pto: %c1_index
  const int64_t v7 = 1;
  // pto: %c0_index
  const int64_t v8 = 0;
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v9 = v8; v9 < v5; v9 += v7) {
    // pto: %chunk__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v10 = Tile<TileType::Vec, int8_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v6);
    // pto: %chunk__ssa_v0
    ;
    uint64_t v11 = (uint64_t) v4;
    TASSIGN(v10, v11);
    // pto: %0
    ;
    int64_t v12 = v9 < v8 ? v8 : v9;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v13 = 0;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v14 = 8192;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, 8192> v15 = pto::Shape<1, 1, 1, 1, 8192>();
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<8192, 8192, 8192, 8192, 1> v16 = pto::Stride<8192, 8192, 8192, 8192, 1>();
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v17 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>((__gm__ int8_t*) v1 + (v13 + v12 * v14), v15, v16);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v10, v17);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v18 = 0;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v19 = 8192;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, 8192> v20 = pto::Shape<1, 1, 1, 1, 8192>();
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<8192, 8192, 8192, 8192, 1> v21 = pto::Stride<8192, 8192, 8192, 8192, 1>();
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v22 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>((__gm__ int8_t*) v2 + (v18 + v12 * v19), v20, v21);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v22, v10);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 2, 8192> v23 = pto::Shape<1, 1, 1, 2, 8192>();
    // pto: %2
    ;
    pto::Stride<16384, 16384, 16384, 8192, 1> v24 = pto::Stride<16384, 16384, 16384, 8192, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 2, 8192>, pto::Stride<16384, 16384, 16384, 8192, 1>, pto::Layout::ND> v25 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 2, 8192>, pto::Stride<16384, 16384, 16384, 8192, 1>, pto::Layout::ND>((__gm__ int8_t*) v2, v23, v24);
    __gm__ int8_t* v26 = PTOAS__GLOBAL_TENSOR_DATA(v25);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v26);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}