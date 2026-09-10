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

AICORE void fill_counts_step(__gm__ int32_t* v1, __gm__ int32_t* v2, __gm__ int64_t* v3) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c2_index
  const int64_t v4 = 2;
  // pto: %c1_index
  const int64_t v5 = 1;
  // pto: %c0_index
  const int64_t v6 = 0;
  for (int64_t v7 = v6; v7 < v4; v7 += v5) {
    // pto: %v__tile
    ;
    int32_t v8 = (v1)[v7];
    (v2)[v7] = v8;
    // pto: %counts__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 2, 1> v9 = pto::Shape<1, 1, 1, 2, 1>();
    // pto: %counts__ssa_v0_pview
    ;
    pto::Stride<2, 2, 2, 1, 2> v10 = pto::Stride<2, 2, 2, 1, 2>();
    // pto: %counts__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 2, 1>, pto::Stride<2, 2, 2, 1, 2>, pto::Layout::DN> v11 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 2, 1>, pto::Stride<2, 2, 2, 1, 2>, pto::Layout::DN>((__gm__ int32_t*) v2, v9, v10);
    __gm__ int32_t* v12 = PTOAS__GLOBAL_TENSOR_DATA(v11);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v12);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
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