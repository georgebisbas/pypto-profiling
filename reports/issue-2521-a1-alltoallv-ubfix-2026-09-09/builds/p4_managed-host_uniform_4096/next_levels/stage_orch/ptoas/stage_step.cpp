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
  // pto: %c4_index
  const int64_t v5 = 4;
  // pto: %c4096_index
  const int64_t v6 = 4096;
  // pto: %c1_index
  const int64_t v7 = 1;
  // pto: %c0_index
  const int64_t v8 = 0;
  // pto: %inp__ssa_v0_view
  const int64_t v9 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v10 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v11 = 1;
  // pto: %inp__ssa_v0_view
  int64_t v12 = v5 * v6;
  // pto: %inp__ssa_v0_view
  int64_t v13 = v11 * v12;
  // pto: %inp__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v14 = pto::Shape<1, 1, 1, -1, -1>(v9, v10, v11, v5, v6);
  // pto: %inp__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v15 = pto::Stride<-1, -1, -1, -1, -1>(v10 * v13, v13, v12, v6, v7);
  // pto: %inp__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v16 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v14, v15);
  // pto: %stage__ssa_v0_view
  const int64_t v17 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v18 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %stage__ssa_v0_view
  int64_t v20 = v5 * v6;
  // pto: %stage__ssa_v0_view
  int64_t v21 = v19 * v20;
  // pto: %stage__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v22 = pto::Shape<1, 1, 1, -1, -1>(v17, v18, v19, v5, v6);
  // pto: %stage__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v23 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v21, v21, v20, v6, v7);
  // pto: %stage__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v24 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v22, v23);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v25 = v8; v25 < v5; v25 += v7) {
    // pto: %tile__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v26 = Tile<TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v6);
    // pto: %tile__ssa_v0
    ;
    uint64_t v27 = (uint64_t) v4;
    TASSIGN(v26, v27);
    // pto: %0
    ;
    int64_t v28 = v25 < v8 ? v8 : v25;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v29 = 0;
    // pto: %inp__ssa_v0_pview
    ;
    __gm__ int8_t* v30 = PTOAS__GLOBAL_TENSOR_DATA(v16);
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v31 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v32 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v33 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v34 = v7 * v6;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v35 = v33 * v34;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v36 = pto::Shape<1, 1, 1, 1, -1>(v31, v32, v33, v7, v6);
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v37 = pto::Stride<-1, -1, -1, -1, -1>(v32 * v35, v35, v34, v6, v7);
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v38 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v30 + (v29 + v28 * v6 + v8 * v7), v36, v37);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v26, v38);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v39 = 0;
    // pto: %stage__iter_v1_pview
    ;
    __gm__ int8_t* v40 = PTOAS__GLOBAL_TENSOR_DATA(v24);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v41 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v42 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v43 = 1;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v44 = v7 * v6;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v45 = v43 * v44;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v46 = pto::Shape<1, 1, 1, 1, -1>(v41, v42, v43, v7, v6);
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v47 = pto::Stride<-1, -1, -1, -1, -1>(v42 * v45, v45, v44, v6, v7);
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v48 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v40 + (v39 + v28 * v6 + v8 * v7), v46, v47);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v48, v26);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 4, 4096> v49 = pto::Shape<1, 1, 1, 4, 4096>();
    // pto: %2
    ;
    pto::Stride<16384, 16384, 16384, 4096, 1> v50 = pto::Stride<16384, 16384, 16384, 4096, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 4096>, pto::Stride<16384, 16384, 16384, 4096, 1>, pto::Layout::ND> v51 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 4096>, pto::Stride<16384, 16384, 16384, 4096, 1>, pto::Layout::ND>((__gm__ int8_t*) v2, v49, v50);
    __gm__ int8_t* v52 = PTOAS__GLOBAL_TENSOR_DATA(v51);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v52);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}