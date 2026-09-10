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
  // pto: %c4_index
  const int64_t v8 = 4;
  // pto: %c4096_index
  const int64_t v9 = 4096;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c0_index
  const int64_t v11 = 0;
  // pto: %inp__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v13 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %inp__ssa_v0_view
  int64_t v15 = v8 * v9;
  // pto: %inp__ssa_v0_view
  int64_t v16 = v14 * v15;
  // pto: %inp__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v8, v9);
  // pto: %inp__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v16, v16, v15, v9, v10);
  // pto: %inp__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v19 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v17, v18);
  // pto: %stage__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %stage__ssa_v0_view
  int64_t v23 = v8 * v9;
  // pto: %stage__ssa_v0_view
  int64_t v24 = v22 * v23;
  // pto: %stage__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v25 = pto::Shape<1, 1, 1, -1, -1>(v20, v21, v22, v8, v9);
  // pto: %stage__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v26 = pto::Stride<-1, -1, -1, -1, -1>(v21 * v24, v24, v23, v9, v10);
  // pto: %stage__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v27 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v25, v26);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v28 = v11; v28 < v8; v28 += v10) {
    // pto: %tile__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v29 = Tile<TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v9);
    // pto: %tile__ssa_v0
    ;
    uint64_t v30 = (uint64_t) v7;
    TASSIGN(v29, v30);
    // pto: %0
    ;
    int64_t v31 = v28 < v11 ? v11 : v28;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v32 = 0;
    // pto: %inp__ssa_v0_pview
    ;
    __gm__ int8_t* v33 = PTOAS__GLOBAL_TENSOR_DATA(v19);
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v34 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v35 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v36 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v37 = v10 * v9;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v38 = v36 * v37;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v39 = pto::Shape<1, 1, 1, 1, -1>(v34, v35, v36, v10, v9);
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v40 = pto::Stride<-1, -1, -1, -1, -1>(v35 * v38, v38, v37, v9, v10);
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v41 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v33 + (v32 + v31 * v9 + v11 * v10), v39, v40);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v29, v41);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v42 = 0;
    // pto: %stage__iter_v1_pview
    ;
    __gm__ int8_t* v43 = PTOAS__GLOBAL_TENSOR_DATA(v27);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v44 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v45 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v46 = 1;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v47 = v10 * v9;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v48 = v46 * v47;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v49 = pto::Shape<1, 1, 1, 1, -1>(v44, v45, v46, v10, v9);
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v50 = pto::Stride<-1, -1, -1, -1, -1>(v45 * v48, v48, v47, v9, v10);
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v51 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v43 + (v42 + v31 * v9 + v11 * v10), v49, v50);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v51, v29);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 4, 4096> v52 = pto::Shape<1, 1, 1, 4, 4096>();
    // pto: %2
    ;
    pto::Stride<16384, 16384, 16384, 4096, 1> v53 = pto::Stride<16384, 16384, 16384, 4096, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 4096>, pto::Stride<16384, 16384, 16384, 4096, 1>, pto::Layout::ND> v54 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 4096>, pto::Stride<16384, 16384, 16384, 4096, 1>, pto::Layout::ND>((__gm__ int8_t*) v3, v52, v53);
    __gm__ int8_t* v55 = PTOAS__GLOBAL_TENSOR_DATA(v54);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v55);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  for (int64_t v56 = v11; v56 < v8; v56 += v10) {
    // pto: %v__tile
    ;
    int32_t v57 = (v2)[v56];
    (v4)[v56] = v57;
    // pto: %counts__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 4, 1> v58 = pto::Shape<1, 1, 1, 4, 1>();
    // pto: %counts__ssa_v0_pview
    ;
    pto::Stride<4, 4, 4, 1, 4> v59 = pto::Stride<4, 4, 4, 1, 4>();
    // pto: %counts__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN> v60 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN>((__gm__ int32_t*) v4, v58, v59);
    __gm__ int32_t* v61 = PTOAS__GLOBAL_TENSOR_DATA(v60);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v61);
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