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
  // pto: %c32768_index
  const int64_t v9 = 32768;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c0_index
  const int64_t v11 = 0;
  // pto: %c16384_index
  const int64_t v12 = 16384;
  // pto: %inp__ssa_v0_view
  const int64_t v13 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v15 = 1;
  // pto: %inp__ssa_v0_view
  int64_t v16 = v8 * v9;
  // pto: %inp__ssa_v0_view
  int64_t v17 = v15 * v16;
  // pto: %inp__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v18 = pto::Shape<1, 1, 1, -1, -1>(v13, v14, v15, v8, v9);
  // pto: %inp__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v19 = pto::Stride<-1, -1, -1, -1, -1>(v14 * v17, v17, v16, v9, v10);
  // pto: %inp__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v20 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v18, v19);
  // pto: %stage__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %stage__ssa_v0_view
  int64_t v24 = v8 * v9;
  // pto: %stage__ssa_v0_view
  int64_t v25 = v23 * v24;
  // pto: %stage__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v26 = pto::Shape<1, 1, 1, -1, -1>(v21, v22, v23, v8, v9);
  // pto: %stage__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v27 = pto::Stride<-1, -1, -1, -1, -1>(v22 * v25, v25, v24, v9, v10);
  // pto: %stage__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v28 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v26, v27);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v29 = v11; v29 < v8; v29 += v10) {
    // pto: %tile__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v0
    ;
    uint64_t v31 = (uint64_t) v7;
    TASSIGN(v30, v31);
    // pto: %0
    ;
    int64_t v32 = v29 < v11 ? v11 : v29;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v33 = 0;
    // pto: %inp__ssa_v0_pview
    ;
    __gm__ int8_t* v34 = PTOAS__GLOBAL_TENSOR_DATA(v20);
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v35 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v36 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v37 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v38 = v10 * v9;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v39 = v37 * v38;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v40 = pto::Shape<1, 1, 1, 1, -1>(v35, v36, v37, v10, v12);
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v41 = pto::Stride<-1, -1, -1, -1, -1>(v36 * v39, v39, v38, v9, v10);
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v42 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v34 + (v33 + v32 * v9 + v11 * v10), v40, v41);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v30, v42);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v43 = 0;
    // pto: %stage__iter_v1_pview
    ;
    __gm__ int8_t* v44 = PTOAS__GLOBAL_TENSOR_DATA(v28);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v45 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v46 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v47 = 1;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v48 = v10 * v9;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v49 = v47 * v48;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v50 = pto::Shape<1, 1, 1, 1, -1>(v45, v46, v47, v10, v12);
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v51 = pto::Stride<-1, -1, -1, -1, -1>(v46 * v49, v49, v48, v9, v10);
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v52 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v44 + (v43 + v32 * v9 + v11 * v10), v50, v51);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v52, v30);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 4, 32768> v53 = pto::Shape<1, 1, 1, 4, 32768>();
    // pto: %2
    ;
    pto::Stride<131072, 131072, 131072, 32768, 1> v54 = pto::Stride<131072, 131072, 131072, 32768, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 32768>, pto::Stride<131072, 131072, 131072, 32768, 1>, pto::Layout::ND> v55 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 32768>, pto::Stride<131072, 131072, 131072, 32768, 1>, pto::Layout::ND>((__gm__ int8_t*) v3, v53, v54);
    __gm__ int8_t* v56 = PTOAS__GLOBAL_TENSOR_DATA(v55);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v56);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v1
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v1
    ;
    uint64_t v58 = (uint64_t) v7;
    TASSIGN(v57, v58);
    // pto: %4
    ;
    const int64_t v59 = 0;
    // pto: %4
    ;
    __gm__ int8_t* v60 = PTOAS__GLOBAL_TENSOR_DATA(v20);
    // pto: %4
    ;
    const int64_t v61 = 1;
    // pto: %4
    ;
    const int64_t v62 = 1;
    // pto: %4
    ;
    const int64_t v63 = 1;
    // pto: %4
    ;
    int64_t v64 = v10 * v9;
    // pto: %4
    ;
    int64_t v65 = v63 * v64;
    // pto: %4
    ;
    pto::Shape<1, 1, 1, 1, -1> v66 = pto::Shape<1, 1, 1, 1, -1>(v61, v62, v63, v10, v12);
    // pto: %4
    ;
    pto::Stride<-1, -1, -1, -1, -1> v67 = pto::Stride<-1, -1, -1, -1, -1>(v62 * v65, v65, v64, v9, v10);
    // pto: %4
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v68 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v60 + (v59 + v32 * v9 + v12 * v10), v66, v67);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v57, v68);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v69 = 0;
    // pto: %stage__ssa_v3_pview
    ;
    __gm__ int8_t* v70 = PTOAS__GLOBAL_TENSOR_DATA(v28);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v71 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v72 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v73 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v74 = v10 * v9;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v75 = v73 * v74;
    // pto: %stage__ssa_v3_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v76 = pto::Shape<1, 1, 1, 1, -1>(v71, v72, v73, v10, v12);
    // pto: %stage__ssa_v3_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v77 = pto::Stride<-1, -1, -1, -1, -1>(v72 * v75, v75, v74, v9, v10);
    // pto: %stage__ssa_v3_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v78 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v70 + (v69 + v32 * v9 + v12 * v10), v76, v77);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    TSTORE(v78, v57);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    __gm__ int8_t* v79 = PTOAS__GLOBAL_TENSOR_DATA(v55);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v79);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  for (int64_t v80 = v11; v80 < v8; v80 += v10) {
    // pto: %v__tile
    ;
    int32_t v81 = (v2)[v80];
    (v4)[v80] = v81;
    // pto: %counts__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 4, 1> v82 = pto::Shape<1, 1, 1, 4, 1>();
    // pto: %counts__ssa_v0_pview
    ;
    pto::Stride<4, 4, 4, 1, 4> v83 = pto::Stride<4, 4, 4, 1, 4>();
    // pto: %counts__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN> v84 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN>((__gm__ int32_t*) v4, v82, v83);
    __gm__ int32_t* v85 = PTOAS__GLOBAL_TENSOR_DATA(v84);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v85);
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