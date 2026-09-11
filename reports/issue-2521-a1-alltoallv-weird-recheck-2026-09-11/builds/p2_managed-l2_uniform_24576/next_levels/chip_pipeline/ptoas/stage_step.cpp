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
  // pto: %c24576_index
  const int64_t v9 = 24576;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c0_index
  const int64_t v11 = 0;
  // pto: %c16384_index
  const int64_t v12 = 16384;
  // pto: %c8192_index
  const int64_t v13 = 8192;
  // pto: %inp__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v15 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v16 = 1;
  // pto: %inp__ssa_v0_view
  int64_t v17 = v8 * v9;
  // pto: %inp__ssa_v0_view
  int64_t v18 = v16 * v17;
  // pto: %inp__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v19 = pto::Shape<1, 1, 1, -1, -1>(v14, v15, v16, v8, v9);
  // pto: %inp__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v20 = pto::Stride<-1, -1, -1, -1, -1>(v15 * v18, v18, v17, v9, v10);
  // pto: %inp__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v21 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v19, v20);
  // pto: %stage__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %stage__ssa_v0_view
  int64_t v25 = v8 * v9;
  // pto: %stage__ssa_v0_view
  int64_t v26 = v24 * v25;
  // pto: %stage__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v27 = pto::Shape<1, 1, 1, -1, -1>(v22, v23, v24, v8, v9);
  // pto: %stage__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v28 = pto::Stride<-1, -1, -1, -1, -1>(v23 * v26, v26, v25, v9, v10);
  // pto: %stage__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v29 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v27, v28);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v30 = v11; v30 < v8; v30 += v10) {
    // pto: %tile__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v31 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v0
    ;
    uint64_t v32 = (uint64_t) v7;
    TASSIGN(v31, v32);
    // pto: %0
    ;
    int64_t v33 = v30 < v11 ? v11 : v30;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v34 = 0;
    // pto: %inp__ssa_v0_pview
    ;
    __gm__ int8_t* v35 = PTOAS__GLOBAL_TENSOR_DATA(v21);
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v36 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v37 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v38 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v39 = v10 * v9;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v40 = v38 * v39;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v41 = pto::Shape<1, 1, 1, 1, -1>(v36, v37, v38, v10, v12);
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v42 = pto::Stride<-1, -1, -1, -1, -1>(v37 * v40, v40, v39, v9, v10);
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v43 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v35 + (v34 + v33 * v9 + v11 * v10), v41, v42);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v31, v43);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v44 = 0;
    // pto: %stage__iter_v1_pview
    ;
    __gm__ int8_t* v45 = PTOAS__GLOBAL_TENSOR_DATA(v29);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v46 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v47 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v48 = 1;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v49 = v10 * v9;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v50 = v48 * v49;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v51 = pto::Shape<1, 1, 1, 1, -1>(v46, v47, v48, v10, v12);
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v52 = pto::Stride<-1, -1, -1, -1, -1>(v47 * v50, v50, v49, v9, v10);
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v53 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v45 + (v44 + v33 * v9 + v11 * v10), v51, v52);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v53, v31);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 2, 24576> v54 = pto::Shape<1, 1, 1, 2, 24576>();
    // pto: %2
    ;
    pto::Stride<49152, 49152, 49152, 24576, 1> v55 = pto::Stride<49152, 49152, 49152, 24576, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 2, 24576>, pto::Stride<49152, 49152, 49152, 24576, 1>, pto::Layout::ND> v56 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 2, 24576>, pto::Stride<49152, 49152, 49152, 24576, 1>, pto::Layout::ND>((__gm__ int8_t*) v3, v54, v55);
    __gm__ int8_t* v57 = PTOAS__GLOBAL_TENSOR_DATA(v56);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v57);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v1
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v58 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v13);
    // pto: %tile__ssa_v1
    ;
    uint64_t v59 = (uint64_t) v7;
    TASSIGN(v58, v59);
    // pto: %4
    ;
    const int64_t v60 = 0;
    // pto: %4
    ;
    __gm__ int8_t* v61 = PTOAS__GLOBAL_TENSOR_DATA(v21);
    // pto: %4
    ;
    const int64_t v62 = 1;
    // pto: %4
    ;
    const int64_t v63 = 1;
    // pto: %4
    ;
    const int64_t v64 = 1;
    // pto: %4
    ;
    int64_t v65 = v10 * v9;
    // pto: %4
    ;
    int64_t v66 = v64 * v65;
    // pto: %4
    ;
    pto::Shape<1, 1, 1, 1, -1> v67 = pto::Shape<1, 1, 1, 1, -1>(v62, v63, v64, v10, v13);
    // pto: %4
    ;
    pto::Stride<-1, -1, -1, -1, -1> v68 = pto::Stride<-1, -1, -1, -1, -1>(v63 * v66, v66, v65, v9, v10);
    // pto: %4
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v69 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v61 + (v60 + v33 * v9 + v12 * v10), v67, v68);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v58, v69);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v70 = 0;
    // pto: %stage__ssa_v3_pview
    ;
    __gm__ int8_t* v71 = PTOAS__GLOBAL_TENSOR_DATA(v29);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v72 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v73 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v74 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v75 = v10 * v9;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v76 = v74 * v75;
    // pto: %stage__ssa_v3_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v77 = pto::Shape<1, 1, 1, 1, -1>(v72, v73, v74, v10, v13);
    // pto: %stage__ssa_v3_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v78 = pto::Stride<-1, -1, -1, -1, -1>(v73 * v76, v76, v75, v9, v10);
    // pto: %stage__ssa_v3_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v79 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v71 + (v70 + v33 * v9 + v12 * v10), v77, v78);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    TSTORE(v79, v58);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    __gm__ int8_t* v80 = PTOAS__GLOBAL_TENSOR_DATA(v56);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v80);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  for (int64_t v81 = v11; v81 < v8; v81 += v10) {
    // pto: %v__tile
    ;
    int32_t v82 = (v2)[v81];
    (v4)[v81] = v82;
    // pto: %counts__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 2, 1> v83 = pto::Shape<1, 1, 1, 2, 1>();
    // pto: %counts__ssa_v0_pview
    ;
    pto::Stride<2, 2, 2, 1, 2> v84 = pto::Stride<2, 2, 2, 1, 2>();
    // pto: %counts__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 2, 1>, pto::Stride<2, 2, 2, 1, 2>, pto::Layout::DN> v85 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 2, 1>, pto::Stride<2, 2, 2, 1, 2>, pto::Layout::DN>((__gm__ int32_t*) v4, v83, v84);
    __gm__ int32_t* v86 = PTOAS__GLOBAL_TENSOR_DATA(v85);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v86);
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