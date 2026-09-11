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
  // pto: %c24576_index
  const int64_t v6 = 24576;
  // pto: %c1_index
  const int64_t v7 = 1;
  // pto: %c0_index
  const int64_t v8 = 0;
  // pto: %c16384_index
  const int64_t v9 = 16384;
  // pto: %c8192_index
  const int64_t v10 = 8192;
  // pto: %inp__ssa_v0_view
  const int64_t v11 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v13 = 1;
  // pto: %inp__ssa_v0_view
  int64_t v14 = v5 * v6;
  // pto: %inp__ssa_v0_view
  int64_t v15 = v13 * v14;
  // pto: %inp__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v16 = pto::Shape<1, 1, 1, -1, -1>(v11, v12, v13, v5, v6);
  // pto: %inp__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v17 = pto::Stride<-1, -1, -1, -1, -1>(v12 * v15, v15, v14, v6, v7);
  // pto: %inp__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v18 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v16, v17);
  // pto: %stage__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %stage__ssa_v0_view
  int64_t v22 = v5 * v6;
  // pto: %stage__ssa_v0_view
  int64_t v23 = v21 * v22;
  // pto: %stage__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v24 = pto::Shape<1, 1, 1, -1, -1>(v19, v20, v21, v5, v6);
  // pto: %stage__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v25 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v23, v23, v22, v6, v7);
  // pto: %stage__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v26 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v24, v25);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v27 = v8; v27 < v5; v27 += v7) {
    // pto: %tile__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v28 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v0
    ;
    uint64_t v29 = (uint64_t) v4;
    TASSIGN(v28, v29);
    // pto: %0
    ;
    int64_t v30 = v27 < v8 ? v8 : v27;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v31 = 0;
    // pto: %inp__ssa_v0_pview
    ;
    __gm__ int8_t* v32 = PTOAS__GLOBAL_TENSOR_DATA(v18);
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v33 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v34 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v35 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v36 = v7 * v6;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v37 = v35 * v36;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v38 = pto::Shape<1, 1, 1, 1, -1>(v33, v34, v35, v7, v9);
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v39 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v37, v37, v36, v6, v7);
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v40 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v32 + (v31 + v30 * v6 + v8 * v7), v38, v39);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v28, v40);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v41 = 0;
    // pto: %stage__iter_v1_pview
    ;
    __gm__ int8_t* v42 = PTOAS__GLOBAL_TENSOR_DATA(v26);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v43 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v44 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v45 = 1;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v46 = v7 * v6;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v47 = v45 * v46;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v48 = pto::Shape<1, 1, 1, 1, -1>(v43, v44, v45, v7, v9);
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v49 = pto::Stride<-1, -1, -1, -1, -1>(v44 * v47, v47, v46, v6, v7);
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v50 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v42 + (v41 + v30 * v6 + v8 * v7), v48, v49);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v50, v28);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 2, 24576> v51 = pto::Shape<1, 1, 1, 2, 24576>();
    // pto: %2
    ;
    pto::Stride<49152, 49152, 49152, 24576, 1> v52 = pto::Stride<49152, 49152, 49152, 24576, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 2, 24576>, pto::Stride<49152, 49152, 49152, 24576, 1>, pto::Layout::ND> v53 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 2, 24576>, pto::Stride<49152, 49152, 49152, 24576, 1>, pto::Layout::ND>((__gm__ int8_t*) v2, v51, v52);
    __gm__ int8_t* v54 = PTOAS__GLOBAL_TENSOR_DATA(v53);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v54);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v1
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v55 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v10);
    // pto: %tile__ssa_v1
    ;
    uint64_t v56 = (uint64_t) v4;
    TASSIGN(v55, v56);
    // pto: %4
    ;
    const int64_t v57 = 0;
    // pto: %4
    ;
    __gm__ int8_t* v58 = PTOAS__GLOBAL_TENSOR_DATA(v18);
    // pto: %4
    ;
    const int64_t v59 = 1;
    // pto: %4
    ;
    const int64_t v60 = 1;
    // pto: %4
    ;
    const int64_t v61 = 1;
    // pto: %4
    ;
    int64_t v62 = v7 * v6;
    // pto: %4
    ;
    int64_t v63 = v61 * v62;
    // pto: %4
    ;
    pto::Shape<1, 1, 1, 1, -1> v64 = pto::Shape<1, 1, 1, 1, -1>(v59, v60, v61, v7, v10);
    // pto: %4
    ;
    pto::Stride<-1, -1, -1, -1, -1> v65 = pto::Stride<-1, -1, -1, -1, -1>(v60 * v63, v63, v62, v6, v7);
    // pto: %4
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v66 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v58 + (v57 + v30 * v6 + v9 * v7), v64, v65);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v55, v66);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v67 = 0;
    // pto: %stage__ssa_v3_pview
    ;
    __gm__ int8_t* v68 = PTOAS__GLOBAL_TENSOR_DATA(v26);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v69 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v70 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v71 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v72 = v7 * v6;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v73 = v71 * v72;
    // pto: %stage__ssa_v3_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v74 = pto::Shape<1, 1, 1, 1, -1>(v69, v70, v71, v7, v10);
    // pto: %stage__ssa_v3_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v75 = pto::Stride<-1, -1, -1, -1, -1>(v70 * v73, v73, v72, v6, v7);
    // pto: %stage__ssa_v3_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v76 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v68 + (v67 + v30 * v6 + v9 * v7), v74, v75);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    TSTORE(v76, v55);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    __gm__ int8_t* v77 = PTOAS__GLOBAL_TENSOR_DATA(v53);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v77);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}