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
  // pto: %c65536_index
  const int64_t v6 = 65536;
  // pto: %c1_index
  const int64_t v7 = 1;
  // pto: %c0_index
  const int64_t v8 = 0;
  // pto: %c16384_index
  const int64_t v9 = 16384;
  // pto: %c32768_index
  const int64_t v10 = 32768;
  // pto: %c49152_index
  const int64_t v11 = 49152;
  // pto: %inp__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v13 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %inp__ssa_v0_view
  int64_t v15 = v5 * v6;
  // pto: %inp__ssa_v0_view
  int64_t v16 = v14 * v15;
  // pto: %inp__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v5, v6);
  // pto: %inp__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v16, v16, v15, v6, v7);
  // pto: %inp__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v19 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v17, v18);
  // pto: %stage__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %stage__ssa_v0_view
  int64_t v23 = v5 * v6;
  // pto: %stage__ssa_v0_view
  int64_t v24 = v22 * v23;
  // pto: %stage__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v25 = pto::Shape<1, 1, 1, -1, -1>(v20, v21, v22, v5, v6);
  // pto: %stage__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v26 = pto::Stride<-1, -1, -1, -1, -1>(v21 * v24, v24, v23, v6, v7);
  // pto: %stage__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v27 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v25, v26);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v28 = v8; v28 < v5; v28 += v7) {
    // pto: %tile__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v29 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v0
    ;
    uint64_t v30 = (uint64_t) v4;
    TASSIGN(v29, v30);
    // pto: %0
    ;
    int64_t v31 = v28 < v8 ? v8 : v28;
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
    int64_t v37 = v7 * v6;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v38 = v36 * v37;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v39 = pto::Shape<1, 1, 1, 1, -1>(v34, v35, v36, v7, v9);
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v40 = pto::Stride<-1, -1, -1, -1, -1>(v35 * v38, v38, v37, v6, v7);
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v41 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v33 + (v32 + v31 * v6 + v8 * v7), v39, v40);
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
    int64_t v47 = v7 * v6;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v48 = v46 * v47;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v49 = pto::Shape<1, 1, 1, 1, -1>(v44, v45, v46, v7, v9);
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v50 = pto::Stride<-1, -1, -1, -1, -1>(v45 * v48, v48, v47, v6, v7);
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v51 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v43 + (v42 + v31 * v6 + v8 * v7), v49, v50);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v51, v29);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 4, 65536> v52 = pto::Shape<1, 1, 1, 4, 65536>();
    // pto: %2
    ;
    pto::Stride<262144, 262144, 262144, 65536, 1> v53 = pto::Stride<262144, 262144, 262144, 65536, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 65536>, pto::Stride<262144, 262144, 262144, 65536, 1>, pto::Layout::ND> v54 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 65536>, pto::Stride<262144, 262144, 262144, 65536, 1>, pto::Layout::ND>((__gm__ int8_t*) v2, v52, v53);
    __gm__ int8_t* v55 = PTOAS__GLOBAL_TENSOR_DATA(v54);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v55);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v1
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v56 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v1
    ;
    uint64_t v57 = (uint64_t) v4;
    TASSIGN(v56, v57);
    // pto: %4
    ;
    const int64_t v58 = 0;
    // pto: %4
    ;
    __gm__ int8_t* v59 = PTOAS__GLOBAL_TENSOR_DATA(v19);
    // pto: %4
    ;
    const int64_t v60 = 1;
    // pto: %4
    ;
    const int64_t v61 = 1;
    // pto: %4
    ;
    const int64_t v62 = 1;
    // pto: %4
    ;
    int64_t v63 = v7 * v6;
    // pto: %4
    ;
    int64_t v64 = v62 * v63;
    // pto: %4
    ;
    pto::Shape<1, 1, 1, 1, -1> v65 = pto::Shape<1, 1, 1, 1, -1>(v60, v61, v62, v7, v9);
    // pto: %4
    ;
    pto::Stride<-1, -1, -1, -1, -1> v66 = pto::Stride<-1, -1, -1, -1, -1>(v61 * v64, v64, v63, v6, v7);
    // pto: %4
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v67 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v59 + (v58 + v31 * v6 + v9 * v7), v65, v66);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v56, v67);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v68 = 0;
    // pto: %stage__ssa_v3_pview
    ;
    __gm__ int8_t* v69 = PTOAS__GLOBAL_TENSOR_DATA(v27);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v70 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v71 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v72 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v73 = v7 * v6;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v74 = v72 * v73;
    // pto: %stage__ssa_v3_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v75 = pto::Shape<1, 1, 1, 1, -1>(v70, v71, v72, v7, v9);
    // pto: %stage__ssa_v3_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v76 = pto::Stride<-1, -1, -1, -1, -1>(v71 * v74, v74, v73, v6, v7);
    // pto: %stage__ssa_v3_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v77 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v69 + (v68 + v31 * v6 + v9 * v7), v75, v76);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    TSTORE(v77, v56);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    __gm__ int8_t* v78 = PTOAS__GLOBAL_TENSOR_DATA(v54);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v78);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v2
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v2
    ;
    uint64_t v80 = (uint64_t) v4;
    TASSIGN(v79, v80);
    // pto: %8
    ;
    const int64_t v81 = 0;
    // pto: %8
    ;
    __gm__ int8_t* v82 = PTOAS__GLOBAL_TENSOR_DATA(v19);
    // pto: %8
    ;
    const int64_t v83 = 1;
    // pto: %8
    ;
    const int64_t v84 = 1;
    // pto: %8
    ;
    const int64_t v85 = 1;
    // pto: %8
    ;
    int64_t v86 = v7 * v6;
    // pto: %8
    ;
    int64_t v87 = v85 * v86;
    // pto: %8
    ;
    pto::Shape<1, 1, 1, 1, -1> v88 = pto::Shape<1, 1, 1, 1, -1>(v83, v84, v85, v7, v9);
    // pto: %8
    ;
    pto::Stride<-1, -1, -1, -1, -1> v89 = pto::Stride<-1, -1, -1, -1, -1>(v84 * v87, v87, v86, v6, v7);
    // pto: %8
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v90 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v82 + (v81 + v31 * v6 + v10 * v7), v88, v89);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v79, v90);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v91 = 0;
    // pto: %stage__ssa_v4_pview
    ;
    __gm__ int8_t* v92 = PTOAS__GLOBAL_TENSOR_DATA(v27);
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v93 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v94 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v95 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    int64_t v96 = v7 * v6;
    // pto: %stage__ssa_v4_pview
    ;
    int64_t v97 = v95 * v96;
    // pto: %stage__ssa_v4_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v98 = pto::Shape<1, 1, 1, 1, -1>(v93, v94, v95, v7, v9);
    // pto: %stage__ssa_v4_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v99 = pto::Stride<-1, -1, -1, -1, -1>(v94 * v97, v97, v96, v6, v7);
    // pto: %stage__ssa_v4_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v100 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v92 + (v91 + v31 * v6 + v10 * v7), v98, v99);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    TSTORE(v100, v79);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    __gm__ int8_t* v101 = PTOAS__GLOBAL_TENSOR_DATA(v54);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v101);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v3
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v3
    ;
    uint64_t v103 = (uint64_t) v4;
    TASSIGN(v102, v103);
    // pto: %12
    ;
    const int64_t v104 = 0;
    // pto: %12
    ;
    __gm__ int8_t* v105 = PTOAS__GLOBAL_TENSOR_DATA(v19);
    // pto: %12
    ;
    const int64_t v106 = 1;
    // pto: %12
    ;
    const int64_t v107 = 1;
    // pto: %12
    ;
    const int64_t v108 = 1;
    // pto: %12
    ;
    int64_t v109 = v7 * v6;
    // pto: %12
    ;
    int64_t v110 = v108 * v109;
    // pto: %12
    ;
    pto::Shape<1, 1, 1, 1, -1> v111 = pto::Shape<1, 1, 1, 1, -1>(v106, v107, v108, v7, v9);
    // pto: %12
    ;
    pto::Stride<-1, -1, -1, -1, -1> v112 = pto::Stride<-1, -1, -1, -1, -1>(v107 * v110, v110, v109, v6, v7);
    // pto: %12
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v113 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v105 + (v104 + v31 * v6 + v11 * v7), v111, v112);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    TLOAD(v102, v113);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v114 = 0;
    // pto: %stage__ssa_v5_pview
    ;
    __gm__ int8_t* v115 = PTOAS__GLOBAL_TENSOR_DATA(v27);
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v116 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v117 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v118 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    int64_t v119 = v7 * v6;
    // pto: %stage__ssa_v5_pview
    ;
    int64_t v120 = v118 * v119;
    // pto: %stage__ssa_v5_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v121 = pto::Shape<1, 1, 1, 1, -1>(v116, v117, v118, v7, v9);
    // pto: %stage__ssa_v5_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v122 = pto::Stride<-1, -1, -1, -1, -1>(v117 * v120, v120, v119, v6, v7);
    // pto: %stage__ssa_v5_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v123 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v115 + (v114 + v31 * v6 + v11 * v7), v121, v122);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
    TSTORE(v123, v102);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    __gm__ int8_t* v124 = PTOAS__GLOBAL_TENSOR_DATA(v54);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v124);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}