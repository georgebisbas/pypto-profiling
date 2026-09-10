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
  // pto: %c65536_index
  const int64_t v9 = 65536;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c0_index
  const int64_t v11 = 0;
  // pto: %c16384_index
  const int64_t v12 = 16384;
  // pto: %c32768_index
  const int64_t v13 = 32768;
  // pto: %c49152_index
  const int64_t v14 = 49152;
  // pto: %inp__ssa_v0_view
  const int64_t v15 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v16 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v17 = 1;
  // pto: %inp__ssa_v0_view
  int64_t v18 = v8 * v9;
  // pto: %inp__ssa_v0_view
  int64_t v19 = v17 * v18;
  // pto: %inp__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v20 = pto::Shape<1, 1, 1, -1, -1>(v15, v16, v17, v8, v9);
  // pto: %inp__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v21 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v19, v19, v18, v9, v10);
  // pto: %inp__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v22 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v20, v21);
  // pto: %stage__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %stage__ssa_v0_view
  int64_t v26 = v8 * v9;
  // pto: %stage__ssa_v0_view
  int64_t v27 = v25 * v26;
  // pto: %stage__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v28 = pto::Shape<1, 1, 1, -1, -1>(v23, v24, v25, v8, v9);
  // pto: %stage__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v29 = pto::Stride<-1, -1, -1, -1, -1>(v24 * v27, v27, v26, v9, v10);
  // pto: %stage__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v30 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v28, v29);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v31 = v11; v31 < v8; v31 += v10) {
    // pto: %tile__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v32 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v0
    ;
    uint64_t v33 = (uint64_t) v7;
    TASSIGN(v32, v33);
    // pto: %0
    ;
    int64_t v34 = v31 < v11 ? v11 : v31;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v35 = 0;
    // pto: %inp__ssa_v0_pview
    ;
    __gm__ int8_t* v36 = PTOAS__GLOBAL_TENSOR_DATA(v22);
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v37 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v38 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v39 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v40 = v10 * v9;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v41 = v39 * v40;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v42 = pto::Shape<1, 1, 1, 1, -1>(v37, v38, v39, v10, v12);
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v43 = pto::Stride<-1, -1, -1, -1, -1>(v38 * v41, v41, v40, v9, v10);
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v44 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v36 + (v35 + v34 * v9 + v11 * v10), v42, v43);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v32, v44);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v45 = 0;
    // pto: %stage__iter_v1_pview
    ;
    __gm__ int8_t* v46 = PTOAS__GLOBAL_TENSOR_DATA(v30);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v47 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v48 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v49 = 1;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v50 = v10 * v9;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v51 = v49 * v50;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v52 = pto::Shape<1, 1, 1, 1, -1>(v47, v48, v49, v10, v12);
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v53 = pto::Stride<-1, -1, -1, -1, -1>(v48 * v51, v51, v50, v9, v10);
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v54 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v46 + (v45 + v34 * v9 + v11 * v10), v52, v53);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v54, v32);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 4, 65536> v55 = pto::Shape<1, 1, 1, 4, 65536>();
    // pto: %2
    ;
    pto::Stride<262144, 262144, 262144, 65536, 1> v56 = pto::Stride<262144, 262144, 262144, 65536, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 65536>, pto::Stride<262144, 262144, 262144, 65536, 1>, pto::Layout::ND> v57 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 65536>, pto::Stride<262144, 262144, 262144, 65536, 1>, pto::Layout::ND>((__gm__ int8_t*) v3, v55, v56);
    __gm__ int8_t* v58 = PTOAS__GLOBAL_TENSOR_DATA(v57);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v58);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v1
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v1
    ;
    uint64_t v60 = (uint64_t) v7;
    TASSIGN(v59, v60);
    // pto: %4
    ;
    const int64_t v61 = 0;
    // pto: %4
    ;
    __gm__ int8_t* v62 = PTOAS__GLOBAL_TENSOR_DATA(v22);
    // pto: %4
    ;
    const int64_t v63 = 1;
    // pto: %4
    ;
    const int64_t v64 = 1;
    // pto: %4
    ;
    const int64_t v65 = 1;
    // pto: %4
    ;
    int64_t v66 = v10 * v9;
    // pto: %4
    ;
    int64_t v67 = v65 * v66;
    // pto: %4
    ;
    pto::Shape<1, 1, 1, 1, -1> v68 = pto::Shape<1, 1, 1, 1, -1>(v63, v64, v65, v10, v12);
    // pto: %4
    ;
    pto::Stride<-1, -1, -1, -1, -1> v69 = pto::Stride<-1, -1, -1, -1, -1>(v64 * v67, v67, v66, v9, v10);
    // pto: %4
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v70 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v62 + (v61 + v34 * v9 + v12 * v10), v68, v69);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v59, v70);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v71 = 0;
    // pto: %stage__ssa_v3_pview
    ;
    __gm__ int8_t* v72 = PTOAS__GLOBAL_TENSOR_DATA(v30);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v73 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v74 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v75 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v76 = v10 * v9;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v77 = v75 * v76;
    // pto: %stage__ssa_v3_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v78 = pto::Shape<1, 1, 1, 1, -1>(v73, v74, v75, v10, v12);
    // pto: %stage__ssa_v3_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v79 = pto::Stride<-1, -1, -1, -1, -1>(v74 * v77, v77, v76, v9, v10);
    // pto: %stage__ssa_v3_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v80 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v72 + (v71 + v34 * v9 + v12 * v10), v78, v79);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    TSTORE(v80, v59);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    __gm__ int8_t* v81 = PTOAS__GLOBAL_TENSOR_DATA(v57);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v81);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v2
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v2
    ;
    uint64_t v83 = (uint64_t) v7;
    TASSIGN(v82, v83);
    // pto: %8
    ;
    const int64_t v84 = 0;
    // pto: %8
    ;
    __gm__ int8_t* v85 = PTOAS__GLOBAL_TENSOR_DATA(v22);
    // pto: %8
    ;
    const int64_t v86 = 1;
    // pto: %8
    ;
    const int64_t v87 = 1;
    // pto: %8
    ;
    const int64_t v88 = 1;
    // pto: %8
    ;
    int64_t v89 = v10 * v9;
    // pto: %8
    ;
    int64_t v90 = v88 * v89;
    // pto: %8
    ;
    pto::Shape<1, 1, 1, 1, -1> v91 = pto::Shape<1, 1, 1, 1, -1>(v86, v87, v88, v10, v12);
    // pto: %8
    ;
    pto::Stride<-1, -1, -1, -1, -1> v92 = pto::Stride<-1, -1, -1, -1, -1>(v87 * v90, v90, v89, v9, v10);
    // pto: %8
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v93 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v85 + (v84 + v34 * v9 + v13 * v10), v91, v92);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v82, v93);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v94 = 0;
    // pto: %stage__ssa_v4_pview
    ;
    __gm__ int8_t* v95 = PTOAS__GLOBAL_TENSOR_DATA(v30);
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v96 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v97 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v98 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    int64_t v99 = v10 * v9;
    // pto: %stage__ssa_v4_pview
    ;
    int64_t v100 = v98 * v99;
    // pto: %stage__ssa_v4_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v101 = pto::Shape<1, 1, 1, 1, -1>(v96, v97, v98, v10, v12);
    // pto: %stage__ssa_v4_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v102 = pto::Stride<-1, -1, -1, -1, -1>(v97 * v100, v100, v99, v9, v10);
    // pto: %stage__ssa_v4_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v103 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v95 + (v94 + v34 * v9 + v13 * v10), v101, v102);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    TSTORE(v103, v82);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    __gm__ int8_t* v104 = PTOAS__GLOBAL_TENSOR_DATA(v57);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v104);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v3
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v3
    ;
    uint64_t v106 = (uint64_t) v7;
    TASSIGN(v105, v106);
    // pto: %12
    ;
    const int64_t v107 = 0;
    // pto: %12
    ;
    __gm__ int8_t* v108 = PTOAS__GLOBAL_TENSOR_DATA(v22);
    // pto: %12
    ;
    const int64_t v109 = 1;
    // pto: %12
    ;
    const int64_t v110 = 1;
    // pto: %12
    ;
    const int64_t v111 = 1;
    // pto: %12
    ;
    int64_t v112 = v10 * v9;
    // pto: %12
    ;
    int64_t v113 = v111 * v112;
    // pto: %12
    ;
    pto::Shape<1, 1, 1, 1, -1> v114 = pto::Shape<1, 1, 1, 1, -1>(v109, v110, v111, v10, v12);
    // pto: %12
    ;
    pto::Stride<-1, -1, -1, -1, -1> v115 = pto::Stride<-1, -1, -1, -1, -1>(v110 * v113, v113, v112, v9, v10);
    // pto: %12
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v116 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v108 + (v107 + v34 * v9 + v14 * v10), v114, v115);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    TLOAD(v105, v116);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v117 = 0;
    // pto: %stage__ssa_v5_pview
    ;
    __gm__ int8_t* v118 = PTOAS__GLOBAL_TENSOR_DATA(v30);
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v119 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v120 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v121 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    int64_t v122 = v10 * v9;
    // pto: %stage__ssa_v5_pview
    ;
    int64_t v123 = v121 * v122;
    // pto: %stage__ssa_v5_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v124 = pto::Shape<1, 1, 1, 1, -1>(v119, v120, v121, v10, v12);
    // pto: %stage__ssa_v5_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v125 = pto::Stride<-1, -1, -1, -1, -1>(v120 * v123, v123, v122, v9, v10);
    // pto: %stage__ssa_v5_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v126 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v118 + (v117 + v34 * v9 + v14 * v10), v124, v125);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
    TSTORE(v126, v105);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    __gm__ int8_t* v127 = PTOAS__GLOBAL_TENSOR_DATA(v57);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v127);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  for (int64_t v128 = v11; v128 < v8; v128 += v10) {
    // pto: %v__tile
    ;
    int32_t v129 = (v2)[v128];
    (v4)[v128] = v129;
    // pto: %counts__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 4, 1> v130 = pto::Shape<1, 1, 1, 4, 1>();
    // pto: %counts__ssa_v0_pview
    ;
    pto::Stride<4, 4, 4, 1, 4> v131 = pto::Stride<4, 4, 4, 1, 4>();
    // pto: %counts__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN> v132 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN>((__gm__ int32_t*) v4, v130, v131);
    __gm__ int32_t* v133 = PTOAS__GLOBAL_TENSOR_DATA(v132);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v133);
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