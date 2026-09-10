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
  // pto: %c262144_index
  const int64_t v9 = 262144;
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
  // pto: %c65536_index
  const int64_t v15 = 65536;
  // pto: %c81920_index
  const int64_t v16 = 81920;
  // pto: %c98304_index
  const int64_t v17 = 98304;
  // pto: %c114688_index
  const int64_t v18 = 114688;
  // pto: %c131072_index
  const int64_t v19 = 131072;
  // pto: %c147456_index
  const int64_t v20 = 147456;
  // pto: %c163840_index
  const int64_t v21 = 163840;
  // pto: %c180224_index
  const int64_t v22 = 180224;
  // pto: %c196608_index
  const int64_t v23 = 196608;
  // pto: %c212992_index
  const int64_t v24 = 212992;
  // pto: %c229376_index
  const int64_t v25 = 229376;
  // pto: %c245760_index
  const int64_t v26 = 245760;
  // pto: %inp__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v28 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %inp__ssa_v0_view
  int64_t v30 = v8 * v9;
  // pto: %inp__ssa_v0_view
  int64_t v31 = v29 * v30;
  // pto: %inp__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v32 = pto::Shape<1, 1, 1, -1, -1>(v27, v28, v29, v8, v9);
  // pto: %inp__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v33 = pto::Stride<-1, -1, -1, -1, -1>(v28 * v31, v31, v30, v9, v10);
  // pto: %inp__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v34 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v32, v33);
  // pto: %stage__ssa_v0_view
  const int64_t v35 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v36 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v37 = 1;
  // pto: %stage__ssa_v0_view
  int64_t v38 = v8 * v9;
  // pto: %stage__ssa_v0_view
  int64_t v39 = v37 * v38;
  // pto: %stage__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v40 = pto::Shape<1, 1, 1, -1, -1>(v35, v36, v37, v8, v9);
  // pto: %stage__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v41 = pto::Stride<-1, -1, -1, -1, -1>(v36 * v39, v39, v38, v9, v10);
  // pto: %stage__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v42 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v40, v41);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v43 = v11; v43 < v8; v43 += v10) {
    // pto: %tile__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v44 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v0
    ;
    uint64_t v45 = (uint64_t) v7;
    TASSIGN(v44, v45);
    // pto: %0
    ;
    int64_t v46 = v43 < v11 ? v11 : v43;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v47 = 0;
    // pto: %inp__ssa_v0_pview
    ;
    __gm__ int8_t* v48 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v49 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v50 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v51 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v52 = v10 * v9;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v53 = v51 * v52;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v54 = pto::Shape<1, 1, 1, 1, -1>(v49, v50, v51, v10, v12);
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v55 = pto::Stride<-1, -1, -1, -1, -1>(v50 * v53, v53, v52, v9, v10);
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v56 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v48 + (v47 + v46 * v9 + v11 * v10), v54, v55);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v44, v56);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v57 = 0;
    // pto: %stage__iter_v1_pview
    ;
    __gm__ int8_t* v58 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v59 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v60 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v61 = 1;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v62 = v10 * v9;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v63 = v61 * v62;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v64 = pto::Shape<1, 1, 1, 1, -1>(v59, v60, v61, v10, v12);
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v65 = pto::Stride<-1, -1, -1, -1, -1>(v60 * v63, v63, v62, v9, v10);
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v66 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v58 + (v57 + v46 * v9 + v11 * v10), v64, v65);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v66, v44);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 4, 262144> v67 = pto::Shape<1, 1, 1, 4, 262144>();
    // pto: %2
    ;
    pto::Stride<1048576, 1048576, 1048576, 262144, 1> v68 = pto::Stride<1048576, 1048576, 1048576, 262144, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 262144>, pto::Stride<1048576, 1048576, 1048576, 262144, 1>, pto::Layout::ND> v69 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 262144>, pto::Stride<1048576, 1048576, 1048576, 262144, 1>, pto::Layout::ND>((__gm__ int8_t*) v3, v67, v68);
    __gm__ int8_t* v70 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v70);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v1
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v71 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v1
    ;
    uint64_t v72 = (uint64_t) v7;
    TASSIGN(v71, v72);
    // pto: %4
    ;
    const int64_t v73 = 0;
    // pto: %4
    ;
    __gm__ int8_t* v74 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %4
    ;
    const int64_t v75 = 1;
    // pto: %4
    ;
    const int64_t v76 = 1;
    // pto: %4
    ;
    const int64_t v77 = 1;
    // pto: %4
    ;
    int64_t v78 = v10 * v9;
    // pto: %4
    ;
    int64_t v79 = v77 * v78;
    // pto: %4
    ;
    pto::Shape<1, 1, 1, 1, -1> v80 = pto::Shape<1, 1, 1, 1, -1>(v75, v76, v77, v10, v12);
    // pto: %4
    ;
    pto::Stride<-1, -1, -1, -1, -1> v81 = pto::Stride<-1, -1, -1, -1, -1>(v76 * v79, v79, v78, v9, v10);
    // pto: %4
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v82 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v74 + (v73 + v46 * v9 + v12 * v10), v80, v81);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v71, v82);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v83 = 0;
    // pto: %stage__ssa_v3_pview
    ;
    __gm__ int8_t* v84 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v85 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v86 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v87 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v88 = v10 * v9;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v89 = v87 * v88;
    // pto: %stage__ssa_v3_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v90 = pto::Shape<1, 1, 1, 1, -1>(v85, v86, v87, v10, v12);
    // pto: %stage__ssa_v3_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v91 = pto::Stride<-1, -1, -1, -1, -1>(v86 * v89, v89, v88, v9, v10);
    // pto: %stage__ssa_v3_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v92 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v84 + (v83 + v46 * v9 + v12 * v10), v90, v91);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    TSTORE(v92, v71);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    __gm__ int8_t* v93 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v93);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v2
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v2
    ;
    uint64_t v95 = (uint64_t) v7;
    TASSIGN(v94, v95);
    // pto: %8
    ;
    const int64_t v96 = 0;
    // pto: %8
    ;
    __gm__ int8_t* v97 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %8
    ;
    const int64_t v98 = 1;
    // pto: %8
    ;
    const int64_t v99 = 1;
    // pto: %8
    ;
    const int64_t v100 = 1;
    // pto: %8
    ;
    int64_t v101 = v10 * v9;
    // pto: %8
    ;
    int64_t v102 = v100 * v101;
    // pto: %8
    ;
    pto::Shape<1, 1, 1, 1, -1> v103 = pto::Shape<1, 1, 1, 1, -1>(v98, v99, v100, v10, v12);
    // pto: %8
    ;
    pto::Stride<-1, -1, -1, -1, -1> v104 = pto::Stride<-1, -1, -1, -1, -1>(v99 * v102, v102, v101, v9, v10);
    // pto: %8
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v105 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v97 + (v96 + v46 * v9 + v13 * v10), v103, v104);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v94, v105);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v106 = 0;
    // pto: %stage__ssa_v4_pview
    ;
    __gm__ int8_t* v107 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v108 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v109 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v110 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    int64_t v111 = v10 * v9;
    // pto: %stage__ssa_v4_pview
    ;
    int64_t v112 = v110 * v111;
    // pto: %stage__ssa_v4_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v113 = pto::Shape<1, 1, 1, 1, -1>(v108, v109, v110, v10, v12);
    // pto: %stage__ssa_v4_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v114 = pto::Stride<-1, -1, -1, -1, -1>(v109 * v112, v112, v111, v9, v10);
    // pto: %stage__ssa_v4_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v115 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v107 + (v106 + v46 * v9 + v13 * v10), v113, v114);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    TSTORE(v115, v94);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    __gm__ int8_t* v116 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v116);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v3
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v3
    ;
    uint64_t v118 = (uint64_t) v7;
    TASSIGN(v117, v118);
    // pto: %12
    ;
    const int64_t v119 = 0;
    // pto: %12
    ;
    __gm__ int8_t* v120 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %12
    ;
    const int64_t v121 = 1;
    // pto: %12
    ;
    const int64_t v122 = 1;
    // pto: %12
    ;
    const int64_t v123 = 1;
    // pto: %12
    ;
    int64_t v124 = v10 * v9;
    // pto: %12
    ;
    int64_t v125 = v123 * v124;
    // pto: %12
    ;
    pto::Shape<1, 1, 1, 1, -1> v126 = pto::Shape<1, 1, 1, 1, -1>(v121, v122, v123, v10, v12);
    // pto: %12
    ;
    pto::Stride<-1, -1, -1, -1, -1> v127 = pto::Stride<-1, -1, -1, -1, -1>(v122 * v125, v125, v124, v9, v10);
    // pto: %12
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v128 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v120 + (v119 + v46 * v9 + v14 * v10), v126, v127);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    TLOAD(v117, v128);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v129 = 0;
    // pto: %stage__ssa_v5_pview
    ;
    __gm__ int8_t* v130 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v131 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v132 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v133 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    int64_t v134 = v10 * v9;
    // pto: %stage__ssa_v5_pview
    ;
    int64_t v135 = v133 * v134;
    // pto: %stage__ssa_v5_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v136 = pto::Shape<1, 1, 1, 1, -1>(v131, v132, v133, v10, v12);
    // pto: %stage__ssa_v5_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v137 = pto::Stride<-1, -1, -1, -1, -1>(v132 * v135, v135, v134, v9, v10);
    // pto: %stage__ssa_v5_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v138 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v130 + (v129 + v46 * v9 + v14 * v10), v136, v137);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
    TSTORE(v138, v117);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    __gm__ int8_t* v139 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v139);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v4
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v140 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v4
    ;
    uint64_t v141 = (uint64_t) v7;
    TASSIGN(v140, v141);
    // pto: %16
    ;
    const int64_t v142 = 0;
    // pto: %16
    ;
    __gm__ int8_t* v143 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %16
    ;
    const int64_t v144 = 1;
    // pto: %16
    ;
    const int64_t v145 = 1;
    // pto: %16
    ;
    const int64_t v146 = 1;
    // pto: %16
    ;
    int64_t v147 = v10 * v9;
    // pto: %16
    ;
    int64_t v148 = v146 * v147;
    // pto: %16
    ;
    pto::Shape<1, 1, 1, 1, -1> v149 = pto::Shape<1, 1, 1, 1, -1>(v144, v145, v146, v10, v12);
    // pto: %16
    ;
    pto::Stride<-1, -1, -1, -1, -1> v150 = pto::Stride<-1, -1, -1, -1, -1>(v145 * v148, v148, v147, v9, v10);
    // pto: %16
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v151 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v143 + (v142 + v46 * v9 + v15 * v10), v149, v150);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    TLOAD(v140, v151);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
    // pto: %stage__ssa_v6_pview
    ;
    const int64_t v152 = 0;
    // pto: %stage__ssa_v6_pview
    ;
    __gm__ int8_t* v153 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v6_pview
    ;
    const int64_t v154 = 1;
    // pto: %stage__ssa_v6_pview
    ;
    const int64_t v155 = 1;
    // pto: %stage__ssa_v6_pview
    ;
    const int64_t v156 = 1;
    // pto: %stage__ssa_v6_pview
    ;
    int64_t v157 = v10 * v9;
    // pto: %stage__ssa_v6_pview
    ;
    int64_t v158 = v156 * v157;
    // pto: %stage__ssa_v6_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v159 = pto::Shape<1, 1, 1, 1, -1>(v154, v155, v156, v10, v12);
    // pto: %stage__ssa_v6_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v160 = pto::Stride<-1, -1, -1, -1, -1>(v155 * v158, v158, v157, v9, v10);
    // pto: %stage__ssa_v6_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v161 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v153 + (v152 + v46 * v9 + v15 * v10), v159, v160);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
    TSTORE(v161, v140);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    __gm__ int8_t* v162 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v162);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v5
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v163 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v5
    ;
    uint64_t v164 = (uint64_t) v7;
    TASSIGN(v163, v164);
    // pto: %20
    ;
    const int64_t v165 = 0;
    // pto: %20
    ;
    __gm__ int8_t* v166 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %20
    ;
    const int64_t v167 = 1;
    // pto: %20
    ;
    const int64_t v168 = 1;
    // pto: %20
    ;
    const int64_t v169 = 1;
    // pto: %20
    ;
    int64_t v170 = v10 * v9;
    // pto: %20
    ;
    int64_t v171 = v169 * v170;
    // pto: %20
    ;
    pto::Shape<1, 1, 1, 1, -1> v172 = pto::Shape<1, 1, 1, 1, -1>(v167, v168, v169, v10, v12);
    // pto: %20
    ;
    pto::Stride<-1, -1, -1, -1, -1> v173 = pto::Stride<-1, -1, -1, -1, -1>(v168 * v171, v171, v170, v9, v10);
    // pto: %20
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v174 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v166 + (v165 + v46 * v9 + v16 * v10), v172, v173);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    TLOAD(v163, v174);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID5);
    // pto: %stage__ssa_v7_pview
    ;
    const int64_t v175 = 0;
    // pto: %stage__ssa_v7_pview
    ;
    __gm__ int8_t* v176 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v7_pview
    ;
    const int64_t v177 = 1;
    // pto: %stage__ssa_v7_pview
    ;
    const int64_t v178 = 1;
    // pto: %stage__ssa_v7_pview
    ;
    const int64_t v179 = 1;
    // pto: %stage__ssa_v7_pview
    ;
    int64_t v180 = v10 * v9;
    // pto: %stage__ssa_v7_pview
    ;
    int64_t v181 = v179 * v180;
    // pto: %stage__ssa_v7_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v182 = pto::Shape<1, 1, 1, 1, -1>(v177, v178, v179, v10, v12);
    // pto: %stage__ssa_v7_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v183 = pto::Stride<-1, -1, -1, -1, -1>(v178 * v181, v181, v180, v9, v10);
    // pto: %stage__ssa_v7_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v184 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v176 + (v175 + v46 * v9 + v16 * v10), v182, v183);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID5);
    TSTORE(v184, v163);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
    __gm__ int8_t* v185 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v185);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v6
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v186 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v6
    ;
    uint64_t v187 = (uint64_t) v7;
    TASSIGN(v186, v187);
    // pto: %24
    ;
    const int64_t v188 = 0;
    // pto: %24
    ;
    __gm__ int8_t* v189 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %24
    ;
    const int64_t v190 = 1;
    // pto: %24
    ;
    const int64_t v191 = 1;
    // pto: %24
    ;
    const int64_t v192 = 1;
    // pto: %24
    ;
    int64_t v193 = v10 * v9;
    // pto: %24
    ;
    int64_t v194 = v192 * v193;
    // pto: %24
    ;
    pto::Shape<1, 1, 1, 1, -1> v195 = pto::Shape<1, 1, 1, 1, -1>(v190, v191, v192, v10, v12);
    // pto: %24
    ;
    pto::Stride<-1, -1, -1, -1, -1> v196 = pto::Stride<-1, -1, -1, -1, -1>(v191 * v194, v194, v193, v9, v10);
    // pto: %24
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v197 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v189 + (v188 + v46 * v9 + v17 * v10), v195, v196);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
    TLOAD(v186, v197);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
    // pto: %stage__ssa_v8_pview
    ;
    const int64_t v198 = 0;
    // pto: %stage__ssa_v8_pview
    ;
    __gm__ int8_t* v199 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v8_pview
    ;
    const int64_t v200 = 1;
    // pto: %stage__ssa_v8_pview
    ;
    const int64_t v201 = 1;
    // pto: %stage__ssa_v8_pview
    ;
    const int64_t v202 = 1;
    // pto: %stage__ssa_v8_pview
    ;
    int64_t v203 = v10 * v9;
    // pto: %stage__ssa_v8_pview
    ;
    int64_t v204 = v202 * v203;
    // pto: %stage__ssa_v8_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v205 = pto::Shape<1, 1, 1, 1, -1>(v200, v201, v202, v10, v12);
    // pto: %stage__ssa_v8_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v206 = pto::Stride<-1, -1, -1, -1, -1>(v201 * v204, v204, v203, v9, v10);
    // pto: %stage__ssa_v8_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v207 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v199 + (v198 + v46 * v9 + v17 * v10), v205, v206);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
    TSTORE(v207, v186);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    __gm__ int8_t* v208 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v208);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v7
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v209 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v7
    ;
    uint64_t v210 = (uint64_t) v7;
    TASSIGN(v209, v210);
    // pto: %28
    ;
    const int64_t v211 = 0;
    // pto: %28
    ;
    __gm__ int8_t* v212 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %28
    ;
    const int64_t v213 = 1;
    // pto: %28
    ;
    const int64_t v214 = 1;
    // pto: %28
    ;
    const int64_t v215 = 1;
    // pto: %28
    ;
    int64_t v216 = v10 * v9;
    // pto: %28
    ;
    int64_t v217 = v215 * v216;
    // pto: %28
    ;
    pto::Shape<1, 1, 1, 1, -1> v218 = pto::Shape<1, 1, 1, 1, -1>(v213, v214, v215, v10, v12);
    // pto: %28
    ;
    pto::Stride<-1, -1, -1, -1, -1> v219 = pto::Stride<-1, -1, -1, -1, -1>(v214 * v217, v217, v216, v9, v10);
    // pto: %28
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v220 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v212 + (v211 + v46 * v9 + v18 * v10), v218, v219);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    TLOAD(v209, v220);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    // pto: %stage__ssa_v9_pview
    ;
    const int64_t v221 = 0;
    // pto: %stage__ssa_v9_pview
    ;
    __gm__ int8_t* v222 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v9_pview
    ;
    const int64_t v223 = 1;
    // pto: %stage__ssa_v9_pview
    ;
    const int64_t v224 = 1;
    // pto: %stage__ssa_v9_pview
    ;
    const int64_t v225 = 1;
    // pto: %stage__ssa_v9_pview
    ;
    int64_t v226 = v10 * v9;
    // pto: %stage__ssa_v9_pview
    ;
    int64_t v227 = v225 * v226;
    // pto: %stage__ssa_v9_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v228 = pto::Shape<1, 1, 1, 1, -1>(v223, v224, v225, v10, v12);
    // pto: %stage__ssa_v9_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v229 = pto::Stride<-1, -1, -1, -1, -1>(v224 * v227, v227, v226, v9, v10);
    // pto: %stage__ssa_v9_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v230 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v222 + (v221 + v46 * v9 + v18 * v10), v228, v229);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    TSTORE(v230, v209);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v231 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v231);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v8
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v8
    ;
    uint64_t v233 = (uint64_t) v7;
    TASSIGN(v232, v233);
    // pto: %32
    ;
    const int64_t v234 = 0;
    // pto: %32
    ;
    __gm__ int8_t* v235 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %32
    ;
    const int64_t v236 = 1;
    // pto: %32
    ;
    const int64_t v237 = 1;
    // pto: %32
    ;
    const int64_t v238 = 1;
    // pto: %32
    ;
    int64_t v239 = v10 * v9;
    // pto: %32
    ;
    int64_t v240 = v238 * v239;
    // pto: %32
    ;
    pto::Shape<1, 1, 1, 1, -1> v241 = pto::Shape<1, 1, 1, 1, -1>(v236, v237, v238, v10, v12);
    // pto: %32
    ;
    pto::Stride<-1, -1, -1, -1, -1> v242 = pto::Stride<-1, -1, -1, -1, -1>(v237 * v240, v240, v239, v9, v10);
    // pto: %32
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v243 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v235 + (v234 + v46 * v9 + v19 * v10), v241, v242);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v232, v243);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v10_pview
    ;
    const int64_t v244 = 0;
    // pto: %stage__ssa_v10_pview
    ;
    __gm__ int8_t* v245 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v10_pview
    ;
    const int64_t v246 = 1;
    // pto: %stage__ssa_v10_pview
    ;
    const int64_t v247 = 1;
    // pto: %stage__ssa_v10_pview
    ;
    const int64_t v248 = 1;
    // pto: %stage__ssa_v10_pview
    ;
    int64_t v249 = v10 * v9;
    // pto: %stage__ssa_v10_pview
    ;
    int64_t v250 = v248 * v249;
    // pto: %stage__ssa_v10_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v251 = pto::Shape<1, 1, 1, 1, -1>(v246, v247, v248, v10, v12);
    // pto: %stage__ssa_v10_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v252 = pto::Stride<-1, -1, -1, -1, -1>(v247 * v250, v250, v249, v9, v10);
    // pto: %stage__ssa_v10_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v253 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v245 + (v244 + v46 * v9 + v19 * v10), v251, v252);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v253, v232);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v254 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v254);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v9
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v255 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v9
    ;
    uint64_t v256 = (uint64_t) v7;
    TASSIGN(v255, v256);
    // pto: %36
    ;
    const int64_t v257 = 0;
    // pto: %36
    ;
    __gm__ int8_t* v258 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %36
    ;
    const int64_t v259 = 1;
    // pto: %36
    ;
    const int64_t v260 = 1;
    // pto: %36
    ;
    const int64_t v261 = 1;
    // pto: %36
    ;
    int64_t v262 = v10 * v9;
    // pto: %36
    ;
    int64_t v263 = v261 * v262;
    // pto: %36
    ;
    pto::Shape<1, 1, 1, 1, -1> v264 = pto::Shape<1, 1, 1, 1, -1>(v259, v260, v261, v10, v12);
    // pto: %36
    ;
    pto::Stride<-1, -1, -1, -1, -1> v265 = pto::Stride<-1, -1, -1, -1, -1>(v260 * v263, v263, v262, v9, v10);
    // pto: %36
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v266 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v258 + (v257 + v46 * v9 + v20 * v10), v264, v265);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v255, v266);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v11_pview
    ;
    const int64_t v267 = 0;
    // pto: %stage__ssa_v11_pview
    ;
    __gm__ int8_t* v268 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v11_pview
    ;
    const int64_t v269 = 1;
    // pto: %stage__ssa_v11_pview
    ;
    const int64_t v270 = 1;
    // pto: %stage__ssa_v11_pview
    ;
    const int64_t v271 = 1;
    // pto: %stage__ssa_v11_pview
    ;
    int64_t v272 = v10 * v9;
    // pto: %stage__ssa_v11_pview
    ;
    int64_t v273 = v271 * v272;
    // pto: %stage__ssa_v11_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v274 = pto::Shape<1, 1, 1, 1, -1>(v269, v270, v271, v10, v12);
    // pto: %stage__ssa_v11_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v275 = pto::Stride<-1, -1, -1, -1, -1>(v270 * v273, v273, v272, v9, v10);
    // pto: %stage__ssa_v11_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v276 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v268 + (v267 + v46 * v9 + v20 * v10), v274, v275);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v276, v255);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v277 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v277);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v10
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v278 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v10
    ;
    uint64_t v279 = (uint64_t) v7;
    TASSIGN(v278, v279);
    // pto: %40
    ;
    const int64_t v280 = 0;
    // pto: %40
    ;
    __gm__ int8_t* v281 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %40
    ;
    const int64_t v282 = 1;
    // pto: %40
    ;
    const int64_t v283 = 1;
    // pto: %40
    ;
    const int64_t v284 = 1;
    // pto: %40
    ;
    int64_t v285 = v10 * v9;
    // pto: %40
    ;
    int64_t v286 = v284 * v285;
    // pto: %40
    ;
    pto::Shape<1, 1, 1, 1, -1> v287 = pto::Shape<1, 1, 1, 1, -1>(v282, v283, v284, v10, v12);
    // pto: %40
    ;
    pto::Stride<-1, -1, -1, -1, -1> v288 = pto::Stride<-1, -1, -1, -1, -1>(v283 * v286, v286, v285, v9, v10);
    // pto: %40
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v289 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v281 + (v280 + v46 * v9 + v21 * v10), v287, v288);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v278, v289);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v12_pview
    ;
    const int64_t v290 = 0;
    // pto: %stage__ssa_v12_pview
    ;
    __gm__ int8_t* v291 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v12_pview
    ;
    const int64_t v292 = 1;
    // pto: %stage__ssa_v12_pview
    ;
    const int64_t v293 = 1;
    // pto: %stage__ssa_v12_pview
    ;
    const int64_t v294 = 1;
    // pto: %stage__ssa_v12_pview
    ;
    int64_t v295 = v10 * v9;
    // pto: %stage__ssa_v12_pview
    ;
    int64_t v296 = v294 * v295;
    // pto: %stage__ssa_v12_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v297 = pto::Shape<1, 1, 1, 1, -1>(v292, v293, v294, v10, v12);
    // pto: %stage__ssa_v12_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v298 = pto::Stride<-1, -1, -1, -1, -1>(v293 * v296, v296, v295, v9, v10);
    // pto: %stage__ssa_v12_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v299 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v291 + (v290 + v46 * v9 + v21 * v10), v297, v298);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v299, v278);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v300 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v300);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v11
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v11
    ;
    uint64_t v302 = (uint64_t) v7;
    TASSIGN(v301, v302);
    // pto: %44
    ;
    const int64_t v303 = 0;
    // pto: %44
    ;
    __gm__ int8_t* v304 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %44
    ;
    const int64_t v305 = 1;
    // pto: %44
    ;
    const int64_t v306 = 1;
    // pto: %44
    ;
    const int64_t v307 = 1;
    // pto: %44
    ;
    int64_t v308 = v10 * v9;
    // pto: %44
    ;
    int64_t v309 = v307 * v308;
    // pto: %44
    ;
    pto::Shape<1, 1, 1, 1, -1> v310 = pto::Shape<1, 1, 1, 1, -1>(v305, v306, v307, v10, v12);
    // pto: %44
    ;
    pto::Stride<-1, -1, -1, -1, -1> v311 = pto::Stride<-1, -1, -1, -1, -1>(v306 * v309, v309, v308, v9, v10);
    // pto: %44
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v312 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v304 + (v303 + v46 * v9 + v22 * v10), v310, v311);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v301, v312);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v13_pview
    ;
    const int64_t v313 = 0;
    // pto: %stage__ssa_v13_pview
    ;
    __gm__ int8_t* v314 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v13_pview
    ;
    const int64_t v315 = 1;
    // pto: %stage__ssa_v13_pview
    ;
    const int64_t v316 = 1;
    // pto: %stage__ssa_v13_pview
    ;
    const int64_t v317 = 1;
    // pto: %stage__ssa_v13_pview
    ;
    int64_t v318 = v10 * v9;
    // pto: %stage__ssa_v13_pview
    ;
    int64_t v319 = v317 * v318;
    // pto: %stage__ssa_v13_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v320 = pto::Shape<1, 1, 1, 1, -1>(v315, v316, v317, v10, v12);
    // pto: %stage__ssa_v13_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v321 = pto::Stride<-1, -1, -1, -1, -1>(v316 * v319, v319, v318, v9, v10);
    // pto: %stage__ssa_v13_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v322 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v314 + (v313 + v46 * v9 + v22 * v10), v320, v321);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v322, v301);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v323 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v323);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v12
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v324 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v12
    ;
    uint64_t v325 = (uint64_t) v7;
    TASSIGN(v324, v325);
    // pto: %48
    ;
    const int64_t v326 = 0;
    // pto: %48
    ;
    __gm__ int8_t* v327 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %48
    ;
    const int64_t v328 = 1;
    // pto: %48
    ;
    const int64_t v329 = 1;
    // pto: %48
    ;
    const int64_t v330 = 1;
    // pto: %48
    ;
    int64_t v331 = v10 * v9;
    // pto: %48
    ;
    int64_t v332 = v330 * v331;
    // pto: %48
    ;
    pto::Shape<1, 1, 1, 1, -1> v333 = pto::Shape<1, 1, 1, 1, -1>(v328, v329, v330, v10, v12);
    // pto: %48
    ;
    pto::Stride<-1, -1, -1, -1, -1> v334 = pto::Stride<-1, -1, -1, -1, -1>(v329 * v332, v332, v331, v9, v10);
    // pto: %48
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v335 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v327 + (v326 + v46 * v9 + v23 * v10), v333, v334);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v324, v335);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v14_pview
    ;
    const int64_t v336 = 0;
    // pto: %stage__ssa_v14_pview
    ;
    __gm__ int8_t* v337 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v14_pview
    ;
    const int64_t v338 = 1;
    // pto: %stage__ssa_v14_pview
    ;
    const int64_t v339 = 1;
    // pto: %stage__ssa_v14_pview
    ;
    const int64_t v340 = 1;
    // pto: %stage__ssa_v14_pview
    ;
    int64_t v341 = v10 * v9;
    // pto: %stage__ssa_v14_pview
    ;
    int64_t v342 = v340 * v341;
    // pto: %stage__ssa_v14_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v343 = pto::Shape<1, 1, 1, 1, -1>(v338, v339, v340, v10, v12);
    // pto: %stage__ssa_v14_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v344 = pto::Stride<-1, -1, -1, -1, -1>(v339 * v342, v342, v341, v9, v10);
    // pto: %stage__ssa_v14_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v345 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v337 + (v336 + v46 * v9 + v23 * v10), v343, v344);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v345, v324);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v346 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v346);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v13
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v347 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v13
    ;
    uint64_t v348 = (uint64_t) v7;
    TASSIGN(v347, v348);
    // pto: %52
    ;
    const int64_t v349 = 0;
    // pto: %52
    ;
    __gm__ int8_t* v350 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %52
    ;
    const int64_t v351 = 1;
    // pto: %52
    ;
    const int64_t v352 = 1;
    // pto: %52
    ;
    const int64_t v353 = 1;
    // pto: %52
    ;
    int64_t v354 = v10 * v9;
    // pto: %52
    ;
    int64_t v355 = v353 * v354;
    // pto: %52
    ;
    pto::Shape<1, 1, 1, 1, -1> v356 = pto::Shape<1, 1, 1, 1, -1>(v351, v352, v353, v10, v12);
    // pto: %52
    ;
    pto::Stride<-1, -1, -1, -1, -1> v357 = pto::Stride<-1, -1, -1, -1, -1>(v352 * v355, v355, v354, v9, v10);
    // pto: %52
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v358 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v350 + (v349 + v46 * v9 + v24 * v10), v356, v357);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v347, v358);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v15_pview
    ;
    const int64_t v359 = 0;
    // pto: %stage__ssa_v15_pview
    ;
    __gm__ int8_t* v360 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v15_pview
    ;
    const int64_t v361 = 1;
    // pto: %stage__ssa_v15_pview
    ;
    const int64_t v362 = 1;
    // pto: %stage__ssa_v15_pview
    ;
    const int64_t v363 = 1;
    // pto: %stage__ssa_v15_pview
    ;
    int64_t v364 = v10 * v9;
    // pto: %stage__ssa_v15_pview
    ;
    int64_t v365 = v363 * v364;
    // pto: %stage__ssa_v15_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v366 = pto::Shape<1, 1, 1, 1, -1>(v361, v362, v363, v10, v12);
    // pto: %stage__ssa_v15_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v367 = pto::Stride<-1, -1, -1, -1, -1>(v362 * v365, v365, v364, v9, v10);
    // pto: %stage__ssa_v15_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v368 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v360 + (v359 + v46 * v9 + v24 * v10), v366, v367);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v368, v347);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v369 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v369);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v14
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v370 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v14
    ;
    uint64_t v371 = (uint64_t) v7;
    TASSIGN(v370, v371);
    // pto: %56
    ;
    const int64_t v372 = 0;
    // pto: %56
    ;
    __gm__ int8_t* v373 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %56
    ;
    const int64_t v374 = 1;
    // pto: %56
    ;
    const int64_t v375 = 1;
    // pto: %56
    ;
    const int64_t v376 = 1;
    // pto: %56
    ;
    int64_t v377 = v10 * v9;
    // pto: %56
    ;
    int64_t v378 = v376 * v377;
    // pto: %56
    ;
    pto::Shape<1, 1, 1, 1, -1> v379 = pto::Shape<1, 1, 1, 1, -1>(v374, v375, v376, v10, v12);
    // pto: %56
    ;
    pto::Stride<-1, -1, -1, -1, -1> v380 = pto::Stride<-1, -1, -1, -1, -1>(v375 * v378, v378, v377, v9, v10);
    // pto: %56
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v381 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v373 + (v372 + v46 * v9 + v25 * v10), v379, v380);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v370, v381);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v16_pview
    ;
    const int64_t v382 = 0;
    // pto: %stage__ssa_v16_pview
    ;
    __gm__ int8_t* v383 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v16_pview
    ;
    const int64_t v384 = 1;
    // pto: %stage__ssa_v16_pview
    ;
    const int64_t v385 = 1;
    // pto: %stage__ssa_v16_pview
    ;
    const int64_t v386 = 1;
    // pto: %stage__ssa_v16_pview
    ;
    int64_t v387 = v10 * v9;
    // pto: %stage__ssa_v16_pview
    ;
    int64_t v388 = v386 * v387;
    // pto: %stage__ssa_v16_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v389 = pto::Shape<1, 1, 1, 1, -1>(v384, v385, v386, v10, v12);
    // pto: %stage__ssa_v16_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v390 = pto::Stride<-1, -1, -1, -1, -1>(v385 * v388, v388, v387, v9, v10);
    // pto: %stage__ssa_v16_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v391 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v383 + (v382 + v46 * v9 + v25 * v10), v389, v390);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v391, v370);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v392 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v392);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v15
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v393 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %tile__ssa_v15
    ;
    uint64_t v394 = (uint64_t) v7;
    TASSIGN(v393, v394);
    // pto: %60
    ;
    const int64_t v395 = 0;
    // pto: %60
    ;
    __gm__ int8_t* v396 = PTOAS__GLOBAL_TENSOR_DATA(v34);
    // pto: %60
    ;
    const int64_t v397 = 1;
    // pto: %60
    ;
    const int64_t v398 = 1;
    // pto: %60
    ;
    const int64_t v399 = 1;
    // pto: %60
    ;
    int64_t v400 = v10 * v9;
    // pto: %60
    ;
    int64_t v401 = v399 * v400;
    // pto: %60
    ;
    pto::Shape<1, 1, 1, 1, -1> v402 = pto::Shape<1, 1, 1, 1, -1>(v397, v398, v399, v10, v12);
    // pto: %60
    ;
    pto::Stride<-1, -1, -1, -1, -1> v403 = pto::Stride<-1, -1, -1, -1, -1>(v398 * v401, v401, v400, v9, v10);
    // pto: %60
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v404 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v396 + (v395 + v46 * v9 + v26 * v10), v402, v403);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v393, v404);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v17_pview
    ;
    const int64_t v405 = 0;
    // pto: %stage__ssa_v17_pview
    ;
    __gm__ int8_t* v406 = PTOAS__GLOBAL_TENSOR_DATA(v42);
    // pto: %stage__ssa_v17_pview
    ;
    const int64_t v407 = 1;
    // pto: %stage__ssa_v17_pview
    ;
    const int64_t v408 = 1;
    // pto: %stage__ssa_v17_pview
    ;
    const int64_t v409 = 1;
    // pto: %stage__ssa_v17_pview
    ;
    int64_t v410 = v10 * v9;
    // pto: %stage__ssa_v17_pview
    ;
    int64_t v411 = v409 * v410;
    // pto: %stage__ssa_v17_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v412 = pto::Shape<1, 1, 1, 1, -1>(v407, v408, v409, v10, v12);
    // pto: %stage__ssa_v17_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v413 = pto::Stride<-1, -1, -1, -1, -1>(v408 * v411, v411, v410, v9, v10);
    // pto: %stage__ssa_v17_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v414 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v406 + (v405 + v46 * v9 + v26 * v10), v412, v413);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v414, v393);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    __gm__ int8_t* v415 = PTOAS__GLOBAL_TENSOR_DATA(v69);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v415);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  for (int64_t v416 = v11; v416 < v8; v416 += v10) {
    // pto: %v__tile
    ;
    int32_t v417 = (v2)[v416];
    (v4)[v416] = v417;
    // pto: %counts__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 4, 1> v418 = pto::Shape<1, 1, 1, 4, 1>();
    // pto: %counts__ssa_v0_pview
    ;
    pto::Stride<4, 4, 4, 1, 4> v419 = pto::Stride<4, 4, 4, 1, 4>();
    // pto: %counts__ssa_v0_pview
    ;
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN> v420 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN>((__gm__ int32_t*) v4, v418, v419);
    __gm__ int32_t* v421 = PTOAS__GLOBAL_TENSOR_DATA(v420);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v421);
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