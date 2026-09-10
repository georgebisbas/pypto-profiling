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
  // pto: %c262144_index
  const int64_t v6 = 262144;
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
  // pto: %c65536_index
  const int64_t v12 = 65536;
  // pto: %c81920_index
  const int64_t v13 = 81920;
  // pto: %c98304_index
  const int64_t v14 = 98304;
  // pto: %c114688_index
  const int64_t v15 = 114688;
  // pto: %c131072_index
  const int64_t v16 = 131072;
  // pto: %c147456_index
  const int64_t v17 = 147456;
  // pto: %c163840_index
  const int64_t v18 = 163840;
  // pto: %c180224_index
  const int64_t v19 = 180224;
  // pto: %c196608_index
  const int64_t v20 = 196608;
  // pto: %c212992_index
  const int64_t v21 = 212992;
  // pto: %c229376_index
  const int64_t v22 = 229376;
  // pto: %c245760_index
  const int64_t v23 = 245760;
  // pto: %inp__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %inp__ssa_v0_view
  const int64_t v26 = 1;
  // pto: %inp__ssa_v0_view
  int64_t v27 = v5 * v6;
  // pto: %inp__ssa_v0_view
  int64_t v28 = v26 * v27;
  // pto: %inp__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v29 = pto::Shape<1, 1, 1, -1, -1>(v24, v25, v26, v5, v6);
  // pto: %inp__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v30 = pto::Stride<-1, -1, -1, -1, -1>(v25 * v28, v28, v27, v6, v7);
  // pto: %inp__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v31 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v29, v30);
  // pto: %stage__ssa_v0_view
  const int64_t v32 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %stage__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %stage__ssa_v0_view
  int64_t v35 = v5 * v6;
  // pto: %stage__ssa_v0_view
  int64_t v36 = v34 * v35;
  // pto: %stage__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v37 = pto::Shape<1, 1, 1, -1, -1>(v32, v33, v34, v5, v6);
  // pto: %stage__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v38 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v36, v36, v35, v6, v7);
  // pto: %stage__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v39 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v37, v38);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v40 = v8; v40 < v5; v40 += v7) {
    // pto: %tile__ssa_v0
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v0
    ;
    uint64_t v42 = (uint64_t) v4;
    TASSIGN(v41, v42);
    // pto: %0
    ;
    int64_t v43 = v40 < v8 ? v8 : v40;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v44 = 0;
    // pto: %inp__ssa_v0_pview
    ;
    __gm__ int8_t* v45 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v46 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v47 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    const int64_t v48 = 1;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v49 = v7 * v6;
    // pto: %inp__ssa_v0_pview
    ;
    int64_t v50 = v48 * v49;
    // pto: %inp__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v51 = pto::Shape<1, 1, 1, 1, -1>(v46, v47, v48, v7, v9);
    // pto: %inp__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v52 = pto::Stride<-1, -1, -1, -1, -1>(v47 * v50, v50, v49, v6, v7);
    // pto: %inp__ssa_v0_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v53 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v45 + (v44 + v43 * v6 + v8 * v7), v51, v52);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v41, v53);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v54 = 0;
    // pto: %stage__iter_v1_pview
    ;
    __gm__ int8_t* v55 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v56 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v57 = 1;
    // pto: %stage__iter_v1_pview
    ;
    const int64_t v58 = 1;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v59 = v7 * v6;
    // pto: %stage__iter_v1_pview
    ;
    int64_t v60 = v58 * v59;
    // pto: %stage__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v61 = pto::Shape<1, 1, 1, 1, -1>(v56, v57, v58, v7, v9);
    // pto: %stage__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v62 = pto::Stride<-1, -1, -1, -1, -1>(v57 * v60, v60, v59, v6, v7);
    // pto: %stage__iter_v1_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v63 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v55 + (v54 + v43 * v6 + v8 * v7), v61, v62);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v63, v41);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %2
    ;
    pto::Shape<1, 1, 1, 4, 262144> v64 = pto::Shape<1, 1, 1, 4, 262144>();
    // pto: %2
    ;
    pto::Stride<1048576, 1048576, 1048576, 262144, 1> v65 = pto::Stride<1048576, 1048576, 1048576, 262144, 1>();
    // pto: %2
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 262144>, pto::Stride<1048576, 1048576, 1048576, 262144, 1>, pto::Layout::ND> v66 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 4, 262144>, pto::Stride<1048576, 1048576, 1048576, 262144, 1>, pto::Layout::ND>((__gm__ int8_t*) v2, v64, v65);
    __gm__ int8_t* v67 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v67);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v1
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v1
    ;
    uint64_t v69 = (uint64_t) v4;
    TASSIGN(v68, v69);
    // pto: %4
    ;
    const int64_t v70 = 0;
    // pto: %4
    ;
    __gm__ int8_t* v71 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %4
    ;
    const int64_t v72 = 1;
    // pto: %4
    ;
    const int64_t v73 = 1;
    // pto: %4
    ;
    const int64_t v74 = 1;
    // pto: %4
    ;
    int64_t v75 = v7 * v6;
    // pto: %4
    ;
    int64_t v76 = v74 * v75;
    // pto: %4
    ;
    pto::Shape<1, 1, 1, 1, -1> v77 = pto::Shape<1, 1, 1, 1, -1>(v72, v73, v74, v7, v9);
    // pto: %4
    ;
    pto::Stride<-1, -1, -1, -1, -1> v78 = pto::Stride<-1, -1, -1, -1, -1>(v73 * v76, v76, v75, v6, v7);
    // pto: %4
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v79 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v71 + (v70 + v43 * v6 + v9 * v7), v77, v78);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v68, v79);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v80 = 0;
    // pto: %stage__ssa_v3_pview
    ;
    __gm__ int8_t* v81 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v82 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v83 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    const int64_t v84 = 1;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v85 = v7 * v6;
    // pto: %stage__ssa_v3_pview
    ;
    int64_t v86 = v84 * v85;
    // pto: %stage__ssa_v3_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v87 = pto::Shape<1, 1, 1, 1, -1>(v82, v83, v84, v7, v9);
    // pto: %stage__ssa_v3_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v88 = pto::Stride<-1, -1, -1, -1, -1>(v83 * v86, v86, v85, v6, v7);
    // pto: %stage__ssa_v3_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v89 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v81 + (v80 + v43 * v6 + v9 * v7), v87, v88);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    TSTORE(v89, v68);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    __gm__ int8_t* v90 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v90);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v2
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v2
    ;
    uint64_t v92 = (uint64_t) v4;
    TASSIGN(v91, v92);
    // pto: %8
    ;
    const int64_t v93 = 0;
    // pto: %8
    ;
    __gm__ int8_t* v94 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %8
    ;
    const int64_t v95 = 1;
    // pto: %8
    ;
    const int64_t v96 = 1;
    // pto: %8
    ;
    const int64_t v97 = 1;
    // pto: %8
    ;
    int64_t v98 = v7 * v6;
    // pto: %8
    ;
    int64_t v99 = v97 * v98;
    // pto: %8
    ;
    pto::Shape<1, 1, 1, 1, -1> v100 = pto::Shape<1, 1, 1, 1, -1>(v95, v96, v97, v7, v9);
    // pto: %8
    ;
    pto::Stride<-1, -1, -1, -1, -1> v101 = pto::Stride<-1, -1, -1, -1, -1>(v96 * v99, v99, v98, v6, v7);
    // pto: %8
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v102 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v94 + (v93 + v43 * v6 + v10 * v7), v100, v101);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v91, v102);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v103 = 0;
    // pto: %stage__ssa_v4_pview
    ;
    __gm__ int8_t* v104 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v105 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v106 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    const int64_t v107 = 1;
    // pto: %stage__ssa_v4_pview
    ;
    int64_t v108 = v7 * v6;
    // pto: %stage__ssa_v4_pview
    ;
    int64_t v109 = v107 * v108;
    // pto: %stage__ssa_v4_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v110 = pto::Shape<1, 1, 1, 1, -1>(v105, v106, v107, v7, v9);
    // pto: %stage__ssa_v4_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v111 = pto::Stride<-1, -1, -1, -1, -1>(v106 * v109, v109, v108, v6, v7);
    // pto: %stage__ssa_v4_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v112 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v104 + (v103 + v43 * v6 + v10 * v7), v110, v111);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    TSTORE(v112, v91);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    __gm__ int8_t* v113 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v113);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v3
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v3
    ;
    uint64_t v115 = (uint64_t) v4;
    TASSIGN(v114, v115);
    // pto: %12
    ;
    const int64_t v116 = 0;
    // pto: %12
    ;
    __gm__ int8_t* v117 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %12
    ;
    const int64_t v118 = 1;
    // pto: %12
    ;
    const int64_t v119 = 1;
    // pto: %12
    ;
    const int64_t v120 = 1;
    // pto: %12
    ;
    int64_t v121 = v7 * v6;
    // pto: %12
    ;
    int64_t v122 = v120 * v121;
    // pto: %12
    ;
    pto::Shape<1, 1, 1, 1, -1> v123 = pto::Shape<1, 1, 1, 1, -1>(v118, v119, v120, v7, v9);
    // pto: %12
    ;
    pto::Stride<-1, -1, -1, -1, -1> v124 = pto::Stride<-1, -1, -1, -1, -1>(v119 * v122, v122, v121, v6, v7);
    // pto: %12
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v125 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v117 + (v116 + v43 * v6 + v11 * v7), v123, v124);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    TLOAD(v114, v125);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v126 = 0;
    // pto: %stage__ssa_v5_pview
    ;
    __gm__ int8_t* v127 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v128 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v129 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    const int64_t v130 = 1;
    // pto: %stage__ssa_v5_pview
    ;
    int64_t v131 = v7 * v6;
    // pto: %stage__ssa_v5_pview
    ;
    int64_t v132 = v130 * v131;
    // pto: %stage__ssa_v5_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v133 = pto::Shape<1, 1, 1, 1, -1>(v128, v129, v130, v7, v9);
    // pto: %stage__ssa_v5_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v134 = pto::Stride<-1, -1, -1, -1, -1>(v129 * v132, v132, v131, v6, v7);
    // pto: %stage__ssa_v5_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v135 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v127 + (v126 + v43 * v6 + v11 * v7), v133, v134);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
    TSTORE(v135, v114);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    __gm__ int8_t* v136 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v136);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v4
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v4
    ;
    uint64_t v138 = (uint64_t) v4;
    TASSIGN(v137, v138);
    // pto: %16
    ;
    const int64_t v139 = 0;
    // pto: %16
    ;
    __gm__ int8_t* v140 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %16
    ;
    const int64_t v141 = 1;
    // pto: %16
    ;
    const int64_t v142 = 1;
    // pto: %16
    ;
    const int64_t v143 = 1;
    // pto: %16
    ;
    int64_t v144 = v7 * v6;
    // pto: %16
    ;
    int64_t v145 = v143 * v144;
    // pto: %16
    ;
    pto::Shape<1, 1, 1, 1, -1> v146 = pto::Shape<1, 1, 1, 1, -1>(v141, v142, v143, v7, v9);
    // pto: %16
    ;
    pto::Stride<-1, -1, -1, -1, -1> v147 = pto::Stride<-1, -1, -1, -1, -1>(v142 * v145, v145, v144, v6, v7);
    // pto: %16
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v148 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v140 + (v139 + v43 * v6 + v12 * v7), v146, v147);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    TLOAD(v137, v148);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
    // pto: %stage__ssa_v6_pview
    ;
    const int64_t v149 = 0;
    // pto: %stage__ssa_v6_pview
    ;
    __gm__ int8_t* v150 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v6_pview
    ;
    const int64_t v151 = 1;
    // pto: %stage__ssa_v6_pview
    ;
    const int64_t v152 = 1;
    // pto: %stage__ssa_v6_pview
    ;
    const int64_t v153 = 1;
    // pto: %stage__ssa_v6_pview
    ;
    int64_t v154 = v7 * v6;
    // pto: %stage__ssa_v6_pview
    ;
    int64_t v155 = v153 * v154;
    // pto: %stage__ssa_v6_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v156 = pto::Shape<1, 1, 1, 1, -1>(v151, v152, v153, v7, v9);
    // pto: %stage__ssa_v6_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v157 = pto::Stride<-1, -1, -1, -1, -1>(v152 * v155, v155, v154, v6, v7);
    // pto: %stage__ssa_v6_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v158 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v150 + (v149 + v43 * v6 + v12 * v7), v156, v157);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
    TSTORE(v158, v137);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    __gm__ int8_t* v159 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v159);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v5
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v160 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v5
    ;
    uint64_t v161 = (uint64_t) v4;
    TASSIGN(v160, v161);
    // pto: %20
    ;
    const int64_t v162 = 0;
    // pto: %20
    ;
    __gm__ int8_t* v163 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %20
    ;
    const int64_t v164 = 1;
    // pto: %20
    ;
    const int64_t v165 = 1;
    // pto: %20
    ;
    const int64_t v166 = 1;
    // pto: %20
    ;
    int64_t v167 = v7 * v6;
    // pto: %20
    ;
    int64_t v168 = v166 * v167;
    // pto: %20
    ;
    pto::Shape<1, 1, 1, 1, -1> v169 = pto::Shape<1, 1, 1, 1, -1>(v164, v165, v166, v7, v9);
    // pto: %20
    ;
    pto::Stride<-1, -1, -1, -1, -1> v170 = pto::Stride<-1, -1, -1, -1, -1>(v165 * v168, v168, v167, v6, v7);
    // pto: %20
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v171 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v163 + (v162 + v43 * v6 + v13 * v7), v169, v170);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    TLOAD(v160, v171);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID5);
    // pto: %stage__ssa_v7_pview
    ;
    const int64_t v172 = 0;
    // pto: %stage__ssa_v7_pview
    ;
    __gm__ int8_t* v173 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v7_pview
    ;
    const int64_t v174 = 1;
    // pto: %stage__ssa_v7_pview
    ;
    const int64_t v175 = 1;
    // pto: %stage__ssa_v7_pview
    ;
    const int64_t v176 = 1;
    // pto: %stage__ssa_v7_pview
    ;
    int64_t v177 = v7 * v6;
    // pto: %stage__ssa_v7_pview
    ;
    int64_t v178 = v176 * v177;
    // pto: %stage__ssa_v7_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v179 = pto::Shape<1, 1, 1, 1, -1>(v174, v175, v176, v7, v9);
    // pto: %stage__ssa_v7_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v180 = pto::Stride<-1, -1, -1, -1, -1>(v175 * v178, v178, v177, v6, v7);
    // pto: %stage__ssa_v7_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v181 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v173 + (v172 + v43 * v6 + v13 * v7), v179, v180);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID5);
    TSTORE(v181, v160);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
    __gm__ int8_t* v182 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v182);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v6
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v6
    ;
    uint64_t v184 = (uint64_t) v4;
    TASSIGN(v183, v184);
    // pto: %24
    ;
    const int64_t v185 = 0;
    // pto: %24
    ;
    __gm__ int8_t* v186 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %24
    ;
    const int64_t v187 = 1;
    // pto: %24
    ;
    const int64_t v188 = 1;
    // pto: %24
    ;
    const int64_t v189 = 1;
    // pto: %24
    ;
    int64_t v190 = v7 * v6;
    // pto: %24
    ;
    int64_t v191 = v189 * v190;
    // pto: %24
    ;
    pto::Shape<1, 1, 1, 1, -1> v192 = pto::Shape<1, 1, 1, 1, -1>(v187, v188, v189, v7, v9);
    // pto: %24
    ;
    pto::Stride<-1, -1, -1, -1, -1> v193 = pto::Stride<-1, -1, -1, -1, -1>(v188 * v191, v191, v190, v6, v7);
    // pto: %24
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v194 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v186 + (v185 + v43 * v6 + v14 * v7), v192, v193);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
    TLOAD(v183, v194);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
    // pto: %stage__ssa_v8_pview
    ;
    const int64_t v195 = 0;
    // pto: %stage__ssa_v8_pview
    ;
    __gm__ int8_t* v196 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v8_pview
    ;
    const int64_t v197 = 1;
    // pto: %stage__ssa_v8_pview
    ;
    const int64_t v198 = 1;
    // pto: %stage__ssa_v8_pview
    ;
    const int64_t v199 = 1;
    // pto: %stage__ssa_v8_pview
    ;
    int64_t v200 = v7 * v6;
    // pto: %stage__ssa_v8_pview
    ;
    int64_t v201 = v199 * v200;
    // pto: %stage__ssa_v8_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v202 = pto::Shape<1, 1, 1, 1, -1>(v197, v198, v199, v7, v9);
    // pto: %stage__ssa_v8_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v203 = pto::Stride<-1, -1, -1, -1, -1>(v198 * v201, v201, v200, v6, v7);
    // pto: %stage__ssa_v8_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v204 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v196 + (v195 + v43 * v6 + v14 * v7), v202, v203);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
    TSTORE(v204, v183);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    __gm__ int8_t* v205 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v205);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v7
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v7
    ;
    uint64_t v207 = (uint64_t) v4;
    TASSIGN(v206, v207);
    // pto: %28
    ;
    const int64_t v208 = 0;
    // pto: %28
    ;
    __gm__ int8_t* v209 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %28
    ;
    const int64_t v210 = 1;
    // pto: %28
    ;
    const int64_t v211 = 1;
    // pto: %28
    ;
    const int64_t v212 = 1;
    // pto: %28
    ;
    int64_t v213 = v7 * v6;
    // pto: %28
    ;
    int64_t v214 = v212 * v213;
    // pto: %28
    ;
    pto::Shape<1, 1, 1, 1, -1> v215 = pto::Shape<1, 1, 1, 1, -1>(v210, v211, v212, v7, v9);
    // pto: %28
    ;
    pto::Stride<-1, -1, -1, -1, -1> v216 = pto::Stride<-1, -1, -1, -1, -1>(v211 * v214, v214, v213, v6, v7);
    // pto: %28
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v217 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v209 + (v208 + v43 * v6 + v15 * v7), v215, v216);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    TLOAD(v206, v217);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    // pto: %stage__ssa_v9_pview
    ;
    const int64_t v218 = 0;
    // pto: %stage__ssa_v9_pview
    ;
    __gm__ int8_t* v219 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v9_pview
    ;
    const int64_t v220 = 1;
    // pto: %stage__ssa_v9_pview
    ;
    const int64_t v221 = 1;
    // pto: %stage__ssa_v9_pview
    ;
    const int64_t v222 = 1;
    // pto: %stage__ssa_v9_pview
    ;
    int64_t v223 = v7 * v6;
    // pto: %stage__ssa_v9_pview
    ;
    int64_t v224 = v222 * v223;
    // pto: %stage__ssa_v9_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v225 = pto::Shape<1, 1, 1, 1, -1>(v220, v221, v222, v7, v9);
    // pto: %stage__ssa_v9_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v226 = pto::Stride<-1, -1, -1, -1, -1>(v221 * v224, v224, v223, v6, v7);
    // pto: %stage__ssa_v9_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v227 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v219 + (v218 + v43 * v6 + v15 * v7), v225, v226);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
    TSTORE(v227, v206);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v228 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v228);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v8
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v229 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v8
    ;
    uint64_t v230 = (uint64_t) v4;
    TASSIGN(v229, v230);
    // pto: %32
    ;
    const int64_t v231 = 0;
    // pto: %32
    ;
    __gm__ int8_t* v232 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %32
    ;
    const int64_t v233 = 1;
    // pto: %32
    ;
    const int64_t v234 = 1;
    // pto: %32
    ;
    const int64_t v235 = 1;
    // pto: %32
    ;
    int64_t v236 = v7 * v6;
    // pto: %32
    ;
    int64_t v237 = v235 * v236;
    // pto: %32
    ;
    pto::Shape<1, 1, 1, 1, -1> v238 = pto::Shape<1, 1, 1, 1, -1>(v233, v234, v235, v7, v9);
    // pto: %32
    ;
    pto::Stride<-1, -1, -1, -1, -1> v239 = pto::Stride<-1, -1, -1, -1, -1>(v234 * v237, v237, v236, v6, v7);
    // pto: %32
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v240 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v232 + (v231 + v43 * v6 + v16 * v7), v238, v239);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v229, v240);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v10_pview
    ;
    const int64_t v241 = 0;
    // pto: %stage__ssa_v10_pview
    ;
    __gm__ int8_t* v242 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v10_pview
    ;
    const int64_t v243 = 1;
    // pto: %stage__ssa_v10_pview
    ;
    const int64_t v244 = 1;
    // pto: %stage__ssa_v10_pview
    ;
    const int64_t v245 = 1;
    // pto: %stage__ssa_v10_pview
    ;
    int64_t v246 = v7 * v6;
    // pto: %stage__ssa_v10_pview
    ;
    int64_t v247 = v245 * v246;
    // pto: %stage__ssa_v10_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v248 = pto::Shape<1, 1, 1, 1, -1>(v243, v244, v245, v7, v9);
    // pto: %stage__ssa_v10_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v249 = pto::Stride<-1, -1, -1, -1, -1>(v244 * v247, v247, v246, v6, v7);
    // pto: %stage__ssa_v10_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v250 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v242 + (v241 + v43 * v6 + v16 * v7), v248, v249);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v250, v229);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v251 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v251);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v9
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v252 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v9
    ;
    uint64_t v253 = (uint64_t) v4;
    TASSIGN(v252, v253);
    // pto: %36
    ;
    const int64_t v254 = 0;
    // pto: %36
    ;
    __gm__ int8_t* v255 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %36
    ;
    const int64_t v256 = 1;
    // pto: %36
    ;
    const int64_t v257 = 1;
    // pto: %36
    ;
    const int64_t v258 = 1;
    // pto: %36
    ;
    int64_t v259 = v7 * v6;
    // pto: %36
    ;
    int64_t v260 = v258 * v259;
    // pto: %36
    ;
    pto::Shape<1, 1, 1, 1, -1> v261 = pto::Shape<1, 1, 1, 1, -1>(v256, v257, v258, v7, v9);
    // pto: %36
    ;
    pto::Stride<-1, -1, -1, -1, -1> v262 = pto::Stride<-1, -1, -1, -1, -1>(v257 * v260, v260, v259, v6, v7);
    // pto: %36
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v263 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v255 + (v254 + v43 * v6 + v17 * v7), v261, v262);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v252, v263);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v11_pview
    ;
    const int64_t v264 = 0;
    // pto: %stage__ssa_v11_pview
    ;
    __gm__ int8_t* v265 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v11_pview
    ;
    const int64_t v266 = 1;
    // pto: %stage__ssa_v11_pview
    ;
    const int64_t v267 = 1;
    // pto: %stage__ssa_v11_pview
    ;
    const int64_t v268 = 1;
    // pto: %stage__ssa_v11_pview
    ;
    int64_t v269 = v7 * v6;
    // pto: %stage__ssa_v11_pview
    ;
    int64_t v270 = v268 * v269;
    // pto: %stage__ssa_v11_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v271 = pto::Shape<1, 1, 1, 1, -1>(v266, v267, v268, v7, v9);
    // pto: %stage__ssa_v11_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v272 = pto::Stride<-1, -1, -1, -1, -1>(v267 * v270, v270, v269, v6, v7);
    // pto: %stage__ssa_v11_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v273 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v265 + (v264 + v43 * v6 + v17 * v7), v271, v272);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v273, v252);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v274 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v274);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v10
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v275 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v10
    ;
    uint64_t v276 = (uint64_t) v4;
    TASSIGN(v275, v276);
    // pto: %40
    ;
    const int64_t v277 = 0;
    // pto: %40
    ;
    __gm__ int8_t* v278 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %40
    ;
    const int64_t v279 = 1;
    // pto: %40
    ;
    const int64_t v280 = 1;
    // pto: %40
    ;
    const int64_t v281 = 1;
    // pto: %40
    ;
    int64_t v282 = v7 * v6;
    // pto: %40
    ;
    int64_t v283 = v281 * v282;
    // pto: %40
    ;
    pto::Shape<1, 1, 1, 1, -1> v284 = pto::Shape<1, 1, 1, 1, -1>(v279, v280, v281, v7, v9);
    // pto: %40
    ;
    pto::Stride<-1, -1, -1, -1, -1> v285 = pto::Stride<-1, -1, -1, -1, -1>(v280 * v283, v283, v282, v6, v7);
    // pto: %40
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v286 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v278 + (v277 + v43 * v6 + v18 * v7), v284, v285);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v275, v286);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v12_pview
    ;
    const int64_t v287 = 0;
    // pto: %stage__ssa_v12_pview
    ;
    __gm__ int8_t* v288 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v12_pview
    ;
    const int64_t v289 = 1;
    // pto: %stage__ssa_v12_pview
    ;
    const int64_t v290 = 1;
    // pto: %stage__ssa_v12_pview
    ;
    const int64_t v291 = 1;
    // pto: %stage__ssa_v12_pview
    ;
    int64_t v292 = v7 * v6;
    // pto: %stage__ssa_v12_pview
    ;
    int64_t v293 = v291 * v292;
    // pto: %stage__ssa_v12_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v294 = pto::Shape<1, 1, 1, 1, -1>(v289, v290, v291, v7, v9);
    // pto: %stage__ssa_v12_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v295 = pto::Stride<-1, -1, -1, -1, -1>(v290 * v293, v293, v292, v6, v7);
    // pto: %stage__ssa_v12_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v296 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v288 + (v287 + v43 * v6 + v18 * v7), v294, v295);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v296, v275);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v297 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v297);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v11
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v298 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v11
    ;
    uint64_t v299 = (uint64_t) v4;
    TASSIGN(v298, v299);
    // pto: %44
    ;
    const int64_t v300 = 0;
    // pto: %44
    ;
    __gm__ int8_t* v301 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %44
    ;
    const int64_t v302 = 1;
    // pto: %44
    ;
    const int64_t v303 = 1;
    // pto: %44
    ;
    const int64_t v304 = 1;
    // pto: %44
    ;
    int64_t v305 = v7 * v6;
    // pto: %44
    ;
    int64_t v306 = v304 * v305;
    // pto: %44
    ;
    pto::Shape<1, 1, 1, 1, -1> v307 = pto::Shape<1, 1, 1, 1, -1>(v302, v303, v304, v7, v9);
    // pto: %44
    ;
    pto::Stride<-1, -1, -1, -1, -1> v308 = pto::Stride<-1, -1, -1, -1, -1>(v303 * v306, v306, v305, v6, v7);
    // pto: %44
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v309 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v301 + (v300 + v43 * v6 + v19 * v7), v307, v308);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v298, v309);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v13_pview
    ;
    const int64_t v310 = 0;
    // pto: %stage__ssa_v13_pview
    ;
    __gm__ int8_t* v311 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v13_pview
    ;
    const int64_t v312 = 1;
    // pto: %stage__ssa_v13_pview
    ;
    const int64_t v313 = 1;
    // pto: %stage__ssa_v13_pview
    ;
    const int64_t v314 = 1;
    // pto: %stage__ssa_v13_pview
    ;
    int64_t v315 = v7 * v6;
    // pto: %stage__ssa_v13_pview
    ;
    int64_t v316 = v314 * v315;
    // pto: %stage__ssa_v13_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v317 = pto::Shape<1, 1, 1, 1, -1>(v312, v313, v314, v7, v9);
    // pto: %stage__ssa_v13_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v318 = pto::Stride<-1, -1, -1, -1, -1>(v313 * v316, v316, v315, v6, v7);
    // pto: %stage__ssa_v13_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v319 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v311 + (v310 + v43 * v6 + v19 * v7), v317, v318);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v319, v298);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v320 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v320);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v12
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v12
    ;
    uint64_t v322 = (uint64_t) v4;
    TASSIGN(v321, v322);
    // pto: %48
    ;
    const int64_t v323 = 0;
    // pto: %48
    ;
    __gm__ int8_t* v324 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %48
    ;
    const int64_t v325 = 1;
    // pto: %48
    ;
    const int64_t v326 = 1;
    // pto: %48
    ;
    const int64_t v327 = 1;
    // pto: %48
    ;
    int64_t v328 = v7 * v6;
    // pto: %48
    ;
    int64_t v329 = v327 * v328;
    // pto: %48
    ;
    pto::Shape<1, 1, 1, 1, -1> v330 = pto::Shape<1, 1, 1, 1, -1>(v325, v326, v327, v7, v9);
    // pto: %48
    ;
    pto::Stride<-1, -1, -1, -1, -1> v331 = pto::Stride<-1, -1, -1, -1, -1>(v326 * v329, v329, v328, v6, v7);
    // pto: %48
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v332 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v324 + (v323 + v43 * v6 + v20 * v7), v330, v331);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v321, v332);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v14_pview
    ;
    const int64_t v333 = 0;
    // pto: %stage__ssa_v14_pview
    ;
    __gm__ int8_t* v334 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v14_pview
    ;
    const int64_t v335 = 1;
    // pto: %stage__ssa_v14_pview
    ;
    const int64_t v336 = 1;
    // pto: %stage__ssa_v14_pview
    ;
    const int64_t v337 = 1;
    // pto: %stage__ssa_v14_pview
    ;
    int64_t v338 = v7 * v6;
    // pto: %stage__ssa_v14_pview
    ;
    int64_t v339 = v337 * v338;
    // pto: %stage__ssa_v14_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v340 = pto::Shape<1, 1, 1, 1, -1>(v335, v336, v337, v7, v9);
    // pto: %stage__ssa_v14_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v341 = pto::Stride<-1, -1, -1, -1, -1>(v336 * v339, v339, v338, v6, v7);
    // pto: %stage__ssa_v14_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v342 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v334 + (v333 + v43 * v6 + v20 * v7), v340, v341);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v342, v321);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v343 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v343);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v13
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v13
    ;
    uint64_t v345 = (uint64_t) v4;
    TASSIGN(v344, v345);
    // pto: %52
    ;
    const int64_t v346 = 0;
    // pto: %52
    ;
    __gm__ int8_t* v347 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %52
    ;
    const int64_t v348 = 1;
    // pto: %52
    ;
    const int64_t v349 = 1;
    // pto: %52
    ;
    const int64_t v350 = 1;
    // pto: %52
    ;
    int64_t v351 = v7 * v6;
    // pto: %52
    ;
    int64_t v352 = v350 * v351;
    // pto: %52
    ;
    pto::Shape<1, 1, 1, 1, -1> v353 = pto::Shape<1, 1, 1, 1, -1>(v348, v349, v350, v7, v9);
    // pto: %52
    ;
    pto::Stride<-1, -1, -1, -1, -1> v354 = pto::Stride<-1, -1, -1, -1, -1>(v349 * v352, v352, v351, v6, v7);
    // pto: %52
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v355 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v347 + (v346 + v43 * v6 + v21 * v7), v353, v354);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v344, v355);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v15_pview
    ;
    const int64_t v356 = 0;
    // pto: %stage__ssa_v15_pview
    ;
    __gm__ int8_t* v357 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v15_pview
    ;
    const int64_t v358 = 1;
    // pto: %stage__ssa_v15_pview
    ;
    const int64_t v359 = 1;
    // pto: %stage__ssa_v15_pview
    ;
    const int64_t v360 = 1;
    // pto: %stage__ssa_v15_pview
    ;
    int64_t v361 = v7 * v6;
    // pto: %stage__ssa_v15_pview
    ;
    int64_t v362 = v360 * v361;
    // pto: %stage__ssa_v15_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v363 = pto::Shape<1, 1, 1, 1, -1>(v358, v359, v360, v7, v9);
    // pto: %stage__ssa_v15_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v364 = pto::Stride<-1, -1, -1, -1, -1>(v359 * v362, v362, v361, v6, v7);
    // pto: %stage__ssa_v15_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v365 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v357 + (v356 + v43 * v6 + v21 * v7), v363, v364);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v365, v344);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v366 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v366);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v14
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v367 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v14
    ;
    uint64_t v368 = (uint64_t) v4;
    TASSIGN(v367, v368);
    // pto: %56
    ;
    const int64_t v369 = 0;
    // pto: %56
    ;
    __gm__ int8_t* v370 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %56
    ;
    const int64_t v371 = 1;
    // pto: %56
    ;
    const int64_t v372 = 1;
    // pto: %56
    ;
    const int64_t v373 = 1;
    // pto: %56
    ;
    int64_t v374 = v7 * v6;
    // pto: %56
    ;
    int64_t v375 = v373 * v374;
    // pto: %56
    ;
    pto::Shape<1, 1, 1, 1, -1> v376 = pto::Shape<1, 1, 1, 1, -1>(v371, v372, v373, v7, v9);
    // pto: %56
    ;
    pto::Stride<-1, -1, -1, -1, -1> v377 = pto::Stride<-1, -1, -1, -1, -1>(v372 * v375, v375, v374, v6, v7);
    // pto: %56
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v378 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v370 + (v369 + v43 * v6 + v22 * v7), v376, v377);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v367, v378);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v16_pview
    ;
    const int64_t v379 = 0;
    // pto: %stage__ssa_v16_pview
    ;
    __gm__ int8_t* v380 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v16_pview
    ;
    const int64_t v381 = 1;
    // pto: %stage__ssa_v16_pview
    ;
    const int64_t v382 = 1;
    // pto: %stage__ssa_v16_pview
    ;
    const int64_t v383 = 1;
    // pto: %stage__ssa_v16_pview
    ;
    int64_t v384 = v7 * v6;
    // pto: %stage__ssa_v16_pview
    ;
    int64_t v385 = v383 * v384;
    // pto: %stage__ssa_v16_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v386 = pto::Shape<1, 1, 1, 1, -1>(v381, v382, v383, v7, v9);
    // pto: %stage__ssa_v16_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v387 = pto::Stride<-1, -1, -1, -1, -1>(v382 * v385, v385, v384, v6, v7);
    // pto: %stage__ssa_v16_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v388 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v380 + (v379 + v43 * v6 + v22 * v7), v386, v387);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v388, v367);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    __gm__ int8_t* v389 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v389);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    // pto: %tile__ssa_v15
    ;
    Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v390 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v7, v9);
    // pto: %tile__ssa_v15
    ;
    uint64_t v391 = (uint64_t) v4;
    TASSIGN(v390, v391);
    // pto: %60
    ;
    const int64_t v392 = 0;
    // pto: %60
    ;
    __gm__ int8_t* v393 = PTOAS__GLOBAL_TENSOR_DATA(v31);
    // pto: %60
    ;
    const int64_t v394 = 1;
    // pto: %60
    ;
    const int64_t v395 = 1;
    // pto: %60
    ;
    const int64_t v396 = 1;
    // pto: %60
    ;
    int64_t v397 = v7 * v6;
    // pto: %60
    ;
    int64_t v398 = v396 * v397;
    // pto: %60
    ;
    pto::Shape<1, 1, 1, 1, -1> v399 = pto::Shape<1, 1, 1, 1, -1>(v394, v395, v396, v7, v9);
    // pto: %60
    ;
    pto::Stride<-1, -1, -1, -1, -1> v400 = pto::Stride<-1, -1, -1, -1, -1>(v395 * v398, v398, v397, v6, v7);
    // pto: %60
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v401 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v393 + (v392 + v43 * v6 + v23 * v7), v399, v400);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    TLOAD(v390, v401);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %stage__ssa_v17_pview
    ;
    const int64_t v402 = 0;
    // pto: %stage__ssa_v17_pview
    ;
    __gm__ int8_t* v403 = PTOAS__GLOBAL_TENSOR_DATA(v39);
    // pto: %stage__ssa_v17_pview
    ;
    const int64_t v404 = 1;
    // pto: %stage__ssa_v17_pview
    ;
    const int64_t v405 = 1;
    // pto: %stage__ssa_v17_pview
    ;
    const int64_t v406 = 1;
    // pto: %stage__ssa_v17_pview
    ;
    int64_t v407 = v7 * v6;
    // pto: %stage__ssa_v17_pview
    ;
    int64_t v408 = v406 * v407;
    // pto: %stage__ssa_v17_pview
    ;
    pto::Shape<1, 1, 1, 1, -1> v409 = pto::Shape<1, 1, 1, 1, -1>(v404, v405, v406, v7, v9);
    // pto: %stage__ssa_v17_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v410 = pto::Stride<-1, -1, -1, -1, -1>(v405 * v408, v408, v407, v6, v7);
    // pto: %stage__ssa_v17_pview
    ;
    GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v411 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v403 + (v402 + v43 * v6 + v23 * v7), v409, v410);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v411, v390);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    __gm__ int8_t* v412 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v412);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}