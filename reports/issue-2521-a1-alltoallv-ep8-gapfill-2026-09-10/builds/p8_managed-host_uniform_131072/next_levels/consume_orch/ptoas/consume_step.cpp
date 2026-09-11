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

AICORE void consume_step(__gm__ int8_t* v1, __gm__ int32_t* v2, __gm__ int8_t* v3, __gm__ int32_t* v4, __gm__ int64_t* v5, __gm__ int64_t* v6) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v7 = 0;
  // pto: %c8_index
  const int64_t v8 = 8;
  // pto: %c131072_index
  const int64_t v9 = 131072;
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
  // pto: %data__ssa_v0_view
  const int64_t v19 = 1;
  // pto: %data__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %data__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %data__ssa_v0_view
  int64_t v22 = v8 * v9;
  // pto: %data__ssa_v0_view
  int64_t v23 = v21 * v22;
  // pto: %data__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v24 = pto::Shape<1, 1, 1, -1, -1>(v19, v20, v21, v8, v9);
  // pto: %data__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v25 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v23, v23, v22, v9, v10);
  // pto: %data__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v26 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v24, v25);
  // pto: %out__ssa_v0_view
  const int64_t v27 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v28 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %out__ssa_v0_view
  int64_t v30 = v8 * v9;
  // pto: %out__ssa_v0_view
  int64_t v31 = v29 * v30;
  // pto: %out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v32 = pto::Shape<1, 1, 1, -1, -1>(v27, v28, v29, v8, v9);
  // pto: %out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v33 = pto::Stride<-1, -1, -1, -1, -1>(v28 * v31, v31, v30, v9, v10);
  // pto: %out__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v34 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v32, v33);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t v35 = v11; v35 < v8; v35 += v10) {
    // pto: %n_rows_i32__tile
    ;
    int32_t v36 = (v2)[v35];
    (v4)[v35] = v36;
    // pto: %2
    ;
    int64_t v37 = (int64_t) v36;
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    for (int64_t v38 = v11; v38 < v37; v38 += v10) {
      // pto: %3
      ;
      int64_t v39 = (int64_t) ((uint64_t) v35 + (uint64_t) v38);
      // pto: %tile__ssa_v0
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v0
      ;
      uint64_t v41 = (uint64_t) v7;
      TASSIGN(v40, v41);
      // pto: %4
      ;
      int64_t v42 = v39 < v11 ? v11 : v39;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v43 = 0;
      // pto: %data__ssa_v0_pview
      ;
      __gm__ int8_t* v44 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v45 = 1;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v46 = 1;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v47 = 1;
      // pto: %data__ssa_v0_pview
      ;
      int64_t v48 = v10 * v9;
      // pto: %data__ssa_v0_pview
      ;
      int64_t v49 = v47 * v48;
      // pto: %data__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v50 = pto::Shape<1, 1, 1, 1, -1>(v45, v46, v47, v10, v12);
      // pto: %data__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v51 = pto::Stride<-1, -1, -1, -1, -1>(v46 * v49, v49, v48, v9, v10);
      // pto: %data__ssa_v0_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v52 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v44 + (v43 + v42 * v9 + v11 * v10), v50, v51);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v40, v52);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__iter_v3_pview
      ;
      const int64_t v53 = 0;
      // pto: %out__iter_v3_pview
      ;
      __gm__ int8_t* v54 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__iter_v3_pview
      ;
      const int64_t v55 = 1;
      // pto: %out__iter_v3_pview
      ;
      const int64_t v56 = 1;
      // pto: %out__iter_v3_pview
      ;
      const int64_t v57 = 1;
      // pto: %out__iter_v3_pview
      ;
      int64_t v58 = v10 * v9;
      // pto: %out__iter_v3_pview
      ;
      int64_t v59 = v57 * v58;
      // pto: %out__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v60 = pto::Shape<1, 1, 1, 1, -1>(v55, v56, v57, v10, v12);
      // pto: %out__iter_v3_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v61 = pto::Stride<-1, -1, -1, -1, -1>(v56 * v59, v59, v58, v9, v10);
      // pto: %out__iter_v3_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v62 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v54 + (v53 + v42 * v9 + v11 * v10), v60, v61);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v62, v40);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v1
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v1
      ;
      uint64_t v64 = (uint64_t) v7;
      TASSIGN(v63, v64);
      // pto: %7
      ;
      const int64_t v65 = 0;
      // pto: %7
      ;
      __gm__ int8_t* v66 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %7
      ;
      const int64_t v67 = 1;
      // pto: %7
      ;
      const int64_t v68 = 1;
      // pto: %7
      ;
      const int64_t v69 = 1;
      // pto: %7
      ;
      int64_t v70 = v10 * v9;
      // pto: %7
      ;
      int64_t v71 = v69 * v70;
      // pto: %7
      ;
      pto::Shape<1, 1, 1, 1, -1> v72 = pto::Shape<1, 1, 1, 1, -1>(v67, v68, v69, v10, v12);
      // pto: %7
      ;
      pto::Stride<-1, -1, -1, -1, -1> v73 = pto::Stride<-1, -1, -1, -1, -1>(v68 * v71, v71, v70, v9, v10);
      // pto: %7
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v74 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v66 + (v65 + v42 * v9 + v12 * v10), v72, v73);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v63, v74);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v75 = 0;
      // pto: %out__ssa_v5_pview
      ;
      __gm__ int8_t* v76 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v77 = 1;
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v78 = 1;
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v79 = 1;
      // pto: %out__ssa_v5_pview
      ;
      int64_t v80 = v10 * v9;
      // pto: %out__ssa_v5_pview
      ;
      int64_t v81 = v79 * v80;
      // pto: %out__ssa_v5_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v82 = pto::Shape<1, 1, 1, 1, -1>(v77, v78, v79, v10, v12);
      // pto: %out__ssa_v5_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v83 = pto::Stride<-1, -1, -1, -1, -1>(v78 * v81, v81, v80, v9, v10);
      // pto: %out__ssa_v5_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v84 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v76 + (v75 + v42 * v9 + v12 * v10), v82, v83);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      TSTORE(v84, v63);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %tile__ssa_v2
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v2
      ;
      uint64_t v86 = (uint64_t) v7;
      TASSIGN(v85, v86);
      // pto: %10
      ;
      const int64_t v87 = 0;
      // pto: %10
      ;
      __gm__ int8_t* v88 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %10
      ;
      const int64_t v89 = 1;
      // pto: %10
      ;
      const int64_t v90 = 1;
      // pto: %10
      ;
      const int64_t v91 = 1;
      // pto: %10
      ;
      int64_t v92 = v10 * v9;
      // pto: %10
      ;
      int64_t v93 = v91 * v92;
      // pto: %10
      ;
      pto::Shape<1, 1, 1, 1, -1> v94 = pto::Shape<1, 1, 1, 1, -1>(v89, v90, v91, v10, v12);
      // pto: %10
      ;
      pto::Stride<-1, -1, -1, -1, -1> v95 = pto::Stride<-1, -1, -1, -1, -1>(v90 * v93, v93, v92, v9, v10);
      // pto: %10
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v96 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v88 + (v87 + v42 * v9 + v13 * v10), v94, v95);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      TLOAD(v85, v96);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v97 = 0;
      // pto: %out__ssa_v6_pview
      ;
      __gm__ int8_t* v98 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v99 = 1;
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v100 = 1;
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v101 = 1;
      // pto: %out__ssa_v6_pview
      ;
      int64_t v102 = v10 * v9;
      // pto: %out__ssa_v6_pview
      ;
      int64_t v103 = v101 * v102;
      // pto: %out__ssa_v6_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v104 = pto::Shape<1, 1, 1, 1, -1>(v99, v100, v101, v10, v12);
      // pto: %out__ssa_v6_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v105 = pto::Stride<-1, -1, -1, -1, -1>(v100 * v103, v103, v102, v9, v10);
      // pto: %out__ssa_v6_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v106 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v98 + (v97 + v42 * v9 + v13 * v10), v104, v105);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      TSTORE(v106, v85);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %tile__ssa_v3
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v3
      ;
      uint64_t v108 = (uint64_t) v7;
      TASSIGN(v107, v108);
      // pto: %13
      ;
      const int64_t v109 = 0;
      // pto: %13
      ;
      __gm__ int8_t* v110 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %13
      ;
      const int64_t v111 = 1;
      // pto: %13
      ;
      const int64_t v112 = 1;
      // pto: %13
      ;
      const int64_t v113 = 1;
      // pto: %13
      ;
      int64_t v114 = v10 * v9;
      // pto: %13
      ;
      int64_t v115 = v113 * v114;
      // pto: %13
      ;
      pto::Shape<1, 1, 1, 1, -1> v116 = pto::Shape<1, 1, 1, 1, -1>(v111, v112, v113, v10, v12);
      // pto: %13
      ;
      pto::Stride<-1, -1, -1, -1, -1> v117 = pto::Stride<-1, -1, -1, -1, -1>(v112 * v115, v115, v114, v9, v10);
      // pto: %13
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v118 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v110 + (v109 + v42 * v9 + v14 * v10), v116, v117);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      TLOAD(v107, v118);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
      // pto: %out__ssa_v7_pview
      ;
      const int64_t v119 = 0;
      // pto: %out__ssa_v7_pview
      ;
      __gm__ int8_t* v120 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v7_pview
      ;
      const int64_t v121 = 1;
      // pto: %out__ssa_v7_pview
      ;
      const int64_t v122 = 1;
      // pto: %out__ssa_v7_pview
      ;
      const int64_t v123 = 1;
      // pto: %out__ssa_v7_pview
      ;
      int64_t v124 = v10 * v9;
      // pto: %out__ssa_v7_pview
      ;
      int64_t v125 = v123 * v124;
      // pto: %out__ssa_v7_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v126 = pto::Shape<1, 1, 1, 1, -1>(v121, v122, v123, v10, v12);
      // pto: %out__ssa_v7_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v127 = pto::Stride<-1, -1, -1, -1, -1>(v122 * v125, v125, v124, v9, v10);
      // pto: %out__ssa_v7_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v128 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v120 + (v119 + v42 * v9 + v14 * v10), v126, v127);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
      TSTORE(v128, v107);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      // pto: %tile__ssa_v4
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v4
      ;
      uint64_t v130 = (uint64_t) v7;
      TASSIGN(v129, v130);
      // pto: %16
      ;
      const int64_t v131 = 0;
      // pto: %16
      ;
      __gm__ int8_t* v132 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %16
      ;
      const int64_t v133 = 1;
      // pto: %16
      ;
      const int64_t v134 = 1;
      // pto: %16
      ;
      const int64_t v135 = 1;
      // pto: %16
      ;
      int64_t v136 = v10 * v9;
      // pto: %16
      ;
      int64_t v137 = v135 * v136;
      // pto: %16
      ;
      pto::Shape<1, 1, 1, 1, -1> v138 = pto::Shape<1, 1, 1, 1, -1>(v133, v134, v135, v10, v12);
      // pto: %16
      ;
      pto::Stride<-1, -1, -1, -1, -1> v139 = pto::Stride<-1, -1, -1, -1, -1>(v134 * v137, v137, v136, v9, v10);
      // pto: %16
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v140 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v132 + (v131 + v42 * v9 + v15 * v10), v138, v139);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      TLOAD(v129, v140);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
      // pto: %out__ssa_v8_pview
      ;
      const int64_t v141 = 0;
      // pto: %out__ssa_v8_pview
      ;
      __gm__ int8_t* v142 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v8_pview
      ;
      const int64_t v143 = 1;
      // pto: %out__ssa_v8_pview
      ;
      const int64_t v144 = 1;
      // pto: %out__ssa_v8_pview
      ;
      const int64_t v145 = 1;
      // pto: %out__ssa_v8_pview
      ;
      int64_t v146 = v10 * v9;
      // pto: %out__ssa_v8_pview
      ;
      int64_t v147 = v145 * v146;
      // pto: %out__ssa_v8_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v148 = pto::Shape<1, 1, 1, 1, -1>(v143, v144, v145, v10, v12);
      // pto: %out__ssa_v8_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v149 = pto::Stride<-1, -1, -1, -1, -1>(v144 * v147, v147, v146, v9, v10);
      // pto: %out__ssa_v8_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v150 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v142 + (v141 + v42 * v9 + v15 * v10), v148, v149);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
      TSTORE(v150, v129);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
      // pto: %tile__ssa_v5
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v5
      ;
      uint64_t v152 = (uint64_t) v7;
      TASSIGN(v151, v152);
      // pto: %19
      ;
      const int64_t v153 = 0;
      // pto: %19
      ;
      __gm__ int8_t* v154 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %19
      ;
      const int64_t v155 = 1;
      // pto: %19
      ;
      const int64_t v156 = 1;
      // pto: %19
      ;
      const int64_t v157 = 1;
      // pto: %19
      ;
      int64_t v158 = v10 * v9;
      // pto: %19
      ;
      int64_t v159 = v157 * v158;
      // pto: %19
      ;
      pto::Shape<1, 1, 1, 1, -1> v160 = pto::Shape<1, 1, 1, 1, -1>(v155, v156, v157, v10, v12);
      // pto: %19
      ;
      pto::Stride<-1, -1, -1, -1, -1> v161 = pto::Stride<-1, -1, -1, -1, -1>(v156 * v159, v159, v158, v9, v10);
      // pto: %19
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v162 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v154 + (v153 + v42 * v9 + v16 * v10), v160, v161);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
      TLOAD(v151, v162);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID5);
      // pto: %out__ssa_v9_pview
      ;
      const int64_t v163 = 0;
      // pto: %out__ssa_v9_pview
      ;
      __gm__ int8_t* v164 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v9_pview
      ;
      const int64_t v165 = 1;
      // pto: %out__ssa_v9_pview
      ;
      const int64_t v166 = 1;
      // pto: %out__ssa_v9_pview
      ;
      const int64_t v167 = 1;
      // pto: %out__ssa_v9_pview
      ;
      int64_t v168 = v10 * v9;
      // pto: %out__ssa_v9_pview
      ;
      int64_t v169 = v167 * v168;
      // pto: %out__ssa_v9_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v170 = pto::Shape<1, 1, 1, 1, -1>(v165, v166, v167, v10, v12);
      // pto: %out__ssa_v9_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v171 = pto::Stride<-1, -1, -1, -1, -1>(v166 * v169, v169, v168, v9, v10);
      // pto: %out__ssa_v9_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v172 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v164 + (v163 + v42 * v9 + v16 * v10), v170, v171);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID5);
      TSTORE(v172, v151);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
      // pto: %tile__ssa_v6
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v173 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v6
      ;
      uint64_t v174 = (uint64_t) v7;
      TASSIGN(v173, v174);
      // pto: %22
      ;
      const int64_t v175 = 0;
      // pto: %22
      ;
      __gm__ int8_t* v176 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %22
      ;
      const int64_t v177 = 1;
      // pto: %22
      ;
      const int64_t v178 = 1;
      // pto: %22
      ;
      const int64_t v179 = 1;
      // pto: %22
      ;
      int64_t v180 = v10 * v9;
      // pto: %22
      ;
      int64_t v181 = v179 * v180;
      // pto: %22
      ;
      pto::Shape<1, 1, 1, 1, -1> v182 = pto::Shape<1, 1, 1, 1, -1>(v177, v178, v179, v10, v12);
      // pto: %22
      ;
      pto::Stride<-1, -1, -1, -1, -1> v183 = pto::Stride<-1, -1, -1, -1, -1>(v178 * v181, v181, v180, v9, v10);
      // pto: %22
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v184 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v176 + (v175 + v42 * v9 + v17 * v10), v182, v183);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
      TLOAD(v173, v184);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
      // pto: %out__ssa_v10_pview
      ;
      const int64_t v185 = 0;
      // pto: %out__ssa_v10_pview
      ;
      __gm__ int8_t* v186 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v10_pview
      ;
      const int64_t v187 = 1;
      // pto: %out__ssa_v10_pview
      ;
      const int64_t v188 = 1;
      // pto: %out__ssa_v10_pview
      ;
      const int64_t v189 = 1;
      // pto: %out__ssa_v10_pview
      ;
      int64_t v190 = v10 * v9;
      // pto: %out__ssa_v10_pview
      ;
      int64_t v191 = v189 * v190;
      // pto: %out__ssa_v10_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v192 = pto::Shape<1, 1, 1, 1, -1>(v187, v188, v189, v10, v12);
      // pto: %out__ssa_v10_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v193 = pto::Stride<-1, -1, -1, -1, -1>(v188 * v191, v191, v190, v9, v10);
      // pto: %out__ssa_v10_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v194 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v186 + (v185 + v42 * v9 + v17 * v10), v192, v193);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
      TSTORE(v194, v173);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v7
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v195 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v7
      ;
      uint64_t v196 = (uint64_t) v7;
      TASSIGN(v195, v196);
      // pto: %25
      ;
      const int64_t v197 = 0;
      // pto: %25
      ;
      __gm__ int8_t* v198 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %25
      ;
      const int64_t v199 = 1;
      // pto: %25
      ;
      const int64_t v200 = 1;
      // pto: %25
      ;
      const int64_t v201 = 1;
      // pto: %25
      ;
      int64_t v202 = v10 * v9;
      // pto: %25
      ;
      int64_t v203 = v201 * v202;
      // pto: %25
      ;
      pto::Shape<1, 1, 1, 1, -1> v204 = pto::Shape<1, 1, 1, 1, -1>(v199, v200, v201, v10, v12);
      // pto: %25
      ;
      pto::Stride<-1, -1, -1, -1, -1> v205 = pto::Stride<-1, -1, -1, -1, -1>(v200 * v203, v203, v202, v9, v10);
      // pto: %25
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v206 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v198 + (v197 + v42 * v9 + v18 * v10), v204, v205);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v195, v206);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v207 = 0;
      // pto: %out__ssa_v11_pview
      ;
      __gm__ int8_t* v208 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v209 = 1;
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v210 = 1;
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v211 = 1;
      // pto: %out__ssa_v11_pview
      ;
      int64_t v212 = v10 * v9;
      // pto: %out__ssa_v11_pview
      ;
      int64_t v213 = v211 * v212;
      // pto: %out__ssa_v11_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v214 = pto::Shape<1, 1, 1, 1, -1>(v209, v210, v211, v10, v12);
      // pto: %out__ssa_v11_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v215 = pto::Stride<-1, -1, -1, -1, -1>(v210 * v213, v213, v212, v9, v10);
      // pto: %out__ssa_v11_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v216 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v208 + (v207 + v42 * v9 + v18 * v10), v214, v215);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
      TSTORE(v216, v195);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    for (int64_t v217 = v37; v217 < v10; v217 += v10) {
      // pto: %28
      ;
      int64_t v218 = (int64_t) ((uint64_t) v35 + (uint64_t) v217);
      // pto: %tile__ssa_v8
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v219 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v8
      ;
      uint64_t v220 = (uint64_t) v7;
      TASSIGN(v219, v220);
      // pto: %29
      ;
      int64_t v221 = v218 < v11 ? v11 : v218;
      // pto: %30
      ;
      const int64_t v222 = 0;
      // pto: %30
      ;
      __gm__ int8_t* v223 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %30
      ;
      const int64_t v224 = 1;
      // pto: %30
      ;
      const int64_t v225 = 1;
      // pto: %30
      ;
      const int64_t v226 = 1;
      // pto: %30
      ;
      int64_t v227 = v10 * v9;
      // pto: %30
      ;
      int64_t v228 = v226 * v227;
      // pto: %30
      ;
      pto::Shape<1, 1, 1, 1, -1> v229 = pto::Shape<1, 1, 1, 1, -1>(v224, v225, v226, v10, v12);
      // pto: %30
      ;
      pto::Stride<-1, -1, -1, -1, -1> v230 = pto::Stride<-1, -1, -1, -1, -1>(v225 * v228, v228, v227, v9, v10);
      // pto: %30
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v231 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v223 + (v222 + v221 * v9 + v11 * v10), v229, v230);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v219, v231);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__iter_v13_pview
      ;
      const int64_t v232 = 0;
      // pto: %out__iter_v13_pview
      ;
      __gm__ int8_t* v233 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__iter_v13_pview
      ;
      const int64_t v234 = 1;
      // pto: %out__iter_v13_pview
      ;
      const int64_t v235 = 1;
      // pto: %out__iter_v13_pview
      ;
      const int64_t v236 = 1;
      // pto: %out__iter_v13_pview
      ;
      int64_t v237 = v10 * v9;
      // pto: %out__iter_v13_pview
      ;
      int64_t v238 = v236 * v237;
      // pto: %out__iter_v13_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v239 = pto::Shape<1, 1, 1, 1, -1>(v234, v235, v236, v10, v12);
      // pto: %out__iter_v13_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v240 = pto::Stride<-1, -1, -1, -1, -1>(v235 * v238, v238, v237, v9, v10);
      // pto: %out__iter_v13_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v241 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v233 + (v232 + v221 * v9 + v11 * v10), v239, v240);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v241, v219);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v9
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v242 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v9
      ;
      uint64_t v243 = (uint64_t) v7;
      TASSIGN(v242, v243);
      // pto: %33
      ;
      const int64_t v244 = 0;
      // pto: %33
      ;
      __gm__ int8_t* v245 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %33
      ;
      const int64_t v246 = 1;
      // pto: %33
      ;
      const int64_t v247 = 1;
      // pto: %33
      ;
      const int64_t v248 = 1;
      // pto: %33
      ;
      int64_t v249 = v10 * v9;
      // pto: %33
      ;
      int64_t v250 = v248 * v249;
      // pto: %33
      ;
      pto::Shape<1, 1, 1, 1, -1> v251 = pto::Shape<1, 1, 1, 1, -1>(v246, v247, v248, v10, v12);
      // pto: %33
      ;
      pto::Stride<-1, -1, -1, -1, -1> v252 = pto::Stride<-1, -1, -1, -1, -1>(v247 * v250, v250, v249, v9, v10);
      // pto: %33
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v253 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v245 + (v244 + v221 * v9 + v12 * v10), v251, v252);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v242, v253);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__ssa_v15_pview
      ;
      const int64_t v254 = 0;
      // pto: %out__ssa_v15_pview
      ;
      __gm__ int8_t* v255 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v15_pview
      ;
      const int64_t v256 = 1;
      // pto: %out__ssa_v15_pview
      ;
      const int64_t v257 = 1;
      // pto: %out__ssa_v15_pview
      ;
      const int64_t v258 = 1;
      // pto: %out__ssa_v15_pview
      ;
      int64_t v259 = v10 * v9;
      // pto: %out__ssa_v15_pview
      ;
      int64_t v260 = v258 * v259;
      // pto: %out__ssa_v15_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v261 = pto::Shape<1, 1, 1, 1, -1>(v256, v257, v258, v10, v12);
      // pto: %out__ssa_v15_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v262 = pto::Stride<-1, -1, -1, -1, -1>(v257 * v260, v260, v259, v9, v10);
      // pto: %out__ssa_v15_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v263 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v255 + (v254 + v221 * v9 + v12 * v10), v261, v262);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v263, v242);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v10
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v264 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v10
      ;
      uint64_t v265 = (uint64_t) v7;
      TASSIGN(v264, v265);
      // pto: %36
      ;
      const int64_t v266 = 0;
      // pto: %36
      ;
      __gm__ int8_t* v267 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %36
      ;
      const int64_t v268 = 1;
      // pto: %36
      ;
      const int64_t v269 = 1;
      // pto: %36
      ;
      const int64_t v270 = 1;
      // pto: %36
      ;
      int64_t v271 = v10 * v9;
      // pto: %36
      ;
      int64_t v272 = v270 * v271;
      // pto: %36
      ;
      pto::Shape<1, 1, 1, 1, -1> v273 = pto::Shape<1, 1, 1, 1, -1>(v268, v269, v270, v10, v12);
      // pto: %36
      ;
      pto::Stride<-1, -1, -1, -1, -1> v274 = pto::Stride<-1, -1, -1, -1, -1>(v269 * v272, v272, v271, v9, v10);
      // pto: %36
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v275 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v267 + (v266 + v221 * v9 + v13 * v10), v273, v274);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v264, v275);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__ssa_v16_pview
      ;
      const int64_t v276 = 0;
      // pto: %out__ssa_v16_pview
      ;
      __gm__ int8_t* v277 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v16_pview
      ;
      const int64_t v278 = 1;
      // pto: %out__ssa_v16_pview
      ;
      const int64_t v279 = 1;
      // pto: %out__ssa_v16_pview
      ;
      const int64_t v280 = 1;
      // pto: %out__ssa_v16_pview
      ;
      int64_t v281 = v10 * v9;
      // pto: %out__ssa_v16_pview
      ;
      int64_t v282 = v280 * v281;
      // pto: %out__ssa_v16_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v283 = pto::Shape<1, 1, 1, 1, -1>(v278, v279, v280, v10, v12);
      // pto: %out__ssa_v16_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v284 = pto::Stride<-1, -1, -1, -1, -1>(v279 * v282, v282, v281, v9, v10);
      // pto: %out__ssa_v16_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v285 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v277 + (v276 + v221 * v9 + v13 * v10), v283, v284);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v285, v264);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v11
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v286 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v11
      ;
      uint64_t v287 = (uint64_t) v7;
      TASSIGN(v286, v287);
      // pto: %39
      ;
      const int64_t v288 = 0;
      // pto: %39
      ;
      __gm__ int8_t* v289 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %39
      ;
      const int64_t v290 = 1;
      // pto: %39
      ;
      const int64_t v291 = 1;
      // pto: %39
      ;
      const int64_t v292 = 1;
      // pto: %39
      ;
      int64_t v293 = v10 * v9;
      // pto: %39
      ;
      int64_t v294 = v292 * v293;
      // pto: %39
      ;
      pto::Shape<1, 1, 1, 1, -1> v295 = pto::Shape<1, 1, 1, 1, -1>(v290, v291, v292, v10, v12);
      // pto: %39
      ;
      pto::Stride<-1, -1, -1, -1, -1> v296 = pto::Stride<-1, -1, -1, -1, -1>(v291 * v294, v294, v293, v9, v10);
      // pto: %39
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v297 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v289 + (v288 + v221 * v9 + v14 * v10), v295, v296);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v286, v297);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__ssa_v17_pview
      ;
      const int64_t v298 = 0;
      // pto: %out__ssa_v17_pview
      ;
      __gm__ int8_t* v299 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v17_pview
      ;
      const int64_t v300 = 1;
      // pto: %out__ssa_v17_pview
      ;
      const int64_t v301 = 1;
      // pto: %out__ssa_v17_pview
      ;
      const int64_t v302 = 1;
      // pto: %out__ssa_v17_pview
      ;
      int64_t v303 = v10 * v9;
      // pto: %out__ssa_v17_pview
      ;
      int64_t v304 = v302 * v303;
      // pto: %out__ssa_v17_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v305 = pto::Shape<1, 1, 1, 1, -1>(v300, v301, v302, v10, v12);
      // pto: %out__ssa_v17_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v306 = pto::Stride<-1, -1, -1, -1, -1>(v301 * v304, v304, v303, v9, v10);
      // pto: %out__ssa_v17_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v307 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v299 + (v298 + v221 * v9 + v14 * v10), v305, v306);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v307, v286);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v12
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v12
      ;
      uint64_t v309 = (uint64_t) v7;
      TASSIGN(v308, v309);
      // pto: %42
      ;
      const int64_t v310 = 0;
      // pto: %42
      ;
      __gm__ int8_t* v311 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %42
      ;
      const int64_t v312 = 1;
      // pto: %42
      ;
      const int64_t v313 = 1;
      // pto: %42
      ;
      const int64_t v314 = 1;
      // pto: %42
      ;
      int64_t v315 = v10 * v9;
      // pto: %42
      ;
      int64_t v316 = v314 * v315;
      // pto: %42
      ;
      pto::Shape<1, 1, 1, 1, -1> v317 = pto::Shape<1, 1, 1, 1, -1>(v312, v313, v314, v10, v12);
      // pto: %42
      ;
      pto::Stride<-1, -1, -1, -1, -1> v318 = pto::Stride<-1, -1, -1, -1, -1>(v313 * v316, v316, v315, v9, v10);
      // pto: %42
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v319 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v311 + (v310 + v221 * v9 + v15 * v10), v317, v318);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v308, v319);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__ssa_v18_pview
      ;
      const int64_t v320 = 0;
      // pto: %out__ssa_v18_pview
      ;
      __gm__ int8_t* v321 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v18_pview
      ;
      const int64_t v322 = 1;
      // pto: %out__ssa_v18_pview
      ;
      const int64_t v323 = 1;
      // pto: %out__ssa_v18_pview
      ;
      const int64_t v324 = 1;
      // pto: %out__ssa_v18_pview
      ;
      int64_t v325 = v10 * v9;
      // pto: %out__ssa_v18_pview
      ;
      int64_t v326 = v324 * v325;
      // pto: %out__ssa_v18_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v327 = pto::Shape<1, 1, 1, 1, -1>(v322, v323, v324, v10, v12);
      // pto: %out__ssa_v18_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v328 = pto::Stride<-1, -1, -1, -1, -1>(v323 * v326, v326, v325, v9, v10);
      // pto: %out__ssa_v18_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v329 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v321 + (v320 + v221 * v9 + v15 * v10), v327, v328);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v329, v308);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v13
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v13
      ;
      uint64_t v331 = (uint64_t) v7;
      TASSIGN(v330, v331);
      // pto: %45
      ;
      const int64_t v332 = 0;
      // pto: %45
      ;
      __gm__ int8_t* v333 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %45
      ;
      const int64_t v334 = 1;
      // pto: %45
      ;
      const int64_t v335 = 1;
      // pto: %45
      ;
      const int64_t v336 = 1;
      // pto: %45
      ;
      int64_t v337 = v10 * v9;
      // pto: %45
      ;
      int64_t v338 = v336 * v337;
      // pto: %45
      ;
      pto::Shape<1, 1, 1, 1, -1> v339 = pto::Shape<1, 1, 1, 1, -1>(v334, v335, v336, v10, v12);
      // pto: %45
      ;
      pto::Stride<-1, -1, -1, -1, -1> v340 = pto::Stride<-1, -1, -1, -1, -1>(v335 * v338, v338, v337, v9, v10);
      // pto: %45
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v341 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v333 + (v332 + v221 * v9 + v16 * v10), v339, v340);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v330, v341);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__ssa_v19_pview
      ;
      const int64_t v342 = 0;
      // pto: %out__ssa_v19_pview
      ;
      __gm__ int8_t* v343 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v19_pview
      ;
      const int64_t v344 = 1;
      // pto: %out__ssa_v19_pview
      ;
      const int64_t v345 = 1;
      // pto: %out__ssa_v19_pview
      ;
      const int64_t v346 = 1;
      // pto: %out__ssa_v19_pview
      ;
      int64_t v347 = v10 * v9;
      // pto: %out__ssa_v19_pview
      ;
      int64_t v348 = v346 * v347;
      // pto: %out__ssa_v19_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v349 = pto::Shape<1, 1, 1, 1, -1>(v344, v345, v346, v10, v12);
      // pto: %out__ssa_v19_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v350 = pto::Stride<-1, -1, -1, -1, -1>(v345 * v348, v348, v347, v9, v10);
      // pto: %out__ssa_v19_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v351 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v343 + (v342 + v221 * v9 + v16 * v10), v349, v350);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v351, v330);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v14
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v352 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v14
      ;
      uint64_t v353 = (uint64_t) v7;
      TASSIGN(v352, v353);
      // pto: %48
      ;
      const int64_t v354 = 0;
      // pto: %48
      ;
      __gm__ int8_t* v355 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %48
      ;
      const int64_t v356 = 1;
      // pto: %48
      ;
      const int64_t v357 = 1;
      // pto: %48
      ;
      const int64_t v358 = 1;
      // pto: %48
      ;
      int64_t v359 = v10 * v9;
      // pto: %48
      ;
      int64_t v360 = v358 * v359;
      // pto: %48
      ;
      pto::Shape<1, 1, 1, 1, -1> v361 = pto::Shape<1, 1, 1, 1, -1>(v356, v357, v358, v10, v12);
      // pto: %48
      ;
      pto::Stride<-1, -1, -1, -1, -1> v362 = pto::Stride<-1, -1, -1, -1, -1>(v357 * v360, v360, v359, v9, v10);
      // pto: %48
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v363 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v355 + (v354 + v221 * v9 + v17 * v10), v361, v362);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v352, v363);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__ssa_v20_pview
      ;
      const int64_t v364 = 0;
      // pto: %out__ssa_v20_pview
      ;
      __gm__ int8_t* v365 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v20_pview
      ;
      const int64_t v366 = 1;
      // pto: %out__ssa_v20_pview
      ;
      const int64_t v367 = 1;
      // pto: %out__ssa_v20_pview
      ;
      const int64_t v368 = 1;
      // pto: %out__ssa_v20_pview
      ;
      int64_t v369 = v10 * v9;
      // pto: %out__ssa_v20_pview
      ;
      int64_t v370 = v368 * v369;
      // pto: %out__ssa_v20_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v371 = pto::Shape<1, 1, 1, 1, -1>(v366, v367, v368, v10, v12);
      // pto: %out__ssa_v20_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v372 = pto::Stride<-1, -1, -1, -1, -1>(v367 * v370, v370, v369, v9, v10);
      // pto: %out__ssa_v20_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v373 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v365 + (v364 + v221 * v9 + v17 * v10), v371, v372);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v373, v352);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v15
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v374 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v15
      ;
      uint64_t v375 = (uint64_t) v7;
      TASSIGN(v374, v375);
      // pto: %51
      ;
      const int64_t v376 = 0;
      // pto: %51
      ;
      __gm__ int8_t* v377 = PTOAS__GLOBAL_TENSOR_DATA(v26);
      // pto: %51
      ;
      const int64_t v378 = 1;
      // pto: %51
      ;
      const int64_t v379 = 1;
      // pto: %51
      ;
      const int64_t v380 = 1;
      // pto: %51
      ;
      int64_t v381 = v10 * v9;
      // pto: %51
      ;
      int64_t v382 = v380 * v381;
      // pto: %51
      ;
      pto::Shape<1, 1, 1, 1, -1> v383 = pto::Shape<1, 1, 1, 1, -1>(v378, v379, v380, v10, v12);
      // pto: %51
      ;
      pto::Stride<-1, -1, -1, -1, -1> v384 = pto::Stride<-1, -1, -1, -1, -1>(v379 * v382, v382, v381, v9, v10);
      // pto: %51
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v385 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v377 + (v376 + v221 * v9 + v18 * v10), v383, v384);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v374, v385);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__ssa_v21_pview
      ;
      const int64_t v386 = 0;
      // pto: %out__ssa_v21_pview
      ;
      __gm__ int8_t* v387 = PTOAS__GLOBAL_TENSOR_DATA(v34);
      // pto: %out__ssa_v21_pview
      ;
      const int64_t v388 = 1;
      // pto: %out__ssa_v21_pview
      ;
      const int64_t v389 = 1;
      // pto: %out__ssa_v21_pview
      ;
      const int64_t v390 = 1;
      // pto: %out__ssa_v21_pview
      ;
      int64_t v391 = v10 * v9;
      // pto: %out__ssa_v21_pview
      ;
      int64_t v392 = v390 * v391;
      // pto: %out__ssa_v21_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v393 = pto::Shape<1, 1, 1, 1, -1>(v388, v389, v390, v10, v12);
      // pto: %out__ssa_v21_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v394 = pto::Stride<-1, -1, -1, -1, -1>(v389 * v392, v392, v391, v9, v10);
      // pto: %out__ssa_v21_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v395 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v387 + (v386 + v221 * v9 + v18 * v10), v393, v394);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v395, v374);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
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