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
  // pto: %data__ssa_v0_view
  const int64_t v15 = 1;
  // pto: %data__ssa_v0_view
  const int64_t v16 = 1;
  // pto: %data__ssa_v0_view
  const int64_t v17 = 1;
  // pto: %data__ssa_v0_view
  int64_t v18 = v8 * v9;
  // pto: %data__ssa_v0_view
  int64_t v19 = v17 * v18;
  // pto: %data__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v20 = pto::Shape<1, 1, 1, -1, -1>(v15, v16, v17, v8, v9);
  // pto: %data__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v21 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v19, v19, v18, v9, v10);
  // pto: %data__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v22 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v20, v21);
  // pto: %out__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v25 = 1;
  // pto: %out__ssa_v0_view
  int64_t v26 = v8 * v9;
  // pto: %out__ssa_v0_view
  int64_t v27 = v25 * v26;
  // pto: %out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v28 = pto::Shape<1, 1, 1, -1, -1>(v23, v24, v25, v8, v9);
  // pto: %out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v29 = pto::Stride<-1, -1, -1, -1, -1>(v24 * v27, v27, v26, v9, v10);
  // pto: %out__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v30 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v28, v29);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
  for (int64_t v31 = v11; v31 < v8; v31 += v10) {
    // pto: %n_rows_i32__tile
    ;
    int32_t v32 = (v2)[v31];
    (v4)[v31] = v32;
    // pto: %2
    ;
    int64_t v33 = (int64_t) v32;
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (int64_t v34 = v11; v34 < v33; v34 += v10) {
      // pto: %3
      ;
      int64_t v35 = (int64_t) ((uint64_t) v31 + (uint64_t) v34);
      // pto: %tile__ssa_v0
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v36 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v0
      ;
      uint64_t v37 = (uint64_t) v7;
      TASSIGN(v36, v37);
      // pto: %4
      ;
      int64_t v38 = v35 < v11 ? v11 : v35;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v39 = 0;
      // pto: %data__ssa_v0_pview
      ;
      __gm__ int8_t* v40 = PTOAS__GLOBAL_TENSOR_DATA(v22);
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v41 = 1;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v42 = 1;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v43 = 1;
      // pto: %data__ssa_v0_pview
      ;
      int64_t v44 = v10 * v9;
      // pto: %data__ssa_v0_pview
      ;
      int64_t v45 = v43 * v44;
      // pto: %data__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v46 = pto::Shape<1, 1, 1, 1, -1>(v41, v42, v43, v10, v12);
      // pto: %data__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v47 = pto::Stride<-1, -1, -1, -1, -1>(v42 * v45, v45, v44, v9, v10);
      // pto: %data__ssa_v0_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v48 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v40 + (v39 + v38 * v9 + v11 * v10), v46, v47);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v36, v48);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__iter_v3_pview
      ;
      const int64_t v49 = 0;
      // pto: %out__iter_v3_pview
      ;
      __gm__ int8_t* v50 = PTOAS__GLOBAL_TENSOR_DATA(v30);
      // pto: %out__iter_v3_pview
      ;
      const int64_t v51 = 1;
      // pto: %out__iter_v3_pview
      ;
      const int64_t v52 = 1;
      // pto: %out__iter_v3_pview
      ;
      const int64_t v53 = 1;
      // pto: %out__iter_v3_pview
      ;
      int64_t v54 = v10 * v9;
      // pto: %out__iter_v3_pview
      ;
      int64_t v55 = v53 * v54;
      // pto: %out__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v56 = pto::Shape<1, 1, 1, 1, -1>(v51, v52, v53, v10, v12);
      // pto: %out__iter_v3_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v57 = pto::Stride<-1, -1, -1, -1, -1>(v52 * v55, v55, v54, v9, v10);
      // pto: %out__iter_v3_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v58 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v50 + (v49 + v38 * v9 + v11 * v10), v56, v57);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v58, v36);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v1
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v1
      ;
      uint64_t v60 = (uint64_t) v7;
      TASSIGN(v59, v60);
      // pto: %7
      ;
      const int64_t v61 = 0;
      // pto: %7
      ;
      __gm__ int8_t* v62 = PTOAS__GLOBAL_TENSOR_DATA(v22);
      // pto: %7
      ;
      const int64_t v63 = 1;
      // pto: %7
      ;
      const int64_t v64 = 1;
      // pto: %7
      ;
      const int64_t v65 = 1;
      // pto: %7
      ;
      int64_t v66 = v10 * v9;
      // pto: %7
      ;
      int64_t v67 = v65 * v66;
      // pto: %7
      ;
      pto::Shape<1, 1, 1, 1, -1> v68 = pto::Shape<1, 1, 1, 1, -1>(v63, v64, v65, v10, v12);
      // pto: %7
      ;
      pto::Stride<-1, -1, -1, -1, -1> v69 = pto::Stride<-1, -1, -1, -1, -1>(v64 * v67, v67, v66, v9, v10);
      // pto: %7
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v70 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v62 + (v61 + v38 * v9 + v12 * v10), v68, v69);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v59, v70);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v71 = 0;
      // pto: %out__ssa_v5_pview
      ;
      __gm__ int8_t* v72 = PTOAS__GLOBAL_TENSOR_DATA(v30);
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v73 = 1;
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v74 = 1;
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v75 = 1;
      // pto: %out__ssa_v5_pview
      ;
      int64_t v76 = v10 * v9;
      // pto: %out__ssa_v5_pview
      ;
      int64_t v77 = v75 * v76;
      // pto: %out__ssa_v5_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v78 = pto::Shape<1, 1, 1, 1, -1>(v73, v74, v75, v10, v12);
      // pto: %out__ssa_v5_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v79 = pto::Stride<-1, -1, -1, -1, -1>(v74 * v77, v77, v76, v9, v10);
      // pto: %out__ssa_v5_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v80 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v72 + (v71 + v38 * v9 + v12 * v10), v78, v79);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      TSTORE(v80, v59);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %tile__ssa_v2
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v2
      ;
      uint64_t v82 = (uint64_t) v7;
      TASSIGN(v81, v82);
      // pto: %10
      ;
      const int64_t v83 = 0;
      // pto: %10
      ;
      __gm__ int8_t* v84 = PTOAS__GLOBAL_TENSOR_DATA(v22);
      // pto: %10
      ;
      const int64_t v85 = 1;
      // pto: %10
      ;
      const int64_t v86 = 1;
      // pto: %10
      ;
      const int64_t v87 = 1;
      // pto: %10
      ;
      int64_t v88 = v10 * v9;
      // pto: %10
      ;
      int64_t v89 = v87 * v88;
      // pto: %10
      ;
      pto::Shape<1, 1, 1, 1, -1> v90 = pto::Shape<1, 1, 1, 1, -1>(v85, v86, v87, v10, v12);
      // pto: %10
      ;
      pto::Stride<-1, -1, -1, -1, -1> v91 = pto::Stride<-1, -1, -1, -1, -1>(v86 * v89, v89, v88, v9, v10);
      // pto: %10
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v92 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v84 + (v83 + v38 * v9 + v13 * v10), v90, v91);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      TLOAD(v81, v92);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v93 = 0;
      // pto: %out__ssa_v6_pview
      ;
      __gm__ int8_t* v94 = PTOAS__GLOBAL_TENSOR_DATA(v30);
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v95 = 1;
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v96 = 1;
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v97 = 1;
      // pto: %out__ssa_v6_pview
      ;
      int64_t v98 = v10 * v9;
      // pto: %out__ssa_v6_pview
      ;
      int64_t v99 = v97 * v98;
      // pto: %out__ssa_v6_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v100 = pto::Shape<1, 1, 1, 1, -1>(v95, v96, v97, v10, v12);
      // pto: %out__ssa_v6_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v101 = pto::Stride<-1, -1, -1, -1, -1>(v96 * v99, v99, v98, v9, v10);
      // pto: %out__ssa_v6_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v102 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v94 + (v93 + v38 * v9 + v13 * v10), v100, v101);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      TSTORE(v102, v81);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %tile__ssa_v3
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v3
      ;
      uint64_t v104 = (uint64_t) v7;
      TASSIGN(v103, v104);
      // pto: %13
      ;
      const int64_t v105 = 0;
      // pto: %13
      ;
      __gm__ int8_t* v106 = PTOAS__GLOBAL_TENSOR_DATA(v22);
      // pto: %13
      ;
      const int64_t v107 = 1;
      // pto: %13
      ;
      const int64_t v108 = 1;
      // pto: %13
      ;
      const int64_t v109 = 1;
      // pto: %13
      ;
      int64_t v110 = v10 * v9;
      // pto: %13
      ;
      int64_t v111 = v109 * v110;
      // pto: %13
      ;
      pto::Shape<1, 1, 1, 1, -1> v112 = pto::Shape<1, 1, 1, 1, -1>(v107, v108, v109, v10, v12);
      // pto: %13
      ;
      pto::Stride<-1, -1, -1, -1, -1> v113 = pto::Stride<-1, -1, -1, -1, -1>(v108 * v111, v111, v110, v9, v10);
      // pto: %13
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v114 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v106 + (v105 + v38 * v9 + v14 * v10), v112, v113);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      TLOAD(v103, v114);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
      // pto: %out__ssa_v7_pview
      ;
      const int64_t v115 = 0;
      // pto: %out__ssa_v7_pview
      ;
      __gm__ int8_t* v116 = PTOAS__GLOBAL_TENSOR_DATA(v30);
      // pto: %out__ssa_v7_pview
      ;
      const int64_t v117 = 1;
      // pto: %out__ssa_v7_pview
      ;
      const int64_t v118 = 1;
      // pto: %out__ssa_v7_pview
      ;
      const int64_t v119 = 1;
      // pto: %out__ssa_v7_pview
      ;
      int64_t v120 = v10 * v9;
      // pto: %out__ssa_v7_pview
      ;
      int64_t v121 = v119 * v120;
      // pto: %out__ssa_v7_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v122 = pto::Shape<1, 1, 1, 1, -1>(v117, v118, v119, v10, v12);
      // pto: %out__ssa_v7_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v123 = pto::Stride<-1, -1, -1, -1, -1>(v118 * v121, v121, v120, v9, v10);
      // pto: %out__ssa_v7_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v124 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v116 + (v115 + v38 * v9 + v14 * v10), v122, v123);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
      TSTORE(v124, v103);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    for (int64_t v125 = v33; v125 < v10; v125 += v10) {
      // pto: %16
      ;
      int64_t v126 = (int64_t) ((uint64_t) v31 + (uint64_t) v125);
      // pto: %tile__ssa_v4
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v4
      ;
      uint64_t v128 = (uint64_t) v7;
      TASSIGN(v127, v128);
      // pto: %17
      ;
      int64_t v129 = v126 < v11 ? v11 : v126;
      // pto: %18
      ;
      const int64_t v130 = 0;
      // pto: %18
      ;
      __gm__ int8_t* v131 = PTOAS__GLOBAL_TENSOR_DATA(v22);
      // pto: %18
      ;
      const int64_t v132 = 1;
      // pto: %18
      ;
      const int64_t v133 = 1;
      // pto: %18
      ;
      const int64_t v134 = 1;
      // pto: %18
      ;
      int64_t v135 = v10 * v9;
      // pto: %18
      ;
      int64_t v136 = v134 * v135;
      // pto: %18
      ;
      pto::Shape<1, 1, 1, 1, -1> v137 = pto::Shape<1, 1, 1, 1, -1>(v132, v133, v134, v10, v12);
      // pto: %18
      ;
      pto::Stride<-1, -1, -1, -1, -1> v138 = pto::Stride<-1, -1, -1, -1, -1>(v133 * v136, v136, v135, v9, v10);
      // pto: %18
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v139 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v131 + (v130 + v129 * v9 + v11 * v10), v137, v138);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
      TLOAD(v127, v139);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
      // pto: %out__iter_v9_pview
      ;
      const int64_t v140 = 0;
      // pto: %out__iter_v9_pview
      ;
      __gm__ int8_t* v141 = PTOAS__GLOBAL_TENSOR_DATA(v30);
      // pto: %out__iter_v9_pview
      ;
      const int64_t v142 = 1;
      // pto: %out__iter_v9_pview
      ;
      const int64_t v143 = 1;
      // pto: %out__iter_v9_pview
      ;
      const int64_t v144 = 1;
      // pto: %out__iter_v9_pview
      ;
      int64_t v145 = v10 * v9;
      // pto: %out__iter_v9_pview
      ;
      int64_t v146 = v144 * v145;
      // pto: %out__iter_v9_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v147 = pto::Shape<1, 1, 1, 1, -1>(v142, v143, v144, v10, v12);
      // pto: %out__iter_v9_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v148 = pto::Stride<-1, -1, -1, -1, -1>(v143 * v146, v146, v145, v9, v10);
      // pto: %out__iter_v9_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v149 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v141 + (v140 + v129 * v9 + v11 * v10), v147, v148);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
      TSTORE(v149, v127);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
      // pto: %tile__ssa_v5
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v150 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v5
      ;
      uint64_t v151 = (uint64_t) v7;
      TASSIGN(v150, v151);
      // pto: %21
      ;
      const int64_t v152 = 0;
      // pto: %21
      ;
      __gm__ int8_t* v153 = PTOAS__GLOBAL_TENSOR_DATA(v22);
      // pto: %21
      ;
      const int64_t v154 = 1;
      // pto: %21
      ;
      const int64_t v155 = 1;
      // pto: %21
      ;
      const int64_t v156 = 1;
      // pto: %21
      ;
      int64_t v157 = v10 * v9;
      // pto: %21
      ;
      int64_t v158 = v156 * v157;
      // pto: %21
      ;
      pto::Shape<1, 1, 1, 1, -1> v159 = pto::Shape<1, 1, 1, 1, -1>(v154, v155, v156, v10, v12);
      // pto: %21
      ;
      pto::Stride<-1, -1, -1, -1, -1> v160 = pto::Stride<-1, -1, -1, -1, -1>(v155 * v158, v158, v157, v9, v10);
      // pto: %21
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v161 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v153 + (v152 + v129 * v9 + v12 * v10), v159, v160);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
      TLOAD(v150, v161);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID5);
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v162 = 0;
      // pto: %out__ssa_v11_pview
      ;
      __gm__ int8_t* v163 = PTOAS__GLOBAL_TENSOR_DATA(v30);
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v164 = 1;
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v165 = 1;
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v166 = 1;
      // pto: %out__ssa_v11_pview
      ;
      int64_t v167 = v10 * v9;
      // pto: %out__ssa_v11_pview
      ;
      int64_t v168 = v166 * v167;
      // pto: %out__ssa_v11_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v169 = pto::Shape<1, 1, 1, 1, -1>(v164, v165, v166, v10, v12);
      // pto: %out__ssa_v11_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v170 = pto::Stride<-1, -1, -1, -1, -1>(v165 * v168, v168, v167, v9, v10);
      // pto: %out__ssa_v11_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v171 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v163 + (v162 + v129 * v9 + v12 * v10), v169, v170);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID5);
      TSTORE(v171, v150);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v6
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v172 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v6
      ;
      uint64_t v173 = (uint64_t) v7;
      TASSIGN(v172, v173);
      // pto: %24
      ;
      const int64_t v174 = 0;
      // pto: %24
      ;
      __gm__ int8_t* v175 = PTOAS__GLOBAL_TENSOR_DATA(v22);
      // pto: %24
      ;
      const int64_t v176 = 1;
      // pto: %24
      ;
      const int64_t v177 = 1;
      // pto: %24
      ;
      const int64_t v178 = 1;
      // pto: %24
      ;
      int64_t v179 = v10 * v9;
      // pto: %24
      ;
      int64_t v180 = v178 * v179;
      // pto: %24
      ;
      pto::Shape<1, 1, 1, 1, -1> v181 = pto::Shape<1, 1, 1, 1, -1>(v176, v177, v178, v10, v12);
      // pto: %24
      ;
      pto::Stride<-1, -1, -1, -1, -1> v182 = pto::Stride<-1, -1, -1, -1, -1>(v177 * v180, v180, v179, v9, v10);
      // pto: %24
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v183 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v175 + (v174 + v129 * v9 + v13 * v10), v181, v182);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v172, v183);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
      // pto: %out__ssa_v12_pview
      ;
      const int64_t v184 = 0;
      // pto: %out__ssa_v12_pview
      ;
      __gm__ int8_t* v185 = PTOAS__GLOBAL_TENSOR_DATA(v30);
      // pto: %out__ssa_v12_pview
      ;
      const int64_t v186 = 1;
      // pto: %out__ssa_v12_pview
      ;
      const int64_t v187 = 1;
      // pto: %out__ssa_v12_pview
      ;
      const int64_t v188 = 1;
      // pto: %out__ssa_v12_pview
      ;
      int64_t v189 = v10 * v9;
      // pto: %out__ssa_v12_pview
      ;
      int64_t v190 = v188 * v189;
      // pto: %out__ssa_v12_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v191 = pto::Shape<1, 1, 1, 1, -1>(v186, v187, v188, v10, v12);
      // pto: %out__ssa_v12_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v192 = pto::Stride<-1, -1, -1, -1, -1>(v187 * v190, v190, v189, v9, v10);
      // pto: %out__ssa_v12_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v193 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v185 + (v184 + v129 * v9 + v13 * v10), v191, v192);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
      TSTORE(v193, v172);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v7
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v194 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v7
      ;
      uint64_t v195 = (uint64_t) v7;
      TASSIGN(v194, v195);
      // pto: %27
      ;
      const int64_t v196 = 0;
      // pto: %27
      ;
      __gm__ int8_t* v197 = PTOAS__GLOBAL_TENSOR_DATA(v22);
      // pto: %27
      ;
      const int64_t v198 = 1;
      // pto: %27
      ;
      const int64_t v199 = 1;
      // pto: %27
      ;
      const int64_t v200 = 1;
      // pto: %27
      ;
      int64_t v201 = v10 * v9;
      // pto: %27
      ;
      int64_t v202 = v200 * v201;
      // pto: %27
      ;
      pto::Shape<1, 1, 1, 1, -1> v203 = pto::Shape<1, 1, 1, 1, -1>(v198, v199, v200, v10, v12);
      // pto: %27
      ;
      pto::Stride<-1, -1, -1, -1, -1> v204 = pto::Stride<-1, -1, -1, -1, -1>(v199 * v202, v202, v201, v9, v10);
      // pto: %27
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v205 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v197 + (v196 + v129 * v9 + v14 * v10), v203, v204);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v194, v205);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
      // pto: %out__ssa_v13_pview
      ;
      const int64_t v206 = 0;
      // pto: %out__ssa_v13_pview
      ;
      __gm__ int8_t* v207 = PTOAS__GLOBAL_TENSOR_DATA(v30);
      // pto: %out__ssa_v13_pview
      ;
      const int64_t v208 = 1;
      // pto: %out__ssa_v13_pview
      ;
      const int64_t v209 = 1;
      // pto: %out__ssa_v13_pview
      ;
      const int64_t v210 = 1;
      // pto: %out__ssa_v13_pview
      ;
      int64_t v211 = v10 * v9;
      // pto: %out__ssa_v13_pview
      ;
      int64_t v212 = v210 * v211;
      // pto: %out__ssa_v13_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v213 = pto::Shape<1, 1, 1, 1, -1>(v208, v209, v210, v10, v12);
      // pto: %out__ssa_v13_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v214 = pto::Stride<-1, -1, -1, -1, -1>(v209 * v212, v212, v211, v9, v10);
      // pto: %out__ssa_v13_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v215 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v207 + (v206 + v129 * v9 + v14 * v10), v213, v214);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
      TSTORE(v215, v194);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
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