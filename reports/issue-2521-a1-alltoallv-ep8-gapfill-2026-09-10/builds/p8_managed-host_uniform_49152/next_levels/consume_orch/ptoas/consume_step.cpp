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
  // pto: %c49152_index
  const int64_t v9 = 49152;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c0_index
  const int64_t v11 = 0;
  // pto: %c16384_index
  const int64_t v12 = 16384;
  // pto: %c32768_index
  const int64_t v13 = 32768;
  // pto: %data__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %data__ssa_v0_view
  const int64_t v15 = 1;
  // pto: %data__ssa_v0_view
  const int64_t v16 = 1;
  // pto: %data__ssa_v0_view
  int64_t v17 = v8 * v9;
  // pto: %data__ssa_v0_view
  int64_t v18 = v16 * v17;
  // pto: %data__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v19 = pto::Shape<1, 1, 1, -1, -1>(v14, v15, v16, v8, v9);
  // pto: %data__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v20 = pto::Stride<-1, -1, -1, -1, -1>(v15 * v18, v18, v17, v9, v10);
  // pto: %data__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v21 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v19, v20);
  // pto: %out__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v23 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v24 = 1;
  // pto: %out__ssa_v0_view
  int64_t v25 = v8 * v9;
  // pto: %out__ssa_v0_view
  int64_t v26 = v24 * v25;
  // pto: %out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v27 = pto::Shape<1, 1, 1, -1, -1>(v22, v23, v24, v8, v9);
  // pto: %out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v28 = pto::Stride<-1, -1, -1, -1, -1>(v23 * v26, v26, v25, v9, v10);
  // pto: %out__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v29 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v27, v28);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
  for (int64_t v30 = v11; v30 < v8; v30 += v10) {
    // pto: %n_rows_i32__tile
    ;
    int32_t v31 = (v2)[v30];
    (v4)[v30] = v31;
    // pto: %2
    ;
    int64_t v32 = (int64_t) v31;
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (int64_t v33 = v11; v33 < v32; v33 += v10) {
      // pto: %3
      ;
      int64_t v34 = (int64_t) ((uint64_t) v30 + (uint64_t) v33);
      // pto: %tile__ssa_v0
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v35 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v0
      ;
      uint64_t v36 = (uint64_t) v7;
      TASSIGN(v35, v36);
      // pto: %4
      ;
      int64_t v37 = v34 < v11 ? v11 : v34;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v38 = 0;
      // pto: %data__ssa_v0_pview
      ;
      __gm__ int8_t* v39 = PTOAS__GLOBAL_TENSOR_DATA(v21);
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v40 = 1;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v41 = 1;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v42 = 1;
      // pto: %data__ssa_v0_pview
      ;
      int64_t v43 = v10 * v9;
      // pto: %data__ssa_v0_pview
      ;
      int64_t v44 = v42 * v43;
      // pto: %data__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v45 = pto::Shape<1, 1, 1, 1, -1>(v40, v41, v42, v10, v12);
      // pto: %data__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v46 = pto::Stride<-1, -1, -1, -1, -1>(v41 * v44, v44, v43, v9, v10);
      // pto: %data__ssa_v0_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v47 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v39 + (v38 + v37 * v9 + v11 * v10), v45, v46);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v35, v47);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__iter_v3_pview
      ;
      const int64_t v48 = 0;
      // pto: %out__iter_v3_pview
      ;
      __gm__ int8_t* v49 = PTOAS__GLOBAL_TENSOR_DATA(v29);
      // pto: %out__iter_v3_pview
      ;
      const int64_t v50 = 1;
      // pto: %out__iter_v3_pview
      ;
      const int64_t v51 = 1;
      // pto: %out__iter_v3_pview
      ;
      const int64_t v52 = 1;
      // pto: %out__iter_v3_pview
      ;
      int64_t v53 = v10 * v9;
      // pto: %out__iter_v3_pview
      ;
      int64_t v54 = v52 * v53;
      // pto: %out__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v55 = pto::Shape<1, 1, 1, 1, -1>(v50, v51, v52, v10, v12);
      // pto: %out__iter_v3_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v56 = pto::Stride<-1, -1, -1, -1, -1>(v51 * v54, v54, v53, v9, v10);
      // pto: %out__iter_v3_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v57 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v49 + (v48 + v37 * v9 + v11 * v10), v55, v56);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v57, v35);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %tile__ssa_v1
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v58 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v1
      ;
      uint64_t v59 = (uint64_t) v7;
      TASSIGN(v58, v59);
      // pto: %7
      ;
      const int64_t v60 = 0;
      // pto: %7
      ;
      __gm__ int8_t* v61 = PTOAS__GLOBAL_TENSOR_DATA(v21);
      // pto: %7
      ;
      const int64_t v62 = 1;
      // pto: %7
      ;
      const int64_t v63 = 1;
      // pto: %7
      ;
      const int64_t v64 = 1;
      // pto: %7
      ;
      int64_t v65 = v10 * v9;
      // pto: %7
      ;
      int64_t v66 = v64 * v65;
      // pto: %7
      ;
      pto::Shape<1, 1, 1, 1, -1> v67 = pto::Shape<1, 1, 1, 1, -1>(v62, v63, v64, v10, v12);
      // pto: %7
      ;
      pto::Stride<-1, -1, -1, -1, -1> v68 = pto::Stride<-1, -1, -1, -1, -1>(v63 * v66, v66, v65, v9, v10);
      // pto: %7
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v69 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v61 + (v60 + v37 * v9 + v12 * v10), v67, v68);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v58, v69);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v70 = 0;
      // pto: %out__ssa_v5_pview
      ;
      __gm__ int8_t* v71 = PTOAS__GLOBAL_TENSOR_DATA(v29);
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v72 = 1;
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v73 = 1;
      // pto: %out__ssa_v5_pview
      ;
      const int64_t v74 = 1;
      // pto: %out__ssa_v5_pview
      ;
      int64_t v75 = v10 * v9;
      // pto: %out__ssa_v5_pview
      ;
      int64_t v76 = v74 * v75;
      // pto: %out__ssa_v5_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v77 = pto::Shape<1, 1, 1, 1, -1>(v72, v73, v74, v10, v12);
      // pto: %out__ssa_v5_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v78 = pto::Stride<-1, -1, -1, -1, -1>(v73 * v76, v76, v75, v9, v10);
      // pto: %out__ssa_v5_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v79 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v71 + (v70 + v37 * v9 + v12 * v10), v77, v78);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      TSTORE(v79, v58);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %tile__ssa_v2
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v2
      ;
      uint64_t v81 = (uint64_t) v7;
      TASSIGN(v80, v81);
      // pto: %10
      ;
      const int64_t v82 = 0;
      // pto: %10
      ;
      __gm__ int8_t* v83 = PTOAS__GLOBAL_TENSOR_DATA(v21);
      // pto: %10
      ;
      const int64_t v84 = 1;
      // pto: %10
      ;
      const int64_t v85 = 1;
      // pto: %10
      ;
      const int64_t v86 = 1;
      // pto: %10
      ;
      int64_t v87 = v10 * v9;
      // pto: %10
      ;
      int64_t v88 = v86 * v87;
      // pto: %10
      ;
      pto::Shape<1, 1, 1, 1, -1> v89 = pto::Shape<1, 1, 1, 1, -1>(v84, v85, v86, v10, v12);
      // pto: %10
      ;
      pto::Stride<-1, -1, -1, -1, -1> v90 = pto::Stride<-1, -1, -1, -1, -1>(v85 * v88, v88, v87, v9, v10);
      // pto: %10
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v91 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v83 + (v82 + v37 * v9 + v13 * v10), v89, v90);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      TLOAD(v80, v91);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v92 = 0;
      // pto: %out__ssa_v6_pview
      ;
      __gm__ int8_t* v93 = PTOAS__GLOBAL_TENSOR_DATA(v29);
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v94 = 1;
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v95 = 1;
      // pto: %out__ssa_v6_pview
      ;
      const int64_t v96 = 1;
      // pto: %out__ssa_v6_pview
      ;
      int64_t v97 = v10 * v9;
      // pto: %out__ssa_v6_pview
      ;
      int64_t v98 = v96 * v97;
      // pto: %out__ssa_v6_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v99 = pto::Shape<1, 1, 1, 1, -1>(v94, v95, v96, v10, v12);
      // pto: %out__ssa_v6_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v100 = pto::Stride<-1, -1, -1, -1, -1>(v95 * v98, v98, v97, v9, v10);
      // pto: %out__ssa_v6_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v101 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v93 + (v92 + v37 * v9 + v13 * v10), v99, v100);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      TSTORE(v101, v80);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    for (int64_t v102 = v32; v102 < v10; v102 += v10) {
      // pto: %13
      ;
      int64_t v103 = (int64_t) ((uint64_t) v30 + (uint64_t) v102);
      // pto: %tile__ssa_v3
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v3
      ;
      uint64_t v105 = (uint64_t) v7;
      TASSIGN(v104, v105);
      // pto: %14
      ;
      int64_t v106 = v103 < v11 ? v11 : v103;
      // pto: %15
      ;
      const int64_t v107 = 0;
      // pto: %15
      ;
      __gm__ int8_t* v108 = PTOAS__GLOBAL_TENSOR_DATA(v21);
      // pto: %15
      ;
      const int64_t v109 = 1;
      // pto: %15
      ;
      const int64_t v110 = 1;
      // pto: %15
      ;
      const int64_t v111 = 1;
      // pto: %15
      ;
      int64_t v112 = v10 * v9;
      // pto: %15
      ;
      int64_t v113 = v111 * v112;
      // pto: %15
      ;
      pto::Shape<1, 1, 1, 1, -1> v114 = pto::Shape<1, 1, 1, 1, -1>(v109, v110, v111, v10, v12);
      // pto: %15
      ;
      pto::Stride<-1, -1, -1, -1, -1> v115 = pto::Stride<-1, -1, -1, -1, -1>(v110 * v113, v113, v112, v9, v10);
      // pto: %15
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v116 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v108 + (v107 + v106 * v9 + v11 * v10), v114, v115);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      TLOAD(v104, v116);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
      // pto: %out__iter_v8_pview
      ;
      const int64_t v117 = 0;
      // pto: %out__iter_v8_pview
      ;
      __gm__ int8_t* v118 = PTOAS__GLOBAL_TENSOR_DATA(v29);
      // pto: %out__iter_v8_pview
      ;
      const int64_t v119 = 1;
      // pto: %out__iter_v8_pview
      ;
      const int64_t v120 = 1;
      // pto: %out__iter_v8_pview
      ;
      const int64_t v121 = 1;
      // pto: %out__iter_v8_pview
      ;
      int64_t v122 = v10 * v9;
      // pto: %out__iter_v8_pview
      ;
      int64_t v123 = v121 * v122;
      // pto: %out__iter_v8_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v124 = pto::Shape<1, 1, 1, 1, -1>(v119, v120, v121, v10, v12);
      // pto: %out__iter_v8_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v125 = pto::Stride<-1, -1, -1, -1, -1>(v120 * v123, v123, v122, v9, v10);
      // pto: %out__iter_v8_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v126 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v118 + (v117 + v106 * v9 + v11 * v10), v124, v125);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
      TSTORE(v126, v104);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
      // pto: %tile__ssa_v4
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v4
      ;
      uint64_t v128 = (uint64_t) v7;
      TASSIGN(v127, v128);
      // pto: %18
      ;
      const int64_t v129 = 0;
      // pto: %18
      ;
      __gm__ int8_t* v130 = PTOAS__GLOBAL_TENSOR_DATA(v21);
      // pto: %18
      ;
      const int64_t v131 = 1;
      // pto: %18
      ;
      const int64_t v132 = 1;
      // pto: %18
      ;
      const int64_t v133 = 1;
      // pto: %18
      ;
      int64_t v134 = v10 * v9;
      // pto: %18
      ;
      int64_t v135 = v133 * v134;
      // pto: %18
      ;
      pto::Shape<1, 1, 1, 1, -1> v136 = pto::Shape<1, 1, 1, 1, -1>(v131, v132, v133, v10, v12);
      // pto: %18
      ;
      pto::Stride<-1, -1, -1, -1, -1> v137 = pto::Stride<-1, -1, -1, -1, -1>(v132 * v135, v135, v134, v9, v10);
      // pto: %18
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v138 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v130 + (v129 + v106 * v9 + v12 * v10), v136, v137);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
      TLOAD(v127, v138);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
      // pto: %out__ssa_v10_pview
      ;
      const int64_t v139 = 0;
      // pto: %out__ssa_v10_pview
      ;
      __gm__ int8_t* v140 = PTOAS__GLOBAL_TENSOR_DATA(v29);
      // pto: %out__ssa_v10_pview
      ;
      const int64_t v141 = 1;
      // pto: %out__ssa_v10_pview
      ;
      const int64_t v142 = 1;
      // pto: %out__ssa_v10_pview
      ;
      const int64_t v143 = 1;
      // pto: %out__ssa_v10_pview
      ;
      int64_t v144 = v10 * v9;
      // pto: %out__ssa_v10_pview
      ;
      int64_t v145 = v143 * v144;
      // pto: %out__ssa_v10_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v146 = pto::Shape<1, 1, 1, 1, -1>(v141, v142, v143, v10, v12);
      // pto: %out__ssa_v10_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v147 = pto::Stride<-1, -1, -1, -1, -1>(v142 * v145, v145, v144, v9, v10);
      // pto: %out__ssa_v10_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v148 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v140 + (v139 + v106 * v9 + v12 * v10), v146, v147);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
      TSTORE(v148, v127);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
      // pto: %tile__ssa_v5
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v149 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v5
      ;
      uint64_t v150 = (uint64_t) v7;
      TASSIGN(v149, v150);
      // pto: %21
      ;
      const int64_t v151 = 0;
      // pto: %21
      ;
      __gm__ int8_t* v152 = PTOAS__GLOBAL_TENSOR_DATA(v21);
      // pto: %21
      ;
      const int64_t v153 = 1;
      // pto: %21
      ;
      const int64_t v154 = 1;
      // pto: %21
      ;
      const int64_t v155 = 1;
      // pto: %21
      ;
      int64_t v156 = v10 * v9;
      // pto: %21
      ;
      int64_t v157 = v155 * v156;
      // pto: %21
      ;
      pto::Shape<1, 1, 1, 1, -1> v158 = pto::Shape<1, 1, 1, 1, -1>(v153, v154, v155, v10, v12);
      // pto: %21
      ;
      pto::Stride<-1, -1, -1, -1, -1> v159 = pto::Stride<-1, -1, -1, -1, -1>(v154 * v157, v157, v156, v9, v10);
      // pto: %21
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v160 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v152 + (v151 + v106 * v9 + v13 * v10), v158, v159);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
      TLOAD(v149, v160);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID5);
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v161 = 0;
      // pto: %out__ssa_v11_pview
      ;
      __gm__ int8_t* v162 = PTOAS__GLOBAL_TENSOR_DATA(v29);
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v163 = 1;
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v164 = 1;
      // pto: %out__ssa_v11_pview
      ;
      const int64_t v165 = 1;
      // pto: %out__ssa_v11_pview
      ;
      int64_t v166 = v10 * v9;
      // pto: %out__ssa_v11_pview
      ;
      int64_t v167 = v165 * v166;
      // pto: %out__ssa_v11_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v168 = pto::Shape<1, 1, 1, 1, -1>(v163, v164, v165, v10, v12);
      // pto: %out__ssa_v11_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v169 = pto::Stride<-1, -1, -1, -1, -1>(v164 * v167, v167, v166, v9, v10);
      // pto: %out__ssa_v11_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v170 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v162 + (v161 + v106 * v9 + v13 * v10), v168, v169);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID5);
      TSTORE(v170, v149);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
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