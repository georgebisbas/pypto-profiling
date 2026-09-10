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
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
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
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v58 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v13);
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
      pto::Shape<1, 1, 1, 1, -1> v67 = pto::Shape<1, 1, 1, 1, -1>(v62, v63, v64, v10, v13);
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
      pto::Shape<1, 1, 1, 1, -1> v77 = pto::Shape<1, 1, 1, 1, -1>(v72, v73, v74, v10, v13);
      // pto: %out__ssa_v5_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v78 = pto::Stride<-1, -1, -1, -1, -1>(v73 * v76, v76, v75, v9, v10);
      // pto: %out__ssa_v5_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v79 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v71 + (v70 + v37 * v9 + v12 * v10), v77, v78);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      TSTORE(v79, v58);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    for (int64_t v80 = v32; v80 < v10; v80 += v10) {
      // pto: %10
      ;
      int64_t v81 = (int64_t) ((uint64_t) v30 + (uint64_t) v80);
      // pto: %tile__ssa_v2
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v12);
      // pto: %tile__ssa_v2
      ;
      uint64_t v83 = (uint64_t) v7;
      TASSIGN(v82, v83);
      // pto: %11
      ;
      int64_t v84 = v81 < v11 ? v11 : v81;
      // pto: %12
      ;
      const int64_t v85 = 0;
      // pto: %12
      ;
      __gm__ int8_t* v86 = PTOAS__GLOBAL_TENSOR_DATA(v21);
      // pto: %12
      ;
      const int64_t v87 = 1;
      // pto: %12
      ;
      const int64_t v88 = 1;
      // pto: %12
      ;
      const int64_t v89 = 1;
      // pto: %12
      ;
      int64_t v90 = v10 * v9;
      // pto: %12
      ;
      int64_t v91 = v89 * v90;
      // pto: %12
      ;
      pto::Shape<1, 1, 1, 1, -1> v92 = pto::Shape<1, 1, 1, 1, -1>(v87, v88, v89, v10, v12);
      // pto: %12
      ;
      pto::Stride<-1, -1, -1, -1, -1> v93 = pto::Stride<-1, -1, -1, -1, -1>(v88 * v91, v91, v90, v9, v10);
      // pto: %12
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v94 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v86 + (v85 + v84 * v9 + v11 * v10), v92, v93);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      TLOAD(v82, v94);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      // pto: %out__iter_v7_pview
      ;
      const int64_t v95 = 0;
      // pto: %out__iter_v7_pview
      ;
      __gm__ int8_t* v96 = PTOAS__GLOBAL_TENSOR_DATA(v29);
      // pto: %out__iter_v7_pview
      ;
      const int64_t v97 = 1;
      // pto: %out__iter_v7_pview
      ;
      const int64_t v98 = 1;
      // pto: %out__iter_v7_pview
      ;
      const int64_t v99 = 1;
      // pto: %out__iter_v7_pview
      ;
      int64_t v100 = v10 * v9;
      // pto: %out__iter_v7_pview
      ;
      int64_t v101 = v99 * v100;
      // pto: %out__iter_v7_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v102 = pto::Shape<1, 1, 1, 1, -1>(v97, v98, v99, v10, v12);
      // pto: %out__iter_v7_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v103 = pto::Stride<-1, -1, -1, -1, -1>(v98 * v101, v101, v100, v9, v10);
      // pto: %out__iter_v7_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v104 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v96 + (v95 + v84 * v9 + v11 * v10), v102, v103);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      TSTORE(v104, v82);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      // pto: %tile__ssa_v3
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v13);
      // pto: %tile__ssa_v3
      ;
      uint64_t v106 = (uint64_t) v7;
      TASSIGN(v105, v106);
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
      pto::Shape<1, 1, 1, 1, -1> v114 = pto::Shape<1, 1, 1, 1, -1>(v109, v110, v111, v10, v13);
      // pto: %15
      ;
      pto::Stride<-1, -1, -1, -1, -1> v115 = pto::Stride<-1, -1, -1, -1, -1>(v110 * v113, v113, v112, v9, v10);
      // pto: %15
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v116 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v108 + (v107 + v84 * v9 + v12 * v10), v114, v115);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      TLOAD(v105, v116);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
      // pto: %out__ssa_v9_pview
      ;
      const int64_t v117 = 0;
      // pto: %out__ssa_v9_pview
      ;
      __gm__ int8_t* v118 = PTOAS__GLOBAL_TENSOR_DATA(v29);
      // pto: %out__ssa_v9_pview
      ;
      const int64_t v119 = 1;
      // pto: %out__ssa_v9_pview
      ;
      const int64_t v120 = 1;
      // pto: %out__ssa_v9_pview
      ;
      const int64_t v121 = 1;
      // pto: %out__ssa_v9_pview
      ;
      int64_t v122 = v10 * v9;
      // pto: %out__ssa_v9_pview
      ;
      int64_t v123 = v121 * v122;
      // pto: %out__ssa_v9_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v124 = pto::Shape<1, 1, 1, 1, -1>(v119, v120, v121, v10, v13);
      // pto: %out__ssa_v9_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v125 = pto::Stride<-1, -1, -1, -1, -1>(v120 * v123, v123, v122, v9, v10);
      // pto: %out__ssa_v9_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v126 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v118 + (v117 + v84 * v9 + v12 * v10), v124, v125);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
      TSTORE(v126, v105);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
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