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
  // pto: %c16384_index
  const int64_t v9 = 16384;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c0_index
  const int64_t v11 = 0;
  // pto: %data__ssa_v0_view
  const int64_t v12 = 1;
  // pto: %data__ssa_v0_view
  const int64_t v13 = 1;
  // pto: %data__ssa_v0_view
  const int64_t v14 = 1;
  // pto: %data__ssa_v0_view
  int64_t v15 = v8 * v9;
  // pto: %data__ssa_v0_view
  int64_t v16 = v14 * v15;
  // pto: %data__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v17 = pto::Shape<1, 1, 1, -1, -1>(v12, v13, v14, v8, v9);
  // pto: %data__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v18 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v16, v16, v15, v9, v10);
  // pto: %data__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v19 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v17, v18);
  // pto: %out__ssa_v0_view
  const int64_t v20 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v21 = 1;
  // pto: %out__ssa_v0_view
  const int64_t v22 = 1;
  // pto: %out__ssa_v0_view
  int64_t v23 = v8 * v9;
  // pto: %out__ssa_v0_view
  int64_t v24 = v22 * v23;
  // pto: %out__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v25 = pto::Shape<1, 1, 1, -1, -1>(v20, v21, v22, v8, v9);
  // pto: %out__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v26 = pto::Stride<-1, -1, -1, -1, -1>(v21 * v24, v24, v23, v9, v10);
  // pto: %out__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v27 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v25, v26);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  for (int64_t v28 = v11; v28 < v8; v28 += v10) {
    // pto: %n_rows_i32__tile
    ;
    int32_t v29 = (v2)[v28];
    (v4)[v28] = v29;
    // pto: %2
    ;
    int64_t v30 = (int64_t) v29;
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (int64_t v31 = v11; v31 < v30; v31 += v10) {
      // pto: %3
      ;
      int64_t v32 = (int64_t) ((uint64_t) v28 + (uint64_t) v31);
      // pto: %tile__ssa_v0
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v33 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v9);
      // pto: %tile__ssa_v0
      ;
      uint64_t v34 = (uint64_t) v7;
      TASSIGN(v33, v34);
      // pto: %4
      ;
      int64_t v35 = v32 < v11 ? v11 : v32;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v36 = 0;
      // pto: %data__ssa_v0_pview
      ;
      __gm__ int8_t* v37 = PTOAS__GLOBAL_TENSOR_DATA(v19);
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v38 = 1;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v39 = 1;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v40 = 1;
      // pto: %data__ssa_v0_pview
      ;
      int64_t v41 = v10 * v9;
      // pto: %data__ssa_v0_pview
      ;
      int64_t v42 = v40 * v41;
      // pto: %data__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v43 = pto::Shape<1, 1, 1, 1, -1>(v38, v39, v40, v10, v9);
      // pto: %data__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v44 = pto::Stride<-1, -1, -1, -1, -1>(v39 * v42, v42, v41, v9, v10);
      // pto: %data__ssa_v0_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v45 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v37 + (v36 + v35 * v9 + v11 * v10), v43, v44);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v33, v45);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__iter_v3_pview
      ;
      const int64_t v46 = 0;
      // pto: %out__iter_v3_pview
      ;
      __gm__ int8_t* v47 = PTOAS__GLOBAL_TENSOR_DATA(v27);
      // pto: %out__iter_v3_pview
      ;
      const int64_t v48 = 1;
      // pto: %out__iter_v3_pview
      ;
      const int64_t v49 = 1;
      // pto: %out__iter_v3_pview
      ;
      const int64_t v50 = 1;
      // pto: %out__iter_v3_pview
      ;
      int64_t v51 = v10 * v9;
      // pto: %out__iter_v3_pview
      ;
      int64_t v52 = v50 * v51;
      // pto: %out__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v53 = pto::Shape<1, 1, 1, 1, -1>(v48, v49, v50, v10, v9);
      // pto: %out__iter_v3_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v54 = pto::Stride<-1, -1, -1, -1, -1>(v49 * v52, v52, v51, v9, v10);
      // pto: %out__iter_v3_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v55 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v47 + (v46 + v35 * v9 + v11 * v10), v53, v54);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v55, v33);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    for (int64_t v56 = v30; v56 < v10; v56 += v10) {
      // pto: %7
      ;
      int64_t v57 = (int64_t) ((uint64_t) v28 + (uint64_t) v56);
      // pto: %tile__ssa_v1
      ;
      Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v58 = Tile<TileType::Vec, int8_t, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v9);
      // pto: %tile__ssa_v1
      ;
      uint64_t v59 = (uint64_t) v7;
      TASSIGN(v58, v59);
      // pto: %8
      ;
      int64_t v60 = v57 < v11 ? v11 : v57;
      // pto: %9
      ;
      const int64_t v61 = 0;
      // pto: %9
      ;
      __gm__ int8_t* v62 = PTOAS__GLOBAL_TENSOR_DATA(v19);
      // pto: %9
      ;
      const int64_t v63 = 1;
      // pto: %9
      ;
      const int64_t v64 = 1;
      // pto: %9
      ;
      const int64_t v65 = 1;
      // pto: %9
      ;
      int64_t v66 = v10 * v9;
      // pto: %9
      ;
      int64_t v67 = v65 * v66;
      // pto: %9
      ;
      pto::Shape<1, 1, 1, 1, -1> v68 = pto::Shape<1, 1, 1, 1, -1>(v63, v64, v65, v10, v9);
      // pto: %9
      ;
      pto::Stride<-1, -1, -1, -1, -1> v69 = pto::Stride<-1, -1, -1, -1, -1>(v64 * v67, v67, v66, v9, v10);
      // pto: %9
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v70 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v62 + (v61 + v60 * v9 + v11 * v10), v68, v69);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      TLOAD(v58, v70);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      // pto: %out__iter_v6_pview
      ;
      const int64_t v71 = 0;
      // pto: %out__iter_v6_pview
      ;
      __gm__ int8_t* v72 = PTOAS__GLOBAL_TENSOR_DATA(v27);
      // pto: %out__iter_v6_pview
      ;
      const int64_t v73 = 1;
      // pto: %out__iter_v6_pview
      ;
      const int64_t v74 = 1;
      // pto: %out__iter_v6_pview
      ;
      const int64_t v75 = 1;
      // pto: %out__iter_v6_pview
      ;
      int64_t v76 = v10 * v9;
      // pto: %out__iter_v6_pview
      ;
      int64_t v77 = v75 * v76;
      // pto: %out__iter_v6_pview
      ;
      pto::Shape<1, 1, 1, 1, -1> v78 = pto::Shape<1, 1, 1, 1, -1>(v73, v74, v75, v10, v9);
      // pto: %out__iter_v6_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v79 = pto::Stride<-1, -1, -1, -1, -1>(v74 * v77, v77, v76, v9, v10);
      // pto: %out__iter_v6_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v80 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v72 + (v71 + v60 * v9 + v11 * v10), v78, v79);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      TSTORE(v80, v58);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
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