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
  // pto: %c32768_index
  const int64_t v9 = 32768;
  // pto: %c1_index
  const int64_t v10 = 1;
  // pto: %c0_index
  const int64_t v11 = 0;
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  for (int64_t v12 = v11; v12 < v8; v12 += v10) {
    // pto: %n_rows_i32__tile
    ;
    int32_t v13 = (v2)[v12];
    (v4)[v12] = v13;
    // pto: %2
    ;
    int64_t v14 = (int64_t) v13;
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (int64_t v15 = v11; v15 < v14; v15 += v10) {
      // pto: %3
      ;
      int64_t v16 = (int64_t) ((uint64_t) v12 + (uint64_t) v15);
      // pto: %chunk__ssa_v0
      ;
      Tile<TileType::Vec, int8_t, 1, 32768, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v17 = Tile<TileType::Vec, int8_t, 1, 32768, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v9);
      // pto: %chunk__ssa_v0
      ;
      uint64_t v18 = (uint64_t) v7;
      TASSIGN(v17, v18);
      // pto: %4
      ;
      int64_t v19 = v16 < v11 ? v11 : v16;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v20 = 0;
      // pto: %data__ssa_v0_pview
      ;
      const int64_t v21 = 32768;
      // pto: %data__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 32768> v22 = pto::Shape<1, 1, 1, 1, 32768>();
      // pto: %data__ssa_v0_pview
      ;
      pto::Stride<32768, 32768, 32768, 32768, 1> v23 = pto::Stride<32768, 32768, 32768, 32768, 1>();
      // pto: %data__ssa_v0_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 32768>, pto::Stride<32768, 32768, 32768, 32768, 1>, pto::Layout::ND> v24 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 32768>, pto::Stride<32768, 32768, 32768, 32768, 1>, pto::Layout::ND>((__gm__ int8_t*) v1 + (v20 + v19 * v21), v22, v23);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v17, v24);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %out__iter_v3_pview
      ;
      const int64_t v25 = 0;
      // pto: %out__iter_v3_pview
      ;
      const int64_t v26 = 32768;
      // pto: %out__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 1, 32768> v27 = pto::Shape<1, 1, 1, 1, 32768>();
      // pto: %out__iter_v3_pview
      ;
      pto::Stride<32768, 32768, 32768, 32768, 1> v28 = pto::Stride<32768, 32768, 32768, 32768, 1>();
      // pto: %out__iter_v3_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 32768>, pto::Stride<32768, 32768, 32768, 32768, 1>, pto::Layout::ND> v29 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 32768>, pto::Stride<32768, 32768, 32768, 32768, 1>, pto::Layout::ND>((__gm__ int8_t*) v3 + (v25 + v19 * v26), v27, v28);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v29, v17);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    for (int64_t v30 = v14; v30 < v10; v30 += v10) {
      // pto: %7
      ;
      int64_t v31 = (int64_t) ((uint64_t) v12 + (uint64_t) v30);
      // pto: %chunk__ssa_v1
      ;
      Tile<TileType::Vec, int8_t, 1, 32768, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v32 = Tile<TileType::Vec, int8_t, 1, 32768, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v9);
      // pto: %chunk__ssa_v1
      ;
      uint64_t v33 = (uint64_t) v7;
      TASSIGN(v32, v33);
      // pto: %8
      ;
      int64_t v34 = v31 < v11 ? v11 : v31;
      // pto: %9
      ;
      const int64_t v35 = 0;
      // pto: %9
      ;
      const int64_t v36 = 32768;
      // pto: %9
      ;
      pto::Shape<1, 1, 1, 1, 32768> v37 = pto::Shape<1, 1, 1, 1, 32768>();
      // pto: %9
      ;
      pto::Stride<32768, 32768, 32768, 32768, 1> v38 = pto::Stride<32768, 32768, 32768, 32768, 1>();
      // pto: %9
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 32768>, pto::Stride<32768, 32768, 32768, 32768, 1>, pto::Layout::ND> v39 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 32768>, pto::Stride<32768, 32768, 32768, 32768, 1>, pto::Layout::ND>((__gm__ int8_t*) v1 + (v35 + v34 * v36), v37, v38);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      TLOAD(v32, v39);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      // pto: %out__iter_v6_pview
      ;
      const int64_t v40 = 0;
      // pto: %out__iter_v6_pview
      ;
      const int64_t v41 = 32768;
      // pto: %out__iter_v6_pview
      ;
      pto::Shape<1, 1, 1, 1, 32768> v42 = pto::Shape<1, 1, 1, 1, 32768>();
      // pto: %out__iter_v6_pview
      ;
      pto::Stride<32768, 32768, 32768, 32768, 1> v43 = pto::Stride<32768, 32768, 32768, 32768, 1>();
      // pto: %out__iter_v6_pview
      ;
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 32768>, pto::Stride<32768, 32768, 32768, 32768, 1>, pto::Layout::ND> v44 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 32768>, pto::Stride<32768, 32768, 32768, 32768, 1>, pto::Layout::ND>((__gm__ int8_t*) v3 + (v40 + v34 * v41), v42, v43);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      TSTORE(v44, v32);
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