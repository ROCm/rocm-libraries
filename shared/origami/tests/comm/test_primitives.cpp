/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Primitive work-resolution regression. Each primitive's resolve() is
// exercised at a fixed (cl_per_iter, instrs_per_cl, elements_per_iter)
// triple and compared field-by-field against the reference values.
#include "test_harness.hpp"

#include "origami/comm/primitives.hpp"

using namespace origami::comm;

// Reference args matching the typical RCCL inner-loop config:
//   bytes_per_iter = vgprs_for_data * 4 = 128 * 4 = 512 → cl_per_iter = 8
//   load_width = DWORDX16 → instrs_per_cl = 64 / 64 = 1
//   elements_per_iter = 256 (BF16 — chosen so we can verify VALU counts)
inline constexpr iter_dims_t kIter{/*cl_per_iter=*/8,
                                   /*instrs_per_cl=*/1,
                                   /*elements_per_iter=*/256};

// ─── Individual primitives ─────────────────────────────────────
TEST(load_resolves_full_read_path) {
  const auto w = load_t{}.resolve(kIter);
  CHECK(w.vmem_read_instrs == 8);  // 8 cl × 1 instr/cl
  CHECK(w.tcp_read_cl == 8);
  CHECK(w.l2_read_cl == 8);
  CHECK(w.mall_read_cl == 8);
  CHECK(w.hbm_read_cl == 8);
  // Everything else zero.
  CHECK(w.vmem_write_instrs == 0);
  CHECK(w.xgmi_read_cl == 0);
  CHECK(w.xgmi_write_cl == 0);
  CHECK(w.valu_ops == 0);
  CHECK(w.atomic_count == 0);
}

TEST(store_default_writes_through_all_levels) {
  const auto w = store_t{}.resolve(kIter);
  CHECK(w.vmem_write_instrs == 8);
  CHECK(w.tcp_write_cl == 8);
  CHECK(w.l2_write_cl == 8);  // not write-through → L2 charged
  CHECK(w.mall_write_cl == 8);
  CHECK(w.hbm_write_cl == 8);
}

TEST(store_write_through_skips_l2) {
  const auto w = store_t{/*write_through=*/true}.resolve(kIter);
  CHECK(w.l2_write_cl == 0);
  CHECK(w.tcp_write_cl == 8);
  CHECK(w.mall_write_cl == 8);
  CHECK(w.hbm_write_cl == 8);
}

TEST(pull_charges_xgmi_read_not_hbm) {
  const auto w = pull_t{/*peer=*/3}.resolve(kIter);
  CHECK(w.vmem_read_instrs == 8);
  CHECK(w.tcp_read_cl == 8);
  CHECK(w.l2_read_cl == 8);
  CHECK(w.xgmi_read_cl == 8);
  CHECK(w.hbm_read_cl == 0);  // remote read, no local HBM
  CHECK(w.mall_read_cl == 0);
}

TEST(push_charges_full_read_plus_xgmi_write) {
  const auto w = push_t{/*peer=*/3}.resolve(kIter);
  CHECK(w.vmem_read_instrs == 8);
  CHECK(w.tcp_read_cl == 8);
  CHECK(w.l2_read_cl == 8);
  CHECK(w.mall_read_cl == 8);
  CHECK(w.hbm_read_cl == 8);
  CHECK(w.xgmi_write_cl == 8);
}

TEST(reduce_charges_only_valu) {
  const auto w = reduce_t{/*op=*/reduce_op_t::SUM}.resolve(kIter);
  CHECK(w.valu_ops == 256);
  CHECK(w.vmem_read_instrs == 0);
  CHECK(w.hbm_read_cl == 0);
}

TEST(signal_one_atomic_and_one_xgmi_write) {
  const auto w = signal_t{/*peer=*/2}.resolve(kIter);
  CHECK(w.atomic_count == 1);
  CHECK(w.xgmi_write_cl == 1);
  CHECK(w.vmem_read_instrs == 0);
  CHECK(w.hbm_read_cl == 0);
}

TEST(wait_one_atomic_and_one_l2_read) {
  const auto w = wait_t{/*peer=*/2}.resolve(kIter);
  CHECK(w.atomic_count == 1);
  CHECK(w.l2_read_cl == 1);
  CHECK(w.xgmi_read_cl == 0);
}

// ─── op_t variant + resolve_work_graph ──────────────────────────
TEST(work_graph_partitions_sync_from_iter) {
  // A "pull from peer" pattern: wait_t(peer), pull_t(peer), signal_t(self).
  std::vector<op_t> ops = {
      wait_t{/*peer=*/1},
      pull_t{/*peer=*/1},
      signal_t{/*peer=*/0},
  };
  const auto resolved = resolve_work_graph(ops, kIter);

  // sync_work: wait_t + signal_t.
  CHECK(resolved.sync_work.atomic_count == 2);
  CHECK(resolved.sync_work.l2_read_cl == 1);     // from wait_t
  CHECK(resolved.sync_work.xgmi_write_cl == 1);  // from signal_t

  // iter_work: pull_t only.
  CHECK(resolved.iter_work.vmem_read_instrs == 8);
  CHECK(resolved.iter_work.tcp_read_cl == 8);
  CHECK(resolved.iter_work.l2_read_cl == 8);
  CHECK(resolved.iter_work.xgmi_read_cl == 8);
  CHECK(resolved.iter_work.atomic_count == 0);
}

TEST(work_graph_aggregates_multiple_iter_ops) {
  // Local copy: load_t → store_t (matches a kernel that just memcpys).
  std::vector<op_t> ops = {load_t{}, store_t{}};
  const auto resolved   = resolve_work_graph(ops, kIter);

  CHECK(resolved.iter_work.vmem_read_instrs == 8);
  CHECK(resolved.iter_work.vmem_write_instrs == 8);
  CHECK(resolved.iter_work.tcp_read_cl == 8);
  CHECK(resolved.iter_work.tcp_write_cl == 8);
  CHECK(resolved.iter_work.hbm_read_cl == 8);
  CHECK(resolved.iter_work.hbm_write_cl == 8);
  CHECK(resolved.iter_work.l2_read_cl == 8);
  CHECK(resolved.iter_work.l2_write_cl == 8);
  CHECK(resolved.iter_work.mall_read_cl == 8);
  CHECK(resolved.iter_work.mall_write_cl == 8);

  CHECK(resolved.sync_work.atomic_count == 0);
}

TEST(work_graph_empty_is_zero) {
  std::vector<op_t> ops;
  const auto resolved = resolve_work_graph(ops, kIter);
  CHECK(resolved.iter_work.vmem_read_instrs == 0);
  CHECK(resolved.iter_work.hbm_read_cl == 0);
  CHECK(resolved.sync_work.atomic_count == 0);
}

TEST(work_graph_resolves_from_iter_dims) {
  std::vector<op_t> ops = {load_t{}};
  const auto resolved   = resolve_work_graph(
      ops, iter_dims_t{/*cl_per_iter=*/4, /*instrs_per_cl=*/4, /*elements_per_iter=*/128});
  CHECK(resolved.iter_work.vmem_read_instrs == 16);  // 4 × 4
  CHECK(resolved.iter_work.hbm_read_cl == 4);
}

ORIGAMI_TEST_MAIN()
