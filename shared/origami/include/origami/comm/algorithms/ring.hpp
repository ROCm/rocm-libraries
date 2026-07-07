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

// Ring collective algorithms: every hop forwards the slice to the fixed
// next-rank neighbour, so the workgroups spread across the ring links via the
// shared ring_distribute helper below. Some are true pipelined rings
// (signal_t/wait_t producer→consumer dependency) priced by aggregate throughput
// rather than a sum of per-step latencies.
#pragma once

#include "origami/comm/algorithms/base.hpp"

#include <vector>

namespace origami::comm {

/**
 * @brief Spread num_wgs workgroups over the available ring links, conserving the total.
 *
 * The defining constraint is conservation: the WG counts returned must sum to
 * exactly num_wgs — every launched workgroup is doing real work on some link,
 * none invented, none dropped. (An earlier even-division form violated this,
 * dropping WGs at high channel counts and fabricating them at low counts,
 * mispricing contention.)
 *
 * There can be at most N-1 distinct ring links, so we use
 * nrings = min(num_wgs, N-1) and hand each ring floor(num_wgs/nrings), giving
 * the first `extra` rings one more so the remainder is absorbed and the total
 * is preserved.
 *
 * @param num_wgs Total workgroups to distribute.
 * @param num_gpus Communicator size (bounds the ring link count at N-1).
 * @return Per-ring workgroup count (one entry per ring link); the values sum to num_wgs.
 */
std::vector<int> ring_distribute(int num_wgs, int num_gpus);

/**
 * @brief Pipelined ring whose every hop crosses the fixed next_rank neighbour link.
 *
 * All hops go to next_rank. Used by ring all-reduce (older form); the work graph
 * carries the wait_t/signal_t producer→consumer dependency between adjacent ranks.
 */
class ring_fixed_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size with an optional work-graph closure.
   *
   * @param num_gpus Communicator size (number of ranks).
   * @param wg_fn Per-hop work-graph builder; defaults to the reduce-ring graph with signal/wait.
   */
  explicit ring_fixed_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {});

  /// @brief Every hop pushes to the fixed next-rank neighbour.
  schedule_entry_t link_of(int pid, int timestep, int my_rank) const override;

  /// @brief Workgroups distributed across the ring links (see ring_distribute).
  std::vector<int> wgs_per_active_link(int timestep, int num_wgs) const override;

  /// @brief N-1 ring hops.
  int num_timesteps() const override;
  /// @brief Each hop moves 1/N of the buffer.
  int chunks_per_timestep() const override;
  /// @brief Ring-class for the per-step overhead heuristic.
  bool is_ring_class() const override;
  /// @brief Priced with the closed-form pipelined-ring throughput model.
  bool is_ring_pipeline() const override;

 private:
  /**
   * @brief Default reduce-ring work graph: wait on prev, pull, reduce, store, signal next.
   *
   * @param my_rank Calling rank (determines its prev/next neighbours).
   * @param num_gpus Communicator size.
   * @return Ordered primitives for the hop, including the signal/wait dependency.
   */
  static std::vector<op_t> default_work_graph(int peer, int my_rank, int num_gpus, bool is_self);

  int num_gpus_;           ///< Communicator size.
  work_graph_fn_t wg_fn_;  ///< Per-hop work-graph builder.
};

/**
 * @brief All-gather ring: N-1 hops forwarding each rank's slice around the ring.
 *
 * N-1 step ring. Each step does load_t (local) + store_t (local) + push_t (forward to next).
 */
class ring_all_gather_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size.
   *
   * @param num_gpus Communicator size (number of ranks).
   */
  explicit ring_all_gather_algorithm_t(int num_gpus);

  /// @brief Each step loads locally, stores, and pushes forward to the next rank.
  schedule_entry_t link_of(int pid, int timestep, int my_rank) const override;

  /// @brief Workgroups distributed across the ring links (see ring_distribute).
  std::vector<int> wgs_per_active_link(int timestep, int num_wgs) const override;

  /// @brief N-1 ring hops.
  int num_timesteps() const override;
  /// @brief One chunk per step (AG convention: the message is the per-rank send).
  int chunks_per_timestep() const override;
  /// @brief Ring-class for the per-step overhead heuristic.
  bool is_ring_class() const override;
  // Sequential-style throughput model (no pipelined ring).

 private:
  int num_gpus_;  ///< Communicator size.
};

/**
 * @brief Reduce-scatter ring: the all-gather ring with a reduce on each hop.
 *
 * Structurally identical to the all-gather ring plus a reduce_t, so each step
 * loads, reduces, stores, and forwards the slice to the next rank.
 */
class ring_reduce_scatter_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size.
   *
   * @param num_gpus Communicator size (number of ranks).
   */
  explicit ring_reduce_scatter_algorithm_t(int num_gpus);

  /// @brief Each step loads, reduces, stores, and pushes forward to the next rank.
  schedule_entry_t link_of(int pid, int timestep, int my_rank) const override;

  /// @brief Workgroups distributed across the ring links (see ring_distribute).
  std::vector<int> wgs_per_active_link(int timestep, int num_wgs) const override;

  /// @brief N-1 ring hops.
  int num_timesteps() const override;
  /// @brief One chunk per step (the per-rank slice).
  int chunks_per_timestep() const override;
  /// @brief Ring-class for the per-step overhead heuristic.
  bool is_ring_class() const override;

 private:
  int num_gpus_;  ///< Communicator size.
};

/**
 * @brief Bandwidth-optimal all-reduce: a reduce-scatter ring then an all-gather ring.
 *
 * A reduce-scatter ring (N-1 steps, each pulls from prev, sums, signals next)
 * followed by an all-gather ring (N-1 steps, each pulls the finished slice and
 * forwards it). 2(N-1) steps total, every step crossing the same neighbour link
 * — which is why it is a true pipelined ring (is_ring_pipeline) priced by
 * aggregate throughput, not a sum of per-step latencies. The wait_t/signal_t in
 * the work graph are the producer→consumer dependency that serializes adjacent
 * ranks within a step.
 */
class ring_all_reduce_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size.
   *
   * @param num_gpus Communicator size (number of ranks).
   */
  explicit ring_all_reduce_algorithm_t(int num_gpus);

  /// @brief Reduce-scatter phase (pull+reduce) for the first N-1 steps, then all-gather (pull).
  schedule_entry_t link_of(int pid, int timestep, int my_rank) const override;

  /// @brief Workgroups distributed across the ring links (see ring_distribute).
  std::vector<int> wgs_per_active_link(int timestep, int num_wgs) const override;

  /// @brief 2(N-1) hops: a reduce-scatter ring followed by an all-gather ring.
  int num_timesteps() const override;
  /// @brief Each hop moves 1/N of the buffer.
  int chunks_per_timestep() const override;
  /// @brief Ring-class for the per-step overhead heuristic.
  bool is_ring_class() const override;
  /// @brief Priced with the closed-form pipelined-ring throughput model.
  bool is_ring_pipeline() const override;

 private:
  int num_gpus_;  ///< Communicator size.
};

/**
 * @brief Broadcast as an N-1 hop pipeline forwarding the root's data around the ring.
 *
 * N-1 hop pipeline on one link. Each GPU does load_t + store_t + push_t,
 * forwarding the data to its next-rank neighbour.
 */
class ring_broadcast_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size.
   *
   * @param num_gpus Communicator size (number of ranks).
   */
  explicit ring_broadcast_algorithm_t(int num_gpus);

  /// @brief Each step loads locally, stores, and pushes forward to the next rank.
  schedule_entry_t link_of(int pid, int timestep, int my_rank) const override;

  /// @brief Workgroups distributed across the ring links (see ring_distribute).
  std::vector<int> wgs_per_active_link(int timestep, int num_wgs) const override;

  /// @brief N-1 ring hops.
  int num_timesteps() const override;
  /// @brief Each hop moves 1/N of the buffer.
  int chunks_per_timestep() const override;

 private:
  int num_gpus_;  ///< Communicator size.
};

}  // namespace origami::comm
