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

// Direct (non-pipelined) collective algorithms: every workgroup reaches its
// peers by a closed-form rank offset and spreads itself across the remote links
// each step, rather than forwarding around a ring. They share no ring helpers —
// each computes its own per-link workgroup share inline.
#pragma once

#include "origami/comm/algorithms/base.hpp"

#include <vector>

namespace origami::comm {

/**
 * @brief Sequential collective where all workgroups target the same link each timestep.
 *
 * All WGs → same link each timestep. Used by one_shot AR + A2A (sequential).
 * Skips self; visits the N-1 remote peers one per round.
 */
class all_to_same_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size with an optional work-graph closure.
   *
   * @param num_gpus Communicator size (number of ranks).
   * @param wg_fn Per-hop work-graph builder; defaults to pull+store (load+store when local).
   */
  explicit all_to_same_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {});

  /// @brief Target the remote peer (my_rank + timestep + 1), pulling from it (load when local).
  schedule_entry_t link_of(int pid, int timestep, int my_rank) const override;

  /// @brief Exactly one link is active per timestep, carrying all num_wgs workgroups.
  std::vector<int> wgs_per_active_link(int timestep, int num_wgs) const override;

  /// @brief N-1 rounds, one per remote peer.
  int num_timesteps() const override;

 private:
  /**
   * @brief Default work graph: pull from the peer then store (load+store when local).
   *
   * @param peer Peer rank for this hop.
   * @param is_self True when the hop is local.
   * @return Ordered primitives for the hop.
   */
  static std::vector<op_t> default_work_graph(int peer, int my_rank, int num_gpus, bool is_self);

  int num_gpus_;           ///< Communicator size.
  work_graph_fn_t wg_fn_;  ///< Per-hop work-graph builder.
};

/**
 * @brief Direct collective with each workgroup's starting peer staggered by pid.
 *
 * pid % world_size offsets the starting peer; workgroups spread uniformly over
 * the remote links. Used by two-shot AR/RS and a2a. Includes a self-timestep.
 */
class pid_staggered_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size with an optional work-graph closure.
   *
   * @param num_gpus Communicator size (number of ranks).
   * @param wg_fn Per-hop work-graph builder; defaults to pull+reduce (load+reduce when local).
   */
  explicit pid_staggered_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {});

  /// @brief Closed-form peer for one round: stagger the start by pid, advance by timestep, pull.
  schedule_entry_t link_of(int pid, int timestep, int my_rank) const override;

  /// @brief All N-1 links are active each timestep with the remote workgroups spread evenly.
  std::vector<int> wgs_per_active_link(int timestep, int num_wgs) const override;

  /// @brief N rounds, including the self-timestep.
  int num_timesteps() const override;
  /// @brief Each step moves 1/N of the buffer.
  int chunks_per_timestep() const override;

 private:
  /**
   * @brief Default work graph: pull from the peer then reduce (load+reduce when local).
   *
   * @param peer Peer rank for this hop.
   * @param is_self True when the hop is local.
   * @return Ordered primitives for the hop.
   */
  static std::vector<op_t> default_work_graph(int peer, int my_rank, int num_gpus, bool is_self);

  int num_gpus_;           ///< Communicator size.
  work_graph_fn_t wg_fn_;  ///< Per-hop work-graph builder.
};

/**
 * @brief Single-step collective with each workgroup permanently bound to one link.
 *
 * Each workgroup is permanently assigned to one link by its pid (partitioned
 * all-gather): every destination is pushed to in a single timestep.
 */
class pid_partitioned_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size with an optional work-graph closure.
   *
   * @param num_gpus Communicator size (number of ranks).
   * @param wg_fn Per-hop work-graph builder; defaults to load+push (load+store when local).
   */
  explicit pid_partitioned_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {});

  /// @brief Bind the workgroup to the destination chosen by its pid, pushing to that peer.
  schedule_entry_t link_of(int pid, int timestep, int my_rank) const override;

  /// @brief All N-1 remote links are active with workgroups partitioned evenly across them.
  std::vector<int> wgs_per_active_link(int timestep, int num_wgs) const override;

  /// @brief A single timestep: every destination is served at once.
  int num_timesteps() const override;

 private:
  /**
   * @brief Default work graph: load locally then push to the peer (load+store when local).
   *
   * @param peer Peer rank for this hop.
   * @param is_self True when the hop is local.
   * @return Ordered primitives for the hop.
   */
  static std::vector<op_t> default_work_graph(int peer, int my_rank, int num_gpus, bool is_self);

  int num_gpus_;           ///< Communicator size.
  work_graph_fn_t wg_fn_;  ///< Per-hop work-graph builder.
};

/**
 * @brief All-reduce factored into a reduce-scatter shot then an all-gather shot.
 *
 * All-reduce factored as reduce-scatter then all-gather ("two shots"): each rank
 * first pulls and sums every peer's slice (N reduce steps, including its own
 * self-step), then pushes the finished slice out to all others (N-1 broadcast
 * steps). Hence num_timesteps = 2N-1 and each step moves 1/N of the buffer
 * (chunks_per_timestep = N). is_reduce_phase_ just splits the timeline at step N
 * into the two shots.
 */
class two_shot_all_reduce_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size.
   *
   * @param num_gpus Communicator size (number of ranks).
   */
  explicit two_shot_all_reduce_algorithm_t(int num_gpus);

  /// @brief Reduce phase (steps < N) pulls and sums each slice; broadcast phase pushes it out.
  schedule_entry_t link_of(int pid, int timestep, int my_rank) const override;

  /// @brief All N-1 links active each step, workgroups spread evenly across them.
  std::vector<int> wgs_per_active_link(int timestep, int num_wgs) const override;

  /// @brief 2N-1 rounds: N reduce steps (incl. self) then N-1 broadcast steps.
  int num_timesteps() const override;
  /// @brief Each step moves 1/N of the buffer.
  int chunks_per_timestep() const override;

 private:
  /// @brief True for the first N timesteps (the reduce-scatter shot); false in the broadcast shot.
  constexpr bool is_reduce_phase_(int timestep) const noexcept { return timestep < num_gpus_; }
  int num_gpus_;  ///< Communicator size.
};

}  // namespace origami::comm
