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

#include "origami/comm/algorithms/direct.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace origami::comm {

// ═════════════════════════════════════════════════════════════════════════════
// all_to_same_algorithm_t
// ═════════════════════════════════════════════════════════════════════════════
// Sequential collective where every workgroup targets the same single link each
// timestep, visiting the N-1 remote peers one per round (self is skipped). Backs
// one-shot all-reduce and the sequential all-to-all.

// Store the communicator size and the per-hop work-graph closure (default: pull+store).
all_to_same_algorithm_t::all_to_same_algorithm_t(int num_gpus, work_graph_fn_t wg_fn)
    : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

// Target the next remote peer (my_rank + timestep + 1) and pull from it; load locally on a
// self-step. (pid does not steer this schedule.)
schedule_entry_t all_to_same_algorithm_t::link_of(int pid, int timestep, int my_rank) const {
  const int peer     = floor_mod(my_rank + timestep + 1, num_gpus_);
  const bool is_self = (peer == my_rank);
  auto work          = wg_fn_(peer, my_rank, num_gpus_, is_self);
  return {is_self ? SELF_LINK : peer, peer, direction_t::PULL, std::move(work), is_self};
}

// Exactly one link is active this step, carrying all the workgroups (independent of timestep).
std::vector<int> all_to_same_algorithm_t::wgs_per_active_link(int timestep, int num_wgs) const {
  return {num_wgs};
}

// One round per remote peer.
int all_to_same_algorithm_t::num_timesteps() const { return num_gpus_ - 1; }

// Default hop: pull then store; load+store when the data is already local. (Depends only on
// peer/is_self; my_rank and num_gpus are unused.)
std::vector<op_t> all_to_same_algorithm_t::default_work_graph(int peer,
                                                              int my_rank,
                                                              int num_gpus,
                                                              bool is_self) {
  if (is_self) return {load_t{}, store_t{}};
  return {pull_t{peer}, store_t{}};
}

// ═════════════════════════════════════════════════════════════════════════════
// pid_staggered_algorithm_t
// ═════════════════════════════════════════════════════════════════════════════
// Direct collective whose starting peer is offset by the workgroup's pid, so
// workgroups fan out uniformly over the N-1 remote links; includes a self-timestep.
// Backs two-shot all-reduce/reduce-scatter and all-to-all.

// Store the communicator size and the per-hop work-graph closure (default: pull+reduce).
pid_staggered_algorithm_t::pid_staggered_algorithm_t(int num_gpus, work_graph_fn_t wg_fn)
    : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

// Closed-form peer for one round: offset the start by pid, then advance by `timestep`
// so successive calls sweep the ranks in order; pull from the resulting peer.
schedule_entry_t pid_staggered_algorithm_t::link_of(int pid, int timestep, int my_rank) const {
  const int start    = floor_mod(pid, num_gpus_);
  const int peer_idx = floor_mod(start + timestep, num_gpus_);
  const int peer     = floor_mod(my_rank + peer_idx, num_gpus_);
  const bool is_self = (peer == my_rank);
  auto work          = wg_fn_(peer, my_rank, num_gpus_, is_self);
  return {is_self ? SELF_LINK : peer, peer, direction_t::PULL, std::move(work), is_self};
}

// All N-1 links active each step, remote workgroups split evenly across them (independent
// of timestep).
std::vector<int> pid_staggered_algorithm_t::wgs_per_active_link(int timestep, int num_wgs) const {
  const int num_links  = num_gpus_ - 1;
  const int remote_wgs = num_wgs * (num_gpus_ - 1) / num_gpus_;
  const int per_link   = std::max(remote_wgs / std::max(num_links, 1), 1);
  return std::vector<int>(std::max(num_links, 0), per_link);
}

// N rounds, counting the self-step.
int pid_staggered_algorithm_t::num_timesteps() const { return num_gpus_; }
// Each step moves 1/N of the buffer.
int pid_staggered_algorithm_t::chunks_per_timestep() const { return num_gpus_; }

// Default hop: pull then reduce; load+reduce when local. (Depends only on peer/is_self.)
std::vector<op_t> pid_staggered_algorithm_t::default_work_graph(int peer,
                                                                int my_rank,
                                                                int num_gpus,
                                                                bool is_self) {
  if (is_self) return {load_t{}, reduce_t{}};
  return {pull_t{peer}, reduce_t{}};
}

// ═════════════════════════════════════════════════════════════════════════════
// pid_partitioned_algorithm_t
// ═════════════════════════════════════════════════════════════════════════════
// Single-step collective where each workgroup is permanently bound to one
// destination link by its pid and pushes there, so every destination is served in
// a single timestep (partitioned all-gather).

// Store the communicator size and the per-hop work-graph closure (default: load+push).
pid_partitioned_algorithm_t::pid_partitioned_algorithm_t(int num_gpus, work_graph_fn_t wg_fn)
    : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

// Permanently bind the workgroup to the destination chosen by its pid and push there. The
// schedule is a single step, so timestep does not steer it.
schedule_entry_t pid_partitioned_algorithm_t::link_of(int pid, int timestep, int my_rank) const {
  const int dest     = floor_mod(pid, num_gpus_);
  const int peer     = floor_mod(my_rank + dest, num_gpus_);
  const bool is_self = (peer == my_rank);
  auto work          = wg_fn_(peer, my_rank, num_gpus_, is_self);
  return {is_self ? SELF_LINK : peer, peer, direction_t::PUSH, std::move(work), is_self};
}

// All N-1 remote links active, workgroups partitioned evenly across them (independent of
// timestep).
std::vector<int> pid_partitioned_algorithm_t::wgs_per_active_link(int timestep, int num_wgs) const {
  const int per_link = std::max(num_wgs / num_gpus_, 1);
  return std::vector<int>(std::max(num_gpus_ - 1, 0), per_link);
}

// A single timestep: every destination is served at once.
int pid_partitioned_algorithm_t::num_timesteps() const { return 1; }

// Default hop: load then push; load+store when local. (Depends only on peer/is_self.)
std::vector<op_t> pid_partitioned_algorithm_t::default_work_graph(int peer,
                                                                  int my_rank,
                                                                  int num_gpus,
                                                                  bool is_self) {
  if (is_self) return {load_t{}, store_t{}};
  return {load_t{}, push_t{peer}};
}

// ═════════════════════════════════════════════════════════════════════════════
// two_shot_all_reduce_algorithm_t
// ═════════════════════════════════════════════════════════════════════════════
// All-reduce factored into two shots — a reduce-scatter (N reduce steps, including
// the self-step) followed by an all-gather (N-1 broadcast steps). 2N-1 rounds
// total, each moving 1/N of the buffer.

// Store the communicator size.
two_shot_all_reduce_algorithm_t::two_shot_all_reduce_algorithm_t(int num_gpus)
    : num_gpus_{num_gpus} {}

// Reduce phase (step < N) pulls and sums each peer's slice; broadcast phase pushes
// the finished slice out, skipping self in the peer ordering.
schedule_entry_t two_shot_all_reduce_algorithm_t::link_of(int pid,
                                                          int timestep,
                                                          int my_rank) const {
  const int N     = num_gpus_;
  const int start = floor_mod(pid, N);
  if (is_reduce_phase_(timestep)) {
    const int peer_idx     = floor_mod(start + timestep, N);
    const int peer         = floor_mod(my_rank + peer_idx, N);
    const bool is_self     = (peer == my_rank);
    std::vector<op_t> work = is_self ? std::vector<op_t>{load_t{}, reduce_t{}}
                                     : std::vector<op_t>{pull_t{peer}, reduce_t{}};
    return {is_self ? SELF_LINK : peer, peer, direction_t::PULL, std::move(work), is_self};
  }
  // Broadcast phase. Skip self in peer ordering.
  const int bcast_idx    = timestep - N;
  const int peer_offset  = floor_mod(start + bcast_idx, N - 1) + 1;
  const int peer         = floor_mod(my_rank + peer_offset, N);
  std::vector<op_t> work = {
      load_t{},
      push_t{peer},
  };
  return {peer, peer, direction_t::PUSH, std::move(work), false};
}

// All N-1 links active each step, workgroups spread evenly across them.
std::vector<int> two_shot_all_reduce_algorithm_t::wgs_per_active_link(int timestep,
                                                                      int num_wgs) const {
  const int num_links = num_gpus_ - 1;
  const int remote_wgs =
      is_reduce_phase_(timestep) ? num_wgs * (num_gpus_ - 1) / num_gpus_ : num_wgs;
  const int per_link = std::max(remote_wgs / std::max(num_links, 1), 1);
  return std::vector<int>(std::max(num_links, 0), per_link);
}

// 2N-1 rounds: N reduce steps (incl. self) then N-1 broadcast steps.
int two_shot_all_reduce_algorithm_t::num_timesteps() const { return 2 * num_gpus_ - 1; }
// Each step moves 1/N of the buffer.
int two_shot_all_reduce_algorithm_t::chunks_per_timestep() const { return num_gpus_; }

}  // namespace origami::comm
