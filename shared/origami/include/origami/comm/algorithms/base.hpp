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

// origami::comm — analytical communication cost model
//
// Collective algorithms: each algorithm emits the *schedule* of a collective,
// expressed as a pure function (pid, timestep) → which link a workgroup uses
// and what primitives it runs there. This is the bridge between an algorithm
// and the cost model: the model never hard-codes "all-gather costs X"; it asks
// the algorithm for the per-step work graph and lets latency.hpp /
// collective.hpp price it.
//
// Three quantities an algorithm exposes drive the whole cost, and each is a
// direct consequence of the algorithm's dataflow:
//   • num_timesteps()       — how many dependent communication rounds the
//     algorithm takes (e.g. a ring visits N-1 peers; two-shot does N reduce
//     rounds then N-1 broadcast rounds = 2N-1). More steps -> more serial
//     handshakes and, for sequential algorithms, more added latency.
//   • chunks_per_timestep() — how finely each GPU's tile is sliced per step.
//     A ring sends 1/N of the buffer per hop (chunks = N); a whole-tile step
//     sends all of it (chunks = 1). This sets the per-step wire bytes.
//   • wgs_per_active_link() — how the workgroups spread across the links lit
//     up this step, which sets per-link contention.
//
// Closed-form vs iterative — where the loop lives:
//   • Closed-form (here): every algorithm method answers ONE query directly from
//     the rank topology, holding no per-round state. link_of(pid, timestep,
//     my_rank) returns the schedule for exactly that one timestep; calling it
//     again with timestep+1 is independent of the previous call. Nothing in an
//     algorithm loops over the communication rounds.
//   • Iterative (the caller): the cost engine in collective.cpp walks the
//     timeline — `for timestep in [0, num_timesteps())` — invoking link_of and
//     wgs_per_active_link once per round and summing the priced work. That external
//     loop is the only place the schedule is actually "stepped through"; the
//     staggered sweep through peers, for instance, emerges from successive
//     timesteps, not from any loop inside link_of.
//   • The small bounded loops you do see inside wgs_per_active_link / ring_distribute
//     fill a vector over the <= N-1 links of a SINGLE step — they build one
//     step's data, they do not iterate the schedule.
// Because the per-call work is closed-form and stateless, the algorithms fold
// cheaply and virtual dispatch happens only once per collective.
//
// Self-timestep distinction: when a step maps a rank to itself, the data is
// already local, so the work graph uses load_t (local HBM) instead of pull_t
// (xGMI). Self-steps consume no fabric bandwidth and are not MSHR-limited —
// the model must not bill them as remote transfers.
//
// This header defines the shared vocabulary every algorithm family builds on:
// the per-timestep schedule entry, the work-graph closure type, the floored
// modulo used by every rank-offset computation, and the abstract base class.
// The concrete algorithms live in the family headers (direct.hpp, ring.hpp)
// and the public builders in resolve.hpp; algorithms.hpp re-includes them all.
#pragma once

#include "origami/comm/primitives.hpp"
#include "origami/comm/types.hpp"

#include <functional>
#include <vector>

namespace origami::comm {

/// Sentinel link id: the timestep is local (a rank maps to itself, no fabric link is used).
inline constexpr int SELF_LINK = -1;

/**
 * @brief One workgroup's communication assignment for a single timestep.
 *
 * Returned by collective_algorithm_t::link_of: it names the fabric link to use,
 * the peer rank on the other end, the transfer direction, and the ordered work
 * graph of primitives to run there. When the step maps a rank to itself the
 * entry is flagged local (is_self) and link_id is SELF_LINK.
 */
struct schedule_entry_t {
  int link_id;                   ///< Fabric link id, or SELF_LINK when the step is local.
  int peer_rank;                 ///< Rank on the other end of the link (== own rank if self).
  direction_t direction;         ///< Whether data is pulled from or pushed to the peer.
  std::vector<op_t> work_graph;  ///< Ordered primitives the workgroup runs this timestep.
  bool is_self = false;          ///< True when the step is local (data already resident).
};

/**
 * @brief Closure that builds the per-timestep work graph for one workgroup.
 *
 * Signature: (peer, my_rank, num_gpus, is_self) -> [op_t]. Concrete algorithms
 * accept one to let callers customize the primitives emitted on each hop while
 * reusing the algorithm's link/timestep schedule.
 *
 * @param peer Peer rank for this hop.
 * @param my_rank Calling rank.
 * @param num_gpus Communicator size.
 * @param is_self True when the hop maps the rank to itself (data is local).
 * @return Ordered list of primitives to run for this hop.
 */
using work_graph_fn_t =
    std::function<std::vector<op_t>(int peer, int my_rank, int num_gpus, bool is_self)>;

/**
 * @brief Floored modulo: always returns a value in the half-open range [0, n).
 *
 * Ring schedules index peers as my_rank ± offset, which can go negative or past
 * N; this wraps those into a valid rank so a single closed form expresses "the
 * neighbour k hops away".
 *
 * @param a Possibly negative or out-of-range index.
 * @param n Modulus (communicator size); must be positive.
 * @return a reduced into the range [0, n).
 */
constexpr int floor_mod(int a, int n) noexcept {
  const int r = a % n;
  return (r < 0) ? r + n : r;
}

/**
 * @brief Abstract base for collective algorithms: the per-step schedule of a collective.
 *
 * Each algorithm emits a pure function of (pid, timestep) describing which link
 * a workgroup uses and what primitives it runs there, decoupling the cost model
 * from any hard-coded per-collective cost. num_gpus is fixed at construction
 * (num_gpus_ on each concrete algorithm), so the schedule queries below take
 * only the per-call coordinates (pid, timestep, my_rank, num_wgs) and read the
 * communicator size from the object.
 *
 * The schedule queries take a uniform (pid, timestep, ...) signature so the cost
 * engine can drive every algorithm the same way. Many algorithms do not read
 * every coordinate: phase-changing ones (two-shot, ring all-reduce) steer on
 * timestep, while round-invariant ones (the rings, all-to-same) emit the same
 * schedule every round and ignore it. Such parameters are kept named — rather
 * than dropped or marked unused — so the interface stays consistent and the
 * intent is documented per override; leaving a named-but-unused parameter is
 * warning-free here (no -Wextra/-Wunused-parameter in the build).
 */
class collective_algorithm_t {
 public:
  /// @brief Virtual destructor for safe polymorphic deletion.
  virtual ~collective_algorithm_t() = default;

  /**
   * @brief Resolve the link, peer, and work graph for one workgroup at a timestep.
   *
   * Closed-form and stateless: this returns the schedule for the single given
   * timestep only and does not loop over rounds. The cost engine drives the
   * timeline by calling it once per round (timestep = 0 .. num_timesteps()-1),
   * so any "sweep" across peers emerges from successive calls, not from here.
   *
   * @param pid Workgroup (partition) id within the launch.
   * @param timestep Communication round index (0-based).
   * @param my_rank Calling rank in the communicator.
   * @return schedule_entry_t naming the link, peer, direction, and primitives to run.
   */
  virtual schedule_entry_t link_of(int pid, int timestep, int my_rank) const = 0;

  /**
   * @brief Workgroup count on each link lit up this timestep.
   *
   * One entry per active link, holding the workgroups assigned to it; the
   * vector's length is the number of active links and each entry is a workgroup
   * count (a link's position in the vector is its dense 0-based id, which no
   * consumer needs beyond iterating the counts). Each count prices that link's
   * contention, and this is the sole per-link quantity the cost engine consumes.
   * The ring algorithms conserve exactly (the counts sum to num_wgs via
   * ring_distribute); the direct staggered/partitioned/two-shot algorithms
   * instead report the floored remote share per link, excluding the self-step
   * workgroups that use no fabric link, so their counts sum to less than num_wgs.
   *
   * @param timestep Communication round index (0-based).
   * @param num_wgs Total workgroups participating in the collective.
   * @return Workgroup count per active link.
   */
  virtual std::vector<int> wgs_per_active_link(int timestep, int num_wgs) const = 0;

  /**
   * @brief Number of dependent communication rounds the algorithm takes.
   *
   * More steps mean more serial handshakes and, for sequential algorithms, more
   * added latency (e.g. a ring visits N-1 peers; two-shot does 2N-1 rounds).
   *
   * @return Count of timesteps in the schedule.
   */
  virtual int num_timesteps() const = 0;

  /**
   * @brief How finely each GPU's tile is sliced per timestep.
   *
   * Default: each timestep moves the whole gpu_tile (= 1). Chunked algorithms
   * override (ring, two-shot, a2a → N). This sets the per-step wire bytes.
   *
   * @return Number of chunks the per-GPU tile is split into per timestep.
   */
  virtual int chunks_per_timestep() const { return 1; }

  /**
   * @brief Whether the algorithm is ring-class for the per-step overhead heuristic.
   *
   * One of the two algorithm-kind checks the collective engine makes to identify
   * ring-class algorithms: a ring-class algorithm is eligible for the per-step
   * proxy/sync overhead heuristic. Covers AG, RS, ring AR, ring fixed.
   *
   * @return True for ring-class algorithms; false otherwise.
   */
  virtual bool is_ring_class() const { return false; }

  /**
   * @brief Whether to price the algorithm with the closed-form pipelined-ring model.
   *
   * The second algorithm-kind check: a pipelined ring uses the closed-form
   * pipelined-ring throughput model rather than per-timestep wg_tile sums.
   * Covers ring AR + ring fixed (the algorithms whose work graph carries
   * signal_t/wait_t ops).
   *
   * @return True for pipelined-ring algorithms; false otherwise.
   */
  virtual bool is_ring_pipeline() const { return false; }
};

}  // namespace origami::comm
