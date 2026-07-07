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

#include "origami/comm/algorithms/resolve.hpp"

#include "origami/comm/algorithms/direct.hpp"
#include "origami/comm/algorithms/ring.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace origami::comm {

// All-gather: ring.
std::unique_ptr<collective_algorithm_t> allgather_algorithm(int num_gpus) {
  return std::make_unique<ring_all_gather_algorithm_t>(num_gpus);
}

// Reduce-scatter: ring.
std::unique_ptr<collective_algorithm_t> reduce_scatter_algorithm(int num_gpus) {
  return std::make_unique<ring_reduce_scatter_algorithm_t>(num_gpus);
}

// Broadcast: ring.
std::unique_ptr<collective_algorithm_t> broadcast_algorithm(int num_gpus) {
  return std::make_unique<ring_broadcast_algorithm_t>(num_gpus);
}

// One-shot all-reduce: all-to-same schedule with a pull+reduce hop.
std::unique_ptr<collective_algorithm_t> allreduce_one_shot_algorithm(int num_gpus) {
  auto wg = [](int peer, int my_rank, int num_gpus, bool is_self) -> std::vector<op_t> {
    if (is_self) return {load_t{}, reduce_t{}};
    return {pull_t{peer}, reduce_t{}};
  };
  return std::make_unique<all_to_same_algorithm_t>(num_gpus, wg);
}

// Two-shot all-reduce: reduce-scatter shot then all-gather shot.
std::unique_ptr<collective_algorithm_t> allreduce_two_shot_algorithm(int num_gpus) {
  return std::make_unique<two_shot_all_reduce_algorithm_t>(num_gpus);
}

// Ring all-reduce: bandwidth-optimal pipelined ring.
std::unique_ptr<collective_algorithm_t> allreduce_ring_algorithm(int num_gpus) {
  return std::make_unique<ring_all_reduce_algorithm_t>(num_gpus);
}

// All-to-all: pid-staggered schedule with a load+push hop.
std::unique_ptr<collective_algorithm_t> alltoall_algorithm(int num_gpus) {
  auto wg = [](int peer, int my_rank, int num_gpus, bool is_self) -> std::vector<op_t> {
    if (is_self) return {load_t{}, store_t{}};
    return {load_t{}, push_t{peer}};
  };
  return std::make_unique<pid_staggered_algorithm_t>(num_gpus, wg);
}

// Map a (collective, algorithm) pair to a concrete algorithm; `automatic` picks each
// collective's canonical default, and any undefined pairing throws rather than costing nonsense.
std::unique_ptr<collective_algorithm_t> resolve_algorithm(primitive_t collective,
                                                          algorithm_t algorithm,
                                                          int num_gpus) {
  const bool automatic = (algorithm == algorithm_t::automatic);
  switch (collective) {
    case primitive_t::all_gather:
      if (automatic || algorithm == algorithm_t::ring) return allgather_algorithm(num_gpus);
      break;
    case primitive_t::reduce_scatter:
      if (automatic || algorithm == algorithm_t::ring) return reduce_scatter_algorithm(num_gpus);
      break;
    case primitive_t::broadcast:
      if (automatic || algorithm == algorithm_t::ring) return broadcast_algorithm(num_gpus);
      break;
    case primitive_t::all_reduce:
      switch (algorithm) {
        case algorithm_t::automatic:
        case algorithm_t::two_shot: return allreduce_two_shot_algorithm(num_gpus);
        case algorithm_t::one_shot: return allreduce_one_shot_algorithm(num_gpus);
        case algorithm_t::ring: return allreduce_ring_algorithm(num_gpus);
        default: break;
      }
      break;
    case primitive_t::all_to_all:
      if (automatic || algorithm == algorithm_t::direct) return alltoall_algorithm(num_gpus);
      break;
  }
  throw std::invalid_argument(std::string{"algorithm '"} + std::string{algorithm_name(algorithm)} +
                              "' is not a valid implementation of collective '" +
                              std::string{primitive_name(collective)} + "'");
}

// Engine-facing overload: collective + num_gpus come from the problem, algorithm from the config.
std::unique_ptr<collective_algorithm_t> resolve_algorithm(const comm_problem_t& problem,
                                                          const comm_config_t& config) {
  return resolve_algorithm(problem.collective, config.algorithm, problem.num_gpus);
}

}  // namespace origami::comm
