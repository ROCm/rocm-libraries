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

// Public algorithm builders: the one place that maps a (collective, algorithm)
// pair to a concrete implementation. Callers work entirely through these
// factories and the collective_algorithm_t base — the concrete classes in
// direct.hpp / ring.hpp are an implementation detail of resolve.cpp.
#pragma once

#include "origami/comm/algorithms/base.hpp"
#include "origami/comm/types.hpp"

#include <memory>

namespace origami::comm {

/**
 * @brief Build the all-gather algorithm (ring).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a ring all-gather algorithm.
 */
std::unique_ptr<collective_algorithm_t> allgather_algorithm(int num_gpus);

/**
 * @brief Build the reduce-scatter algorithm (ring).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a ring reduce-scatter algorithm.
 */
std::unique_ptr<collective_algorithm_t> reduce_scatter_algorithm(int num_gpus);

/**
 * @brief Build the broadcast algorithm (ring).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a ring broadcast algorithm.
 */
std::unique_ptr<collective_algorithm_t> broadcast_algorithm(int num_gpus);

/**
 * @brief Build the one-shot all-reduce algorithm (all-to-same with pull+reduce work graph).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to an all-to-same algorithm configured for one-shot all-reduce.
 */
std::unique_ptr<collective_algorithm_t> allreduce_one_shot_algorithm(int num_gpus);

/**
 * @brief Build the two-shot all-reduce algorithm (reduce-scatter then all-gather).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a two-shot all-reduce algorithm.
 */
std::unique_ptr<collective_algorithm_t> allreduce_two_shot_algorithm(int num_gpus);

/**
 * @brief Build the ring all-reduce algorithm (bandwidth-optimal pipelined ring).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a ring all-reduce algorithm.
 */
std::unique_ptr<collective_algorithm_t> allreduce_ring_algorithm(int num_gpus);

/**
 * @brief Build the all-to-all algorithm (direct, pid-staggered with load+push work graph).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a pid-staggered algorithm configured for all-to-all.
 */
std::unique_ptr<collective_algorithm_t> alltoall_algorithm(int num_gpus);

/**
 * @brief Map a (collective, algorithm) pair to a concrete algorithm implementation.
 *
 * The single place the validity rule lives: an algorithm is a valid config only
 * if it is defined for the problem's collective. `automatic` maps to each
 * collective's canonical algorithm (preserving the historical defaults); a named
 * value selects a specific one and any undefined (collective, algorithm) pair
 * throws rather than silently costing a nonsense schedule.
 *
 *   all_gather / reduce_scatter / broadcast : ring only
 *   all_reduce                              : two_shot (default), one_shot, ring
 *   all_to_all                              : direct (pid-staggered) only
 *
 * @param collective Collective primitive to implement.
 * @param algorithm Requested algorithm, or algorithm_t::automatic for the canonical default.
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to the resolved algorithm.
 * @throws std::invalid_argument if the algorithm is not valid for the collective.
 */
std::unique_ptr<collective_algorithm_t> resolve_algorithm(primitive_t collective,
                                                          algorithm_t algorithm,
                                                          int num_gpus);

/**
 * @brief Resolve the algorithm from a problem/config bundle (the engine-facing overload).
 *
 * The collective comes from the problem (correctness), the algorithm from the
 * config (performance), and num_gpus is the communicator size on the problem.
 * This is the form the latency engine calls; the scalar overload above stays for
 * unit tests that probe (collective, algorithm) pairs directly.
 *
 * @param problem Communication problem (provides the collective and num_gpus).
 * @param config Communication config (provides the requested algorithm).
 * @return Owning pointer to the resolved algorithm.
 * @throws std::invalid_argument if the algorithm is not valid for the collective.
 */
std::unique_ptr<collective_algorithm_t> resolve_algorithm(const comm_problem_t& problem,
                                                          const comm_config_t& config);

}  // namespace origami::comm
