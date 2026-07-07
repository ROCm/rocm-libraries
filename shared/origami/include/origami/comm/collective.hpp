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
// End-to-end collective latency: composes the per-WG-tile atom from
// latency.hpp over a collective's full communication schedule.
//
// The composition depends on how the schedule's timesteps relate in time,
// which is itself a property of the algorithm:
//
//   1. Sequential timesteps (latency-composed). The timesteps are data-
//      dependent — each must finish before the next begins (e.g. the two
//      phases of a two-shot all-reduce). They may use different links, so the
//      total is the *sum* of per-timestep tile latencies. Small messages live
//      here: the cost is a chain of fill/drain/handshake constants.
//
//   2. Pipelined ring (throughput-composed). All steps stream over the same
//      ring link and the steps overlap across WGs, so the ring behaves as one
//      long pipe. Latency does not add up step-by-step; instead total wire
//      bytes are divided by the *aggregate* sustainable throughput, then a
//      fixed sync + per-step overhead is added. Large messages live here: the
//      cost approaches bytes ÷ bandwidth.
//
// Every path also adds the once-per-launch kernel overhead, which is what
// dominates the sub-kilobyte regime.
#pragma once

#include "origami/comm/algorithms.hpp"
#include "origami/comm/hardware.hpp"
#include "origami/comm/heuristics.hpp"
#include "origami/comm/latency.hpp"
#include "origami/comm/primitives.hpp"
#include "origami/comm/types.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace origami::comm {

// ─── ring-step heuristic ────────────────────────────────────────
/**
 * @brief Per-launch ring-step proxy/handshake overhead, in cycles.
 *
 * Empirical, CPU-mediated overhead that the bandwidth model cannot see, charged
 * once per ring timestep. Returns 0 for non-ring algorithms.
 *
 * @param primitive Collective being run (keys the per-step heuristic).
 * @param algorithm Resolved collective algorithm (provides timestep count and class).
 * @param heur Tunable heuristic parameters.
 * @param fabric Fabric hardware, whose clock converts the host-time heuristic to
 *        cycles (the heuristic is stored in nanoseconds, so this carries no
 *        fixed-clock assumption).
 * @return double Total ring-step overhead in GPU cycles (0 if not ring-class).
 */
double ring_step_overhead_cycles(primitive_t primitive,
                                 const collective_algorithm_t& algorithm,
                                 const heuristics_t& heur,
                                 const comm_hardware_t& fabric);

// The (collective, algorithm) → implementation resolution lives in
// algorithms.hpp as resolve_algorithm(); compute_collective_latency_for_rank
// calls it below. No default factory is needed here.

/**
 * @brief Throughput-composed latency of a pipelined ring collective, in cycles.
 *
 * All steps stream over the same ring link and overlap across WGs, so the ring
 * behaves as one long pipe: total wire bytes ÷ aggregate sustainable throughput
 * (the slower of fabric and local HBM), plus serial per-step sync and empirical
 * per-step overhead, on top of the fixed kernel launch floor.
 *
 * @param algorithm Resolved ring algorithm (timesteps, chunks, link schedule).
 * @param problem Collective problem (shape, dtype, world size, collective op).
 * @param config Communication kernel configuration (WG count, load width, etc.).
 * @param system GPU + fabric hardware description (@see origami::comm::system_t).
 * @param my_rank Rank whose link schedule is sampled for the sync-op count.
 * @param heur Tunable heuristic parameters.
 * @return double Predicted latency for this rank in GPU cycles (not microseconds;
 *         the cycles→µs conversion happens at the public boundary, predict_row).
 */
double compute_ring_latency(const collective_algorithm_t& algorithm,
                            const comm_problem_t& problem,
                            const comm_config_t& config,
                            const system_t& system,
                            int my_rank,
                            const heuristics_t& heur);

/**
 * @brief Latency-composed latency of a sequential-timestep collective, in cycles.
 *
 * Timesteps are data-dependent, so their latencies add up; within a timestep the
 * links run in parallel, so a timestep costs the *slowest* link (max over links),
 * with "self" steps bound by local HBM instead of a fabric link. Adds per-step
 * overhead on top of the fixed kernel launch floor.
 *
 * @param algorithm Resolved sequential algorithm (timesteps, link schedule).
 * @param problem Collective problem (shape, dtype, world size, collective op).
 * @param config Communication kernel configuration (WG count, load width, etc.).
 * @param system GPU + fabric hardware description (@see origami::comm::system_t).
 * @param my_rank Rank whose per-timestep link schedule is evaluated.
 * @param heur Tunable heuristic parameters.
 * @return double Predicted latency for this rank in GPU cycles (not microseconds;
 *         the cycles→µs conversion happens at the public boundary, predict_row).
 */
double compute_sequential_latency(const collective_algorithm_t& algorithm,
                                  const comm_problem_t& problem,
                                  const comm_config_t& config,
                                  const system_t& system,
                                  int my_rank,
                                  const heuristics_t& heur);

/**
 * @brief Predicted GPU cycles for *one* rank's timeline.
 *
 * This is the per-rank atom and the diagnostic entry point: call it directly to
 * inspect whether ranks diverge. Caller converts cycles→µs at the public
 * boundary.
 *
 * The operation comes from problem.collective (what to compute) and the
 * implementation from config.algorithm (how) — resolve_algorithm maps that pair
 * to a concrete algorithm (or rejects an invalid pair). A non-null
 * config.algorithm_override bypasses resolution with a caller-supplied object.
 * This is the problem/config split: correctness inputs in the problem,
 * performance inputs in the config.
 *
 * @param problem Collective problem (shape, dtype, world size, collective op).
 * @param config Communication kernel configuration; algorithm_override, if set,
 *        bypasses resolve_algorithm.
 * @param system GPU + fabric hardware description (@see origami::comm::system_t).
 * @param my_rank Rank whose timeline is predicted.
 * @param heur Tunable heuristic parameters (defaults to DEFAULT_HEURISTICS).
 * @return double Predicted latency for this rank in GPU cycles.
 */
double compute_collective_latency_for_rank(const comm_problem_t& problem,
                                           const comm_config_t& config,
                                           const system_t& system,
                                           int my_rank,
                                           const heuristics_t& heur = DEFAULT_HEURISTICS);

/**
 * @brief Predicted GPU cycles for the whole collective.
 *
 * The operation completes only when its slowest participant does, so the cost is
 * the *max* of every rank's timeline — this loop is where rank asymmetry, if any
 * algorithm ever introduces it, would surface.
 *
 * Shortcut: with heur.assume_rank_symmetry the loop collapses to rank 0 alone
 * (see heuristics_t — exact for the rank-symmetric algorithms we ship today, an
 * N× speedup). Default is the honest max so the engine stays correct for any
 * future asymmetric algorithm without a flag change.
 *
 * @param problem Collective problem (shape, dtype, world size, collective op).
 * @param config Communication kernel configuration.
 * @param system GPU + fabric hardware description (@see origami::comm::system_t).
 * @param heur Tunable heuristic parameters (defaults to DEFAULT_HEURISTICS).
 * @return double Predicted latency for the whole collective in GPU cycles.
 */
double compute_collective_latency(const comm_problem_t& problem,
                                  const comm_config_t& config,
                                  const system_t& system,
                                  const heuristics_t& heur = DEFAULT_HEURISTICS);

/**
 * @brief Byte-level public entry point: predict one collective call's latency
 *        in microseconds.
 *
 * Its job is to translate a benchmark row's conventions into a
 * comm_problem_t/comm_config_t and then defer to the model above.
 *
 * The one subtlety it owns is the message-size convention: most collectives
 * report msg_bytes as the per-rank buffer, but reduce_scatter reports the full
 * pre-scatter buffer, so its per-rank share is msg_bytes / world_size. When no
 * explicit [M,N] shape is given, the buffer is treated as a 1×N row of bf16
 * elements. cl/sync contention, algorithm, and unit conversion are all
 * delegated; the cycles→µs conversion happens here, at the boundary.
 *
 * @param primitive Collective name (e.g. "all_reduce"); mapped to an enum here.
 * @param msg_bytes Message size in the benchmark's convention (per-rank buffer,
 *        except reduce_scatter which passes the aggregate pre-scatter buffer).
 * @param world_size Number of participating ranks.
 * @param nchannels Number of channels/workgroups driving the collective.
 * @param system GPU + fabric hardware description (@see origami::comm::system_t).
 * @param M Optional row count of the logical tensor; 0 means derive from msg_bytes.
 * @param N Optional column count of the logical tensor; 0 means derive from msg_bytes.
 * @param split_dim Sharded axis (0 = rows, 1 = columns).
 * @param heur Tunable heuristic parameters (defaults to DEFAULT_HEURISTICS).
 * @return double Predicted latency in microseconds.
 */
double predict_row(std::string_view primitive,
                   std::size_t msg_bytes,
                   int world_size,
                   int nchannels,
                   const system_t& system,
                   std::size_t M            = 0,
                   std::size_t N            = 0,
                   int split_dim            = 0,
                   const heuristics_t& heur = DEFAULT_HEURISTICS);

}  // namespace origami::comm
