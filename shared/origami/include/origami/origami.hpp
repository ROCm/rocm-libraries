/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#pragma once

#include <functional>
#include <set>
#include <tuple>
#include <vector>

#include "origami/hardware.hpp"
#include "origami/types.hpp"
#include "origami/origami_export.h"

namespace origami {

/**
 * @brief Based on the provided problem and configs; selects the best config.
 *
 * With the default (empty) pipeline this performs a single ranking pass that
 * honors each config's own target and prediction fidelity. Passing a non-empty
 * pipeline runs the coarse-to-fine cascade instead (see @ref rank_configs).
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs Vector of all possible valid configurations.
 * @param model Operation model for the single-pass case (gemm or attention).
 * @param pipeline Optional multi-phase pipeline; empty means a single pass.
 * @return prediction_result_t Configurations with best latency.
 */
ORIGAMI_EXPORT prediction_result_t select_config(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const std::vector<config_t>& configs,
                                  model_t model = model_t::gemm,
                                  const ranking_pipeline_t& pipeline = ranking_pipeline_t{});

/**
 * @brief Select best workgroup-mapping for the given tile size.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param skGrid StreamK grid size.
 * @return workgroup_mapping_t Workgroup mapping parameters (wgmxccchunk, wgmxcc, wgm).
 */
ORIGAMI_EXPORT workgroup_mapping_t select_workgroup_mapping(const problem_t& problem,
                                             const hardware_t& hardware,
                                             const config_t& config,
                                             size_t skGrid);

/**
 * @brief Select best staggerU for the given tile size.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration
 * @param skGrid StreamK grid size
 * @param wgm Workgroup mapping
 * @return staggerU_t StaggerU parameters (staggerUMapping, staggerU, staggerUStrideShift).
 */
ORIGAMI_EXPORT staggerU_t select_staggerU(const problem_t& problem,
                           const hardware_t& hardware,
                           const config_t& config,
                           size_t skGrid,
                           int32_t wgm);

/**
 * @brief Rank configurations based on predicted performance.
 *
 * Two modes, selected by @p pipeline:
 * - Empty pipeline (default): a single ranking pass. Each config resolves its
 *   own cost model from @p model plus its target and prediction fidelity, is
 *   gated for feasibility, scored, and the whole set is ranked (best first).
 * - Non-empty pipeline: a coarse-to-fine cascade. Each phase scores the current
 *   survivors with the cost model fixed by its (model, target, fidelity), drops
 *   infeasible configs, then prunes per its policy before the next phase. The
 *   result is ranked by the final phase's latencies. (@p model is unused here;
 *   each phase carries its own.) A single no-prune phase is thus the cascade
 *   form of a single pass.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs List of candidate kernel configurations to rank
 * @param model Operation model for the single-pass case (gemm or attention)
 * @param pipeline Optional multi-phase pipeline; empty means a single pass
 * @return std::vector<prediction_result_t> Configurations with latencies ranked by
 * performance (best first). If every candidate is rejected, returns a ranking of the
 * rejected configs (max latency) instead of throwing.
 * @throws std::runtime_error if @p configs is empty.
 */
ORIGAMI_EXPORT std::vector<prediction_result_t> rank_configs(
    const problem_t& problem,
    const hardware_t& hardware,
    const std::vector<config_t>& configs,
    model_t model = model_t::gemm,
    const ranking_pipeline_t& pipeline = ranking_pipeline_t{});

/**
 * @brief Select best configuration based only on M, N, K dimensions with default settings.
 *
 * @param M Problem dimension M
 * @param N Problem dimension N
 * @param K Problem dimension K
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs List of candidate configurations
 * @return prediction_result_t Configurations with best latency.
 */
ORIGAMI_EXPORT prediction_result_t select_config_mnk(std::size_t M,
                                      std::size_t N,
                                      std::size_t K,
                                      const hardware_t& hardware,
                                      const std::vector<config_t>& configs);

/**
 * @brief Select top K configurations based on performance ranking.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs List of candidate configurations to rank
 * @param topk Number of top configurations to return
 * @return std::vector<prediction_result_t> Top K configurations ranked by performance (best first)
 */
ORIGAMI_EXPORT std::vector<prediction_result_t> select_topk_configs(const problem_t& problem,
                                                     const hardware_t& hardware,
                                                     const std::vector<config_t>& configs,
                                                     std::size_t topk,
                                                     model_t model = model_t::gemm);

/**
 * @brief Build a two-phase estimation-then-simulation cascade pipeline.
 *
 * Phase 1 runs the cheap analytical estimation and keeps the best
 * @p topk_after_estimation configs; phase 2 re-ranks those survivors with the
 * simulation model (no further pruning). Simulation must be registered for
 * (@p model, @p target); today that is gemm + tensilelite.
 *
 * @param model Operation model (gemm or attention).
 * @param target Target backend whose kernels are modeled.
 * @param topk_after_estimation Survivors kept after the estimation phase.
 * @return ranking_pipeline_t The configured cascade pipeline.
 */
ORIGAMI_EXPORT ranking_pipeline_t make_cascade_pipeline(model_t model,
                                                        target_t target,
                                                        std::size_t topk_after_estimation);

/**
 * @brief Given a latency, compute the achieved throughput in gflops.
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param problem Problem description (M, N, K, etc.)
 * @param latency Kernel latency.
 * @return double Throughput in gflops/s.
 */
ORIGAMI_EXPORT double compute_perf_gflops(const hardware_t& hardware,
                           const problem_t& problem,
                           const double latency);

}  // namespace origami
