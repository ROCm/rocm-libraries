// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <vector>

#include "origami/math.hpp"
#include "origami/streamk.hpp"
#include "origami/targets/triton/gemm.hpp"
#include "origami/types.hpp"

namespace origami {

namespace {

// ----- compute_triton_sk_grid heuristic constants -----

// Caller-side workspace cap (matches Tensile's StreamK budget).
constexpr std::size_t kMaxWorkspaceBytes = 128ull * 1024 * 1024;

// Fractional grid candidates explored when tiles > N_CU.
const std::vector<double> kFracs = {0.0, 0.5, 0.125, 0.2, 0.25, 1.0 / 3.0};

// K-split factors explored when tiles < N_CU.
const std::vector<std::size_t> kSplitFactors = {8, 6, 4, 3, 2, 1};

// When the trailing partial wave on the n_cu mapping holds fewer than this
// many tiles, fall back to prev_pow2(n_cu) so the workload distributes more
// evenly. Empirically tuned on MI300X.
constexpr std::size_t kSmallLastWaveTileCount = 128;

}  // namespace

// Compute Triton-specific StreamK grid size.
//
// Reuses the shared streamk fractional-grid and k-split helpers (no logic
// duplication with streamk::grid_k_split_aware), and adds the Triton-only
// "last partial wave -> prev_pow2(n_cu)" compensation that was empirically
// tuned on MI300X.
//
// Differences from the previous primitive-arg signature:
//   - Tile count is batch-aware via streamk::compute_number_of_output_tiles
//     (fixes a latent under-count for batched problems).
//   - Per-tile workspace bytes are derived sub-byte-safely from the C dtype
//     (fixes a latent zero-divide for F4/F6 outputs).
//   - Falls back to config.workspace_size_per_elem_c when the caller has
//     populated it (matches Tensile semantics in streamk::grid_k_split_aware).
std::size_t compute_triton_sk_grid(const problem_t&  problem,
                                   const config_t&   config,
                                   const hardware_t& hardware) {
  const std::size_t n_cu = hardware.N_CU;
  if (n_cu == 0) return 0;

  const std::size_t tiles = streamk::compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
  if (tiles == 0) return 0;

  const std::size_t k_iters = std::max(math::safe_ceil_div(problem.size.k, config.mt.k),
                                       static_cast<std::size_t>(1));

  // Per-tile workspace bytes. Prefer a caller-supplied value; otherwise derive
  // sub-byte-safely from problem.c_dtype.
  std::size_t tile_ws = 0;
  if (config.workspace_size_per_elem_c > 0) {
    tile_ws = config.mt.mn() * config.workspace_size_per_elem_c;
  } else {
    const std::size_t bits_c = datatype_to_bits(problem.c_dtype);
    tile_ws                  = math::safe_ceil_div(config.mt.mn() * bits_c, std::size_t{8});
  }

  std::size_t sk_grid = tiles;
  if (tiles > n_cu) {
    if (std::size_t cand = streamk::pick_fractional_grid(
            tiles, n_cu, tile_ws, kMaxWorkspaceBytes, kFracs)) {
      sk_grid = cand;
    }
  } else if (tiles < n_cu) {
    if (std::size_t split = streamk::pick_k_split(
            tiles, n_cu, k_iters, kSplitFactors, /*min_iters_per_cu=*/8)) {
      sk_grid = split;
    }
  }

  if (sk_grid == 0 || tiles % sk_grid != 0) sk_grid = tiles;

  // Triton-only last-wave compensation: when only a small partial wave
  // (< kSmallLastWaveTileCount tiles) would land on the trailing CUs, fall
  // back to the largest power-of-two grid <= n_cu so the workload distributes
  // more evenly. No-op when n_cu is already a power of two.
  if (tiles >= n_cu) {
    const std::size_t remainder = tiles % n_cu;
    if (remainder > 0 && remainder < kSmallLastWaveTileCount) {
      sk_grid = math::prev_pow2(n_cu);
    }
  }

  return sk_grid;
}

// Default Triton tile candidate configs for the given problem and hardware.
//
// Architecture-aware cross-product of (block_m, block_n, block_k) candidates
// expanded into a flat std::vector<config_t>. The MN range is gated by the
// narrower of (problem.a_dtype, problem.b_dtype) per the MFMA shape support
// matrix; the K range is the same for every (arch, dtype). Only mt is
// populated (via dim3_t aggregate-init); the caller is responsible for
// setting mi and any other config fields per its selection policy.
std::vector<config_t> get_triton_default_configs(const problem_t&  problem,
                                                 const hardware_t& hardware) {
  const int narrow_bits =
      std::min(datatype_to_bits(problem.a_dtype), datatype_to_bits(problem.b_dtype));

  std::vector<std::size_t> block_mn;
  if (hardware.arch == hardware_t::architecture_t::gfx950 && narrow_bits <= 8) {
    // gfx950 F8/F4: no 16-MN MFMA available for sub-16-bit inputs.
    block_mn = {32, 64, 128, 256};
  } else if (hardware.arch == hardware_t::architecture_t::gfx942 && narrow_bits == 8) {
    // gfx942 F8: additive 512-MN MFMA available on top of the default range.
    block_mn = {16, 32, 64, 128, 256, 512};
  } else {
    // Default search space. F4/F6 on gfx942 falls here; the dtype is not
    // natively supported on gfx942 MFMA, so consumers should filter
    // empirically downstream.
    block_mn = {16, 32, 64, 128, 256};
  }

  static const std::vector<std::size_t> block_k = {16, 32, 64, 128, 256, 512};

  std::vector<config_t> configs;
  configs.reserve(block_mn.size() * block_mn.size() * block_k.size());
  for (std::size_t bm : block_mn) {
    for (std::size_t bn : block_mn) {
      for (std::size_t bk : block_k) {
        config_t c;
        c.mt = dim3_t{bm, bn, bk};
        configs.push_back(c);
      }
    }
  }
  return configs;
}

}  // namespace origami
