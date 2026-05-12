// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <vector>

#include "origami/math.hpp"
#include "origami/targets/triton/gemm.hpp"

namespace origami {

// Select work-stealing parameters for Triton.
// Empirically tuned on MI300X via autotune sweeps.
triton_ws_params_t select_triton_ws_params(size_t m, size_t n, size_t block_m, size_t block_n) {
  const size_t grid_m     = math::safe_ceil_div(m, block_m);
  const size_t num_tiles  = grid_m * math::safe_ceil_div(n, block_n);

  int counters_per_xcd;
  if (num_tiles <= 512)
    counters_per_xcd = 8;
  else if (num_tiles <= 1536)
    counters_per_xcd = 4;
  else if (num_tiles <= 2048)
    counters_per_xcd = 2;
  else
    counters_per_xcd = 1;

  int wgm = std::min(static_cast<size_t>(8), grid_m);

  return {counters_per_xcd, wgm};
}

// Compute optimal local/global tile split for hierarchical work-stealing.
// Adaptive split based on tiles-per-CU density:
//   <= 4 tiles/CU: 100% local (global counter overhead dominates)
//   > 4 tiles/CU:  local_frac decreases linearly, floor at 50%
triton_hierarchical_split_t compute_triton_hierarchical_split(
    size_t m, size_t n, size_t block_m, size_t block_n,
    size_t num_xcds, size_t n_cu) {
  const size_t total = math::safe_ceil_div(m, block_m) * math::safe_ceil_div(n, block_n);

  if (num_xcds == 0) num_xcds = 1;
  const double tiles_per_cu =
      static_cast<double>(total) / std::max(n_cu, static_cast<size_t>(1));

  double local_frac = std::max(0.5, 1.0 - std::max(0.0, tiles_per_cu - 4.0) * 0.05);
  size_t local = static_cast<size_t>(total * local_frac) / num_xcds;
  if (local == 0) local = 1;
  size_t global = total > (local * num_xcds) ? total - (local * num_xcds) : 0;

  return {local, global};
}

// Compute Triton-specific StreamK grid size.
// Full Tensile-style fractional split + K-dimension splitting + last-wave fix.
size_t compute_triton_sk_grid(size_t m, size_t n, size_t k,
                              size_t block_m, size_t block_n, size_t block_k,
                              size_t n_cu, size_t out_dtype_bits) {
  const size_t tiles_m = math::safe_ceil_div(m, block_m);
  const size_t tiles_n = math::safe_ceil_div(n, block_n);
  const size_t tiles   = tiles_m * tiles_n;
  const size_t k_iters = std::max(math::safe_ceil_div(k, block_k), static_cast<size_t>(1));

  size_t sk_grid = tiles;

  constexpr size_t kMaxWorkspace = 128 * 1024 * 1024;
  const size_t bytes_per_elem = out_dtype_bits / 8;
  const size_t tile_ws = block_m * block_n * bytes_per_elem;

  // Fractional denominators to try, in priority order
  constexpr double kFracs[] = {0.0, 0.5, 0.125, 0.2, 0.25, 1.0 / 3.0};
  constexpr int kSplitFactors[] = {8, 6, 4, 3, 2, 1};

  if (tiles > n_cu) {
    double min_even = static_cast<double>(tiles) / n_cu;
    for (double frac : kFracs) {
      size_t cand = static_cast<size_t>(tiles / (min_even + frac) + 0.5);
      if (cand == 0) continue;
      if (tiles % cand != 0 && tile_ws * cand > kMaxWorkspace)
        continue;
      if (cand <= n_cu) {
        sk_grid = cand;
        break;
      }
    }
  } else if (tiles < n_cu) {
    for (int factor : kSplitFactors) {
      size_t split = tiles * factor;
      size_t iters_per = k_iters / factor;
      if (split <= n_cu && iters_per >= 8) {
        sk_grid = split;
        break;
      }
    }
  }

  if (tiles % sk_grid != 0)
    sk_grid = tiles;

  // Last-wave compensation: when only a small partial wave (< 128 tiles)
  // would land on the trailing CUs, fall back to the largest power-of-two
  // grid <= n_cu so the workload distributes more evenly. No-op when n_cu
  // is already a power of two.
  if (tiles >= n_cu) {
    size_t remainder = tiles % n_cu;
    if (remainder > 0 && remainder < 128) {
      sk_grid = math::prev_pow2(n_cu);
    }
  }

  return sk_grid;
}

// Get default Triton tile search ranges for the given architecture and dtype.
triton_tile_ranges_t get_triton_default_tile_ranges(const hardware_t& hardware,
                                                    size_t dtype_bits) {
  std::vector<size_t> block_mn = {16, 32, 64, 128, 256};
  std::vector<size_t> block_k  = {16, 32, 64, 128, 256, 512};

  if (hardware.arch == hardware_t::architecture_t::gfx950) {
    if (dtype_bits <= 8) {
      // Restrict MN to >=32 for F8/F4 on gfx950; K range already covers
      // {16..512} including 128 and 256, so no extra K entries are needed.
      block_mn = {32, 64, 128, 256};
    }
  } else if (hardware.arch == hardware_t::architecture_t::gfx942) {
    if (dtype_bits == 8) {
      // 512 MN is genuinely additive on gfx942 F8; 128/256 K are already
      // present in the default block_k range, so don't re-push them.
      block_mn.push_back(512);
    } else if (dtype_bits < 8) {
      // F4/F6 unsupported on gfx942
    }
  }

  return {block_mn, block_k};
}

}  // namespace origami
