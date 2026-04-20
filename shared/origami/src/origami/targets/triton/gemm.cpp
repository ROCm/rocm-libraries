// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <vector>

#include "origami/math.hpp"
#include "origami/targets/triton/gemm.hpp"

namespace origami {

// Estimate Triton kernel LDS usage in bytes.
// Validated against actual Triton 3.6.0 compiled kernel metadata (n_shared_bytes).
// The compiler allocates:
//   stages == 1  →  max(A_tile_bytes, B_tile_bytes)        (A & B share the same LDS region)
//   stages >= 2  →  (stages - 1) * (A_tile_bytes + B_tile_bytes)
size_t estimate_triton_lds_bytes(dim3_t mt,
                                 data_type_t a_dtype,
                                 data_type_t b_dtype,
                                 int num_stages) {
  const size_t bytes_a = static_cast<size_t>(data_type_to_bytes(a_dtype));
  const size_t bytes_b = static_cast<size_t>(data_type_to_bytes(b_dtype));

  const size_t a_tile = mt.m * mt.k * bytes_a;
  const size_t b_tile = mt.k * mt.n * bytes_b;

  if (num_stages <= 1)
    return std::max(a_tile, b_tile);

  return static_cast<size_t>(num_stages - 1) * (a_tile + b_tile);
}

// Check if tile fits in LDS for Triton kernels.
bool check_triton_lds_capacity(const hardware_t& hardware,
                               dim3_t mt,
                               data_type_t a_dtype,
                               data_type_t b_dtype,
                               int num_stages) {
  return estimate_triton_lds_bytes(mt, a_dtype, b_dtype, num_stages) <= hardware.lds_capacity;
}

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
    size_t num_xcds, size_t n_cu, size_t cu_per_l2) {
  const size_t total = math::safe_ceil_div(m, block_m) * math::safe_ceil_div(n, block_n);

  if (num_xcds == 0) num_xcds = 1;
  const size_t hw_cus = num_xcds * (cu_per_l2 > 0 ? cu_per_l2 : 1);
  const double tiles_per_cu = static_cast<double>(total) / std::max(hw_cus, static_cast<size_t>(1));

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

  // Last-wave compensation for gfx942 CU counts
  if (tiles >= n_cu) {
    size_t remainder = tiles % n_cu;
    if (remainder > 0 && remainder < 128 &&
        (n_cu == 304 || n_cu == 80 || n_cu == 64)) {
      sk_grid = (n_cu == 304) ? 256 : 64;
    }
  }

  return sk_grid;
}

// Get default Triton tile search ranges for the given architecture and dtype.
triton_tile_ranges_t get_triton_default_tile_ranges(const hardware_t& hardware,
                                                    size_t dtype_bits,
                                                    size_t k) {
  std::vector<size_t> block_mn = {16, 32, 64, 128, 256};
  std::vector<size_t> block_k  = {16, 32, 64, 128, 256, 512};

  if (hardware.arch == hardware_t::architecture_t::gfx950) {
    if (dtype_bits <= 8) {
      block_mn = {32, 64, 128, 256};
      block_k.push_back(k % 256 == 0 ? 256 : 128);
    }
  } else if (hardware.arch == hardware_t::architecture_t::gfx942) {
    if (dtype_bits == 8) {
      block_mn.push_back(512);
      block_k.push_back(128);
      block_k.push_back(256);
    } else if (dtype_bits < 8) {
      // F4/F6 unsupported on gfx942
    }
  }

  return {block_mn, block_k};
}

}  // namespace origami
