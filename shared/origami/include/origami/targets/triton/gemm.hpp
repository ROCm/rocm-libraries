// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief Result of Triton work-stealing parameter selection.
 */
struct triton_ws_params_t {
  int counters_per_xcd;
  int workgroup_mapping;
};

/**
 * @brief Select work-stealing parameters for Triton based on tile count.
 *
 * Empirically tuned on MI300X.
 *
 * @param m Problem M dimension
 * @param n Problem N dimension
 * @param block_m Tile M dimension
 * @param block_n Tile N dimension
 * @return triton_ws_params_t Work-stealing parameters.
 */
triton_ws_params_t select_triton_ws_params(size_t m, size_t n, size_t block_m, size_t block_n);

/**
 * @brief Result of Triton hierarchical split computation.
 */
struct triton_hierarchical_split_t {
  size_t local_per_xcd;
  size_t global_tiles;
};

/**
 * @brief Compute optimal local/global tile split for hierarchical work-stealing.
 *
 * @param m        Problem M dimension
 * @param n        Problem N dimension
 * @param block_m  Tile M dimension
 * @param block_n  Tile N dimension
 * @param num_xcds Number of XCDs (used for the local/global tile split)
 * @param n_cu     Total CU count (must be > 0)
 * @return triton_hierarchical_split_t Split parameters.
 */
triton_hierarchical_split_t compute_triton_hierarchical_split(
    size_t m, size_t n, size_t block_m, size_t block_n,
    size_t num_xcds, size_t n_cu);

/**
 * @brief Compute Triton-specific StreamK grid size.
 *
 * @param m Problem M dimension
 * @param n Problem N dimension
 * @param k Problem K dimension
 * @param block_m Tile M dimension
 * @param block_n Tile N dimension
 * @param block_k Tile K dimension
 * @param n_cu Total CU count
 * @param out_dtype_bits Bits per output element
 * @return size_t StreamK grid size.
 */
size_t compute_triton_sk_grid(size_t m, size_t n, size_t k,
                              size_t block_m, size_t block_n, size_t block_k,
                              size_t n_cu, size_t out_dtype_bits);

/**
 * @brief Architecture-specific tile search ranges for Triton config generation.
 */
struct triton_tile_ranges_t {
  std::vector<size_t> block_mn;
  std::vector<size_t> block_k;
};

/**
 * @brief Get default Triton tile search ranges for the given architecture and dtype.
 *
 * Returns architecture-specific block_m/n and block_k ranges that should be used
 * when generating candidate Triton GEMM configs. Accounts for:
 *   - gfx950 F8/F4: restricted MN to 32-256
 *   - gfx942 F8:    adds 512 to the MN range
 *   - Default:      MN in {16,32,64,128,256}, K in {16,32,64,128,256,512}
 *
 * @param hardware   Hardware characteristics
 * @param dtype_bits Bits per element of the narrower input dtype
 * @return triton_tile_ranges_t Tile search ranges
 */
triton_tile_ranges_t get_triton_default_tile_ranges(const hardware_t& hardware,
                                                    size_t dtype_bits);

}  // namespace origami
