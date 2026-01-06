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

#include "origami/hardware.hpp"
#include "origami/types.hpp"

#include <vector>

namespace origami {
namespace streamk {
/**
 * @brief Number of output tiles.
 *
 * @param mt_m Tile size in M-dimension.
 * @param mt_n Tile size in N-dimension.
 * @param m Matrix's m-dimension.
 * @param n Matrix's n-dimension.
 * @param batch Number of batches.
 * @return size_t Total number of output tiles.
 */
size_t compute_number_of_output_tiles(size_t mt_m, size_t mt_n, size_t m, size_t n, size_t batch);

/**
 * @brief Select the best reduction strategy for StreamK.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param algorithm Grid selection algorithm
 * @return reduction_t Selected reduction strategy
 */
reduction_t select_reduction(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             grid_selection_t algorithm);

/**
 * @brief Based on the provided kernel config, select the best grid dimension.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_selection_t grid selection algorithm (@see origami::grid_selection_t)
 * @param max_cus Maximum number of CUs to use.
 * @return size_t Dimensions of the grid launched.
 */
size_t select_grid_size(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config,
                        grid_selection_t algorithm,
                        size_t max_cus = 0);

}  // namespace streamk
}  // namespace origami
