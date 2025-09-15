/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/analytical/Hardware.hpp>
#include <vector>

namespace TensileLite
{
    namespace analytical
    {
        namespace streamk
        {
            enum class ReductionType
            {
                // BasicReduction,
                Tree,
                Parallel,
                // AtomicReduction,
                Count,
                None = Count
            };

            inline ReductionType intToReductionType(int rt)
            {
                return (ReductionType)rt;
            }

            size_t get_streamk_workspace(
                size_t x,
                size_t y,
                size_t mt_m,
                size_t mt_n,
                size_t bpe_c,
                size_t grid,
                size_t tiles,
                ReductionType reduction);

            ReductionType select_streamk_reduction(
                size_t x,
                size_t y,
                size_t z,
                size_t batch,
                size_t mt_m,
                size_t mt_n,
                size_t mt_k,
                const Hardware& analytical_hardware,
                int dynamic_grid_version);

            size_t select_streamk_grid(size_t x,
                            size_t y,
                            size_t z,
                            size_t batch,
                            bool            trans_a,
                            bool            trans_b,
                            size_t          element_size_A,
                            size_t          element_size_B,
                            size_t          element_size_out,
                            DataType        mi_datatype,
                            size_t          workspace_size,
                            size_t          mt_m,
                            size_t          mt_n,
                            size_t          mt_k,
                            size_t          mi_m,
                            size_t          mi_n,
                            size_t          mi_k,
                            size_t          workgroup_mapping,
                            size_t          workspace_size_per_elem_c,
                            int             occupancy,
                            const Hardware& analytical_hardware,
                            int dynamic_grid_version,
                            ReductionType reduction_strategy);
                            // max workspace
                            // max cus

        } // namespace streamk
    }
}
