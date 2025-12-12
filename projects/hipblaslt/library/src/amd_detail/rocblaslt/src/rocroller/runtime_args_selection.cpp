/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "analytical_utils.hpp"
#include "gemm.hpp"
#include "runtime_args_selection.hpp"

#include "origami/streamk.hpp"

int chooseStreamKGridSize(std::shared_ptr<GemmKernel>        gemm,
                          const RocblasltContractionProblem& prob)
{
    const origami::hardware_t analytical_hardware = origami::hardware_t::get_hardware_for_device(0);

    const origami::grid_selection_t DEFAULT_DYNAMIC_MODE = origami::grid_selection_t::k_split_aware;

    //setting max_cu's
    size_t max_cus = analytical_hardware.N_CU;

    size_t elementSizeA_bits = rocRoller::DataTypeInfo::Get(gemm->params->kernelType.typeA).elementBits;
    size_t elementSizeB_bits = rocRoller::DataTypeInfo::Get(gemm->params->kernelType.typeB).elementBits;
    size_t elementSizeAcc = rocRoller::DataTypeInfo::Get(gemm->params->kernelType.typeAcc).elementBytes;

    origami::problem_t origami_problem = {
        .size = {prob.m, prob.n, prob.k},
        .batch = prob.batch_count,
        .a_dtype = rocroller_type_to_analytical_type(gemm->params->kernelType.typeA),
        .b_dtype = rocroller_type_to_analytical_type(gemm->params->kernelType.typeB),
        .mi_dtype = rocroller_type_to_analytical_type(elementSizeA_bits < elementSizeB_bits ? gemm->params->kernelType.typeB : gemm->params->kernelType.typeA),
    };
    origami::config_t origami_config = {
        .mt = {
            static_cast<size_t>(gemm->params->workgroupTile.m), 
            static_cast<size_t>(gemm->params->workgroupTile.n), 
            static_cast<size_t>(gemm->params->workgroupTile.k)
        },
        .occupancy = gemm->occupancy,
        .workspace_size = prob.workspaceSize,
        .workspace_size_per_elem_c = elementSizeAcc,
    };

    auto reduction_type = origami::streamk::select_reduction(origami_problem,
                                                            analytical_hardware,
                                                            origami_config,
                                                            DEFAULT_DYNAMIC_MODE);

    origami_config.reduction_strategy = reduction_type;

    auto result = origami::streamk::select_grid_size(origami_problem,
                                                    analytical_hardware,
                                                    origami_config,
                                                    DEFAULT_DYNAMIC_MODE,
                                                    max_cus);

    return result;
}
