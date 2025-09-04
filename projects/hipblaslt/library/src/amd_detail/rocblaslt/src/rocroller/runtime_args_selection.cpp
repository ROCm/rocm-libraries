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

#include "gemm.hpp"
#include "runtime_args_selection.hpp"

#include <Tensile/analytical/StreamK.hpp>

int chooseStreamKGridSize(std::shared_ptr<GemmKernel>        gemm,
                          const RocblasltContractionProblem& prob)
{
    const TensileLite::analytical::Hardware analaytical_hardware = TensileLite::analytical::Hardware::getHardwareForDevice(0);

    size_t elementSizeA_bits = rocRoller::DataTypeInfo::Get(gemm->params->kernelType.typeA).elementBits;
    size_t elementSizeB_bits = rocRoller::DataTypeInfo::Get(gemm->params->kernelType.typeB).elementBits;
    size_t elementSizeD_bits = rocRoller::DataTypeInfo::Get(gemm->params->kernelType.typeD).elementBits;
    size_t elementSizeAcc = rocRoller::DataTypeInfo::Get(gemm->params->kernelType.typeAcc).elementBytes;

    TensileLite::analytical::DataType dataType;
    if (elementSizeA_bits < elementSizeB_bits)
        dataType = rocroller_type_to_analytical_type(gemm->params->kernelType.typeB);
    else
        dataType = rocroller_type_to_analytical_type(gemm->params->kernelType.typeA);

    auto result = TensileLite::analytical::streamk::select_streamk_grid(prob.m,
        prob.n,
        prob.k,
        prob.batch_count,
        prob.trans_a == HIPBLAS_OP_T,
        prob.trans_b == HIPBLAS_OP_T,
        elementSizeA_bits,
        elementSizeB_bits,
        elementSizeD_bits,
        dataType,
        prob.workspaceSize,
        gemm->params->workgroupTile.m,
        gemm->params->workgroupTile.n,
        gemm->params->workgroupTile.k,
        gemm->params->machineInstruction.m,
        gemm->params->machineInstruction.n,
        gemm->params->machineInstruction.k,
        DEFAULT_WGM,
        elementSizeAcc,
        gemm->occupancy,
        analaytical_hardware,
        6);

    return result;
}
