/*! \file */
/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
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

#pragma once

#include "parameter_selection.hpp"

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/TensorDescriptor.hpp>

/**
 * @brief GemmKernel
 *
 * Everything needed to launch a kernel
 *
 */
struct GemmKernel
{
    rocRoller::CommandPtr                          command;
    rocRoller::CommandKernelPtr                    commandKernel;
    std::shared_ptr<SolutionParameters> params;

    rocRoller::Operations::OperationTag tagTensorA;
    rocRoller::Operations::OperationTag tagTensorB;
    rocRoller::Operations::OperationTag tagTensorC;
    rocRoller::Operations::OperationTag tagTensorD;

    rocRoller::Operations::OperationTag tagScalarAlpha;
    rocRoller::Operations::OperationTag tagScalarBeta;

    rocRoller::Operations::OperationTag tagTensorScaleA;
    rocRoller::Operations::OperationTag tagTensorScaleB;

    rocRoller::Operations::OperationTag tagScratch;
    rocRoller::Operations::OperationTag tagSKGrid;
    rocRoller::Operations::OperationTag tagWGM;

    int occupancy;
};

/**
 * @brief Generate a GEMM Kernel
 *
 * This involves creating the Command describing the KernelType
 * and setting all of the parameters.
 *
 * @param gemm
 * @return std::shared_ptr<GemmKernel>
 */
std::shared_ptr<GemmKernel> genGemmKernel(std::shared_ptr<SolutionParameters> gemm);

/**
 * @brief Return the amount of workspace that is required to execute a kernel.
 *
 * Note: This only takes into account the workspace required for StreamK kernels.
 */
size_t workspaceRequired(std::shared_ptr<GemmKernel> gemm, const RocblasltContractionProblem& prob);

/**
 * @brief Set the arguments to call a rocRoller kernel
 *
 * @param gemm
 * @param prob
 * @return CommandArguments
 */
rocRoller::CommandArguments createCommandArguments(std::shared_ptr<GemmKernel>        gemm,
    const RocblasltContractionProblem& prob,
    int wgm);

std::string genKernelName(std::shared_ptr<SolutionParameters> gemm);

/**
 * @brief Execute a GEMM operation.
 *
 * @param gemm
 * @param prob
 * @return rocblaslt_status
 */
rocblaslt_status runGemmKernel(std::shared_ptr<GemmKernel>        gemm,
    const RocblasltContractionProblem& prob);
