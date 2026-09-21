/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <miopen/scalar.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/conv/tensors.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
namespace miopen {
namespace conv {

struct DataInvokeParams : InvokeParams
{
    ConvDataTensors tensors;
    Data_t workSpace;
    std::size_t workSpaceSize;
    bool gfx90aFp16alt;
    Scalar alpha;
    Scalar beta;

    DataInvokeParams(ConvDataTensors tensors_,
                     Data_t workSpace_,
                     std::size_t workSpaceSize_,
                     bool gfx90aFp16alt_,
                     const Scalar& alpha_ = Scalar(1.0),
                     const Scalar& beta_  = Scalar(0.0))
        : tensors(tensors_),
          workSpace(workSpace_),
          workSpaceSize(workSpaceSize_),
          gfx90aFp16alt(gfx90aFp16alt_),
          alpha(alpha_),
          beta(beta_)
    {
    }

    DataInvokeParams(InvokeType type_,
                     ConvDataTensors tensors_,
                     Data_t workSpace_,
                     std::size_t workSpaceSize_,
                     bool gfx90aFp16alt_,
                     const Scalar& alpha_ = Scalar(1.0),
                     const Scalar& beta_  = Scalar(0.0))
        : InvokeParams{type_},
          tensors(tensors_),
          workSpace(workSpace_),
          workSpaceSize(workSpaceSize_),
          gfx90aFp16alt(gfx90aFp16alt_),
          alpha(alpha_),
          beta(beta_)
    {
    }

    std::size_t GetWorkspaceSize() const { return workSpaceSize; }
    Data_t GetWorkspace() const { return workSpace; }
};

struct TransposeConvInvokeParams : InvokeParams
{
    // Flat tensor descriptors (required by TransposingSolver member pointers)
    TensorDescriptor inDesc;
    TensorDescriptor wDesc;
    TensorDescriptor outDesc;

    // Flat data pointers (required by TransposingSolver member pointers)
    ConstData_t in = nullptr;
    ConstData_t w  = nullptr;
    Data_t out     = nullptr;

    // Workspace
    Data_t workspace          = nullptr;
    std::size_t workspaceSize = 0;

    // Additional conv params
    bool gfx90aFp16alt = false;
    Scalar alpha{1.0};
    Scalar beta{0.0};

    // WrW-specific data pointers (used when is_wrw == true)
    // For WrW: w (weights slot) holds dw which is the output (Data_t),
    //          out (output slot) holds x which is an input (ConstData_t)
    Data_t w_as_output       = nullptr; // dw buffer pointer for WrW output transpose
    ConstData_t out_as_input = nullptr; // x buffer pointer for WrW input transpose
    bool is_wrw              = false;

    TransposeConvInvokeParams() = default;

    // Constructor from DataInvokeParams only (uses tensor descs from params.tensors)
    explicit TransposeConvInvokeParams(const DataInvokeParams& params)
        : InvokeParams{params.type},
          inDesc(params.tensors.inDesc),
          wDesc(params.tensors.wDesc),
          outDesc(params.tensors.outDesc),
          in(params.tensors.in),
          w(params.tensors.w),
          out(params.tensors.out),
          workspace(params.workSpace),
          workspaceSize(params.workSpaceSize),
          gfx90aFp16alt(params.gfx90aFp16alt),
          alpha(params.alpha),
          beta(params.beta)
    {
    }

    // Constructor from WrWInvokeParams
    // Maps: in/inDesc = dy, w/wDesc = dw, out/outDesc = x
    // Also sets w_as_output and out_as_input for correct transpose I/O semantics
    explicit TransposeConvInvokeParams(const WrWInvokeParams& params)
        : InvokeParams{params.type},
          inDesc(params.tensors.dyDesc),
          wDesc(params.tensors.dwDesc),
          outDesc(params.tensors.xDesc),
          in(params.tensors.dy),
          w(params.tensors.dw),
          out(nullptr), // not used for WrW — x is accessed via out_as_input
          workspace(params.workSpace),
          workspaceSize(params.workSpaceSize),
          gfx90aFp16alt(params.gfx90aFp16alt),
          alpha(params.alpha),
          beta(params.beta),
          w_as_output(params.tensors.dw),
          out_as_input(params.tensors.x),
          is_wrw(true)
    {
    }

    // Constructor from existing DataInvokeParams + ProblemDescription
    TransposeConvInvokeParams(const DataInvokeParams& params, const ProblemDescription& problem)
        : InvokeParams{params.type},
          inDesc(problem.GetIn()),
          wDesc(problem.GetWeights()),
          outDesc(problem.GetOut()),
          in(params.tensors.in),
          w(params.tensors.w),
          out(params.tensors.out),
          workspace(params.workSpace),
          workspaceSize(params.workSpaceSize),
          gfx90aFp16alt(params.gfx90aFp16alt),
          alpha(params.alpha),
          beta(params.beta)
    {
    }

    std::size_t GetWorkspaceSize() const { return workspaceSize; }
    Data_t GetWorkspace() const { return workspace; }

    /// Convert to DataInvokeParams for inner solver invocation (Fwd/Bwd).
    DataInvokeParams ToDataInvokeParams() const
    {
        return DataInvokeParams{type,
                                ConvDataTensors{inDesc, in, wDesc, w, outDesc, out},
                                workspace,
                                workspaceSize,
                                gfx90aFp16alt,
                                alpha,
                                beta};
    }

    /// Convert to WrWInvokeParams for inner solver invocation (WrW).
    /// Maps back: inDesc/in → dy, wDesc/w_as_output → dw, outDesc/out_as_input → x
    WrWInvokeParams ToWrWInvokeParams() const
    {
        return WrWInvokeParams{
            type,
            ConvWrwTensors{inDesc, in, outDesc, out_as_input, wDesc, w_as_output},
            workspace,
            workspaceSize,
            gfx90aFp16alt,
            alpha,
            beta};
    }
};

} // namespace conv
} // namespace miopen
