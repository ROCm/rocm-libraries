/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_BATCHED_TRANSPOSE_SOL_HPP
#define GUARD_MIOPEN_BATCHED_TRANSPOSE_SOL_HPP

#include <miopen/miopen.h>
#include <miopen/errors.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/execution_context.hpp>
#include <vector>

struct transpose_invoke_param : public miopen::InvokeParams
{
    ConstData_t src = nullptr;
    Data_t dst      = nullptr;

    transpose_invoke_param(ConstData_t src_, Data_t dst_) : InvokeParams{}, src(src_), dst(dst_) {}
    transpose_invoke_param(miopen::InvokeType type_, ConstData_t src_, Data_t dst_)
        : InvokeParams{type_}, src(src_), dst(dst_)
    {
    }

    Data_t GetWorkspace() const { return nullptr; }
    std::size_t GetWorkspaceSize() const { return 0; }
};

namespace miopen {

struct BatchedTransposeParam
{
    int tile_x{0};
    int tile_y{0};
    int pack_x{0};
    int pack_y{0};
    int ediv_x{0};
    int ediv_y{0};
};

struct MIOPEN_INTERNALS_EXPORT BatchedTransposeSolution
{
    BatchedTransposeSolution(const ExecutionContext& ctx_,
                             miopenDataType_t data_type_,
                             uint32_t batch_,
                             uint32_t height_,
                             uint32_t width_);
    solver::KernelInfo GetKernelInfo() const;
    std::vector<OpKernelArg> GetKernelArg() const;
    std::string GetKernelName() const;
    bool IsSkippable() const;
    size_t GetOutputTensorSize() const;

    /// Check if the given data type is supported by batched transpose
    static bool IsApplicable(miopenDataType_t data_type)
    {
        return data_type == miopenHalf || data_type == miopenFloat || data_type == miopenInt32 ||
               data_type == miopenInt8 || data_type == miopenBFloat16;
    }

    /// Check if dimensions are supported by batched transpose (works for both 4D and 5D)
    /// For 4D: lens = {N, C, H, W}
    /// For 5D: lens = {N, C, D, H, W}
    static bool IsApplicable(miopenDataType_t data_type, const std::vector<size_t>& lens)
    {
        // Check data type first
        if(!IsApplicable(data_type))
            return false;

        // Must be 4D or 5D
        if(lens.size() != 4 && lens.size() != 5)
            return false;

        // Check all dimensions fit in uint32_t
        for(auto dim : lens)
        {
            if(dim > std::numeric_limits<uint32_t>::max())
                return false;
        }

        const size_t n = lens[0];
        const size_t c = lens[1];

        // Compute spatial product (H*W for 4D, D*H*W for 5D) with overflow protection
        size_t spatial_product = 1;
        for(size_t i = 2; i < lens.size(); ++i)
        {
            // Check for overflow before multiplying
            if(lens[i] > 0 && spatial_product > std::numeric_limits<uint32_t>::max() / lens[i])
                return false;
            spatial_product *= lens[i];
        }

        // Check c*spatial doesn't overflow uint32_t
        if(c > 0 && spatial_product > std::numeric_limits<uint32_t>::max() / c)
            return false;

        const size_t c_spatial = c * spatial_product;

        // Check n*c*spatial doesn't overflow uint32_t
        if(n > 0 && c_spatial > std::numeric_limits<uint32_t>::max() / n)
            return false;

        return true;
    }

    miopenDataType_t data_type;
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    int num_cu;

    BatchedTransposeParam kernel_param_heuristic;

    InvokerFactory MakeBatchedTransposeInvokerFactory() const;
};

struct TransposeSolutionDefault2Nhwc : public BatchedTransposeSolution
{
    TransposeSolutionDefault2Nhwc(const ExecutionContext& ctx_,
                                  miopenDataType_t data_type_,
                                  uint32_t n_,
                                  uint32_t c_,
                                  uint32_t h_,
                                  uint32_t w_)
        : BatchedTransposeSolution(ctx_, data_type_, n_, c_, h_ * w_)
    {
        MIOPEN_THROW_IF(size_t(h_ * w_) != (size_t(h_) * size_t(w_)), "integer overflow");
    }
};

struct TransposeSolutionNhwc2Default : public BatchedTransposeSolution
{
    TransposeSolutionNhwc2Default(const ExecutionContext& ctx_,
                                  miopenDataType_t data_type_,
                                  uint32_t n_,
                                  uint32_t c_,
                                  uint32_t h_,
                                  uint32_t w_)
        : BatchedTransposeSolution(ctx_, data_type_, n_, h_ * w_, c_)
    {
        MIOPEN_THROW_IF(size_t(h_ * w_) != (size_t(h_) * size_t(w_)), "integer overflow");
    }
};

struct TransposeSolutionDefault2Ndhwc : public BatchedTransposeSolution
{
    TransposeSolutionDefault2Ndhwc(const ExecutionContext& ctx_,
                                   miopenDataType_t data_type_,
                                   uint32_t n_,
                                   uint32_t c_,
                                   uint32_t d_,
                                   uint32_t h_,
                                   uint32_t w_)
        : BatchedTransposeSolution(ctx_, data_type_, n_, c_, d_ * h_ * w_)
    {
        MIOPEN_THROW_IF(size_t(d_ * h_ * w_) != (size_t(d_) * size_t(h_) * size_t(w_)),
                        "integer overflow");
    }
};

struct TransposeSolutionNdhwc2Default : public BatchedTransposeSolution
{
    TransposeSolutionNdhwc2Default(const ExecutionContext& ctx_,
                                   miopenDataType_t data_type_,
                                   uint32_t n_,
                                   uint32_t c_,
                                   uint32_t d_,
                                   uint32_t h_,
                                   uint32_t w_)
        : BatchedTransposeSolution(ctx_, data_type_, n_, d_ * h_ * w_, c_)
    {
        MIOPEN_THROW_IF(size_t(d_ * h_ * w_) != (size_t(d_) * size_t(h_) * size_t(w_)),
                        "integer overflow");
    }
};

} // namespace miopen

#endif
