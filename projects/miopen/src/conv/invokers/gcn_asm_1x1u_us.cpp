/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/conv/invokers/gcn_asm_1x1u_us.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

namespace miopen {
namespace conv {

void RunUpSampleKernel(const solver::KernelInfo& us_kernel_info,
                       const Handle& handle,
                       const ConvDataTensors& tensors,
                       Data_t workSpace)
{

    auto&& kernels = handle.GetKernels(us_kernel_info.kernel_name, us_kernel_info.comp_options);
    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(workSpace, tensors.out);
    }
    else
    {
        handle.AddKernel(us_kernel_info.kernel_name,
                         us_kernel_info.comp_options,
                         us_kernel_info.kernel_file,
                         us_kernel_info.kernel_name,
                         us_kernel_info.l_wk,
                         us_kernel_info.g_wk,
                         us_kernel_info.comp_options)(workSpace, tensors.out);
    }
}

InvokerFactory MakeGcnAsm1x1UUSInvokerFactory(const solver::KernelInfo& us_kernel_info,
                                              int N,
                                              int C,
                                              int K,
                                              int n_groups,
                                              int H,
                                              int W,
                                              std::size_t workspace_sz)
{
    return [=](const std::vector<Kernel>& kernels) {
        const auto kernel = kernels[0];

        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& params        = primitive_parameters.CastTo<DataInvokeParams>();
            const auto& tensors       = params.tensors;
            const auto& workSpace     = params.workSpace;
            const auto& workSpaceSize = params.workSpaceSize;

            if(workSpace == nullptr || workSpaceSize == 0)
                MIOPEN_THROW("Workspace is required for SubSample");

            if(workSpaceSize < workspace_sz)
                MIOPEN_THROW("Not enough workspace has been provided for SubSample.");

            const int unused = 0;
            int* return_addr = nullptr;
            handle.Run(kernel)(N,
                               C,
                               H,
                               W,
                               K,
                               n_groups,
                               unused,
                               unused,
                               tensors.in,
                               tensors.w,
                               workSpace,
                               return_addr);

            if(params.type != InvokeType::AutoTune)
            {
                float elapsed = 0;

                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                /// \todo Initialization is required for upsampling. This leads to small perf drop.
                /// 1: Add kernel (from SetTensor) to the Solution in the Solver.
                /// 2: Fix UpSample kernel, probably by means of conditional compilation.
                float zero = 0.f;
                SetTensor(handle, tensors.outDesc, tensors.out, &zero);

                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                RunUpSampleKernel(us_kernel_info, handle, tensors, workSpace);

                if(handle.IsProfilingEnabled())
                {
                    elapsed += handle.GetKernelTime();
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            }
        };
    };
}

} // namespace conv
} // namespace miopen
