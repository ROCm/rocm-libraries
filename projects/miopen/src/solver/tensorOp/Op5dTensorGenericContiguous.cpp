/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "tensor_op_helpers.hpp"
#include <miopen/tensorOp/solvers.hpp>
#include <miopen/tensorOp/invoke_params.hpp>
#include <miopen/tensor.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/datatype.hpp>

namespace miopen {

namespace solver {

namespace tensorOp {

bool Op5dTensorGenericContiguous::IsApplicable(
    [[maybe_unused]] const ExecutionContext&,
    const miopen::tensorOp::ProblemDescription& problem) const
{
    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& bTensorDesc = problem.GetBTensorDesc();
    const auto& cTensorDesc = problem.GetCTensorDesc();

    if(aTensorDesc.GetLengths().size() != 5)
        return false;
    if(aTensorDesc.GetLengths() != bTensorDesc.GetLengths())
        return false;
    if(bTensorDesc.GetLengths() != cTensorDesc.GetLengths())
        return false;

    if(!(aTensorDesc.IsContiguous() && bTensorDesc.IsContiguous() && cTensorDesc.IsContiguous()))
        return false;

    if(!(aTensorDesc.AllLengthsFitIntoInt() && bTensorDesc.AllLengthsFitIntoInt() &&
         cTensorDesc.AllLengthsFitIntoInt()))
        return false;

    const auto& cl = cTensorDesc.GetLengths();
    uint64_t total = 1;
    for(int i = 0; i < 5; ++i)
    {
        const uint64_t t = static_cast<uint64_t>(cl[i]);
        if(t == 0)
            return false;
        if(total > (std::numeric_limits<uint64_t>::max() / t))
            return false;
        total *= t;
    }
    if(total > std::numeric_limits<uint32_t>::max())
        return false;

    return true;
}

std::size_t Op5dTensorGenericContiguous::GetWorkspaceSize(
    [[maybe_unused]] const ExecutionContext&,
    [[maybe_unused]] const miopen::tensorOp::ProblemDescription&) const
{
    return 0;
}

ConvSolution
Op5dTensorGenericContiguous::GetSolution(const ExecutionContext& context,
                                         const miopen::tensorOp::ProblemDescription& problem) const
{
    ConvSolution result{miopenStatusSuccess};

    const auto& cTensorDesc = problem.GetCTensorDesc();
    const auto& clens       = cTensorDesc.GetLengths();

    KernelBuildParameters build_params;
    GetCommonParams(build_params, problem, false);
    build_params.Define("USE_5D_TENSOR_GENERIC_CONTIGUOUS");

    // Use 32-bit verified all dimensions and total elements fit
    build_params.Define("USE_INDEX32");

    // PACK_T - how many consecutive elements each thread handles per inner iteration.
    // declaration + define as main source, in kernel defined as fallback - only
    constexpr size_t k_PACK_T = 8;
    build_params.Define("PACK_T", k_PACK_T);

    const uint64_t total_elems_u64 =
        static_cast<uint64_t>(clens[0]) * static_cast<uint64_t>(clens[1]) *
        static_cast<uint64_t>(clens[2]) * static_cast<uint64_t>(clens[3]) *
        static_cast<uint64_t>(clens[4]);

    const size_t total_elems   = static_cast<size_t>(total_elems_u64);
    const size_t total_packets = (total_elems + k_PACK_T - 1) / k_PACK_T;

    size_t local_threads = 256; // default
    if(total_packets < 512)
        local_threads = 128;
    if(total_packets < 128)
        local_threads = 64;

    const size_t cu_count   = context.GetStream().GetMaxComputeUnits();
    const size_t need_wg    = (total_packets + local_threads - 1) / local_threads;
    const size_t max_num_wg = std::max<std::size_t>(cu_count * 32, 1);
    const size_t num_wg     = std::max<std::size_t>(size_t{1}, std::min(need_wg, max_num_wg));

    const size_t global_threads = num_wg * local_threads;
    const std::array<size_t, 3> vld{local_threads, 1, 1};
    const std::array<size_t, 3> vgd{global_threads, 1, 1};

    miopenDataType_t data_type = cTensorDesc.GetType();

    KernelInfo kernel;
    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
    kernel.kernel_file  = "MIOpenTensorKernelsHip.cpp";
    kernel.kernel_name  = "Op5dTensorGenericContiguous";
    kernel.l_wk         = {vld[0], vld[1], vld[2]};
    kernel.g_wk         = {vgd[0], vgd[1], vgd[2]};

    result.invoker_factory =
        [data_type, clens, total_elems_u64](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
                auto kernel       = handle.Run(kernels.front());
                const auto params = raw_params.CastTo<miopen::tensorOp::InvokeParams>();

                const uint64_t total_work = total_elems_u64;

                visit_float(data_type, [&](auto as_float) {
                    const float alpha0f = *static_cast<const float*>(params.alpha0);
                    const float alpha1f = *static_cast<const float*>(params.alpha1);
                    const float betaf   = *static_cast<const float*>(params.beta);
                    const auto alpha0   = as_float(alpha0f);
                    const auto alpha1   = as_float(alpha1f);
                    const auto beta     = as_float(betaf);

                    // assuming all forms equality
                    kernel(params.ATensor,
                           params.BTensor,
                           params.CTensor,
                           static_cast<uint64_t>(params.Aoffset),
                           static_cast<uint64_t>(params.Boffset),
                           static_cast<uint64_t>(params.Coffset),
                           uint64_t(clens[0]), // b_n = c_n
                           uint64_t(clens[1]), // b_c = c_c
                           uint64_t(clens[2]), // b_d = c_d
                           uint64_t(clens[3]), // b_h = c_h
                           uint64_t(clens[4]), // b_w = c_w
                           uint64_t(clens[0]), // c_n
                           uint64_t(clens[1]), // c_c
                           uint64_t(clens[2]), // c_d
                           uint64_t(clens[3]), // c_h
                           uint64_t(clens[4]), // c_w
                           alpha0,
                           alpha1,
                           beta,
                           total_work,
                           !float_equal(beta, 0.0f));
                });
            };
        };

    result.construction_params.push_back(kernel);
    return result;
}

} // namespace tensorOp

} // namespace solver

} // namespace miopen
