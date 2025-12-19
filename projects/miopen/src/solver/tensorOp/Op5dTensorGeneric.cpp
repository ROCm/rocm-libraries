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

// Software-level caps for the number of work-groups, used as a safety net
// on top of hardware / CU-based heuristics.
//
// For small tensors, launching too many WGs increases dispatch overhead
// and leads to tail inefficiency, as for larger tensors - allow more WGs to improve latency hiding.
// Values are empirical and intentionally conservative.
constexpr size_t k_SmallTensorElemsThreshold = 1024 * 1024;
constexpr size_t k_MaxNumWgSmallTensor       = 1024;
constexpr size_t k_MaxNumWgLargeTensor       = 4096;

bool Op5dTensorGeneric::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                     const miopen::tensorOp::ProblemDescription& problem) const
{
    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& alens       = aTensorDesc.GetLengths();
    auto asize              = alens.size();

    if(asize == 5)
    {
        return true;
    }

    return false;
}

std::size_t Op5dTensorGeneric::GetWorkspaceSize(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::tensorOp::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution
Op5dTensorGeneric::GetSolution([[maybe_unused]] const ExecutionContext& context,
                               const miopen::tensorOp::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& bTensorDesc = problem.GetBTensorDesc();
    const auto& cTensorDesc = problem.GetCTensorDesc();

    const auto& alens = aTensorDesc.GetLengths();
    const auto& blens = bTensorDesc.GetLengths();
    const auto& clens = cTensorDesc.GetLengths();

    std::array<size_t, 5> astrides;
    std::array<size_t, 5> bstrides;
    std::array<size_t, 5> cstrides;
    std::tie(astrides[0], astrides[1], astrides[2], astrides[3], astrides[4]) =
        miopen::tien<5>(aTensorDesc.GetStrides());
    std::tie(bstrides[0], bstrides[1], bstrides[2], bstrides[3], bstrides[4]) =
        miopen::tien<5>(bTensorDesc.GetStrides());
    std::tie(cstrides[0], cstrides[1], cstrides[2], cstrides[3], cstrides[4]) =
        miopen::tien<5>(cTensorDesc.GetStrides());

    auto bstrides_fix = bstrides;
    auto astrides_fix = astrides;
    for(int i = 0; i < 5; ++i)
    {
        if(blens[i] == 1)
            bstrides_fix[i] = 0;
        if(alens[i] == 1)
            astrides_fix[i] = 0;
    }

    KernelBuildParameters build_params = KernelBuildParameters{};
    GetCommonParams(build_params, problem, false);
    build_params.Define("USE_5D_TENSOR_GENERIC");

    // PACK_T - how many consecutive elements each thread handles per inner iteration.
    // declaration + define as main source, in kernel defined as fallback - only
    constexpr size_t k_PACK_T = 8;
    build_params.Define("PACK_T", k_PACK_T);

    // ========================================================================
    // Division Strength Reduction: Power-of-2 Optimization for W Dimension
    // ========================================================================
    // When c_w (W dimension) is a power of 2, we can replace expensive 64-bit
    // integer division and modulo with fast bit operations:
    //   x % c_w  ->  x & (c_w - 1)     (bitwise AND with mask)
    //   x / c_w  ->  x >> log2(c_w)    (right shift)
    //
    // This optimization is applied at JIT compile time when detected, with a
    // runtime fallback in the kernel for cases where the solver doesn't set
    // these macros.
    // ========================================================================
    const size_t c_w_val = clens[4];

    // Check if c_w is a power of 2: (x & (x-1)) == 0 for powers of 2, and x > 0
    const bool cw_is_pow2 = (c_w_val > 0) && ((c_w_val & (c_w_val - 1)) == 0);

    if(cw_is_pow2)
    {
        build_params.Define("CW_IS_POW2", 1);

        // Compute log2(c_w) using bit counting
        // For power of 2, log2 equals the position of the single set bit
        unsigned int cw_log2 = 0;
        size_t temp          = c_w_val;
        while(temp > 1)
        {
            temp >>= 1;
            ++cw_log2;
        }
        build_params.Define("CW_LOG2", cw_log2);
    }

    const TensorDescriptor aTempDesc(aTensorDesc.GetType(), alens);
    const TensorDescriptor bTempDesc(bTensorDesc.GetType(), blens);

    const uint64_t a_max = static_cast<uint64_t>(aTempDesc.GetElementSpace());
    const uint64_t b_max = static_cast<uint64_t>(bTempDesc.GetElementSpace());
    const uint64_t c_max = static_cast<uint64_t>(cTensorDesc.GetElementSpace());

    // determine necessity to use 32-bit indexing
    bool use_i32 = (a_max < 0x7fffffffULL) && (b_max < 0x7fffffffULL) && (c_max < 0x7fffffffULL);
    use_i32      = use_i32 && aTensorDesc.AllDimsFitIntoInt() && bTensorDesc.AllDimsFitIntoInt() &&
              cTensorDesc.AllDimsFitIntoInt();

    // check if total work fits in u32 or not
    auto mul_u64_sat = [](uint64_t a, uint64_t b) -> uint64_t {
        if(a == 0 || b == 0)
            return uint64_t{0};
        if(a > (std::numeric_limits<uint64_t>::max() / b))
            return std::numeric_limits<uint64_t>::max();
        return a * b;
    };

    uint64_t total_work_u64 = 1;
    for(int i = 0; i < 5; ++i)
        total_work_u64 = mul_u64_sat(total_work_u64, static_cast<uint64_t>(clens[i]));
    const bool total_work_fits_u32 = (total_work_u64 <= uint64_t{0xFFFFFFFFu});

    use_i32 = use_i32 && total_work_fits_u32;
    if(use_i32)
        build_params.Define("USE_INDEX32");

    auto kernel                = KernelInfo{};
    miopenDataType_t data_type = bTensorDesc.GetType();

    const size_t warp_size   = context.GetStream().GetWavefrontWidth();
    const size_t total_elems = clens[0] * clens[1] * clens[2] * clens[3] * clens[4];
    const size_t cu_count    = context.GetStream().GetMaxComputeUnits();

    size_t local_threads;
    size_t num_wg;
    size_t global_threads;

    // Linear packet distribution: each thread processes PACK_T consecutive elements.
    // Optimal for general tensor operations.
    const size_t total_packets = (total_elems + k_PACK_T - 1) / k_PACK_T;

    /*!
     * @anchor cu_scaled_wg_cap
     * @brief Heuristic: CU-scaled work-group cap for latency hiding
     *
     * Heuristic cap for the number of work-groups to launch, proportional to the number of
     * compute units (CU). The goal is to keep enough wavefronts resident to hide memory latency,
     * while avoiding a cascade of tiny work-groups on small problems (scheduler overhead & tail
     * waste).
     *
     * On AMD GPUs, a wavefront consists of 64 threads. With local_threads = 256, each work-group
     * contains 256 / 64 = 4 wavefronts. The multipliers below roughly target a desired number of
     * wavefronts per CU:
     *
     *  - cu_count * 32  -> ~32 WGs/CU × 4 waves/WG ≈ 128 waves/CU (large problems)
     *  - cu_count * 16  -> ~16 WGs/CU × 4 waves/WG ≈  64 waves/CU (mid-size problems)
     *  - cu_count *  8  -> ~ 8 WGs/CU × 4 waves/WG ≈  32 waves/CU (small problems)
     *
     * The cap is downscaled when total_packets is small to reduce launch/dispatch overhead and
     * avoid over-fragmentation, while still keeping enough wavefronts per CU to hide latency.
     *
     * For the special case b_w == 1 (W-broadcast on B), memory access patterns tend to be less
     * favorable; the cap is slightly increased to allow more work-groups in flight and improve
     * latency hiding.
     */
    size_t max_num_wg_hw = cu_count * 32;
    if(total_packets < 512)
        max_num_wg_hw = cu_count * 16;
    if(total_packets < 128)
        max_num_wg_hw = cu_count * 8;

    local_threads = warp_size * 4; // 256 threads = 4 wavefronts

    // Small adjustments for narrow problems and W-broadcast cases:
    // For small total packet counts - reduce local_threads to avoid spawning many half-empty waves
    // For b_w == 1 - i.e. "heavy broadcast", limit local_threads to 128 to allow more WGs per CU.
    const bool bw_is_one = (blens[4] == 1);

    if(total_packets < 512)
        local_threads = 128;
    if(total_packets < 128)
        local_threads = 64;
    if(bw_is_one && local_threads > 128)
        local_threads = 128;

    // For W-broadcast, slightly increase the upper WG cap to better hide memory latency.
    if(bw_is_one)
        max_num_wg_hw = std::max(max_num_wg_hw, cu_count * 48);

    // Required number of WGs given the chosen local_threads
    const size_t need_wg = (total_packets + local_threads - 1) / local_threads;

    // Software-level safety cap on top of the hardware heuristic limit
    const size_t max_num_wg_sw =
        (total_elems < k_SmallTensorElemsThreshold) ? k_MaxNumWgSmallTensor : k_MaxNumWgLargeTensor;

    // Final cap
    const size_t max_num_wg = std::min(max_num_wg_hw, max_num_wg_sw);

    // Final number of WGs to launch
    num_wg = std::clamp(need_wg, static_cast<size_t>(1), max_num_wg);

    global_threads = num_wg * local_threads;

    const std::array<size_t, 3> vld{local_threads, 1, 1};
    const std::array<size_t, 3> vgd{global_threads, 1, 1};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
    kernel.kernel_file  = "MIOpenTensorKernelsHip.cpp";
    kernel.kernel_name  = "Op5dTensorGeneric";

    using std::begin, std::end;
    kernel.l_wk.insert(end(kernel.l_wk), begin(vld), end(vld));
    kernel.g_wk.insert(end(kernel.g_wk), begin(vgd), end(vgd));

    result.invoker_factory =
        [data_type, blens, clens, cstrides, astrides_fix, bstrides_fix, mul_u64_sat](
            const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
                auto kernel = handle.Run(kernels.front());
                auto params = raw_params.CastTo<miopen::tensorOp::InvokeParams>();

                visit_float(data_type, [&](auto as_float) {
                    const float alpha0f = *static_cast<const float*>(params.alpha0);
                    const float alpha1f = *static_cast<const float*>(params.alpha1);
                    const float betaf   = *static_cast<const float*>(params.beta);
                    const auto alpha0   = as_float(alpha0f);
                    const auto alpha1   = as_float(alpha1f);
                    const auto beta     = as_float(betaf);

                    uint64_t total_work = 1;
                    for(int i = 0; i < 5; ++i)
                        total_work = mul_u64_sat(total_work, static_cast<uint64_t>(clens[i]));

                    kernel(params.ATensor,            // Input Tensor A
                           params.BTensor,            // Operand Tensor B
                           params.CTensor,            // Output Tensor C
                           uint64_t(params.Aoffset),  // A-offset
                           uint64_t(params.Boffset),  // B-offset
                           uint64_t(params.Coffset),  // C-offset
                           uint64_t(blens[0]),        // b_n
                           uint64_t(blens[1]),        // b_c
                           uint64_t(blens[2]),        // b_d
                           uint64_t(blens[3]),        // b_h
                           uint64_t(blens[4]),        // b_w
                           uint64_t(clens[0]),        // c_n
                           uint64_t(clens[1]),        // c_c
                           uint64_t(clens[2]),        // c_d
                           uint64_t(clens[3]),        // c_h
                           uint64_t(clens[4]),        // c_w
                           uint64_t(astrides_fix[0]), // a_n fixed stride
                           uint64_t(astrides_fix[1]), // a_c fixed stride
                           uint64_t(astrides_fix[2]), // a_d fixed stride
                           uint64_t(astrides_fix[3]), // a_h fixed stride
                           uint64_t(astrides_fix[4]), // a_w fixed stride
                           uint64_t(bstrides_fix[0]), // b_n fixed stride
                           uint64_t(bstrides_fix[1]), // b_c fixed stride
                           uint64_t(bstrides_fix[2]), // b_d fixed stride
                           uint64_t(bstrides_fix[3]), // b_h fixed stride
                           uint64_t(bstrides_fix[4]), // b_w fixed stride
                           uint64_t(cstrides[0]),     // c_n strides
                           uint64_t(cstrides[1]),     // c_c strides
                           uint64_t(cstrides[2]),     // c_d strides
                           uint64_t(cstrides[3]),     // c_h strides
                           uint64_t(cstrides[4]),     // c_w strides
                           alpha0,                    // a_0 coeff - scalar
                           alpha1,                    // a_1 coeff - scalar
                           beta,                      // beta-factor
                           total_work,                // total number of operations
                           !float_equal(beta, 0.0f)); // use beta
                });
            };
        };
    result.construction_params.push_back(kernel);
    return result;
}

} // namespace tensorOp

} // namespace solver

} // namespace miopen
