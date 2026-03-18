// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/pooling/solvers.hpp>

#include <miopen/pooling/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/pooling.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

namespace miopen {

namespace solver {

namespace pooling {

namespace {

struct kernel_params
{
    int kernel_size_h;
    int kernel_size_w;
    int kernel_stride_h;
    int kernel_stride_w;
    int out_height;
    int out_width;
    int out_pix_tile0;
    int out_pix_tile1;

    kernel_params(const miopen::pooling::ProblemDescription& p,
                  const std::optional<PerformanceConfigPooling2dForward>& config = std::nullopt)
    {
        const auto& pd  = p.GetPooling();
        const auto& yd  = p.GetYDesc();
        kernel_size_h   = pd.lens[0];
        kernel_size_w   = pd.lens[1];
        kernel_stride_h = pd.strides[0];
        kernel_stride_w = pd.strides[1];
        out_height      = yd.GetLengths()[2];
        out_width       = yd.GetLengths()[3];
        out_pix_tile0   = 1;
        if(config)
        {
            out_pix_tile1 = config->out_pix_tile1;
        }
        else
        {
            out_pix_tile1 = out_height <= 8    ? 1 //
                            : out_height <= 32 ? 4 //
                                               : 8;
            if(out_height > 16 && out_height % 32 > 16)
                out_pix_tile1 = std::min(16, std::max(1, prePow2(out_pix_tile1 * kernel_stride_h)));
        }
    }
};

std::size_t sizeof_kernel_FLOAT(const miopen::pooling::ProblemDescription& problem)
{
    const auto datatype = problem.GetXDesc().GetType();
    return get_data_size(datatype);
}

std::size_t sizeof_kernel_FLOAT_ACCUM(const miopen::pooling::ProblemDescription& problem)
{
    const auto datatype = problem.GetXDesc().GetType();
    if(datatype == miopenHalf || datatype == miopenBFloat16)
        return get_data_size(miopenFloat); // mixed precision
    return get_data_size(datatype);
}

inline std::size_t RoundUpToMultiple(std::size_t v, std::size_t m)
{
    assert(m > 0);
    return ((v + m - 1) / m) * m;
}

// Compute amount of private memory required for holding the arrays defined
// in the "mloPoolingG" kernel:
//
// #define MLO_BOT_DATA_SZ0
//     ((MLO_POOLING_N_HORIZ_OUT_PIX - 1) * MLO_POOLING_STRIDE0 + MLO_POOLING_KERNEL_SZ0)
//
// #define MLO_BOT_DATA_SZ1
//    ((MLO_POOLING_N_VERT_OUT_PIX - 1) * MLO_POOLING_STRIDE1 + MLO_POOLING_KERNEL_SZ1)
//
// _FLOAT bot_data[MLO_BOT_DATA_SZ1][MLO_BOT_DATA_SZ0];
// _FLOAT_ACCUM res[MLO_POOLING_N_VERT_OUT_PIX][MLO_POOLING_N_HORIZ_OUT_PIX];
//
std::size_t sizeof_private_memory(const miopen::pooling::ProblemDescription& problem)
{
    const kernel_params kp(problem);

    // aliases to ease programming
    const auto& MLO_POOLING_KERNEL_SZ1      = kp.kernel_size_h;
    const auto& MLO_POOLING_STRIDE1         = kp.kernel_stride_h;
    const auto& MLO_POOLING_KERNEL_SZ0      = kp.kernel_size_w;
    const auto& MLO_POOLING_STRIDE0         = kp.kernel_stride_w;
    const auto& MLO_POOLING_N_HORIZ_OUT_PIX = kp.out_pix_tile0;
    // safer estimate of memory with max_out_pix_tile1
    assert(kp.out_pix_tile1 <= PerformanceConfigPooling2dForward::max_out_pix_tile1);
    const auto& MLO_POOLING_N_VERT_OUT_PIX = PerformanceConfigPooling2dForward::max_out_pix_tile1;

    const auto MLO_BOT_DATA_SZ0 =
        (static_cast<std::size_t>(MLO_POOLING_N_HORIZ_OUT_PIX) - 1) * MLO_POOLING_STRIDE0 +
        MLO_POOLING_KERNEL_SZ0;
    const auto MLO_BOT_DATA_SZ1 =
        (static_cast<std::size_t>(MLO_POOLING_N_VERT_OUT_PIX) - 1) * MLO_POOLING_STRIDE1 +
        MLO_POOLING_KERNEL_SZ1;

    const auto sizeof_bot_data = sizeof_kernel_FLOAT(problem) * MLO_BOT_DATA_SZ1 * MLO_BOT_DATA_SZ0;
    const auto sizeof_res      = sizeof_kernel_FLOAT_ACCUM(problem) * MLO_POOLING_N_VERT_OUT_PIX *
                            MLO_POOLING_N_HORIZ_OUT_PIX;

    MIOPEN_LOG_T("sizeof_bot_data " << sizeof_bot_data << "sizeof_res" << sizeof_res);

    /// \ref alignment_of_arrays_in_gpu_memory
    return RoundUpToMultiple(sizeof_bot_data, 16) + RoundUpToMultiple(sizeof_res, 16);
}

} // namespace

bool PoolingForward2d::IsApplicable(const ExecutionContext& context,
                                    const miopen::pooling::ProblemDescription& problem) const
{
    static const auto strict = TensorDescriptor::LayoutValidationMode::StrictDecreasingStrides;

    return problem.GetDirection() == miopen::pooling::Direction::Forward &&
           problem.GetXDesc().GetNumDims() == 4 &&
           problem.GetXDesc().GetType() == problem.GetYDesc().GetType() &&
           (problem.GetXDesc().GetType() == miopenFloat ||
            problem.GetXDesc().GetType() == miopenHalf ||
            problem.GetXDesc().GetType() == miopenBFloat16) &&
           problem.GetXDesc().IsPossibleLayout4D5D("NCHW", strict) &&
           problem.GetYDesc().IsPossibleLayout4D5D("NCHW", strict) &&
           sizeof_private_memory(problem) <=
               TargetProperties::GetMaxWaveScratchSize() / context.GetStream().GetWavefrontWidth();
}

ConvSolution PoolingForward2d::GetSolution(const ExecutionContext&,
                                           const miopen::pooling::ProblemDescription& problem,
                                           const PerformanceConfigPooling2dForward& config) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenPooling.cpp";
        kernel.kernel_name = "mloPoolingG";

        const kernel_params kp(problem, config);

        int batch_sz, n_outputs;
        std::tie(batch_sz, n_outputs, std::ignore, std::ignore) =
            miopen::tien<4>(problem.GetYDesc().GetLengths(), 1);

        const auto& pool_d   = problem.GetPooling();
        const auto wsp_index = pool_d.GetWorkspaceIndexMode();

        int grp_tile0 = config.local_size0;
        int grp_tile1 = config.local_size1;
        grp_tile1 /= kp.out_pix_tile1;
        while(grp_tile0 * grp_tile1 > 256 && grp_tile0 > 1)
            grp_tile0 >>= 1;

        int pooling_method =
            (pool_d.GetMode() == miopenPoolingMax)
                ? MLO_POOLING_OP_MAX
                : ((pool_d.GetMode() == miopenPoolingAverage) ? MLO_POOLING_OP_AVE
                                                              : MLO_POOLING_OP_AVE_INCLUSIVE);

        auto build_params = KernelBuildParameters{
            {"MLO_POOLING_OP_ID", pooling_method},
            {"MLO_POOLING_KERNEL_SZ1", kp.kernel_size_h},
            {"MLO_POOLING_STRIDE1", kp.kernel_stride_h},
            {"MLO_POOLING_KERNEL_SZ0", kp.kernel_size_w},
            {"MLO_POOLING_STRIDE0", kp.kernel_stride_w},
            {"MLO_POOLING_N_HORIZ_OUT_PIX", kp.out_pix_tile0},
            {"MLO_POOLING_N_VERT_OUT_PIX", kp.out_pix_tile1},
            {"MLO_POOLING_GROUP_SZ0", grp_tile0},
            {"MLO_POOLING_GROUP_SZ1", grp_tile1},
            {"MLO_POOLING_INDEX_TYPE", get_pooling_index_type_name(pool_d.GetIndexType())},
        };

        if(problem.SaveIndex())
        {
            build_params << KernelBuildParameters{
                {"MLO_POOLING_SAVE_INDEX"},
                {"USE_IMG_INDEX", (wsp_index == miopenPoolingWorkspaceIndexImage ? 1 : 0)},
            };
        }

        build_params << GetDataTypeKBP(problem.GetXDesc().GetType());

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(grp_tile0);
        kernel.l_wk.push_back(grp_tile1);
        kernel.l_wk.push_back(1);

        int g_wk_width =
            ((kp.out_width + grp_tile0 * kp.out_pix_tile0 - 1) / (grp_tile0 * kp.out_pix_tile0));
        int g_wk_height =
            ((kp.out_height + grp_tile1 * kp.out_pix_tile1 - 1) / (grp_tile1 * kp.out_pix_tile1));

        kernel.g_wk.push_back(static_cast<std::size_t>(g_wk_width) * grp_tile0);
        kernel.g_wk.push_back(static_cast<std::size_t>(g_wk_height) * grp_tile1);
        kernel.g_wk.push_back(static_cast<std::size_t>(n_outputs) * batch_sz);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::pooling::FwdInvokeParams>();

            kernel(params.x,
                   params.y,
                   params.workspace,
                   static_cast<int>(params.pooling.pads[0]),
                   static_cast<int>(params.pooling.pads[1]),
                   static_cast<int>(params.xDesc.GetLengths()[1]),
                   static_cast<int>(params.xDesc.GetLengths()[2]),
                   static_cast<int>(params.xDesc.GetLengths()[3]),
                   static_cast<int>(params.yDesc.GetLengths()[2]),
                   static_cast<int>(params.yDesc.GetLengths()[3]),
                   static_cast<int>(params.xDesc.GetStrides()[0]),
                   static_cast<int>(params.xDesc.GetStrides()[1]),
                   static_cast<int>(params.xDesc.GetStrides()[2]),
                   static_cast<int>(params.yDesc.GetStrides()[0]),
                   static_cast<int>(params.yDesc.GetStrides()[1]),
                   static_cast<int>(params.yDesc.GetStrides()[2]));
        };
    };

    return result;
}

bool PerformanceConfigPooling2dForward::IsValidValue(
    const miopen::pooling::ProblemDescription&) const
{
    if(!IsTwoPower<min_out_pix_tile1, max_out_pix_tile1>(out_pix_tile1))
        return false;
    if(!IsTwoPower<min_local_size0, max_local_size0>(local_size0))
        return false;
    if(!IsTwoPower<min_local_size1, max_local_size1>(local_size1))
        return false;
    // this constraint is enforced to avoid grp_tile1 becoming zero in GetSolution
    if(local_size1 / out_pix_tile1 < 1)
        return false;
    return true;
}

std::size_t
PoolingForward2d::GetWorkspaceSize(const ExecutionContext&,
                                   const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax)
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

bool PoolingForward2d::IsValidPerformanceConfig(
    const ExecutionContext& context,
    const miopen::pooling::ProblemDescription& problem,
    const PerformanceConfigPooling2dForward& config) const
{
    return config.IsValid(context, problem);
}

PerformanceConfigPooling2dForward PoolingForward2d::GetDefaultPerformanceConfig(
    const ExecutionContext&, const miopen::pooling::ProblemDescription& problem) const
{
    PerformanceConfigPooling2dForward config;
    config.HeuristicInit(problem);
    return config;
}

} // namespace pooling

} // namespace solver

} // namespace miopen
