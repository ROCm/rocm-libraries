// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/pooling/solvers.hpp>

#include <miopen/pooling/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/pooling.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/target_properties.hpp>
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
    int out_pix_tile0;
    int out_pix_tile1;
    std::size_t batch_sz;
    std::size_t n_inputs;
    std::size_t in_height;
    std::size_t in_width;
    std::size_t grp_tile0;
    std::size_t grp_tile1;

    kernel_params(const miopen::pooling::ProblemDescription& problem,
                  const PerformanceConfigPooling2dBackward& config)
    {
        const auto& pd = problem.GetPooling();

        kernel_size_w   = pd.lens[1];
        kernel_size_h   = pd.lens[0];
        kernel_stride_w = pd.strides[1];
        kernel_stride_h = pd.strides[0];

        std::tie(batch_sz, n_inputs, in_height, in_width) =
            miopen::tien<4>(problem.GetXDesc().GetLengths(), 1);

        out_pix_tile0 = config.out_pix_tile0;
        out_pix_tile1 = config.out_pix_tile1;
        grp_tile0     = config.local_size0;
        grp_tile1     = config.local_size1;
    }
};

std::size_t sizeof_kernel_FLOAT(const miopen::pooling::ProblemDescription& problem)
{
    const auto datatype = problem.GetXDesc().GetType();
    return get_data_size(datatype);
}

std::size_t sizeof_kernel_index_t(const miopen::pooling::ProblemDescription& problem)
{
    return get_data_size(problem.GetPooling().GetIndexType());
}

inline std::size_t RoundUpToMultiple(std::size_t v, std::size_t m)
{
    assert(m > 0);
    return ((v + m - 1) / m) * m;
}

// Compute amount of local memory required for holding the arrays defined
// in the "mloPoolingAveBwd" and "mloPoolingMaxBwd" kernels.
std::size_t sizeof_local_memory(const miopen::pooling::ProblemDescription& problem,
                                const PerformanceConfigPooling2dBackward& config)
{
    const kernel_params kp(problem, config);

    // aliases to ease programming
    const auto& MLO_POOLING_KERNEL_SZ0      = kp.kernel_size_w;
    const auto& MLO_POOLING_KERNEL_SZ1      = kp.kernel_size_h;
    const auto& MLO_POOLBWD_N_HORIZ_OUT_PIX = kp.out_pix_tile0;
    const auto& MLO_POOLBWD_N_VERT_OUT_PIX  = kp.out_pix_tile1;
    const auto& MLO_POOLING_STRIDE0         = kp.kernel_stride_w;
    const auto& MLO_POOLING_STRIDE1         = kp.kernel_stride_h;
    const auto& MLO_POOLBWD_GROUP_SZ0       = kp.grp_tile0;
    const auto& MLO_POOLBWD_GROUP_SZ1       = kp.grp_tile1;

    const auto MLO_POOLBWD_LCL_DATA_WIDTH =
        (static_cast<std::size_t>(MLO_POOLBWD_GROUP_SZ0) * MLO_POOLBWD_N_HORIZ_OUT_PIX +
         MLO_POOLING_KERNEL_SZ0 + MLO_POOLING_STRIDE0 - 2) /
        MLO_POOLING_STRIDE0;
    const auto MLO_POOLBWD_LCL_DATA_HEIGHT =
        (static_cast<std::size_t>(MLO_POOLBWD_GROUP_SZ1) * MLO_POOLBWD_N_VERT_OUT_PIX +
         MLO_POOLING_KERNEL_SZ1 + MLO_POOLING_STRIDE1 - 2) /
        MLO_POOLING_STRIDE1;

    std::size_t rv   = 0;
    const auto nelem = MLO_POOLBWD_LCL_DATA_WIDTH * MLO_POOLBWD_LCL_DATA_HEIGHT;
    if(problem.GetPooling().GetMode() == miopenPoolingMax)
    {
        const auto sizeof_lcl_top_df = sizeof_kernel_FLOAT(problem) * nelem;
        const auto sizeof_lcl_mask   = sizeof_kernel_index_t(problem) * nelem;
        /// \anchor alignment_of_arrays_in_gpu_memory
        /// The total amount of memory calculated here is slightly less than the amount calculated
        /// by the compiler. As a result, the check here may pass, while then the compiler might
        /// refuse to build the kernel. The most likely reason for the difference is padding (due to
        /// alignment requirements). We don't know exactly how the compiler takes alignment into
        /// account, but what can we do is applying an alignment that imposes a slightly tighter
        /// constraints than the compiler. So far, 16-byte (4xDWORD) alignment works well.
        rv = RoundUpToMultiple(sizeof_lcl_top_df, 16) + RoundUpToMultiple(sizeof_lcl_mask, 16);
    }
    else
    {
        const auto sizeof_lcl_top_diff = sizeof_kernel_FLOAT(problem) * nelem;
        rv                             = RoundUpToMultiple(sizeof_lcl_top_diff, 16);
    }
    MIOPEN_LOG_T(rv);
    return rv;
}

} // namespace

bool PoolingBackward2d::IsApplicable(const ExecutionContext&,
                                     const miopen::pooling::ProblemDescription& problem) const
{
    static const auto strict = TensorDescriptor::LayoutValidationMode::StrictDecreasingStrides;

    return problem.GetDirection() == miopen::pooling::Direction::Backward &&
           problem.GetXDesc().GetType() == problem.GetYDesc().GetType() &&
           (problem.GetXDesc().GetType() == miopenFloat ||
            problem.GetXDesc().GetType() == miopenHalf ||
            problem.GetXDesc().GetType() == miopenBFloat16) &&
           (problem.GetPooling().GetMode() == miopenPoolingMax ||
            problem.GetPooling().GetMode() == miopenPoolingAverage ||
            problem.GetPooling().GetMode() == miopenPoolingAverageInclusive) &&
           problem.GetXDesc().GetNumDims() == 4 &&
           problem.GetXDesc().IsPossibleLayout4D5D("NCHW", strict) &&
           problem.GetYDesc().IsPossibleLayout4D5D("NCHW", strict);
}

ConvSolution PoolingBackward2d::GetSolution(const ExecutionContext&,
                                            const miopen::pooling::ProblemDescription& problem,
                                            const PerformanceConfigPooling2dBackward& config) const
{
    // check local memory requirement
    if(sizeof_local_memory(problem, config) > TargetProperties::GetMaxLocalMemorySize())
    {
        MIOPEN_THROW(
            "The local memory requirement in PoolingBackward2d solver exceeds the device limit.");
    }

    auto result = ConvSolution{miopenStatusSuccess};

    const kernel_params kp(problem, config);

    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenPoolingBwd.cpp";

        if(problem.GetPooling().GetMode() == miopenPoolingMax)
        {
            kernel.kernel_name = "mloPoolingMaxBwd";
        }
        else if(problem.GetPooling().GetMode() == miopenPoolingAverage ||
                problem.GetPooling().GetMode() == miopenPoolingAverageInclusive)
        {
            kernel.kernel_name = "mloPoolingAveBwd";
        }

        const int pooling_method = (problem.GetPooling().GetMode() == miopenPoolingMax)
                                       ? MLO_POOLING_OP_MAX
                                       : ((problem.GetPooling().GetMode() == miopenPoolingAverage)
                                              ? MLO_POOLING_OP_AVE
                                              : MLO_POOLING_OP_AVE_INCLUSIVE);

        const int g_wk_width  = ((kp.in_width + kp.grp_tile0 * kp.out_pix_tile0 - 1) /
                                (kp.grp_tile0 * kp.out_pix_tile0));
        const int g_wk_height = ((kp.in_height + kp.grp_tile1 * kp.out_pix_tile1 - 1) /
                                 (kp.grp_tile1 * kp.out_pix_tile1));

        const auto build_params =
            KernelBuildParameters{
                {"MLO_POOLING_OP_ID", pooling_method},
                {"MLO_POOLING_KERNEL_SZ1", kp.kernel_size_h},
                {"MLO_POOLING_STRIDE1", kp.kernel_stride_h},
                {"MLO_POOLING_KERNEL_SZ0", kp.kernel_size_w},
                {"MLO_POOLING_STRIDE0", kp.kernel_stride_w},
                {"MLO_POOLBWD_N_HORIZ_OUT_PIX", kp.out_pix_tile0},
                {"MLO_POOLBWD_N_VERT_OUT_PIX", kp.out_pix_tile1},
                {"MLO_POOLBWD_GROUP_SZ0", kp.grp_tile0},
                {"MLO_POOLBWD_GROUP_SZ1", kp.grp_tile1},
                {"MLO_POOLING_INDEX_TYPE",
                 get_pooling_index_type_name(problem.GetPooling().GetIndexType())},
                {"USE_IMG_INDEX",
                 problem.GetPooling().GetWorkspaceIndexMode() == miopenPoolingWorkspaceIndexImage
                     ? 1
                     : 0},
            }
            << GetDataTypeKBP(problem.GetXDesc().GetType());

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk = {kp.grp_tile0, kp.grp_tile1, 1};
        kernel.g_wk = {
            g_wk_width * kp.grp_tile0, g_wk_height * kp.grp_tile1, kp.n_inputs * kp.batch_sz};

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::pooling::BwdInvokeParams>();

            if(params.pooling.GetMode() == miopenPoolingMax)
            {
                kernel(params.dy,
                       params.dx,
                       params.workspace,
                       static_cast<int>(params.pooling.pads[0]),
                       static_cast<int>(params.pooling.pads[1]),
                       static_cast<int>(params.dyDesc.GetLengths()[1]),
                       static_cast<int>(params.dxDesc.GetLengths()[2]),
                       static_cast<int>(params.dxDesc.GetLengths()[3]),
                       static_cast<int>(params.dyDesc.GetLengths()[2]),
                       static_cast<int>(params.dyDesc.GetLengths()[3]),
                       static_cast<int>(params.dxDesc.GetStrides()[0]),
                       static_cast<int>(params.dxDesc.GetStrides()[1]),
                       static_cast<int>(params.dxDesc.GetStrides()[2]),
                       static_cast<int>(params.dyDesc.GetStrides()[0]),
                       static_cast<int>(params.dyDesc.GetStrides()[1]),
                       static_cast<int>(params.dyDesc.GetStrides()[2]));
            }
            else
            {
                kernel(params.dy,
                       params.dx,
                       static_cast<int>(params.pooling.pads[0]),
                       static_cast<int>(params.pooling.pads[1]),
                       static_cast<int>(params.dyDesc.GetLengths()[1]),
                       static_cast<int>(params.dxDesc.GetLengths()[2]),
                       static_cast<int>(params.dxDesc.GetLengths()[3]),
                       static_cast<int>(params.dyDesc.GetLengths()[2]),
                       static_cast<int>(params.dyDesc.GetLengths()[3]),
                       static_cast<int>(params.dxDesc.GetStrides()[0]),
                       static_cast<int>(params.dxDesc.GetStrides()[1]),
                       static_cast<int>(params.dxDesc.GetStrides()[2]),
                       static_cast<int>(params.dyDesc.GetStrides()[0]),
                       static_cast<int>(params.dyDesc.GetStrides()[1]),
                       static_cast<int>(params.dyDesc.GetStrides()[2]));
            }
        };
    };

    return result;
}

bool PerformanceConfigPooling2dBackward::IsValidValue(
    const miopen::pooling::ProblemDescription& problem) const
{
    if(!IsTwoPower<min_out_pix_tile0, max_out_pix_tile0>(out_pix_tile0))
        return false;
    if(!IsTwoPower<min_out_pix_tile1, max_out_pix_tile1>(out_pix_tile1))
        return false;
    if(!IsTwoPower<min_local_size0, max_local_size0>(local_size0))
        return false;
    if(!IsTwoPower<min_local_size1, max_local_size1>(local_size1))
        return false;
    // this constraint is enforced to avoid exceedance of local memory limit
    if(sizeof_local_memory(problem, *this) > TargetProperties::GetMaxLocalMemorySize())
        return false;
    return true;
}

std::size_t
PoolingBackward2d::GetWorkspaceSize(const ExecutionContext&,
                                    const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax)
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

bool PoolingBackward2d::IsValidPerformanceConfig(
    const ExecutionContext& context,
    const miopen::pooling::ProblemDescription& problem,
    const PerformanceConfigPooling2dBackward& config) const
{
    return config.IsValid(context, problem);
}

PerformanceConfigPooling2dBackward PoolingBackward2d::GetDefaultPerformanceConfig(
    const ExecutionContext&, const miopen::pooling::ProblemDescription& problem) const
{
    PerformanceConfigPooling2dBackward config;
    config.HeuristicInit(problem);
    return config;
}

} // namespace pooling

} // namespace solver

} // namespace miopen
