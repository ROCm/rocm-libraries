// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::tensor_layout::convolution;

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

#ifdef CK_ENABLE_FP8
using F8 = ck::f8_t;
#endif

#ifdef CK_ENABLE_BF8
using BF8 = ck::bf8_t;
#endif

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

static constexpr auto ConvBwdWeightFilter1x1Stride1Pad0 =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0;

#if defined(CK_USE_GFX1250)
// The direct low-precision scalar store is safe only for a single K batch. A split K
// would use the packed half/bfloat16 atomic path, which requires paired output lanes.
template <typename DeviceOp>
struct DeviceGroupedConvBwdWeight_Wmma_CShuffleV3_Split1 : DeviceOp
{
    using Argument = typename DeviceOp::Argument;

    static bool IsSupportedArgument(const Argument& arg)
    {
        return arg.k_batch_ == 1 && DeviceOp::IsSupportedArgument(arg);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const auto* arg = dynamic_cast<const Argument*>(p_arg);
        return arg != nullptr && IsSupportedArgument(*arg);
    }

    struct Invoker : DeviceOp::Invoker
    {
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            const auto* arg = dynamic_cast<const Argument*>(p_arg);
            if(arg == nullptr || arg->k_batch_ != 1)
            {
                throw std::runtime_error("Scalar direct WRW requires a split-1 argument");
            }
            return DeviceOp::Invoker::Run(*arg, stream_config);
        }
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }

    std::string GetTypeString() const override { return DeviceOp::GetTypeString() + "_Split1"; }
};
#endif

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec,
          BlockGemmPipelineScheduler Scheduler     = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion PipelineVersion = BlockGemmPipelineVersion::v1>
using device_grouped_conv_bwd_weight_v3_wmma_c_shuffle_f16_instances = std::tuple<
    // clang-format off
    //#########################################|        Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|   ConvBackward| Block|  MPer|  NPer|  KPer| ABK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CShuffleBlockTransfer|  CShuffleBlockTransfer| BlockGemm|       BlockGemm|
    //#########################################|        Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|         Weight|  Size| Block| Block| Block|     | Wmma| Wmma|        |        |     ThreadCluster|    ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|        ClusterLengths|        ScalarPerVector|  Pipeline|        Pipeline|
    //#########################################|    Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation| Specialization|      |      |      |      |     |     |     |        |        | Lengths_AK0_M_AK1|     ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |              |      PerVector|  PerVector_BK1|          |  PerShuffle|  PerShuffle|      MBlock_MPerBlock|             _NPerBlock| Scheduler|         Version|
    //#########################################|           |         |          |          |       |        |        |        |            |            |            |               |      |      |      |      |     |     |     |        |        |                  |                 |               |               |               |               |          |                  |               |               |              |               |               |          |            |            |      NBlock_NPerBlock|                       |          |                |
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    64,    32,    32,    32,    8,   16,   16,       2,       1,        S<4, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              2,              2,         0,       S<4, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              2,              2,         0,           1,           1,         S<1, 8, 1, 8>,                      2, Scheduler, PipelineVersion>,
#if defined(CK_USE_GFX1250)
    // Scalar source/store permits odd per-group K/C; only the single-batch Set epilogue is safe.
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3_Split1<
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    32,    16,    16,    32,    8,   16,   16,       1,       1,        S<4, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              1,              4,         0,        S<4, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              1,              4,         0,           1,           1,         S<1, 4, 1, 8>,                      1, Scheduler, PipelineVersion>>,
#endif
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,   128,   128,    32,    8,   16,   16,       8,       2,       S<4, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         0,       S<4, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              4,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,   128,   128,    32,    8,   16,   16,       8,       2,       S<4, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         1,       S<4, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              4,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    64,    64,    64,    64,    8,   16,   16,       4,       2,        S<8, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         0,        S<8, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 4>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,   128,   256,    64,    8,   16,   16,       8,       2,       S<8, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         1,       S<8, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,       S<1, 16, 1, 16>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    48,    64,   128,    8,   16,   16,       3,       1,       S<16, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<16, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    96,   128,    64,    8,   16,   16,       6,       2,       S<8, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<8, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    64,    64,   128,    8,   16,   16,       4,       1,       S<16, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         0,       S<16, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,    96,   128,   128,    8,   16,   16,       6,       1,      S<16, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,      S<16, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,       S<1, 16, 1, 16>,                      8, Scheduler, PipelineVersion>
    // DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    96,    96,    96,    48,    8,   16,   16,       6,       2,       S<6, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         0,       S<6, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              6,              8,         0,           1,           1,        S<1, 16, 1, 6>,                      8, Scheduler, PipelineVersion>, // Incorrect results for at least GemmDefault
    // DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    96,    96,    96,    48,    8,   16,   16,       6,       2,       S<6, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<6, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              6,              8,         0,           1,           1,        S<1, 16, 1, 6>,                      8, Scheduler, PipelineVersion>  // Incorrect results for at least GemmDefault
    // clang-format on
    >;

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec,
          BlockGemmPipelineScheduler Scheduler     = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion PipelineVersion = BlockGemmPipelineVersion::v1>
using device_grouped_conv_bwd_weight_v3_wmma_c_shuffle_bf16_instances = std::tuple<
    // clang-format off
    //#########################################|        Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|   ConvBackward| Block|  MPer|  NPer|  KPer| ABK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CShuffleBlockTransfer|  CShuffleBlockTransfer| BlockGemm|       BlockGemm|
    //#########################################|        Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|         Weight|  Size| Block| Block| Block|     | Wmma| Wmma|        |        |     ThreadCluster|    ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|        ClusterLengths|        ScalarPerVector|  Pipeline|        Pipeline|
    //#########################################|    Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation| Specialization|      |      |      |      |     |     |     |        |        | Lengths_AK0_M_AK1|     ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |              |      PerVector|  PerVector_BK1|          |  PerShuffle|  PerShuffle|      MBlock_MPerBlock|             _NPerBlock| Scheduler|         Version|
    //#########################################|           |         |          |          |       |        |        |        |            |            |            |               |      |      |      |      |     |     |     |        |        |                  |                 |               |               |               |               |          |                  |               |               |              |               |               |          |            |            |      NBlock_NPerBlock|                       |          |                |
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    64,    32,    32,    32,    8,   16,   16,       2,       1,        S<4, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              2,              2,         0,       S<4, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              2,              2,         0,           1,           1,         S<1, 8, 1, 8>,                      2, Scheduler, PipelineVersion>,
#if defined(CK_USE_GFX1250)
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3_Split1<
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    32,    16,    16,    32,    8,   16,   16,       1,       1,        S<4, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              1,              4,         0,        S<4, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              1,              4,         0,           1,           1,         S<1, 4, 1, 8>,                      1, Scheduler, PipelineVersion>>,
#endif
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,   128,   128,    32,    8,   16,   16,       8,       2,       S<4, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         0,       S<4, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              4,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,   128,   128,    32,    8,   16,   16,       8,       2,       S<4, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         1,       S<4, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              4,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    64,    64,    64,    64,    8,   16,   16,       4,       2,        S<8, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         0,        S<8, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 4>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,   128,   256,    64,    8,   16,   16,       8,       2,       S<8, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         1,       S<8, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,       S<1, 16, 1, 16>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    48,    64,   128,    8,   16,   16,       3,       1,       S<16, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<16, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    96,   128,    64,    8,   16,   16,       6,       2,       S<8, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<8, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    64,    64,   128,    8,   16,   16,       4,       1,       S<16, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         0,       S<16, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,    96,   128,   128,    8,   16,   16,       6,       1,      S<16, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,      S<16, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,       S<1, 16, 1, 16>,                      8, Scheduler, PipelineVersion>
#ifndef CK_USE_GFX1250
    ,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    96,    96,    96,    48,    8,   16,   16,       6,       2,       S<6, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         0,       S<6, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              6,              8,         0,           1,           1,        S<1, 16, 1, 6>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    96,    96,    96,    48,    8,   16,   16,       6,       2,       S<6, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<6, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              6,              8,         0,           1,           1,        S<1, 16, 1, 6>,                      8, Scheduler, PipelineVersion>
#endif
    //clang-format on
    >;

// gfx1250-only large-tile instances, ported from the analogous
// device_grouped_conv_bwd_data_wmma_v3_bf16_large_tiles_instances /
// device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_large_tiles families
// (see GFX1250_CONV_OPTIMIZATION_ROADMAP.md item T2-05): fwd and bwd_data both
// ship a dedicated large-tile instance file compiled with
// "-mllvm -enable-post-misched=1 -mllvm --amdgpu-mfma-vgpr-form" (see their
// CMakeLists.txt COMPILE_OPTIONS), while bwd_weight's large tiles previously
// lived inline in the regular (non-flagged) instance file. These are the same
// tile configs moved out verbatim so they can be compiled with the same
// scheduling flags and benchmarked against the un-flagged baseline.
template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec,
          BlockGemmPipelineScheduler Scheduler     = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion PipelineVersion = BlockGemmPipelineVersion::v1>
using device_grouped_conv_bwd_weight_v3_wmma_c_shuffle_bf16_instances_large_tiles = std::tuple<
#if defined(CK_USE_GFX1250)
    // clang-format off
    //#########################################|        Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|   ConvBackward| Block|  MPer|  NPer|  KPer| ABK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CShuffleBlockTransfer|  CShuffleBlockTransfer| BlockGemm|       BlockGemm|
    //#########################################|        Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|         Weight|  Size| Block| Block| Block|     | Wmma| Wmma|        |        |     ThreadCluster|    ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|        ClusterLengths|        ScalarPerVector|  Pipeline|        Pipeline|
    //#########################################|    Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation| Specialization|      |      |      |      |     |     |     |        |        | Lengths_AK0_M_AK1|     ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |              |      PerVector|  PerVector_BK1|          |  PerShuffle|  PerShuffle|      MBlock_MPerBlock|             _NPerBlock| Scheduler|         Version|
    //#########################################|           |         |          |          |       |        |        |        |            |            |            |               |      |      |      |      |     |     |     |        |        |                  |                 |               |               |               |               |          |                  |               |               |              |               |               |          |            |            |      NBlock_NPerBlock|                       |          |                |
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,    64,   256,   256,    8,   16,   16,       2,       4,       S<32, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         1,       S<32, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 32, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,   128,   256,   128,    8,   16,   16,       4,       4,      S<16, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         1,      S<16, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 32, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,   256,   256,    64,    8,   16,   16,       8,       4,       S<8, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         1,       S<8, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 32, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   512,   128,   256,   256,    8,   16,   16,       4,       2,      S<32, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         1,      S<32, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,       S<1, 32, 1, 16>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,   256,   256,    64,    8,   16,   16,       8,       4,      S< 1, 32, 8>,       S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,      S< 1, 32, 8>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,        S<1, 32, 1, 8>,                      8, Scheduler, PipelineVersion, BF16, BF16, 1, 1, true, true>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   512,   128,   256,   128,    8,   16,   16,       4,       2,      S< 4, 16, 8>,       S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,      S< 2, 32, 8>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,        S<1, 32, 1,16>,                      8, Scheduler, PipelineVersion, BF16, BF16, 1, 1, true, true>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   512,   128,   128,    64,    8,   16,   16,       4,       1,      S< 4, 16, 8>,       S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,      S< 4, 16, 8>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,        S<1, 32, 1,16>,                      8, Scheduler, PipelineVersion, BF16, BF16, 1, 1, true, true>
#endif
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
