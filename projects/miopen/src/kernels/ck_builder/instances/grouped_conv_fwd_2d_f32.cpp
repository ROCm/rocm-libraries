#include <miopen/ck_builder/instances/grouped_conv_fwd_2d_f32.hpp>
#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using InLayout                             = ck::tensor_layout::convolution::NGCHW;
using WeiLayout                            = ck::tensor_layout::convolution::GKCYX;
using OutLayout                            = ck::tensor_layout::convolution::NGKHW;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
using EmptyTuple                           = ck::Tuple<>;
static constexpr ck::index_t NumDimSpatial = 2;
template <typename DataType>
using DeviceOpGFwdDefault =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<>,
                                                                  OutLayout,
                                                                  DataType,
                                                                  DataType,
                                                                  ck::Tuple<>,
                                                                  DataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  DataType,
                                                                  DataType>;

using DeviceOpGFWdDefaultFloat = DeviceOpGFwdDefault<float>;

constexpr auto FP32 = ckb::DataType::FP32;

constexpr auto
create_device_grouped_conv_fwd_xdl_f32_instance_data(std::size_t spatialDim,
                                                     ckb::TensorLayout inLayout,
                                                     ckb::TensorLayout weiLayout,
                                                     ckb::TensorLayout outLayout,
                                                     ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp

    // clang-format off
    std::array result = {
        // Instance 1: Generic instance
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 64, 16, 4, 4, 32, 32, 2, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 8, 1, 8}, 1,
            FP32, FP32),
        
        // Instance 2: Small conv.K and conv.C
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 32, 16, 4, 4, 32, 32, 2, 1,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 1,
            FP32, FP32),
        
        // Instance 3
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 16, 4, 4, 32, 32, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 4
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 128, 16, 4, 4, 32, 32, 4, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 5
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 256, 16, 4, 4, 32, 32, 2, 4,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 6
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 128, 16, 4, 4, 32, 32, 4, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 7
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 16, 4, 4, 32, 32, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 8
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 64, 16, 4, 4, 32, 32, 2, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 9
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 128, 16, 4, 4, 32, 32, 2, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 10
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 64, 16, 4, 4, 32, 32, 2, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 11
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 64, 16, 4, 4, 32, 32, 2, 1,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 12
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 64, 128, 16, 4, 4, 32, 32, 1, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 13
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 32, 16, 4, 4, 32, 32, 2, 1,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 14
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 128, 16, 4, 4, 32, 32, 1, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 15
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 32, 16, 4, 4, 32, 32, 2, 1,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 16
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 32, 64, 16, 4, 4, 32, 32, 1, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 17
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 192, 16, 4, 4, 32, 32, 2, 3,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32)

        // clang-format on
    };

    return result;
}

using BaseOperator    = ck::tensor_operation::device::BaseOperator;
using BaseOperatorPtr = std::unique_ptr<BaseOperator>;

template <auto arr>
void build_k()
{
    auto s = arr.size();
    std::cout << s << std::endl;
}

constexpr auto create_device_grouped_conv_fwd_xdl_f32_16x16_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp
    // device_grouped_conv_fwd_xdl_f32_16x16_instances

    return std::array<XdlInstance, 0>{};

    // TODO - Investigate why a_block_transfer_dst_scalar_per_vector_k1 = 8 is invalid according to
    // CK builder even though we are already creating kernels with it

    /*
    // clang-format off
    std::array result = {
        // Instance 1
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 64, 64, 32, 8, 8, 16, 16, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 2, 8, true,
            1, 1, {1, 32, 1, 4}, 1,
            FP32, FP32),

        // Instance 2
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 64, 64, 32, 8, 8, 16, 16, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 2, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 8, true,
            1, 1, {1, 32, 1, 4}, 2,
            FP32, FP32),

        // Instance 3
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 64, 64, 32, 8, 8, 16, 16, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 8, true,
            1, 1, {1, 32, 1, 4}, 4,
            FP32, FP32)
    };
    // clang-format on

    return result;
    //*/
}

constexpr auto create_device_grouped_conv_fwd_xdl_f32_comp_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp
    // device_grouped_conv_fwd_xdl_f32_comp_instances

    return std::array<XdlV3Instance, 0>{};

    // TODO - Investigate why c_block_transfer_scalar_per_vector = 8 is invalid according to CK
    // builder even though we are already creating kernels with it
    /*
    // clang-format off
    std::array result = {
        // Instance 1: Intrawave v4
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 32, 8, 8, 32, 32, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V4),

        // Instance 2: Intrawave v3
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 64, 8, 8, 32, 32, 2, 2,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V3),

        // Instance 3: Intrawave v5
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 64, 8, 8, 32, 32, 2, 2,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V5),

        // Instance 4: Interwave v1
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 64, 8, 8, 32, 32, 2, 2,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V1)
    };
    // clang-format on

    return result;
    //*/
}

constexpr auto create_device_grouped_conv_fwd_xdl_f32_mem_intra_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_mem_instance.hpp
    // device_grouped_conv_fwd_xdl_f32_mem_instances with Intrawave scheduler

    // clang-format off
    std::array result = {
        // Latency friendly instances (v1)
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V1),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 16, 16, 128, 8, 8, 16, 16, 1, 1,
            {16, 4, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {16, 4, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 4}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V1),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 16, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 8, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 8, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 4}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V1),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 16, 32, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 16, 64, 64, 8, 8, 16, 16, 1, 2,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        // TODO - Investigate why c_block_transfer_scalar_per_vector = 8 is invalid according to CK builder even though we are already creating kernels with it
        /*
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 64, 64, 8, 8, 32, 32, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 8,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        //*/
        
        // Memory friendly instances (v2)
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 32, 64, 8, 8, 32, 32, 2, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 16, 64, 8, 8, 16, 16, 4, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 32, 64, 8, 8, 32, 32, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 16, 64, 8, 8, 16, 16, 2, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP32, FP32,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2)
    };
    // clang-format on

    return result;
}

constexpr auto create_device_grouped_conv_fwd_xdl_f32_mem_inter_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_mem_instance.hpp
    // device_grouped_conv_fwd_xdl_f32_mem_instances with Interwave scheduler

    // clang-format off
    std::array result = {
        // Latency friendly instances (v1)
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V1),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 16, 16, 128, 8, 8, 16, 16, 1, 1,
            {16, 4, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {16, 4, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 4}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V1),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 16, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 8, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 8, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 4}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 16, 32, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 16, 64, 64, 8, 8, 16, 16, 1, 2,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        // TODO - Investigate why c_block_transfer_scalar_per_vector = 8 is invalid according to CK builder even though we are already creating kernels with it
        /*
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 64, 64, 8, 8, 32, 32, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 8,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        //*/
        
        // Memory friendly instances (v2)
        // TODO - This fails to build due to "desired occupancy was 2, final occupancy is 1"
        /*
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 32, 64, 8, 8, 32, 32, 2, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        */
        
        // TODO - This fails to build due to "desired occupancy was 2, final occupancy is 1"
        /*
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 16, 64, 8, 8, 16, 16, 4, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        */
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 32, 64, 8, 8, 32, 32, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 16, 64, 8, 8, 16, 16, 2, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP32, FP32,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2)
    };
    // clang-format on

    return result;
}

constexpr auto create_device_grouped_conv_fwd_xdl_merged_groups_f32_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp

    return std::array<XdlInstance, 0>{};

    // TODO: These instances have a a_block_transfer_src_vector_dim value of 1, which is invalid
    // according to the CK Builder constraints
    /*
    // clang-format off
    std::array result = {
        // Instance 1: NumGroupsToMerge = 8
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 16, 16, 4, 4, 16, 16, 4, 1,
            {4, 16, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 4}, 1,
            FP32, FP32, ckb::PipelineScheduler::DEFAULT, 8),

        // Instance 2: NumGroupsToMerge = 16
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 16, 16, 4, 4, 16, 16, 4, 1,
            {4, 16, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 4}, 1,
            FP32, FP32, ckb::PipelineScheduler::DEFAULT, 16),

        // Instance 3: NumGroupsToMerge = 32
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 16, 16, 4, 4, 16, 16, 4, 1,
            {4, 16, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 4}, 1,
            FP32, FP32, ckb::PipelineScheduler::DEFAULT, 32)
    };
    // clang-format on

    return result;
    //*/
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f32_instance_data(2,
                                                             ckb::TensorLayout::NGCHW,
                                                             ckb::TensorLayout::GKCYX,
                                                             ckb::TensorLayout::NGKHW,
                                                             ckb::ConvSpecialization::DEFAULT);

    constexpr auto filter1x1Pad0InstanceData = create_device_grouped_conv_fwd_xdl_f32_instance_data(
        2,
        ckb::TensorLayout::NGCHW,
        ckb::TensorLayout::GKCYX,
        ckb::TensorLayout::NGKHW,
        ckb::ConvSpecialization::FILTER_1X1_PAD0);

    constexpr auto filter1x1Stride1Pad0InstanceData =
        create_device_grouped_conv_fwd_xdl_f32_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0);

    constexpr auto instanceData =
        concat(defaultInstanceData, filter1x1Pad0InstanceData, filter1x1Stride1Pad0InstanceData);

    return instanceData;
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_16x16_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_16x16_instance.cpp

    constexpr auto defaultInstanceData = create_device_grouped_conv_fwd_xdl_f32_16x16_instance_data(
        2,
        ckb::TensorLayout::NGCHW,
        ckb::TensorLayout::GKCYX,
        ckb::TensorLayout::NGKHW,
        ckb::ConvSpecialization::DEFAULT);

    constexpr auto filter1x1Pad0InstanceData =
        create_device_grouped_conv_fwd_xdl_f32_16x16_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::FILTER_1X1_PAD0);

    constexpr auto filter1x1Stride1Pad0InstanceData =
        create_device_grouped_conv_fwd_xdl_f32_16x16_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0);

    constexpr auto instanceData =
        concat(defaultInstanceData, filter1x1Pad0InstanceData, filter1x1Stride1Pad0InstanceData);

    return instanceData;
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_comp_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_comp_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f32_comp_instance_data(2,
                                                                  ckb::TensorLayout::NGCHW,
                                                                  ckb::TensorLayout::GKCYX,
                                                                  ckb::TensorLayout::NGKHW,
                                                                  ckb::ConvSpecialization::DEFAULT);

    return defaultInstanceData;
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_intra_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_intra_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f32_mem_intra_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::DEFAULT);

    return defaultInstanceData;
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_inter_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_inter_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f32_mem_inter_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::DEFAULT);

    return defaultInstanceData;
}

constexpr auto
create_device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f32_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f32_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_merged_groups_f32_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::DEFAULT);

    constexpr auto filter3x3InstanceData =
        create_device_grouped_conv_fwd_xdl_merged_groups_f32_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::FILTER_3x3);

    constexpr auto instanceData = concat(defaultInstanceData, filter3x3InstanceData);

    return instanceData;
}

void add_grouped_conv_fwd_2d_f32(std::vector<BaseOperatorPtr>& instances)
{
    // Adapted from GetInstances() in the composable_kernel project's file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp

    constexpr auto xdlKernelData = concat(
        create_device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f32_instance_data(),
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_instance_data(),
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_16x16_instance_data());
    build_kernels<xdlKernelData>(instances);

    constexpr auto xdlV3KernelData = concat(
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_comp_instance_data(),
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_intra_instance_data(),
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_inter_instance_data());
    build_kernels<xdlV3KernelData>(instances);
}
} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
