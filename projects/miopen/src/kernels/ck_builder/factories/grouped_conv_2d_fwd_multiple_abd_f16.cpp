#include <miopen/ck_builder/factories/grouped_conv_2d_fwd_multiple_abd.hpp>

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/instances/xdl.hpp>
#include <miopen/ck_builder/instances/xdl_v3.hpp>

#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

constexpr auto FP16 = ckb::DataType::FP16;
constexpr auto FP32 = ckb::DataType::FP32;

constexpr auto
create_device_grouped_conv_fwd_xdl_f16_instance_data(std::size_t spatialDim,
                                                     ckb::TensorLayout inLayout,
                                                     ckb::TensorLayout weiLayout,
                                                     ckb::TensorLayout outLayout,
                                                     ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp
    // device_grouped_conv_fwd_xdl_f16_instances

    // clang-format off
    std::array result = {
        // Instance 1: Generic instance
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 64, 32, 8, 8, 32, 32, 2, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 8, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 8, true,
            1, 1, {1, 16, 1, 4}, 1,
            FP16, FP16),
        
        // Instance 2: Small conv.K and conv.C
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 32, 32, 8, 8, 32, 32, 2, 1,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 16, 1, 4}, 1,
            FP16, FP16),
        
        // Instance 3
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 32, 8, 8, 32, 32, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 8, true,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16),
        
        // Instance 4
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 128, 32, 8, 8, 32, 32, 4, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16),
        
        // Instance 5
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 256, 32, 8, 8, 32, 32, 2, 4,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16),
        
        // Instance 6
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 128, 32, 8, 8, 32, 32, 4, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 16, 1, 8}, 8,
            FP16, FP16),
        
        // Instance 7
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 32, 8, 8, 32, 32, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16),
        
        // Instance 8
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 64, 32, 8, 8, 32, 32, 2, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 4}, 8,
            FP16, FP16),
        
        // Instance 9
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 128, 32, 8, 8, 32, 32, 2, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 16, 1, 8}, 8,
            FP16, FP16),
        
        // Instance 10
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 64, 32, 8, 8, 32, 32, 2, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 16, 1, 4}, 8,
            FP16, FP16),
        
        // Instance 11
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 64, 32, 8, 8, 32, 32, 2, 1,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16),
        
        // Instance 12
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 64, 128, 32, 8, 8, 32, 32, 1, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16),
        
        // Instance 13
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 32, 32, 8, 8, 32, 32, 2, 1,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 4}, 8,
            FP16, FP16),
        
        // Instance 14
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 128, 32, 8, 8, 32, 32, 1, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 16, 1, 8}, 8,
            FP16, FP16),
        
        // Instance 15
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 32, 32, 8, 8, 32, 32, 2, 1,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 16, 1, 4}, 8,
            FP16, FP16),
        
        // Instance 16
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 32, 64, 32, 8, 8, 32, 32, 1, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 16, 1, 4}, 8,
            FP16, FP16)

        // clang-format on
    };

    return result;
}

constexpr auto create_device_grouped_conv_fwd_xdl_f16_16x16_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp
    // device_grouped_conv_fwd_xdl_f16_16x16_instances

    // clang-format off
    std::array result = {
        // Instance 1
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 64, 64, 32, 8, 8, 16, 16, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 2, 8, true,
            1, 1, {1, 32, 1, 4}, 1,
            FP16, FP16),

        // Instance 2
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 64, 64, 32, 8, 8, 16, 16, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 2, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 8, true,
            1, 1, {1, 32, 1, 4}, 2,
            FP16, FP16),

        // Instance 3
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 64, 64, 32, 8, 8, 16, 16, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 8, true,
            1, 1, {1, 32, 1, 4}, 4,
            FP16, FP16),

        // Instance 4
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 64, 64, 32, 8, 8, 16, 16, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 4}, 8,
            FP16, FP16)
    };
    // clang-format on

    return result;
}

constexpr auto create_device_grouped_conv_fwd_xdl_f16_comp_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp
    // device_grouped_conv_fwd_xdl_f16_comp_instances

    // clang-format off
    std::array result = {
        // Instance 1
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 64, 8, 8, 32, 32, 2, 2,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V4),

        // Instance 2
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 64, 8, 8, 32, 32, 2, 2,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V5),

        // Instance 3
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 256, 32, 8, 8, 32, 32, 2, 4,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V1),

        // Instance 4
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 128, 32, 8, 8, 32, 32, 4, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V1),

        // Instance 5
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 64, 8, 8, 32, 32, 2, 2,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V1)
    };
    // clang-format on

    return result;
}

constexpr auto create_device_grouped_conv_fwd_xdl_f16_comp_2x_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp
    // device_grouped_conv_fwd_xdl_f16_comp_instances_2x
    // Double rate mfma instances on gfx950

    // clang-format off
    std::array result = {
        // Instance 1
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 64, 16, 16, 32, 32, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, true,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V1)
    };
    // clang-format on

    return result;
}

constexpr auto create_device_grouped_conv_fwd_xdl_f16_comp_part2_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp
    // device_grouped_conv_fwd_xdl_f16_comp_instances_part2
    // Instances not working on gfx950

    // clang-format off
    std::array result = {
        // Instance 1
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 256, 32, 8, 8, 32, 32, 4, 4,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V4),

        // Instance 2
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 32, 8, 8, 32, 32, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V4),

        // Instance 3
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 256, 32, 8, 8, 32, 32, 4, 4,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V3),

        // Instance 4
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 256, 32, 8, 8, 32, 32, 4, 4,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V5),

        // Instance 5
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 256, 32, 8, 8, 16, 16, 8, 8,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 2, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V3),

        // Instance 6: AGPR Spill when use permuted lds layout, so use padding
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 224, 256, 64, 8, 8, 16, 16, 7, 8,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 2, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V3),

        // Instance 7: AGPR Spill when use permuted lds layout, so use padding
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 224, 64, 8, 8, 16, 16, 8, 7,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            2, 1, {1, 64, 1, 4}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V3),

        // Instance 8
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 64, 8, 8, 32, 32, 2, 2,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V3),

        // Instance 9
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 64, 8, 8, 32, 32, 2, 2,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V5)
    };
    // clang-format on

    return result;
}

constexpr auto create_device_grouped_conv_fwd_xdl_f16_mem_intra_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_mem_instance.hpp
    // device_grouped_conv_fwd_xdl_f16_mem_instances with Intrawave scheduler

    // clang-format off
    std::array result = {
        // Latency friendly instances (v1)
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V1),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 16, 16, 128, 8, 8, 16, 16, 1, 1,
            {16, 4, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {16, 4, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 4}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V1),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 16, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 8, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 8, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 4}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V1),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 16, 32, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V1),
        
        // Memory friendly instances (v2)
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 32, 64, 8, 8, 32, 32, 2, 1,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 16, 64, 8, 8, 16, 16, 4, 1,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 2,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 32, 64, 8, 8, 32, 32, 2, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 16, 64, 8, 8, 16, 16, 4, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 32, 64, 8, 8, 32, 32, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 16, 64, 8, 8, 16, 16, 2, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 16, 16, 128, 8, 8, 16, 16, 1, 1,
            {16, 4, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {16, 4, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 4}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 16, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 8, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 8, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 4}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 16, 32, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTRAWAVE, ckb::PipelineVersion::V1)
    };
    // clang-format on

    return result;
}

constexpr auto create_device_grouped_conv_fwd_xdl_f16_mem_inter_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_mem_instance.hpp
    // device_grouped_conv_fwd_xdl_f16_mem_instances with Interwave scheduler

    // clang-format off
    std::array result = {
        // Memory friendly instances (v2)
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 32, 64, 8, 8, 32, 32, 2, 1,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 16, 64, 8, 8, 16, 16, 4, 1,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 32, 1, 8}, 2,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 32, 64, 8, 8, 32, 32, 2, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 16, 64, 8, 8, 16, 16, 4, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 32, 64, 8, 8, 32, 32, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 16, 64, 8, 8, 16, 16, 2, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 2,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 16, 16, 128, 8, 8, 16, 16, 1, 1,
            {16, 4, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {16, 4, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 4}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 16, 16, 64, 8, 8, 16, 16, 1, 1,
            {8, 8, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 8, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 4}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 16, 32, 64, 8, 8, 16, 16, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 16, 64, 64, 8, 8, 16, 16, 1, 2,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 64, 64, 8, 8, 32, 32, 1, 1,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 16, 128, 64, 8, 8, 16, 16, 1, 4,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 128, 64, 8, 8, 32, 32, 1, 2,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 8}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 16, 256, 64, 8, 8, 16, 16, 1, 4,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 16}, 4,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2),
        
        make_xdl_v3_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 32, 256, 64, 8, 8, 32, 32, 1, 2,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            {8, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false,
            1, 1, {1, 16, 1, 16}, 8,
            FP16, FP16,
            ckb::PipelineScheduler::INTERWAVE, ckb::PipelineVersion::V2)
    };
    // clang-format on

    return result;
}

constexpr auto
create_device_grouped_conv_fwd_xdl_merged_groups_f16_instance_data(
    std::size_t spatialDim,
    ckb::TensorLayout inLayout,
    ckb::TensorLayout weiLayout,
    ckb::TensorLayout outLayout,
    ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp
    // device_grouped_conv_fwd_xdl_merged_groups_f16_instances

    return std::array<XdlInstance, 0>{};

    // TODO: These instances have a a_block_transfer_src_vector_dim value of 1, which is invalid
    /*
    // clang-format off
    std::array result = {
        // Instance 1: NumGroupsToMerge = 8
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 16, 16, 4, 4, 16, 16, 4, 1,
            {4, 16, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 4}, 1,
            FP16, FP16, ckb::PipelineScheduler::DEFAULT, 8),

        // Instance 2: NumGroupsToMerge = 16
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 16, 16, 4, 4, 16, 16, 4, 1,
            {4, 16, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 4}, 1,
            FP16, FP16, ckb::PipelineScheduler::DEFAULT, 16),

        // Instance 3: NumGroupsToMerge = 32
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP16, FP16, FP32, FP16, FP16,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 16, 16, 4, 4, 16, 16, 4, 1,
            {4, 16, 1}, {0, 2, 1}, {0, 2, 1}, 1, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 4}, 1,
            FP16, FP16, ckb::PipelineScheduler::DEFAULT, 32)
    };
    // clang-format on

    return result;
    //*/
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f16_instance_data(2,
                                                             ckb::TensorLayout::NGCHW,
                                                             ckb::TensorLayout::GKCYX,
                                                             ckb::TensorLayout::NGKHW,
                                                             ckb::ConvSpecialization::DEFAULT);

    constexpr auto filter1x1Pad0InstanceData = create_device_grouped_conv_fwd_xdl_f16_instance_data(
        2,
        ckb::TensorLayout::NGCHW,
        ckb::TensorLayout::GKCYX,
        ckb::TensorLayout::NGKHW,
        ckb::ConvSpecialization::FILTER_1X1_PAD0);

    constexpr auto filter1x1Stride1Pad0InstanceData =
        create_device_grouped_conv_fwd_xdl_f16_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0);

    constexpr auto instanceData =
        concat(defaultInstanceData, filter1x1Pad0InstanceData, filter1x1Stride1Pad0InstanceData);

    return instanceData;
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_16x16_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_16x16_instance.cpp

    constexpr auto defaultInstanceData = create_device_grouped_conv_fwd_xdl_f16_16x16_instance_data(
        2,
        ckb::TensorLayout::NGCHW,
        ckb::TensorLayout::GKCYX,
        ckb::TensorLayout::NGKHW,
        ckb::ConvSpecialization::DEFAULT);

    constexpr auto filter1x1Pad0InstanceData =
        create_device_grouped_conv_fwd_xdl_f16_16x16_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::FILTER_1X1_PAD0);

    constexpr auto filter1x1Stride1Pad0InstanceData =
        create_device_grouped_conv_fwd_xdl_f16_16x16_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0);

    constexpr auto instanceData =
        concat(defaultInstanceData, filter1x1Pad0InstanceData, filter1x1Stride1Pad0InstanceData);

    return instanceData;
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f16_comp_instance_data(2,
                                                                  ckb::TensorLayout::NGCHW,
                                                                  ckb::TensorLayout::GKCYX,
                                                                  ckb::TensorLayout::NGKHW,
                                                                  ckb::ConvSpecialization::DEFAULT);

    return defaultInstanceData;
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_2x_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_2x_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f16_comp_2x_instance_data(2,
                                                                     ckb::TensorLayout::NGCHW,
                                                                     ckb::TensorLayout::GKCYX,
                                                                     ckb::TensorLayout::NGKHW,
                                                                     ckb::ConvSpecialization::DEFAULT);

    return defaultInstanceData;
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_part2_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_part2_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f16_comp_part2_instance_data(2,
                                                                        ckb::TensorLayout::NGCHW,
                                                                        ckb::TensorLayout::GKCYX,
                                                                        ckb::TensorLayout::NGKHW,
                                                                        ckb::ConvSpecialization::DEFAULT);

    return defaultInstanceData;
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_intra_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_intra_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f16_mem_intra_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::DEFAULT);

    return defaultInstanceData;
}

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_inter_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_inter_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f16_mem_inter_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::DEFAULT);

    return defaultInstanceData;
}

constexpr auto
create_device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f16_instance_data()
{
    // Adapted from the composable_kernel project, file:
    // library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f16_instance.cpp

    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_merged_groups_f16_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::DEFAULT);

    constexpr auto filter3x3InstanceData =
        create_device_grouped_conv_fwd_xdl_merged_groups_f16_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::FILTER_3x3);

    constexpr auto instanceData = concat(defaultInstanceData, filter3x3InstanceData);

    return instanceData;
}

std::vector<BaseOperatorPtr> DeviceOperationInstanceFactory<DeviceOpGFwdDefault<ck::half_t>>::GetInstances()
{
    // Adapted from GetInstances() in the composable_kernel project's file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp
    std::vector<BaseOperatorPtr> instances{};
    
    constexpr auto xdlKernelData = concat(
        create_device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f16_instance_data(),
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_instance_data(),
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_16x16_instance_data()
    );
    build_kernels<xdlKernelData>(instances);

    constexpr auto xdlV3KernelData = concat(
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_instance_data(),
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_2x_instance_data(),
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_part2_instance_data(),
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_intra_instance_data(),
        create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_inter_instance_data()
    );
    build_kernels<xdlV3KernelData>(instances);
    
    return instances;
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
