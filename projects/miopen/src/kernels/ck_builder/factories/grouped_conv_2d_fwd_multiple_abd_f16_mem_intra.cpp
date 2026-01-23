#include <miopen/ck_builder/factories/grouped_conv_2d_fwd_multiple_abd.hpp>

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/instances/xdl_v3.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

constexpr auto FP16 = ckb::DataType::FP16;
constexpr auto FP32 = ckb::DataType::FP32;

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

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_intra_instance_data()
{
    constexpr auto defaultInstanceData =
        create_device_grouped_conv_fwd_xdl_f16_mem_intra_instance_data(
            2,
            ckb::TensorLayout::NGCHW,
            ckb::TensorLayout::GKCYX,
            ckb::TensorLayout::NGKHW,
            ckb::ConvSpecialization::DEFAULT);

    return defaultInstanceData;
}

void add_f16_mem_intra_instances(std::vector<BaseOperatorPtr>& instances)
{
    constexpr auto kernelData = create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_intra_instance_data();
    build_kernels<kernelData>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
