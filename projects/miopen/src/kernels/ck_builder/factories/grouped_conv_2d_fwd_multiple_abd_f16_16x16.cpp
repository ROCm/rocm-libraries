#include <miopen/ck_builder/factories/grouped_conv_2d_fwd_multiple_abd.hpp>

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/instances/xdl.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

constexpr auto FP16 = ckb::DataType::FP16;
constexpr auto FP32 = ckb::DataType::FP32;

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

constexpr auto create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_16x16_instance_data()
{
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

void add_f16_16x16_instances(std::vector<BaseOperatorPtr>& instances)
{
    constexpr auto kernelData = create_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_16x16_instance_data();
    build_kernels<kernelData>(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
