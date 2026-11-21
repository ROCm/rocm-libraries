// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <gtest/gtest.h>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/image_to_column.hpp"

// Host API implementation
template <typename DataType>
class TestCkTileImageToColumn : public ::testing::Test
{
    static constexpr ck_tile::index_t VectorSize  = 1;
    static constexpr ck_tile::index_t NDimSpatial = 2;

    protected:
    void Run(const ck_tile::conv::ConvParam conv_params)
    {

        using ImLayout = ck_tile::tensor_layout::convolution::NHWGC;

        const auto G = conv_params.G_;
        const auto N = conv_params.N_;
        const auto C = conv_params.C_;

        const ck_tile::long_index_t NDoHoWo =
            N * std::accumulate(conv_params.output_spatial_lengths_.begin(),
                                std::next(conv_params.output_spatial_lengths_.begin(), NDimSpatial),
                                1,
                                std::multiplies<>());

        const ck_tile::long_index_t CZYX =
            C * std::accumulate(conv_params.filter_spatial_lengths_.begin(),
                                std::next(conv_params.filter_spatial_lengths_.begin(), NDimSpatial),
                                1,
                                std::multiplies<>());

        const auto in_desc =
            ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<ImLayout>(
                conv_params);
        const auto out_desc = ck_tile::HostTensorDescriptor({G, NDoHoWo, CZYX});

        // host verify
        ck_tile::HostTensor<DataType> in(in_desc);
        ck_tile::HostTensor<DataType> out_device(out_desc);
        ck_tile::HostTensor<DataType> out_host(out_desc);

        std::cout << "input: " << in.mDesc << std::endl;
        std::cout << "output: " << out_device.mDesc << std::endl;

        ck_tile::FillUniformDistributionIntegerValue<DataType>{-5.f, 5.f}(in);

        ck_tile::DeviceMem in_device_buf(in.get_element_space_size_in_bytes());
        ck_tile::DeviceMem out_device_buf(out_device.get_element_space_size_in_bytes());

        in_device_buf.ToDevice(in.data());

        using thread_tile = ck_tile::sequence<4, 4>;
        using warp_tile   = ck_tile::sequence<8, 128>;
        using block_tile  = ck_tile::sequence<32, 128>;

        using Shape = ck_tile::TileImageToColumnShape<thread_tile, warp_tile, block_tile>;

        using PipelineProblem = ck_tile::BlockImageToColumnProblem<DataType,
                                                                   DataType,
                                                                   Shape,
                                                                   NDimSpatial,
                                                                   VectorSize,
                                                                   VectorSize>;

        using Kernel = ck_tile::ImageToColumn<PipelineProblem>;

        auto kargs = Kernel::MakeKargs(
            in_device_buf.GetDeviceBuffer(),
            out_device_buf.GetDeviceBuffer(),
            G,
            N,
            C,
            ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(
                conv_params.input_spatial_lengths_),
            ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(
                conv_params.filter_spatial_lengths_),
            ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(
                conv_params.output_spatial_lengths_),
            ck_tile::to_array<ck_tile::long_index_t, NDimSpatial + 3>(in_desc.get_strides()),
            ck_tile::to_array<ck_tile::long_index_t, 3>(out_desc.get_strides()),
            ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.conv_filter_strides_),
            ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(
                conv_params.conv_filter_dilations_),
            ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.input_left_pads_),
            ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.input_right_pads_));

        ASSERT_TRUE(Kernel::IsSupportedArgument(kargs));

        const dim3 grids = Kernel::GridSize(
            kargs.N * kargs.output_spatial_lengths[0] * kargs.output_spatial_lengths[1],
            kargs.filter_spatial_lengths[0] * kargs.filter_spatial_lengths[1] * kargs.C,
            kargs.G);
        const dim3 blocks = Kernel::BlockSize();

        constexpr ck_tile::index_t kBlockPerCu = 2;

        ck_tile::launch_kernel(
            ck_tile::stream_config{},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        // reference
        ck_tile::reference_im2col<DataType, DataType, NDimSpatial>(in, out_host, conv_params);

        out_device_buf.FromDevice(out_device.data());
        bool pass = ck_tile::check_err(out_device, out_host);

        EXPECT_TRUE(pass);
    }
};

class TestCkTileImageToColumnFloat : public TestCkTileImageToColumn<float>
{
};

class TestCkTileImageToColumnHalf : public TestCkTileImageToColumn<ck_tile::half_t>
{
};

TEST_F(TestCkTileImageToColumnFloat, TestCorrectness)
{
    this->Run({2, 2, 4, 1, 192, {3, 3}, {28, 28}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->Run({2, 2, 64, 1, 64, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->Run({2, 1, 64, 1, 64, {1, 1}, {7, 7}, {3, 3}, {1, 1}, {0, 0}, {0, 0}});
    this->Run({2, 1, 64, 1, 64, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->Run({2, 2, 64, 1, 64, {3, 3}, {28, 28}, {2, 2}, {2, 2}, {1, 1}, {1, 1}});
}

TEST_F(TestCkTileImageToColumnHalf, TestCorrectness)
{
    this->Run({2, 2, 4, 1, 192, {3, 3}, {28, 28}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->Run({2, 2, 64, 1, 64, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->Run({2, 1, 64, 1, 64, {1, 1}, {7, 7}, {3, 3}, {1, 1}, {0, 0}, {0, 0}});
    this->Run({2, 1, 64, 1, 64, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->Run({2, 2, 64, 1, 64, {3, 3}, {28, 28}, {2, 2}, {2, 2}, {1, 1}, {1, 1}});
}

// IsSupportedArgument must reject arguments whose vector reads of the image would cross a
// filter-window row (https://github.com/ROCm/rocm-libraries/issues/15883). Host only, no launch.
namespace {
template <ck_tile::index_t VectorSize>
bool IsSupported(const ck_tile::conv::ConvParam& conv_params)
{
    constexpr ck_tile::index_t NDimSpatial = 2;

    using ImLayout    = ck_tile::tensor_layout::convolution::NHWGC;
    using thread_tile = ck_tile::sequence<4, 4>;
    using warp_tile   = ck_tile::sequence<64, 16>;
    using block_tile  = ck_tile::sequence<64, 16>;
    using Shape       = ck_tile::TileImageToColumnShape<thread_tile, warp_tile, block_tile>;
    using Problem     = ck_tile::
        BlockImageToColumnProblem<float, float, Shape, NDimSpatial, VectorSize, VectorSize>;
    using Kernel = ck_tile::ImageToColumn<Problem>;

    const auto in_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<ImLayout>(conv_params);
    const ck_tile::long_index_t NHoWo = conv_params.N_ * conv_params.output_spatial_lengths_[0] *
                                        conv_params.output_spatial_lengths_[1];
    const ck_tile::long_index_t CYX = conv_params.C_ * conv_params.filter_spatial_lengths_[0] *
                                      conv_params.filter_spatial_lengths_[1];
    const auto out_desc = ck_tile::HostTensorDescriptor({conv_params.G_, NHoWo, CYX});

    const auto kargs = Kernel::MakeKargs(
        nullptr,
        nullptr,
        conv_params.G_,
        conv_params.N_,
        conv_params.C_,
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.input_spatial_lengths_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.filter_spatial_lengths_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.output_spatial_lengths_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial + 3>(in_desc.get_strides()),
        ck_tile::to_array<ck_tile::long_index_t, 3>(out_desc.get_strides()),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.conv_filter_strides_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.conv_filter_dilations_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.input_left_pads_),
        ck_tile::to_array<ck_tile::long_index_t, NDimSpatial>(conv_params.input_right_pads_));

    return Kernel::IsSupportedArgument(kargs);
}
} // namespace

TEST(TestCkTileImageToColumnArgs, VectorAccessOfFilterWindowRow)
{
    // {NDimSpatial, G, N, K, C, filter, input, strides, dilations, left pads, right pads}

    // The reproduction of issue 15883: C = 3, X = 2, so X * C = 6 is not a multiple of 4.
    EXPECT_FALSE(IsSupported<4>({2, 1, 1, 1, 3, {2, 2}, {4, 4}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}));
    // The same arguments with vectors of 1 are supported.
    EXPECT_TRUE(IsSupported<1>({2, 1, 1, 1, 3, {2, 2}, {4, 4}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}));
    // C = 2, X = 2: X * C = 4 is a multiple of 4, so vectors may span pixels of a window row.
    EXPECT_TRUE(IsSupported<4>({2, 1, 1, 1, 2, {2, 2}, {4, 4}, {1, 1}, {1, 1}, {0, 0}, {0, 0}}));
    // C = 4: every vector stays inside one pixel.
    EXPECT_TRUE(IsSupported<4>({2, 1, 1, 1, 4, {3, 3}, {8, 8}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}));
}

TEST(TestCkTileImageToColumnArgs, VectorAccessOfPadsAndDilation)
{
    // W pad * C = 2 is not a multiple of 4.
    EXPECT_FALSE(IsSupported<4>({2, 1, 1, 1, 2, {2, 2}, {4, 4}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}));
    // W pad * C = 4, but with a W stride of 2 a vector would span two pixels.
    EXPECT_FALSE(IsSupported<4>({2, 1, 1, 1, 2, {2, 2}, {8, 8}, {2, 2}, {1, 1}, {2, 2}, {2, 2}}));
    // With a W dilation of 2, adjacent window pixels are not adjacent in memory.
    EXPECT_FALSE(IsSupported<4>({2, 1, 1, 1, 2, {2, 2}, {8, 8}, {1, 1}, {2, 2}, {0, 0}, {0, 0}}));
    // The same dilation with C = 4 is supported.
    EXPECT_TRUE(IsSupported<4>({2, 1, 1, 1, 4, {2, 2}, {8, 8}, {1, 1}, {2, 2}, {0, 0}, {0, 0}}));
}
