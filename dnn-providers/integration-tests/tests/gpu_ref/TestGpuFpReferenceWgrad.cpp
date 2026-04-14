// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn_gpu_ref/GpuFpReferenceConvolution.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;

namespace
{

// Validates that two tensors are element-wise close using the standard allClose validator.
// Handles NaN/Inf detection, stride-aware indexing, and parallel comparison.
template <typename T>
void assertAllClose(TensorBase<T>& expected, TensorBase<T>& actual, float tolerance)
{
    auto validator = CpuFpReferenceValidation<T>(tolerance, 0.0f);
    ASSERT_TRUE(validator.allClose(expected, actual));
}

// Core helper: fills x and dy tensors, runs GPU and CPU wgrad, compares dw results.
// For wgrad: x and dy are inputs, dw is the output (weight gradient).
template <typename XDataType, typename DwDataType, typename DyDataType, typename ComputeDataType>
void compareGpuVsCpuConvWrw(Tensor<XDataType>& xTensor,
                            Tensor<DyDataType>& dyTensor,
                            Tensor<DwDataType>& dwCpu,
                            Tensor<DwDataType>& dwGpu,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& dilations,
                            const std::vector<int64_t>& prePadding,
                            const std::vector<int64_t>& postPadding,
                            float tolerance,
                            float fillRange)
{
    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(
        static_cast<XDataType>(-fillRange), static_cast<XDataType>(fillRange), seed);
    dyTensor.fillWithRandomValues(
        static_cast<DyDataType>(-fillRange), static_cast<DyDataType>(fillRange), seed + 1);

    CpuFpReferenceConvolution::wgrad<XDataType, DwDataType, DyDataType, ComputeDataType>(
        xTensor, dwCpu, dyTensor, strides, dilations, prePadding, postPadding);

    GpuFpReferenceConvolution::wgrad<XDataType, DwDataType, DyDataType, ComputeDataType>(
        xTensor, dwGpu, dyTensor, strides, dilations, prePadding, postPadding);

    assertAllClose(dwCpu, dwGpu, tolerance);
}

// Convenience wrapper for uniform-type wgrad tests.
// When layout is non-null, x and dy use channel-last strides.
// dw (weight gradient) always uses default packed (KCRS) strides.
template <typename DataType, typename ComputeDataType = double>
void runGpuVsCpuConvWrw(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& dyDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& prePadding,
                        const std::vector<int64_t>& postPadding,
                        float tolerance,
                        const TensorLayout* layout = nullptr,
                        float fillRange = 1.0f)
{
    auto xTensor = layout != nullptr ? Tensor<DataType>(xDims, *layout) : Tensor<DataType>(xDims);
    auto dyTensor
        = layout != nullptr ? Tensor<DataType>(dyDims, *layout) : Tensor<DataType>(dyDims);
    auto dwCpu = Tensor<DataType>(wDims);
    auto dwGpu = Tensor<DataType>(wDims);

    compareGpuVsCpuConvWrw<DataType, DataType, DataType, ComputeDataType>(xTensor,
                                                                          dyTensor,
                                                                          dwCpu,
                                                                          dwGpu,
                                                                          strides,
                                                                          dilations,
                                                                          prePadding,
                                                                          postPadding,
                                                                          tolerance,
                                                                          fillRange);
}

// ============================================================================
// ConvWgradShapeCase — shape parameters for parameterized wgrad tests
// ============================================================================

struct ConvWgradShapeCase
{
    std::vector<int64_t> xDims;
    std::vector<int64_t> wDims;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    std::vector<int64_t> padding;
    int64_t groups = 1;
    std::string tag;

    // When non-null, x and dy tensors use this channel-last layout (NHWC/NDHWC/NLC).
    // dw (weight gradient) always uses default packed (KCRS) strides regardless.
    const TensorLayout* layout = nullptr;

    // Computes the forward output dims (= dy dims for wgrad).
    std::vector<int64_t> computeOutputDims() const
    {
        auto numSpatialDims = xDims.size() - 2;
        std::vector<int64_t> dyDims = {xDims[0], wDims[0]};
        for(size_t i = 0; i < numSpatialDims; ++i)
        {
            auto outputSize
                = (xDims[2 + i] + 2 * padding[i] - dilations[i] * (wDims[2 + i] - 1) - 1)
                      / strides[i]
                  + 1;
            dyDims.push_back(outputSize);
        }
        return dyDims;
    }

    friend std::ostream& operator<<(std::ostream& os, const ConvWgradShapeCase& tc)
    {
        return os << tc.tag;
    }
};

// Returns copies with channel-last layout set on x/dy.
// 3-way branch: 5D → NDHWC, 4D → NHWC, 3D → NLC.
std::vector<ConvWgradShapeCase> withChannelLastLayout(std::vector<ConvWgradShapeCase> cases)
{
    for(auto& tc : cases)
    {
        if(tc.xDims.size() == 5)
        {
            tc.layout = &TensorLayout::NDHWC;
        }
        else if(tc.xDims.size() == 4)
        {
            tc.layout = &TensorLayout::NHWC;
        }
        else
        {
            tc.layout = &TensorLayout::NLC;
        }
    }
    return cases;
}

// ============================================================================
// Shape Catalog — centralized convolution shapes, categorized by size
// ============================================================================

// Small 2D shapes: output < 1K elements, suitable for all types
std::vector<ConvWgradShapeCase> getSmall2dWgradCases()
{
    return {
        // Basic single-channel 3x3 convolution, no padding
        {{1, 1, 8, 8}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 1, "Basic3x3"},
        // Multiple input/output channels with padding
        {{1, 3, 8, 8}, {6, 3, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, "MultiChanPad"},
        // 2-group convolution with multi-batch
        {{2, 4, 8, 8}, {4, 2, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 2, "Grouped2Batch2"},
        // Stride=2 downsampling
        {{1, 1, 8, 8}, {2, 1, 3, 3}, {2, 2}, {1, 1}, {0, 0}, 1, "Stride2"},
        // Dilation=2 (expanded receptive field)
        {{1, 1, 12, 12}, {1, 1, 3, 3}, {1, 1}, {2, 2}, {0, 0}, 1, "Dilation2"},
        // Depthwise convolution (groups == input channels)
        {{1, 3, 8, 8}, {3, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 3, "Depthwise3Chan"},
        // 1x1 pointwise convolution (channel mixing only)
        {{1, 8, 4, 4}, {16, 8, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "Pointwise1x1"},
        // Depthwise with odd group count
        {{1, 7, 8, 8}, {7, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 7, "DepthwiseOdd7"},
        // Minimum output: single element (3x3 input, 3x3 kernel)
        {{1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 1, "SingleElement"},
    };
}

// Medium 2D shapes: ResNet/ResNeXt/Inception-like, suitable for fp32 + fp16
std::vector<ConvWgradShapeCase> getMedium2dWgradCases()
{
    return {
        // ResNeXt-like 2-group block
        {{8, 64, 28, 28}, {128, 32, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 2, "ResNeXt2Group"},
        // ResNeXt-32x4d bottleneck (32 groups, 4 channels/group)
        {{8, 128, 14, 14}, {256, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXt32x4d"},
        // ResNet 1x1 pointwise reduction
        {{4, 64, 56, 56}, {64, 64, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "ResNet1x1Reduce"},
        // ResNet stem layer: 7x7 kernel, stride=2
        {{8, 3, 28, 28}, {64, 3, 7, 7}, {2, 2}, {1, 1}, {3, 3}, 1, "ResNetStem7x7"},
        // 8-group convolution
        {{8, 64, 14, 14}, {64, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "Grouped8"},
        // MobileNet-style depthwise (16 channels)
        {{4, 16, 48, 48}, {16, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 16, "MobileNetDW16"},
        // RGB 3-group with stride-2 downsampling
        {{8, 3, 108, 108}, {63, 1, 3, 3}, {2, 2}, {1, 1}, {1, 1}, 3, "RGB3GroupStride2"},
        // 2-group with 5x5 kernel
        {{4, 32, 28, 28}, {32, 16, 5, 5}, {1, 1}, {1, 1}, {2, 2}, 2, "Grouped2Kernel5x5"},
        // 8-group mid-resolution
        {{8, 128, 28, 28}, {128, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "Grouped8MidRes"},
        // Bottleneck 1x1 channel expansion
        {{2, 256, 14, 14}, {256, 256, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "Bottleneck1x1Expand"},
        // Small depthwise (4 channels)
        {{4, 4, 48, 48}, {16, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 4, "Depthwise4Chan"},
        // Odd channel count grouped (7 groups)
        {{8, 7, 14, 14}, {63, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 7, "OddChanGrouped7"},
    };
}

// Large 2D shapes: stress tests matching real workloads, fp32 only
std::vector<ConvWgradShapeCase> getLarge2dWgradCases()
{
    return {
        // ResNeXt-32x4d high-resolution block
        {{16, 128, 56, 56}, {256, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXt32x4dHiRes"},
        // ResNeXt deep 32-group (512->1024 channels)
        {{16, 512, 14, 14}, {1024, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXtDeep32Group"},
        // ResNeXt stride-2 downsample (256->512)
        {{16, 256, 28, 28}, {512, 8, 3, 3}, {2, 2}, {1, 1}, {1, 1}, 32, "ResNeXtStride2Down"},
        // Large stem: 3-group 7x7 on 224x224 input
        {{16, 3, 224, 224}, {63, 1, 7, 7}, {2, 2}, {1, 1}, {3, 3}, 3, "LargeStem7x7"},
        // Mid-resolution 8-group on 56x56
        {{8, 128, 56, 56}, {128, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "MidRes8Group56x56"},
        // Inception-like 5x5 kernel, 16-group
        {{16, 192, 28, 28}, {32, 12, 5, 5}, {1, 1}, {1, 1}, {2, 2}, 16, "Inception5x5x16Group"},
        // DeepSpeech-like non-square spatial (161x700)
        {{4, 4, 161, 700}, {32, 1, 5, 20}, {2, 2}, {1, 1}, {0, 0}, 4, "DeepSpeechNonSquare"},
        // Non-square spatial with 2-group (79x341)
        {{8, 32, 79, 341}, {32, 16, 5, 10}, {2, 2}, {1, 1}, {0, 0}, 2, "NonSquareGrouped2"},
    };
}

// Small 1D shapes: basic NCW convolution tests
std::vector<ConvWgradShapeCase> getSmall1dWgradCases()
{
    return {
        // Basic 1D: single-channel, kernel=3
        {{1, 1, 8}, {1, 1, 3}, {1}, {1}, {0}, 1, "Basic1d"},
        // 1D with padding
        {{1, 1, 6}, {1, 1, 3}, {1}, {1}, {1}, 1, "Padded1d"},
        // 1D with stride=2
        {{1, 1, 10}, {1, 1, 3}, {2}, {1}, {0}, 1, "Stride2x1d"},
        // 1D with dilation=2
        {{1, 1, 9}, {1, 1, 3}, {1}, {2}, {0}, 1, "Dilation2x1d"},
        // 1D multi-channel (3 in, 2 out)
        {{1, 3, 8}, {2, 3, 3}, {1}, {1}, {0}, 1, "MultiChan1d"},
        // 1D multi-batch
        {{2, 1, 8}, {1, 1, 3}, {1}, {1}, {0}, 1, "MultiBatch1d"},
        // 1D grouped (2 groups)
        {{1, 4, 8}, {4, 2, 3}, {1}, {1}, {0}, 2, "Grouped2x1d"},
        // 1D pointwise (kernel=1)
        {{1, 3, 8}, {2, 3, 1}, {1}, {1}, {0}, 1, "Pointwise1d"},
    };
}

// Medium 1D shapes: larger 1D convolutions for additional coverage
std::vector<ConvWgradShapeCase> getMedium1dWgradCases()
{
    return {
        // Multi-batch multi-channel with padding
        {{4, 16, 64}, {32, 16, 3}, {1}, {1}, {1}, 1, "MediumMultiChan1d"},
        // Grouped 4-group with larger spatial
        {{8, 32, 128}, {32, 8, 5}, {1}, {1}, {2}, 4, "Grouped4x1d"},
        // Stride=2 downsampling
        {{4, 8, 256}, {16, 8, 7}, {2}, {1}, {3}, 1, "Stride2Med1d"},
        // Depthwise 1D (8 channels)
        {{4, 8, 64}, {8, 1, 3}, {1}, {1}, {1}, 8, "Depthwise8x1d"},
        // Dilation=2 with padding
        {{2, 4, 32}, {8, 4, 3}, {1}, {2}, {2}, 1, "Dilation2Med1d"},
        // Large kernel pointwise (1x1)
        {{8, 64, 128}, {128, 64, 1}, {1}, {1}, {0}, 1, "Pointwise64to128x1d"},
    };
}

// Small 3D shapes: basic 3D convolution tests
std::vector<ConvWgradShapeCase> getSmall3dWgradCases()
{
    return {
        // Basic 3D: single-channel 3x3x3
        {{1, 1, 4, 4, 4}, {1, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "Basic3d"},
        // 3D with padding
        {{1, 1, 6, 6, 6}, {1, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "Padded3d"},
        // 3D grouped (2 groups)
        {{2, 4, 4, 4, 4}, {8, 2, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 2, "Grouped2x3d"},
        // 3D with stride=2
        {{1, 1, 5, 5, 5}, {1, 1, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, 1, "Stride2x3d"},
        // 3D with dilation=2
        {{1, 1, 7, 7, 7}, {1, 1, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}, 1, "Dilation2x3d"},
        // 3D multi-channel (3 in, 2 out)
        {{1, 3, 4, 4, 4}, {2, 3, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "MultiChan3d"},
    };
}

// Medium 3D shapes: larger 3D convolutions
std::vector<ConvWgradShapeCase> getMedium3dWgradCases()
{
    return {
        // Standard 3D with 16 input channels and padding
        {{2, 16, 8, 8, 8}, {32, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "Standard16Ch3d"},
        // Non-cube spatial dimensions (4x14x14)
        {{1, 16, 4, 14, 14}, {16, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "NonCube3d"},
        // Large 5x5x5 kernel
        {{2, 16, 8, 8, 8}, {32, 16, 5, 5, 5}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "Kernel5x5x5"},
    };
}

// Alias to avoid verbose braced-init-list issues inside EXPECT_THROW macros
using Vec = std::vector<int64_t>;

} // namespace

// ============================================================================
// TestGpuConvWrwRefValidation — validateInput throw paths (via wgrad)
// ============================================================================

TEST(TestGpuConvWrwRefValidation, ThrowsOnInvalidDimCount)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({8, 8});
    Tensor<float> dw({8, 8});
    Tensor<float> dy({8, 8});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(x, dw, dy, Vec{1}, Vec{1}, Vec{0}, Vec{0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnWeightDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnOutputDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnStridesSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::wgrad<float>(x, dw, dy, Vec{1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnDilationsSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnPrePaddingSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::wgrad<float>(x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnPostPaddingSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnZeroStride)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{0, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnNegativeDilation)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, -1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnNegativePrePadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{-1, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnNegativePostPadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, -1}),
                 std::invalid_argument);
}

TEST(TestGpuConvWrwRefValidation, ThrowsOnOutputDimValueMismatch)
{
    SKIP_IF_NO_DEVICES();
    // Input [1,1,4,4], kernel [1,1,3,3], no padding, stride 1 -> expected dy [1,1,2,2]
    // Provide wrong dy dims [1,1,3,3]
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> dw({1, 1, 3, 3});
    Tensor<float> dy({1, 1, 3, 3});

    EXPECT_THROW(GpuFpReferenceConvolution::wgrad<float>(
                     x, dw, dy, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

// ============================================================================
// TestGpuConvWrwRefAsymPad — asymmetric (pre != post) padding tests
// ============================================================================

TEST(TestGpuConvWrwRefAsymPadFp32, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 1e-5f);
}

TEST(TestGpuConvWrwRefAsymPadFp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<half>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 5e-2f);
}

TEST(TestGpuConvWrwRefAsymPadBfp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvWrw<bfloat16>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 0.1f);
}

// ============================================================================
// TestGpuConvWrwRefAlphaBeta — alpha/beta scaling tests
// ============================================================================

TEST(TestGpuConvWrwRefAlphaBeta, AlphaOnly)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dwRef({1, 1, 3, 3});
    Tensor<float> dwScaled({1, 1, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Compute with alpha=1.0
    GpuFpReferenceConvolution::wgrad<float>(xTensor, dwRef, dyTensor, {1, 1}, {1, 1}, {0, 0});

    // Compute with alpha=2.0
    GpuFpReferenceConvolution::wgrad<float>(
        xTensor, dwScaled, dyTensor, {1, 1}, {1, 1}, {0, 0}, 2.0);

    const auto* refData = dwRef.memory().hostData();
    const auto* scaledData = dwScaled.memory().hostData();

    for(size_t i = 0; i < 9; ++i)
    {
        ASSERT_NEAR(scaledData[i], 2.0f * refData[i], 1e-5f) << "Alpha scaling failed at " << i;
    }
}

TEST(TestGpuConvWrwRefAlphaBeta, BetaAccumulate)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dwTensor({1, 1, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    dwTensor.fillWithValue(1.0f);

    // Pre-fill dw with 1.0, then compute with alpha=1.0, beta=1.0
    // Result should be wgrad(x, dy) + 1.0
    Tensor<float> dwNoAccum({1, 1, 3, 3});
    GpuFpReferenceConvolution::wgrad<float>(xTensor, dwNoAccum, dyTensor, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::wgrad<float>(
        xTensor, dwTensor, dyTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 1.0);

    const auto* noAccumData = dwNoAccum.memory().hostData();
    const auto* accumData = dwTensor.memory().hostData();

    for(size_t i = 0; i < 9; ++i)
    {
        ASSERT_NEAR(accumData[i], noAccumData[i] + 1.0f, 1e-5f)
            << "Beta accumulation failed at " << i;
    }
}

TEST(TestGpuConvWrwRefAlphaBeta, BetaZeroSkipsRead)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dwBetaZero({1, 1, 3, 3});
    Tensor<float> dwDefault({1, 1, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Pre-fill with garbage — should be ignored when beta=0
    dwBetaZero.fillWithValue(999.0f);

    GpuFpReferenceConvolution::wgrad<float>(xTensor, dwDefault, dyTensor, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::wgrad<float>(
        xTensor, dwBetaZero, dyTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0);

    const auto* defaultData = dwDefault.memory().hostData();
    const auto* betaZeroData = dwBetaZero.memory().hostData();

    for(size_t i = 0; i < 9; ++i)
    {
        ASSERT_NEAR(betaZeroData[i], defaultData[i], 1e-5f)
            << "Beta=0 should ignore pre-filled data at " << i;
    }
}

// ============================================================================
// TestGpuConvWrwRefStridedFp32 — non-packed (strided) tensor tests
// Verifies stride-based indexing with memory gaps between elements.
// ============================================================================

TEST(TestGpuConvWrwRefStridedFp32, NonPackedInput)
{
    SKIP_IF_NO_DEVICES();

    // x: [1, 2, 4, 4] with inter-channel gap (stride[1]=32 vs packed 16)
    const std::vector<int64_t> xDims = {1, 2, 4, 4};
    const std::vector<int64_t> xStrides = {64, 32, 4, 1}; // packed would be {32, 16, 4, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> dyTensor({1, 1, 2, 2});
    Tensor<float> dwCpu({1, 2, 3, 3});
    Tensor<float> dwGpu({1, 2, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwCpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwGpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(dwCpu, dwGpu, 1e-5f);
}

TEST(TestGpuConvWrwRefStridedFp32, NonPackedDy)
{
    SKIP_IF_NO_DEVICES();

    // dy: [1, 1, 4, 4] with inter-row gap (stride[2]=8 vs packed 4)
    const std::vector<int64_t> dyDims = {1, 1, 4, 4};
    const std::vector<int64_t> dyStrides = {32, 32, 8, 1}; // packed would be {16, 16, 4, 1}

    Tensor<float> xTensor({1, 1, 6, 6});
    Tensor<float> dyTensor(dyDims, dyStrides);
    Tensor<float> dwCpu({1, 1, 3, 3});
    Tensor<float> dwGpu({1, 1, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwCpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwGpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(dwCpu, dwGpu, 1e-5f);
}

TEST(TestGpuConvWrwRefStridedFp32, NonPackedInputAndDy)
{
    SKIP_IF_NO_DEVICES();

    // Both x and dy have non-packed strides with inter-row gaps
    const std::vector<int64_t> xDims = {1, 2, 4, 4};
    const std::vector<int64_t> xStrides = {64, 32, 6, 1}; // packed would be {32, 16, 4, 1}

    const std::vector<int64_t> dyDims = {1, 1, 2, 2};
    const std::vector<int64_t> dyStrides = {8, 8, 4, 1}; // packed would be {4, 4, 2, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> dyTensor(dyDims, dyStrides);
    Tensor<float> dwCpu({1, 2, 3, 3});
    Tensor<float> dwGpu({1, 2, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwCpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwGpu, dyTensor, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(dwCpu, dwGpu, 1e-5f);
}

TEST(TestGpuConvWrwRefStridedFp32, NonPackedWithPadding)
{
    SKIP_IF_NO_DEVICES();

    // Non-packed input with padding to exercise both features together
    const std::vector<int64_t> xDims = {1, 2, 3, 3};
    const std::vector<int64_t> xStrides = {36, 18, 3, 1}; // packed would be {18, 9, 3, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> dyTensor({1, 1, 3, 3});
    Tensor<float> dwCpu({1, 2, 3, 3});
    Tensor<float> dwGpu({1, 2, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    dyTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwCpu, dyTensor, {1, 1}, {1, 1}, {1, 1});

    GpuFpReferenceConvolution::wgrad<float, float, float, double>(
        xTensor, dwGpu, dyTensor, {1, 1}, {1, 1}, {1, 1});

    assertAllClose(dwCpu, dwGpu, 1e-5f);
}

// ============================================================================
// TestGpuConvWrwRefShapes — parameterized shape coverage across types
// ============================================================================

template <typename DataType>
class ConvWgradShapeSuite : public ::testing::TestWithParam<ConvWgradShapeCase>
{
protected:
    static float tolerance(const ConvWgradShapeCase& tc)
    {
        constexpr double FILL_RANGE = 1.0;
        auto dyDims = tc.computeOutputDims();
        return hipdnn_test_sdk::utilities::conv::
            calculateConvWrwTolerance<DataType, DataType, double>(
                -FILL_RANGE, FILL_RANGE, -FILL_RANGE, FILL_RANGE, dyDims);
    }

    void runConvWgradShapeTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        auto dyDims = tc.computeOutputDims();
        runGpuVsCpuConvWrw<DataType>(tc.xDims,
                                     tc.wDims,
                                     dyDims,
                                     tc.strides,
                                     tc.dilations,
                                     tc.padding,
                                     tc.padding,
                                     tolerance(tc),
                                     tc.layout);
    }
};

using TestGpuConvWrwRefShapesFp32 = ConvWgradShapeSuite<float>;
using TestGpuConvWrwRefShapesFp16 = ConvWgradShapeSuite<half>;
using TestGpuConvWrwRefShapesBfp16 = ConvWgradShapeSuite<bfloat16>;

TEST_P(TestGpuConvWrwRefShapesFp32, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefShapesFp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}
TEST_P(TestGpuConvWrwRefShapesBfp16, MatchesCpuRef)
{
    this->runConvWgradShapeTest();
}

// ============================================================================
// Default layout (NCHW/NCDHW/NCW) instantiations — packed strides, no layout set.
// ============================================================================

// fp32 NCHW/NCDHW: all sizes (small + medium + large 2D, small + medium 3D, small + medium 1D)
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getSmall2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Medium2d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getMedium2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Large2d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getLarge2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getSmall3dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Medium3d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getMedium3dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });

// fp32 NCW: 1D shapes (small + medium)
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getSmall1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Medium1d,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(getMedium1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16 NCHW/NCDHW/NCW: small + medium 2D, small + medium 1D, small 3D
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(getSmall2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Medium2d,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(getMedium2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(getSmall1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Medium1d,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(getMedium1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(getSmall3dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16 NCHW/NCDHW/NCW: small + medium 2D, small + medium 1D, small 3D
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(getSmall2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Medium2d,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(getMedium2dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(getSmall1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Medium1d,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(getMedium1dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(getSmall3dWgradCases()),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });

// ============================================================================
// Channel-last (NHWC/NDHWC/NLC) instantiations — same suites, same catalog,
// but withChannelLastLayout() sets tc.layout so the fixture uses channel-last
// strides on x and dy tensors. dw (weight gradient) always stays packed (KCRS).
// ============================================================================

// fp32 NHWC/NDHWC/NLC: all sizes
INSTANTIATE_TEST_SUITE_P(Nhwc2dSmall,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Nhwc2dMedium,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Nhwc2dLarge,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dSmall,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Ndhwc3dMedium,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nlc1dSmall,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Nlc1dMedium,
                         TestGpuConvWrwRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16 NHWC/NDHWC/NLC: small + medium 2D, small 3D, small + medium 1D
INSTANTIATE_TEST_SUITE_P(Nhwc2dSmall,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Nhwc2dMedium,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dSmall,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nlc1dSmall,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Nlc1dMedium,
                         TestGpuConvWrwRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16 NHWC/NDHWC/NLC: small + medium 2D, small 3D, small + medium 1D
INSTANTIATE_TEST_SUITE_P(Nhwc2dSmall,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Nhwc2dMedium,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dSmall,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Nlc1dSmall,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Nlc1dMedium,
                         TestGpuConvWrwRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium1dWgradCases())),
                         [](const ::testing::TestParamInfo<ConvWgradShapeCase>& info) {
                             return info.param.tag;
                         });
