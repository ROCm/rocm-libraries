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
#include <hipdnn_gpu_ref/detail/GpuRefHipError.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn_test_sdk/utilities/ConvolutionValidation.hpp>

#include <chrono>
#include <cmath>
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

// Core helper: fills tensors, runs GPU and CPU convolution, compares results.
// Separate template params support mixed input/weight types (WDataType != XDataType).
template <typename XDataType, typename WDataType, typename YDataType, typename ComputeDataType>
void compareGpuVsCpuConvFwd(Tensor<XDataType>& xTensor,
                            Tensor<WDataType>& wTensor,
                            Tensor<YDataType>& yCpu,
                            Tensor<YDataType>& yGpu,
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
    wTensor.fillWithRandomValues(
        static_cast<WDataType>(-fillRange), static_cast<WDataType>(fillRange), seed + 1);

    CpuFpReferenceConvolution::fprop<XDataType, WDataType, YDataType, ComputeDataType>(
        xTensor, wTensor, yCpu, strides, dilations, prePadding, postPadding);

    GpuFpReferenceConvolution::fprop<XDataType, WDataType, YDataType, ComputeDataType>(
        xTensor, wTensor, yGpu, strides, dilations, prePadding, postPadding);

    assertAllClose(yCpu, yGpu, tolerance);
}

// --- Forward convolution helper overloads ---
// fillRange controls the magnitude of random fill values [-fillRange, +fillRange].
// For small output types (e.g. fp8), reduce fillRange to prevent overflow:
// each output element accumulates cPerGroup * Kh * Kw products, so
// max output ≈ numMACs * fillRange². Keep numMACs * fillRange² < type max.

// Asymmetric padding with optional layout.
// When layout is non-null, input/output tensors use channel-last strides (e.g. NHWC, NDHWC)
// generated via Tensor(dims, layout). Weights always use default packed (KCRS) strides.
// When layout is null, all tensors use default packed strides (NCHW/NCDHW).
template <typename DataType, typename ComputeDataType = double>
void runGpuVsCpuConvFwd(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        const std::vector<int64_t>& prePadding,
                        const std::vector<int64_t>& postPadding,
                        float tolerance,
                        const TensorLayout* layout = nullptr,
                        float fillRange = 1.0f)
{
    auto xTensor = layout != nullptr ? Tensor<DataType>(xDims, *layout) : Tensor<DataType>(xDims);
    auto wTensor = Tensor<DataType>(wDims);
    auto yCpu = layout != nullptr ? Tensor<DataType>(yDims, *layout) : Tensor<DataType>(yDims);
    auto yGpu = layout != nullptr ? Tensor<DataType>(yDims, *layout) : Tensor<DataType>(yDims);

    compareGpuVsCpuConvFwd<DataType, DataType, DataType, ComputeDataType>(xTensor,
                                                                          wTensor,
                                                                          yCpu,
                                                                          yGpu,
                                                                          strides,
                                                                          dilations,
                                                                          prePadding,
                                                                          postPadding,
                                                                          tolerance,
                                                                          fillRange);
}

// ============================================================================
// ConvFwdShapeCase — shape parameters for parameterized convolution tests
// ============================================================================

struct ConvFwdShapeCase
{
    std::vector<int64_t> xDims;
    std::vector<int64_t> wDims;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    std::vector<int64_t> padding;
    int64_t groups = 1;
    std::string tag;

    // When non-null, input/output tensors use this channel-last layout (NHWC/NDHWC).
    // Weights always use default packed (KCRS) strides regardless.
    // When null, all tensors use default packed strides (NCHW/NCDHW).
    const TensorLayout* layout = nullptr;

    std::vector<int64_t> computeOutputDims() const
    {
        auto numSpatialDims = xDims.size() - 2;
        std::vector<int64_t> yDims = {xDims[0], wDims[0]};
        for(size_t i = 0; i < numSpatialDims; ++i)
        {
            auto outputSize
                = (xDims[2 + i] + 2 * padding[i] - dilations[i] * (wDims[2 + i] - 1) - 1)
                      / strides[i]
                  + 1;
            yDims.push_back(outputSize);
        }
        return yDims;
    }

    friend std::ostream& operator<<(std::ostream& os, const ConvFwdShapeCase& tc)
    {
        return os << tc.tag;
    }
};

// Returns copies of the given cases with channel-last layout set.
// Uses NHWC for 4D (2D conv) and NDHWC for 5D (3D conv).
// Points to the static TensorLayout constants which have program lifetime.
std::vector<ConvFwdShapeCase> withChannelLastLayout(std::vector<ConvFwdShapeCase> cases)
{
    for(auto& tc : cases)
    {
        tc.layout = (tc.xDims.size() == 5) ? &TensorLayout::NDHWC : &TensorLayout::NHWC;
    }
    return cases;
}

// ============================================================================
// Shape Catalog — centralized convolution shapes, categorized by size
// ============================================================================

// Small 2D shapes: output < 1K elements, suitable for all types
std::vector<ConvFwdShapeCase> getSmall2dConvCases()
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
std::vector<ConvFwdShapeCase> getMedium2dConvCases()
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
std::vector<ConvFwdShapeCase> getLarge2dConvCases()
{
    return {
        // ResNeXt-32x4d high-resolution block
        {{16, 128, 56, 56}, {256, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXt32x4dHiRes"},
        // ResNeXt deep 32-group (512→1024 channels)
        {{16, 512, 14, 14}, {1024, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXtDeep32Group"},
        // ResNeXt stride-2 downsample (256→512)
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
std::vector<ConvFwdShapeCase> getSmall1dConvCases()
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

// Small 3D shapes: basic 3D convolution tests
std::vector<ConvFwdShapeCase> getSmall3dConvCases()
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
std::vector<ConvFwdShapeCase> getMedium3dConvCases()
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
// TestConvolutionValidation — direct tests for validateConvolutionParams
// (no GPU needed, tests the standalone validation function)
// ============================================================================

TEST(TestConvolutionValidation, AcceptsValidParams)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_NO_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
        x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}));
}

TEST(TestConvolutionValidation, ThrowsOnWeightDimMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnOutputDimMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnStridesSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnDilationsSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnPrePaddingSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnPostPaddingSizeMismatch)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnZeroStride)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{0, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnNegativeDilation)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, -1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnNegativePrePadding)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{-1, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnNegativePostPadding)
{
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, -1}),
                 std::invalid_argument);
}

TEST(TestConvolutionValidation, ThrowsOnOutputDimValueMismatch)
{
    // Input [1,1,4,4], kernel [1,1,3,3], no padding, stride 1 → expected output [1,1,2,2]
    // Provide wrong output dims [1,1,3,3]
    const Tensor<float> x({1, 1, 4, 4});
    const Tensor<float> w({1, 1, 3, 3});
    const Tensor<float> y({1, 1, 3, 3});

    EXPECT_THROW(hipdnn_test_sdk::utilities::validateConvolutionParams(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

// ============================================================================
// TestGpuConvFwdRefValidation — validateInput throw paths (via GpuFpReferenceConvolution)
// ============================================================================

TEST(TestGpuConvFwdRefValidation, ThrowsOnInvalidDimCount)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({8, 8});
    Tensor<float> w({8, 8});
    Tensor<float> y({8, 8});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1}, Vec{1}, Vec{0}, Vec{0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnWeightDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnOutputDimMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnStridesSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnDilationsSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnPrePaddingSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(
        GpuFpReferenceConvolution::fprop<float>(x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0}, Vec{0, 0}),
        std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnPostPaddingSizeMismatch)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnZeroStride)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{0, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativeDilation)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, -1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativePrePadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{-1, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnNegativePostPadding)
{
    SKIP_IF_NO_DEVICES();
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, -1}),
                 std::invalid_argument);
}

TEST(TestGpuConvFwdRefValidation, ThrowsOnOutputDimValueMismatch)
{
    SKIP_IF_NO_DEVICES();
    // Input [1,1,4,4], kernel [1,1,3,3], no padding, stride 1 → expected output [1,1,2,2]
    // Provide wrong output dims [1,1,3,3]
    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> w({1, 1, 3, 3});
    Tensor<float> y({1, 1, 3, 3});

    EXPECT_THROW(GpuFpReferenceConvolution::fprop<float>(
                     x, w, y, Vec{1, 1}, Vec{1, 1}, Vec{0, 0}, Vec{0, 0}),
                 std::invalid_argument);
}

// ============================================================================
// Standalone tests (TEST, not TEST_P)
// These test features with unique verification logic that doesn't fit the
// shape-catalog pattern (fill → run GPU + CPU → compare). Each suite exercises
// a different code path: asymmetric padding, alpha/beta scaling, non-packed
// strides, integer types, TF32 truncation, mixed input/weight types, and timing.
// All use default packed strides (NCHW). The features tested here (scaling,
// truncation, type conversion, padding) are layout-independent — NHWC/NDHWC
// correctness is covered by the parameterized shape catalog below.
// ============================================================================

// ============================================================================
// TestGpuConvFwdRefAsymPad — asymmetric (pre != post) padding tests
// ============================================================================

TEST(TestGpuConvFwdRefAsymPadFp32, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<float>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 1e-5f);
}

TEST(TestGpuConvFwdRefAsymPadFp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<half>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 5e-2f);
}

TEST(TestGpuConvFwdRefAsymPadBfp16, MatchesCpuRef)
{
    SKIP_IF_NO_DEVICES();
    runGpuVsCpuConvFwd<bfloat16>(
        {1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 2}, {1, 1}, {1, 1}, {1, 0}, {0, 1}, 0.1f);
}

// ============================================================================
// TestGpuConvFwdRefAlphaBeta — alpha/beta scaling tests
// ============================================================================

TEST(TestGpuConvFwdRefAlphaBeta, AlphaOnly)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yRef({1, 1, 2, 2});
    Tensor<float> yScaled({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Compute with alpha=1.0
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yRef, {1, 1}, {1, 1}, {0, 0});

    // Compute with alpha=2.0
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yScaled, {1, 1}, {1, 1}, {0, 0}, 2.0);

    const auto* refData = yRef.memory().hostData();
    const auto* scaledData = yScaled.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(scaledData[i], 2.0f * refData[i], 1e-5f) << "Alpha scaling failed at " << i;
    }
}

TEST(TestGpuConvFwdRefAlphaBeta, BetaAccumulate)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yTensor({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    yTensor.fillWithValue(1.0f);

    // Pre-fill y with 1.0, then compute with alpha=1.0, beta=1.0
    // Result should be conv(x,w) + 1.0
    Tensor<float> yNoAccum({1, 1, 2, 2});
    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yNoAccum, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::fprop<float>(
        xTensor, wTensor, yTensor, {1, 1}, {1, 1}, {0, 0}, 1.0, 1.0);

    const auto* noAccumData = yNoAccum.memory().hostData();
    const auto* accumData = yTensor.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(accumData[i], noAccumData[i] + 1.0f, 1e-5f)
            << "Beta accumulation failed at " << i;
    }
}

TEST(TestGpuConvFwdRefAlphaBeta, BetaZeroSkipsRead)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yBetaZero({1, 1, 2, 2});
    Tensor<float> yDefault({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Pre-fill with garbage — should be ignored when beta=0
    yBetaZero.fillWithValue(999.0f);

    GpuFpReferenceConvolution::fprop<float>(xTensor, wTensor, yDefault, {1, 1}, {1, 1}, {0, 0});
    GpuFpReferenceConvolution::fprop<float>(
        xTensor, wTensor, yBetaZero, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0);

    const auto* defaultData = yDefault.memory().hostData();
    const auto* betaZeroData = yBetaZero.memory().hostData();

    for(size_t i = 0; i < 4; ++i)
    {
        ASSERT_NEAR(betaZeroData[i], defaultData[i], 1e-5f)
            << "Beta=0 should ignore pre-filled data at " << i;
    }
}

// ============================================================================
// TestGpuConvFwdRefStridedFp32 — non-packed (strided) tensor tests
// Verifies stride-based indexing with memory gaps between elements.
// ============================================================================

TEST(TestGpuConvFwdRefStridedFp32, NonPackedInput)
{
    SKIP_IF_NO_DEVICES();

    // x: [1, 2, 4, 4] with inter-channel gap (stride[1]=32 vs packed 16)
    const std::vector<int64_t> xDims = {1, 2, 4, 4};
    const std::vector<int64_t> xStrides = {64, 32, 4, 1}; // packed would be {32, 16, 4, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> wTensor({1, 2, 3, 3});
    Tensor<float> yCpu({1, 1, 2, 2});
    Tensor<float> yGpu({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

TEST(TestGpuConvFwdRefStridedFp32, NonPackedOutput)
{
    SKIP_IF_NO_DEVICES();

    // y: [1, 1, 4, 4] with inter-row gap (stride[2]=8 vs packed 4)
    const std::vector<int64_t> yDims = {1, 1, 4, 4};
    const std::vector<int64_t> yStrides = {32, 32, 8, 1}; // packed would be {16, 16, 4, 1}

    Tensor<float> xTensor({1, 1, 6, 6});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yCpu(yDims, yStrides);
    Tensor<float> yGpu(yDims, yStrides);

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

TEST(TestGpuConvFwdRefStridedFp32, NonPackedInputAndOutput)
{
    SKIP_IF_NO_DEVICES();

    // Both x and y have non-packed strides with inter-row gaps
    const std::vector<int64_t> xDims = {1, 2, 4, 4};
    const std::vector<int64_t> xStrides = {64, 32, 6, 1}; // packed would be {32, 16, 4, 1}

    const std::vector<int64_t> yDims = {1, 1, 2, 2};
    const std::vector<int64_t> yStrides = {8, 8, 4, 1}; // packed would be {4, 4, 2, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> wTensor({1, 2, 3, 3});
    Tensor<float> yCpu(yDims, yStrides);
    Tensor<float> yGpu(yDims, yStrides);

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {0, 0});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

TEST(TestGpuConvFwdRefStridedFp32, NonPackedWithPadding)
{
    SKIP_IF_NO_DEVICES();

    // Non-packed input with padding to exercise both features together
    const std::vector<int64_t> xDims = {1, 2, 3, 3};
    const std::vector<int64_t> xStrides = {36, 18, 3, 1}; // packed would be {18, 9, 3, 1}

    Tensor<float> xTensor(xDims, xStrides);
    Tensor<float> wTensor({1, 2, 3, 3});
    Tensor<float> yCpu({1, 1, 3, 3});
    Tensor<float> yGpu({1, 1, 3, 3});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    CpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yCpu, {1, 1}, {1, 1}, {1, 1});

    GpuFpReferenceConvolution::fprop<float, float, float, double>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {1, 1});

    assertAllClose(yCpu, yGpu, 1e-5f);
}

// ============================================================================
// TestGpuConvFwdRefInt8 — int8 input with int32 or float output
// ============================================================================

TEST(TestGpuConvFwdRefInt8, Int8ToInt32)
{
    SKIP_IF_NO_DEVICES();

    Tensor<int8_t> xTensor({1, 1, 4, 4});
    Tensor<int8_t> wTensor({1, 1, 3, 3});
    Tensor<int32_t> yGpu({1, 1, 2, 2});

    // Fill with small values that won't overflow
    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<int8_t>(-3), static_cast<int8_t>(3), seed);
    wTensor.fillWithRandomValues(static_cast<int8_t>(-2), static_cast<int8_t>(2), seed + 1);

    GpuFpReferenceConvolution::fprop<int8_t, int8_t, int32_t, int32_t>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    // Verify manually: compute expected output
    const auto* x = xTensor.memory().hostData();
    const auto* w = wTensor.memory().hostData();
    const auto* y = yGpu.memory().hostData();

    // Each output element should be the sum of element-wise products of a 3x3 patch
    for(int ho = 0; ho < 2; ++ho)
    {
        for(int wo = 0; wo < 2; ++wo)
        {
            int32_t expected = 0;
            for(int kh = 0; kh < 3; ++kh)
            {
                for(int kw = 0; kw < 3; ++kw)
                {
                    expected += static_cast<int32_t>(x[(ho + kh) * 4 + (wo + kw)])
                                * static_cast<int32_t>(w[kh * 3 + kw]);
                }
            }
            ASSERT_EQ(y[ho * 2 + wo], expected)
                << "Int8->Int32 mismatch at (" << ho << "," << wo << ")";
        }
    }
}

TEST(TestGpuConvFwdRefInt8, Int8ToFloat)
{
    SKIP_IF_NO_DEVICES();

    Tensor<int8_t> xTensor({1, 1, 4, 4});
    Tensor<int8_t> wTensor({1, 1, 3, 3});
    Tensor<float> yGpu({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<int8_t>(-3), static_cast<int8_t>(3), seed);
    wTensor.fillWithRandomValues(static_cast<int8_t>(-2), static_cast<int8_t>(2), seed + 1);

    GpuFpReferenceConvolution::fprop<int8_t, int8_t, float, float>(
        xTensor, wTensor, yGpu, {1, 1}, {1, 1}, {0, 0});

    const auto* x = xTensor.memory().hostData();
    const auto* w = wTensor.memory().hostData();
    const auto* y = yGpu.memory().hostData();

    for(int ho = 0; ho < 2; ++ho)
    {
        for(int wo = 0; wo < 2; ++wo)
        {
            float expected = 0.0f;
            for(int kh = 0; kh < 3; ++kh)
            {
                for(int kw = 0; kw < 3; ++kw)
                {
                    expected += static_cast<float>(x[(ho + kh) * 4 + (wo + kw)])
                                * static_cast<float>(w[kh * 3 + kw]);
                }
            }
            ASSERT_NEAR(y[ho * 2 + wo], expected, 1e-5f)
                << "Int8->Float mismatch at (" << ho << "," << wo << ")";
        }
    }
}

// ============================================================================
// TestGpuConvFwdRefTf32 — TF32 truncation test
// ============================================================================

TEST(TestGpuConvFwdRefTf32, DiffersFromNonTf32)
{
    SKIP_IF_NO_DEVICES();

    // Use values with enough mantissa bits to show TF32 truncation
    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<float> yNoTf32({1, 1, 2, 2});
    Tensor<float> yTf32({1, 1, 2, 2});

    const unsigned int seed = 42;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Regular computation with float accumulation
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yNoTf32, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0, false);

    // TF32 computation
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yTf32, {1, 1}, {1, 1}, {0, 0}, 1.0, 0.0, true);

    const auto* noTf32Data = yNoTf32.memory().hostData();
    const auto* tf32Data = yTf32.memory().hostData();

    // TF32 results should be close but not identical to full-precision
    bool hasDifference = false;
    for(size_t i = 0; i < 4; ++i)
    {
        if(std::abs(noTf32Data[i] - tf32Data[i]) > 1e-10f)
        {
            hasDifference = true;
        }
        // But should still be close
        ASSERT_NEAR(noTf32Data[i], tf32Data[i], 0.1f)
            << "TF32 too far from full precision at " << i;
    }
    ASSERT_TRUE(hasDifference) << "TF32 should produce different results from full precision";
}

// ============================================================================
// TestGpuConvFwdRefMixedType — separate WEI_TYPE tests
// ============================================================================

TEST(TestGpuConvFwdRefMixedType, FloatInputHalfWeight)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> xTensor({1, 1, 4, 4});
    Tensor<half> wTensor({1, 1, 3, 3});
    Tensor<float> yCpu({1, 1, 2, 2});
    Tensor<float> yGpu({1, 1, 2, 2});

    compareGpuVsCpuConvFwd<float, half, float, double>(
        xTensor, wTensor, yCpu, yGpu, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 5e-2f, 1.0f);
}

TEST(TestGpuConvFwdRefMixedType, HalfInputFloatWeight)
{
    SKIP_IF_NO_DEVICES();

    Tensor<half> xTensor({1, 1, 4, 4});
    Tensor<float> wTensor({1, 1, 3, 3});
    Tensor<half> yCpu({1, 1, 2, 2});
    Tensor<half> yGpu({1, 1, 2, 2});

    compareGpuVsCpuConvFwd<half, float, half, double>(
        xTensor, wTensor, yCpu, yGpu, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 5e-2f, 1.0f);
}

// ============================================================================
// TestGpuConvFwdRefPerformance — timing comparisons
// ============================================================================

TEST(TestGpuConvFwdRefPerformance, MediumTensorTimingComparison)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> xDims = {2, 64, 14, 14};
    const std::vector<int64_t> wDims = {128, 64, 3, 3};
    const std::vector<int64_t> yDims = {2, 128, 12, 12};
    const std::vector<int64_t> strides = {1, 1};
    const std::vector<int64_t> dilations = {1, 1};
    const std::vector<int64_t> padding = {0, 0};

    Tensor<float> xTensor(xDims);
    Tensor<float> wTensor(wDims);
    Tensor<float> yCpu(yDims);
    Tensor<float> yGpu(yDims);

    const unsigned int seed = 123;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Warm-up run (includes HipRTC compilation on first call)
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);

    // Time CPU
    auto cpuStart = std::chrono::high_resolution_clock::now();
    CpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yCpu, strides, dilations, padding);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuUs = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    auto cpuMs = static_cast<double>(cpuUs) / 1000.0;

    // Time GPU (kernel already compiled from warm-up)
    hipEvent_t gpuStart = nullptr;
    hipEvent_t gpuStop = nullptr;
    static_cast<void>(hipEventCreate(&gpuStart));
    static_cast<void>(hipEventCreate(&gpuStop));

    static_cast<void>(hipEventRecord(gpuStart, nullptr));
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);
    static_cast<void>(hipEventRecord(gpuStop, nullptr));
    static_cast<void>(hipEventSynchronize(gpuStop));

    float gpuMs = 0.0f;
    static_cast<void>(hipEventElapsedTime(&gpuMs, gpuStart, gpuStop));

    static_cast<void>(hipEventDestroy(gpuStart));
    static_cast<void>(hipEventDestroy(gpuStop));

    RecordProperty("cpu_ms", std::to_string(cpuMs));
    RecordProperty("gpu_ms", std::to_string(static_cast<double>(gpuMs)));

    assertAllClose(yCpu, yGpu, 1e-3f);
}

TEST(TestGpuConvFwdRefPerformance, DISABLED_LargeTensorTimingComparison)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> xDims = {8, 128, 28, 28};
    const std::vector<int64_t> wDims = {256, 128, 3, 3};
    const std::vector<int64_t> yDims = {8, 256, 26, 26};
    const std::vector<int64_t> strides = {1, 1};
    const std::vector<int64_t> dilations = {1, 1};
    const std::vector<int64_t> padding = {0, 0};

    Tensor<float> xTensor(xDims);
    Tensor<float> wTensor(wDims);
    Tensor<float> yCpu(yDims);
    Tensor<float> yGpu(yDims);

    const unsigned int seed = 456;
    xTensor.fillWithRandomValues(-1.0f, 1.0f, seed);
    wTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);

    // Warm-up run
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);

    // Time CPU
    auto cpuStart = std::chrono::high_resolution_clock::now();
    CpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yCpu, strides, dilations, padding);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuUs = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    auto cpuMs = static_cast<double>(cpuUs) / 1000.0;

    // Time GPU
    hipEvent_t gpuStart = nullptr;
    hipEvent_t gpuStop = nullptr;
    static_cast<void>(hipEventCreate(&gpuStart));
    static_cast<void>(hipEventCreate(&gpuStop));

    static_cast<void>(hipEventRecord(gpuStart, nullptr));
    GpuFpReferenceConvolution::fprop<float, float, float, float>(
        xTensor, wTensor, yGpu, strides, dilations, padding);
    static_cast<void>(hipEventRecord(gpuStop, nullptr));
    static_cast<void>(hipEventSynchronize(gpuStop));

    float gpuMs = 0.0f;
    static_cast<void>(hipEventElapsedTime(&gpuMs, gpuStart, gpuStop));

    static_cast<void>(hipEventDestroy(gpuStart));
    static_cast<void>(hipEventDestroy(gpuStop));

    RecordProperty("cpu_ms", std::to_string(cpuMs));
    RecordProperty("gpu_ms", std::to_string(static_cast<double>(gpuMs)));

    assertAllClose(yCpu, yGpu, 1e-2f);
}

// ============================================================================
// TestGpuConvFwdRefShapes — parameterized shape coverage across types
// ============================================================================

template <typename DataType>
class ConvFwdShapeSuite : public ::testing::TestWithParam<ConvFwdShapeCase>
{
protected:
    static float tolerance(const ConvFwdShapeCase& tc)
    {
        constexpr double FILL_RANGE = 1.0;
        return hipdnn_test_sdk::utilities::conv::
            calculateConvFpropTolerance<DataType, DataType, double>(
                -FILL_RANGE, FILL_RANGE, -FILL_RANGE, FILL_RANGE, tc.wDims);
    }

    void runConvFwdShapeTest()
    {
        SKIP_IF_NO_DEVICES();
        const auto& tc = GetParam();
        auto yDims = tc.computeOutputDims();
        runGpuVsCpuConvFwd<DataType>(tc.xDims,
                                     tc.wDims,
                                     yDims,
                                     tc.strides,
                                     tc.dilations,
                                     tc.padding,
                                     tc.padding,
                                     tolerance(tc),
                                     tc.layout);
    }
};

using TestGpuConvFwdRefShapesFp32 = ConvFwdShapeSuite<float>;
using TestGpuConvFwdRefShapesFp16 = ConvFwdShapeSuite<half>;
using TestGpuConvFwdRefShapesBfp16 = ConvFwdShapeSuite<bfloat16>;

TEST_P(TestGpuConvFwdRefShapesFp32, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefShapesFp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}
TEST_P(TestGpuConvFwdRefShapesBfp16, MatchesCpuRef)
{
    this->runConvFwdShapeTest();
}

// ============================================================================
// Default layout (NCHW/NCDHW/NCW) instantiations — packed strides, no layout set.
// ============================================================================

// fp32 NCHW/NCDHW: all sizes (small + medium + large 2D, small + medium 3D, small 1D)
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Medium2d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Large2d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getLarge2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Medium3d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getMedium3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp32 NCW: 1D shapes
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16 NCHW/NCDHW/NCW: small + medium 2D, small 1D, small 3D
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Medium2d,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16 NCHW/NCDHW/NCW: small + medium 2D, small 1D, small 3D
INSTANTIATE_TEST_SUITE_P(Small2d,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Medium2d,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small1d,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(getSmall1dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Small3d,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(getSmall3dConvCases()),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// ============================================================================
// Channel-last (NHWC/NDHWC) instantiations — same suites, same catalog,
// but withChannelLastLayout() sets tc.layout so the fixture uses channel-last
// strides on input/output tensors. Weights always stay packed (KCRS).
// TensorLayout controls stride generation via generateStrides(dims, strideOrder);
// e.g. NHWC with dims {1,3,8,8} produces strides {192,1,24,3} (C=1 is innermost).
// ============================================================================

// fp32 NHWC/NDHWC: all sizes (small + medium + large 2D, small + medium 3D)
INSTANTIATE_TEST_SUITE_P(Nhwc2dSmall,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Nhwc2dMedium,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Nhwc2dLarge,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getLarge2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dSmall,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dMedium,
                         TestGpuConvFwdRefShapesFp32,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// fp16 NHWC/NDHWC: small + medium 2D, small 3D
INSTANTIATE_TEST_SUITE_P(Nhwc2dSmall,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Nhwc2dMedium,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dSmall,
                         TestGpuConvFwdRefShapesFp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });

// bfp16 NHWC/NDHWC: small + medium 2D, small 3D
INSTANTIATE_TEST_SUITE_P(Nhwc2dSmall,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(DISABLED_Nhwc2dMedium,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getMedium2dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
INSTANTIATE_TEST_SUITE_P(Ndhwc3dSmall,
                         TestGpuConvFwdRefShapesBfp16,
                         ::testing::ValuesIn(withChannelLastLayout(getSmall3dConvCases())),
                         [](const ::testing::TestParamInfo<ConvFwdShapeCase>& info) {
                             return info.param.tag;
                         });
