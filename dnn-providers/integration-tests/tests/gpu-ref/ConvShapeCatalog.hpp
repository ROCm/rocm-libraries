// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ConvShapeCase.hpp"

#include <stdexcept>

// ============================================================================
// Shape Catalog — centralized convolution shapes shared across all directions
// (forward, dgrad, wgrad). Adding a new shape = editing this one file.
// The existing INSTANTIATE_TEST_SUITE_P calls pick it up automatically.
//
// Function family         → Tier            → Purpose
// ──────────────────────────────────────────────────────────────────────
// getSmall*ConvCases()    → Smoke           → Minimal shapes, fast
// getMedium*ConvCases()   → Standard        → Moderate shapes, PR-level
// getLargeEdge*ConvCases()→ Comprehensive   → Corner cases (odd channels,
//                                              asymmetric filters, prime K)
// getLargeStress*()       → Full            → Real-workload shapes (ResNeXt,
//                                              DeepSpeech, large stem)
//
// Organization: Small (1D, 2D, 3D) → Medium (1D, 2D, 3D) → Large (1D, 2D, 3D)
//
// Medium 1D shapes intentionally differ for dgrad — use getMedium1dDgradCases()
// for dgrad and getMedium1dConvCases() for fwd/wgrad.
// ============================================================================

namespace gpu_conv_ref_test
{

// Returns copies of the given cases with channel-last layout set.
// 3D (NLC) for 1D conv, 4D (NHWC) for 2D conv, 5D (NDHWC) for 3D conv.
// Points to the static TensorLayout constants which have program lifetime.
inline std::vector<ConvShapeCase> withChannelLastLayout(std::vector<ConvShapeCase> cases)
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
        else if(tc.xDims.size() == 3)
        {
            tc.layout = &TensorLayout::NLC;
        }
        else
        {
            throw std::invalid_argument("Unsupported tensor rank for channel-last layout: "
                                        + std::to_string(tc.xDims.size()));
        }
    }
    return cases;
}

// ============================================================================
// Small shapes — fast binary (CI gate)
// ============================================================================

// Small 1D shapes: basic NCL convolution tests [fwd, dgrad, wgrad]
inline std::vector<ConvShapeCase> getSmall1dConvCases()
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

// Small 2D shapes: output < 1K elements, suitable for all types [fwd, dgrad, wgrad]
inline std::vector<ConvShapeCase> getSmall2dConvCases()
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
        // 5x5 kernel with padding (larger receptive field)
        {{1, 2, 10, 10}, {4, 2, 5, 5}, {1, 1}, {1, 1}, {2, 2}, 1, "Kernel5x5"},
        // Non-square spatial dimensions
        {{1, 2, 6, 10}, {4, 2, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 1, "NonSquare6x10"},
        // Minimum output: single element (3x3 input, 3x3 kernel)
        {{1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 1, "SingleElement"},
    };
}

// Small 3D shapes: basic 3D convolution tests [fwd, dgrad, wgrad]
inline std::vector<ConvShapeCase> getSmall3dConvCases()
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
        // 3D depthwise (groups == input channels)
        {{1, 3, 4, 4, 4}, {3, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 3, "Depthwise3d"},
        // 3D pointwise (1x1x1 kernel — channel mixing only)
        {{1, 8, 4, 4, 4}, {16, 8, 1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "Pointwise3d"},
    };
}

// ============================================================================
// Medium shapes — Standard tier (PR gate)
// ============================================================================

// Medium 1D shapes [fwd, wgrad] — dgrad uses getMedium1dDgradCases() instead
inline std::vector<ConvShapeCase> getMedium1dConvCases()
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

// Medium 1D shapes [dgrad] — intentionally different from fwd/wgrad
inline std::vector<ConvShapeCase> getMedium1dDgradCases()
{
    return {
        {{8, 64, 128}, {128, 64, 3}, {1}, {1}, {1}, 1, "WaveNet64Ch"},
        {{4, 32, 256}, {32, 32, 5}, {1}, {1}, {2}, 1, "Kernel5Pad2"},
        {{8, 128, 64}, {128, 16, 3}, {1}, {1}, {1}, 8, "Grouped8x1d"},
        {{4, 16, 512}, {16, 16, 7}, {2}, {1}, {3}, 1, "Stride2Kernel7"},
        {{8, 32, 128}, {32, 1, 3}, {1}, {1}, {1}, 32, "Depthwise32x1d"},
        {{4, 64, 64}, {128, 64, 1}, {1}, {1}, {0}, 1, "Pointwise64Ch"},
    };
}

// Medium 2D shapes: ResNet/ResNeXt/Inception-like, suitable for fp32 + fp16 [fwd, dgrad, wgrad]
inline std::vector<ConvShapeCase> getMedium2dConvCases()
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
        // 4-group convolution (C=4, K=16, C/G=1 per group, but K/G=4 output channels per group)
        {{4, 4, 48, 48}, {16, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 4, "Grouped4Chan"},
        // Odd channel count grouped (7 groups)
        {{8, 7, 14, 14}, {63, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 7, "OddChanGrouped7"},
        // Dilation=2 at medium scale
        {{4, 32, 28, 28}, {32, 32, 3, 3}, {1, 1}, {2, 2}, {2, 2}, 1, "Dilation2MedScale"},
    };
}

// Medium 3D shapes: larger 3D convolutions [fwd, dgrad, wgrad]
inline std::vector<ConvShapeCase> getMedium3dConvCases()
{
    return {
        // Standard 3D with 16 input channels and padding
        {{2, 16, 8, 8, 8}, {32, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "Standard16Ch3d"},
        // Non-cube spatial dimensions (4x14x14)
        {{1, 16, 4, 14, 14}, {16, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "NonCube3d"},
        // Large 5x5x5 kernel
        {{2, 16, 8, 8, 8}, {32, 16, 5, 5, 5}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "Kernel5x5x5"},
        // 3D depthwise at medium scale (MobileNet pattern)
        {{4, 16, 8, 8, 8}, {16, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 16, "Depthwise16x3d"},
        // 3D pointwise channel expansion (1x1x1 bottleneck)
        {{4, 16, 8, 8, 8},
         {32, 16, 1, 1, 1},
         {1, 1, 1},
         {1, 1, 1},
         {0, 0, 0},
         1,
         "Pointwise16to32x3d"},
        // 8-group 3D — from MIOpen grouped conv3d (N=128,C=32,K=32,28³,G=8)
        {{4, 32, 8, 8, 8}, {32, 4, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 8, "Grouped8x3d"},
        // Stride-2 with real channels (downsampling block)
        {{4, 16, 14, 14, 14},
         {32, 16, 3, 3, 3},
         {2, 2, 2},
         {1, 1, 1},
         {1, 1, 1},
         1,
         "Stride2Med3d"},
        // Dilation=2 at medium scale
        {{2, 16, 14, 14, 14},
         {32, 16, 3, 3, 3},
         {1, 1, 1},
         {2, 2, 2},
         {2, 2, 2},
         1,
         "Dilation2Med3d"},
    };
}

// ============================================================================
// Large shapes — split into edge cases (Comprehensive / nightly) and
// stress tests (Full / weekly) for tiered CI execution.
//
// Edge cases: odd channels, tiny output, prime K, asymmetric filters — shapes
//             that exercise corner cases at larger-than-medium scale.
// Stress tests: real-workload shapes (ResNeXt, DeepSpeech, WaveNet, medical
//               imaging) — genuinely expensive to run.
//
// ============================================================================

// Large 1D edge cases — moderate-cost shapes that exercise corner cases [fwd, dgrad, wgrad]
inline std::vector<ConvShapeCase> getLargeEdge1dConvCases()
{
    return {
        // 8-group on medium-long sequence
        {{16, 64, 512}, {64, 8, 5}, {1}, {1}, {2}, 8, "Grouped8Kernel5"},
        // Odd input channels — vector alignment edge case (SWDEV-502833 pattern)
        {{8, 5, 512}, {32, 5, 3}, {1}, {1}, {1}, 1, "OddChan5x1d"},
        // Kernel = spatial → output length 1
        {{4, 32, 7}, {64, 32, 7}, {1}, {1}, {0}, 1, "Output1x1d"},
        // Prime output channels — non-power-of-2 K
        {{8, 32, 256}, {127, 32, 5}, {1}, {1}, {2}, 1, "PrimeK127x1d"},
    };
}

// Large 1D stress tests — heavy real-workload shapes [fwd, dgrad, wgrad]
inline std::vector<ConvShapeCase> getLargeStress1dConvCases()
{
    return {
        // WaveNet-style high-channel long sequence
        {{16, 128, 1024}, {256, 128, 3}, {1}, {1}, {1}, 1, "WaveNetLarge"},
        // Large depthwise on long sequence
        {{16, 64, 2048}, {64, 1, 3}, {1}, {1}, {1}, 64, "Depthwise64Long"},
        // Stride-2 on long sequence with large kernel
        {{8, 32, 4096}, {64, 32, 7}, {2}, {1}, {3}, 1, "Stride2LargeKernel7"},
        // Large dilation on long sequence
        {{4, 16, 2048}, {32, 16, 3}, {1}, {4}, {4}, 1, "Dilation4Long1d"},
    };
}

// Large 2D edge cases — moderate-cost shapes that exercise corner cases [fwd, dgrad, wgrad]
inline std::vector<ConvShapeCase> getLargeEdge2dConvCases()
{
    return {
        // Odd input channels C=5 — vector alignment (SWDEV-502833)
        {{8, 5, 56, 56}, {32, 5, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, "OddChanIn5"},
        // Asymmetric filter 5x10 (MIOpen #540)
        {{4, 32, 79, 141}, {64, 32, 5, 10}, {2, 2}, {1, 1}, {0, 0}, 1, "AsymFilter5x10"},
        // Stride-2 on tiny spatial → output 1x1 (ho=wo=1)
        {{16, 128, 2, 2}, {256, 128, 1, 1}, {2, 2}, {1, 1}, {0, 0}, 1, "Output1x1Stride2"},
        // Prime output channels K=127
        {{8, 64, 14, 14}, {127, 64, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, "PrimeK127"},
        // High-channel 1x1 reduction (MIOpen #2012 / SWDEV-305815)
        {{8, 256, 14, 14}, {128, 256, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "HiChan1x1Reduce"},
        // Asymmetric stride (1,2) — CK edge case
        {{8, 32, 28, 56}, {64, 32, 3, 3}, {1, 2}, {1, 1}, {1, 1}, 1, "AsymStride1x2"},
        // Non-square spatial with 2-group (79x341)
        {{8, 32, 79, 341}, {32, 16, 5, 10}, {2, 2}, {1, 1}, {0, 0}, 2, "NonSquareGrouped2"},
    };
}

// Large 2D stress tests — heavy real-workload shapes [fwd, dgrad, wgrad]
inline std::vector<ConvShapeCase> getLargeStress2dConvCases()
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
    };
}

// Large 3D edge cases — moderate-cost shapes that exercise corner cases [fwd, dgrad, wgrad]
inline std::vector<ConvShapeCase> getLargeEdge3dConvCases()
{
    return {
        // 3D medical imaging style: 16ch, 32x32x32
        {{4, 16, 32, 32, 32},
         {32, 16, 3, 3, 3},
         {1, 1, 1},
         {1, 1, 1},
         {1, 1, 1},
         1,
         "MedImg32cube"},
        // 3D with stride-2 downsampling
        {{4, 32, 16, 16, 16},
         {64, 32, 3, 3, 3},
         {2, 2, 2},
         {1, 1, 1},
         {1, 1, 1},
         1,
         "Stride2x3dLarge"},
        // 3D 4-group on larger spatial
        {{2, 32, 16, 16, 16},
         {32, 8, 3, 3, 3},
         {1, 1, 1},
         {1, 1, 1},
         {1, 1, 1},
         4,
         "Grouped4x3dLarge"},
        // D=1 degeneracy — collapses depth to 2D-like behavior
        {{4, 16, 1, 32, 32},
         {32, 16, 1, 3, 3},
         {1, 1, 1},
         {1, 1, 1},
         {0, 1, 1},
         1,
         "Degenerate3dD1"},
    };
}

// Large 3D stress tests — heavy real-workload shapes [fwd, dgrad, wgrad]
inline std::vector<ConvShapeCase> getLargeStress3dConvCases()
{
    return {
        // 3D non-cube spatial (8x32x32)
        {{4, 16, 8, 32, 32},
         {32, 16, 3, 3, 3},
         {1, 1, 1},
         {1, 1, 1},
         {1, 1, 1},
         1,
         "NonCube3dLarge"},
        // 3D ResNet / C3D block — matches CK's standard 3D test shape
        // (CK: N=64,C=64,K=128,28³; scaled for reference impl)
        {{8, 64, 14, 14, 14}, {128, 64, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "ResNet3d"},
        // Large-spatial 5x5x5 kernel — from MIOpen conv3d_test (N=2,C=16,50³,K=32)
        {{2, 16, 50, 50, 50},
         {32, 16, 5, 5, 5},
         {1, 1, 1},
         {1, 1, 1},
         {2, 2, 2},
         1,
         "MedImg50cube5x5x5"},
        // Video-style temporal: small D, large HW — from MIOpen grouped conv3d
        {{8, 32, 4, 56, 56},
         {64, 32, 3, 3, 3},
         {1, 2, 2},
         {1, 1, 1},
         {1, 1, 1},
         1,
         "VideoTemporal"},
    };
}

} // namespace gpu_conv_ref_test
