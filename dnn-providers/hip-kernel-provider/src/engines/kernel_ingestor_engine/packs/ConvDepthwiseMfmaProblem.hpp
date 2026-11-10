// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "engines/kernel_ingestor_engine/descriptors/conv/depthwise_mfma/kernels/ConvDepthwiseMfmaArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseValuProblem.hpp"

/**
 * @file ConvDepthwiseMfmaProblem.hpp
 * @brief The host half of the matrix-core depthwise engines (CONV_DEPTHWISE_MFMA_FWD/BWD): the
 *        config a kernel's metadata declares, whether it serves a problem on a device, and its
 *        launch.
 *
 * The problem is the VALU depthwise engines' (parseDepthwiseValuProblem): forward, and unit-stride
 * backward data as the forward convolution of dy with the filter reflected. These kernels serve
 * its 2-D unit-dilation 16-bit form, channels-last or channels-first, at stride 1 or (forward) 2. Kept apart from the dispatch code
 * so every gate can be tested without a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_mfma
{

/// One kernel config: the KMD fields minus dtype and direction.
struct MfmaConfig
{
    /// "NHWC" (the channel-diagonal core) or "NCHW" (the channel-pair core).
    std::string layout;
    /// The MFMA's K: 32 (v_mfma_f32_16x16x32, gfx950) or 16 (v_mfma_f32_16x16x16, gfx908+).
    int64_t atom = 0;
    /// Compile-time filter extent.
    int64_t kh = 0;
    int64_t kw = 0;
    /// The stride along both axes: 1, or 2 (forward only).
    int64_t stride = 0;
    /// Waves per block: 8 channels each (NHWC) or a channel pair each (NCHW).
    int64_t waves = 0;
    /// Input rows in flight.
    int64_t depth = 0;
    /// Images sharing the MFMA's output columns (32 NHWC, 128 NCHW): 1, 2, 4 (or 8, NCHW).
    int64_t fold = 0;
    /// Output rows a block streams.
    int64_t rows = 0;
};

/// The launch of a config on a problem.
struct MfmaLaunch
{
    int64_t block = 0;
    int64_t grid = 0;
    /// The kernel's argument block, pointers left null.
    ConvDepthwiseMfmaArgs args{};
    /// The filter is read reflected (CONV_DWM_FLIP).
    bool flip = false;
};

/// LDS bytes a config's block claims: NHWC's two input and two output rows; NCHW none.
int64_t mfmaLdsBytes(const MfmaConfig& config);

/**
 * @brief Whether a device @p gcnArchName (features intact) runs @p atom's MFMA in the element
 *        type (@p bf16 or fp16) for @p layout's core: 32 on gfx950, and on gfx94x as the 2:4-sparse
 *        MFMA for NHWC; 16 on gfx908 (fp16 only), gfx90a, gfx94x, gfx950.
 */
bool mfmaRunsOn(int64_t atom, std::string_view gcnArchName, bool bf16, std::string_view layout);

/**
 * @brief The launch of @p config on @p problem, or nullopt when the config does not compute it
 *        correctly or cannot launch it.
 *
 * Reads and gates every field of @p config: the layout, the atom, the filter extent against the
 * problem's, the waves against the channels, the depth, the fold, the rows, the LDS, and that
 * the folded images' bytes stay within the 32-bit buffer offsets. The problem must be 2-D,
 * channels-last, unit dilation, with the config's stride on both axes (backward data: its
 * unit-stride forward form).
 */
std::optional<MfmaLaunch> mfmaLaunchFor(const conv_depthwise_valu::DepthwiseValuProblem& problem,
                                        const MfmaConfig& config);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_mfma
