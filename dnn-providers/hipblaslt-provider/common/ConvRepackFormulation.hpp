// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <optional>

#include "common/ConvGemmFormulation.hpp"

namespace hipblaslt_plugin
{

/// The most workspace the repack steps of one plan take by default. A column matrix that needs
/// more is built a slice of images at a time; a single image that needs more is a poor fit for
/// explicit GEMM and is declined. Measured with dnnb autotune on 134 graphs near the cap from a
/// real-model convolution corpus (gfx950, MI325X, gfx906): unsliced column matrices of several
/// GiB run the explicit-GEMM engines 0.81x (gfx950, MI325X) to 0.94x (gfx906) as fast, down to
/// 0.08x at 9 GiB, and a 256 MiB cap 0.96x (MI325X).
inline constexpr size_t MAX_REPACK_WORKSPACE_BYTES = static_cast<size_t>(1) << 30;

/// The cap of the explicit-GEMM engines' further backward weights benchmarking candidate
/// (ConvGemmPlanBuilder): slicing backward weights costs an accumulating GEMM per slice (and an
/// fp32 accumulator for 16-bit w). Measured with dnnb autotune on the real-model corpus's backward
/// weights graphs whose explicit-GEMM plan is within 2x of the best engine, every plan at 2 GiB:
/// gfx906 explicit GEMM 1.19x geomean, best engine 3.05x on a 7x7 fp32 stem and 1.17x on a 1x1
/// fp16 layer; MI325X explicit GEMM 1.02x, but that stem 0.88x (it was best). 4 GiB measured no
/// better than 2 GiB.
inline constexpr size_t MAX_BACKWARD_WEIGHTS_REPACK_WORKSPACE_BYTES = static_cast<size_t>(2) << 30;

/// How a plan that has a choice lays out what it computes through: an IM2COL plan's column
/// matrix, and a 1x1 channels-first backward weights of several images (batched fp32 partials
/// per image, or x and dy staged channels-last for one GEMM).
enum class ColumnLayout
{
    /// The default: columns like y; for the 1x1 backward weights, whichever moves fewer bytes.
    FOLLOW_Y,
    /// Channels-first: columns like a channels-first y; the 1x1 backward weights as batched
    /// partials.
    CHANNELS_FIRST,
    /// Channels-last whatever y's layout; the 1x1 backward weights staged channels-last.
    CHANNELS_LAST,
    /// Channels-last columns, and a channels-first y of several images staged channels-last
    /// in the workspace, so one GEMM covers every image of a slice instead of a batch of
    /// per-image GEMMs (MIOpen's 1x1 stride-2 form); otherwise CHANNELS_LAST.
    STAGED_CHANNELS_LAST,
    /// As FOLLOW_Y, but a 16-bit backward weights over the cap slices its images into an fp32
    /// accumulator rather than running blocks of w's columns over every image (convGemmCalls).
    IMAGE_SLICES,
};

/**
 * @brief The IM2COL plan for @p problem within @p maxWorkspaceBytes of repack workspace, or
 *        nullopt when explicit GEMM is a poor fit for it or cannot read its operands.
 *
 * im2col writes the column matrix (kernels/ConvRepackArgs.h) to the workspace, which turns the
 * convolution into a ONE_BY_ONE one with C * filterSize channels over the output positions;
 * directGemmCalls formulates that. Backward data runs the same GEMM into an fp32 column matrix
 * and col2im sums it into dx. Columns follow the filter's own layout (KCRS or KRSC), so w is
 * read in place; a filter that is neither is declined. When the GEMM cannot read y (dy) as
 * given -- channels-first backward weights over several images, say -- y is staged in the
 * workspace channels-last by a strided copy.
 *
 * When the column matrix of every image does not fit, the images are processed a slice at a
 * time, each slice a stage reusing the same workspace. Backward weights sums its slices with
 * accumulating GEMMs: into w for fp32, otherwise into an fp32 accumulator in the workspace that
 * a final stage rounds into w once. Declined when one image does not fit.
 */
std::optional<ConvGemmPlanCalls> im2colGemmCalls(const ConvProblem& problem,
                                                 size_t maxWorkspaceBytes
                                                 = MAX_REPACK_WORKSPACE_BYTES,
                                                 ColumnLayout layout = ColumnLayout::FOLLOW_Y);

/// The plan for @p problem: a direct row (directGemmCalls) when one applies; for a 1x1
/// channels-first backward weights of several images, batched partials or a channels-last
/// staging (@p layout); for a 16-bit backward weights whose column matrix over every image
/// exceeds the cap, blocks of w's columns, each one GEMM over every image writing its block
/// once (16-bit output; not with @p layout IMAGE_SLICES); IM2COL otherwise; or nullopt when none
/// does. A long reduction into a
/// small backward-weights output -- and with @p narrowPartials, a forward or backward data one
/// straight into a 16-bit y or dx -- runs as chunks with fp32 partials summed in order and
/// rounded once. @p splitReduction: such
/// a reduction whose GEMMs leave too few output tiles to fill a GPU (fewer than
/// SPLIT_TARGET_TILES) runs its reduction as enough fp32 partials to fill one, summed in order
/// and rounded once -- a choice for the benchmarking builder, which times it against the default.
std::optional<ConvGemmPlanCalls> convGemmCalls(const ConvProblem& problem,
                                               size_t maxWorkspaceBytes
                                               = MAX_REPACK_WORKSPACE_BYTES,
                                               ColumnLayout layout = ColumnLayout::FOLLOW_Y,
                                               bool splitReduction = false,
                                               bool narrowPartials = false);

/// The output tiles (128 x 128) a split reduction aims to fill a GPU with.
inline constexpr int64_t SPLIT_TARGET_TILES = 512;

/**
 * @brief The reduced-precision-reduction plan for @p problem, or nullopt when it has none: the
 *        plan of the ROCBLAS_CONV_RPR and HIPBLASLT_CONV_RPR engines (REDUCED_PRECISION_REDUCTION).
 *
 * A 1x1, unit-stride, unpadded fp16 backward weights of several channels-first images, which no
 * single GEMM reads, as one GEMM per image into dw, each after the first adding to it (beta 1):
 * MIOpen's GemmWrw1x1_stride1. Each image's sum is exact in fp32, but dw is rounded to fp16 once
 * per image (cuDNN's REDUCED_PRECISION_REDUCTION) where convGemmCalls rounds once. No workspace.
 * bf16 has no reduced form: cuDNN serves it with FLOAT compute only.
 */
std::optional<ConvGemmPlanCalls> reducedPrecisionGemmCalls(const ConvProblem& problem);

} // namespace hipblaslt_plugin
