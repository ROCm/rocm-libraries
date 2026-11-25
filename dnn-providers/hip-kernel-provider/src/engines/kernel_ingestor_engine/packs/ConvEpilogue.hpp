// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ConvEpilogueMatch.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "compilation/KernelCompileOptions.hpp"
#include "engines/kernel_ingestor_engine/descriptors/conv/igemm/kernels/ConvEpilogueArgs.h"

/**
 * @file ConvEpilogue.hpp
 * @brief The pointwise epilogue a forward convolution graph fuses after its conv node: parsed
 *        from the graph, generated into kernel code, and marshalled at launch.
 *
 * A fused graph is the convolution node followed by a chain of pointwise nodes, each taking the
 * previous one's result, the last one writing the graph's single non-virtual output. The chain
 * runs in fp32 on the convolution's accumulator and the kernel rounds once to the output type.
 *
 * The generated code depends on the pattern only: the operations, the operand dtypes and their
 * broadcast kind. Operand pointers and every value (clip bounds, slope, alpha, beta, scalar
 * operands) are runtime arguments (ConvEpiArgs), so one code object serves every graph of a
 * pattern, whatever its values or shapes.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_epilogue
{

using hipdnn_plugin_sdk::conv_epilogue::ConvEpilogue;
using hipdnn_plugin_sdk::conv_epilogue::Op;
using hipdnn_plugin_sdk::conv_epilogue::OpCode;
using hipdnn_plugin_sdk::conv_epilogue::Operand;
using hipdnn_plugin_sdk::conv_epilogue::Scalar;

static_assert(hipdnn_plugin_sdk::conv_epilogue::MAX_OPERANDS == CONV_EPI_OPERANDS
                  && hipdnn_plugin_sdk::conv_epilogue::MAX_SCALARS == CONV_EPI_SCALARS,
              "the shared parser's bounds are the kernel argument block's");

/// The graph's pointwise chain after its convolution node, or nullopt when the graph has nodes
/// after the convolution that the generator cannot fuse. @p convOutputUid is the conv node's y and
/// @p outputDims its dims (n, k, spatial...); every tensor of the chain has them, and per-channel
/// operands broadcast along k. A single-node graph gives an empty epilogue with outputUid the
/// conv's y.
///
/// Gates every field: the node kinds and order, each pointwise mode and its attributes (relu
/// clips and slope, swish beta, elu alpha, softplus beta, axis, in_2), each node's compute type,
/// and for every tensor its dims (the broadcast kind), strides, data type, virtual flag,
/// pass-by-value and ragged fields and alignment.
std::optional<ConvEpilogue>
    parseConvEpilogue(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                      int64_t convOutputUid,
                      const std::vector<int64_t>& outputDims);

/// The canonical text of the epilogue's pattern (empty for none): what the generated code
/// depends on, and nothing else.
std::string patternOf(const ConvEpilogue& epilogue);

/// Adds the epilogue's generated code to @p options: nothing for an empty epilogue, so an
/// unfused build's options (and code object) are exactly those of a build without fusion.
void addEpilogueDefines(compilation::KernelCompileOptions& options, const ConvEpilogue& epilogue);

/// Removes what addEpilogueDefines added, for kernels of the same build that take no epilogue
/// (staging copies, transposes).
void removeEpilogueDefines(compilation::KernelCompileOptions& options);

/// The kernel argument a fused launch passes: operand pointers from @p deviceBuffers, values
/// resolved (runtime scalars read now). @p chanStride and @p chans describe the flat channel walk
/// of a pass that needs it (ConvEpiArgs).
ConvEpiArgs epilogueArgs(const ConvEpilogue& epilogue,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         long long chanStride = 1,
                         int chans = 1);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_epilogue

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
