// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
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

/// One step of the chain, applied to the running fp32 value v.
enum class OpCode
{
    ADD_OPERAND, ///< v + operand (bias)
    ADD_SCALAR, ///< v + value
    MUL_OPERAND, ///< v * operand (scale)
    MUL_SCALAR, ///< v * value
    RELU, ///< relu / clipped relu / clamp / leaky relu: values lower, upper, slope
    GELU, ///< exact (erf)
    GELU_TANH, ///< tanh approximation
    SWISH, ///< v * sigmoid(beta v): value beta (silu at 1)
    SIGMOID,
    TANH,
    ELU, ///< value alpha
    ABS,
    SOFTPLUS, ///< log(1 + e^(beta v)) / beta, v past 20 / beta: value beta
};

/// A tensor operand the epilogue reads: a per-channel vector ([1, K, 1, ...]) or a one-element
/// device tensor, indexed by the channel or by 0.
struct Operand
{
    int64_t uid = 0;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    bool perChannel = true;
};

/// A value slot of ConvEpiArgs::scalar: fixed at plan build (an attribute, a constant scalar
/// tensor, a runtime scalar with a default), or read at execute (a runtime pass-by-value scalar
/// tensor the caller supplies).
struct Scalar
{
    float value = 0.0F;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> runtime;
};

struct Op
{
    OpCode code = OpCode::ADD_OPERAND;
    /// The operand index (ADD_OPERAND, MUL_OPERAND) or the first scalar slot.
    int slot = 0;
};

/// The epilogue of a forward convolution graph. Empty for a single-node graph.
struct ConvEpilogue
{
    std::vector<Op> ops;
    std::vector<Operand> operands;
    std::vector<Scalar> scalars;
    /// The convolution node's own output (virtual in a fused graph).
    int64_t convOutputUid = 0;
    /// The graph's written output: the last pointwise node's.
    int64_t outputUid = 0;

    bool empty() const
    {
        return ops.empty();
    }
};

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
