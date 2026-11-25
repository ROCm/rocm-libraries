// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvRdnaCommon.hpp"

/**
 * @file ConvRdnaFusedEpilogue.hpp
 * @brief The epilogue the fused RDNA forward convolution engines apply to the convolution result
 *        before the one store: an optional per-output-channel bias ADD and a forward activation,
 *        the graph `conv_fprop -> [pointwise ADD(bias) ->] pointwise activation`. Matching that
 *        graph, the activation the kernels take, and the epilogue half of a saved launch.
 *
 * The kernels add the bias and apply the activation in fp32, with the formulas and unset-attribute
 * defaults of hipDNN's pointwise reference, and round once.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna
{

/// The activations the kernels apply, numbered as the kernels' ACT_* values.
enum class ActivationMode : int32_t
{
    RELU = 0,
    SIGMOID = 1,
    TANH = 2,
    ELU = 3,
    GELU = 4,
    GELU_APPROX_TANH = 5,
    SWISH = 6,
    SOFTPLUS = 7,
    IDENTITY = 8,
};

/// An activation and its parameters as the kernels take them: RELU reads p0 as the lower clip,
/// p1 as the upper clip and p2 as the slope below the lower clip; ELU reads p0 as alpha;
/// SWISH and SOFTPLUS read p0 as beta. The others read none.
struct Activation
{
    ActivationMode mode = ActivationMode::IDENTITY;
    float p0 = 0.0F;
    float p1 = 0.0F;
    float p2 = 0.0F;
};

/// A matched `conv_fprop -> [ADD(bias) ->] activation` graph: the conv node, the tensors the
/// kernel reads and writes, the convolution as the kernel runs it (with the activation output as
/// its stored output), and the activation.
struct ConvEpilogueMatch
{
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes* conv = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* w = nullptr;
    /// The bias; null for a graph without one.
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* b = nullptr;
    /// The output of the activation: what the kernel stores.
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y = nullptr;
    std::optional<int64_t> bUid;
    int64_t yUid = 0;
    ConvProblem problem;
    Activation activation;
};

/// Matches `conv_fprop -> ADD(bias) -> activation` (the bias in either ADD operand) or
/// `conv_fprop -> activation`. Admits only what every fused engine can run: the convolution and
/// the activation computed in fp32; every intermediate virtual, of the output shape, and in fp32
/// or the data dtype; a bias of one value per output channel in the data dtype; an activation of
/// ActivationMode with finite parameters (and a nonzero softplus beta). The caller applies its own
/// rules for which convolutions its kernel serves.
std::optional<ConvEpilogueMatch>
    matchConvBiasActivation(const hipdnn_plugin_sdk::ingestor::MatchContext& context);

/// Adds the epilogue values of a saved launch: `b.uid` (only with a bias), `activation` (the
/// hipDNN pointwise mode name, e.g. `RELU_FWD`) and `activation.p0` to `activation.p2`.
void addEpilogueLaunchValues(hipdnn_plugin_sdk::ingestor::MetadataValues& values,
                             std::optional<int64_t> bUid,
                             const Activation& activation);

/// The names addEpilogueLaunchValues writes.
std::vector<std::string_view> epilogueLaunchValueNames();

/// The epilogue half of a saved launch.
struct EpilogueLaunchInputs
{
    std::optional<int64_t> bUid;
    Activation activation;
};

/// Reads the values addEpilogueLaunchValues wrote; `b.uid` is optional.
///
/// @throws HipdnnPluginException with a load refusal when a value is of another type, the
///         activation is not one the kernels apply, or a parameter is not a finite float (or a
///         softplus beta is 0).
EpilogueLaunchInputs
    readEpilogueLaunchValues(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
