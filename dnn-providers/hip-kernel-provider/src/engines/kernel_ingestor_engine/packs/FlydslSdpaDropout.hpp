// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaShared.hpp"

/**
 * @file FlydslSdpaDropout.hpp
 * @brief Dropout on the attention probabilities as every FlyDSL SDPA engine reads it from a
 *        graph: a probability with a seed and an offset tensor, or a mask with a scale, and the
 *        RNG dump output.
 *
 * One function, servableDropout(), decides what of this a graph asks for and whether it is
 * served, and both families' graph matchers call it, so the dropout rules live in one place and
 * a dropout change does not touch the matchers' other checks. Which objects apply it is each
 * kernel matcher's to say (the objects built with dropout, `has_dropout` 1).
 *
 * The mask is AOTriton's (>= 0.15, the release PyTorch-ROCm's flash path bundles from then on):
 * Philox4x32-10 keyed by the seed, one counter per 4 keys of a (batch, query head, row) at
 * offset1 + offset2 + ((b * Hq + h) * Sq + i) * cdiv(Skv, 4) + j / 4, a key kept iff its
 * random, read as int32, exceeds dropoutThreshold(p). Kept probabilities are scaled by
 * 1 / (1 - p); the softmax's normaliser and the LSE are the undropped ones.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa
{

/// The uid a dropout token holds when the graph has no such tensor.
inline constexpr int64_t NO_DROPOUT_TENSOR = -1;

/// What a graph's dropout fields bind: the probability (0 when the graph has none) and the
/// seed and offset tensors (NO_DROPOUT_TENSOR when absent).
struct SdpaDropout
{
    float probability = 0.0F;
    int64_t seed = NO_DROPOUT_TENSOR;
    int64_t offset = NO_DROPOUT_TENSOR;

    /// Whether any score is dropped: a probability of 0 is the undropped result, which every
    /// object computes.
    bool active() const
    {
        return probability > 0.0F;
    }
};

/**
 * @brief Whether @p uid names a Philox seed or offset the kernels can take: one value, as a
 *        `[1, 1, 1, 1]` tensor (cuDNN's form), INT64 or INT32.
 *
 * - On the device: INT64 only, at an 8-byte-aligned base, since the kernels read 8 bytes there
 *   (PyTorch-ROCm passes its seed and offset this way, so such a graph binds with no copy).
 * - Pass-by-value (a host scalar, constant or supplied at execute): INT64 or INT32, widened to
 *   64 bits on the host.
 * - Never virtual or ragged. INT32 on the device declines: the kernels would read 8 bytes.
 */
inline bool isServablePhiloxTensor(const MatchContext& context, int64_t uid)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr || tensor->dims() == nullptr || tensor->strides() == nullptr
       || tensor->dims()->size() != SDPA_RANK || tensor->strides()->size() != SDPA_RANK
       || tensor->virtual_() || tensor->ragged_offset_tensor_uid().has_value())
    {
        return false;
    }
    for(const auto extent : *tensor->dims())
    {
        if(extent != 1)
        {
            return false;
        }
    }
    const auto dataType = tensor->data_type();
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor))
    {
        return dataType == data_objects::DataType::INT64
               || dataType == data_objects::DataType::INT32;
    }
    return dataType == data_objects::DataType::INT64 && declaresAlignment(*tensor, sizeof(int64_t));
}

/**
 * @brief The kernels' keep threshold for a drop probability @p p: a key is kept iff its Philox
 *        random, read as int32, is greater.
 *
 * AOTriton's `((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32)`, evaluated as its kernel evaluates
 * it: in f32, where the constant rounds to 2^32, then truncated toward zero. Measured against
 * PyTorch-ROCm's flash path at p values chosen so the f32 and double forms disagree on a
 * mask element: it applies this one. (`philox.py`'s host `dropout_threshold` computes it in
 * double and differs by up to 2^7, about 3e-8 of the elements.) Clamped to int32, which no p in
 * [0, 1) leaves.
 */
inline int32_t dropoutThreshold(float p)
{
    const float scaled = (p - 0.5F) * 4294967296.0F;
    if(!(scaled < 2147483648.0F))
    {
        return std::numeric_limits<int32_t>::max();
    }
    if(!(scaled > -2147483648.0F))
    {
        return std::numeric_limits<int32_t>::min();
    }
    return static_cast<int32_t>(scaled);
}

/// The scale the kept probabilities take, folded into 1/l: AOTriton's `1.0 / (1 - dropout_p)`,
/// in f32.
inline float dropoutScale(float p)
{
    return 1.0F / (1.0F - p);
}

/**
 * @brief The graph's dropout fields, or nullopt for a graph that asks for dropout the kernels
 *        do not apply.
 *
 * Served: a probability in [0, 1) with a seed and an offset isServablePhiloxTensor accepts
 * (both required above 0; at 0, the undropped result, either may be absent). Declined, each a
 * wrong answer rather than an error if it were ignored: a probability outside [0, 1) or NaN, a
 * seed or offset without a probability, a dropout mask or scale tensor (cuDNN's second form),
 * and the RNG dump output, which no kernel writes.
 */
inline std::optional<SdpaDropout> servableDropout(const MatchContext& context,
                                                  const data_objects::SdpaAttributes& attributes)
{
    // The mask-and-scale form and the RNG dump: no kernel takes a mask or writes one.
    if(attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    const bool hasSeed = attributes.seed_tensor_uid().has_value();
    const bool hasOffset = attributes.offset_tensor_uid().has_value();
    if(!attributes.dropout_probability().has_value())
    {
        // A seed or an offset with no probability is a graph half-stated: decline it rather
        // than guess.
        if(hasSeed || hasOffset)
        {
            return std::nullopt;
        }
        return SdpaDropout{};
    }
    // A probability in [0, 1), and not NaN.
    const float probability = attributes.dropout_probability().value();
    if(std::isnan(probability) || probability < 0.0F || probability >= 1.0F)
    {
        return std::nullopt;
    }
    SdpaDropout dropout;
    dropout.probability = probability == 0.0F ? 0.0F : probability;
    // Each named tensor must be one the kernels can read, whatever the probability: a field
    // the matcher reads is gated. A drop needs both; at 0 nothing is drawn, so neither is.
    if((hasSeed && !isServablePhiloxTensor(context, attributes.seed_tensor_uid().value()))
       || (hasOffset && !isServablePhiloxTensor(context, attributes.offset_tensor_uid().value()))
       || (dropout.active() && (!hasSeed || !hasOffset)))
    {
        return std::nullopt;
    }
    dropout.seed = hasSeed ? attributes.seed_tensor_uid().value() : NO_DROPOUT_TENSOR;
    dropout.offset = hasOffset ? attributes.offset_tensor_uid().value() : NO_DROPOUT_TENSOR;
    return dropout;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
