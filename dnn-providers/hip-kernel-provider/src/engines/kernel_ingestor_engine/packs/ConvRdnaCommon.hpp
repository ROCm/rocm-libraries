// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/**
 * @file ConvRdnaCommon.hpp
 * @brief What the RDNA convolution packs (descriptors/hip/conv_rdna/ and
 *        hip/conv_dgrad_dw_rdna/) share: operand validation, layout detection, the convolution
 *        geometry, and the features a ranker reads. Node-type specific matching and the
 *        dispatch stay in each pack.
 *
 * Every problem is described in the forward view: x is (N, C, spatial), w is
 * (K, C / groups, filter), y is (N, K, out). A backward-data pack passes dx as x and dy as y.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna
{

/// How all three operands are packed. hipDNN keeps dims in NC(D)HW order and expresses the
/// layout through strides only.
enum class Layout
{
    CHANNELS_FIRST, ///< NC(D)HW, KC(Z)YX
    CHANNELS_LAST, ///< N(D)HWC, K(Z)YXC
};

/// The KMD `layout` value of @p layout.
std::string_view layoutName(Layout layout);

/// A convolution over 1, 2 or 3 spatial dimensions. Spatial arrays are (depth, height,
/// width), right-aligned: a 2-D problem has depth 1 and a 1-D problem depth and height 1
/// everywhere, so a 3-D kernel runs either unchanged.
struct ConvProblem
{
    int spatialRank = 2;
    int n = 0;
    int c = 0;
    int k = 0;
    int groups = 1;
    std::array<int, 3> in{1, 1, 1};
    std::array<int, 3> filter{1, 1, 1};
    std::array<int, 3> out{1, 1, 1};
    /// Pre-padding; post-padding only shapes `out`.
    std::array<int, 3> pad{0, 0, 0};
    std::array<int, 3> stride{1, 1, 1};
    std::array<int, 3> dilation{1, 1, 1};
    Layout layout = Layout::CHANNELS_FIRST;
    hipdnn_flatbuffers_sdk::data_objects::DataType dtype
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;

    int channelsPerGroup() const
    {
        return c / groups;
    }
    int outChannelsPerGroup() const
    {
        return k / groups;
    }
    bool depthwise() const
    {
        return groups == c && groups == k;
    }
};

/// The convolution attributes a problem is read from, common to every conv node type.
struct ConvAttributes
{
    hipdnn_flatbuffers_sdk::data_objects::ConvMode mode
        = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    const flatbuffers::Vector<int64_t>* stride = nullptr;
    const flatbuffers::Vector<int64_t>* dilation = nullptr;
    const flatbuffers::Vector<int64_t>* prePadding = nullptr;
    const flatbuffers::Vector<int64_t>* postPadding = nullptr;
};

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    findTensor(const hipdnn_plugin_sdk::ingestor::MatchContext& context, int64_t uid);

/**
 * @brief The problem x, w and y describe under @p attributes, or nullopt if these packs
 *        cannot serve it.
 *
 * Requires rank 3, 4 or 5 for all three, real device data, one dtype among FLOAT, HALF and
 * BFLOAT16, one packed layout shared by all three, cross-correlation, every extent in int,
 * C and K divisible by groups = C / w channels, and y exactly the output shape.
 */
std::optional<ConvProblem>
    convProblem(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& x,
                const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& w,
                const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& y,
                const ConvAttributes& attributes);

/// Binds the operand uids under `<prefix>.x.uid` (w, y alike), the tensor features under
/// `<prefix>.x` (w, y), and the op-intrinsic `<prefix>.flops` and `<prefix>.bytes`.
void bindConvFeatures(hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
                      std::string_view prefix,
                      const ConvProblem& problem,
                      int64_t xUid,
                      int64_t wUid,
                      int64_t yUid,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& x,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& w,
                      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& y);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
