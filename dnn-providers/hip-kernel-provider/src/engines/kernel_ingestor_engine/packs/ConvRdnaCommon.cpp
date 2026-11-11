// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvRdnaCommon.hpp"

#include <limits>
#include <string>

#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::MatchContext;

namespace
{

constexpr int64_t INT_MAX_VALUE = std::numeric_limits<int>::max();

/// Bit per layout a tensor is packed in; a tensor with a unit channel or spatial extent can
/// be packed in both.
constexpr unsigned CHANNELS_FIRST_BIT = 1U;
constexpr unsigned CHANNELS_LAST_BIT = 2U;

/// The layouts @p tensor is packed in (dims in NC(D)HW order). Zero when it is in neither,
/// not rank 4 or 5, or has an extent outside [1, INT_MAX].
unsigned packedLayouts(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || strides->size() != dims->size()
       || (dims->size() != 4 && dims->size() != 5))
    {
        return 0;
    }
    const auto rank = static_cast<flatbuffers::uoffset_t>(dims->size());
    for(flatbuffers::uoffset_t i = 0; i < rank; ++i)
    {
        if(dims->Get(i) < 1 || dims->Get(i) > INT_MAX_VALUE)
        {
            return 0;
        }
    }

    unsigned layouts = 0;

    // NC(D)HW: row-major over the dims as given.
    int64_t expected = 1;
    bool first = true;
    for(auto i = rank; i-- > 0;)
    {
        first = first && strides->Get(i) == expected;
        expected *= dims->Get(i);
    }
    if(first)
    {
        layouts |= CHANNELS_FIRST_BIT;
    }

    // N(D)HWC: channel fastest, then the spatial dims from the last, then the batch.
    expected = 1;
    bool last = strides->Get(1) == expected;
    expected *= dims->Get(1);
    for(auto i = rank; i-- > 2;)
    {
        last = last && strides->Get(i) == expected;
        expected *= dims->Get(i);
    }
    last = last && strides->Get(0) == expected;
    if(last)
    {
        layouts |= CHANNELS_LAST_BIT;
    }
    return layouts;
}

bool isServedDtype(data_objects::DataType dataType)
{
    return dataType == data_objects::DataType::FLOAT || dataType == data_objects::DataType::HALF
           || dataType == data_objects::DataType::BFLOAT16;
}

/// The @p count spatial values of @p values, each at least @p minimum and at most INT_MAX.
std::optional<std::array<int, 3>>
    spatial(const flatbuffers::Vector<int64_t>* values, size_t count, int64_t minimum, int fill)
{
    if(values == nullptr || values->size() != count)
    {
        return std::nullopt;
    }
    std::array<int, 3> result{fill, fill, fill};
    for(size_t i = 0; i < count; ++i)
    {
        const int64_t value = values->Get(static_cast<flatbuffers::uoffset_t>(i));
        if(value < minimum || value > INT_MAX_VALUE)
        {
            return std::nullopt;
        }
        // Right-aligned: a 2-D problem fills height and width and keeps depth at `fill`.
        result[3 - count + i] = static_cast<int>(value);
    }
    return result;
}

} // namespace

std::string_view layoutName(Layout layout)
{
    return layout == Layout::CHANNELS_LAST ? "CHANNELS_LAST" : "CHANNELS_FIRST";
}

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

std::optional<ConvProblem> convProblem(const data_objects::TensorAttributes& x,
                                       const data_objects::TensorAttributes& w,
                                       const data_objects::TensorAttributes& y,
                                       const ConvAttributes& attributes)
{
    if(attributes.mode != data_objects::ConvMode::CROSS_CORRELATION)
    {
        return std::nullopt;
    }

    // Device operands of one served dtype. A pass-by-value tensor holds a host pointer.
    for(const auto* tensor : {&x, &w, &y})
    {
        if(tensor->virtual_() || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor)
           || !isServedDtype(tensor->data_type()))
        {
            return std::nullopt;
        }
    }
    if(x.data_type() != w.data_type() || x.data_type() != y.data_type())
    {
        return std::nullopt;
    }

    // One layout for all three; the kernels take no strides. Channels-first wins a tie.
    const unsigned layouts = packedLayouts(x) & packedLayouts(w) & packedLayouts(y);
    if(layouts == 0)
    {
        return std::nullopt;
    }

    const auto* xDims = x.dims();
    const auto* wDims = w.dims();
    const auto* yDims = y.dims();
    if(wDims->size() != xDims->size() || yDims->size() != xDims->size())
    {
        return std::nullopt;
    }
    const auto spatialRank = static_cast<size_t>(xDims->size() - 2);

    const auto stride = spatial(attributes.stride, spatialRank, 1, 1);
    const auto dilation = spatial(attributes.dilation, spatialRank, 1, 1);
    const auto prePadding = spatial(attributes.prePadding, spatialRank, 0, 0);
    const auto postPadding = spatial(attributes.postPadding, spatialRank, 0, 0);
    if(!stride || !dilation || !prePadding || !postPadding)
    {
        return std::nullopt;
    }

    ConvProblem problem;
    problem.spatialRank = static_cast<int>(spatialRank);
    problem.layout
        = (layouts & CHANNELS_FIRST_BIT) != 0 ? Layout::CHANNELS_FIRST : Layout::CHANNELS_LAST;
    problem.dtype = x.data_type();
    problem.n = static_cast<int>(xDims->Get(0));
    problem.c = static_cast<int>(xDims->Get(1));
    problem.k = static_cast<int>(wDims->Get(0));
    problem.stride = *stride;
    problem.dilation = *dilation;
    problem.pad = *prePadding;

    // hipDNN carries no group attribute: groups = x channels / w channels.
    const int64_t wChannels = wDims->Get(1);
    if(problem.c % wChannels != 0)
    {
        return std::nullopt;
    }
    problem.groups = static_cast<int>(problem.c / wChannels);
    if(problem.k % problem.groups != 0 || yDims->Get(0) != problem.n || yDims->Get(1) != problem.k)
    {
        return std::nullopt;
    }

    for(size_t i = 0; i < spatialRank; ++i)
    {
        const auto axis = static_cast<flatbuffers::uoffset_t>(2 + i);
        const size_t slot = 3 - spatialRank + i;
        problem.in[slot] = static_cast<int>(xDims->Get(axis));
        problem.filter[slot] = static_cast<int>(wDims->Get(axis));

        // Output extent from the padded input and the dilated filter, in int64_t.
        const int64_t span = int64_t{problem.in[slot]} + (*prePadding)[slot] + (*postPadding)[slot]
                             - int64_t{problem.dilation[slot]} * (problem.filter[slot] - 1) - 1;
        if(span < 0)
        {
            return std::nullopt;
        }
        const int64_t out = span / problem.stride[slot] + 1;
        // y must be exactly what a kernel writes.
        if(yDims->Get(axis) != out)
        {
            return std::nullopt;
        }
        problem.out[slot] = static_cast<int>(out);
    }
    return problem;
}

void bindConvFeatures(BoundTokens& bound,
                      std::string_view prefix,
                      const ConvProblem& problem,
                      int64_t xUid,
                      int64_t wUid,
                      int64_t yUid,
                      const data_objects::TensorAttributes& x,
                      const data_objects::TensorAttributes& w,
                      const data_objects::TensorAttributes& y)
{
    const std::string root(prefix);
    bound[root + ".x.uid"] = xUid;
    bound[root + ".w.uid"] = wUid;
    bound[root + ".y.uid"] = yUid;

    pack_features::bindTensorFields(bound, root + ".x", x);
    pack_features::bindTensorFields(bound, root + ".w", w);
    pack_features::bindTensorFields(bound, root + ".y", y);

    // 2 x N x K x out positions x (C / groups) x filter taps, padded taps counted as the
    // dense convention does.
    const auto flops = pack_features::checkedProduct({2,
                                                      problem.n,
                                                      problem.k,
                                                      problem.out[0],
                                                      problem.out[1],
                                                      problem.out[2],
                                                      problem.channelsPerGroup(),
                                                      problem.filter[0],
                                                      problem.filter[1],
                                                      problem.filter[2]});
    if(flops.has_value())
    {
        bound[root + ".flops"] = *flops;
    }
    const auto bytes = pack_features::movedBytes({&x, &w, &y});
    if(bytes.has_value())
    {
        bound[root + ".bytes"] = *bytes;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
