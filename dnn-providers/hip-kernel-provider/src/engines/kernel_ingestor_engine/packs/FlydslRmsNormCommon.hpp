// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/**
 * @file FlydslRmsNormCommon.hpp
 * @brief What the flyDSL RMS-norm forward and backward packs share: the kernel descriptor
 *        structs, and the graph checks both apply the same way -- the normalised-axes
 *        split, the packed-layout and per-row-statistic rules, and the dtype vocabulary.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_norm
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// `N: 0` in a UKD's metadata means "this kernel takes N at runtime". A real graph never
/// has a zero-width normalised axis, so the value is free to carry the sentinel; see
/// `GENERIC_N` in `flydsl/gen_descriptors.py` for why null could not be used instead.
constexpr int64_t GENERIC_N_SENTINEL = 0;

/// A row-major 2-D operand as the kernel reads it: `fly.make_layout((M, N), (stride, 1))`
/// lowered to a packed LLVM struct. The innermost stride is a compile-time 1 and so has no
/// slot here -- the reason `isPackedRowMajor` below is an applicability condition and not a
/// convenience.
struct Desc2D
{
    int32_t shape0 = 0; ///< M, the number of rows.
    int32_t shape1 = 0; ///< N, the normalised axis.
    int64_t stride0 = 0; ///< Row stride in elements.
};

/// A 1-D operand (the gamma vector): length only, unit stride implied.
struct Desc1D
{
    int32_t shape0 = 0;
};

// The kernels' kernarg segments hold these descriptors packed, each after its pointer;
// natural C++ alignment happens to reproduce that layout exactly, which is worth
// asserting rather than assuming, since a mismatch would corrupt every launch silently.
static_assert(sizeof(Desc2D) == 16, "Desc2D must occupy the kernel's 16-byte 2-D layout");
static_assert(offsetof(Desc2D, shape1) == 4, "Desc2D.shape1 must sit at +4");
static_assert(offsetof(Desc2D, stride0) == 8, "Desc2D.stride0 must sit at +8");
static_assert(sizeof(Desc1D) == 4, "Desc1D must occupy the kernel's 4-byte 1-D layout");

inline const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// Runs on an unvalidated graph, so must be total: a caller can present a tensor the
/// frontend would have rejected, with null or mismatched dims/strides.
inline bool hasUsableShape(const data_objects::TensorAttributes& tensor, uint32_t minimumRank)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    return dims != nullptr && strides != nullptr && strides->size() == dims->size()
           && dims->size() >= minimumRank;
}

/// Dense row-major: innermost stride 1 and every outer stride the product of the extents
/// inside it. Stronger than the descriptor strictly needs -- it could carry a padded row
/// stride -- but it is what makes collapsing rank > 2 to (M, N) sound, since one row stride
/// has to describe all of the leading axes at once.
///
/// An axis of extent 1 is never stepped along, so its stride is not checked: frameworks
/// write anything there (hipdnn_torch gives an `F.rms_norm` weight stride 0 on its unit
/// axes), and the data SDK's own packed test exempts it the same way.
inline bool isPackedRowMajor(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    int64_t expected = 1;
    for(auto axis = static_cast<int64_t>(dims->size()) - 1; axis >= 0; --axis)
    {
        const auto index = static_cast<flatbuffers::uoffset_t>(axis);
        const int64_t extent = dims->Get(index);
        // A non-positive extent, or one whose product would overflow, describes no tensor
        // the kernel can address.
        if((extent != 1 && strides->Get(index) != expected) || extent <= 0
           || expected > std::numeric_limits<int64_t>::max() / extent)
        {
            return false;
        }
        expected *= extent;
    }
    return true;
}

/// The largest operand the kernel addresses: it views each operand as one buffer whose
/// offsets are 32-bit bytes, and takes M and N as int32. The matchers bound the row
/// operands' bytes (x, and y/dy/dx of the same shape and dtype) by it; gamma's bytes are
/// not bounded separately, so an f32 gamma beside bf16/f16 x at 2^29 or more columns is
/// not covered by this bound.
constexpr int64_t MAX_OPERAND_BYTES = 2147483647LL;

/// Product of x's extents over [begin, end). -1 for a non-positive extent or a product
/// past MAX_OPERAND_BYTES, which no launch could address anyway.
inline int64_t extentProduct(const flatbuffers::Vector<int64_t>& dims,
                             flatbuffers::uoffset_t begin,
                             flatbuffers::uoffset_t end)
{
    int64_t product = 1;
    for(auto axis = begin; axis < end; ++axis)
    {
        const int64_t extent = dims.Get(axis);
        if(extent <= 0 || product > MAX_OPERAND_BYTES / extent)
        {
            return -1;
        }
        product *= extent;
    }
    return product;
}

/// x as the kernel sees it: `rows` independent rows of `columns` normalised elements.
struct RowSplit
{
    int64_t rows = 0;
    int64_t columns = 0;
    /// The first normalised axis: axes before it are rows, it and those after are columns.
    flatbuffers::uoffset_t start = 0;
};

/// The frontend's normalised shape (`validateScaleNormalizedShape`): the longest trailing
/// run of axes where gamma's extent equals x's, never including the batch axis 0, with
/// gamma 1 on every axis before it. So cuDNN's `x {B*S, D, 1, 1}` with
/// `gamma {1, D, 1, 1}` normalises over D. A gamma of lower rank -- which the frontend
/// rejects, but the matcher must be total -- is aligned at the innermost axis, its missing
/// leading axes read as 1. Packed x then makes the axes before the run the rows and the
/// run itself one contiguous row of `columns` elements; packed gamma makes it one vector.
inline std::optional<RowSplit> normalisedSplit(const data_objects::TensorAttributes& x,
                                               const data_objects::TensorAttributes& scale)
{
    const auto& xDims = *x.dims();
    const auto& scaleDims = *scale.dims();
    const auto rank = xDims.size();
    if(scaleDims.size() > rank)
    {
        return std::nullopt;
    }
    const auto offset = rank - scaleDims.size();
    const auto scaleExtent = [&](flatbuffers::uoffset_t axis) -> int64_t {
        return axis < offset ? 1 : scaleDims.Get(axis - offset);
    };

    auto start = rank;
    while(start > 1 && scaleExtent(start - 1) == xDims.Get(start - 1))
    {
        --start;
    }
    if(start == rank)
    {
        return std::nullopt;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < start; ++axis)
    {
        if(scaleExtent(axis) != 1)
        {
            return std::nullopt;
        }
    }

    const RowSplit split{extentProduct(xDims, 0, start), extentProduct(xDims, start, rank), start};
    if(split.rows <= 0 || split.columns <= 0)
    {
        return std::nullopt;
    }
    return split;
}

inline bool sameShape(const data_objects::TensorAttributes& lhs,
                      const data_objects::TensorAttributes& rhs)
{
    const auto* lhsDims = lhs.dims();
    const auto* rhsDims = rhs.dims();
    if(lhsDims->size() != rhsDims->size())
    {
        return false;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < lhsDims->size(); ++axis)
    {
        if(lhsDims->Get(axis) != rhsDims->Get(axis))
        {
            return false;
        }
    }
    return true;
}

/// A regular device operand: real storage, readable by the kernel through a pointer, and not
/// a ragged view the kernel would read as dense.
inline bool isDeviceOperand(const data_objects::TensorAttributes& tensor)
{
    return !tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor)
           && !tensor.ragged_offset_tensor_uid().has_value();
}

/// Whether @p tensor promises the 16-byte base alignment the kernels' vector copies assume.
inline bool declaresVectorAlignment(const data_objects::TensorAttributes& tensor)
{
    constexpr int64_t VECTOR_BYTES = 16;
    return tensor.alignment() >= VECTOR_BYTES && tensor.alignment() % VECTOR_BYTES == 0;
}

/// The saved statistic as the frontend shapes it (`RMSNormNode`): x's rank, x's extent on
/// every row axis and 1 on every normalised one, so one f32 per row. The kernel writes it
/// densely, one 32-bit store per row, so it must be packed (unit axes exempt) and 4-byte
/// aligned rather than vector aligned.
inline bool isPerRowStatistic(const data_objects::TensorAttributes& invRms,
                              const data_objects::TensorAttributes& x,
                              const RowSplit& split)
{
    const auto* dims = invRms.dims();
    const auto& xDims = *x.dims();
    if(dims->size() != xDims.size() || invRms.data_type() != data_objects::DataType::FLOAT
       || invRms.alignment() < 4 || invRms.alignment() % 4 != 0)
    {
        return false;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        if(dims->Get(axis) != (axis < split.start ? xDims.Get(axis) : 1))
        {
            return false;
        }
    }
    return isPackedRowMajor(invRms);
}

/// The dtypes the shipped instances were built for; see the `dtype` values in the UKDs.
inline bool isSupportedDataType(data_objects::DataType dataType)
{
    return dataType == data_objects::DataType::BFLOAT16 || dataType == data_objects::DataType::HALF
           || dataType == data_objects::DataType::FLOAT;
}

/// Bytes per element of a supported dtype, for the 32-bit addressing bound.
inline int64_t elementBytes(data_objects::DataType dataType)
{
    return dataType == data_objects::DataType::FLOAT ? 4 : 2;
}

/// The kernel metadata spelling of a graph dtype. Deliberately not `EnumNameDataType`: the
/// descriptors say `bf16`/`f16`/`f32`, the flatbuffer enum says `BFLOAT16`/`HALF`/`FLOAT`,
/// and this is the one place the two vocabularies meet.
inline std::optional<std::string> kernelDataTypeName(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::BFLOAT16:
        return std::string("bf16");
    case data_objects::DataType::HALF:
        return std::string("f16");
    case data_objects::DataType::FLOAT:
        return std::string("f32");
    default:
        return std::nullopt;
    }
}

inline const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    if(it == tensors.end() || it->second == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched flydsl rmsnorm graph has no tensor for uid " + std::to_string(uid));
    }
    return *it->second;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_norm

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
