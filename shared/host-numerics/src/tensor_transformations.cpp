// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <complex>
#include <cstring>
#include <limits>
#include <roc/host_numerics/tensor.hpp>
#include <type_traits>

#include "detail/layout_iteration.hpp"
#include "detail/tensor_conversion.hpp"
#include "detail/tensor_storage.hpp"

namespace roc::host_numerics {
namespace {
constexpr size_t conversionBlockElements = 64 * 1024;
// Opposite contiguous orders use square tiles. A side spans at least one cache line in the wider
// encoding, while the minimum avoids excessive block counts for wide scalar representations.
constexpr size_t assumedCacheLineBytes = 64;
constexpr size_t minimumTileExtent = 16;
constexpr size_t float16LookupMinimumElements = 4 * conversionBlockElements;

const std::array<float, size_t{1} << 16>& float16DecodeTable() {
    static const std::array<float, size_t{1} << 16> values = [] {
        std::array<float, size_t{1} << 16> result{};
        for (size_t value = 0; value < result.size(); ++value)
            result[value] = detail::decodeFloat16(static_cast<uint16_t>(value));
        return result;
    }();
    return values;
}

using CopyConvertedRangeFunction = void (*)(std::span<const std::byte>, ptrdiff_t, ptrdiff_t,
                                            std::span<std::byte>, ptrdiff_t, ptrdiff_t, size_t,
                                            const ScalarConversionOptions&);

template <typename SourceTag, typename DestinationTag, bool UseFloat16Lookup>
void copyConvertedRange(std::span<const std::byte> sourceStorage, ptrdiff_t sourceOffset,
                        ptrdiff_t sourceStride, std::span<std::byte> destinationStorage,
                        ptrdiff_t destinationOffset, ptrdiff_t destinationStride, size_t count,
                        const ScalarConversionOptions& options) {
    if constexpr (SourceTag::type == DestinationTag::type) {
        if constexpr (!std::is_void_v<typename SourceTag::Storage>) {
            using Storage = typename SourceTag::Storage;
            if (sourceStride == 1 && destinationStride == 1) {
                std::memcpy(
                    destinationStorage.data() +
                        static_cast<size_t>(destinationOffset) * sizeof(Storage),
                    sourceStorage.data() + static_cast<size_t>(sourceOffset) * sizeof(Storage),
                    count * sizeof(Storage));
                return;
            }
            for (size_t index = 0; index < count; ++index) {
                Storage encoded;
                std::memcpy(
                    &encoded,
                    sourceStorage.data() + static_cast<size_t>(sourceOffset) * sizeof(Storage),
                    sizeof(encoded));
                std::memcpy(destinationStorage.data() +
                                static_cast<size_t>(destinationOffset) * sizeof(Storage),
                            &encoded, sizeof(encoded));
                sourceOffset += sourceStride;
                destinationOffset += destinationStride;
            }
        } else {
            constexpr uint16_t bits = scalarTypeInfo(SourceTag::type).storageBits;
            for (size_t index = 0; index < count; ++index) {
                detail::copyBitRange(
                    sourceStorage, detail::bitOffset(SourceTag::type, sourceOffset),
                    destinationStorage, detail::bitOffset(DestinationTag::type, destinationOffset),
                    bits);
                sourceOffset += sourceStride;
                destinationOffset += destinationStride;
            }
        }
        return;
    }

    const auto convert = [&]<typename Intermediate>() {
        const float* float16Values = nullptr;
        if constexpr (UseFloat16Lookup && SourceTag::type == ScalarType::Float16)
            float16Values = float16DecodeTable().data();
        for (size_t index = 0; index < count; ++index) {
            Intermediate value;
            if constexpr (!std::is_void_v<typename SourceTag::Storage>) {
                typename SourceTag::Storage encoded;
                std::memcpy(&encoded,
                            sourceStorage.data() + static_cast<size_t>(sourceOffset) *
                                                       sizeof(typename SourceTag::Storage),
                            sizeof(encoded));
                if constexpr (UseFloat16Lookup && SourceTag::type == ScalarType::Float16)
                    value =
                        detail::convertScalarValue<Intermediate>(float16Values[encoded], options);
                else
                    value =
                        detail::decodeScalarValueKnown<SourceTag, Intermediate>(encoded, options);
            } else {
                value = detail::decodeScalarKnown<SourceTag, Intermediate>(sourceStorage,
                                                                           sourceOffset, options);
            }

            if constexpr (!std::is_void_v<typename DestinationTag::Storage>) {
                const auto encoded = detail::encodeScalarValueKnown<DestinationTag>(value, options);
                std::memcpy(destinationStorage.data() +
                                static_cast<size_t>(destinationOffset) * sizeof(encoded),
                            &encoded, sizeof(encoded));
            } else {
                detail::encodeScalarKnown<DestinationTag>(destinationStorage, destinationOffset,
                                                          value, options);
            }
            sourceOffset += sourceStride;
            destinationOffset += destinationStride;
        }
    };

    constexpr ScalarCategory sourceCategory = scalarTypeInfo(SourceTag::type).category;
    if constexpr (sourceCategory == ScalarCategory::Boolean ||
                  sourceCategory == ScalarCategory::UnsignedInteger)
        convert.template operator()<uint64_t>();
    else if constexpr (sourceCategory == ScalarCategory::SignedInteger)
        convert.template operator()<int64_t>();
    else if constexpr (sourceCategory == ScalarCategory::Complex)
        convert.template operator()<std::complex<double>>();
    else
        convert.template operator()<double>();
}

CopyConvertedRangeFunction copyConvertedRangeFunction(ScalarType sourceType,
                                                      ScalarType destinationType,
                                                      bool useFloat16Lookup) {
    return visitScalarType(
        sourceType,
        [destinationType, useFloat16Lookup]<typename SourceTag>() -> CopyConvertedRangeFunction {
            if constexpr (SourceTag::type == ScalarType::Float16) {
                return visitScalarType(
                    destinationType,
                    [useFloat16Lookup]<typename DestinationTag>() -> CopyConvertedRangeFunction {
                        return useFloat16Lookup
                                   ? &copyConvertedRange<SourceTag, DestinationTag, true>
                                   : &copyConvertedRange<SourceTag, DestinationTag, false>;
                    });
            } else {
                return visitScalarType(
                    destinationType, []<typename DestinationTag>() -> CopyConvertedRangeFunction {
                        return &copyConvertedRange<SourceTag, DestinationTag, false>;
                    });
            }
        });
}

IndexOrder preferredTraversalOrder(const Layout& source, const Layout& destination) {
    if (detail::isContiguous(source, IndexOrder::FirstDimensionFastest))
        return IndexOrder::FirstDimensionFastest;
    if (detail::isContiguous(source, IndexOrder::LastDimensionFastest))
        return IndexOrder::LastDimensionFastest;
    if (source.shape().rank() == 0) return IndexOrder::LastDimensionFastest;

    const auto endpointStride = [](const Layout& layout, size_t dimension) {
        return layout.shape()[dimension] <= 1 ? std::numeric_limits<uint64_t>::max()
                                              : detail::strideMagnitude(layout.stride(dimension));
    };
    const size_t last = source.shape().rank() - 1;
    const uint64_t sourceFirst = endpointStride(source, 0);
    const uint64_t sourceLast = endpointStride(source, last);
    if (sourceFirst != sourceLast)
        return sourceFirst < sourceLast ? IndexOrder::FirstDimensionFastest
                                        : IndexOrder::LastDimensionFastest;

    const uint64_t destinationFirst = endpointStride(destination, 0);
    const uint64_t destinationLast = endpointStride(destination, last);
    return destinationFirst < destinationLast ? IndexOrder::FirstDimensionFastest
                                              : IndexOrder::LastDimensionFastest;
}
}  // namespace

detail::TensorConversion::TensorConversion(const Tensor& source, Tensor destination,
                                           const ScalarConversionOptions& options)
    : m_source(source), m_destination(std::move(destination)), m_options(options) {
    if (m_source.shape() != m_destination.shape())
        throw std::invalid_argument(
            "Tensor conversion requires matching source and output shapes.");
    if (m_destination.layout() != m_source.layout() &&
        !detail::hasProvablyDistinctElementOffsets(m_destination.layout()))
        throw std::invalid_argument(
            "Tensor conversion requires non-overlapping destination elements when relayouting.");
    if (detail::byteRangesOverlap(m_source.rawEncodedBackingStorage(),
                                  m_destination.rawEncodedBackingStorage()))
        throw std::invalid_argument("Tensor conversion source and destination storage overlap.");

    const size_t elements = m_source.elementCount();
    m_copyRange = copyConvertedRangeFunction(
        m_source.type(), m_destination.type(),
        m_source.type() == ScalarType::Float16 && elements >= float16LookupMinimumElements);
    if (elements == 0) return;

    const bool sourceFirst =
        detail::isContiguous(m_source.layout(), IndexOrder::FirstDimensionFastest);
    const bool sourceLast =
        detail::isContiguous(m_source.layout(), IndexOrder::LastDimensionFastest);
    const bool destinationFirst =
        detail::isContiguous(m_destination.layout(), IndexOrder::FirstDimensionFastest);
    const bool destinationLast =
        detail::isContiguous(m_destination.layout(), IndexOrder::LastDimensionFastest);
    // Matching dense orders are one physical range even when their logical index order differs.
    if ((sourceFirst && destinationFirst) || (sourceLast && destinationLast)) {
        m_traversal = Traversal::Linear;
        m_blockCount = elements / conversionBlockElements +
                       static_cast<size_t>(elements % conversionBlockElements != 0);
        return;
    }

    const bool byteAddressable = !scalarTypeInfo(m_source.type()).isPacked() &&
                                 !scalarTypeInfo(m_destination.type()).isPacked();
    // Opposite dense matrix orders need cache-local tiles. Every other layout uses complete
    // inner-dimension runs, so arbitrary affine layouts retain the same conversion contract.
    if (byteAddressable && m_source.shape().rank() == 2 &&
        ((sourceFirst && destinationLast) || (sourceLast && destinationFirst))) {
        m_traversal = Traversal::TiledMatrix;
        m_sourceFirstDimensionFastest = sourceFirst;
        const size_t sourceBytes = scalarTypeInfo(m_source.type()).storageBits / 8;
        const size_t destinationBytes = scalarTypeInfo(m_destination.type()).storageBits / 8;
        m_tileExtent = std::max(minimumTileExtent,
                                assumedCacheLineBytes / std::max(sourceBytes, destinationBytes));
        const size_t rowTiles = m_source.shape()[0] / m_tileExtent +
                                static_cast<size_t>(m_source.shape()[0] % m_tileExtent != 0);
        const size_t columnTiles = m_source.shape()[1] / m_tileExtent +
                                   static_cast<size_t>(m_source.shape()[1] % m_tileExtent != 0);
        m_blockCount = rowTiles * columnTiles;
        return;
    }

    m_traversal = Traversal::General;
    m_indexOrder = preferredTraversalOrder(m_source.layout(), m_destination.layout());
    m_blockCount = elements / conversionBlockElements +
                   static_cast<size_t>(elements % conversionBlockElements != 0);
}

size_t detail::TensorConversion::elementCount() const {
    return m_source.elementCount();
}

size_t detail::TensorConversion::blockCount() const {
    return m_blockCount;
}

bool detail::TensorConversion::canParallelize() const {
    return !scalarTypeInfo(m_destination.type()).isPacked() &&
           detail::hasProvablyDistinctElementOffsets(m_destination.layout());
}

void detail::TensorConversion::copyBlock(size_t block) const {
    if (block >= m_blockCount)
        throw std::out_of_range("Tensor conversion block index is out of range.");
    const auto sourceStorage = m_source.rawEncodedBackingStorage();
    const auto destinationStorage = m_destination.rawEncodedBackingStorage();

    if (m_traversal == Traversal::Linear) {
        const size_t first = block * conversionBlockElements;
        const size_t count = std::min(conversionBlockElements, elementCount() - first);
        m_copyRange(sourceStorage, m_source.layout().offset() + static_cast<ptrdiff_t>(first), 1,
                    destinationStorage,
                    m_destination.layout().offset() + static_cast<ptrdiff_t>(first), 1, count,
                    m_options);
        return;
    }

    if (m_traversal == Traversal::TiledMatrix) {
        const size_t rows = m_source.shape()[0];
        const size_t columns = m_source.shape()[1];
        const size_t rowTiles = rows / m_tileExtent + static_cast<size_t>(rows % m_tileExtent != 0);
        const size_t rowBase = block % rowTiles * m_tileExtent;
        const size_t columnBase = block / rowTiles * m_tileExtent;
        const size_t blockRows = std::min(m_tileExtent, rows - rowBase);
        const size_t blockColumns = std::min(m_tileExtent, columns - columnBase);
        if (m_sourceFirstDimensionFastest) {
            for (size_t column = 0; column < blockColumns; ++column) {
                const size_t coordinateColumn = columnBase + column;
                m_copyRange(
                    sourceStorage,
                    m_source.layout().offset() +
                        static_cast<ptrdiff_t>(rowBase) * m_source.layout().stride(0) +
                        static_cast<ptrdiff_t>(coordinateColumn) * m_source.layout().stride(1),
                    m_source.layout().stride(0), destinationStorage,
                    m_destination.layout().offset() +
                        static_cast<ptrdiff_t>(rowBase) * m_destination.layout().stride(0) +
                        static_cast<ptrdiff_t>(coordinateColumn) * m_destination.layout().stride(1),
                    m_destination.layout().stride(0), blockRows, m_options);
            }
        } else {
            for (size_t row = 0; row < blockRows; ++row) {
                const size_t coordinateRow = rowBase + row;
                m_copyRange(
                    sourceStorage,
                    m_source.layout().offset() +
                        static_cast<ptrdiff_t>(coordinateRow) * m_source.layout().stride(0) +
                        static_cast<ptrdiff_t>(columnBase) * m_source.layout().stride(1),
                    m_source.layout().stride(1), destinationStorage,
                    m_destination.layout().offset() +
                        static_cast<ptrdiff_t>(coordinateRow) * m_destination.layout().stride(0) +
                        static_cast<ptrdiff_t>(columnBase) * m_destination.layout().stride(1),
                    m_destination.layout().stride(1), blockColumns, m_options);
            }
        }
        return;
    }

    const size_t first = block * conversionBlockElements;
    const size_t count = std::min(conversionBlockElements, elementCount() - first);
    const size_t pastLast = first + count;
    detail::forEachLayoutOffsetPairRunRange(
        m_source.layout(), m_destination.layout(), m_indexOrder, first, pastLast,
        [&](size_t, ptrdiff_t sourceOffset, ptrdiff_t sourceStride, ptrdiff_t destinationOffset,
            ptrdiff_t destinationStride, size_t count) {
            m_copyRange(sourceStorage, sourceOffset, sourceStride, destinationStorage,
                        destinationOffset, destinationStride, count, m_options);
        });
}

void detail::TensorConversion::copyAll() const {
    for (size_t block = 0; block < blockCount(); ++block) copyBlock(block);
}

Tensor Tensor::reshapeSharingStorage(Shape shape) const {
    if (shape.elementCount() != elementCount())
        throw std::invalid_argument("Tensor reshape requires the same logical element count.");
    if (layout() != Layout::contiguousLastDimensionFastest(this->shape()))
        throw std::invalid_argument(
            "Tensor reshape requires a contiguous last-dimension-fastest layout.");
    return shareStorageWithLayout(Layout::contiguousLastDimensionFastest(shape));
}

Tensor Tensor::expandDims(size_t axis) const {
    if (axis > shape().rank())
        throw std::out_of_range("Tensor expansion axis exceeds the destination rank.");
    std::vector<size_t> dimensions(shape().dimensions().begin(), shape().dimensions().end());
    std::vector<ptrdiff_t> strides(layout().strides().begin(), layout().strides().end());
    dimensions.insert(dimensions.begin() + static_cast<ptrdiff_t>(axis), 1);
    strides.insert(strides.begin() + static_cast<ptrdiff_t>(axis), 0);
    return shareStorageWithLayout(
        Layout(Shape(std::move(dimensions)), std::move(strides), layout().offset()));
}

Tensor Tensor::broadcastTo(Shape shape) const {
    if (shape.rank() < this->shape().rank())
        throw std::invalid_argument("Tensor cannot broadcast to a lower-rank shape.");

    std::vector<ptrdiff_t> strides(shape.rank(), 0);
    const size_t leadingDimensions = shape.rank() - this->shape().rank();
    for (size_t sourceDimension = 0; sourceDimension < this->shape().rank(); ++sourceDimension) {
        const size_t destinationDimension = leadingDimensions + sourceDimension;
        const size_t sourceExtent = this->shape()[sourceDimension];
        const size_t destinationExtent = shape[destinationDimension];
        if (sourceExtent != destinationExtent && sourceExtent != 1)
            throw std::invalid_argument("Tensor cannot broadcast to the requested shape.");
        if (sourceExtent == destinationExtent)
            strides[destinationDimension] = layout().stride(sourceDimension);
    }
    return shareStorageWithLayout(Layout(std::move(shape), std::move(strides), layout().offset()));
}

Tensor Tensor::copyWithZeroPadding(Shape shape) const {
    if (shape.rank() != this->shape().rank())
        throw std::invalid_argument("Tensor padding requires the same rank.");
    for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
        if (shape[dimension] < this->shape()[dimension])
            throw std::invalid_argument("Tensor padding cannot shrink a dimension.");
    }

    Tensor result(type(), std::move(shape));
    const uint16_t bits = scalarTypeInfo(type()).storageBits;
    detail::forEachIndex(this->shape(), [&](std::span<const size_t> indices, size_t) {
        detail::copyBitRange(
            rawEncodedBackingStorage(), detail::bitOffset(type(), layout().elementOffset(indices)),
            result.rawEncodedBackingStorage(),
            detail::bitOffset(type(), result.layout().elementOffset(indices)), bits);
    });
    return result;
}

Tensor Tensor::copyWithPermutedDimensions(std::span<const size_t> destinationToSource) const {
    if (destinationToSource.size() != shape().rank())
        throw std::invalid_argument("Tensor permutation rank does not match the tensor.");

    std::vector<bool> seen(shape().rank(), false);
    std::vector<size_t> destinationDimensions(shape().rank(), 0);
    for (size_t destinationDimension = 0; destinationDimension < shape().rank();
         ++destinationDimension) {
        const size_t sourceDimension = destinationToSource[destinationDimension];
        if (sourceDimension >= shape().rank() || seen[sourceDimension])
            throw std::invalid_argument(
                "Tensor permutation must contain each source dimension exactly once.");
        seen[sourceDimension] = true;
        destinationDimensions[destinationDimension] = shape()[sourceDimension];
    }

    Tensor result(type(), Shape(std::move(destinationDimensions)));
    std::vector<size_t> destinationIndices(shape().rank(), 0);
    const uint16_t bits = scalarTypeInfo(type()).storageBits;
    detail::forEachIndex(shape(), [&](std::span<const size_t> sourceIndices, size_t) {
        for (size_t destinationDimension = 0; destinationDimension < shape().rank();
             ++destinationDimension)
            destinationIndices[destinationDimension] =
                sourceIndices[destinationToSource[destinationDimension]];

        detail::copyBitRange(
            rawEncodedBackingStorage(),
            detail::bitOffset(type(), layout().elementOffset(sourceIndices)),
            result.rawEncodedBackingStorage(),
            detail::bitOffset(type(), result.layout().elementOffset(destinationIndices)), bits);
    });
    return result;
}

Tensor Tensor::copyWithPermutedDimensions(std::initializer_list<size_t> destinationToSource) const {
    return copyWithPermutedDimensions(
        std::span<const size_t>(destinationToSource.begin(), destinationToSource.size()));
}

Tensor Tensor::copyConvertedTo(ScalarType destinationType) const {
    return copyConvertedTo(destinationType, layout(),
                           detail::implicitStorageConversionOptions(destinationType));
}

Tensor Tensor::copyConvertedTo(ScalarType destinationType,
                               const ScalarConversionOptions& options) const {
    return copyConvertedTo(destinationType, layout(), options);
}

Tensor Tensor::copyConvertedTo(ScalarType destinationType, Layout destinationLayout) const {
    return copyConvertedTo(destinationType, std::move(destinationLayout),
                           detail::implicitStorageConversionOptions(destinationType));
}

Tensor Tensor::copyConvertedTo(ScalarType destinationType, Layout destinationLayout,
                               const ScalarConversionOptions& options) const {
    if (destinationLayout.shape() != shape())
        throw std::invalid_argument(
            "Tensor conversion requires matching source and output shapes.");
    if (destinationType == type() && destinationLayout == layout()) return deepCopy();
    if (destinationLayout != layout() &&
        !detail::hasProvablyDistinctElementOffsets(destinationLayout))
        throw std::invalid_argument(
            "Tensor conversion requires non-overlapping destination elements when relayouting.");

    const bool fullyWritesBackingStorage =
        !scalarTypeInfo(destinationType).isPacked() && destinationLayout.offset() == 0 &&
        (detail::isContiguous(destinationLayout, IndexOrder::FirstDimensionFastest) ||
         detail::isContiguous(destinationLayout, IndexOrder::LastDimensionFastest));
    Tensor result =
        fullyWritesBackingStorage
            ? Tensor::allocateUninitialized(destinationType, std::move(destinationLayout))
            : Tensor(destinationType, std::move(destinationLayout));
    detail::TensorConversion(*this, result, options).copyAll();
    return result;
}
}  // namespace roc::host_numerics
