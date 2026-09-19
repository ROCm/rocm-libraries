// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <complex>
#include <cstring>
#include <roc/host_numerics/tensor.hpp>
#include <type_traits>

#include "detail/layout_iteration.hpp"

namespace roc::host_numerics {
namespace {
template <typename SourceTag, typename DestinationTag>
void copyConvertedElement(std::span<const std::byte> sourceStorage, ptrdiff_t sourceOffset,
                          std::span<std::byte> destinationStorage, ptrdiff_t destinationOffset,
                          const ScalarConversionOptions& options) {
    const auto convert = [&]<typename Intermediate>() {
        if constexpr (!std::is_void_v<typename SourceTag::Storage> &&
                      !std::is_void_v<typename DestinationTag::Storage>) {
            typename SourceTag::Storage encodedSource;
            std::memcpy(&encodedSource,
                        sourceStorage.data() +
                            static_cast<size_t>(sourceOffset) * sizeof(typename SourceTag::Storage),
                        sizeof(encodedSource));
            const Intermediate value =
                detail::decodeScalarValueKnown<SourceTag, Intermediate>(encodedSource, options);
            const auto encodedDestination =
                detail::encodeScalarValueKnown<DestinationTag>(value, options);
            std::memcpy(destinationStorage.data() + static_cast<size_t>(destinationOffset) *
                                                        sizeof(typename DestinationTag::Storage),
                        &encodedDestination, sizeof(encodedDestination));
        } else {
            const Intermediate value =
                detail::decodeScalarKnown<SourceTag, Intermediate>(sourceStorage, sourceOffset);
            detail::encodeScalarKnown<DestinationTag>(destinationStorage, destinationOffset, value,
                                                      options);
        }
    };

    constexpr ScalarCategory sourceCategory = scalarTypeInfo(SourceTag::type).category;
    if constexpr (sourceCategory == ScalarCategory::Boolean ||
                  sourceCategory == ScalarCategory::UnsignedInteger) {
        convert.template operator()<uint64_t>();
    } else if constexpr (sourceCategory == ScalarCategory::SignedInteger) {
        convert.template operator()<int64_t>();
    } else if constexpr (sourceCategory == ScalarCategory::Complex) {
        convert.template operator()<std::complex<double>>();
    } else {
        convert.template operator()<double>();
    }
}

using CopyConvertedElementFunction = void (*)(std::span<const std::byte>, ptrdiff_t,
                                              std::span<std::byte>, ptrdiff_t,
                                              const ScalarConversionOptions&);

CopyConvertedElementFunction copyConvertedElementFunction(ScalarType sourceType,
                                                          ScalarType destinationType) {
    return visitScalarType(
        sourceType, [destinationType]<typename SourceTag>() -> CopyConvertedElementFunction {
            return visitScalarType(destinationType,
                                   []<typename DestinationTag>() -> CopyConvertedElementFunction {
                                       return &copyConvertedElement<SourceTag, DestinationTag>;
                                   });
        });
}
}  // namespace

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
    const ScalarType sourceType = type();
    const Layout& sourceLayout = layout();
    const std::span<const std::byte> sourceStorage = rawEncodedBackingStorage();
    if (destinationLayout.shape() != shape())
        throw std::invalid_argument(
            "Tensor conversion requires matching source and output shapes.");
    if (destinationType == sourceType && destinationLayout == sourceLayout) return deepCopy();
    if (destinationLayout != sourceLayout &&
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
    if (destinationType == sourceType) {
        result.copyLogicalElementsFrom(*this);
        return result;
    }
    const Tensor destination = result;
    const auto copyElement = copyConvertedElementFunction(sourceType, destinationType);
    const auto destinationStorage = destination.rawEncodedBackingStorage();
    const bool firstDimensionFastest =
        detail::isContiguous(sourceLayout, IndexOrder::FirstDimensionFastest) &&
        detail::isContiguous(destination.layout(), IndexOrder::FirstDimensionFastest);
    const bool lastDimensionFastest =
        detail::isContiguous(sourceLayout, IndexOrder::LastDimensionFastest) &&
        detail::isContiguous(destination.layout(), IndexOrder::LastDimensionFastest);
    if (firstDimensionFastest || lastDimensionFastest) {
        const ptrdiff_t sourceBase = sourceLayout.offset();
        const ptrdiff_t destinationBase = destination.layout().offset();
        for (size_t index = 0; index < sourceLayout.shape().elementCount(); ++index)
            copyElement(sourceStorage, sourceBase + static_cast<ptrdiff_t>(index),
                        destinationStorage, destinationBase + static_cast<ptrdiff_t>(index),
                        options);
        return result;
    }

    const bool sourceFirstDimensionFastest =
        detail::isContiguous(sourceLayout, IndexOrder::FirstDimensionFastest);
    const bool sourceLastDimensionFastest =
        detail::isContiguous(sourceLayout, IndexOrder::LastDimensionFastest);
    const bool destinationFirstDimensionFastest =
        detail::isContiguous(destination.layout(), IndexOrder::FirstDimensionFastest);
    const bool destinationLastDimensionFastest =
        detail::isContiguous(destination.layout(), IndexOrder::LastDimensionFastest);
    if (sourceLayout.shape().rank() == 2 &&
        ((sourceFirstDimensionFastest && destinationLastDimensionFastest) ||
         (sourceLastDimensionFastest && destinationFirstDimensionFastest))) {
        constexpr size_t tileRows = 64;
        constexpr size_t tileColumns = 64;
        const size_t rows = sourceLayout.shape()[0];
        const size_t columns = sourceLayout.shape()[1];
        const auto copyAt = [&](size_t row, size_t column) {
            copyElement(sourceStorage,
                        sourceLayout.offset() +
                            static_cast<ptrdiff_t>(row) * sourceLayout.stride(0) +
                            static_cast<ptrdiff_t>(column) * sourceLayout.stride(1),
                        destinationStorage,
                        destination.layout().offset() +
                            static_cast<ptrdiff_t>(row) * destination.layout().stride(0) +
                            static_cast<ptrdiff_t>(column) * destination.layout().stride(1),
                        options);
        };
        for (size_t rowBase = 0; rowBase < rows; rowBase += tileRows) {
            const size_t rowEnd = std::min(rows, rowBase + tileRows);
            for (size_t columnBase = 0; columnBase < columns; columnBase += tileColumns) {
                const size_t columnEnd = std::min(columns, columnBase + tileColumns);
                if (sourceFirstDimensionFastest) {
                    for (size_t column = columnBase; column < columnEnd; ++column)
                        for (size_t row = rowBase; row < rowEnd; ++row) copyAt(row, column);
                } else {
                    for (size_t row = rowBase; row < rowEnd; ++row)
                        for (size_t column = columnBase; column < columnEnd; ++column)
                            copyAt(row, column);
                }
            }
        }
        return result;
    }

    const IndexOrder traversalOrder =
        sourceFirstDimensionFastest        ? IndexOrder::FirstDimensionFastest
        : sourceLastDimensionFastest       ? IndexOrder::LastDimensionFastest
        : destinationFirstDimensionFastest ? IndexOrder::FirstDimensionFastest
                                           : IndexOrder::LastDimensionFastest;
    detail::forEachLayoutOffsetPairRange(
        sourceLayout, destination.layout(), traversalOrder, 0, sourceLayout.shape().elementCount(),
        [&](size_t, ptrdiff_t sourceOffset, ptrdiff_t destinationOffset) {
            copyElement(sourceStorage, sourceOffset, destinationStorage, destinationOffset,
                        options);
        });
    return result;
}
}  // namespace roc::host_numerics
