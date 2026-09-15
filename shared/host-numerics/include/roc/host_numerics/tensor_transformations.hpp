// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_numerics/tensor.hpp>

namespace roc::host_numerics {
inline Tensor Tensor::reshapeSharingStorage(Shape shape) const {
    if (shape.elementCount() != elementCount())
        throw std::invalid_argument("Tensor reshape requires the same logical element count.");
    if (layout() != Layout::contiguousLastDimensionFastest(this->shape()))
        throw std::invalid_argument(
            "Tensor reshape requires a contiguous last-dimension-fastest layout.");
    return shareStorageWithLayout(Layout::contiguousLastDimensionFastest(shape));
}

inline Tensor Tensor::expandDims(size_t axis) const {
    if (axis > shape().rank())
        throw std::out_of_range("Tensor expansion axis exceeds the destination rank.");
    std::vector<size_t> dimensions(shape().dimensions().begin(), shape().dimensions().end());
    std::vector<ptrdiff_t> strides(layout().strides().begin(), layout().strides().end());
    dimensions.insert(dimensions.begin() + static_cast<ptrdiff_t>(axis), 1);
    strides.insert(strides.begin() + static_cast<ptrdiff_t>(axis), 0);
    return shareStorageWithLayout(
        Layout(Shape(std::move(dimensions)), std::move(strides), layout().offset()));
}

inline Tensor Tensor::broadcastTo(Shape shape) const {
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

inline Tensor Tensor::copyWithZeroPadding(Shape shape) const {
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

inline Tensor Tensor::copyWithPermutedDimensions(
    std::span<const size_t> destinationToSource) const {
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

inline Tensor Tensor::copyConvertedTo(ScalarType destinationType) const {
    return copyConvertedTo(destinationType, layout(),
                           detail::implicitStorageConversionOptions(destinationType));
}

inline Tensor Tensor::copyConvertedTo(ScalarType destinationType,
                                      const ScalarConversionOptions& options) const {
    return copyConvertedTo(destinationType, layout(), options);
}

inline Tensor Tensor::copyConvertedTo(ScalarType destinationType, Layout destinationLayout) const {
    return copyConvertedTo(destinationType, std::move(destinationLayout),
                           detail::implicitStorageConversionOptions(destinationType));
}

inline Tensor Tensor::copyConvertedTo(ScalarType destinationType, Layout destinationLayout,
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

    Tensor result(destinationType, std::move(destinationLayout));
    if (destinationType == sourceType) {
        result.copyLogicalElementsFrom(*this);
        return result;
    }
    const Tensor destination = result;
    visitScalarType(sourceType, [&]<typename SourceTag>() {
        visitScalarType(destinationType, [&]<typename DestinationTag>() {
            detail::forEachIndex(sourceLayout.shape(), [&](std::span<const size_t> indices,
                                                           size_t) {
                const ptrdiff_t sourceOffset = sourceLayout.elementOffset(indices);
                const ptrdiff_t destinationOffset = destination.layout().elementOffset(indices);
                constexpr ScalarCategory sourceCategory = scalarTypeInfo(SourceTag::type).category;
                if constexpr (sourceCategory == ScalarCategory::Boolean ||
                              sourceCategory == ScalarCategory::UnsignedInteger) {
                    const uint64_t value =
                        detail::decodeScalarKnown<SourceTag, uint64_t>(sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag>(
                        destination.rawEncodedBackingStorage(), destinationOffset, value, options);
                } else if constexpr (sourceCategory == ScalarCategory::SignedInteger) {
                    const int64_t value =
                        detail::decodeScalarKnown<SourceTag, int64_t>(sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag>(
                        destination.rawEncodedBackingStorage(), destinationOffset, value, options);
                } else if constexpr (sourceCategory == ScalarCategory::Complex) {
                    const std::complex<double> value =
                        detail::decodeScalarKnown<SourceTag, std::complex<double>>(sourceStorage,
                                                                                   sourceOffset);
                    detail::encodeScalarKnown<DestinationTag>(
                        destination.rawEncodedBackingStorage(), destinationOffset, value, options);
                } else {
                    const double value =
                        detail::decodeScalarKnown<SourceTag, double>(sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag>(
                        destination.rawEncodedBackingStorage(), destinationOffset, value, options);
                }
            });
        });
    });
    return result;
}

}  // namespace roc::host_numerics
