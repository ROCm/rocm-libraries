// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <roc/host_numerics/index_order.hpp>
#include <roc/host_numerics/scalar.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::host_numerics {
class Shape {
   public:
    Shape() = default;

    explicit Shape(std::vector<size_t> dimensions) : m_dimensions(std::move(dimensions)) {}

    Shape(std::initializer_list<size_t> dimensions) : m_dimensions(dimensions) {}

    explicit Shape(std::span<const size_t> dimensions)
        : m_dimensions(dimensions.begin(), dimensions.end()) {}

    size_t rank() const {
        return m_dimensions.size();
    }

    size_t operator[](size_t dimension) const {
        return extent(dimension);
    }

    size_t extent(size_t dimension) const {
        return m_dimensions.at(dimension);
    }

    std::span<const size_t> dimensions() const {
        return m_dimensions;
    }

    size_t elementCount() const {
        return elementCount(0, rank());
    }

    size_t elementCount(size_t firstDimension, size_t onePastLastDimension) const {
        if (firstDimension > onePastLastDimension || onePastLastDimension > rank())
            throw std::out_of_range("Tensor shape dimension range is invalid.");

        size_t count = 1;
        for (size_t dimension = firstDimension; dimension < onePastLastDimension; ++dimension)
            count = checkedElementProduct(count, extent(dimension));
        return count;
    }

    size_t elementCountExcluding(size_t excludedDimension) const {
        if (excludedDimension >= rank())
            throw std::out_of_range("Excluded tensor shape dimension is invalid.");

        size_t count = 1;
        for (size_t dimension = 0; dimension < rank(); ++dimension) {
            if (dimension != excludedDimension)
                count = checkedElementProduct(count, extent(dimension));
        }
        return count;
    }

    // Converts coordinates to a logical linear index in the selected order.
    // Throws std::invalid_argument for a rank mismatch and std::out_of_range
    // for any coordinate outside the shape.
    size_t linearIndex(std::span<const size_t> indices, IndexOrder order) const {
        if (indices.size() != rank())
            throw std::invalid_argument("Tensor coordinate rank does not match shape.");

        size_t result = 0;
        const auto appendDimension = [&](size_t dimension) {
            if (indices[dimension] >= extent(dimension))
                throw std::out_of_range("Tensor coordinate exceeds shape.");
            result = checkedElementProduct(result, extent(dimension));
            if (indices[dimension] > std::numeric_limits<size_t>::max() - result)
                throw std::overflow_error("Tensor linear index overflow.");
            result += indices[dimension];
        };

        if (order == IndexOrder::LastDimensionFastest) {
            for (size_t dimension = 0; dimension < rank(); ++dimension) appendDimension(dimension);
        } else {
            for (size_t dimension = rank(); dimension > 0; --dimension)
                appendDimension(dimension - 1);
        }
        return result;
    }

    // Writes the coordinates for one logical linear index. Throws
    // std::invalid_argument for a rank mismatch and std::out_of_range when
    // linearIndex is outside [0, elementCount()).
    void coordinates(size_t linearIndex, IndexOrder order, std::span<size_t> result) const {
        if (result.size() != rank())
            throw std::invalid_argument("Tensor coordinate output rank does not match shape.");
        if (linearIndex >= elementCount())
            throw std::out_of_range("Tensor logical index exceeds shape.");

        if (order == IndexOrder::FirstDimensionFastest) {
            for (size_t dimension = 0; dimension < rank(); ++dimension) {
                result[dimension] = linearIndex % extent(dimension);
                linearIndex /= extent(dimension);
            }
        } else {
            for (size_t dimension = rank(); dimension > 0; --dimension) {
                const size_t index = dimension - 1;
                result[index] = linearIndex % extent(index);
                linearIndex /= extent(index);
            }
        }
    }

    std::vector<size_t> coordinates(size_t linearIndex, IndexOrder order) const {
        std::vector<size_t> result(rank(), 0);
        coordinates(linearIndex, order, result);
        return result;
    }

    friend bool operator==(const Shape&, const Shape&) = default;

   private:
    static size_t checkedElementProduct(size_t count, size_t extent) {
        if (extent == 0) return 0;
        if (count > std::numeric_limits<size_t>::max() / extent)
            throw std::overflow_error("Tensor shape element count overflow.");
        return count * extent;
    }

    std::vector<size_t> m_dimensions;
};

class Layout;

namespace detail {
inline std::pair<ptrdiff_t, ptrdiff_t> elementBounds(const Layout& layout);
}

class Layout {
   public:
    Layout() = default;

    Layout(Shape shape, std::vector<ptrdiff_t> strides, ptrdiff_t offset = 0)
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset) {
        if (m_shape.rank() != m_strides.size())
            throw std::invalid_argument("Tensor layout rank and stride count differ.");
    }

    // For a matrix this is row-major/C-order: the last dimension has unit stride.
    static Layout contiguousLastDimensionFastest(const Shape& shape) {
        std::vector<ptrdiff_t> strides(shape.rank(), 1);
        ptrdiff_t stride = 1;
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            strides[index] = stride;
            stride = checkedContiguousStride(stride, shape.extent(index));
        }
        return Layout(shape, std::move(strides));
    }

    // For a matrix this is column-major/Fortran-order: the first dimension has unit stride.
    static Layout contiguousFirstDimensionFastest(const Shape& shape) {
        std::vector<ptrdiff_t> strides(shape.rank(), 1);
        ptrdiff_t stride = 1;
        for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
            strides[dimension] = stride;
            stride = checkedContiguousStride(stride, shape.extent(dimension));
        }
        return Layout(shape, std::move(strides));
    }

    const Shape& shape() const {
        return m_shape;
    }

    std::span<const ptrdiff_t> strides() const {
        return m_strides;
    }

    ptrdiff_t stride(size_t dimension) const {
        return m_strides.at(dimension);
    }

    ptrdiff_t offset() const {
        return m_offset;
    }

    ptrdiff_t elementOffset(std::span<const size_t> indices) const {
        if (indices.size() != m_shape.rank())
            throw std::invalid_argument("Tensor index rank does not match layout rank.");

        ptrdiff_t result = m_offset;
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            if (indices[dimension] >= m_shape.extent(dimension))
                throw std::out_of_range("Tensor index exceeds shape.");
            const ptrdiff_t delta = checkedMultiply(indices[dimension], stride(dimension));
            result = checkedAdd(result, delta);
        }
        return result;
    }

    friend bool operator==(const Layout&, const Layout&) = default;

   private:
    static ptrdiff_t checkedContiguousStride(ptrdiff_t stride, size_t extent) {
        if (extent > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("Tensor extent exceeds ptrdiff_t.");
        const ptrdiff_t signedExtent = static_cast<ptrdiff_t>(extent);
        if (signedExtent != 0 && stride > std::numeric_limits<ptrdiff_t>::max() / signedExtent)
            throw std::overflow_error("Tensor contiguous stride overflow.");
        return stride * signedExtent;
    }

    static ptrdiff_t checkedMultiply(size_t value, ptrdiff_t factor) {
        if (value == 0 || factor == 0) return 0;

        const bool negative = factor < 0;
        const uintmax_t factorMagnitude =
            negative ? static_cast<uintmax_t>(-(factor + 1)) + 1 : static_cast<uintmax_t>(factor);
        const uintmax_t limit =
            negative ? static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max()) + 1
                     : static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max());
        if (static_cast<uintmax_t>(value) > limit / factorMagnitude)
            throw std::overflow_error("Tensor layout offset multiplication overflow.");

        const uintmax_t magnitude = static_cast<uintmax_t>(value) * factorMagnitude;
        if (!negative) return static_cast<ptrdiff_t>(magnitude);
        if (magnitude == limit) return std::numeric_limits<ptrdiff_t>::min();
        return -static_cast<ptrdiff_t>(magnitude);
    }

    static ptrdiff_t checkedAdd(ptrdiff_t left, ptrdiff_t right) {
        if ((right > 0 && left > std::numeric_limits<ptrdiff_t>::max() - right) ||
            (right < 0 && left < std::numeric_limits<ptrdiff_t>::min() - right))
            throw std::overflow_error("Tensor layout offset addition overflow.");
        return left + right;
    }

    std::pair<ptrdiff_t, ptrdiff_t> checkedElementBounds() const {
        for (size_t dimension = 0; dimension < m_shape.rank(); ++dimension) {
            if (m_shape.extent(dimension) == 0) return {0, -1};
        }

        ptrdiff_t lower = m_offset;
        ptrdiff_t upper = m_offset;
        for (size_t dimension = 0; dimension < m_shape.rank(); ++dimension) {
            const ptrdiff_t delta =
                checkedMultiply(m_shape.extent(dimension) - 1, stride(dimension));
            if (delta < 0)
                lower = checkedAdd(lower, delta);
            else
                upper = checkedAdd(upper, delta);
        }
        return {lower, upper};
    }

    friend std::pair<ptrdiff_t, ptrdiff_t> detail::elementBounds(const Layout& layout);

    Shape m_shape;
    std::vector<ptrdiff_t> m_strides;
    ptrdiff_t m_offset = 0;
};

namespace detail {
inline std::pair<ptrdiff_t, ptrdiff_t> elementBounds(const Layout& layout) {
    return layout.checkedElementBounds();
}

inline uint64_t strideMagnitude(ptrdiff_t stride) {
    if (stride >= 0) return static_cast<uint64_t>(stride);
    return static_cast<uint64_t>(-(stride + 1)) + 1;
}

inline bool hasProvablyDistinctElementOffsets(const Layout& layout) {
    if (layout.shape().elementCount() == 0) return true;

    std::vector<std::pair<uint64_t, size_t>> dimensions;
    dimensions.reserve(layout.shape().rank());
    for (size_t dimension = 0; dimension < layout.shape().rank(); ++dimension) {
        const size_t extent = layout.shape()[dimension];
        if (extent <= 1) continue;

        const uint64_t stride = strideMagnitude(layout.strides()[dimension]);
        if (stride == 0) return false;
        dimensions.emplace_back(stride, extent);
    }
    std::ranges::sort(dimensions);

    uint64_t addressedSpan = 1;
    for (const auto& [stride, extent] : dimensions) {
        if (stride < addressedSpan) return false;
        const uint64_t additionalExtent = static_cast<uint64_t>(extent - 1);
        if (additionalExtent > (std::numeric_limits<uint64_t>::max() - addressedSpan) / stride)
            return false;
        addressedSpan += additionalExtent * stride;
    }
    return true;
}

template <typename Function>
void forEachIndex(const Shape& shape, Function&& function) {
    const size_t count = shape.elementCount();
    std::vector<size_t> indices(shape.rank(), 0);
    for (size_t linearIndex = 0; linearIndex < count; ++linearIndex) {
        function(std::span<const size_t>(indices), linearIndex);
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            if (++indices[index] < shape.extent(index)) break;
            indices[index] = 0;
        }
    }
}

inline size_t storageBytesForLayout(ScalarType type, const Layout& layout) {
    const auto [lower, upper] = elementBounds(layout);
    if (upper < lower) return 0;
    if (lower < 0) throw std::invalid_argument("Tensor layout addresses before the storage base.");

    const uint64_t bits = scalarTypeInfo(type).storageBits;
    const uint64_t elementCount = static_cast<uint64_t>(upper) + 1;
    if (elementCount > std::numeric_limits<uint64_t>::max() / bits)
        throw std::overflow_error("Tensor storage size overflow.");
    const uint64_t totalBits = elementCount * bits;
    const uint64_t bytes = totalBits / 8 + static_cast<uint64_t>(totalBits % 8 != 0);
    if (bytes > std::numeric_limits<size_t>::max())
        throw std::overflow_error("Tensor storage byte count overflow.");
    return static_cast<size_t>(bytes);
}
}  // namespace detail

inline size_t storageBytesForLayout(ScalarType type, const Layout& layout) {
    return detail::storageBytesForLayout(type, layout);
}

namespace detail {
inline bool byteRangesOverlap(std::span<const std::byte> left, std::span<const std::byte> right) {
    if (left.empty() || right.empty()) return false;
    const auto less = std::less<const std::byte*>{};
    return less(left.data(), right.data() + right.size()) &&
           less(right.data(), left.data() + left.size());
}
}  // namespace detail

// Runtime tensor handle consisting of a ScalarType, Layout, and owner-anchored
// storage. Copies share storage. deepCopy() duplicates the complete backing storage,
// while shareStorageWithLayout() applies another layout to the same bytes. Const
// Tensor handles retain shallow mutability, so writes are visible through every
// shared handle.
//
// Copy factories allocate owned storage. shareExternalMutableBackingStorage()
// retains a caller-supplied lifetime anchor and performs no copy.
class Tensor {
   public:
    Tensor(ScalarType type, Shape shape)
        : Tensor(type, Layout::contiguousLastDimensionFastest(shape)) {}

    Tensor(ScalarType type, Layout layout)
        : m_type(type),
          m_layout(std::move(layout)),
          m_storage(allocateZeroInitializedStorage(
              ::roc::host_numerics::storageBytesForLayout(m_type, m_layout))) {}

    // Allocates byte-addressable storage without initializing it. Every byte
    // that can be observed must be written before it is read or copied.
    static Tensor allocateUninitialized(ScalarType type, Layout layout) {
        if (scalarTypeInfo(type).isPacked())
            throw std::invalid_argument(
                "Uninitialized Tensor allocation requires a byte-addressable scalar type.");
        const size_t bytes = ::roc::host_numerics::storageBytesForLayout(type, layout);
        return Tensor(type, std::move(layout), allocateUninitializedStorage(bytes));
    }

    static Tensor allocateUninitialized(ScalarType type, Shape shape) {
        return allocateUninitialized(type, Layout::contiguousLastDimensionFastest(shape));
    }

    // Copies the complete encoded backing storage. The span may include
    // product-required padding beyond the elements addressed by layout.
    static Tensor copyEncodedBackingStorage(ScalarType type, Layout layout,
                                            std::span<const std::byte> storage) {
        const size_t required = ::roc::host_numerics::storageBytesForLayout(type, layout);
        if (storage.size() < required)
            throw std::invalid_argument("Encoded Tensor storage is too small for its layout.");
        return Tensor(type, std::move(layout),
                      storageFromVector(std::vector<std::byte>(storage.begin(), storage.end())));
    }

    // Takes ownership of complete encoded backing storage, including any
    // product-required padding beyond the elements addressed by layout.
    static Tensor takeOwnershipOfEncodedBackingStorage(ScalarType type, Layout layout,
                                                       std::vector<std::byte> storage) {
        const size_t required = ::roc::host_numerics::storageBytesForLayout(type, layout);
        if (storage.size() < required)
            throw std::invalid_argument("Encoded Tensor storage is too small for its layout.");
        return Tensor(type, std::move(layout), storageFromVector(std::move(storage)));
    }

    // Shares an external mutable byte range without copying it. lifetimeAnchor
    // must keep that range valid until every Tensor sharing it is destroyed.
    static Tensor shareExternalMutableBackingStorage(ScalarType type, Layout layout,
                                                     std::shared_ptr<void> lifetimeAnchor,
                                                     std::span<std::byte> storage) {
        return Tensor(type, std::move(layout), SharedStorage(std::move(lifetimeAnchor), storage));
    }

    template <typename Source>
    static Tensor copyValuesWithConversion(ScalarType type, Shape shape,
                                           std::span<const Source> values) {
        if (values.size() != shape.elementCount())
            throw std::invalid_argument("Tensor value count does not match shape.");
        Tensor result(type, shape);
        for (size_t index = 0; index < values.size(); ++index)
            detail::encodeScalar(type, result.rawEncodedBackingStorage(),
                                 static_cast<ptrdiff_t>(index), values[index]);
        return result;
    }

    template <typename Source>
    static Tensor copyValuesWithConversion(ScalarType type, Shape shape,
                                           std::span<const Source> values,
                                           const ScalarConversionOptions& options) {
        if (values.size() != shape.elementCount())
            throw std::invalid_argument("Tensor value count does not match shape.");
        Tensor result(type, shape);
        for (size_t index = 0; index < values.size(); ++index)
            detail::encodeScalar(type, result.rawEncodedBackingStorage(),
                                 static_cast<ptrdiff_t>(index), values[index], options);
        return result;
    }

    template <typename Source>
    static Tensor copyNativeValues(Shape shape, std::span<const Source> values) {
        return copyValuesWithConversion(nativeScalarType<Source>, std::move(shape), values);
    }

    template <typename Source>
    static Tensor copyNativeStorage(Layout layout, std::span<const Source> values) {
        constexpr ScalarType type = nativeScalarType<Source>;
        static_assert(scalarTypeInfo(type).storageBits == sizeof(Source) * 8,
                      "Native Tensor storage requires one scalar per C++ object.");
        const std::span<const std::byte> bytes = std::as_bytes(values);
        const size_t required = ::roc::host_numerics::storageBytesForLayout(type, layout);
        if (bytes.size() < required)
            throw std::invalid_argument("Native Tensor storage is too small for its layout.");
        return copyEncodedBackingStorage(type, std::move(layout), bytes);
    }

    template <typename Source>
    static Tensor copyNativeStorage(Layout layout, std::span<Source> values) {
        return copyNativeStorage(std::move(layout), std::span<const Source>(values));
    }

    template <typename Source>
    static Tensor copyNativeStorage(std::span<const Source> values) {
        return copyNativeStorage(Layout::contiguousLastDimensionFastest(Shape{values.size()}),
                                 values);
    }

    template <typename Source>
    static Tensor copyNativeStorage(std::span<Source> values) {
        return copyNativeStorage(std::span<const Source>(values));
    }

    ScalarType type() const {
        return m_type;
    }

    const Shape& shape() const {
        return m_layout.shape();
    }

    const Layout& layout() const {
        return m_layout;
    }

    size_t elementCount() const {
        return m_layout.shape().elementCount();
    }

    std::span<std::byte> rawEncodedBackingStorage() const {
        return m_storage.bytes;
    }

    template <typename Target>
    Target loadAs(std::span<const size_t> indices) const {
        return detail::decodeScalar<Target>(m_type, rawEncodedBackingStorage(),
                                            m_layout.elementOffset(indices));
    }

    template <typename Target>
    Target loadAs(std::span<const size_t> indices, const ScalarConversionOptions& options) const {
        return detail::decodeScalar<Target>(m_type, rawEncodedBackingStorage(),
                                            m_layout.elementOffset(indices), options);
    }

    template <typename Target>
    Target loadAs(std::initializer_list<size_t> indices) const {
        return loadAs<Target>(std::span<const size_t>(indices.begin(), indices.size()));
    }

    template <typename Target>
    Target loadAs(std::initializer_list<size_t> indices,
                  const ScalarConversionOptions& options) const {
        return loadAs<Target>(std::span<const size_t>(indices.begin(), indices.size()), options);
    }

    template <typename Source>
    void storeFrom(std::span<const size_t> indices, Source value) const {
        detail::encodeScalar(m_type, rawEncodedBackingStorage(), m_layout.elementOffset(indices),
                             value);
    }

    template <typename Source>
    void storeFrom(std::span<const size_t> indices, Source value,
                   const ScalarConversionOptions& options) const {
        detail::encodeScalar(m_type, rawEncodedBackingStorage(), m_layout.elementOffset(indices),
                             std::move(value), options);
    }

    template <typename Source>
    void storeFrom(std::initializer_list<size_t> indices, Source value) const {
        storeFrom(std::span<const size_t>(indices.begin(), indices.size()), value);
    }

    template <typename Source>
    void storeFrom(std::initializer_list<size_t> indices, Source value,
                   const ScalarConversionOptions& options) const {
        storeFrom(std::span<const size_t>(indices.begin(), indices.size()), std::move(value),
                  options);
    }

    Tensor shareStorageWithLayout(Layout layout) const {
        return Tensor(m_type, std::move(layout), m_storage);
    }

    Tensor deepCopy() const {
        return copyEncodedBackingStorage(m_type, m_layout, rawEncodedBackingStorage());
    }

    // Copies only logical tensor elements into their encoded destination
    // locations. Bytes and bits used only for layout gaps remain unchanged.
    void copyLogicalElementsToEncodedStorage(std::span<std::byte> destination) const {
        const size_t required = ::roc::host_numerics::storageBytesForLayout(m_type, m_layout);
        if (destination.size() < required)
            throw std::invalid_argument("Tensor copy destination storage is too small.");
        if (detail::byteRangesOverlap(rawEncodedBackingStorage(), destination)) {
            Tensor staged(m_type, m_layout);
            staged.copyLogicalElementsFrom(*this);
            staged.copyLogicalElementsToEncodedStorage(destination);
            return;
        }

        const uint16_t bits = scalarTypeInfo(m_type).storageBits;
        detail::forEachIndex(shape(), [&](std::span<const size_t> indices, size_t) {
            const uint64_t offset = detail::bitOffset(m_type, layout().elementOffset(indices));
            detail::copyBitRange(rawEncodedBackingStorage(), offset, destination, offset, bits);
        });
    }

    void copySelectedElementsToEncodedStorage(std::span<std::byte> destination,
                                              std::span<const size_t> linearIndices,
                                              IndexOrder indexOrder) const {
        const size_t required = ::roc::host_numerics::storageBytesForLayout(m_type, m_layout);
        if (destination.size() < required)
            throw std::invalid_argument("Tensor copy destination storage is too small.");
        if (detail::byteRangesOverlap(rawEncodedBackingStorage(), destination)) {
            Tensor staged(m_type, m_layout);
            copySelectedElementsToEncodedStorage(staged.rawEncodedBackingStorage(), linearIndices,
                                                 indexOrder);
            staged.copySelectedElementsToEncodedStorage(destination, linearIndices, indexOrder);
            return;
        }

        const uint16_t bits = scalarTypeInfo(m_type).storageBits;
        forEachLinearIndex(linearIndices, indexOrder, [&](std::span<const size_t> indices) {
            const uint64_t offset = detail::bitOffset(m_type, layout().elementOffset(indices));
            detail::copyBitRange(rawEncodedBackingStorage(), offset, destination, offset, bits);
        });
    }

    // Copies the selected logical elements into a contiguous rank-one Tensor
    // in selection order. Encoded bits are preserved exactly.
    Tensor copySelectedElements(std::span<const size_t> linearIndices,
                                IndexOrder indexOrder) const {
        Tensor result(m_type, Shape{linearIndices.size()});
        const uint16_t bits = scalarTypeInfo(m_type).storageBits;
        size_t destinationIndex = 0;
        forEachLinearIndex(linearIndices, indexOrder, [&](std::span<const size_t> indices) {
            detail::copyBitRange(rawEncodedBackingStorage(),
                                 detail::bitOffset(m_type, layout().elementOffset(indices)),
                                 result.rawEncodedBackingStorage(),
                                 detail::bitOffset(m_type, destinationIndex), bits);
            ++destinationIndex;
        });
        return result;
    }

    void copyLogicalElementsFrom(const Tensor& source) const {
        if (m_type != source.m_type)
            throw std::invalid_argument("Tensor copy requires matching scalar types.");
        if (shape() != source.shape())
            throw std::invalid_argument("Tensor copy requires matching shapes.");
        if (rawEncodedBackingStorage().data() == source.rawEncodedBackingStorage().data() &&
            layout() == source.layout())
            return;
        if (!detail::hasProvablyDistinctElementOffsets(layout()))
            throw std::invalid_argument(
                "Tensor copy requires non-overlapping destination elements.");
        if (detail::byteRangesOverlap(rawEncodedBackingStorage(),
                                      source.rawEncodedBackingStorage())) {
            Tensor staged(source.type(), source.layout());
            staged.copyLogicalElementsFrom(source);
            copyLogicalElementsFrom(staged);
            return;
        }
        const uint16_t bits = scalarTypeInfo(m_type).storageBits;
        detail::forEachIndex(shape(), [&](std::span<const size_t> indices, size_t) {
            detail::copyBitRange(source.rawEncodedBackingStorage(),
                                 detail::bitOffset(m_type, source.layout().elementOffset(indices)),
                                 rawEncodedBackingStorage(),
                                 detail::bitOffset(m_type, layout().elementOffset(indices)), bits);
        });
    }

    Tensor reshapeSharingStorage(Shape shape) const;

    // New storage bits are zero; existing logical elements retain their encodings.
    Tensor copyWithZeroPadding(Shape shape) const;

    // destinationToSource[d] names the source dimension copied to destination dimension d.
    Tensor copyWithPermutedDimensions(std::span<const size_t> destinationToSource) const;

    Tensor copyWithPermutedDimensions(std::initializer_list<size_t> destinationToSource) const {
        return copyWithPermutedDimensions(
            std::span<const size_t>(destinationToSource.begin(), destinationToSource.size()));
    }

    Tensor copyConvertedTo(ScalarType type) const;

    Tensor copyConvertedTo(ScalarType type, const ScalarConversionOptions& options) const;

    Tensor copyConvertedTo(ScalarType type, Layout layout) const;

    Tensor copyConvertedTo(ScalarType type, Layout layout,
                           const ScalarConversionOptions& options) const;

   private:
    struct SharedStorage {
        SharedStorage() = default;

        SharedStorage(std::shared_ptr<void> lifetimeAnchor, std::span<std::byte> bytes)
            : lifetimeAnchor(std::move(lifetimeAnchor)), bytes(bytes) {
            if (!this->lifetimeAnchor && !this->bytes.empty())
                throw std::invalid_argument("Nonempty Tensor storage requires a lifetime anchor.");
        }

        std::shared_ptr<void> lifetimeAnchor;
        std::span<std::byte> bytes;
    };

    Tensor(ScalarType type, Layout layout, SharedStorage storage)
        : m_type(type), m_layout(std::move(layout)), m_storage(std::move(storage)) {
        validateStorage();
    }

    static SharedStorage allocateZeroInitializedStorage(size_t bytes) {
        auto owner = std::make_shared<std::vector<std::byte>>(bytes);
        return SharedStorage(owner, std::span<std::byte>(*owner));
    }

    static SharedStorage allocateUninitializedStorage(size_t bytes) {
        std::shared_ptr<void> owner(new std::byte[bytes], std::default_delete<std::byte[]>());
        return SharedStorage(owner,
                             std::span<std::byte>(static_cast<std::byte*>(owner.get()), bytes));
    }

    static SharedStorage storageFromVector(std::vector<std::byte> storage) {
        auto owner = std::make_shared<std::vector<std::byte>>(std::move(storage));
        return SharedStorage(owner, std::span<std::byte>(*owner));
    }

    template <typename Function>
    void forEachLinearIndex(std::span<const size_t> linearIndices, IndexOrder indexOrder,
                            Function&& function) const {
        const size_t count = elementCount();
        std::vector<size_t> indices(shape().rank(), 0);
        for (const size_t linearIndex : linearIndices) {
            if (linearIndex >= count)
                throw std::out_of_range("Tensor copy index exceeds the logical element count.");
            shape().coordinates(linearIndex, indexOrder, indices);
            function(std::span<const size_t>(indices));
        }
    }

    void validateStorage() {
        const size_t required = ::roc::host_numerics::storageBytesForLayout(m_type, m_layout);
        if (m_storage.bytes.size() < required)
            throw std::invalid_argument("Tensor storage is too small for its layout.");
    }

    ScalarType m_type;
    Layout m_layout;
    SharedStorage m_storage;
};

inline Tensor Tensor::reshapeSharingStorage(Shape shape) const {
    if (shape.elementCount() != elementCount())
        throw std::invalid_argument("Tensor reshape requires the same logical element count.");
    if (layout() != Layout::contiguousLastDimensionFastest(this->shape()))
        throw std::invalid_argument(
            "Tensor reshape requires a contiguous last-dimension-fastest layout.");
    return shareStorageWithLayout(Layout::contiguousLastDimensionFastest(shape));
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
                    const uint64_t value = detail::decodeScalarKnown<SourceTag::type, uint64_t>(
                        sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag::type>(
                        destination.rawEncodedBackingStorage(), destinationOffset, value, options);
                } else if constexpr (sourceCategory == ScalarCategory::SignedInteger) {
                    const int64_t value = detail::decodeScalarKnown<SourceTag::type, int64_t>(
                        sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag::type>(
                        destination.rawEncodedBackingStorage(), destinationOffset, value, options);
                } else if constexpr (sourceCategory == ScalarCategory::Complex) {
                    const std::complex<double> value =
                        detail::decodeScalarKnown<SourceTag::type, std::complex<double>>(
                            sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag::type>(
                        destination.rawEncodedBackingStorage(), destinationOffset, value, options);
                } else {
                    const double value = detail::decodeScalarKnown<SourceTag::type, double>(
                        sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag::type>(
                        destination.rawEncodedBackingStorage(), destinationOffset, value, options);
                }
            });
        });
    });
    return result;
}
}  // namespace roc::host_numerics
