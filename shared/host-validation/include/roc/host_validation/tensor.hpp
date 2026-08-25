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
#include <roc/host_validation/index_order.hpp>
#include <roc/host_validation/scalar.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::host_validation {
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

    bool empty() const {
        return elementCount() == 0;
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
            result = result * extent(dimension) + indices[dimension];
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
        if (rank() != m_strides.size())
            throw std::invalid_argument("Tensor layout rank and stride count differ.");
    }

    // Compatibility spelling for contiguousLastDimensionFastest(). For a matrix this is
    // row-major/C-order: the last dimension has unit stride.
    static Layout contiguous(const Shape& shape) {
        return contiguousLastDimensionFastest(shape);
    }

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

    size_t rank() const {
        return m_shape.rank();
    }

    size_t extent(size_t dimension) const {
        return m_shape.extent(dimension);
    }

    std::span<const size_t> dimensions() const {
        return m_shape.dimensions();
    }

    size_t elementCount() const {
        return m_shape.elementCount();
    }

    size_t elementCount(size_t firstDimension, size_t onePastLastDimension) const {
        return m_shape.elementCount(firstDimension, onePastLastDimension);
    }

    size_t elementCountExcluding(size_t excludedDimension) const {
        return m_shape.elementCountExcluding(excludedDimension);
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
        if (indices.size() != rank())
            throw std::invalid_argument("Tensor index rank does not match layout rank.");

        ptrdiff_t result = m_offset;
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            if (indices[dimension] >= extent(dimension))
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
        for (size_t dimension = 0; dimension < rank(); ++dimension) {
            if (extent(dimension) == 0) return {0, -1};
        }

        ptrdiff_t lower = m_offset;
        ptrdiff_t upper = m_offset;
        for (size_t dimension = 0; dimension < rank(); ++dimension) {
            const ptrdiff_t delta = checkedMultiply(extent(dimension) - 1, stride(dimension));
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

// Reference-counted lifetime owner plus a mutable byte range. Copies retain the
// same owner and byte range. A nonempty range requires an owner, so this type
// cannot represent unanchored borrowed memory.
class TensorStorage {
   public:
    TensorStorage() = default;

    static TensorStorage wrap(std::shared_ptr<void> owner, std::span<std::byte> bytes) {
        return TensorStorage(std::move(owner), bytes);
    }

    static TensorStorage allocate(size_t bytes) {
        auto owner = std::make_shared<std::vector<std::byte>>(bytes);
        return TensorStorage(owner, std::span<std::byte>(*owner));
    }

    // Callers must write each byte before reading it.
    static TensorStorage allocateUninitialized(size_t bytes) {
        std::shared_ptr<void> owner(new std::byte[bytes], std::default_delete<std::byte[]>());
        return TensorStorage(owner,
                             std::span<std::byte>(static_cast<std::byte*>(owner.get()), bytes));
    }

    std::span<const std::byte> bytes() const {
        return m_bytes;
    }

    std::span<std::byte> mutableBytes() const {
        return m_bytes;
    }

    size_t size() const {
        return m_bytes.size();
    }

   private:
    TensorStorage(std::shared_ptr<void> owner, std::span<std::byte> bytes)
        : m_owner(std::move(owner)), m_bytes(bytes) {
        if (!m_owner && !m_bytes.empty())
            throw std::invalid_argument("Nonempty TensorStorage requires an owner.");
    }

    std::shared_ptr<void> m_owner;
    std::span<std::byte> m_bytes;
};

using TensorStorageAllocator = std::function<TensorStorage(size_t)>;

// Runtime tensor handle consisting of a ScalarType, Layout, and owner-anchored
// storage. Copies share storage. clone() deep-copies the addressed storage and
// alias() applies another layout to the same bytes. Const Tensor handles retain
// shallow mutability, so writes are visible through every alias.
//
// Span and native-value constructors copy into owned storage. wrapStorage()
// retains the supplied TensorStorage owner and performs no copy.
class Tensor {
   public:
    Tensor(ScalarType type, Shape shape) : Tensor(type, Layout::contiguous(shape)) {}

    Tensor(ScalarType type, Layout layout)
        : Tensor(type, std::move(layout), TensorStorage::allocate) {}

    Tensor(ScalarType type, Shape shape, const TensorStorageAllocator& allocator)
        : Tensor(type, Layout::contiguous(shape), allocator) {}

    Tensor(ScalarType type, Layout layout, const TensorStorageAllocator& allocator)
        : m_type(type), m_layout(std::move(layout)) {
        if (!allocator) throw std::invalid_argument("Tensor storage allocator is empty.");
        m_storage = allocator(::roc::host_validation::storageBytesForLayout(m_type, m_layout));
        validateStorage();
    }

    Tensor(ScalarType type, Layout layout, std::vector<std::byte> storage)
        : Tensor(type, std::move(layout), storageFromVector(std::move(storage))) {}

    Tensor(ScalarType type, Layout layout, std::span<const std::byte> storage)
        : Tensor(type, std::move(layout), std::vector<std::byte>(storage.begin(), storage.end())) {}

    Tensor(ScalarType type, Layout layout, std::span<std::byte> storage)
        : Tensor(type, std::move(layout), std::span<const std::byte>(storage)) {}

    static Tensor wrapStorage(ScalarType type, Layout layout, TensorStorage storage) {
        return Tensor(type, std::move(layout), std::move(storage));
    }

    static Tensor fromStorage(ScalarType type, Layout layout, std::vector<std::byte> storage) {
        return Tensor(type, std::move(layout), std::move(storage));
    }

    template <typename Source>
    static Tensor fromValues(ScalarType type, Shape shape, std::span<const Source> values) {
        if (values.size() != shape.elementCount())
            throw std::invalid_argument("Tensor value count does not match shape.");
        Tensor result(type, shape);
        for (size_t index = 0; index < values.size(); ++index)
            detail::encodeScalar(type, result.storage(), static_cast<ptrdiff_t>(index),
                                 values[index]);
        return result;
    }

    template <typename Source>
    static Tensor fromValues(ScalarType type, Shape shape, std::span<const Source> values,
                             const ScalarConversionOptions& options) {
        if (values.size() != shape.elementCount())
            throw std::invalid_argument("Tensor value count does not match shape.");
        Tensor result(type, shape);
        for (size_t index = 0; index < values.size(); ++index)
            detail::encodeScalar(type, result.storage(), static_cast<ptrdiff_t>(index),
                                 values[index], options);
        return result;
    }

    template <typename Source>
    static Tensor fromNativeValues(Shape shape, std::span<const Source> values) {
        return fromValues(nativeScalarType<Source>, std::move(shape), values);
    }

    template <typename Source>
    static Tensor fromNative(Layout layout, std::span<const Source> values) {
        constexpr ScalarType type = nativeScalarType<Source>;
        static_assert(scalarTypeInfo(type).storageBits == sizeof(Source) * 8,
                      "Native Tensor storage requires one scalar per C++ object.");
        const std::span<const std::byte> bytes = std::as_bytes(values);
        const size_t required = ::roc::host_validation::storageBytesForLayout(type, layout);
        if (bytes.size() < required)
            throw std::invalid_argument("Native Tensor storage is too small for its layout.");
        return Tensor(type, std::move(layout), bytes.first(required));
    }

    template <typename Source>
    static Tensor fromNative(Layout layout, std::span<Source> values) {
        return fromNative(std::move(layout), std::span<const Source>(values));
    }

    template <typename Source>
    static Tensor fromNative(std::span<const Source> values) {
        return fromNative(Layout::contiguous(Shape{values.size()}), values);
    }

    template <typename Source>
    static Tensor fromNative(std::span<Source> values) {
        return fromNative(std::span<const Source>(values));
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

    size_t size() const {
        return m_layout.elementCount();
    }

    std::span<std::byte> storage() const {
        return m_storage.mutableBytes();
    }

    template <typename Target>
    Target loadAs(std::span<const size_t> indices) const {
        return detail::decodeScalar<Target>(m_type, storage(), m_layout.elementOffset(indices));
    }

    template <typename Target>
    Target loadAs(std::span<const size_t> indices, const ScalarConversionOptions& options) const {
        return detail::decodeScalar<Target>(m_type, storage(), m_layout.elementOffset(indices),
                                            options);
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
        detail::encodeScalar(m_type, storage(), m_layout.elementOffset(indices), value);
    }

    template <typename Source>
    void storeFrom(std::span<const size_t> indices, Source value,
                   const ScalarConversionOptions& options) const {
        detail::encodeScalar(m_type, storage(), m_layout.elementOffset(indices), std::move(value),
                             options);
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

    Tensor alias(Layout layout) const {
        return Tensor(m_type, std::move(layout), m_storage);
    }

    Tensor clone() const {
        return clone(TensorStorage::allocate);
    }

    Tensor clone(const TensorStorageAllocator& allocator) const {
        Tensor result(m_type, m_layout, allocator);
        const size_t required = ::roc::host_validation::storageBytesForLayout(m_type, m_layout);
        std::ranges::copy(storage().first(required), result.storage().begin());
        return result;
    }

    void copyTo(std::span<std::byte> destination) const {
        const size_t required = ::roc::host_validation::storageBytesForLayout(m_type, m_layout);
        if (destination.size() < required)
            throw std::invalid_argument("Tensor copy destination storage is too small.");
        std::ranges::copy(storage().first(required), destination.begin());
    }

    void copyTo(std::span<std::byte> destination, std::span<const size_t> linearIndices) const {
        const size_t required = ::roc::host_validation::storageBytesForLayout(m_type, m_layout);
        if (destination.size() < required)
            throw std::invalid_argument("Tensor copy destination storage is too small.");

        const uint16_t bits = scalarTypeInfo(m_type).storageBits;
        forEachLinearIndex(linearIndices, [&](std::span<const size_t> indices) {
            const uint64_t offset = detail::bitOffset(m_type, layout().elementOffset(indices));
            detail::copyBitRange(storage(), offset, destination, offset, bits);
        });
    }

    void copyFrom(const Tensor& source) const {
        if (m_type != source.m_type)
            throw std::invalid_argument("Tensor copy requires matching scalar types.");
        if (shape() != source.shape())
            throw std::invalid_argument("Tensor copy requires matching shapes.");
        const uint16_t bits = scalarTypeInfo(m_type).storageBits;
        detail::forEachIndex(shape(), [&](std::span<const size_t> indices, size_t) {
            detail::copyBitRange(
                source.storage(), detail::bitOffset(m_type, source.layout().elementOffset(indices)),
                storage(), detail::bitOffset(m_type, layout().elementOffset(indices)), bits);
        });
    }

    void copyFrom(const Tensor& source, std::span<const size_t> linearIndices) const {
        if (m_type != source.m_type)
            throw std::invalid_argument("Tensor copy requires matching scalar types.");
        if (shape() != source.shape())
            throw std::invalid_argument("Tensor copy requires matching shapes.");

        const uint16_t bits = scalarTypeInfo(m_type).storageBits;
        forEachLinearIndex(linearIndices, [&](std::span<const size_t> indices) {
            detail::copyBitRange(
                source.storage(), detail::bitOffset(m_type, source.layout().elementOffset(indices)),
                storage(), detail::bitOffset(m_type, layout().elementOffset(indices)), bits);
        });
    }

    Tensor reshape(Shape shape) const;

    // New storage bits are zero; existing logical elements retain their encodings.
    Tensor pad(Shape shape) const;

    // destinationToSource[d] names the source dimension copied to destination dimension d.
    Tensor permute(std::span<const size_t> destinationToSource) const;

    Tensor permute(std::initializer_list<size_t> destinationToSource) const {
        return permute(
            std::span<const size_t>(destinationToSource.begin(), destinationToSource.size()));
    }

    Tensor to(ScalarType type) const;

    Tensor to(ScalarType type, const ScalarConversionOptions& options) const;

   private:
    Tensor(ScalarType type, Layout layout, TensorStorage storage)
        : m_type(type), m_layout(std::move(layout)), m_storage(std::move(storage)) {
        validateStorage();
    }

    static TensorStorage storageFromVector(std::vector<std::byte> storage) {
        auto owner = std::make_shared<std::vector<std::byte>>(std::move(storage));
        return TensorStorage::wrap(owner, std::span<std::byte>(*owner));
    }

    template <typename Function>
    void forEachLinearIndex(std::span<const size_t> linearIndices, Function&& function) const {
        const size_t count = size();
        std::vector<size_t> indices(shape().rank(), 0);
        for (const size_t linearIndex : linearIndices) {
            if (linearIndex >= count)
                throw std::out_of_range("Tensor copy index exceeds the logical element count.");

            size_t remaining = linearIndex;
            for (size_t dimension = shape().rank(); dimension > 0; --dimension) {
                const size_t index = dimension - 1;
                indices[index] = remaining % shape().extent(index);
                remaining /= shape().extent(index);
            }
            function(std::span<const size_t>(indices));
        }
    }

    void validateStorage() const {
        if (m_storage.size() < ::roc::host_validation::storageBytesForLayout(m_type, m_layout))
            throw std::invalid_argument("Tensor storage is too small for its layout.");
    }

    ScalarType m_type;
    Layout m_layout;
    TensorStorage m_storage;
};

inline Tensor Tensor::reshape(Shape shape) const {
    if (shape.elementCount() != size())
        throw std::invalid_argument("Tensor reshape requires the same logical element count.");
    if (layout() != Layout::contiguousLastDimensionFastest(this->shape()))
        throw std::invalid_argument(
            "Tensor reshape requires a contiguous last-dimension-fastest layout.");
    return alias(Layout::contiguousLastDimensionFastest(shape));
}

inline Tensor Tensor::pad(Shape shape) const {
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
            storage(), detail::bitOffset(type(), layout().elementOffset(indices)), result.storage(),
            detail::bitOffset(type(), result.layout().elementOffset(indices)), bits);
    });
    return result;
}

inline Tensor Tensor::permute(std::span<const size_t> destinationToSource) const {
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
            storage(), detail::bitOffset(type(), layout().elementOffset(sourceIndices)),
            result.storage(),
            detail::bitOffset(type(), result.layout().elementOffset(destinationIndices)), bits);
    });
    return result;
}

inline Tensor Tensor::to(ScalarType destinationType) const {
    return to(destinationType, detail::implicitStorageConversionOptions(destinationType));
}

inline Tensor Tensor::to(ScalarType destinationType, const ScalarConversionOptions& options) const {
    const ScalarType sourceType = type();
    const Layout& sourceLayout = layout();
    const std::span<const std::byte> sourceStorage = storage();
    const size_t requiredStorage =
        ::roc::host_validation::storageBytesForLayout(sourceType, sourceLayout);
    if (destinationType == sourceType)
        return Tensor::fromStorage(
            destinationType, sourceLayout,
            std::vector<std::byte>(sourceStorage.begin(), sourceStorage.begin() + requiredStorage));

    Tensor result(destinationType, sourceLayout);
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
                        destination.storage(), destinationOffset, value, options);
                } else if constexpr (sourceCategory == ScalarCategory::SignedInteger) {
                    const int64_t value = detail::decodeScalarKnown<SourceTag::type, int64_t>(
                        sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag::type>(
                        destination.storage(), destinationOffset, value, options);
                } else if constexpr (sourceCategory == ScalarCategory::Complex) {
                    const std::complex<double> value =
                        detail::decodeScalarKnown<SourceTag::type, std::complex<double>>(
                            sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag::type>(
                        destination.storage(), destinationOffset, value, options);
                } else {
                    const double value = detail::decodeScalarKnown<SourceTag::type, double>(
                        sourceStorage, sourceOffset);
                    detail::encodeScalarKnown<DestinationTag::type>(
                        destination.storage(), destinationOffset, value, options);
                }
            });
        });
    });
    return result;
}
}  // namespace roc::host_validation
