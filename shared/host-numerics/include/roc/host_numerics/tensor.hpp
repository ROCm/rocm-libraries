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
#include <roc/host_numerics/layout.hpp>
#include <roc/host_numerics/scalar.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::host_numerics {
namespace detail {
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

inline bool byteRangesOverlap(std::span<const std::byte> left, std::span<const std::byte> right) {
    if (left.empty() || right.empty()) return false;
    const auto less = std::less<const std::byte*>{};
    return less(left.data(), right.data() + right.size()) &&
           less(right.data(), left.data() + left.size());
}
}  // namespace detail

inline size_t storageBytesForLayout(ScalarType type, const Layout& layout) {
    return detail::storageBytesForLayout(type, layout);
}

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
    template <typename Source>
        requires NativeScalar<Source>
    Tensor(Source value) : Tensor(nativeScalarType<std::remove_cvref_t<Source>>, Shape{}) {
        detail::encodeScalar(m_type, rawEncodedBackingStorage(), 0, std::move(value));
    }

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

    template <typename Source>
        requires NativeScalar<Source>
    static Tensor scalar(ScalarType type, Source value) {
        Tensor result(type, Shape{});
        detail::encodeScalar(type, result.rawEncodedBackingStorage(), 0, std::move(value));
        return result;
    }

    template <typename Source>
        requires NativeScalar<Source>
    static Tensor scalar(ScalarType type, Source value, const ScalarConversionOptions& options) {
        Tensor result(type, Shape{});
        detail::encodeScalar(type, result.rawEncodedBackingStorage(), 0, std::move(value), options);
        return result;
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

    template <typename Target>
    Target item() const {
        if (shape().rank() != 0)
            throw std::invalid_argument("Tensor item requires a rank-zero tensor.");
        return detail::decodeScalar<Target>(type(), rawEncodedBackingStorage(),
                                            layout().elementOffset(std::span<const size_t>{}));
    }

    template <typename Target>
    Target item(const ScalarConversionOptions& options) const {
        if (shape().rank() != 0)
            throw std::invalid_argument("Tensor item requires a rank-zero tensor.");
        return detail::decodeScalar<Target>(type(), rawEncodedBackingStorage(),
                                            layout().elementOffset(std::span<const size_t>{}),
                                            options);
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

    // Elementwise activation conveniences. The named free functions remain
    // available for callers that prefer functional composition.
    [[nodiscard]] Tensor absolute() const;
    [[nodiscard]] Tensor relu() const;
    [[nodiscard]] Tensor gelu() const;
    [[nodiscard]] Tensor sigmoid() const;
    [[nodiscard]] Tensor tanh(double inputScale = 1.0, double outputScale = 1.0) const;
    [[nodiscard]] Tensor silu() const;
    [[nodiscard]] Tensor swish(double beta = 1.0) const;
    [[nodiscard]] Tensor clip(double minimum, double maximum) const;

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

    // Inserts a singleton dimension without copying storage.
    Tensor expandDims(size_t axis) const;

    // Returns a shallow NumPy-style broadcast view. Missing leading
    // dimensions and expanded singleton dimensions receive stride zero.
    Tensor broadcastTo(Shape shape) const;

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
}  // namespace roc::host_numerics

#include <roc/host_numerics/tensor_transformations.hpp>
