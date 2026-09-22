// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <functional>
#include <limits>
#include <roc/host_numerics/tensor.hpp>

#include "detail/tensor_conversion.hpp"
#include "detail/tensor_storage.hpp"

namespace roc::host_numerics {
namespace {
template <typename Function>
void forEachLinearIndex(const Tensor& tensor, std::span<const size_t> linearIndices,
                        IndexOrder indexOrder, Function&& function) {
    const size_t count = tensor.elementCount();
    std::vector<size_t> indices(tensor.shape().rank(), 0);
    for (const size_t linearIndex : linearIndices) {
        if (linearIndex >= count)
            throw std::out_of_range("Tensor copy index exceeds the logical element count.");
        tensor.shape().coordinates(linearIndex, indexOrder, indices);
        function(std::span<const size_t>(indices));
    }
}
}  // namespace

bool detail::byteRangesOverlap(std::span<const std::byte> left, std::span<const std::byte> right) {
    if (left.empty() || right.empty()) return false;
    const auto less = std::less<const std::byte*>{};
    return less(left.data(), right.data() + right.size()) &&
           less(right.data(), left.data() + left.size());
}

size_t storageBytesForLayout(ScalarType type, const Layout& layout) {
    const auto [lower, upper] = detail::elementBounds(layout);
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

Tensor::Tensor(ScalarType type, Shape shape)
    : Tensor(type, Layout::contiguousLastDimensionFastest(shape)) {}

Tensor::Tensor(ScalarType type, Layout layout)
    : m_type(type),
      m_layout(std::move(layout)),
      m_storage(allocateZeroInitializedStorage(storageBytesForLayout(m_type, m_layout))) {}

Tensor Tensor::allocateUninitialized(ScalarType type, Layout layout) {
    if (scalarTypeInfo(type).isPacked())
        throw std::invalid_argument(
            "Uninitialized Tensor allocation requires a byte-addressable scalar type.");
    const size_t bytes = storageBytesForLayout(type, layout);
    return Tensor(type, std::move(layout), allocateUninitializedStorage(bytes));
}

Tensor Tensor::allocateUninitialized(ScalarType type, Shape shape) {
    return allocateUninitialized(type, Layout::contiguousLastDimensionFastest(shape));
}

Tensor Tensor::copyEncodedBackingStorage(ScalarType type, Layout layout,
                                         std::span<const std::byte> storage) {
    const size_t required = storageBytesForLayout(type, layout);
    if (storage.size() < required)
        throw std::invalid_argument("Encoded Tensor storage is too small for its layout.");
    return Tensor(type, std::move(layout),
                  storageFromVector(std::vector<std::byte>(storage.begin(), storage.end())));
}

Tensor Tensor::takeOwnershipOfEncodedBackingStorage(ScalarType type, Layout layout,
                                                    std::vector<std::byte> storage) {
    const size_t required = storageBytesForLayout(type, layout);
    if (storage.size() < required)
        throw std::invalid_argument("Encoded Tensor storage is too small for its layout.");
    return Tensor(type, std::move(layout), storageFromVector(std::move(storage)));
}

Tensor Tensor::shareExternalMutableBackingStorage(ScalarType type, Layout layout,
                                                  std::shared_ptr<void> lifetimeAnchor,
                                                  std::span<std::byte> storage) {
    return Tensor(type, std::move(layout), SharedStorage(std::move(lifetimeAnchor), storage));
}

Tensor Tensor::shareStorageWithLayout(Layout layout) const {
    return Tensor(m_type, std::move(layout), m_storage);
}

Tensor Tensor::deepCopy() const {
    return copyEncodedBackingStorage(m_type, m_layout, rawEncodedBackingStorage());
}

void Tensor::copyLogicalElementsToEncodedStorage(std::span<std::byte> destination) const {
    const size_t required = storageBytesForLayout(m_type, m_layout);
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

void Tensor::copySelectedElementsToEncodedStorage(std::span<std::byte> destination,
                                                  std::span<const size_t> linearIndices,
                                                  IndexOrder indexOrder) const {
    const size_t required = storageBytesForLayout(m_type, m_layout);
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
    forEachLinearIndex(*this, linearIndices, indexOrder, [&](std::span<const size_t> indices) {
        const uint64_t offset = detail::bitOffset(m_type, layout().elementOffset(indices));
        detail::copyBitRange(rawEncodedBackingStorage(), offset, destination, offset, bits);
    });
}

Tensor Tensor::copySelectedElements(std::span<const size_t> linearIndices,
                                    IndexOrder indexOrder) const {
    Tensor result(m_type, Shape{linearIndices.size()});
    const uint16_t bits = scalarTypeInfo(m_type).storageBits;
    size_t destinationIndex = 0;
    forEachLinearIndex(*this, linearIndices, indexOrder, [&](std::span<const size_t> indices) {
        detail::copyBitRange(
            rawEncodedBackingStorage(), detail::bitOffset(m_type, layout().elementOffset(indices)),
            result.rawEncodedBackingStorage(), detail::bitOffset(m_type, destinationIndex), bits);
        ++destinationIndex;
    });
    return result;
}

void Tensor::copyLogicalElementsFrom(const Tensor& source) const {
    if (m_type != source.m_type)
        throw std::invalid_argument("Tensor copy requires matching scalar types.");
    if (shape() != source.shape())
        throw std::invalid_argument("Tensor copy requires matching shapes.");
    if (rawEncodedBackingStorage().data() == source.rawEncodedBackingStorage().data() &&
        layout() == source.layout())
        return;
    if (!detail::hasProvablyDistinctElementOffsets(layout()))
        throw std::invalid_argument("Tensor copy requires non-overlapping destination elements.");
    if (detail::byteRangesOverlap(rawEncodedBackingStorage(), source.rawEncodedBackingStorage())) {
        Tensor staged(source.type(), source.layout());
        staged.copyLogicalElementsFrom(source);
        copyLogicalElementsFrom(staged);
        return;
    }
    detail::TensorConversion(source, *this, detail::implicitStorageConversionOptions(m_type))
        .copyAll();
}

Tensor::SharedStorage::SharedStorage(std::shared_ptr<void> lifetimeAnchor,
                                     std::span<std::byte> bytes)
    : lifetimeAnchor(std::move(lifetimeAnchor)), bytes(bytes) {
    if (!this->lifetimeAnchor && !this->bytes.empty())
        throw std::invalid_argument("Nonempty Tensor storage requires a lifetime anchor.");
}

Tensor::Tensor(ScalarType type, Layout layout, SharedStorage storage)
    : m_type(type), m_layout(std::move(layout)), m_storage(std::move(storage)) {
    validateStorage();
}

Tensor::SharedStorage Tensor::allocateZeroInitializedStorage(size_t bytes) {
    auto owner = std::make_shared<std::vector<std::byte>>(bytes);
    return SharedStorage(owner, std::span<std::byte>(*owner));
}

Tensor::SharedStorage Tensor::allocateUninitializedStorage(size_t bytes) {
    std::shared_ptr<void> owner(new std::byte[bytes], std::default_delete<std::byte[]>());
    return SharedStorage(owner, std::span<std::byte>(static_cast<std::byte*>(owner.get()), bytes));
}

Tensor::SharedStorage Tensor::storageFromVector(std::vector<std::byte> storage) {
    auto owner = std::make_shared<std::vector<std::byte>>(std::move(storage));
    return SharedStorage(owner, std::span<std::byte>(*owner));
}

void Tensor::validateStorage() {
    const size_t required = storageBytesForLayout(m_type, m_layout);
    if (m_storage.bytes.size() < required)
        throw std::invalid_argument("Tensor storage is too small for its layout.");
}
}  // namespace roc::host_numerics
