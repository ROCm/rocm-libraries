// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <roc/hostnumerics/periodic_initialization.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/index_iteration.hpp"
#include "detail/prepared_scalar_storage.hpp"

namespace roc::hostnumerics {
namespace {
Tensor snapshotSeed(const Tensor& seed, const Shape& shape) {
    if (seed.shape().rank() != shape.rank())
        throw std::runtime_error("Periodic seed and logical shape must have equal rank.");
    const size_t count = shape.elementCount();
    for (size_t d = 0; d < shape.rank(); ++d) {
        if (count != 0 && seed.shape()[d] == 0)
            throw std::runtime_error("A nonempty periodic tensor requires positive periods.");
    }
    Tensor snapshot(seed.type(), seed.shape());
    // An empty logical tensor does not read the seed, even if the seed is nonempty.
    if (count != 0) snapshot.copyLogicalElementsFrom(seed);
    return snapshot;
}
}  // namespace

PeriodicInitialization::PeriodicInitialization(const Tensor& seedTile, Shape logicalShape)
    : m_seed(snapshotSeed(seedTile, logicalShape)), m_shape(std::move(logicalShape)) {}

ScalarType PeriodicInitialization::type() const {
    return m_seed.type();
}

const Shape& PeriodicInitialization::shape() const {
    return m_shape;
}

const Shape& PeriodicInitialization::periods() const {
    return m_seed.shape();
}

size_t PeriodicInitialization::seedStorageBytes() const {
    return m_seed.rawEncodedBackingStorage().size();
}

Tensor PeriodicInitialization::copySeedTile() const {
    return m_seed.deepCopy();
}

void PeriodicInitialization::fillInto(Tensor destination,
                                      std::span<const size_t> logicalOrigin) const {
    if (destination.type() != type())
        throw std::runtime_error("Periodic initialization requires matching encoded types.");
    if (destination.shape().rank() != shape().rank() || logicalOrigin.size() != shape().rank())
        throw std::runtime_error("Periodic region rank does not match the logical shape.");
    for (size_t d = 0; d < shape().rank(); ++d) {
        if (logicalOrigin[d] > shape()[d] || destination.shape()[d] > shape()[d] - logicalOrigin[d])
            throw std::runtime_error("Periodic region exceeds the logical shape.");
    }
    if (!detail::hasProvablyDistinctElementOffsets(destination.layout()))
        throw std::runtime_error("Periodic destination must have distinct element offsets.");

    const uint16_t bits = scalarTypeInfo(type()).storageBits;
    const auto source = m_seed.rawEncodedBackingStorage();
    const auto target = destination.rawEncodedBackingStorage();
    std::vector<size_t> seedIndex(shape().rank());
    destination.shape().forEachCoordinateLastDimensionFastest(
        IndexRange::all(destination.shape().elementCount()),
        [&](size_t, std::span<const size_t> index) {
            for (size_t d = 0; d < index.size(); ++d)
                seedIndex[d] = (logicalOrigin[d] + index[d]) % periods().dimensions()[d];
            const auto src = detail::preparedBitOffset(
                type(), detail::preparedElementOffset(m_seed.layout(), seedIndex));
            const auto dst = detail::preparedBitOffset(
                type(), detail::preparedElementOffset(destination.layout(), index));
            if (bits % 8 == 0)
                std::memcpy(target.data() + dst / 8, source.data() + src / 8, bits / 8);
            else
                detail::preparedCopyBitRange(source, src, target, dst, bits);
        });
}

void PeriodicInitialization::fillInto(Tensor destination) const {
    fillInto(std::move(destination), std::vector<size_t>(shape().rank(), 0));
}
}  // namespace roc::hostnumerics
