// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/tensor.hpp>
#include <span>

namespace roc::hostnumerics {
// Repeats a private snapshot of an encoded seed tensor across a logical shape.
// The seed and logical shape have equal rank. Each logical coordinate i reads
// seed[i[d] % seed.shape()[d]] in every dimension. There are no numerical or
// GEMM restrictions; conversion, if wanted, precedes construction.
//
// Copies share the immutable snapshot. No returned Tensor aliases that snapshot.
class PeriodicInitialization {
   public:
    PeriodicInitialization(const Tensor& seedTile, Shape logicalShape);

    ScalarType type() const;
    const Shape& shape() const;
    const Shape& periods() const;
    size_t seedStorageBytes() const;

    // A new compact copy, useful for computing a reference from stored values.
    Tensor copySeedTile() const;

    // destination's coordinates start at logicalOrigin in the full definition;
    // its shape specifies the region extent. Layout gaps and unaddressed packed
    // bits are preserved. Requires provably distinct destination offsets.
    void fillInto(Tensor destination, std::span<const size_t> logicalOrigin) const;
    void fillInto(Tensor destination) const;

   private:
    Tensor m_seed;
    Shape m_shape;
};
}  // namespace roc::hostnumerics
