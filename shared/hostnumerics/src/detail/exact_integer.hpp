// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cmath>
#include <roc/hostnumerics/scalar.hpp>

#include "prepared_scalar_storage.hpp"

namespace roc::hostnumerics::detail {
inline bool isExactInteger(double value) {
    return std::isfinite(value) && std::trunc(value) == value && std::abs(value) < 0x1p61;
}

// Supported verification formats. Every integer strictly inside this interval
// is representable; a bound reaching either endpoint is deliberately refused.
inline double exactIntegerLimit(ScalarType type) {
    switch (type) {
        case ScalarType::Float32:
            return 0x1p24;
        case ScalarType::Float64:
            return 0x1p53;
        case ScalarType::Float16:
            return 0x1p11;
        case ScalarType::BFloat16:
            return 0x1p8;
        case ScalarType::Int32:
            return 0x1p31;
        case ScalarType::Int8:
            return 128;
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E4M3Fnuz:
            return 16;
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E5M2Fnuz:
            return 8;
        default:
            return 0;
    }
}

inline double exactObservationLimit(ScalarType type) {
    const double limit = exactIntegerLimit(type);
    // Both integer endpoints can be the result of saturation.
    return type == ScalarType::Int8 || type == ScalarType::Int32 ? limit - 1 : limit;
}

inline ScalarConversionOptions exactOutputConversion() {
    return {IntegerRounding::NearestEven, IntegerOverflow::Saturate};
}

class PreparedExactOutput {
   public:
    explicit PreparedExactOutput(ScalarType type)
        : m_convert(visitScalarType(type, []<typename Tag>() -> double (*)(double) {
              return +[](double value) {
                  std::array<std::byte, 16> bytes{};
                  preparedEncodeScalarKnown<Tag>(bytes, 0, value, exactOutputConversion());
                  return preparedDecodeScalarKnown<Tag, double>(bytes, 0);
              };
          })) {}
    double operator()(double value) const {
        return m_convert(value);
    }

   private:
    double (*m_convert)(double);
};

inline double roundExactOutput(double value, ScalarType type) {
    std::array<std::byte, 16> bytes{};
    encodeScalar(type, bytes, 0, value, exactOutputConversion());
    return decodeScalar<double>(type, bytes, 0);
}
}  // namespace roc::hostnumerics::detail
