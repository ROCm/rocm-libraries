//===-- rocwmma_fp4.hpp - FP4 Type Definitions for RDNA4 SWMMAC --*- C++ -*-===//
//
// E2M1, E3M0, Q16 numerical format type definitions for use with
// the rocWMMA SWMMAC backend on RDNA4 (gfx1200/gfx1201).
//
//===----------------------------------------------------------------------===//
//
// This header provides type-level support for FP4 formats used by
// v_swmmac instructions. These are pure type definitions and conversion
// utilities — no hardware simulation, no backend emulation.
//
//===----------------------------------------------------------------------===//

#ifndef ROCWMMA_FP4_HPP
#define ROCWMMA_FP4_HPP

#include <cstdint>
#include <type_traits>

namespace rocwmma {
namespace fp4 {

// --- E2M1 Format (1 sign, 2 exponent, 1 mantissa) ---

struct e2m1_t {
    uint8_t data : 4;
    constexpr e2m1_t() : data(0) {}
    constexpr explicit e2m1_t(uint8_t v) : data(v & 0xF) {}
    constexpr operator uint8_t() const { return data; }
};

// --- E3M0 Format (1 sign, 3 exponent, 0 mantissa) ---

struct e3m0_t {
    uint8_t data : 4;
    constexpr e3m0_t() : data(0) {}
    constexpr explicit e3m0_t(uint8_t v) : data(v & 0xF) {}
    constexpr operator uint8_t() const { return data; }
};

// --- UE8M0 Block Scale (8-bit unsigned power-of-2 exponent) ---

struct ue8m0_t {
    uint8_t data;
    constexpr ue8m0_t() : data(0) {}
    constexpr explicit ue8m0_t(uint8_t v) : data(v) {}
    constexpr operator uint8_t() const { return data; }
};

// --- Packing ratios for WMMA fragment types ---

template <typename T>
struct pack_ratio;

template <> struct pack_ratio<e2m1_t> { static constexpr int value = 8; };
template <> struct pack_ratio<e3m0_t> { static constexpr int value = 8; };

// --- Float conversion (host-side only, not device) ---

inline float e2m1_to_float(e2m1_t v) {
    if (v.data == 0) return 0.0f;
    int sign = (v.data >> 3) & 1;
    int exp  = (v.data >> 1) & 3;
    int mant = v.data & 1;
    float val = (1.0f + mant * 0.5f) * (1 << exp);
    return sign ? -val : val;
}

inline float e3m0_to_float(e3m0_t v) {
    if (v.data == 0) return 0.0f;
    int sign = (v.data >> 3) & 1;
    int exp  = v.data & 7;
    float val = (float)(1 << exp);
    return sign ? -val : val;
}

inline float ue8m0_scale(ue8m0_t v) {
    return (float)(1 << v.data);
}

// --- Traits for fragment types ---

template <typename T>
struct is_fp4 : std::false_type {};
template <> struct is_fp4<e2m1_t> : std::true_type {};
template <> struct is_fp4<e3m0_t> : std::true_type {};

template <typename T>
inline constexpr bool is_fp4_v = is_fp4<T>::value;

} // namespace fp4
} // namespace rocwmma

#endif // ROCWMMA_FP4_HPP
