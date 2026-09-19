/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_TEST_INTENSITY_H
#define RPP_TEST_INTENSITY_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"

// Conversions between a stored pixel and the numeric spaces the golden models compute in, plus
// the quantization back. Every reference model goes through these, so the suite has one
// definition of what a U8/I8/F16/F32 pixel *means* numerically.

namespace rpptest {

// A generic path plus an explicit Rpp16f specialization, since half_float::half
// only exposes an operator float() and a float constructor.

template <typename T>
inline double to_double(const T& v) {
    return static_cast<double>(v);
}
template <>
inline double to_double<Rpp16f>(const Rpp16f& v) {
    return static_cast<double>(static_cast<float>(v));
}

template <typename T>
inline T from_double(double v) {
    return static_cast<T>(v);
}
template <>
inline Rpp16f from_double<Rpp16f>(double v) {
    return static_cast<Rpp16f>(static_cast<float>(v));
}

// Clamps v to [lo, hi]. Shared by the op reference models.
inline double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Normalized [0,1] "unit intensity" conversions, shared by the reference models that compute
// in unit space (hue, saturation, color_twist, gamma_correction). to_unit maps a stored pixel
// into [0,1]; from_unit quantizes a [0,1] result back to the stored dtype, rounding integers to
// nearest (the intended round-to-nearest behavior the golden models hold to, which several
// kernels do not). I8 pixels are the same intensities
// shifted by -128.
inline double to_unit(double v, DType dt) {
    return (dt == DType::U8) ? v / 255.0 : (dt == DType::I8) ? (v + 128.0) / 255.0 : v;
}
inline double from_unit(double x, DType dt) {
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint(x * 255.0), 0.0, 255.0);
        case DType::I8:
            return clampd(std::nearbyint(x * 255.0) - 128.0, -128.0, 127.0);
        default:
            return clampd(x, 0.0, 1.0);  // F16/F32
    }
}

// The dtype's "black" (zero intensity) in stored units: 0 for U8/F16/F32, -128 for I8. Geometric
// ops use this as the out-of-frame border fill.
inline double dtype_black(DType dt) {
    return dt == DType::I8 ? -128.0 : 0.0;
}

// Quantizes a value already expressed in stored units back into the dtype's storable range:
// integers round to nearest and clamp, floats clamp to [0,1]. Round-to-nearest is the intended
// integer behavior the golden models hold to, which several kernels do not (they truncate I8).
// Distinct from from_unit(), which additionally maps [0,1] -> stored.
inline double quantize_stored(double v, DType dt) {
    switch (dt) {
        case DType::U8:
            return clampd(std::nearbyint(v), 0.0, 255.0);
        case DType::I8:
            return clampd(std::nearbyint(v), -128.0, 127.0);
        default:
            return clampd(v, 0.0, 1.0);  // F16/F32
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_INTENSITY_H
