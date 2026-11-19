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
// the quantization back.

namespace rpptest {

// Rpp16f is specialized because half_float::half only converts through float.

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

// M_PI is POSIX, not standard C++, so it is not guaranteed under a strict-conformance build.
constexpr double kPi = 3.14159265358979323846;

inline double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// [0,1] "unit intensity" conversions. to_unit maps a stored pixel into [0,1]; from_unit quantizes
// back, rounding integers to nearest. I8 pixels are the U8 intensities shifted by -128.
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

// Quantizes a value already in stored units: integers round to nearest and clamp, floats clamp
// to [0,1]. Unlike from_unit(), no [0,1] -> stored mapping is applied.
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
