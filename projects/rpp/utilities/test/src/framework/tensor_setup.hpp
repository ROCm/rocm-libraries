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

#ifndef RPP_TEST_TENSOR_SETUP_H
#define RPP_TEST_TENSOR_SETUP_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"

namespace rpptest {

// ---- dtype traits ---------------------------------------------------------

inline RpptDataType to_rpp_dtype(DType d) {
    switch (d) {
        case DType::U8: return U8;
        case DType::F16: return F16;
        case DType::F32: return F32;
        case DType::I8: return I8;
        case DType::I16: return I16;
    }
    return U8;
}

inline std::size_t dtype_size(DType d) {
    switch (d) {
        case DType::U8:
        case DType::I8: return 1;
        case DType::F16:
        case DType::I16: return 2;
        case DType::F32: return 4;
    }
    return 1;
}

inline int channels_of(Layout l) { return l == Layout::PLN1 ? 1 : 3; }

inline RpptLayout to_rpp_layout(Layout l) { return l == Layout::PKD3 ? NHWC : NCHW; }

// ---- element conversions (used by the reference model and the comparator) --
//
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
        case DType::U8: return clampd(std::nearbyint(x * 255.0), 0.0, 255.0);
        case DType::I8: return clampd(std::nearbyint(x * 255.0) - 128.0, -128.0, 127.0);
        default:        return clampd(x, 0.0, 1.0);  // F16/F32
    }
}

// The dtype's "black" (zero intensity) in stored units: 0 for U8/F16/F32, -128 for I8. Geometric
// ops use this as the out-of-frame border fill.
inline double dtype_black(DType dt) { return dt == DType::I8 ? -128.0 : 0.0; }

// Quantizes a value already expressed in stored units back into the dtype's storable range:
// integers round to nearest and clamp, floats clamp to [0,1]. Round-to-nearest is the intended
// integer behavior the golden models hold to, which several kernels do not (they truncate I8).
// Distinct from from_unit(), which additionally maps [0,1] -> stored.
inline double quantize_stored(double v, DType dt) {
    switch (dt) {
        case DType::U8: return clampd(std::nearbyint(v), 0.0, 255.0);
        case DType::I8: return clampd(std::nearbyint(v), -128.0, 127.0);
        default:        return clampd(v, 0.0, 1.0);  // F16/F32
    }
}

// ---- descriptor / ROI construction ----------------------------------------

struct TensorShape {
    Rpp32u n, c, h, w;
};

// RPP's tensor calling convention pads the row width to (w/8)*8+8 elements, so kernels
// that process a full SIMD vector on the row tail read/write into that slack instead of
// overrunning the buffer (and, for a batch, the next image). The official legacy harness
// applies this padding to every op; some kernels tolerate a tight width but others (e.g.
// color_temperature) corrupt memory without it. d.w stays the logical width (so ROIs and
// the reference walk the real image); only the strides carry the padded row stride.
inline Rpp32u padded_width(Rpp32u w) { return (w / 8) * 8 + 8; }

// Builds a 4D descriptor for the given layout, with the padded row stride RPP expects. pad=false
// gives the densely packed strides instead: no op is run that way, but the goldens must agree
// logically under either convention (src/tests/core/golden_layout_test.cpp guards that).
inline RpptDesc make_descriptor(const TensorShape& s, DType dt, Layout layout, bool pad = true) {
    RpptDesc d{};
    d.numDims = 4;
    d.offsetInBytes = 0;
    d.dataType = to_rpp_dtype(dt);
    d.layout = to_rpp_layout(layout);
    d.n = s.n;
    d.c = s.c;
    d.h = s.h;
    d.w = s.w;
    const Rpp32u pw = pad ? padded_width(s.w) : s.w;
    if (d.layout == NHWC) {
        d.strides.nStride = s.c * s.h * pw;
        d.strides.hStride = s.c * pw;
        d.strides.wStride = s.c;
        d.strides.cStride = 1;
    } else {  // NCHW
        d.strides.nStride = s.c * s.h * pw;
        d.strides.cStride = s.h * pw;
        d.strides.hStride = pw;
        d.strides.wStride = 1;
    }
    return d;
}

// Total element count backing the tensor (offsetInBytes is 0 for the test suite).
inline std::size_t element_count(const RpptDesc& d) {
    return static_cast<std::size_t>(d.n) * d.strides.nStride;
}

inline std::size_t byte_size(const RpptDesc& d, DType dt) {
    return element_count(d) * dtype_size(dt);
}

// Per-image XYWH ROIs: the full frame, or a centered half-size window.
inline std::vector<RpptROI> make_roi(const RpptDesc& d, Roi mode) {
    std::vector<RpptROI> roi(d.n);
    for (Rpp32u i = 0; i < d.n; ++i) {
        RpptROI r{};
        if (mode == Roi::Full) {
            r.xywhROI.xy.x = 0;
            r.xywhROI.xy.y = 0;
            r.xywhROI.roiWidth = static_cast<int>(d.w);
            r.xywhROI.roiHeight = static_cast<int>(d.h);
        } else {
            r.xywhROI.xy.x = static_cast<int>(d.w / 4);
            r.xywhROI.xy.y = static_cast<int>(d.h / 4);
            r.xywhROI.roiWidth = static_cast<int>(d.w / 2);
            r.xywhROI.roiHeight = static_cast<int>(d.h / 2);
        }
        roi[i] = r;
    }
    return roi;
}

// ---- element addressing ----------------------------------------------------
//
// Every golden reaches an element through the descriptor's strides and never by walking the
// buffer flat, so a dense and a padded descriptor place the same logical coordinate correctly
// and two operands may use different conventions. These three are that mapping's only
// definition; the traversals below and the op references are all built on them.

// Origin of image n's channel-c plane.
inline std::size_t plane_base(const RpptDesc& d, Rpp32u n, Rpp32u c) {
    return static_cast<std::size_t>(n) * d.strides.nStride +
           static_cast<std::size_t>(c) * d.strides.cStride;
}

// Element (y, x) of the plane whose origin is `base`.
inline std::size_t plane_index(const RpptDesc& d, std::size_t base, std::size_t y, std::size_t x) {
    return base + y * d.strides.hStride + x * d.strides.wStride;
}

// Channel c of the pixel whose channel-0 element sits at `pixel`.
inline std::size_t channel_index(const RpptDesc& d, std::size_t pixel, Rpp32u c) {
    return pixel + static_cast<std::size_t>(c) * d.strides.cStride;
}

// ---- ROI traversal (shared by the reference model and the comparator) ------

struct RoiBounds {
    Rpp32u x0, y0, w, h;
};

inline RoiBounds roi_bounds(const RpptROI& r, RpptRoiType type) {
    if (type == XYWH)
        return {static_cast<Rpp32u>(r.xywhROI.xy.x), static_cast<Rpp32u>(r.xywhROI.xy.y),
                static_cast<Rpp32u>(r.xywhROI.roiWidth), static_cast<Rpp32u>(r.xywhROI.roiHeight)};
    return {static_cast<Rpp32u>(r.ltrbROI.lt.x), static_cast<Rpp32u>(r.ltrbROI.lt.y),
            static_cast<Rpp32u>(r.ltrbROI.rb.x - r.ltrbROI.lt.x + 1),
            static_cast<Rpp32u>(r.ltrbROI.rb.y - r.ltrbROI.lt.y + 1)};
}

// Invokes fn(n, b, c, base) once per image-channel plane, b being the image's ROI bounds and
// base the plane's origin. The outer walk shared by the goldens that address their own
// neighbourhood inside a plane (filters, morphology, warps) rather than a single element.
template <typename Fn>
void for_each_roi_plane(const RpptDesc& d, const RpptROI* roi, RpptRoiType type, Fn fn) {
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        for (Rpp32u c = 0; c < d.c; ++c) fn(n, b, c, plane_base(d, n, c));
    }
}

// Invokes fn(n, c, j, i, srcIdx, dstIdx) for every element of each image's ROI.
//
// RPP pointwise ops read the source from the ROI offset but write the output packed at
// the destination origin: for output row j / col i, the source element is at
// (y0 + j, x0 + i) and the destination element is at (j, i) (see the kernel's
// srcPtrChannel = srcPtrImage + ROI offset, dstPtrChannel = dstPtrImage). This is the
// single definition of that mapping, so the reference and the comparator agree.
template <typename Fn>
void for_each_roi_io(const RpptDesc& d, const RpptROI* roi, RpptRoiType type, Fn fn) {
    for_each_roi_plane(d, roi, type, [&](Rpp32u n, const RoiBounds& b, Rpp32u c,
                                         std::size_t base) {
        for (Rpp32u j = 0; j < b.h; ++j)
            for (Rpp32u i = 0; i < b.w; ++i)
                fn(n, c, j, i, plane_index(d, base, b.y0 + j, b.x0 + i),
                   plane_index(d, base, j, i));
    });
}

// Invokes fn(n, j, i, srcPix, dstPix) once per pixel of each image's ROI, where srcPix/dstPix
// are the channel-0 element offsets; the callback strides channels itself via channel_index().
// Same source-at-ROI-offset / destination-at-origin mapping as for_each_roi_io (that mapping's
// single definition), for ops that need a whole pixel's channels together (e.g. RGB<->HSV).
template <typename Fn>
void for_each_roi_pixel(const RpptDesc& d, const RpptROI* roi, RpptRoiType type, Fn fn) {
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        const std::size_t base = plane_base(d, n, 0);
        for (Rpp32u j = 0; j < b.h; ++j)
            for (Rpp32u i = 0; i < b.w; ++i)
                fn(n, j, i, plane_index(d, base, b.y0 + j, b.x0 + i), plane_index(d, base, j, i));
    }
}

// Visits every logical element of the image, fn(n, c, y, x, idx).
template <typename Fn>
void for_each_image_element(const RpptDesc& d, Fn fn) {
    for (Rpp32u n = 0; n < d.n; ++n)
        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t base = plane_base(d, n, c);
            for (Rpp32u y = 0; y < d.h; ++y)
                for (Rpp32u x = 0; x < d.w; ++x) fn(n, c, y, x, plane_index(d, base, y, x));
        }
}

// Deterministic input fill within each dtype's valid range:
// U8 [0,255], I8 [-128,127], I16 [-32768,32512], F16/F32 [0,1]. salt shifts the pattern so a
// second operand (for two-source ops) differs from the first.
template <typename T>
void fill_input(T* buf, std::size_t count, DType dt, unsigned salt = 0) {
    for (std::size_t i = 0; i < count; ++i) {
        const unsigned v = static_cast<unsigned>((i * 37u + 11u + salt * 101u) & 0xFFu);  // 0..255
        switch (dt) {
            case DType::U8: buf[i] = static_cast<T>(v); break;
            case DType::I8: buf[i] = static_cast<T>(static_cast<int>(v) - 128); break;
            case DType::I16: buf[i] = static_cast<T>((static_cast<int>(v) - 128) * 256); break;
            case DType::F16:
            case DType::F32: buf[i] = from_double<T>(static_cast<double>(v) / 255.0); break;
        }
    }
}

// The image counterpart of fill_input_nd(): the same pattern addressed by coordinate, so a tensor's
// logical content does not depend on its stride convention. The pattern is laid out in (n, c, y, x)
// order, which is memory order for a dense planar descriptor (there this is byte-identical to
// fill_input()) but not for a packed one, whose innermost axis is the channel.
template <typename T>
void fill_input_image(T* buf, const RpptDesc& d, DType dt, unsigned salt = 0) {
    const std::size_t logical = static_cast<std::size_t>(d.n) * d.c * d.h * d.w;
    std::vector<T> pattern(logical);
    fill_input<T>(pattern.data(), logical, dt, salt);
    std::size_t k = 0;
    for_each_image_element(
        d, [&](Rpp32u, Rpp32u, Rpp32u, Rpp32u, std::size_t idx) { buf[idx] = pattern[k++]; });
}

}  // namespace rpptest

#endif  // RPP_TEST_TENSOR_SETUP_H
