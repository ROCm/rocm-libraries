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
    }
    return U8;
}

inline std::size_t dtype_size(DType d) {
    switch (d) {
        case DType::U8:
        case DType::I8: return 1;
        case DType::F16: return 2;
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
// nearest (the intended round-to-nearest behavior the golden models hold to -- see the systemic
// I8 round-vs-truncate finding in section 13 of the plan). I8 pixels are the same intensities
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

// Builds a 4D descriptor for the given layout, with the padded row stride RPP expects.
inline RpptDesc make_descriptor(const TensorShape& s, DType dt, Layout layout) {
    RpptDesc d{};
    d.numDims = 4;
    d.offsetInBytes = 0;
    d.dataType = to_rpp_dtype(dt);
    d.layout = to_rpp_layout(layout);
    d.n = s.n;
    d.c = s.c;
    d.h = s.h;
    d.w = s.w;
    const Rpp32u pw = padded_width(s.w);
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

// Invokes fn(n, c, j, i, srcIdx, dstIdx) for every element of each image's ROI.
//
// RPP pointwise ops read the source from the ROI offset but write the output packed at
// the destination origin: for output row j / col i, the source element is at
// (y0 + j, x0 + i) and the destination element is at (j, i) (see the kernel's
// srcPtrChannel = srcPtrImage + ROI offset, dstPtrChannel = dstPtrImage). This is the
// single definition of that mapping, so the reference and the comparator agree.
template <typename Fn>
void for_each_roi_io(const RpptDesc& d, const RpptROI* roi, RpptRoiType type, Fn fn) {
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        for (Rpp32u c = 0; c < d.c; ++c)
            for (Rpp32u j = 0; j < b.h; ++j)
                for (Rpp32u i = 0; i < b.w; ++i) {
                    const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride +
                                             static_cast<std::size_t>(c) * d.strides.cStride;
                    const std::size_t srcIdx = base + (b.y0 + j) * d.strides.hStride +
                                               (b.x0 + i) * d.strides.wStride;
                    const std::size_t dstIdx =
                        base + j * d.strides.hStride + i * d.strides.wStride;
                    fn(n, c, j, i, srcIdx, dstIdx);
                }
    }
}

// Invokes fn(n, j, i, srcPix, dstPix) once per pixel of each image's ROI, where srcPix/dstPix
// are the channel-0 element offsets; the callback strides channels itself via d.strides.cStride.
// Same source-at-ROI-offset / destination-at-origin mapping as for_each_roi_io (that mapping's
// single definition), for ops that need a whole pixel's channels together (e.g. RGB<->HSV).
template <typename Fn>
void for_each_roi_pixel(const RpptDesc& d, const RpptROI* roi, RpptRoiType type, Fn fn) {
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        for (Rpp32u j = 0; j < b.h; ++j)
            for (Rpp32u i = 0; i < b.w; ++i) {
                const std::size_t base = static_cast<std::size_t>(n) * d.strides.nStride;
                const std::size_t srcPix =
                    base + (b.y0 + j) * d.strides.hStride + (b.x0 + i) * d.strides.wStride;
                const std::size_t dstPix = base + j * d.strides.hStride + i * d.strides.wStride;
                fn(n, j, i, srcPix, dstPix);
            }
    }
}

// Deterministic input fill within each dtype's valid range:
// U8 [0,255], I8 [-128,127], F16/F32 [0,1]. salt shifts the pattern so a second
// operand (for two-source ops) differs from the first.
template <typename T>
void fill_input(T* buf, std::size_t count, DType dt, unsigned salt = 0) {
    for (std::size_t i = 0; i < count; ++i) {
        const unsigned v = static_cast<unsigned>((i * 37u + 11u + salt * 101u) & 0xFFu);  // 0..255
        switch (dt) {
            case DType::U8: buf[i] = static_cast<T>(v); break;
            case DType::I8: buf[i] = static_cast<T>(static_cast<int>(v) - 128); break;
            case DType::F16:
            case DType::F32: buf[i] = from_double<T>(static_cast<double>(v) / 255.0); break;
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_TENSOR_SETUP_H
