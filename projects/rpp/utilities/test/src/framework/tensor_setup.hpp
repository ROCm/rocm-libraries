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
#include "framework/dtype.hpp"
#include "framework/intensity.hpp"

namespace rpptest {

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
inline Rpp32u padded_width(Rpp32u w) {
    return (w / 8) * 8 + 8;
}

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

// The source and destination descriptors for a config, at the config's own extents. Each side's
// channel count follows its own layout, so a conversion that changes it (PKD3 -> PLN1) is described
// correctly without the test restating the shape; for the usual same-layout config the two are
// identical. Ops whose output extents differ from the input's (the resize family) build their own
// shapes and call make_descriptor() directly with cfg.layoutIn / cfg.layoutOut.
inline RpptDesc make_src_descriptor(const TestConfig& c, bool pad = true) {
    const TensorShape s{c.size.n, static_cast<Rpp32u>(channels_of(c.layoutIn)), c.size.h, c.size.w};
    return make_descriptor(s, c.dtype, c.layoutIn, pad);
}

inline RpptDesc make_dst_descriptor(const TestConfig& c, bool pad = true) {
    const TensorShape s{c.size.n, static_cast<Rpp32u>(channels_of(c.layoutOut)), c.size.h,
                        c.size.w};
    return make_descriptor(s, c.dtype, c.layoutOut, pad);
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

// Invokes fn(n, b, c, srcBase, dstBase) once per image-channel plane, the two origins addressed
// through their own descriptor. The neighbourhood-walking counterpart of the dual-descriptor
// for_each_roi_io below, for filters and warps under a toggled output layout.
template <typename Fn>
void for_each_roi_plane(const RpptDesc& sd, const RpptDesc& dd, const RpptROI* roi,
                        RpptRoiType type, Fn fn) {
    for (Rpp32u n = 0; n < sd.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        for (Rpp32u c = 0; c < sd.c; ++c) fn(n, b, c, plane_base(sd, n, c), plane_base(dd, n, c));
    }
}

// Invokes fn(n, c, j, i, srcIdx, dstIdx) for every element of each image's ROI, addressing the
// source through `sd` and the destination through `dd`.
//
// RPP pointwise ops read the source from the ROI offset but write the output packed at
// the destination origin: for output row j / col i, the source element is at
// (y0 + j, x0 + i) and the destination element is at (j, i) (see the kernel's
// srcPtrChannel = srcPtrImage + ROI offset, dstPtrChannel = dstPtrImage). This is the
// single definition of that mapping, so the reference and the comparator agree.
//
// Two descriptors because an op may write a layout other than the one it reads -- RPP's fused
// output-layout toggle, NHWC <-> NCHW. Each index goes through its own descriptor's strides, so
// the transpose is expressed entirely by the descriptors and no golden has to special-case it.
template <typename Fn>
void for_each_roi_io(const RpptDesc& sd, const RpptDesc& dd, const RpptROI* roi, RpptRoiType type,
                     Fn fn) {
    for (Rpp32u n = 0; n < sd.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        for (Rpp32u c = 0; c < sd.c; ++c) {
            const std::size_t srcBase = plane_base(sd, n, c);
            const std::size_t dstBase = plane_base(dd, n, c);
            for (Rpp32u j = 0; j < b.h; ++j)
                for (Rpp32u i = 0; i < b.w; ++i)
                    fn(n, c, j, i, plane_index(sd, srcBase, b.y0 + j, b.x0 + i),
                       plane_index(dd, dstBase, j, i));
        }
    }
}

// Single-descriptor form: source and destination share a layout, which is every op that does not
// exercise the output-layout toggle.
template <typename Fn>
void for_each_roi_io(const RpptDesc& d, const RpptROI* roi, RpptRoiType type, Fn fn) {
    for_each_roi_io(d, d, roi, type, fn);
}

// Invokes fn(n, j, i, srcPix, dstPix) once per pixel of each image's ROI, where srcPix/dstPix
// are the channel-0 element offsets; the callback strides channels itself via channel_index()
// -- through `sd` for srcPix and `dd` for dstPix, which is what makes a toggled layout work.
// Same source-at-ROI-offset / destination-at-origin mapping as for_each_roi_io (that mapping's
// single definition), for ops that need a whole pixel's channels together (e.g. RGB<->HSV).
template <typename Fn>
void for_each_roi_pixel(const RpptDesc& sd, const RpptDesc& dd, const RpptROI* roi,
                        RpptRoiType type, Fn fn) {
    for (Rpp32u n = 0; n < sd.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], type);
        const std::size_t srcBase = plane_base(sd, n, 0);
        const std::size_t dstBase = plane_base(dd, n, 0);
        for (Rpp32u j = 0; j < b.h; ++j)
            for (Rpp32u i = 0; i < b.w; ++i)
                fn(n, j, i, plane_index(sd, srcBase, b.y0 + j, b.x0 + i),
                   plane_index(dd, dstBase, j, i));
    }
}

// Single-descriptor form, as for for_each_roi_io above.
template <typename Fn>
void for_each_roi_pixel(const RpptDesc& d, const RpptROI* roi, RpptRoiType type, Fn fn) {
    for_each_roi_pixel(d, d, roi, type, fn);
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
            case DType::U8:
                buf[i] = static_cast<T>(v);
                break;
            case DType::I8:
                buf[i] = static_cast<T>(static_cast<int>(v) - 128);
                break;
            case DType::I16:
                buf[i] = static_cast<T>((static_cast<int>(v) - 128) * 256);
                break;
            case DType::F16:
            case DType::F32:
                buf[i] = from_double<T>(static_cast<double>(v) / 255.0);
                break;
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
