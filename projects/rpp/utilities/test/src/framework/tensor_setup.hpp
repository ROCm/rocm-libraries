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
#include <utility>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype.hpp"
#include "framework/intensity.hpp"
#include "framework/nd_config_param.hpp"
#include "framework/voxel_config_param.hpp"

namespace rpptest {

// =================================================================================================
// Images (Image domain)
// =================================================================================================

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

// =================================================================================================
// Generic ND tensors (Misc domain)
// =================================================================================================

// Construction, traversal and comparison for the ND "generic tensor" ops (Misc domain), which take
// RpptGenericDesc (numDims/dims[]/strides[]) instead of the image domain's RpptDesc + XYWH ROI.

// ---- descriptors ----------------------------------------------------------

// Row padding, the generic-tensor form of the image domain's padded_width(): RPP's vectorized
// kernels store a full 8-element vector for a row's tail and expect the caller to have allocated
// the slack. It is the *width* axis that gets it -- innermost for planar layouts, one in from the
// end for packed ones, whose innermost axis is the channel axis.
inline Rpp32u nd_padded_width(Rpp32u w) {
    return (w / 8) * 8 + 8;
}

// dims stay logical; padAxis (-1 = none) widens only the strides. layout matters to the ops that
// dispatch on it and require it to agree with the rank (slice); the rest ignore it.
inline RpptGenericDesc make_generic_descriptor(const NdDims& dims, DType dt,
                                               RpptLayout layout = RpptLayout::NCHW,
                                               int padAxis = -1) {
    RpptGenericDesc d{};
    d.numDims = dims.size();
    d.offsetInBytes = 0;
    d.dataType = to_rpp_dtype(dt);
    d.layout = layout;
    for (std::size_t i = 0; i < dims.size(); ++i) d.dims[i] = dims[i];
    Rpp32u v = 1;
    for (int i = static_cast<int>(d.numDims) - 1; i > 0; --i) {
        d.strides[i] = v;
        v *= (i == padAxis) ? nd_padded_width(d.dims[i]) : d.dims[i];
    }
    d.strides[0] = v;
    return d;
}

// The HIP ND kernels read dims/strides on the device at rank >= 4, so the descriptor struct itself
// must be device-addressable -- undocumented, and not required at lower ranks.
class GenericDescriptor {
   public:
    GenericDescriptor(RppBackend backend, const NdDims& dims, DType dt,
                      RpptLayout layout = RpptLayout::NCHW, int padAxis = -1)
        : backend_(backend) {
        if (backend_ == RPP_HIP_BACKEND) {
#if RPP_BACKEND_HIP
            RPP_TEST_CHECK_HIP(
                hipHostMalloc(reinterpret_cast<void**>(&desc_), sizeof(RpptGenericDesc)));
#endif
        } else {
            desc_ = new RpptGenericDesc();
        }
        *desc_ = make_generic_descriptor(dims, dt, layout, padAxis);
    }
    ~GenericDescriptor() {
        if (backend_ == RPP_HIP_BACKEND) {
#if RPP_BACKEND_HIP
            (void)hipHostFree(desc_);
#endif
        } else {
            delete desc_;
        }
    }
    GenericDescriptor(const GenericDescriptor&) = delete;
    GenericDescriptor& operator=(const GenericDescriptor&) = delete;

    RpptGenericDescPtr get() const {
        return desc_;
    }
    const RpptGenericDesc& operator*() const {
        return *desc_;
    }

   private:
    RppBackend backend_;
    RpptGenericDesc* desc_ = nullptr;
};

// Allocation size, i.e. including any padding slack.
inline std::size_t generic_element_count(const RpptGenericDesc& d) {
    return static_cast<std::size_t>(d.dims[0]) * d.strides[0];
}

inline std::size_t generic_byte_size(const RpptGenericDesc& d, DType dt) {
    return generic_element_count(d) * dtype_size(dt);
}

// 2 * nDim values per sample: per-axis starts then per-axis lengths. The suite exercises whole
// tensors, so starts are 0 and lengths are the operand's own extents.
inline std::vector<Rpp32u> make_nd_roi_tensor(const NdDims& dims) {
    const Rpp32u nDim = nd_rank(dims);
    std::vector<Rpp32u> roi(static_cast<std::size_t>(dims[0]) * 2 * nDim, 0);
    for (Rpp32u s = 0; s < dims[0]; ++s) {
        Rpp32u* sample = roi.data() + static_cast<std::size_t>(s) * 2 * nDim;
        for (Rpp32u a = 0; a < nDim; ++a) {
            sample[a] = 0;
            sample[nDim + a] = dims[a + 1];
        }
    }
    return roi;
}

// ---- traversal ------------------------------------------------------------
//
// Tensors are always addressed by logical coordinate through their own strides, never by walking
// the buffer flat, so a dense and a padded descriptor give the same logical answer and operands may
// differ in convention. src/tests/core/golden_layout_test.cpp guards that property.

// The descriptor's logical extents, batch axis first.
inline NdDims nd_dims(const RpptGenericDesc& d) {
    return NdDims(d.dims, d.dims + d.numDims);
}

inline std::size_t generic_logical_count(const RpptGenericDesc& d) {
    std::size_t n = 1;
    for (std::size_t a = 0; a < d.numDims; ++a) n *= d.dims[a];
    return n;
}

// An axis of extent 1 is held at 0 while a larger iteration space advances, which is broadcasting;
// for a tensor walked over its own dims the term is 0 anyway.
inline std::size_t nd_offset(const RpptGenericDesc& d, const NdDims& coord) {
    std::size_t index = 0;
    for (std::size_t a = 0; a < d.numDims; ++a)
        index += static_cast<std::size_t>(d.dims[a] == 1 ? 0 : coord[a]) * d.strides[a];
    return index;
}

// Visits every coordinate of an arbitrary extent list, row-major with the innermost axis fastest.
// The one coordinate walk: a golden whose iteration space is not a whole descriptor (slice's
// per-sample shape, normalize's per-sample sub-tensor) drives it from here rather than
// re-deriving the row-major order.
template <typename Fn>
void for_each_coord(const NdDims& extents, Fn fn) {
    const std::size_t rank = extents.size();
    std::size_t total = 1;
    for (Rpp32u e : extents) total *= e;
    NdDims coord(rank, 0);
    for (std::size_t n = 0; n < total; ++n) {
        fn(static_cast<const NdDims&>(coord));
        for (std::size_t a = rank; a-- > 0;) {
            if (++coord[a] < extents[a]) break;
            coord[a] = 0;
        }
    }
}

// Visits every logical coordinate of the descriptor.
template <typename Fn>
void for_each_nd_coord(const RpptGenericDesc& d, Fn fn) {
    for_each_coord(nd_dims(d), fn);
}

// fn(outIdx, idx1, idx2, coord) over every element of the (broadcast) output.
template <typename Fn>
void for_each_nd_element(const RpptGenericDesc& out, const RpptGenericDesc& s1,
                         const RpptGenericDesc& s2, Fn fn) {
    for_each_nd_coord(out, [&](const NdDims& coord) {
        fn(nd_offset(out, coord), nd_offset(s1, coord), nd_offset(s2, coord), coord);
    });
}

// ---- input fill -----------------------------------------------------------

// Written into the padding slack so a kernel that reads it yields obviously wrong output instead
// of something that looks like data.
template <typename T>
inline T nd_slack_poison(DType dt) {
    switch (dt) {
        case DType::I8:
            return static_cast<T>(-91);
        case DType::I16:
            return static_cast<T>(-21931);
        case DType::F16:
        case DType::F32:
            return from_double<T>(-1.0);  // the pattern only spans [0, 1]
        default:
            return static_cast<T>(0xA5);
    }
}

// The pattern is addressed by coordinate, so a tensor's logical content is the same under any
// stride convention. Byte-identical to fill_input() when the descriptor is dense.
template <typename T>
void fill_input_nd(T* buf, const RpptGenericDesc& d, DType dt, unsigned salt = 0) {
    const std::size_t alloc = generic_element_count(d);
    const std::size_t logical = generic_logical_count(d);
    if (alloc != logical)
        for (std::size_t i = 0; i < alloc; ++i) buf[i] = nd_slack_poison<T>(dt);

    std::vector<T> pattern(logical);
    fill_input<T>(pattern.data(), logical, dt, salt);
    std::size_t n = 0;
    for_each_nd_coord(d, [&](const NdDims& coord) { buf[nd_offset(d, coord)] = pattern[n++]; });
}

// =================================================================================================
// Voxel tensors (Voxel domain)
// =================================================================================================

// Construction, traversal and comparison for the Voxel domain: a 5D RpptGenericDesc (NCDHW or
// NDHWC) plus a per-sample RpptROI3D box. Descriptors are built with the GenericDescriptor
// above (dense strides, as the legacy voxel harness uses), so only the 3D layout
// mapping, the ROI3D box and an in-box traversal are added here.

inline Rpp32u voxel_channels(VoxelLayout l) {
    return l == VoxelLayout::NCDHW1 ? 1 : 3;
}

inline bool voxel_is_packed(VoxelLayout l) {
    return l == VoxelLayout::NDHWC3;
}

inline bool voxel_is_packed(const RpptGenericDesc& d) {
    return d.layout == RpptLayout::NDHWC;
}

inline RpptLayout to_rpp_layout_3d(VoxelLayout l) {
    return voxel_is_packed(l) ? RpptLayout::NDHWC : RpptLayout::NCDHW;
}

// Which descriptor axis each logical axis occupies: {n, d, h, w, c} packed, {n, c, d, h, w}
// planar. The single statement of that order -- extents (voxel_dims) and addressing
// (voxel_plane_base / voxel_plane_index) are both derived from it, so they cannot disagree.
struct VoxelAxes {
    std::size_t c, z, y, x;
};

inline VoxelAxes voxel_axes(bool packed) {
    return packed ? VoxelAxes{4, 1, 2, 3} : VoxelAxes{1, 2, 3, 4};
}

inline NdDims voxel_dims(const VoxelSize& s, VoxelLayout l) {
    const VoxelAxes a = voxel_axes(voxel_is_packed(l));
    NdDims dims(5, s.n);  // dims[0] is the batch axis
    dims[a.c] = voxel_channels(l);
    dims[a.z] = s.d;
    dims[a.y] = s.h;
    dims[a.x] = s.w;
    return dims;
}

inline Rpp32u voxel_channels(const RpptGenericDesc& d) {
    return d.dims[voxel_axes(voxel_is_packed(d)).c];
}

// ---- element addressing ----------------------------------------------------
//
// Addressed through the descriptor's own strides, never by walking the buffer flat, so the layout
// is the descriptor's business and a padded convention would need no change here. The image
// domain's plane_base / plane_index pair, one dimension up.

// Origin of sample n's channel-c volume.
inline std::size_t voxel_plane_base(const RpptGenericDesc& d, Rpp32u n, Rpp32u c) {
    return static_cast<std::size_t>(n) * d.strides[0] +
           static_cast<std::size_t>(c) * d.strides[voxel_axes(voxel_is_packed(d)).c];
}

// Voxel (z, y, x) of the volume whose origin is `base`.
inline std::size_t voxel_plane_index(const RpptGenericDesc& d, std::size_t base, Rpp32u z, Rpp32u y,
                                     Rpp32u x) {
    const VoxelAxes a = voxel_axes(voxel_is_packed(d));
    return base + static_cast<std::size_t>(z) * d.strides[a.z] +
           static_cast<std::size_t>(y) * d.strides[a.y] +
           static_cast<std::size_t>(x) * d.strides[a.x];
}

// ---- ROI3D -----------------------------------------------------------------

// x is the width axis, y the height axis, z the depth axis (RpptRoiXyzwhd's own naming).
struct VoxelBox {
    Rpp32u x0, y0, z0, w, h, d;
};

// LTFRBB is read inclusive of both corners, matching the image domain's LTRB convention
// (roi_bounds() above).
inline VoxelBox voxel_box(const RpptROI3D& r, Roi3D type) {
    if (type == Roi3D::XYZWHD)
        return {
            static_cast<Rpp32u>(r.xyzwhdROI.xyz.x),     static_cast<Rpp32u>(r.xyzwhdROI.xyz.y),
            static_cast<Rpp32u>(r.xyzwhdROI.xyz.z),     static_cast<Rpp32u>(r.xyzwhdROI.roiWidth),
            static_cast<Rpp32u>(r.xyzwhdROI.roiHeight), static_cast<Rpp32u>(r.xyzwhdROI.roiDepth)};
    return {static_cast<Rpp32u>(r.ltfrbbROI.ltf.x),
            static_cast<Rpp32u>(r.ltfrbbROI.ltf.y),
            static_cast<Rpp32u>(r.ltfrbbROI.ltf.z),
            static_cast<Rpp32u>(r.ltfrbbROI.rbb.x - r.ltfrbbROI.ltf.x + 1),
            static_cast<Rpp32u>(r.ltfrbbROI.rbb.y - r.ltfrbbROI.ltf.y + 1),
            static_cast<Rpp32u>(r.ltfrbbROI.rbb.z - r.ltfrbbROI.ltf.z + 1)};
}

inline RpptRoi3DType to_rpp_roi3d_type(Roi3D t) {
    return t == Roi3D::XYZWHD ? RpptRoi3DType::XYZWHD : RpptRoi3DType::LTFRBB;
}

// Per-sample ROI3D: the whole volume, or a centered half-extent box in every axis.
inline std::vector<RpptROI3D> make_voxel_roi(const VoxelSize& s, Roi mode, Roi3D type) {
    // One axis of the box; a half-extent is kept at least 1, so a thin depth axis still yields a
    // usable box.
    auto span = [mode](Rpp32u extent) {
        if (mode == Roi::Full) return std::pair<Rpp32u, Rpp32u>{0, extent};
        return std::pair<Rpp32u, Rpp32u>{extent / 4, extent / 2 ? extent / 2 : 1};
    };
    const auto sx = span(s.w), sy = span(s.h), sz = span(s.d);

    std::vector<RpptROI3D> roi(s.n);
    for (Rpp32u i = 0; i < s.n; ++i) {
        RpptROI3D r{};
        if (type == Roi3D::XYZWHD) {
            r.xyzwhdROI.xyz.x = static_cast<int>(sx.first);
            r.xyzwhdROI.xyz.y = static_cast<int>(sy.first);
            r.xyzwhdROI.xyz.z = static_cast<int>(sz.first);
            r.xyzwhdROI.roiWidth = static_cast<int>(sx.second);
            r.xyzwhdROI.roiHeight = static_cast<int>(sy.second);
            r.xyzwhdROI.roiDepth = static_cast<int>(sz.second);
        } else {
            r.ltfrbbROI.ltf.x = static_cast<int>(sx.first);
            r.ltfrbbROI.ltf.y = static_cast<int>(sy.first);
            r.ltfrbbROI.ltf.z = static_cast<int>(sz.first);
            r.ltfrbbROI.rbb.x = static_cast<int>(sx.first + sx.second - 1);
            r.ltfrbbROI.rbb.y = static_cast<int>(sy.first + sy.second - 1);
            r.ltfrbbROI.rbb.z = static_cast<int>(sz.first + sz.second - 1);
        }
        roi[i] = r;
    }
    return roi;
}

// ---- traversal --------------------------------------------------------------

// Invokes fn(n, c, box, base) once per sample-channel volume, box being that sample's ROI and base
// the volume's origin. The outer walk, for goldens that address a neighbourhood or mirror
// coordinates rather than a single voxel (the image domain's for_each_roi_plane, one dimension up).
template <typename Fn>
void for_each_voxel_roi_plane(const RpptGenericDesc& desc, const RpptROI3D* roi, Roi3D type,
                              Fn fn) {
    const Rpp32u channels = voxel_channels(desc);
    for (Rpp32u n = 0; n < desc.dims[0]; ++n) {
        const VoxelBox box = voxel_box(roi[n], type);
        for (Rpp32u c = 0; c < channels; ++c) fn(n, c, box, voxel_plane_base(desc, n, c));
    }
}

// Invokes fn(n, c, z, y, x, srcIdx, dstIdx) for every voxel of each sample's ROI box, (z, y, x)
// being the box-relative coordinate.
//
// Like the image-domain pointwise ops (for_each_roi_io above), the voxel ops read the
// source at the ROI offset and write the output packed at the destination origin: box voxel
// (z, y, x) comes from source (z0 + z, y0 + y, x0 + x) and lands at destination (z, y, x). This is
// the one definition of that mapping -- the goldens and the comparator both drive it, so they
// cannot disagree.
template <typename Fn>
void for_each_voxel_roi_io(const RpptGenericDesc& desc, const RpptROI3D* roi, Roi3D type, Fn fn) {
    for_each_voxel_roi_plane(
        desc, roi, type, [&](Rpp32u n, Rpp32u c, const VoxelBox& b, std::size_t base) {
            for (Rpp32u z = 0; z < b.d; ++z)
                for (Rpp32u y = 0; y < b.h; ++y)
                    for (Rpp32u x = 0; x < b.w; ++x)
                        fn(n, c, z, y, x,
                           voxel_plane_index(desc, base, b.z0 + z, b.y0 + y, b.x0 + x),
                           voxel_plane_index(desc, base, z, y, x));
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_TENSOR_SETUP_H
