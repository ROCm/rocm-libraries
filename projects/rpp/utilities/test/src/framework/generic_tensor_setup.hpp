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

#ifndef RPP_TEST_GENERIC_TENSOR_SETUP_H
#define RPP_TEST_GENERIC_TENSOR_SETUP_H

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Construction, traversal and comparison for the ND "generic tensor" ops (Misc domain), which take
// RpptGenericDesc (numDims/dims[]/strides[]) instead of the image domain's RpptDesc + XYWH ROI.

// ---- descriptors ----------------------------------------------------------

// Row padding, the generic-tensor form of the image domain's padded_width(): RPP's vectorized
// kernels store a full 8-element vector for a row's tail and expect the caller to have allocated
// the slack. It is the *width* axis that gets it -- innermost for planar layouts, one in from the
// end for packed ones, whose innermost axis is the channel axis.
inline Rpp32u nd_padded_width(Rpp32u w) { return (w / 8) * 8 + 8; }

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
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
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
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
            (void)hipHostFree(desc_);
#endif
        } else {
            delete desc_;
        }
    }
    GenericDescriptor(const GenericDescriptor&) = delete;
    GenericDescriptor& operator=(const GenericDescriptor&) = delete;

    RpptGenericDescPtr get() const { return desc_; }
    const RpptGenericDesc& operator*() const { return *desc_; }

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
        case DType::I8:  return static_cast<T>(-91);
        case DType::I16: return static_cast<T>(-21931);
        case DType::F16:
        case DType::F32: return from_double<T>(-1.0);  // the pattern only spans [0, 1]
        default:         return static_cast<T>(0xA5);
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

// ---- comparison -----------------------------------------------------------

// Compares the logical elements only -- padding slack is not data. The bound is
// absTolerance + relTolerance * |reference|; bit-exact ops pass 0 for both.
template <typename T>
::testing::AssertionResult compare_nd(const T* actual, const T* reference,
                                      const RpptGenericDesc& d, double absTolerance,
                                      double relTolerance = 0.0) {
    bool failed = false;
    std::string coords;
    double got = 0.0, want = 0.0, diff = 0.0, tolerance = 0.0;
    for_each_nd_coord(d, [&](const NdDims& coord) {
        if (failed) return;  // report the first mismatch only
        const std::size_t i = nd_offset(d, coord);
        const double a = to_double(actual[i]);
        const double r = to_double(reference[i]);
        const double delta = std::fabs(a - r);
        const double bound = absTolerance + relTolerance * std::fabs(r);
        if (delta <= bound) return;
        failed = true;
        got = a, want = r, diff = delta, tolerance = bound;
        for (std::size_t axis = 0; axis < coord.size(); ++axis)
            coords += (axis ? "," : "") + std::to_string(coord[axis]);
    });
    if (failed)
        return ::testing::AssertionFailure()
               << "mismatch at [" << coords << "]: actual=" << got << " reference=" << want
               << " diff=" << diff << " tolerance=" << tolerance;
    return ::testing::AssertionSuccess();
}

}  // namespace rpptest

#endif  // RPP_TEST_GENERIC_TENSOR_SETUP_H
