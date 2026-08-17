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

#ifndef RPP_TEST_NORMALIZE_REF_H
#define RPP_TEST_NORMALIZE_REF_H

#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "framework/generic_tensor_setup.hpp"

namespace rpptest {

// Host golden model for rppt_normalize. Modelled from the operation's definition and the
// public API header, NOT from the kernel; computed once on the host and used as the
// reference for both the HOST and HIP backends.
//
// Semantics, per the header: "removing the mean and dividing by the standard deviation",
// with scale "multiplied with data after subtracting from mean" and shift "added finally":
//
//     dst = ((src - mean) / stdDev) * scale + shift
//
// axisMask selects the axes that are REDUCED: bit i set collapses axis i, so mean/stdDev
// carry one value per combination of the remaining (non-reduced) coordinates --
// paramSize = product over i of (bit i set ? 1 : dims[i]), laid out row-major over those
// same axes, one such block per sample.
//
// computeMeanStddev is a bitmask: bit 0 requests that the mean be computed internally,
// bit 1 the standard deviation; a clear bit means the caller's value is used. The model
// mirrors that, computing whichever statistics the flag asks for over each reduction group
// and taking the rest from the supplied tensors.
//
// Assumption (undocumented): the standard deviation is the POPULATION deviation --
// sqrt(sum((x - mean)^2) / N), not the N-1 sample form. Everything is accumulated in
// double; no clamping is applied, since a normalized result is signed and the documented
// output dtypes are floating point.

// Per-axis extents of the mean/stdDev tensor for a given reduction mask: reduced axes
// collapse to 1, the rest keep their extent. dims includes the leading batch axis.
inline std::vector<Rpp32u> normalize_param_dims(const NdDims& dims, Rpp32u axisMask) {
    const Rpp32u nDim = nd_rank(dims);
    std::vector<Rpp32u> paramDims(nDim);
    for (Rpp32u i = 0; i < nDim; ++i)
        paramDims[i] = ((axisMask >> i) & 1u) ? 1u : dims[i + 1];
    return paramDims;
}

// Row-major strides over the param extents, and the total per-sample param count.
inline std::vector<Rpp32u> normalize_param_strides(const std::vector<Rpp32u>& paramDims) {
    const std::size_t nDim = paramDims.size();
    std::vector<Rpp32u> strides(nDim, 1);
    for (int i = static_cast<int>(nDim) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * paramDims[i + 1];
    return strides;
}

inline Rpp32u normalize_param_size(const std::vector<Rpp32u>& paramDims) {
    Rpp32u size = 1;
    for (Rpp32u d : paramDims) size *= d;
    return size;
}

template <typename Tin, typename Tout>
void normalize_reference(const Tin* src, Tout* dst, const RpptGenericDesc& srcDesc,
                         const RpptGenericDesc& dstDesc, Rpp32u axisMask,
                         const Rpp32f* meanTensor, const Rpp32f* stdDevTensor,
                         Rpp8u computeMeanStddev, Rpp32f scale, Rpp32f shift) {
    const NdDims dims = nd_dims(srcDesc);
    const Rpp32u nDim = nd_rank(dims);
    const Rpp32u batch = dims[0];
    const NdDims sampleExtents(dims.begin() + 1, dims.end());

    const std::vector<Rpp32u> paramDims = normalize_param_dims(dims, axisMask);
    const std::vector<Rpp32u> paramStrides = normalize_param_strides(paramDims);
    const Rpp32u paramSize = normalize_param_size(paramDims);

    const bool computeMean = (computeMeanStddev & 1u) != 0;
    const bool computeStdDev = (computeMeanStddev & 2u) != 0;

    std::vector<double> mean(paramSize), stdDev(paramSize);
    std::vector<double> acc(paramSize);
    std::vector<std::size_t> count(paramSize);

    // Walks one sample's logical coordinates, handing the callback the source and destination
    // addresses (each formed from its own descriptor's strides) and the param slot the element
    // reduces into. Addressing by coordinate rather than walking the sample flat matters twice
    // over for a reduction op: padding slack must not enter the mean/stddev, and it must not be
    // mistaken for a coordinate.
    NdDims coord(nDim + 1);
    auto for_each_element = [&](Rpp32u n, auto fn) {
        coord[0] = n;
        for_each_coord(sampleExtents, [&](const NdDims& c) {
            Rpp32u p = 0;
            for (Rpp32u a = 0; a < nDim; ++a) {
                coord[a + 1] = c[a];
                if (!((axisMask >> a) & 1u)) p += c[a] * paramStrides[a];
            }
            fn(nd_offset(srcDesc, coord), nd_offset(dstDesc, coord), p);
        });
    };

    for (Rpp32u n = 0; n < batch; ++n) {
        const Rpp32f* sampleMean = meanTensor + static_cast<std::size_t>(n) * paramSize;
        const Rpp32f* sampleStdDev = stdDevTensor + static_cast<std::size_t>(n) * paramSize;

        if (computeMean) {
            std::fill(acc.begin(), acc.end(), 0.0);
            std::fill(count.begin(), count.end(), std::size_t{0});
            for_each_element(n, [&](std::size_t srcIdx, std::size_t, Rpp32u p) {
                acc[p] += to_double(src[srcIdx]);
                ++count[p];
            });
            for (Rpp32u p = 0; p < paramSize; ++p)
                mean[p] = count[p] ? acc[p] / static_cast<double>(count[p]) : 0.0;
        } else {
            for (Rpp32u p = 0; p < paramSize; ++p) mean[p] = static_cast<double>(sampleMean[p]);
        }

        if (computeStdDev) {
            std::fill(acc.begin(), acc.end(), 0.0);
            std::fill(count.begin(), count.end(), std::size_t{0});
            for_each_element(n, [&](std::size_t srcIdx, std::size_t, Rpp32u p) {
                const double d = to_double(src[srcIdx]) - mean[p];
                acc[p] += d * d;
                ++count[p];
            });
            for (Rpp32u p = 0; p < paramSize; ++p)
                stdDev[p] = count[p] ? std::sqrt(acc[p] / static_cast<double>(count[p])) : 0.0;
        } else {
            for (Rpp32u p = 0; p < paramSize; ++p) stdDev[p] = static_cast<double>(sampleStdDev[p]);
        }

        for_each_element(n, [&](std::size_t srcIdx, std::size_t dstIdx, Rpp32u p) {
            const double inv = (stdDev[p] != 0.0) ? (1.0 / stdDev[p]) : 0.0;
            const double v = (to_double(src[srcIdx]) - mean[p]) * inv * static_cast<double>(scale) +
                             static_cast<double>(shift);
            dst[dstIdx] = from_double<Tout>(v);
        });
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_NORMALIZE_REF_H
