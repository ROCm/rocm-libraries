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

#ifndef RPP_TEST_REMAP_REF_H
#define RPP_TEST_REMAP_REF_H

#include <rpp/rpp.h>

#include <cstddef>

#include "framework/config_param.hpp"
#include "framework/interpolation.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_remap, derived from the op's definition
// output(x,y) = input(colRemapTable(x,y), rowRemapTable(x,y)), NOT from the RPP kernel. Used as the
// reference for both backends so kernel bugs surface as diffs.
//
// The remap tables are a per-image lookup: for output pixel (i,j) the source COLUMN to sample is
// colRemapTable(i,j) and the source ROW is rowRemapTable(i,j). The table value is taken literally as
// the ABSOLUTE source coordinate (image origin = texel (0,0)); the source is sampled in that absolute
// frame with the requested interpolation and per-dtype round-to-nearest quantization (both shared,
// kernel-independent, from interpolation.hpp / tensor_setup.hpp).
//
// Unlike the same-size warps this does not reuse geometric_reference(): remap genuinely has two
// distinct descriptors (the image descriptor vs the unpadded single-channel table descriptor), so the
// walk is written here while the sampler (interpolation.hpp) stays shared.
//
// NOTE (semantics assumption): the public header states neither the coordinate frame nor the boundary
// handling. The reference holds to the literal reading -- absolute-frame coordinates, with the valid
// source rectangle being the ROI rectangle [x0,x0+roiW) x [y0,y0+roiH) and samples outside it
// returning the dtype's black (sample()'s border param). A kernel that uses a different frame or
// border shows up as a diff, which is a finding, not a reference bug.
template <typename T>
void remap_reference(const T* src, T* dst, const RpptDesc& d, DType dt,
                     const Rpp32f* rowRemapTable, const Rpp32f* colRemapTable, const RpptDesc& td,
                     const RpptROI* roi, RpptRoiType roiType, RpptInterpolationType interp) {
    const double border = dtype_black(dt);
    for_each_roi_plane(d, roi, roiType, [&](Rpp32u n, const RoiBounds& b, Rpp32u,
                                            std::size_t imgBase) {
        const int rx0 = static_cast<int>(b.x0), ry0 = static_cast<int>(b.y0);
        const int rx1 = rx0 + static_cast<int>(b.w), ry1 = ry0 + static_cast<int>(b.h);
        const std::size_t tblBase = plane_base(td, n, 0);
        for (Rpp32u j = 0; j < b.h; ++j)
            for (Rpp32u i = 0; i < b.w; ++i) {
                const std::size_t tblIdx = plane_index(td, tblBase, j, i);
                const double sx = colRemapTable[tblIdx];
                const double sy = rowRemapTable[tblIdx];
                const double v =
                    sample(src, d, imgBase, sx, sy, rx0, ry0, rx1, ry1, interp, border);
                dst[plane_index(d, imgBase, j, i)] = from_double<T>(quantize_stored(v, dt));
            }
    });
}

}  // namespace rpptest

#endif  // RPP_TEST_REMAP_REF_H
