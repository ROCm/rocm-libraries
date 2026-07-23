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
    for (Rpp32u n = 0; n < d.n; ++n) {
        const RoiBounds b = roi_bounds(roi[n], roiType);
        const int rx0 = static_cast<int>(b.x0), ry0 = static_cast<int>(b.y0);
        const int rx1 = rx0 + static_cast<int>(b.w), ry1 = ry0 + static_cast<int>(b.h);
        const std::size_t tblBase = static_cast<std::size_t>(n) * td.strides.nStride;
        for (Rpp32u c = 0; c < d.c; ++c) {
            const std::size_t imgBase = static_cast<std::size_t>(n) * d.strides.nStride +
                                        static_cast<std::size_t>(c) * d.strides.cStride;
            for (Rpp32u j = 0; j < b.h; ++j)
                for (Rpp32u i = 0; i < b.w; ++i) {
                    const std::size_t tblIdx = tblBase +
                                               static_cast<std::size_t>(j) * td.strides.hStride +
                                               static_cast<std::size_t>(i) * td.strides.wStride;
                    const double sx = colRemapTable[tblIdx];
                    const double sy = rowRemapTable[tblIdx];
                    const double v =
                        sample(src, d, imgBase, sx, sy, rx0, ry0, rx1, ry1, interp, border);
                    const std::size_t dstIdx = imgBase +
                                               static_cast<std::size_t>(j) * d.strides.hStride +
                                               static_cast<std::size_t>(i) * d.strides.wStride;
                    dst[dstIdx] = from_double<T>(quantize_stored(v, dt));
                }
        }
    }
}

}  // namespace rpptest

#endif  // RPP_TEST_REMAP_REF_H
