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

#ifndef RPP_TEST_WATER_REF_H
#define RPP_TEST_WATER_REF_H

#include <rpp/rpp.h>

#include <cmath>

#include "framework/config_param.hpp"
#include "framework/geometric.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Independent host golden model for rppt_water, derived from the op's definition (a sinusoidal
// "water surface" displacement) and its six documented parameters -- an amplitude, frequency and
// phase per axis -- NOT from the RPP kernel. Used as the reference for both backends so kernel bugs
// surface as diffs.
//
// Each axis is displaced by a wave driven by the *other* coordinate, which is what makes the
// surface ripple and what the separate X and Y parameter triples are for:
//
//     srcX = x + amplitudeX * sin(frequencyX * y + phaseX)
//     srcY = y + amplitudeY * cos(frequencyY * x + phaseY)
//
// The source is then sampled at (srcX, srcY) and the result quantized per dtype -- both handled by
// geometric_reference() / interpolation.hpp, shared with the other warps, so the sampling model is
// the suite's own and not the kernel's. A zero amplitude makes the map the exact identity.
//
// NOTE (semantics assumptions): the public header documents none of the following, so a kernel that
// chose differently shows up as a diff -- a finding, not a reference bug.
//   - Sampling is NEAREST_NEIGHBOR (the op exposes no interpolationType) and a displaced coordinate
//     outside the ROI reads the dtype's black, the border rule the suite's other warps use.
//   - The wave is a function of the position within the processed region (ROI-local x, y), and the
//     ROI origin is then added to reach the source. Under a full ROI the two readings coincide, so
//     that grid validates the formula independently of this choice.

// Destination pixel (outX, outY) of a region whose source origin is (x0, y0) -> the source
// coordinate to sample, in the absolute image frame.
inline void water_map(double outX, double outY, double x0, double y0, double amplitudeX,
                      double amplitudeY, double frequencyX, double frequencyY, double phaseX,
                      double phaseY, double& srcX, double& srcY) {
    srcX = x0 + outX + amplitudeX * std::sin(frequencyX * outY + phaseX);
    srcY = y0 + outY + amplitudeY * std::cos(frequencyY * outX + phaseY);
}

// Each parameter tensor holds one value per image.
template <typename T>
void water_reference(const T* src, T* dst, const RpptDesc& d, DType dt, const RpptROI* roi,
                     RpptRoiType roiType, const Rpp32f* amplitudeX, const Rpp32f* amplitudeY,
                     const Rpp32f* frequencyX, const Rpp32f* frequencyY, const Rpp32f* phaseX,
                     const Rpp32f* phaseY) {
    geometric_reference<T>(src, dst, d, dt, roi, roiType, roi_out_sizes(d, roi, roiType),
                           NEAREST_NEIGHBOR,
                           [&](Rpp32u n, double ox, double oy, double& sx, double& sy) {
                               const RoiBounds b = roi_bounds(roi[n], roiType);
                               water_map(ox, oy, b.x0, b.y0, amplitudeX[n], amplitudeY[n],
                                         frequencyX[n], frequencyY[n], phaseX[n], phaseY[n], sx,
                                         sy);
                           });
}

}  // namespace rpptest

#endif  // RPP_TEST_WATER_REF_H
