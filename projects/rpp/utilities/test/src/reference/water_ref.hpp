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

/*
Reference model: water

RPP op
  rppt_water   (Image / Effects augmentation)

Description
  Sinusoidal "water surface" displacement with an amplitude, frequency and
  phase per axis. Each axis is displaced by a wave driven by the other
  coordinate; a zero amplitude is the identity. Sampling and quantization are
  handled by geometric_reference().

Expression
  srcX = x + amplitudeX * sin(frequencyX * y + phaseX)
  srcY = y + amplitudeY * cos(frequencyY * x + phaseY)

Notes
  The public header does not specify the following:

  - Sampling is NEAREST_NEIGHBOR (the op has no interpolationType) and
    out-of-ROI samples read the type's black, as in the other warps.
  - The wave uses ROI-local x, y, and the ROI origin is then added to reach
    the source.
*/

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
void water_reference(const T* src, const RpptDesc& sd, T* dst, const RpptDesc& dd, DType dt,
                     const RpptROI* roi, RpptRoiType roiType, const Rpp32f* amplitudeX,
                     const Rpp32f* amplitudeY, const Rpp32f* frequencyX, const Rpp32f* frequencyY,
                     const Rpp32f* phaseX, const Rpp32f* phaseY) {
    geometric_reference<T>(
        src, sd, dst, dd, dt, roi, roiType, roi_out_sizes(sd, roi, roiType), NEAREST_NEIGHBOR,
        [&](Rpp32u n, double ox, double oy, double& sx, double& sy) {
            const RoiBounds b = roi_bounds(roi[n], roiType);
            water_map(ox, oy, b.x0, b.y0, amplitudeX[n], amplitudeY[n], frequencyX[n],
                      frequencyY[n], phaseX[n], phaseY[n], sx, sy);
        });
}

}  // namespace rpptest

#endif  // RPP_TEST_WATER_REF_H
