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

#include "host_tensor_executors.hpp"

/*
bayer_to_rgb -- HOST kernel

Spec (the gtest golden, utilities/test/src/reference/bayer_to_rgb_ref.hpp, is the authority):

  Source: single-channel NCHW mosaic. Destination: 3 channels, R G B order, NCHW or NHWC.
  For every image n and every output pixel (j, i) of the ROI (ROI-local coordinates):

    colour(y, x) = bayerPattern's colour at (y % 2, x % 2)      pattern anchored at the ROI origin
    s(y, x)      = src(y0 + m(y, roiHeight), x0 + m(x, roiWidth))
    m(r, n)      = -r for r < 0, 2(n-1) - r for r >= n, else r  (REFLECT_NO_EDGE, mirror without
                                                                  repeating the edge pixel)

    For each output channel C:
      C == colour(j, i)                       -> s(j, i)
      C == G (site is R or B)                 -> mean of s(j-1, i), s(j+1, i), s(j, i-1), s(j, i+1)
      site is G, C == colour(j, i-1)          -> mean of s(j, i-1), s(j, i+1)
      site is G, C == colour(j-1, i)          -> mean of s(j-1, i), s(j+1, i)
      otherwise (B at R, R at B)              -> mean of the 4 diagonal neighbours

  - The mirror keeps neighbour colours intact, so colour() of a neighbour can be read from its
    unmirrored coordinates.
  - Read the source at the ROI offset (x0, y0); write the destination from its origin.
  - I8: a mean of stored values is shift-invariant, so no +128 is needed.
  - Store: U8/I8 round to nearest then clamp; F16/F32 clamp to [0,1].
  - layoutParams describes the DESTINATION (3-channel NCHW or NHWC); the source is always PLN1.

  Test: rpp_tests --gtest_filter='Image_DataExchange/BayerToRgb*HOST_*'
*/

// Bilinear demosaic. Every pattern is RGGB shifted by 0 or 1 pixel in x and/or y, so a pixel is
// classified by whether it shares a row and/or a column with R:
//   (R row, R col) = R    (R row, other col) = G
//   (other row, R col) = G    (other row, other col) = B
// Each missing channel is the average of its nearest same-color neighbors, assuming color varies
// smoothly between nearby samples.

// Parity (0 = even, 1 = odd), relative to the ROI origin, of the row and column that hold R
inline void demosaic_red_parity(RpptBayerPattern bayerPattern, Rpp32s& redRowParity,
                                Rpp32s& redColParity) {
    switch (bayerPattern) {
        case RpptBayerPattern_BGGR:
            redRowParity = 1;
            redColParity = 1;
            break;
        case RpptBayerPattern_GRBG:
            redRowParity = 0;
            redColParity = 1;
            break;
        case RpptBayerPattern_GBRG:
            redRowParity = 1;
            redColParity = 0;
            break;
        case RpptBayerPattern_RGGB:
        default:
            redRowParity = 0;
            redColParity = 0;
            break;
    }
}

// Reflects an index back into [lo, hi) without repeating the edge sample (lo - 1 -> lo + 1), so a
// reflected neighbor keeps the same parity and therefore lands on the same Bayer color
inline Rpp32s demosaic_reflect_index(Rpp32s idx, Rpp32s lo, Rpp32s hi) {
    if (hi - lo == 1) return lo;
    if (idx < lo) return 2 * lo - idx;
    if (idx >= hi) return 2 * (hi - 1) - idx;
    return idx;
}

// Loads a pixel as Rpp32f in the range expected by saturate_pixel(). i8 is shifted to 0 - 255 here
// and saturate_pixel() subtracts the 128 again, so the shift cancels, as the spec requires
inline Rpp32f demosaic_load_pixel(Rpp8u pixel) {
    return static_cast<Rpp32f>(pixel);
}
inline Rpp32f demosaic_load_pixel(Rpp8s pixel) {
    return static_cast<Rpp32f>(pixel) + 128.0f;
}
inline Rpp32f demosaic_load_pixel(Rpp32f pixel) {
    return pixel;
}
inline Rpp32f demosaic_load_pixel(Rpp16f pixel) {
    return static_cast<Rpp32f>(pixel);
}

template <typename T>
RppStatus bayer_to_rgb_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                   RpptDescPtr dstDescPtr, RpptBayerPattern bayerPattern,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   RppLayoutParams layoutParams, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    Rpp32s redRowParity, redColParity;
    demosaic_red_parity(bayerPattern, redRowParity, redColParity);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        // ROI bounds in image coordinates; the pattern phase is ROI-local
        Rpp32s xStart = roi.xywhROI.xy.x;
        Rpp32s yStart = roi.xywhROI.xy.y;
        Rpp32s xEnd = xStart + roi.xywhROI.roiWidth;
        Rpp32s yEnd = yStart + roi.xywhROI.roiHeight;

        // Neighbors outside the ROI are mirrored back inside it, so padding is never read
        auto srcPixel = [&](Rpp32s y, Rpp32s x) -> Rpp32f {
            y = demosaic_reflect_index(y, yStart, yEnd);
            x = demosaic_reflect_index(x, xStart, xEnd);
            return demosaic_load_pixel(
                srcPtrImage[y * srcDescPtr->strides.hStride + x * srcDescPtr->strides.wStride]);
        };

        for (Rpp32s y = yStart; y < yEnd; y++) {
            T* dstPtrRow = dstPtrImage + (y - yStart) * dstDescPtr->strides.hStride;
            bool isRedRow = (((y - yStart) & 1) == redRowParity);
            for (Rpp32s x = xStart; x < xEnd; x++) {
                bool isRedCol = (((x - xStart) & 1) == redColParity);
                Rpp32f center = srcPixel(y, x);
                Rpp32f red, green, blue;

                if (isRedRow && isRedCol) {
                    // Red location - green from the 4 cross neighbors, blue from the 4 diagonals
                    red = center;
                    green = 0.25f * (srcPixel(y - 1, x) + srcPixel(y + 1, x) + srcPixel(y, x - 1) +
                                     srcPixel(y, x + 1));
                    blue = 0.25f * (srcPixel(y - 1, x - 1) + srcPixel(y - 1, x + 1) +
                                    srcPixel(y + 1, x - 1) + srcPixel(y + 1, x + 1));
                } else if (!isRedRow && !isRedCol) {
                    // Blue location - green from the 4 cross neighbors, red from the 4 diagonals
                    blue = center;
                    green = 0.25f * (srcPixel(y - 1, x) + srcPixel(y + 1, x) + srcPixel(y, x - 1) +
                                     srcPixel(y, x + 1));
                    red = 0.25f * (srcPixel(y - 1, x - 1) + srcPixel(y - 1, x + 1) +
                                   srcPixel(y + 1, x - 1) + srcPixel(y + 1, x + 1));
                } else {
                    // Green location - red and blue from the 2 horizontal or 2 vertical neighbors
                    Rpp32f horizontal = 0.5f * (srcPixel(y, x - 1) + srcPixel(y, x + 1));
                    Rpp32f vertical = 0.5f * (srcPixel(y - 1, x) + srcPixel(y + 1, x));
                    green = center;
                    red = isRedRow ? horizontal : vertical;   // red row: R is left/right
                    blue = isRedRow ? vertical : horizontal;  // blue row: B is left/right
                }

                // wStride/cStride handle both NCHW (planar) and NHWC (packed) destinations
                T* dstPtrTemp = dstPtrRow + (x - xStart) * dstDescPtr->strides.wStride;
                saturate_pixel(red, dstPtrTemp);
                saturate_pixel(green, dstPtrTemp + dstDescPtr->strides.cStride);
                saturate_pixel(blue, dstPtrTemp + 2 * dstDescPtr->strides.cStride);
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus bayer_to_rgb_host_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                   RpptBayerPattern, RpptROIPtr, RpptRoiType,
                                                   RppLayoutParams, rpp::Handle&);
template RppStatus bayer_to_rgb_host_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                    RpptBayerPattern, RpptROIPtr, RpptRoiType,
                                                    RppLayoutParams, rpp::Handle&);
template RppStatus bayer_to_rgb_host_tensor<Rpp16f>(Rpp16f*, RpptDescPtr, Rpp16f*, RpptDescPtr,
                                                    RpptBayerPattern, RpptROIPtr, RpptRoiType,
                                                    RppLayoutParams, rpp::Handle&);
template RppStatus bayer_to_rgb_host_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                   RpptBayerPattern, RpptROIPtr, RpptRoiType,
                                                   RppLayoutParams, rpp::Handle&);
