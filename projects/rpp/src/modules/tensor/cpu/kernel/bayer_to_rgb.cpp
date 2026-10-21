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

template <typename T>
RppStatus bayer_to_rgb_host_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                   RpptDescPtr dstDescPtr, RpptBayerPattern bayerPattern,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   RppLayoutParams layoutParams, rpp::Handle& handle) {
    // TODO: implement. Replace this whole body.
    return RPP_ERROR_NOT_IMPLEMENTED;
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
