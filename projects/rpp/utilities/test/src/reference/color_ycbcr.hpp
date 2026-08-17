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

#ifndef RPP_TEST_COLOR_YCBCR_H
#define RPP_TEST_COLOR_YCBCR_H

namespace rpptest {

// Shared BT.601 full-range ("JPEG") RGB<->YCbCr building blocks, derived from the color transform's
// definition (NOT any RPP kernel), so every op that routes through luma/chroma agrees on it and a
// single reviewed copy backs them all -- the arrangement color_hsv.hpp provides for RGB<->HSV.
// Users: histogram_equalize (equalizes Y, keeps Cb/Cr) and jpeg_compression_distortion (quantizes
// the three planes separately).
//
// Both directions operate on continuous [0,255] intensities, and take their inputs by value, so a
// caller may pass the same variables as inputs and outputs to convert a pixel in place. Quantizing the result is the caller's
// business, since the two ops legitimately differ there: histogram_equalize rounds to 8 bits as an
// integer YCbCr pipeline does, jpeg_compression_distortion keeps the samples continuous so the
// distortion comes only from its coefficient quantization.
//
//   Y  =  0.299 R + 0.587 G + 0.114 B
//   Cb = -0.168736 R - 0.331264 G + 0.5 B + 128
//   Cr =  0.5 R - 0.418688 G - 0.081312 B + 128
//   R  = Y + 1.402 (Cr-128)
//   G  = Y - 0.344136 (Cb-128) - 0.714136 (Cr-128)
//   B  = Y + 1.772 (Cb-128)

inline double ycbcr_y(double r, double g, double b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}
inline double ycbcr_cb(double r, double g, double b) {
    return -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
}
inline double ycbcr_cr(double r, double g, double b) {
    return 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
}

inline void rgb_to_ycbcr(double r, double g, double b, double& y, double& cb, double& cr) {
    y = ycbcr_y(r, g, b);
    cb = ycbcr_cb(r, g, b);
    cr = ycbcr_cr(r, g, b);
}

inline void ycbcr_to_rgb(double y, double cb, double cr, double& r, double& g, double& b) {
    r = y + 1.402 * (cr - 128.0);
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0);
    b = y + 1.772 * (cb - 128.0);
}

}  // namespace rpptest

#endif  // RPP_TEST_COLOR_YCBCR_H
