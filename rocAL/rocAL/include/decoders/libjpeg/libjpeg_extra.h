/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#include <turbojpeg.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "libjpeg_utils.h"

extern "C" {

//! extra apis for rocal to support partial decoding

//! * Helper function to se the source
//! * This function doesn't scale the decoded image

//! * Decompress a subregion of JPEG image to an RGB, grayscale, or CMYK image.
//! * This function doesn't scale the decoded image

/*!
  \param handle  TJPeg handle
  \param jpegBuf compressed jpeg image buffer
  \param jpegSize Size of the compressed data provided in the input_buffer
  \param dstBuf user provided output buffer
  \param width, pitch, height  width, stride and height of the allocated buffer
  \param flags  TJPEG flags
  \param pixelFormat  pixel format of the image
  \param crop_x_diff,  crop_width_diff Actual crop_x and crop_w (adjusted to MB boundery)
  \param x1, y1, crop_width, crop_height requested crop window
*/

int tjDecompress2_partial(tjhandle handle, const unsigned char *jpegBuf,
                                    unsigned long jpegSize, unsigned char *dstBuf,
                                    int width, int pitch, int height, int pixelFormat,
                                    int flags, unsigned int *crop_x_diff, unsigned int *crop_width_diff,
                                    unsigned int x1, unsigned int y1, unsigned int crop_width, unsigned int crop_height);


//! * Decompress a subregion of JPEG image to an RGB, grayscale, or CMYK image.
//! * This function scale the decoded image to fit the output dims
/*!
  \param handle  TJPeg handle
  \param jpegBuf compressed jpeg image buffer
  \param jpegSize Size of the compressed data provided in the input_buffer
  \param dstBuf user provided output buffer
  \param width, pitch, height  width, stride and height of the allocated buffer
  \param flags  TJPEG flags
  \param crop_width, crop_height requested crop window
*/

int tjDecompress2_partial_scale(tjhandle handle, const unsigned char *jpegBuf,
                            unsigned long jpegSize, unsigned char *dstBuf,
                            int width, int pitch, int height, int pixelFormat,
                            int flags, unsigned int crop_width, unsigned int crop_height);
}