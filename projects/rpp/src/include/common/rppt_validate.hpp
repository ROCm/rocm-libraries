/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#ifndef RPPT_VALIDATE_OPERATIONS_FUNCTIONS
#define RPPT_VALIDATE_OPERATIONS_FUNCTIONS

#include <stdlib.h>

#include <iostream>

#include "rpp.h"
#include "rppdefs.h"

inline RppLayoutParams get_layout_params(RpptLayout layout, Rpp32u channels) {
    RppLayoutParams layoutParams;
    if (layout == RpptLayout::NCHW || layout == RpptLayout::NCDHW) {
        if (channels == 1)  // PLN1
        {
            layoutParams.channelParam = 1;
            layoutParams.bufferMultiplier = 1;
        } else if (channels == 3)  // PLN3
        {
            layoutParams.channelParam = 3;
            layoutParams.bufferMultiplier = 1;
        }
    } else if (layout == RpptLayout::NHWC || layout == RpptLayout::NDHWC) {
        // PKD
        layoutParams.channelParam = 1;
        layoutParams.bufferMultiplier = channels;
    }
    return layoutParams;
}

inline int check_roi_out_of_bounds(RpptROIPtr roiPtrImage, RpptDescPtr srcDescPtr,
                                   RpptRoiType type) {
    int x = 0, y = 0, w = 0, h = 0;
    if (type == RpptRoiType::XYWH) {
        x = ((0 <= roiPtrImage->xywhROI.xy.x) && (roiPtrImage->xywhROI.xy.x < srcDescPtr->w))
                ? roiPtrImage->xywhROI.xy.x
                : -1;
        y = ((0 <= roiPtrImage->xywhROI.xy.y) && (roiPtrImage->xywhROI.xy.y < srcDescPtr->h))
                ? roiPtrImage->xywhROI.xy.y
                : -1;
        w = ((roiPtrImage->xywhROI.roiWidth) <= srcDescPtr->w) ? roiPtrImage->xywhROI.roiWidth : -1;
        h = ((roiPtrImage->xywhROI.roiHeight) <= srcDescPtr->h) ? roiPtrImage->xywhROI.roiHeight
                                                                : -1;
    } else if (type == RpptRoiType::LTRB) {
        x = ((0 <= roiPtrImage->ltrbROI.lt.x) && (roiPtrImage->ltrbROI.lt.x < srcDescPtr->w))
                ? roiPtrImage->ltrbROI.lt.x
                : -1;
        y = ((0 <= roiPtrImage->ltrbROI.lt.y) && (roiPtrImage->ltrbROI.lt.y < srcDescPtr->h))
                ? roiPtrImage->ltrbROI.lt.y
                : -1;
        w = ((0 <= roiPtrImage->ltrbROI.rb.x) && (roiPtrImage->ltrbROI.rb.x < srcDescPtr->w))
                ? roiPtrImage->ltrbROI.rb.x - roiPtrImage->ltrbROI.lt.x + 1
                : -1;
        h = ((0 <= roiPtrImage->ltrbROI.rb.y) && (roiPtrImage->ltrbROI.rb.y < srcDescPtr->h))
                ? roiPtrImage->ltrbROI.rb.y - roiPtrImage->ltrbROI.lt.y + 1
                : -1;
    } else {
        // Invalid ROI type
        return -1;
    }

    if ((x < 0) || (y < 0) || (w < 0) || (h < 0)) {
        return -1;
    }

    return 0;
}

// Generic-tensor ops (Misc domain, RpptGenericDesc-based) require strides that are densely packed
// per sample - no row/plane padding slack between the innermost extent and the stride of the next
// outer dimension. This walks the per-sample axes (index 1..numDims-1) from innermost outward and
// checks each stride equals the running dense product; the batch axis (index 0) is left unchecked,
// since batch spacing is unrelated to the row-padding convention this guards against.
inline bool rppt_generic_desc_is_dense(RpptGenericDescPtr descPtr) {
    Rpp32u expectedStride = 1;
    for (int i = static_cast<int>(descPtr->numDims) - 1; i >= 1; i--) {
        if (descPtr->strides[i] != expectedStride) return false;
        expectedStride *= descPtr->dims[i];
    }
    return true;
}

#endif  // RPPT_VALIDATE_OPERATIONS_FUNCTIONS
