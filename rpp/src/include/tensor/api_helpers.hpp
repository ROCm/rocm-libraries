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

#ifndef API_HELPERS_HPP
#define API_HELPERS_HPP

#include "rpp.h"
#include "rppdefs.h"

// sets descriptor dimensions and strides for descriptor used for fog augmentation
inline void set_fog_mask_descriptor(RpptDescPtr descPtr, Rpp32s batchSize, Rpp32s maxHeight, Rpp32s maxWidth, Rpp32s numChannels)
{
    descPtr->numDims = 4;
    descPtr->offsetInBytes = 0;
    descPtr->dataType = RpptDataType::F32;
    descPtr->layout = RpptLayout::NCHW;
    descPtr->n = batchSize;
    descPtr->h = maxHeight;
    descPtr->w = maxWidth;
    descPtr->c = numChannels;
    descPtr->strides = {descPtr->c * descPtr->w * descPtr->h,  1, descPtr->w, 1};
}

#endif /* API_HELPERS_HPP */
