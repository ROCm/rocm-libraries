/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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


#ifndef __MERGE_H__
#define __MERGE_H__

#include "kernels.h"
#include "lens_distortion_remap.h"

//////////////////////////////////////////////////////////////////////
//! \brief The merge cameraId packing within U016 pixel entry.
typedef struct {
	vx_uint16 camId0    : 5; // values 0..30 are valid; 31 indicates invalid cameraId
	vx_uint16 camId1    : 5; // values 0..30 are valid; 31 indicates invalid cameraId
	vx_uint16 camId2    : 5; // values 0..30 are valid; 31 indicates invalid cameraId
	vx_uint16 reserved0 : 1; // reserved (shall be zero)
} StitchMergeCamIdEntry;

//////////////////////////////////////////////////////////////////////
//! \brief The kernel registration functions.
vx_status merge_publish(vx_context context);

//////////////////////////////////////////////////////////////////////
// Generate data in buffers for merge
vx_status GenerateMergeBuffers(
	vx_uint32 numCamera,                  // [in] number of cameras
	vx_uint32 eqrWidth,                   // [in] output equirectangular image width
	vx_uint32 eqrHeight,                  // [in] output equirectangular image height
	const vx_uint32 * validPixelCamMap,   // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_uint32 * paddedPixelCamMap,  // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight] (optional)
	vx_uint32  camIdStride,               // [in] stride (in bytes) of camId table (image)
	vx_uint32  camGroup1Stride,           // [in] stride (in bytes) of camGroup1 table (image)
	vx_uint32  camGroup2Stride,           // [in] stride (in bytes) of camGroup2 table (image)
	vx_uint8 * camIdBuf,                  // [out] camId table (image)
	StitchMergeCamIdEntry * camGroup1Buf, // [out] camId Group1 table (image)
	StitchMergeCamIdEntry * camGroup2Buf  // [out] camId Group2 table (image)
	);

//////////////////////////////////////////////////////////////////////
// Generate default merge mask image
vx_status GenerateDefaultMergeMaskImage(
	vx_uint32 numCamera,                  // [in] number of cameras
	vx_uint32 eqrWidth,                   // [in] output equirectangular image width
	vx_uint32 eqrHeight,                  // [in] output equirectangular image height
	const vx_uint8 * defaultCamIndex,     // [in] default camera index (255 refers to no camera): size: [eqrWidth * eqrHeight] (optional)
	vx_uint32  maskStride,                // [in] stride (in bytes) of mask image
	vx_uint8 * maskBuf                    // [out] mask image buffer: size: [eqrWidth * eqrHeight * numCamera]
	);

#endif //__MERGE_H__
