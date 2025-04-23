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


#ifndef __WARP_H__
#define __WARP_H__

#include "kernels.h"
#include "lens_distortion_remap.h"

//////////////////////////////////////////////////////////////////////
//! \brief The gray scale compute method modes
enum {
	STITCH_GRAY_SCALE_COMPUTE_METHOD_AVG  = 0, // Use: Y = (R + G + B) / 3
	STITCH_GRAY_SCALE_COMPUTE_METHOD_DIST = 1, // Use: Y = sqrt(R*R + G*G + B*B)
};

//////////////////////////////////////////////////////////////////////
//! \brief The valid pixel entry for 8 consecutive pixel locations.
//  For dummy entries in the buffer for alignment to 32 items, the invalid items shall be (vx_uint32)0xFFFFFFFF
typedef struct {
	vx_uint32 camId     :  5; // destination buffer/camera ID
	vx_uint32 reserved0 :  2; // reserved (shall be zero)
	vx_uint32 allValid  :  1; // all 8 consecutive pixels are valid
	vx_uint32 dstX      : 11; // destination pixel x-coordinate/8 (integer)
	vx_uint32 dstY      : 13; // destination pixel y-coordinate   (integer)
} StitchValidPixelEntry;

//////////////////////////////////////////////////////////////////////
//! \brief The warp pixel remap entry for 8 consecutive pixel locations.
//  Entry is invalid if srcX and srcY has all bits set to 1s.
//  For srcX and srcY coordinates, below fixed-point representation is used:
//     Q13.3
typedef struct {
	vx_uint16 srcX0; // source pixel (x,y) for (dstX*8+0,dstY) in Q13.3 format
	vx_uint16 srcY0;
	vx_uint16 srcX1; // source pixel (x,y) for (dstX*8+1,dstY) in Q13.3 format
	vx_uint16 srcY1;
	vx_uint16 srcX2; // source pixel (x,y) for (dstX*8+2,dstY) in Q13.3 format
	vx_uint16 srcY2;
	vx_uint16 srcX3; // source pixel (x,y) for (dstX*8+3,dstY) in Q13.3 format
	vx_uint16 srcY3;
	vx_uint16 srcX4; // source pixel (x,y) for (dstX*8+4,dstY) in Q13.3 format
	vx_uint16 srcY4;
	vx_uint16 srcX5; // source pixel (x,y) for (dstX*8+5,dstY) in Q13.3 format
	vx_uint16 srcY5;
	vx_uint16 srcX6; // source pixel (x,y) for (dstX*8+6,dstY) in Q13.3 format
	vx_uint16 srcY6;
	vx_uint16 srcX7; // source pixel (x,y) for (dstX*8+7,dstY) in Q13.3 format
	vx_uint16 srcY7;
} StitchWarpRemapEntry;

//////////////////////////////////////////////////////////////////////
//! \brief The kernel registration functions.
vx_status warp_publish(vx_context context);

//////////////////////////////////////////////////////////////////////
// Calculate buffer sizes and generate data in buffers for warp
//   CalculateLargestWarpBufferSizes  - useful when reinitialize is enabled
//   CalculateSmallestWarpBufferSizes - useful when reinitialize is disabled
//   GenerateWarpBuffers              - generate tables

vx_status CalculateLargestWarpBufferSizes(
	vx_uint32 numCamera,                  // [in] number of cameras
	vx_uint32 eqrWidth,                   // [in] output equirectangular image width
	vx_uint32 eqrHeight,                  // [in] output equirectangular image height
	vx_size * warpMapEntryCount           // [out] number of entries needed by warp map table
	);

vx_status CalculateSmallestWarpBufferSizes(
	vx_uint32 numCamera,                         // [in] number of cameras
	vx_uint32 eqrWidth,                          // [in] output equirectangular image width
	vx_uint32 eqrHeight,                         // [in] output equirectangular image height
	const vx_uint32 * validPixelCamMap,          // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_uint32 * paddedPixelCamMap,         // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight]
	vx_size * warpMapEntryCount                  // [out] number of entries needed by warp map table
	);

vx_status GenerateWarpBuffers(
	vx_uint32 numCamera,                         // [in] number of cameras
	vx_uint32 eqrWidth,                          // [in] output equirectangular image width
	vx_uint32 eqrHeight,                         // [in] output equirectangular image height
	const vx_uint32 * validPixelCamMap,          // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_uint32 * paddedPixelCamMap,         // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight]
	const StitchCoord2dFloat * camSrcMap,        // [in] camera coordinate mapping: size: [numCamera * eqrWidth * eqrHeight] (optional)
	vx_uint32 numCameraColumns,                  // [in] number of camera columns
	vx_uint32 camWidth,                          // [in] input camera image width
	vx_size   mapTableSize,                      // [in] size of warp/valid map table, in terms of number of entries
	StitchValidPixelEntry * validMap,            // [in] valid map table
	StitchWarpRemapEntry * warpMap,              // [in] warp map table
	vx_size * mapEntryCount                      // [out] number of entries added to warp/valid map table
	);

#endif //__WARP_H__
