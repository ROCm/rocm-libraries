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


#ifndef __MULTIBAND_BLENDER_H__
#define __MULTIBAND_BLENDER_H__

#include "kernels.h"

/*********************************************************************
Multiband blend Data Structures
**********************************************************************/
typedef struct{
	vx_image WeightPyrImgGaussian;
	vx_image DstPyrImgGaussian;
	vx_image DstPyrImgLaplacian;
	vx_image DstPyrImgLaplacianRec;
	vx_node WeightHSGNode;
	vx_node SourceHSGNode;
	vx_node UpscaleSubtractNode;
	vx_node BlendNode;
	vx_node UpscaleAddNode;
	vx_node LaplacianReconNode;
	vx_uint32 valid_array_offset;		// in number of elements
}StitchMultibandData;

typedef struct {
	vx_uint32 camId   :  5; // destination buffer/camera ID
	vx_uint32 dstX    : 14; // destination pixel x-coordinate (integer)
	vx_uint32 dstY    : 13; // destination pixel y-coordinate (integer)
	vx_uint32 last_x  :  8; // ending pixel x-coordinate within the 64x16 block
	vx_uint32 last_y  :  8; // ending pixel y-coordinate within the 64x16 block
	vx_uint32 skip_x  :  8; // starting pixel x-coordinate within the 64x16 block
	vx_uint32 skip_y  :  8; // starting pixel y-coordinate within the 64x16 block
} StitchBlendValidEntry;

//////////////////////////////////////////////////////////////////////
//! \brief The kernel registration functions.
vx_status multiband_blend_publish(vx_context context);

//////////////////////////////////////////////////////////////////////
// Calculate buffer sizes and generate data in buffers for blend
//   CalculateLargestBlendBufferSizes  - useful when reinitialize is enabled
//   CalculateSmallestBlendBufferSizes - useful when reinitialize is disabled
//   GenerateBlendBuffers              - generate tables

vx_uint32 CalculateLargestBlendBufferSizes(
	vx_uint32 numCamera,                    // [in] number of cameras
	vx_uint32 eqrWidth,                     // [in] output equirectangular image width
	vx_uint32 eqrHeight,                    // [in] output equirectangular image height
	vx_uint32 numBands,						// [in] number of bands in multiband blend
	vx_size * blendOffsetIntoBuffer,        // [out] individual level offset table: size [numBands]
	vx_size * blendOffsetEntryCount         // [out] number of entries needed by blend offset table
	);

vx_uint32 CalculateSmallestBlendBufferSizes(
	vx_uint32 numCamera,                           // [in] number of cameras
	vx_uint32 eqrWidth,                            // [in] output equirectangular image width
	vx_uint32 eqrHeight,                           // [in] output equirectangular image height
	vx_uint32 numBands,						       // [in] number of bands in multiband blend
	const vx_uint32 * validPixelCamMap,            // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_uint32 * paddedPixelCamMap,           // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapPadded,  // [in] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	const vx_uint32 * paddedCamOverlapInfo,        // [in] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32]
	vx_size * blendOffsetIntoBuffer,               // [out] individual level offset table: size [numBands]
	vx_size * blendOffsetEntryCount                // [out] number of entries needed by blend offset table
	);

vx_uint32 GenerateBlendBuffers(
	vx_uint32 numCamera,                             // [in] number of cameras
	vx_uint32 eqrWidth,                              // [in] output equirectangular image width
	vx_uint32 eqrHeight,                             // [in] output equirectangular image height
	vx_uint32 numBands,						         // [in] number of bands in multiband blend
	const vx_uint32 * validPixelCamMap,              // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_uint32 * paddedPixelCamMap,             // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapPadded,    // [in] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	const vx_uint32 * paddedCamOverlapInfo,          // [in] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32]
	const vx_size * blendOffsetIntoBuffer,           // [in] individual level offset table: size [numBands]
	vx_size blendOffsetTableSize,                    // [in] size of blend offset table
	StitchBlendValidEntry * blendOffsetTable         // [out] blend offset table
	);

#endif //__MULTIBAND_BLENDER_H__
