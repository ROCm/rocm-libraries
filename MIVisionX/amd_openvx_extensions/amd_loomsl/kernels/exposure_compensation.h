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

#ifndef __EXPOSURE_COMPENSATION_H__
#define __EXPOSURE_COMPENSATION_H__

#include "exp_comp.h"

//////////////////////////////////////////////////////////////////////
//! \brief The exposure comp calc entry.
typedef struct {
	vx_uint32 camId   :  6; // destination buffer/camera ID
	vx_uint32 dstX    : 12; // destination pixel x-coordinate/8 (integer)
	vx_uint32 dstY    : 14; // destination pixel y-coordinate/2 (integer)
	vx_uint32 start_x :  8; // starting pixel x-coordinate within the 128x32 block
	vx_uint32 start_y :  8; // starting pixel y-coordinate within the 128x32 block
	vx_uint32 end_x   :  8; // ending pixel x-coordinate within the 128x32 block
	vx_uint32 end_y   :  8; // ending pixel y-coordinate within the 128x32 block
} StitchExpCompCalcEntry;

//////////////////////////////////////////////////////////////////////
//! \brief The overlap pixel entry.
typedef struct {
	vx_uint32 camId0  :  5; // destination buffer/camera ID
	vx_uint32 start_x : 14; // destination start pixel x-coordinate
	vx_uint32 start_y : 13; // destination start pixel y-coordinate
	vx_uint32 end_x   :  7; // ending pixel x-coordinate within the 128x32 block
	vx_uint32 end_y   :  5; // ending pixel y-coordinate within the 128x32 block
	vx_uint32 camId1  :  5; // values [0..30] overlapping cameraId; 31 indicates invalid cameraId
	vx_uint32 camId2  :  5; // values [0..30] overlapping cameraId; 31 indicates invalid cameraId
	vx_uint32 camId3  :  5; // values [0..30] overlapping cameraId; 31 indicates invalid cameraId
	vx_uint32 camId4  :  5; // values [0..30] overlapping cameraId; 31 indicates invalid cameraId
} StitchOverlapPixelEntry;

//////////////////////////////////////////////////////////////////////
//! \brief The kernel registration functions.
vx_status exposure_comp_calcErrorFn_publish(vx_context context);
vx_status exposure_comp_solvegains_publish(vx_context context);
vx_status exposure_comp_applygains_publish(vx_context context);
vx_status exposure_comp_calcRGBErrorFn_publish(vx_context context);

//////////////////////////////////////////////////////////////////////
// Calculate buffer sizes and generate data in buffers for exposure compensation
//   CalculateLargestExpCompBufferSizes  - useful when reinitialize is enabled
//   CalculateSmallestExpCompBufferSizes - useful when reinitialize is disabled
//   GenerateExpCompBuffers              - generate tables

vx_status CalculateLargestExpCompBufferSizes(
	vx_uint32 numCamera,                    // [in] number of cameras
	vx_uint32 eqrWidth,                     // [in] output equirectangular image width
	vx_uint32 eqrHeight,                    // [in] output equirectangular image height
	vx_size * validTableEntryCount,         // [out] number of entries needed by expComp valid table
	vx_size * overlapTableEntryCount        // [out] number of entries needed by expComp overlap table
	);

vx_status CalculateSmallestExpCompBufferSizes(
	vx_uint32 numCamera,                           // [in] number of cameras
	vx_uint32 eqrWidth,                            // [in] output equirectangular image width
	vx_uint32 eqrHeight,                           // [in] output equirectangular image height
	const vx_uint32 * validPixelCamMap,            // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapValid,   // [in] overlap regions: overlapValid[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	const vx_uint32 * validCamOverlapInfo,         // [in] camera overlap info - use "validCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32]
	const vx_uint32 * paddedPixelCamMap,           // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight](optional)
	const vx_rectangle_t * const * overlapPadded,  // [in] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i(optional)
	const vx_uint32 * paddedCamOverlapInfo,        // [in] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32](optional)
	vx_size * validTableEntryCount,                // [out] number of entries needed by expComp valid table
	vx_size * overlapTableEntryCount               // [out] number of entries needed by expComp overlap table
	);

vx_status GenerateExpCompBuffers(
	vx_uint32 numCamera,                           // [in] number of cameras
	vx_uint32 eqrWidth,                            // [in] output equirectangular image width
	vx_uint32 eqrHeight,                           // [in] output equirectangular image height
	const vx_uint32 * validPixelCamMap,            // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapValid,   // [in] overlap regions: overlapValid[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	const vx_uint32 * validCamOverlapInfo,         // [in] camera overlap info - use "validCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32]
	const vx_uint32 * paddedPixelCamMap,           // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight](optional)
	const vx_rectangle_t * const * overlapPadded,  // [in] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i(optional)
	const vx_uint32 * paddedCamOverlapInfo,        // [in] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32](optional)
	vx_size validTableSize,                        // [in] size of valid table, in terms of number of entries
	vx_size overlapTableSize,                      // [in] size of overlap table, in terms of number of entries
	StitchExpCompCalcEntry * validTable,           // [out] expComp valid table
	StitchOverlapPixelEntry * overlapTable,        // [out] expComp overlap table
	vx_size * validTableEntryCount,                // [out] number of entries needed by expComp valid table
	vx_size * overlapTableEntryCount,              // [out] number of entries needed by expComp overlap table
	vx_int32 * overlapPixelCountMatrix             // [out] expComp overlap pixel count matrix: size: [numCamera * numCamera]
	);

#endif // __EXPOSURE_COMPENSATION_H__
