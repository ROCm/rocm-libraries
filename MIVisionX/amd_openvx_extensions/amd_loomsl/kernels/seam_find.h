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


#ifndef __SEAM_FIND_H__
#define __SEAM_FIND_H__

#include "kernels.h"

/*********************************************************************
SeamFind Data Structures
**********************************************************************/
#define ENABLE_VERTICAL_SEAM 1
#define ENABLE_HORIZONTAL_SEAM 1

#define VERTICAL_SEAM 0
#define HORIZONTAL_SEAM 1
#define DIAGONAL_SEAM 2

//! \brief The Seam Information struct.
typedef struct {
	vx_int16 cam_id_1;		// Overlap CAM ID - 1
	vx_int16 cam_id_2;		// Overlap CAM ID - 2
	vx_int16 start_x;		// Overlap Rectangle start x
	vx_int16 end_x;			// Overlap Rectangle end x
	vx_int16 start_y;		// Overlap Rectangle start y
	vx_int16 end_y;			// Overlap Rectangle end y
	vx_int32 offset;		// Offset location in accumulate buffer
} StitchSeamFindInformation;

//! \brief The Seam Find Preference attributes.
typedef struct {
	vx_int16 type;			// Overlap type: 0 - Vertical Overlap, 1 - Hortzontal Overlap,  2 - Diagonal Overlap
	vx_int16 seam_type_num;	// Overlap type ID - vertical/horizontal overlap ID
	vx_int16 start_frame;	// Start frame to calculate the seam
	vx_int16 frequency;		// Frequency to calculate the seam		
	vx_int16 quality;		// Quality of the calculated the seam		
	vx_int16 priority;		// Priority to calculate the seam	
	vx_int16 seam_lock;		// Lock the seam after scene change is detected for n frames
	vx_int16 scene_flag;	// Scene change detection flag
} StitchSeamFindPreference;

//! \brief The valid pixel entry for Seam Find.
typedef struct {
	vx_int16 dstX;		 // destination pixel x-coordinate (integer)
	vx_int16 dstY;		 // destination pixel y-coordinate  (integer)
	vx_int16 height;	 // y - Height
	vx_int16 width;		 // x - Width
	vx_int16 OverLapX;   // Absolute Overlap destination pixel x-coordinate (integer)
	vx_int16 OverLapY;   // Absolute Overlap destination pixel y-coordinate (integer)
	vx_int16 CAMERA_ID_1;// Overlap Camera i
	vx_int16 ID;		 // Overlap Number	
} StitchSeamFindValidEntry;

//! \brief The valid pixel entry for Weight Manipulation Seam Find.
typedef struct {
	vx_int16 x;				//pixel x-coordinate (integer)
	vx_int16 y;				//pixel y-coordinate (integer)
	vx_int16 cam_id_1;		// Overlap Camera i
	vx_int16 cam_id_2;		// Overlap Camera j
	vx_int16 overlap_id;	// Overlap Number	
	vx_int16 overlap_type;	// Overlap Type: 0: Vert Seam 1: Hort Seam 2: Diag Seam
} StitchSeamFindWeightEntry;

//! \brief The Output Accum entry for Seam Find.
typedef struct{
	vx_int16 parent_x;		//pixel x-coordinate of parent (integer)
	vx_int16 parent_y;		//pixel y-coordinate of parent (integer)
	vx_int32 value;			//value accumulated
	vx_int32 propagate;		//propogate paths from start to finish
}StitchSeamFindAccumEntry;

//! \brief The path entry for Seam Find.
typedef struct {
	vx_int16 min_pixel;			//pixel x/y - coordinate (integer)
	vx_int16 weight_value_i;    //mask Value             (integer)
} StitchSeamFindPathEntry;

//! \brief The Scene Change Segments for Seam Find.
#define MAX_SEGMENTS 24
#define MAX_SEAM_BYTES 8
typedef struct{
	vx_uint8 segment[MAX_SEGMENTS][MAX_SEAM_BYTES];
}StitchSeamFindSceneEntry;

//! \brief The Output Accum entry for Seam Find.
typedef struct{
	vx_int16 parent_x;
	vx_int16 parent_y;
	vx_int32 value;
}StitchSeamFindAccum;

//////////////////////////////////////////////////////////////////////
//! \brief The kernel registration functions.
vx_status seamfind_model_publish(vx_context context);
vx_status seamfind_scene_detect_publish(vx_context context);
vx_status seamfind_cost_generate_publish(vx_context context);
vx_status seamfind_cost_accumulate_publish(vx_context context);
vx_status seamfind_path_trace_publish(vx_context context);
vx_status seamfind_set_weights_publish(vx_context context);
vx_status seamfind_analyze_publish(vx_context context);

//////////////////////////////////////////////////////////////////////
// Calculate buffer sizes and generate data in buffers for seam find
//   CalculateLargestSeamFindBufferSizes  - useful when reinitialize is enabled
//   CalculateSmallestSeamFindBufferSizes - useful when reinitialize is disabled
//   GenerateSeamFindBuffers              - generate tables

vx_status CalculateLargestSeamFindBufferSizes(
	vx_uint32 numCamera,                    // [in] number of cameras
	vx_uint32 eqrWidth,                     // [in] output equirectangular image width
	vx_uint32 eqrHeight,                    // [in] output equirectangular image height
	vx_size * seamFindValidEntryCount,      // [out] number of entries needed by seamFind valid table
	vx_size * seamFindWeightEntryCount,     // [out] number of entries needed by seamFind weight table
	vx_size * seamFindAccumEntryCount,      // [out] number of entries needed by seamFind accum table
	vx_size * seamFindPrefInfoEntryCount,   // [out] number of entries needed by seamFind pref/info table
	vx_size * seamFindPathEntryCount        // [out] number of entries needed by seamFind path table
	);

vx_status CalculateSmallestSeamFindBufferSizes(
	vx_uint32 numCamera,                           // [in] number of cameras
	vx_uint32 eqrWidth,                            // [in] output equirectangular image width
	vx_uint32 eqrHeight,                           // [in] output equirectangular image height
	const camera_params * camera_par,               // [in] camera parameters
	const vx_uint32 * validPixelCamMap,            // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapValid,   // [in] overlap regions: overlapValid[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	const vx_uint32 * validCamOverlapInfo,         // [in] camera overlap info - use "validCamOverlapInfo[cam_i] & (1 << cam_j)": size: [LIVE_STITCH_MAX_CAMERAS]
	const vx_uint32 * paddedPixelCamMap,           // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight](optional)
	const vx_rectangle_t * const * overlapPadded,  // [in] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i(optional)
	const vx_uint32 * paddedCamOverlapInfo,        // [in] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [LIVE_STITCH_MAX_CAMERAS](optional)
	const vx_float32 * live_stitch_attr,           // [in] attributes
	vx_size * seamFindValidEntryCount,             // [out] number of entries needed by seamFind valid table
	vx_size * seamFindWeightEntryCount,            // [out] number of entries needed by seamFind weight table
	vx_size * seamFindAccumEntryCount,             // [out] number of entries needed by seamFind accum table
	vx_size * seamFindPrefInfoEntryCount,          // [out] number of entries needed by seamFind pref/info table
	vx_size * seamFindPathEntryCount               // [out] number of entries needed by seamFind path table
	);

vx_status GenerateSeamFindBuffers(
	vx_uint32 numCamera,                            // [in] number of cameras
	vx_uint32 eqrWidth,                             // [in] output equirectangular image width
	vx_uint32 eqrHeight,                            // [in] output equirectangular image height
	const camera_params * camera_par,               // [in] camera parameters
	const vx_uint32 * validPixelCamMap,             // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapValid,    // [in] overlap regions: overlapValid[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	const vx_uint32 * validCamOverlapInfo,          // [in] camera overlap info - use "validCamOverlapInfo[cam_i] & (1 << cam_j)": size: [LIVE_STITCH_MAX_CAMERAS]
	const vx_uint32 * paddedPixelCamMap,            // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight](optional)
	const vx_rectangle_t * const * overlapPadded,   // [in] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i(optional)
	const vx_uint32 * paddedCamOverlapInfo,         // [in] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [LIVE_STITCH_MAX_CAMERAS](optional)
	const vx_float32 * live_stitch_attr,            // [in] attributes
	vx_size validTableSize,                         // [in] size of seamFind valid table in terms of number of entries
	vx_size weightTableSize,                        // [in] size of seamFind weight table in terms of number of entries
	vx_size accumTableSize,                         // [in] size of seamFind accum table in terms of number of entries
	vx_size prefInfoTableSize,                      // [in] size of seamFind pref/info table in terms of number of entries
	StitchSeamFindValidEntry * validTable,          // [out] valid table
	StitchSeamFindWeightEntry * weightTable,        // [out] weight table
	StitchSeamFindAccumEntry * accumTable,          // [out] accum table
	StitchSeamFindPreference * prefTable,           // [out] preference table
	StitchSeamFindInformation * infoTable,          // [out] info table
	vx_size * seamFindValidEntryCount,              // [out] number of entries needed by seamFind valid table
	vx_size * seamFindWeightEntryCount,             // [out] number of entries needed by seamFind weight table
	vx_size * seamFindAccumEntryCount,              // [out] number of entries needed by seamFind accum table
	vx_size * seamFindPrefInfoEntryCount,           // [out] number of entries needed by seamFind pref/info table
	vx_size * seamFindPathEntryCount                // [out] number of entries needed by seamFind path table
	);

#endif //__SEAM_FIND_H__
