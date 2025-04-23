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
#define _CRT_SECURE_NO_WARNINGS
#include "lens_distortion_remap.h"
#include "kernels.h"
#define DUMP_BUFFERS_INITIALIZE	0
#define PROFILE_STARTUP_TIME	0

template<class T> inline const T& max(const T& a, const T& b)
{
	return b < a ? a : b;
}

template<class T> inline const T& min(const T& a, const T& b)
{
	return b < a ? b : a;
}

inline void rect_bound(vx_rectangle_t &rect, const vx_uint32 &x, const vx_uint32 &y)
{
	if (x < rect.start_x) rect.start_x = x;
	if (y < rect.start_y) rect.start_y = y;
	if (x > rect.end_x) rect.end_x = x;
	if (y > rect.end_y) rect.end_y = y;
}

#if PROFILE_STARTUP_TIME
#define NOMINMAX
#include <Windows.h>
#endif
extern void ls_printf(const char * format, ...);
extern vx_status DumpImage(vx_image img, const char * fileName);
extern vx_status DumpArray(vx_array arr, const char * fileName);
extern vx_status DumpBuffer(const vx_uint8 * buf, vx_size size, const char * fileName);


//! \brief Function to Compute M.
static void ComputeM(float * M, float th, float fi, float sy)
{
	float sth = sinf(th), cth = cosf(th);
	float sfi = sinf(fi), cfi = cosf(fi);
	float ssy = sinf(sy), csy = cosf(sy);
	M[0] = sth*ssy*sfi + cth*csy; M[1] = ssy*cfi; M[2] = cth*ssy*sfi - sth*csy;
	M[3] = sth*csy*sfi - cth*ssy; M[4] = csy*cfi; M[5] = cth*csy*sfi + sth*ssy;
	M[6] = sth    *cfi;          M[7] = -sfi;     M[8] = cth    *cfi;
}

//! \brief Matix Multiplication Function.
static void MatMul3x3(float * C, const float * A, const float * B)
{
	const float * At = A;
	for (int i = 0; i < 3; i++, At += 3) {
		const float * Bt = B;
		for (int j = 0; j < 3; j++, Bt++, C++) {
			*C = At[0] * Bt[0] + At[1] * Bt[3] + At[2] * Bt[6];
		}
	}
}

//! \brief Matix Multiplication Function.
static inline void MatMul3x1(float * Y, const float * M, const float * X)
{
	Y[0] = M[0] * X[0] + M[1] * X[1] + M[2] * X[2];
	Y[1] = M[3] * X[0] + M[4] * X[1] + M[5] * X[2];
	Y[2] = M[6] * X[0] + M[7] * X[1] + M[8] * X[2];
}

vx_status CalculateCameraWarpParameters(
	vx_uint32 numCamera,               // number of cameras
	vx_uint32 camWidth, vx_uint32 camHeight, // [in] individual camera dimensions
	const rig_params * rigParam,       // rig configuration
	const camera_params * camParam,    // individual camera configuration
	float * Mcam,                      // M matrices: one 3x3 per camera
	float * Tcam,                      // T vector: one 3x1 per camera
	float * fcam,                      // f vector: one 2x1 per camera
	float * Mr                         // M matrix: 3x3 for the rig
	)
{
	float deg2rad = (float)M_PI / 180.0f;
	ComputeM(Mr, rigParam->yaw * deg2rad, rigParam->pitch * deg2rad, rigParam->roll * deg2rad);
	for (vx_uint32 cam = 0; cam < numCamera; cam++)
	{
		float Mc[9];
		ComputeM(Mc, camParam[cam].focal.yaw * deg2rad, camParam[cam].focal.pitch * deg2rad, camParam[cam].focal.roll * deg2rad);
		MatMul3x3(&Mcam[cam * 9], Mc, Mr);
		if (rigParam->d > 0.0f) {
			Tcam[cam * 3 + 0] = camParam[cam].focal.tx / rigParam->d;
			Tcam[cam * 3 + 1] = camParam[cam].focal.ty / rigParam->d;
			Tcam[cam * 3 + 2] = camParam[cam].focal.tz / rigParam->d;
		}
		else {
			Tcam[cam * 3 + 0] = Tcam[cam * 3 + 1] = Tcam[cam * 3 + 2] = 0.0f;
		}
		fcam[cam * 2 + 1] = 0.5f * (camParam[cam].lens.haw > 0 ? camParam[cam].lens.haw : (float)camWidth);
		if (camParam[cam].lens.lens_type == ptgui_lens_rectilinear || camParam[cam].lens.lens_type == adobe_lens_rectilinear) { // rectilinear
			fcam[cam * 2 + 0] = 1.0f / tanf(0.5f * camParam[cam].lens.hfov * deg2rad);
		}
		else if (camParam[cam].lens.lens_type == ptgui_lens_fisheye_ff || camParam[cam].lens.lens_type == ptgui_lens_fisheye_circ || camParam[cam].lens.lens_type == adobe_lens_fisheye) { // fisheye
			fcam[cam * 2 + 0] = 1.0f / (0.5f * camParam[cam].lens.hfov * deg2rad);
		}
		else{ // unsupported lens
			printf("ERROR: CalculateCameraWarpParameters: lens_type = %d not supported [cam#%d]\n", camParam[cam].lens.lens_type, cam);
			return VX_ERROR_INVALID_TYPE;
		}
	}
	return VX_SUCCESS;
}

//////////////////////////////////////////////////////////////////////
// lens models
static inline float ptgui_lens_rectilinear_model(float th, float fr, float a, float b, float c, float d)
{
	float r = tanf(th) * fr;
	return r * (d + r * (c + r * (b + r * a)));
}
static inline float ptgui_lens_fisheye_model(float th, float fr, float a, float b, float c, float d)
{
	float r = th * fr;
	return r * (d + r * (c + r * (b + r * a)));
}
static inline float adobe_lens_rectilinear_model(float th, float fr, float k1, float k2, float k3, float k0)
{
	float r = tanf(th) * fr, r2 = r * r;
	return r * (1 + r2 * (k1 + r2 * (k2 + r2 * k3)));
}
static inline float adobe_lens_fisheye_model(float th, float fr, float k1, float k2, float k3, float k0)
{
	float r = th * fr, r2 = r * r;
	return r * (1 + r2 * (k1 + r2 * k2));
}

//////////////////////////////////////////////////////////////////////
// calculate padded region for the circular fisheye unwarpped image
static void CalculatePaddedRegion(
	vx_uint32 eqrWidth, vx_uint32 eqrHeight, // [in] output equirectangular dimensions
	vx_uint32 camId,                         // [in] camera index
	vx_uint32 * validPixelCamMap,            // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	vx_uint32 paddingPixelCount,             // [in] padding pixels around valid region
	vx_uint32 * paddedPixelCamMap            // [out] padded pixel camera index map: size: [eqrWidth * eqrHeight]
	)
{
	vx_uint32 camMapBit = 1 << camId;
	vx_uint32 loopPixels = (2 * paddingPixelCount) + 1;
	// dilate using separable filter for (N x 1) & (1 x N)
	for (vx_uint32 y_eqr = 0, pixelPosition = 0; y_eqr < (int)eqrHeight; y_eqr++) {
		for (vx_uint32 x_eqr = 0; x_eqr < (int)eqrWidth; x_eqr++, pixelPosition++) {
			vx_uint32 val = 0;
			vx_int32 X = (vx_int32)x_eqr - paddingPixelCount;
			// get the neighborhood of (x_eqr,y_eqr)
			for (vx_uint32 i = 0; i < loopPixels; i++){
				int pixel_location = (y_eqr * eqrWidth) + (X + i);
				if (pixel_location >= 0 && pixel_location < (int)(eqrWidth * eqrHeight)){
					val = validPixelCamMap[pixel_location] & camMapBit;
				}
				if (val) break;
			}
			// set padded pixel
			if (val){
				if (!(validPixelCamMap[pixelPosition] & camMapBit)){
					paddedPixelCamMap[pixelPosition] |= camMapBit;
				}
			}
		}
	}
	for (vx_uint32 y_eqr = 0, pixelPosition = 0; y_eqr < (int)eqrHeight; y_eqr++) {
		for (vx_uint32 x_eqr = 0; x_eqr < (int)eqrWidth; x_eqr++, pixelPosition++) {
			vx_uint32 val = 0;
			vx_int32 Y = (vx_int32)y_eqr - paddingPixelCount;
			// get the neighborhood of (x_eqr,y_eqr)
			for (vx_uint32 j = 0; j < loopPixels; j++){
				int pixel_location = ((Y + j) * eqrWidth) + x_eqr;
				if (pixel_location >= 0 && pixel_location < (int)(eqrWidth * eqrHeight)){
					val = validPixelCamMap[pixel_location] & camMapBit;
				}
				if (val) break;
			}
			// set padded pixel
			if (val){
				if (!(validPixelCamMap[pixelPosition] & camMapBit)){
					paddedPixelCamMap[pixelPosition] |= camMapBit;
				}
			}
		}
	}
	return;
}

//////////////////////////////////////////////////////////////////////
// calculate lens distorion and warp maps using lens model
static void CalculateLensDistortionAndWarpMapsUsingLensModel(
	vx_uint32 camWidth, vx_uint32 camHeight, // [in] individual camera dimensions
	vx_uint32 eqrWidth, vx_uint32 eqrHeight, // [in] output equirectangular dimensions
	vx_uint32 * validPixelCamMap,            // [out] valid pixel camera index map: size: [eqrWidth * eqrHeight] (optional)
	vx_uint32 paddingPixelCount,             // [in] padding pixels around valid region
	vx_uint32 * paddedPixelCamMap,           // [out] padded pixel camera index map: size: [eqrWidth * eqrHeight] (optional)
	StitchCoord2dFloat * camSrcMap,          // [out] camera coordinate mapping: size: [numCamera * eqrWidth * eqrHeight] (optional)
	vx_float32 * internalBufferForCamIndex,  // [tmp] buffer for internal use: size: [eqrWidth * eqrHeight] (optional)
	vx_uint8 * defaultCamIndex,              // [out] default camera index (255 refers to no camera): size: [eqrWidth * eqrHeight] (optional)
	vx_uint32 camId,                         // [in] camera index
	const float * M, const float * T, const float * f,
	float k1, float k2, float k3, float k0, float du0, float dv0, float r_crop,
	float left, float top, float right, float bottom,
	float(&lens_model_f)(float th, float fr, float k1, float k2, float k3, float k0),
	camera_lens_type lens_type
	)
{
	vx_uint32 camMapBit = 1 << camId;
	float pi_by_h = (float)M_PI / (float)eqrHeight;
	float center_x = du0 + (float)camWidth * 0.5f, center_y = dv0 + (float)camHeight * 0.5f;
	float rightMinus1 = right - 1, right2Minus2 = rightMinus1 * 2;
	float bottomMinus1 = bottom - 1, bottom2Minus2 = bottomMinus1 * 2;
	for (vx_uint32 y_eqr = 0, pixelPosition=0; y_eqr < (int)eqrHeight; y_eqr++) {
		float pe = (float)y_eqr * pi_by_h - (float)M_PI_2;
		float sin_pe = sinf(pe);
		float cos_pe = cosf(pe);
		for (vx_uint32 x_eqr = 0; x_eqr < (int)eqrWidth; x_eqr++, pixelPosition++) {
			float x_src = -1, y_src = -1;
			float te = (float)x_eqr * pi_by_h - (float)M_PI;
			float sin_te = sinf(te);
			float cos_te = cosf(te);
			float X[3] = { sin_te*cos_pe, sin_pe, cos_te*cos_pe };
			float Xt[3] = { X[0] - T[0], X[1] - T[1], X[2] - T[2] };
			float nfactor = sqrtf(Xt[0] * Xt[0] + Xt[1] * Xt[1] + Xt[2] * Xt[2]);
			Xt[0] /= nfactor;
			Xt[1] /= nfactor;
			Xt[2] /= nfactor;
			float Y[3];
			MatMul3x1(Y, M, Xt);
				// only consider pixels within 180 degrees field of view for non-circular fisheye lens
			if (Y[2] > 0.0f || lens_type == ptgui_lens_fisheye_circ) {
				float ph = atan2f(Y[1], Y[0]);
				float th = asinf(sqrtf((float)fmin(fmax(Y[0] * Y[0] + Y[1] * Y[1], 0.0f), 1.0f)));
				float rd = lens_model_f(th, f[0], k1, k2, k3, k0);
				float rr;
				x_src = f[1] * rd * cosf(ph);
				y_src = f[1] * rd * sinf(ph);
				rr = sqrtf(x_src*x_src + y_src*y_src);
				x_src += center_x;
				y_src += center_y;
				bool validCamIndex = false;
				if ((Y[2] > 0.0f) && validPixelCamMap &&
					(x_src >= left && x_src <= rightMinus1) && (y_src >= top && y_src <= bottomMinus1) &&
					(r_crop <= 0.0f || rr <= r_crop))
				{
					validCamIndex = true;
					// update camera map
					validPixelCamMap[pixelPosition] |= camMapBit;
				}
				else if (paddedPixelCamMap &&
					(x_src >= left - (float)paddingPixelCount) && (x_src <= rightMinus1 + paddingPixelCount) &&
					(y_src >= top - (float)paddingPixelCount) && (y_src <= bottomMinus1 + paddingPixelCount) &&
					((r_crop <= 0.0f) || (rr <= (r_crop + paddingPixelCount))))
				{
					// reflect the source coordinates
					if (x_src < left) x_src = left - x_src; else if (x_src >= rightMinus1) x_src = right2Minus2 - x_src;
					if (y_src < top) y_src = top - y_src; else if (y_src >= bottomMinus1) y_src = bottom2Minus2 - y_src;
					// update camera map
					if (lens_type != ptgui_lens_fisheye_circ){
						paddedPixelCamMap[pixelPosition] |= camMapBit;
					}
				}
				else{ x_src = y_src = -1.0f; }
				// pick default camera index
				if (validCamIndex) {
					vx_float32 zindicator = (float)fabs(Y[2]);
					if (zindicator > internalBufferForCamIndex[pixelPosition]) {
						defaultCamIndex[pixelPosition] = camId;
						internalBufferForCamIndex[pixelPosition] = zindicator;
					}
				}
			}
			// save source pixel coordinates, if requested
			if (camSrcMap) {
				if (x_src < left) x_src = left - x_src; else if (x_src >= rightMinus1) x_src = right2Minus2 - x_src;
				if (y_src < top) y_src = top - y_src; else if (y_src >= bottomMinus1) y_src = bottom2Minus2 - y_src;
				camSrcMap[pixelPosition].x = x_src;
				camSrcMap[pixelPosition].y = y_src;
			}
		}
	}
	// calculate paddedPixelCamMap for circular fisheye lens
	if (paddedPixelCamMap && (lens_type == ptgui_lens_fisheye_circ)){
		CalculatePaddedRegion(eqrWidth, eqrHeight, camId, validPixelCamMap, paddingPixelCount, paddedPixelCamMap);
	}
}

//////////////////////////////////////////////////////////////////////
// calculate lens distorion and warp maps from rig and camera configuration
vx_status CalculateLensDistortionAndWarpMaps(
	StitchInitializeData *pInitData,		 // [in] data pointer for init
	vx_uint32 numCamera,                     // [in] number of cameras
	vx_uint32 camWidth, vx_uint32 camHeight, // [in] individual camera dimensions
	vx_uint32 eqrWidth, vx_uint32 eqrHeight, // [in] output equirectangular dimensions
	const rig_params * rigParam,             // [in] rig configuration
	const camera_params * camParam,          // [in] individual camera configuration: size: [numCamera]
	vx_uint32 * validPixelCamMap,            // [out] valid pixel camera index map: size: [eqrWidth * eqrHeight] (optional)
	vx_uint32 paddingPixelCount,             // [in] padding pixels around valid region
	vx_uint32 * paddedPixelCamMap,           // [out] padded pixel camera index map: size: [eqrWidth * eqrHeight] (optional)
	StitchCoord2dFloat * camSrcMap,          // [out] camera coordinate mapping: size: [numCamera * eqrWidth * eqrHeight] (optional)
	vx_float32 * internalBufferForCamIndex,  // [tmp] buffer for internal use: size: [eqrWidth * eqrHeight] (optional)
	vx_uint8 * defaultCamIndex               // [out] default camera index (255 refers to no camera): size: [eqrWidth * eqrHeight] (optional)
	)
{
	// disable defaultCamIndex if tmp buffer is not specified (and vice versa)
	if (!internalBufferForCamIndex || !defaultCamIndex) {
		internalBufferForCamIndex = nullptr;
		defaultCamIndex = nullptr;
	}
	// supports upto 32 cameras
	if (numCamera > 32) {
		printf("ERROR: CalculateValidPixelMap: can't support %d cameras -- 32 is the current limit\n", numCamera);
		return VX_ERROR_NOT_SUPPORTED;
	}
#if PROFILE_STARTUP_TIME
	__int64 stime, etime;
	LARGE_INTEGER v;
	QueryPerformanceCounter(&v);
	stime = v.QuadPart;
#endif
	if (pInitData && pInitData->graphInitialize)
	{	
		ERROR_CHECK_STATUS(vxProcessGraph(pInitData->graphInitialize));

		// copy the data back to cpu memory for the rest of the processing
		vx_uint32 plane = 0;
		vx_rectangle_t rectFull = { 0, 0, 0, 0 };
		vx_imagepatch_addressing_t addr = { 0 };
		vx_uint8 * src = NULL;
		ERROR_CHECK_STATUS(vxQueryImage(pInitData->ValidPixelMap, VX_IMAGE_ATTRIBUTE_WIDTH, &rectFull.end_x, sizeof(rectFull.end_x)));
		ERROR_CHECK_STATUS(vxQueryImage(pInitData->ValidPixelMap, VX_IMAGE_ATTRIBUTE_HEIGHT, &rectFull.end_y, sizeof(rectFull.end_y)));
		// write all image planes from vx_image
		ERROR_CHECK_STATUS(vxAccessImagePatch(pInitData->ValidPixelMap, &rectFull, plane, &addr, (void **)&src, VX_READ_ONLY));
		vx_size width_in_bytes = (addr.dim_x * addr.stride_x);
		vx_uint8 *dst = (vx_uint8 *)validPixelCamMap;
		for (vx_uint32 y = 0; y < addr.dim_y; y += addr.step_y){
			vx_uint8 *srcp = (vx_uint8 *)vxFormatImagePatchAddress2d(src, 0, y, &addr);
			memcpy(dst, srcp, width_in_bytes); dst += width_in_bytes;
		}
		ERROR_CHECK_STATUS(vxCommitImagePatch(pInitData->ValidPixelMap, &rectFull, plane, &addr, src));

		if (pInitData->PaddedPixMap){
			ERROR_CHECK_STATUS(vxQueryImage(pInitData->PaddedPixMap, VX_IMAGE_ATTRIBUTE_WIDTH, &rectFull.end_x, sizeof(rectFull.end_x)));
			ERROR_CHECK_STATUS(vxQueryImage(pInitData->PaddedPixMap, VX_IMAGE_ATTRIBUTE_HEIGHT, &rectFull.end_y, sizeof(rectFull.end_y)));
			// write all image planes from vx_image
			addr = { 0 };
			src = NULL;
			ERROR_CHECK_STATUS(vxAccessImagePatch(pInitData->PaddedPixMap, &rectFull, plane, &addr, (void **)&src, VX_READ_ONLY));
			width_in_bytes = (addr.dim_x * addr.stride_x);
			dst = (vx_uint8 *)paddedPixelCamMap;
			for (vx_uint32 y = 0; y < addr.dim_y; y += addr.step_y){
				vx_uint8 *srcp = (vx_uint8 *)vxFormatImagePatchAddress2d(src, 0, y, &addr);
				memcpy(dst, srcp, width_in_bytes); dst += width_in_bytes;
			}
			ERROR_CHECK_STATUS(vxCommitImagePatch(pInitData->PaddedPixMap, &rectFull, plane, &addr, src));
		}

		// write all image planes from vx_image
		if (pInitData->DefaultCamMap){
			addr = { 0 };
			src = NULL;
			ERROR_CHECK_STATUS(vxQueryImage(pInitData->DefaultCamMap, VX_IMAGE_ATTRIBUTE_WIDTH, &rectFull.end_x, sizeof(rectFull.end_x)));
			ERROR_CHECK_STATUS(vxQueryImage(pInitData->DefaultCamMap, VX_IMAGE_ATTRIBUTE_HEIGHT, &rectFull.end_y, sizeof(rectFull.end_y)));
			ERROR_CHECK_STATUS(vxAccessImagePatch(pInitData->DefaultCamMap, &rectFull, plane, &addr, (void **)&src, VX_READ_ONLY));
			width_in_bytes = (addr.dim_x * addr.stride_x);
			dst = defaultCamIndex;
			for (vx_uint32 y = 0; y < addr.dim_y; y += addr.step_y){
				vx_uint8 *srcp = (vx_uint8 *)vxFormatImagePatchAddress2d(src, 0, y, &addr);
				memcpy(dst, srcp, width_in_bytes); dst += width_in_bytes;
			}
			ERROR_CHECK_STATUS(vxCommitImagePatch(pInitData->DefaultCamMap, &rectFull, plane, &addr, src));
		}

		if (pInitData->SrcCoordMap){
			// write all image planes from vx_image
			addr = { 0 };
			src = NULL;
			ERROR_CHECK_STATUS(vxQueryImage(pInitData->SrcCoordMap, VX_IMAGE_ATTRIBUTE_WIDTH, &rectFull.end_x, sizeof(rectFull.end_x)));
			ERROR_CHECK_STATUS(vxQueryImage(pInitData->SrcCoordMap, VX_IMAGE_ATTRIBUTE_HEIGHT, &rectFull.end_y, sizeof(rectFull.end_y)));
			ERROR_CHECK_STATUS(vxAccessImagePatch(pInitData->SrcCoordMap, &rectFull, plane, &addr, (void **)&src, VX_READ_ONLY));
			width_in_bytes = (addr.dim_x * addr.stride_x);
			dst = (vx_uint8 *)camSrcMap;
			for (vx_uint32 y = 0; y < addr.dim_y; y += addr.step_y){
				vx_uint8 *srcp = (vx_uint8 *)vxFormatImagePatchAddress2d(src, 0, y, &addr);
				memcpy(dst, srcp, width_in_bytes); dst += width_in_bytes;
			}
			ERROR_CHECK_STATUS(vxCommitImagePatch(pInitData->SrcCoordMap, &rectFull, plane, &addr, src));
		}
#if DUMP_BUFFERS_INITIALIZE
//		DumpImage(pInitData->ValidPixelMap, "ValidCamMapGpu.bin");
//		DumpImage(pInitData->PaddedPixMap, "PaddedCamMapGpu.bin");
//		DumpImage(pInitData->SrcCoordMap, "SrcCordMapGpu.bin");
//		DumpImage(pInitData->DefaultCamMap, "DefCamMapGpu.bin");
#endif
	}
	else
	{
		// compute camera warp parameters and check for supported lens types
		float Mcam[32 * 9], Tcam[32 * 3], fcam[32 * 2], Mr[3 * 3];
		vx_status status = CalculateCameraWarpParameters(numCamera, camWidth, camHeight, rigParam, camParam, Mcam, Tcam, fcam, Mr);
		if (status != VX_SUCCESS) return status;

		// cpu version
		// initialize buffers
		size_t totSize = eqrWidth * eqrHeight;
		if (validPixelCamMap) {
			memset(validPixelCamMap, 0, totSize*sizeof(vx_uint32));
		}
		if (paddedPixelCamMap) {
			memset(paddedPixelCamMap, 0, totSize*sizeof(vx_uint32));
		}
		if (defaultCamIndex) {
			memset(internalBufferForCamIndex, 0, totSize*sizeof(vx_uint32));
			memset(defaultCamIndex, 0xFF, totSize);
		}
		// compute valid pixels based on warp parameters
		const float * T = Tcam, *M = Mcam, *f = fcam;
		for (vx_uint32 cam = 0; cam < numCamera; cam++, T += 3, M += 9, f += 2) {
			// perform lens distortion and warp for each pixel in the equirectangular destination image
			const camera_lens_params * lens = &camParam[cam].lens;
			float k0 = 1.0f - (lens->k1 + lens->k2 + lens->k3);
			float left = 0, top = 0, right = (float)camWidth, bottom = (float)camHeight;
			if (lens->lens_type <= ptgui_lens_fisheye_circ && (lens->reserved[3] != 0 || lens->reserved[4] != 0 || lens->reserved[5] != 0 || lens->reserved[6] != 0)) {
				left = std::max(left, lens->reserved[3]);
				top = std::max(top, lens->reserved[4]);
				right = std::min(right, lens->reserved[5]);
				bottom = std::min(bottom, lens->reserved[6]);
			}
			if (lens->lens_type == ptgui_lens_rectilinear) {
				CalculateLensDistortionAndWarpMapsUsingLensModel(camWidth, camHeight, eqrWidth, eqrHeight,
					validPixelCamMap, paddingPixelCount, paddedPixelCamMap, &camSrcMap[cam * eqrWidth * eqrHeight],
					internalBufferForCamIndex, defaultCamIndex,
					cam, M, T, f, lens->k1, lens->k2, lens->k3, k0, lens->du0, lens->dv0, lens->r_crop,
					left, top, right, bottom, ptgui_lens_rectilinear_model, lens->lens_type);
			}
			else if (lens->lens_type == ptgui_lens_fisheye_ff || lens->lens_type == ptgui_lens_fisheye_circ) {
				CalculateLensDistortionAndWarpMapsUsingLensModel(camWidth, camHeight, eqrWidth, eqrHeight,
					validPixelCamMap, paddingPixelCount, paddedPixelCamMap, &camSrcMap[cam * eqrWidth * eqrHeight],
					internalBufferForCamIndex, defaultCamIndex,
					cam, M, T, f, lens->k1, lens->k2, lens->k3, k0, lens->du0, lens->dv0, lens->r_crop,
					left, top, right, bottom, ptgui_lens_fisheye_model, lens->lens_type);
			}
			else if (lens->lens_type == adobe_lens_rectilinear) {
				CalculateLensDistortionAndWarpMapsUsingLensModel(camWidth, camHeight, eqrWidth, eqrHeight,
					validPixelCamMap, paddingPixelCount, paddedPixelCamMap, &camSrcMap[cam * eqrWidth * eqrHeight],
					internalBufferForCamIndex, defaultCamIndex,
					cam, M, T, f, lens->k1, lens->k2, lens->k3, k0, lens->du0, lens->dv0, lens->r_crop,
					left, top, right, bottom, adobe_lens_rectilinear_model, lens->lens_type);
			}
			else if (lens->lens_type == adobe_lens_fisheye) {
				CalculateLensDistortionAndWarpMapsUsingLensModel(camWidth, camHeight, eqrWidth, eqrHeight,
					validPixelCamMap, paddingPixelCount, paddedPixelCamMap, &camSrcMap[cam * eqrWidth * eqrHeight],
					internalBufferForCamIndex, defaultCamIndex,
					cam, M, T, f, lens->k1, lens->k2, lens->k3, k0, lens->du0, lens->dv0, lens->r_crop,
					left, top, right, bottom, adobe_lens_fisheye_model, lens->lens_type);
			}
		}
#if DUMP_BUFFERS_INITIALIZE
		DumpBuffer((vx_uint8 *)paddedPixelCamMap, eqrWidth*eqrHeight * 4, "PaddedCamMap.bin");
#endif
	}
#if PROFILE_STARTUP_TIME
	QueryPerformanceCounter(&v);
	etime = v.QuadPart;
	QueryPerformanceFrequency(&v);
	__int64 denom = v.QuadPart;
	__int64 tot_time = ((etime - stime) * 1000) / denom;
	printf("CalculateLensDistortionAndWarpMaps:: tot time:%d ms\n", tot_time);
#endif

	return VX_SUCCESS;
}


//////////////////////////////////////////////////////////////////////
// calculate overlap regions and returns number of overlaps
vx_uint32 CalculateValidOverlapRegions(
	vx_uint32 numCamera,                 // [in] number of cameras
	vx_uint32 eqrWidth,                  // [in] output equirectangular image width
	vx_uint32 eqrHeight,                 // [in] output equirectangular image height
	const vx_uint32 * validPixelCamMap,  // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	vx_rectangle_t ** overlapValid,      // [out] overlap regions: overlapValid[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	vx_uint32 * validCamOverlapInfo,     // [out] camera overlap info - use "validCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32]
	const vx_uint32 * paddedPixelCamMap, // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight](optional)
	vx_rectangle_t ** overlapPadded,     // [out] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i(optional)
	vx_uint32 * paddedCamOverlapInfo     // [out] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32](optional)
	)
{
	// initialize camera overlap info
	memset(validCamOverlapInfo, 0, LIVE_STITCH_MAX_CAMERAS * sizeof(vx_uint32));
	if (paddedCamOverlapInfo) memset(paddedCamOverlapInfo, 0, LIVE_STITCH_MAX_CAMERAS * sizeof(vx_uint32));
	if (!paddedPixelCamMap)
	{
		overlapPadded = nullptr;
		paddedCamOverlapInfo = nullptr;
	}
	vx_uint32 validPixelOverlapCountMax = 0;
	vx_uint32 paddedPixelOverlapCountMax = 0;
	// initialize overlap region rectangles
	for (vx_uint32 cam = 0; cam < numCamera; cam++) {
		for (vx_uint32 i = 0; i <= cam; i++) {
			overlapValid[cam][i] = { eqrWidth, eqrHeight, 0, 0 };
			if (overlapPadded) {
				overlapPadded[cam][i] = { eqrWidth, eqrHeight, 0, 0 };
			}
		}
	}
	// calculate overlap region rectangles
	if (paddedPixelCamMap){
		for (vx_uint32 y_eqr = 0, pixelPosition = 0; y_eqr < (vx_uint32)eqrHeight; y_eqr++) {
			for (vx_uint32 x_eqr = 0; x_eqr < (vx_uint32)eqrWidth; x_eqr++, pixelPosition++) {
				vx_uint32 validCamMap = validPixelCamMap[pixelPosition];
				// update each of valid and overlaped regions
				validPixelOverlapCountMax = max(validPixelOverlapCountMax, GetOneBitCount(validCamMap));
				for (unsigned long camMapI = validCamMap; camMapI;) {
					// get cam_i
					vx_uint32 cam_i = GetOneBitPosition(camMapI);
					camMapI &= ~(1 << cam_i);
					// update overlapValid[cam_i][cam_i]
					rect_bound(overlapValid[cam_i][cam_i], x_eqr, y_eqr);
					for (unsigned long camMapJ = camMapI; camMapJ;) {
						vx_uint32 cam_j = GetOneBitPosition(camMapJ);
						validCamOverlapInfo[cam_i] |= (1 << cam_j);
						camMapJ &= ~(1 << cam_j);
						// update overlapValid[cam_i][cam_j]
						rect_bound(overlapValid[cam_i][cam_j], x_eqr, y_eqr);
					}
				}
				vx_uint32 paddedCamMap = validCamMap | paddedPixelCamMap[pixelPosition];
				paddedPixelOverlapCountMax = max(paddedPixelOverlapCountMax, GetOneBitCount(paddedCamMap));
				// update each of padded overlaped region
				for (unsigned long camMapI = paddedCamMap; camMapI;) {
					// get cam_i
					vx_uint32 cam_i = GetOneBitPosition(camMapI);
					camMapI &= ~(1 << cam_i);
					// update overlapPadded[cam_i][cam_i]
					rect_bound(overlapPadded[cam_i][cam_i], x_eqr, y_eqr);
					for (unsigned long camMapJ = camMapI; camMapJ;) {
						vx_uint32 cam_j = GetOneBitPosition(camMapJ);
						paddedCamOverlapInfo[cam_i] |= (1 << cam_j);
						camMapJ &= ~(1 << cam_j);
						// update overlapPadded[cam_i][cam_j]
						rect_bound(overlapPadded[cam_i][cam_j], x_eqr, y_eqr);
					}
				}
			}
		}
	}
	else
	{
		for (vx_uint32 y_eqr = 0, pixelPosition = 0; y_eqr < (vx_uint32)eqrHeight; y_eqr++) {
			for (vx_uint32 x_eqr = 0; x_eqr < (vx_uint32)eqrWidth; x_eqr++, pixelPosition++) {
				vx_uint32 validCamMap = validPixelCamMap[pixelPosition];
				// update each of valid and overlaped regions
				validPixelOverlapCountMax = max(validPixelOverlapCountMax, GetOneBitCount(validCamMap));
				for (unsigned long camMapI = validCamMap; camMapI;) {
					// get cam_i
					vx_uint32 cam_i = GetOneBitPosition(camMapI);
					camMapI &= ~(1 << cam_i);
					// update overlapValid[cam_i][cam_i]
					rect_bound(overlapValid[cam_i][cam_i], x_eqr, y_eqr);
					for (unsigned long camMapJ = camMapI; camMapJ;) {
						vx_uint32 cam_j = GetOneBitPosition(camMapJ);
						validCamOverlapInfo[cam_i] |= (1 << cam_j);
						camMapJ &= ~(1 << cam_j);
						// update overlapValid[cam_i][cam_j]
						rect_bound(overlapValid[cam_i][cam_j], x_eqr, y_eqr);
					}
				}
			}
		}
	}

	for (vx_uint32 cam = 0; cam < numCamera; cam++) {
		for (vx_uint32 i = 0; i <= cam; i++) {
			overlapValid[cam][i].end_x += 1;
			overlapValid[cam][i].end_y += 1;
			if (overlapPadded) {
				overlapPadded[cam][i].end_x += 1;
				overlapPadded[cam][i].end_y += 1;
			}
		}
	}
	// count number of overlaps (use overlap if specified)
	vx_uint32 overlapCount = 0;
	if (paddedCamOverlapInfo) {
		overlapCount = paddedPixelOverlapCountMax;
	}
	else {
		overlapCount = validPixelOverlapCountMax;
	}
	return overlapCount;
}

//////////////////////////////////////////////////////////////////////
// Generate valid mask image
vx_status GenerateValidMaskImage(
	vx_uint32 numCamera,                  // [in] number of cameras
	vx_uint32 eqrWidth,                   // [in] output equirectangular image width
	vx_uint32 eqrHeight,                  // [in] output equirectangular image height
	const vx_uint32 * validPixelCamMap,   // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	vx_uint32  maskStride,                // [in] stride (in bytes) of mask image
	vx_uint8 * maskBuf                    // [out] valid mask image buffer: size: [eqrWidth * eqrHeight * numCamera]
	)
{
	vx_uint32 maskPosition = 0;
	for (vx_uint32 camId = 0; camId < numCamera; camId++) {
		vx_uint32 camMaskBit = 1 << camId;
		for (vx_uint32 y = 0, pixelPosition = 0; y < eqrHeight; y++) {
			for (vx_uint32 x = 0; x < eqrWidth; x++, pixelPosition++) {
				maskBuf[maskPosition + x] = (validPixelCamMap[pixelPosition] & camMaskBit) ? 255 : 0;
			}
			maskPosition += maskStride;
		}
	}
	return VX_SUCCESS;
}
