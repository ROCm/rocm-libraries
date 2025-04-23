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

#ifndef __KERNELS_H__
#define __KERNELS_H__

//////////////////////////////////////////////////////////////////////
// SHARED_PUBLIC - shared sybols for export
// STITCH_API_ENTRY - export API symbols
#if _WIN32
#define SHARED_PUBLIC extern "C" __declspec(dllexport)
#else
#define SHARED_PUBLIC extern "C" __attribute__ ((visibility ("default")))
#endif
#define LIVE_STITCH_API_ENTRY SHARED_PUBLIC

//////////////////////////////////////////////////////////////////////
// common header files
#include "live_stitch_api.h"
#include <VX/vx.h>
#include <vx_ext_amd.h>
#include <VX/vx_compatibility.h>
#if !__APPLE__
#include <omp.h>
#endif
#include <vector>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>

#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#if !_WIN32
#include <strings.h>
#define _strnicmp strncasecmp
#define _stricmp  strcasecmp
#endif

#if _WIN32
#pragma intrinsic(_BitScanReverse)
static inline vx_uint32 GetOneBitPosition(vx_uint32 a)
{
	unsigned long index;
	_BitScanReverse(&index, (unsigned long)a);
	return (vx_uint32)index;
}
static inline vx_uint32 GetOneBitCount(vx_uint32 a)
{
	return __popcnt(a);
}
#else
static inline vx_uint32 GetOneBitPosition(vx_uint32 a)
{
	return ((31 - __builtin_clz(a)) < 0) ? 32 : (31 - __builtin_clz(a));
}
static inline vx_uint32 GetOneBitCount(vx_uint32 a)
{
	return __builtin_popcount(a);
}
#endif

//////////////////////////////////////////////////////////////////////
//! \brief The AMD extension library for stitching
#define	AMDOVX_LIBRARY_STITCHING          2

//////////////////////////////////////////////////////////////////////
//! \brief The additional image formats supported by stitching module
enum vx_df_image_amd_stitching_e {
	VX_DF_IMAGE_Y210_AMD = VX_DF_IMAGE('Y', '2', '1', '0'),  // AGO image with YUV 4:2:2 10-bit (Y210)
	VX_DF_IMAGE_Y212_AMD = VX_DF_IMAGE('Y', '2', '1', '2'),  // AGO image with YUV 4:2:2 12-bit (Y212)
	VX_DF_IMAGE_Y216_AMD = VX_DF_IMAGE('Y', '2', '1', '6'),  // AGO image with YUV 4:2:2 16-bit (Y216)
	VX_DF_IMAGE_RGB4_AMD = VX_DF_IMAGE('R', 'G', 'B', '4'),  // AGO image with RGB-48 16bit per channel (RGB4)
};

//////////////////////////////////////////////////////////////////////
//! \brief The list of kernels in the stitching library.
enum vx_kernel_stitching_amd_e {
	//! \brief The Color Convert with optional 2x2 scale down function kernel. Kernel name is "com.amd.loomsl.color_convert".
	AMDOVX_KERNEL_STITCHING_COLOR_CONVERT = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x001,

	//! \brief The Warp function kernel. Kernel name is "com.amd.loomsl.warp".
	AMDOVX_KERNEL_STITCHING_WARP = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x002,

	//! \brief The Merge kernel. Kernel name is "com.amd.loomsl.merge".
	AMDOVX_KERNEL_STITCHING_MERGE = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x003,

	//! \brief The Alpha Blend kernel. Kernel name is "com.amd.loomsl.alpha_blend".
	AMDOVX_KERNEL_STITCHING_ALPHA_BLEND = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x004,

	//! \brief The Exposure Comp Stage#1 kernel. Kernel name is "com.amd.loomsl.expcomp_compute_gainmatrix".
	AMDOVX_KERNEL_STITCHING_EXPCOMP_COMPUTE_GAINMAT = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x005,

	//! \brief The Exposure Comp Stage#2 kernel. Kernel name is "com.amd.loomsl.expcomp_solvegains".
	AMDOVX_KERNEL_STITCHING_EXPCOMP_SOLVE = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x006,

	//! \brief The Exposure Comp Stage#3 kernel. Kernel name is "com.amd.loomsl.expcomp_applygains".
	AMDOVX_KERNEL_STITCHING_EXPCOMP_APPLYGAINS = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x007,

	//! \brief The Multi-band blend kernel. Kernel name is "com.amd.loomsl.multiband_blend".
	AMDOVX_KERNEL_STITCHING_MULTIBAND_BLEND = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x008,

	//! \brief The Half scale Gaussian kernel. Kernel name is "com.amd.loomsl.half_scale_gaussian".
	AMDOVX_KERNEL_STITCHING_HALF_SCALE_GAUSSIAN = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x009,

	//! \brief The Half scale Gaussian kernel. Kernel name is "com.amd.loomsl.upscale_gaussian_subtract".
	AMDOVX_KERNEL_STITCHING_UPSCALE_GAUSSIAN_SUBTRACT = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x00a,

	//! \brief The Half scale Gaussian kernel. Kernel name is "com.amd.loomsl.upscale_gaussian_add".
	AMDOVX_KERNEL_STITCHING_UPSCALE_GAUSSIAN_ADD = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x00b,

	//! \brief The Half scale Gaussian kernel. Kernel name is "com.amd.loomsl.laplacian_reconstruct".
	AMDOVX_KERNEL_STITCHING_LAPLACIAN_RECONSTRUCT = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x00c,

	//! \brief The Seam Finding kernel 0. Kernel name is "com.amd.loomsl.seamfind_scene_detect".
	AMDOVX_KERNEL_STITCHING_SEAMFIND_SCENE_DETECT = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x00d,

	//! \brief The Seam Finding kernel 1. Kernel name is "com.amd.loomsl.seamfind_cost_generate".
	AMDOVX_KERNEL_STITCHING_SEAMFIND_COST_GENERATE = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x00e,

	//! \brief The Seam Finding kernel 2. Kernel name is "com.amd.loomsl.seamfind_cost_accumulate".
	AMDOVX_KERNEL_STITCHING_SEAMFIND_COST_ACCUMULATE = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x00f,

	//! \brief The Seam Finding kernel 3. Kernel name is "com.amd.loomsl.seamfind_path_trace".
	AMDOVX_KERNEL_STITCHING_SEAMFIND_PATH_TRACE = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x010,

	//! \brief The Seam Finding kernel 4. Kernel name is "com.amd.loomsl.seamfind_set_weights".
	AMDOVX_KERNEL_STITCHING_SEAMFIND_SET_WEIGHTS = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x011,

	//! \brief The Seam Finding kernel. Kernel name is "com.amd.loomsl.seamfind_analyze".
	AMDOVX_KERNEL_STITCHING_SEAMFIND_ANALYZE = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x012,

	//! \brief The Exposure Comp Stage#1 kernel. Kernel name is "com.amd.loomsl.expcomp_compute_gainmatrix_rgb".
	AMDOVX_KERNEL_STITCHING_EXPCOMP_COMPUTE_GAINMAT_RGB = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x013,
	
	//! \brief The Seam Finding kernel 4. Kernel name is "com.amd.loomsl.chroma_key_mask_generation".
	AMDOVX_KERNEL_STITCHING_CHROMA_KEY_MASK_GENERATION = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x015,

	//! \brief The Seam Finding kernel. Kernel name is "com.amd.loomsl.chroma_key_merge".
	AMDOVX_KERNEL_STITCHING_CHROMA_KEY_MERGE = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x016,

	//! \brief The Noise Filter at input. Kernel name is "com.amd.loomsl.color_convert".
	AMDOVX_KERNEL_STITCHING_NOISE_FILTER = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x017,

	//! \brief The warp to sphere kernel. Kernel name is "com.amd.loomsl.warp_eqr_to_aze".
	AMDOVX_KERNEL_STITCHING_WARP_EQR_TO_AZE = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x018,

	//! \brief The warp to sphere kernel. Kernel name is "com.amd.loomsl.calc_lens_distortionwarp_map".
	AMDOVX_KERNEL_STITCHING_INIT_CALC_CAMERA_VALID_MAP = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x019,

	//! \brief The warp to sphere kernel. Kernel name is "com.amd.loomsl.compute_default_camIdx".
	AMDOVX_KERNEL_STITCHING_INIT_COMPUTE_DEFAULT_CAMERA_IDX = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x01a,

	//! \brief The warp to sphere kernel. Kernel name is "com.amd.loomsl.extend_padding".
	AMDOVX_KERNEL_STITCHING_INIT_EXTEND_PAD_DILATE = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x01b,

	//! \brief The warp to sphere kernel. Kernel name is "com.amd.loomsl.extend_padding_vert".
	AMDOVX_KERNEL_STITCHING_INIT_EXTEND_PAD_VERT = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x01c,

	// TBD: remove

	//! \brief The Exposure Compensation kernel. Kernel name is "com.amd.loomsl.exposure_compensation_model".
	AMDOVX_KERNEL_STITCHING_EXPOSURE_COMPENSATION_MODEL = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x100,

	//! \brief The Seam Finding kernel. Kernel name is "com.amd.loomsl.seamfind_model".
	AMDOVX_KERNEL_STITCHING_SEAMFIND_MODEL = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_STITCHING) + 0x101,

};

//////////////////////////////////////////////////////////////////////
//! \brief The module entry point for publishing kernel.
SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context);

//////////////////////////////////////////////////////////////////////

/*! \brief [Graph] Creates a Color Convert node.
* \param [in] graph The reference to the graph.
* \param [in] input The input image.
* \param [out] output The output image.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchColorConvertNode(vx_graph graph, vx_image input, vx_image output);

/*! \brief [Graph] Creates a Warp node.
* \param [in] graph The reference to the graph.
* \param [in] input The input computation method type.
* \param [in] input The input scalar number of cameras.
* \param [in] input The input array of StitchValidPixel.
* \param [in] input The input array of StitchWarpRemap
* \param [in] input The input image.
* \param [out] output The output image.
* \param [in] num_camera_columns The number of camera columns (optional)
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchWarpNode(vx_graph graph, vx_enum method, vx_uint32 num_cam,
	vx_array ValidPixelEntry, vx_array WarpRemapEntry, vx_image input, vx_image output, vx_image outputLuma, vx_uint32 num_camera_columns);

/*! \brief [Graph] Creates a Stitch Merge node.
* \param [in] graph The reference to the graph.
* \param [in] input The camera id image.
* \param [in] input The group1 id image.
* \param [in] input The group2 id image.
* \param [in] input The input image.
* \param [in] input The weight image.
* \param [out] output The output image.
* \param [out/optional] output The output U8 image containing the X plane of output RGBX
* \param [in/optional] input The number of camera columns (uint32 scalar)
* \param [in/optional] input The external alpha value (uint32 scalar)
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchMergeNode(vx_graph graph, vx_image camera_id_image, vx_image group1_image, vx_image group2_image, vx_image input, vx_image weight_image, vx_image output);

/*! \brief [Graph] Creates a AlphaBlend node.
* \param [in] graph The reference to the graph.
* \param [in] input_rgb Input RGB image.
* \param [in] input_rgba Input RGBX image with alpha channel.
* \param [out] output_rgb Output RGB image.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchAlphaBlendNode(vx_graph graph, vx_image input_rgb, vx_image input_rgba, vx_image output_rgb);

/*! \brief [Graph] Creates a ExposureCompCalcErrorFn node.
* \param [in] graph      The reference to the graph.
* \param [in] numCameras Scalar (uint32: number of cameras)
* \param [in] input      Input image
* \param [in] exp_data   Input Array of expdata.
* \param [in] mask       Mask image.
* \param [out] out_intensity     Output matrix for sum of overlapping pixels.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchExposureCompCalcErrorFnNode(vx_graph graph, vx_uint32 numCameras,
	vx_image input, vx_array exp_data, vx_image mask, vx_matrix out_intensity);


/*! \brief [Graph] Creates a ExposureCompSolveForGain node.
* \param [in] graph         The reference to the graph.
* \param [in] alpha         Input scalar (float32 alpha value: typically square of standard deviation of normalized gain error)
* \param [in] beta          Input scalar (float32 beta value: typically square of standard deviation of normalized intensity error)
* \param [in] in_intensity  Input matrix for sum of overlapping pixels.
* \param [in] in_count      Input matrix for count of overlapping pixels.
* \param [out] out_gains    Output array for gains.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchExposureCompSolveForGainNode(vx_graph graph, vx_float32 alpha,
	vx_float32 beta, vx_matrix in_intensity, vx_matrix in_count, vx_array out_gains);

/*! \brief [Graph] Creates a ExposureCompApplyGain node.
* \param [in] graph      The reference to the graph.
* \param [in] input      Input image
* \param [in] in_gains   Array of valid regions of rectangles
* \param [in] in_offsets Array of StitchExpCompCalcEntry
* \param [in] num_cam      scalar 
* \param [in] gain_width scalar for gain_array width
* \param [in] gain_height scalar for gain_array height
* \param [out] output    Exposure adjusted image.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchExposureCompApplyGainNode(vx_graph graph, vx_image input, vx_array in_gains, vx_array in_offsets, 
	vx_uint32 num_cam, vx_uint32 gain_width, vx_uint32 gain_height, vx_image output);

/*! \brief [Graph] Creates a ExposureCompCalcErrorFnRGB node.
* \param [in] graph      The reference to the graph.
* \param [in] numCameras Scalar (uint32: number of cameras)
* \param [in] input      Input image
* \param [in] exp_data   Input Array of expdata.
* \param [in] mask       Mask image.
* \param [out] out_intensity     Output matrix for sum of overlapping pixels.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchExposureCompCalcErrorFnRGBNode(vx_graph graph, vx_uint32 numCameras,
	vx_image input, vx_array exp_data, vx_image mask, vx_matrix out_intensity);


/*! \brief [Graph] Creates a stitchBlendMultiBandMerge node.
* \param [in] graph         The reference to the graph.
* \param [in] num_cameras   Input Scalar (uint32: number of cameras)
* \param [in] blend_array_offs Scalar (uint32: start_offset to valid_arr in #of elements)
* \param [in] input         Src image (RGBA)
* \param [in] weight_img    Weight image(source U8 image)
* \param [in] valid_arr     Offsets/valid rect array (offsets will be useful for GPU kernel)
* \param [out] output       Reconstructed from multibands.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchMultiBandMergeNode(vx_graph graph, vx_uint32 num_cameras, vx_uint32 blend_array_offs,
	vx_image input, vx_image weight_img, vx_array valid_arr, vx_image output);

/*! \brief [Graph] Creates a stitchMultiBandHalfScaleGaussian node.
* \param [in] graph         The reference to the graph.
* \param [in] num_cameras   Scalar (uint32: number of cameras)
* \param [in] blend_array_offs  Scalar (uint32: start_offset to valid_arr in #of elements)
* \param [in] valid_arr     Offsets/valid rect array (offsets will be useful for GPU kernel)
* \param [in] input         Src image (RGBA)
* \param [out] output   Output image
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchMultiBandHalfScaleGaussianNode(vx_graph graph, vx_uint32 num_cameras, vx_uint32 blend_array_offs,
	vx_array valid_arr, vx_image input, vx_image output);

/*! \brief [Graph] Creates a stitchMultiBandUpscaleGaussianSubtract node.
* \param [in] graph The reference to the graph.
* \param [in] num_cameras Scalar (uint32: number of cameras)
* \param [in] blend_array_offs Scalar (uint32: start_offset to valid_arr in #of elements)
* \param [in] input1 Src_image1
* \param [in] input2 Src_image2
* \param [in] valid_arr Offsets/valid rect array (offsets will be useful for GPU kernel)
* \param [in/optional] Weight_img input weight image for blending
* \param [out] output Output image.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchMultiBandUpscaleGaussianSubtractNode(vx_graph graph, vx_uint32 num_cameras, vx_uint32 blend_array_offs,
	vx_image input1, vx_image input2, vx_array valid_arr, vx_image weight_img, vx_image output);

/*! \brief [Graph] Creates a stitchMultiBandUpscaleGaussianAdd node.
* \param [in] graph The reference to the graph.
* \param [in] num_cameras scalar (uint32: number of cameras)
* \param [in] blend_array_offs scalar (uint32: start_offset to valid_arr in #of elements)
* \param [in] input1 src image1
* \param [in] input2 src image2
* \param [in] valid_arr Offsets/valid rect array (offsets will be useful for GPU kernel)
* \param [out] output (img)
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchMultiBandUpscaleGaussianAddNode(vx_graph graph, vx_uint32 num_cameras, vx_uint32 blend_array_offs,
	vx_image input1, vx_image input2, vx_array valid_arr, vx_image output);

/*! \brief [Graph] Creates a stitchMultiBandLaplacianReconstruct node.
* \param [in] graph The reference to the graph.
* \param [in] num_cameras The number of cameras
* \param [in] blend_array_offs The start_offsets to valid_arr in #of elements
* \param [in] input1 The src image1
* \param [in] input2 The src image2
* \param [in] valid_arr The offsets/valid rect array (offsets will be useful for GPU kernel)
* \param [out] output image.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchMultiBandLaplacianReconstructNode(vx_graph graph, vx_uint32 num_cameras, vx_uint32 blend_array_offs,
	vx_image input1, vx_image input2, vx_array valid_arr, vx_image output);

/*! \brief [Graph] Creates a SeamFind Accumulate node K1 - GPU/CPU.
* \param [in] graph The reference to the graph.
* \param [in] current_frame The input scalar current frame.
* \param [in] scene_threshold The input scalar threshold.
* \param [in] input_image The input U8 Image.
* \param [in] seam_info The input array of seam info.
* \param [out] seam_pref The array of seam preference.
* \param [out] output The array of seam scene change.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindSceneDetectNode(vx_graph graph, vx_scalar current_frame, vx_scalar scene_threshold,
	vx_image input_image, vx_array seam_info, vx_array seam_pref, vx_array seam_scene_change);

/*! \brief [Graph] Creates a SeamFind Cost Generate node - K2 - GPU.
* \param [in] graph The reference to the graph.
* \param [in] executeFlag The input scalar to bypass the execution of kernel.
* \param [in] input_weight_image The input U8 weight image from Warp.
* \param [out] magnitude_image The output magnitude image.
* \param [out] phase_image The output phase image.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindCostGenerateNode(vx_graph graph, vx_scalar executeFlag,
	vx_image input_weight_image, vx_image magnitude_image, vx_image phase_image);

/*! \brief [Graph] Creates a SeamFind Cost Accumulate node - K3 - GPU.
* \param [in] graph The reference to the graph.
* \param [in] current_frame The Current Frame.
* \param [in] output_width  The output image width.
* \param [in] output_height The output image height.
* \param [in] magnitude_img The input Magnitude Image from K1.
* \param [in] phase_img     The input Phase Image from K1.
* \param [in] mask_img      The input Mask Image fromInitialize Stitch Config.
* \param [in] valid_seam    The input array of valid_seam pixels.
* \param [in] pref_seam     The input array of seam preference.
* \param [in] info_seam     The input seam info array.
* \param [out] output       The output seam_accum array.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindCostAccumulateNode(vx_graph graph, vx_scalar current_frame,
	vx_uint32 output_width, vx_uint32 output_height, vx_image magnitude_img, vx_image phase_img,
	vx_image mask_img, vx_array valid_seam, vx_array pref_seam, vx_array info_seam, vx_array accum_seam);

/*! \brief [Graph] Creates a SeamFind Accumulate node K4 - GPU/CPU.
* \param [in] graph The reference to the graph.
* \param [in] current_frame The Current Frame.
* \param [in] weight_image  The input Weight Image
* \param [in] seam_info     The input seam info array.
* \param [in] seam_accum    The input seam_accum array.
* \param [in] seam_pref     The input array of seam preference
* \param [out] output       The Path Array.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindPathTraceNode(vx_graph graph, vx_scalar current_frame, vx_image weight_image, vx_array seam_info,
	vx_array seam_accum, vx_array seam_pref, vx_array paths);

/*! \brief [Graph] Creates a SeamFind Accumulate node K5 - GPU.
* \param [in] graph         The reference to the graph.
* \param [in] current_frame The input Current Frame.
* \param [in] NumCam        The input scalar number of cameras.
* \param [in] output_width  The input output width.
* \param [in] output_height The input output height.
* \param [in] seam_weight   The input array of seam weights.
* \param [in] seam_path     The input array of seam path .
* \param [in] seam_pref     The input array of seam preference.
* \param [out] output       The weight image.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindSetWeightsNode(vx_graph graph, vx_scalar current_frame, vx_uint32 NumCam,
	vx_uint32 output_width, vx_uint32 output_height, vx_array seam_weight, vx_array seam_path,
	vx_array seam_pref, vx_image weight_image, vx_uint32 flags);

/*!\brief Function to create SeamFind Seam Analyze Node - CPU
* \param [in] graph         The reference to the graph.
* \param [in] current_frame The input Current Frame.
* \param [in] seam_pref     The input array of seam preference.
* \param [out] output       The output flag.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindAnalyzeNode(vx_graph graph, vx_scalar current_frame, vx_array seam_pref, vx_scalar flag);

//////////////////////////////////////////////////////////////////////
// Chroma Key Kernels
//////////////////////////////////////////////////////////////////////

/*! \brief [Graph] Creates a stitch Chroma Key Mask Generator Node- GPU/CPU.
* \param [in] graph         The reference to the graph.
* \param [in] ChromaKey		The input Chroma Key.
* \param [in] ChromaKeyTol  The input Chroma Key tolerance.
* \param [in] input_rgb_img The input stitched output image.
* \param [out] output       The output mask image.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchChromaKeyMaskGeneratorNode(vx_graph graph, vx_uint32 ChromaKey, vx_uint32 ChromaKeyTol, vx_image input_rgb_img, vx_image output_mask_img);

/*! \brief [Graph] Creates a stitch Chroma Key Merge Node- GPU/CPU.
* \param [in] graph				The reference to the graph.
* \param [in] input_rgb_img		The input stitched output image.
* \param [in] input_chroma_img	The input Chroma image.
* \param [in] input_mask_img	The input mask image.
* \param [out] output			The output chorma merged image.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchChromaKeyMergeNode(vx_graph graph, vx_image input_rgb_img, vx_image input_chroma_img, vx_image input_mask_img, vx_image output_merged_img);


/*! \brief [Graph] Creates a stitch Noise Filter Node- GPU.
* \param [in] graph				The reference to the graph.
* \param [in] lambda			The input scalar lambda.
* \param [in] input_rgb_img		The input camera image.
* \param [in] input_rgb_img		The input delayed camera image.
* \param [out] output			The output denoised image.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchNoiseFilterNode(vx_graph graph, vx_scalar lambda, vx_image input_rgb_img_1, vx_image input_rgb_img_2, vx_image denoised_image);

/*! \brief [Graph] Creates a stitch Equirectangular to Azimuthal Equidistant projection Node- GPU.
* \param [in] graph					The reference to the graph.
* \param [in] input_rgb				The input equirectangular image in RGB format.
* \param [in] rad_lat_map			The input radius to latitude map array.
* \param [out] output_warped_rgb	The output warped image in Azimuthal equidistant projection
* \param [in/optional] a			The parameter a to control the scale of the projection
* \param [in/optional] b			The parameter a to control the rotation of the projection
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL stitchWarpEqrToAzE(vx_graph graph, vx_image input_rgb, vx_array rad_lat_map, vx_image output_warped_rgb, vx_scalar a, vx_scalar b);

//////////////////////////////////////////////////////////////////////
//! \brief Nodes for Stitch Initialize
VX_API_ENTRY vx_node VX_API_CALL stitchInitCalcCamWarpMaps(vx_graph graph, void *scalar_params, vx_array cam_params, vx_image valid_map, vx_image padding_map, vx_image src_coord_map, vx_array z_buffer_map);
VX_API_ENTRY vx_node VX_API_CALL stitchInitCalcDefCamIdxNode(vx_graph graph, vx_uint32 numCam, vx_uint32 out_width, vx_uint32 out_height, vx_array z_buffer_map, vx_image cam_idx_image);
VX_API_ENTRY vx_node VX_API_CALL stitchInitExtendPadDilateNode(vx_graph graph, vx_uint32 paddingPixels, vx_image valid_map, vx_image padding_map);
//VX_API_ENTRY vx_node VX_API_CALL stitchInitPadVertNode(vx_graph graph, vx_uint32 paddingPixels, vx_uint32 out_W, vx_uint32 out_H, vx_image valid_map, vx_image padding_map);

//////////////////////////////////////////////////////////////////////
//! \brief The utility functions
vx_reference avxGetNodeParamRef(vx_node node, vx_uint32 index);
vx_node stitchCreateNode(vx_graph graph, vx_enum kernelEnum, vx_reference params[], vx_uint32 num);
vx_node stitchCreateNode(vx_graph graph, const char * kernelName, vx_reference params[], vx_uint32 num);
bool StitchGetEnvironmentVariable(const char * name, char * value, size_t valueSize);

//////////////////////////////////////////////////////////////////////
//! \brief The macro for error checking from OpenVX status.
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS){ printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }}
//! \brief The macro for error checking from OpenVX object.
#define ERROR_CHECK_OBJECT(obj)  { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS){ vxAddLogEntry((vx_reference)(obj), status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }}
//! \brief The macro for getting the dimensions.
#define dimof(x)                    (sizeof(x)/sizeof(x[0]))

#endif //__KERNELS_H__
