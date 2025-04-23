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

#include "kernels.h"
#include "chroma_key.h"
#include "color_convert.h"
#include "warp.h"
#include "seam_find.h"
#include "exposure_compensation.h"
#include "multiband_blender.h"
#include "pyramid_scale.h"
#include "merge.h"
#include "alpha_blend.h"
#include "noise_filter.h"
#include "warp_eqr_to_aze.h"
#include "lens_distortion_remap.h"

#if _WIN32
#include <Windows.h>
#endif

////////////////////////////////////////////////////////////////////////////
//! \brief The module entry point for publishing kernel.
SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
	// register image formats
	AgoImageFormatDescription desc = { 3, 1, 32, VX_COLOR_SPACE_DEFAULT, VX_CHANNEL_RANGE_FULL };
	vxSetContextImageFormatDescription(context, VX_DF_IMAGE_Y210_AMD, &desc);
	vxSetContextImageFormatDescription(context, VX_DF_IMAGE_Y212_AMD, &desc);
	vxSetContextImageFormatDescription(context, VX_DF_IMAGE_Y216_AMD, &desc);
	desc = { 3, 1, 48, VX_COLOR_SPACE_DEFAULT, VX_CHANNEL_RANGE_FULL };
	vxSetContextImageFormatDescription(context, VX_DF_IMAGE_RGB4_AMD, &desc);
	// register kernels
	ERROR_CHECK_STATUS(color_convert_publish(context));
	ERROR_CHECK_STATUS(warp_publish(context));
	ERROR_CHECK_STATUS(exposure_compensation_publish(context));
	ERROR_CHECK_STATUS(exposure_comp_calcErrorFn_publish(context));
	ERROR_CHECK_STATUS(exposure_comp_solvegains_publish(context));
	ERROR_CHECK_STATUS(exposure_comp_applygains_publish(context));
	ERROR_CHECK_STATUS(merge_publish(context));
	ERROR_CHECK_STATUS(alpha_blend_publish(context));
	ERROR_CHECK_STATUS(multiband_blend_publish(context));
	ERROR_CHECK_STATUS(half_scale_gaussian_publish(context));
	ERROR_CHECK_STATUS(upscale_gaussian_subtract_publish(context));
	ERROR_CHECK_STATUS(upscale_gaussian_add_publish(context));
	ERROR_CHECK_STATUS(laplacian_reconstruct_publish(context));
	ERROR_CHECK_STATUS(seamfind_model_publish(context));
	ERROR_CHECK_STATUS(seamfind_scene_detect_publish(context));
	ERROR_CHECK_STATUS(seamfind_cost_generate_publish(context));
	ERROR_CHECK_STATUS(seamfind_cost_accumulate_publish(context));
	ERROR_CHECK_STATUS(seamfind_path_trace_publish(context));
	ERROR_CHECK_STATUS(seamfind_set_weights_publish(context));
	ERROR_CHECK_STATUS(seamfind_analyze_publish(context));
	ERROR_CHECK_STATUS(exposure_comp_calcRGBErrorFn_publish(context));
	ERROR_CHECK_STATUS(chroma_key_mask_generation_publish(context));
	ERROR_CHECK_STATUS(chroma_key_merge_publish(context));
	ERROR_CHECK_STATUS(noise_filter_publish(context));
	ERROR_CHECK_STATUS(warp_eqr_to_aze_publish(context));
	ERROR_CHECK_STATUS(calc_lens_distortionwarp_map_publish(context));
	ERROR_CHECK_STATUS(compute_default_camIdx_publish(context));
	ERROR_CHECK_STATUS(extend_padding_dilate_publish(context));
	//ERROR_CHECK_STATUS(extend_padding_vert_publish(context));

	return VX_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////
//! \brief The module entry point for unpublishing kernel.
SHARED_PUBLIC vx_status VX_API_CALL vxUnpublishKernels(vx_context context)
{
	return VX_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////
// local utility functions

//! \brief The utility function to get reference of a node parameter at specified index
vx_reference avxGetNodeParamRef(vx_node node, vx_uint32 index)
{
	vx_reference ref = nullptr;
	vx_parameter param = vxGetParameterByIndex(node, index);
	if (vxGetStatus((vx_reference)param) == VX_SUCCESS) {
		vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &ref, sizeof(ref));
		vxReleaseParameter(&param);
	}
	return ref;
}


//! \brief Utility function to create nodes
vx_node stitchCreateNode(vx_graph graph, vx_enum kernelEnum, vx_reference params[], vx_uint32 num)
{
	vx_status status = VX_SUCCESS;
	vx_node node = 0;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByEnum(context, kernelEnum);
	if (kernel) {
		node = vxCreateGenericNode(graph, kernel);
		if (node) {
			vx_uint32 p = 0;
			for (p = 0; p < num; p++) {
				if (params[p]) {
					status = vxSetParameterByIndex(node, p, params[p]);
					if (status != VX_SUCCESS) {
						char kernelName[VX_MAX_KERNEL_NAME];
						vxQueryKernel(kernel, VX_KERNEL_ATTRIBUTE_NAME, kernelName, VX_MAX_KERNEL_NAME);
						vxAddLogEntry((vx_reference)graph, status, "stitchCreateNode: vxSetParameterByIndex(%s, %d, 0x%p) => %d\n", kernelName, p, params[p], status);
						vxReleaseNode(&node);
						node = 0;
						break;
					}
				}
			}
		}
		else {
			vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "Failed to create node with kernel enum %d\n", kernelEnum);
			status = VX_ERROR_NO_MEMORY;
		}
		vxReleaseKernel(&kernel);
	}
	else {
		vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "failed to retrieve kernel enum %d\n", kernelEnum);
		status = VX_ERROR_NOT_SUPPORTED;
	}
	return node;
}

vx_node stitchCreateNode(vx_graph graph, const char * kernelName, vx_reference params[], vx_uint32 num)
{
	vx_status status = VX_SUCCESS;
	vx_node node = 0;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByName(context, kernelName);
	if (kernel) {
		node = vxCreateGenericNode(graph, kernel);
		if (node) {
			vx_uint32 p = 0;
			for (p = 0; p < num; p++) {
				if (params[p]) {
					status = vxSetParameterByIndex(node, p, params[p]);
					if (status != VX_SUCCESS) {
						vxAddLogEntry((vx_reference)graph, status, "stitchCreateNode: vxSetParameterByIndex(%s, %d, 0x%p) => %d\n", kernelName, p, params[p], status);
						vxReleaseNode(&node);
						node = 0;
						break;
					}
				}
			}
		}
		else {
			vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "Failed to create node with kernel %s\n", kernelName);
			status = VX_ERROR_NO_MEMORY;
		}
		vxReleaseKernel(&kernel);
	}
	else {
		vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "failed to retrieve kernel %s\n", kernelName);
		status = VX_ERROR_NOT_SUPPORTED;
	}
	return node;
}

bool StitchGetEnvironmentVariable(const char * name, char * value, size_t valueSize)
{
#if _WIN32
	DWORD len = GetEnvironmentVariableA(name, value, (DWORD)valueSize);
	value[valueSize - 1] = 0;
	return (len > 0) ? true : false;
#else
	const char * v = getenv(name);
	if (v) {
		strncpy(value, v, valueSize);
		value[valueSize - 1] = 0;
	}
	return v ? true : false;
#endif
}

/***********************************************************************************************************************************
OVX Stich Nodes
************************************************************************************************************************************/
/**
* \brief Function to create Color Convert node
*/
VX_API_ENTRY vx_node VX_API_CALL stitchColorConvertNode(vx_graph graph, vx_image input, vx_image output)
{

	vx_reference params[] = {
		(vx_reference)input,
		(vx_reference)output
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_COLOR_CONVERT,
		params,
		dimof(params));

	return node;
}

/**
* \brief Function to create Stitch Warp node
*/
VX_API_ENTRY vx_node VX_API_CALL stitchWarpNode(vx_graph graph, vx_enum method, vx_uint32 num_cam, vx_array ValidPixelEntry, vx_array WarpRemapEntry, vx_image input, vx_image output, vx_image outputLuma, vx_uint32 num_camera_columns)
{
	vx_scalar METHOD = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &method);
	vx_scalar NUM_CAM = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &num_cam);
	vx_scalar s_num_camera_columns = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &num_camera_columns);

	vx_reference params[] = {
		(vx_reference)METHOD,
		(vx_reference)NUM_CAM,
		(vx_reference)ValidPixelEntry,
		(vx_reference)WarpRemapEntry,
		(vx_reference)input,
		(vx_reference)output,
		(vx_reference)outputLuma,
		(vx_reference)s_num_camera_columns,
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_WARP,
		params,
		dimof(params));
	vxReleaseScalar(&METHOD);
	vxReleaseScalar(&NUM_CAM);
	vxReleaseScalar(&s_num_camera_columns);
	return node;
}

/**
* \brief Function to create Stitch Merge node
*/
VX_API_ENTRY vx_node VX_API_CALL stitchMergeNode(vx_graph graph, vx_image camera_id_image, vx_image group1_image, vx_image group2_image, vx_image input, vx_image weight_image, vx_image output)
{
	vx_reference params[] = {
		(vx_reference)camera_id_image,
		(vx_reference)group1_image,
		(vx_reference)group2_image,
		(vx_reference)input,
		(vx_reference)weight_image,
		(vx_reference)output
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_MERGE,
		params,
		dimof(params));

	return node;
}

/**
* \brief Function to create Stitch AlphaBlendnode
*/
VX_API_ENTRY vx_node VX_API_CALL stitchAlphaBlendNode(vx_graph graph, vx_image input_rgb, vx_image input_rgba, vx_image output_rgb)
{
	vx_reference params[] = {
		(vx_reference)input_rgb,
		(vx_reference)input_rgba,
		(vx_reference)output_rgb
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_ALPHA_BLEND,
		params,
		dimof(params));

	return node;
}

/**
* \brief Function to create Calculate Error Function node
*/
VX_API_ENTRY vx_node VX_API_CALL stitchExposureCompCalcErrorFnNode(vx_graph graph, vx_uint32 numCameras, vx_image input, vx_array exp_data, vx_image mask, vx_matrix out_intensity)
{
	vx_scalar Num_Camera = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &numCameras);

	vx_reference params[] = {
		(vx_reference)Num_Camera,
		(vx_reference)input,
		(vx_reference)exp_data,
		(vx_reference)mask,
		(vx_reference)out_intensity,
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_EXPCOMP_COMPUTE_GAINMAT,
		params,
		dimof(params));

	vxReleaseScalar(&Num_Camera);
	return node;
}

/**
* \brief Function to create Calculate RGB Error Function node
*/
VX_API_ENTRY vx_node VX_API_CALL stitchExposureCompCalcErrorFnRGBNode(vx_graph graph, vx_uint32 numCameras, vx_image input, vx_array exp_data, vx_image mask, vx_matrix out_intensity)
{
	vx_scalar Num_Camera = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &numCameras);

	vx_reference params[] = {
		(vx_reference)Num_Camera,
		(vx_reference)input,
		(vx_reference)exp_data,
		(vx_reference)mask,
		(vx_reference)out_intensity,
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_EXPCOMP_COMPUTE_GAINMAT_RGB,
		params,
		dimof(params));

	vxReleaseScalar(&Num_Camera);
	return node;
}

/**
* \brief Function to create Calculate Gains node
*/
VX_API_ENTRY vx_node VX_API_CALL stitchExposureCompSolveForGainNode(vx_graph graph, vx_float32 alpha, vx_float32 beta, vx_matrix in_intensity, vx_matrix in_count, vx_array out_gains)
{
	vx_scalar Alpha = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &alpha);
	vx_scalar Beta = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &beta);

	vx_reference params[] = {
		(vx_reference)Alpha,
		(vx_reference)Beta,
		(vx_reference)in_intensity,
		(vx_reference)in_count,
		(vx_reference)out_gains,
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_EXPCOMP_SOLVE,
		params,
		dimof(params));

	vxReleaseScalar(&Alpha);
	vxReleaseScalar(&Beta);
	return node;
}

/**
* \brief Function to create Apply Gains node
*/
VX_API_ENTRY vx_node VX_API_CALL stitchExposureCompApplyGainNode(vx_graph graph, vx_image input, vx_array in_gains, vx_array in_offsets, vx_uint32 num_cam, vx_uint32 gain_width, vx_uint32 gain_height, vx_image output)
{
	vx_scalar Num_Camera = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &num_cam);
	vx_scalar bg_width = nullptr; vx_scalar bg_height = nullptr;
	if ((gain_width > 1) || (gain_height > 1)) {
		bg_width = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &gain_width);
		bg_height = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &gain_height);
	}

	vx_reference params[] = {
		(vx_reference)input,
		(vx_reference)in_gains,
		(vx_reference)in_offsets,
		(vx_reference)Num_Camera,
		(vx_reference)bg_width,
		(vx_reference)bg_height,
		(vx_reference)output,
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_EXPCOMP_APPLYGAINS,
		params,
		dimof(params));

	vxReleaseScalar(&Num_Camera);
	if (bg_width) vxReleaseScalar(&bg_width);
	if (bg_height) vxReleaseScalar(&bg_height);
	return node;
}

//*\brief Function to create SeamFind Scene Change Detect Node - CPU/GPU
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindSceneDetectNode(vx_graph graph, vx_scalar current_frame, vx_scalar scene_threshold,
	vx_image input_image, vx_array seam_info, vx_array seam_pref, vx_array seam_scene_change)
{
	vx_reference params[] = {
		(vx_reference)current_frame,
		(vx_reference)scene_threshold,
		(vx_reference)input_image,
		(vx_reference)seam_info,
		(vx_reference)seam_pref,
		(vx_reference)seam_scene_change
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_SEAMFIND_SCENE_DETECT,
		params,
		dimof(params));

	return node;
}

//*\brief Function to create SeamFind Cost Generate node - GPU
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindCostGenerateNode(vx_graph graph, vx_scalar executeFlag, vx_image input_weight_image, vx_image magnitude_image, vx_image phase_image)
{
	vx_reference params[] = {
		(vx_reference)executeFlag,
		(vx_reference)input_weight_image,
		(vx_reference)magnitude_image,
		(vx_reference)phase_image,
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_SEAMFIND_COST_GENERATE,
		params,
		dimof(params));

	return node;
}

//*\brief Function to create SeamFind Cost Accumulate Node - GPU 
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindCostAccumulateNode(vx_graph graph, vx_scalar current_frame, vx_uint32 output_width, vx_uint32 output_height,
	vx_image magnitude_img, vx_image phase_img, vx_image mask_img, vx_array valid_seam, vx_array pref_seam, vx_array info_seam, vx_array accum_seam)
{
	vx_scalar OUTPUT_WIDTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &output_width);
	vx_scalar OUTPUT_HEIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &output_height);

	vx_reference params[] = {
		(vx_reference)current_frame,
		(vx_reference)OUTPUT_WIDTH,
		(vx_reference)OUTPUT_HEIGHT,
		(vx_reference)magnitude_img,
		(vx_reference)phase_img,
		(vx_reference)mask_img,
		(vx_reference)valid_seam,
		(vx_reference)pref_seam,
		(vx_reference)info_seam,
		(vx_reference)accum_seam
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_SEAMFIND_COST_ACCUMULATE,
		params,
		dimof(params));

	vxReleaseScalar(&OUTPUT_WIDTH);
	vxReleaseScalar(&OUTPUT_HEIGHT);
	return node;
}

//*\brief Function to create SeamFind Path Trace node - GPU/CPU
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindPathTraceNode(vx_graph graph, vx_scalar current_frame, vx_image weight_image, vx_array seam_info,
	vx_array seam_accum, vx_array seam_pref, vx_array paths)
{
	vx_reference params[] = {
		(vx_reference)current_frame,
		(vx_reference)weight_image,
		(vx_reference)seam_info,
		(vx_reference)seam_accum,
		(vx_reference)seam_pref,
		(vx_reference)paths
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_SEAMFIND_PATH_TRACE,
		params,
		dimof(params));

	return node;
}

//*\brief Function to create SeamFind Set Weights node - GPU
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindSetWeightsNode(vx_graph graph, vx_scalar current_frame, vx_uint32 NumCam, vx_uint32 output_width, vx_uint32 output_height, vx_array seam_weight, vx_array seam_path,
	vx_array seam_pref, vx_image weight_image, vx_uint32 flags)
{
	vx_scalar NUM_CAM = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &NumCam);
	vx_scalar OUTPUT_WIDTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &output_width);
	vx_scalar OUTPUT_HEIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &output_height);
	vx_scalar FLAGS = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &flags);

	vx_reference params[] = {
		(vx_reference)current_frame,
		(vx_reference)NUM_CAM,
		(vx_reference)OUTPUT_WIDTH,
		(vx_reference)OUTPUT_HEIGHT,
		(vx_reference)seam_weight,
		(vx_reference)seam_path,
		(vx_reference)seam_pref,
		(vx_reference)weight_image,
		(vx_reference)FLAGS
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_SEAMFIND_SET_WEIGHTS,
		params,
		dimof(params));

	vxReleaseScalar(&NUM_CAM);
	vxReleaseScalar(&OUTPUT_WIDTH);
	vxReleaseScalar(&OUTPUT_HEIGHT);
	vxReleaseScalar(&FLAGS);
	return node;
}

//*\brief Function to create SeamFind Seam Analyze Node - CPU
VX_API_ENTRY vx_node VX_API_CALL stitchSeamFindAnalyzeNode(vx_graph graph, vx_scalar current_frame, vx_array seam_pref, vx_scalar flag)
{
	vx_reference params[] = {
		(vx_reference)current_frame,
		(vx_reference)seam_pref,
		(vx_reference)flag
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_SEAMFIND_ANALYZE,
		params,
		dimof(params));

	return node;
}

/***********************************************************************************************************************************
Stitch Multiband blending nodes.
************************************************************************************************************************************/
VX_API_ENTRY vx_node VX_API_CALL stitchMultiBandMergeNode(vx_graph graph, vx_uint32 num_cameras, vx_uint32 blend_array_offs,
	vx_image input, vx_image weight_img, vx_array valid_arr, vx_image output)
{
	vx_scalar numCam = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &num_cameras);
	vx_scalar array_offs = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &blend_array_offs);

	vx_reference params[] = {
		(vx_reference)numCam,
		(vx_reference)array_offs,
		(vx_reference)input,
		(vx_reference)weight_img,
		(vx_reference)valid_arr,
		(vx_reference)output
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_MULTIBAND_BLEND,
		params,
		dimof(params));

	vxReleaseScalar(&numCam);
	vxReleaseScalar(&array_offs);
	return node;
}

/***********************************************************************************************************************************
Stitch Multiband blending nodes.
************************************************************************************************************************************/
VX_API_ENTRY vx_node VX_API_CALL stitchMultiBandHalfScaleGaussianNode(vx_graph graph, vx_uint32 num_cameras, vx_uint32 blend_array_offs,
							vx_array valid_arr, vx_image input, vx_image output)
{
	vx_scalar numCam = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &num_cameras);
	vx_scalar array_offs = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &blend_array_offs);

	vx_reference params[] = {
		(vx_reference)numCam,
		(vx_reference)array_offs,
		(vx_reference)valid_arr,
		(vx_reference)input,
		(vx_reference)output
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_HALF_SCALE_GAUSSIAN,
		params,
		dimof(params));

	vxReleaseScalar(&numCam);
	vxReleaseScalar(&array_offs);
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL stitchMultiBandUpscaleGaussianSubtractNode(vx_graph graph, vx_uint32 num_cameras, vx_uint32 blend_array_offs,
	vx_image input1, vx_image input2, vx_array valid_arr, vx_image weight_img, vx_image output)
{
	vx_scalar numCam = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &num_cameras);
	vx_scalar array_offs = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &blend_array_offs);

	vx_reference params[] = {
		(vx_reference)numCam,
		(vx_reference)array_offs,
		(vx_reference)input1,
		(vx_reference)input2,
		(vx_reference)valid_arr,
		(vx_reference)weight_img,
		(vx_reference)output
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_UPSCALE_GAUSSIAN_SUBTRACT,
		params,
		dimof(params));

	vxReleaseScalar(&numCam);
	vxReleaseScalar(&array_offs);
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL stitchMultiBandUpscaleGaussianAddNode(vx_graph graph, vx_uint32 num_cameras, vx_uint32 blend_array_offs,
	vx_image input1, vx_image input2, vx_array valid_arr, vx_image output)
{
	vx_scalar numCam = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &num_cameras);
	vx_scalar array_offs = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &blend_array_offs);

	vx_reference params[] = {
		(vx_reference)numCam,
		(vx_reference)array_offs,
		(vx_reference)input1,
		(vx_reference)input2,
		(vx_reference)valid_arr,
		(vx_reference)output
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_UPSCALE_GAUSSIAN_ADD,
		params,
		dimof(params));

	vxReleaseScalar(&numCam);
	vxReleaseScalar(&array_offs);
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL stitchMultiBandLaplacianReconstructNode(vx_graph graph, vx_uint32 num_cameras, vx_uint32 blend_array_offs,
	vx_image input1, vx_image input2, vx_array valid_arr, vx_image output)
{
	vx_scalar numCam = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &num_cameras);
	vx_scalar array_offs = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &blend_array_offs);

	vx_reference params[] = {
		(vx_reference)numCam,
		(vx_reference)array_offs,
		(vx_reference)input1,
		(vx_reference)input2,
		(vx_reference)valid_arr,
		(vx_reference)output
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_LAPLACIAN_RECONSTRUCT,
		params,
		dimof(params));

	vxReleaseScalar(&numCam);
	vxReleaseScalar(&array_offs);
	return node;

}

/***********************************************************************************************************************************
													Stitch CHROMA KEY
************************************************************************************************************************************/
VX_API_ENTRY vx_node VX_API_CALL stitchChromaKeyMaskGeneratorNode(vx_graph graph, vx_uint32 ChromaKey, vx_uint32 Tolerance, vx_image input_rgb_img, vx_image output_mask_img)
{
	vx_scalar CHROMA_KEY = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &ChromaKey);
	vx_scalar TOLERANCE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &Tolerance);

	vx_reference params[] = {
		(vx_reference)CHROMA_KEY,
		(vx_reference)TOLERANCE,
		(vx_reference)input_rgb_img,
		(vx_reference)output_mask_img
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_CHROMA_KEY_MASK_GENERATION,
		params,
		dimof(params));

	vxReleaseScalar(&CHROMA_KEY);
	vxReleaseScalar(&TOLERANCE);
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL stitchChromaKeyMergeNode(vx_graph graph, vx_image input_rgb_img, vx_image input_chroma_img, vx_image input_mask_img, vx_image output_merged_img)
{
	vx_reference params[] = {
		(vx_reference)input_rgb_img,
		(vx_reference)input_chroma_img,
		(vx_reference)input_mask_img,
		(vx_reference)output_merged_img
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_CHROMA_KEY_MERGE,
		params,
		dimof(params));

	return node;
}

/***********************************************************************************************************************************
Stitch Noise Filter
************************************************************************************************************************************/
VX_API_ENTRY vx_node VX_API_CALL stitchNoiseFilterNode(vx_graph graph, vx_scalar lambda, vx_image input_rgb_img_1, vx_image input_rgb_img_2, vx_image denoised_image)
{
	vx_reference params[] = {
		(vx_reference)lambda,
		(vx_reference)input_rgb_img_1,
		(vx_reference)input_rgb_img_2,
		(vx_reference)denoised_image
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_NOISE_FILTER,
		params,
		dimof(params));

	return node;

}

/***********************************************************************************************************************************
Stitch warp to sphere kernel
************************************************************************************************************************************/
VX_API_ENTRY vx_node VX_API_CALL stitchWarpEqrToAzE(vx_graph graph, vx_image input_rgb, vx_array rad_lat_map, vx_image output_warped_rgb, vx_scalar a, vx_scalar b)
{
	vx_reference params[] = {
		(vx_reference)input_rgb,
		(vx_reference)rad_lat_map,
		(vx_reference)output_warped_rgb,
		(vx_reference)a,
		(vx_reference)b
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_WARP_EQR_TO_AZE,
		params,
		dimof(params));

	return node;

}

/***********************************************************************************************************************************
stitchInitCalcCamWarpMaps node
************************************************************************************************************************************/
VX_API_ENTRY vx_node VX_API_CALL stitchInitCalcCamWarpMaps(vx_graph graph, void *scalar_params, vx_array cam_params, vx_image valid_map, vx_image padding_map, vx_image src_coord_map, vx_array z_buffer_map)
{
	cam_warp_map_params *cam_warp = (cam_warp_map_params *)scalar_params;
	vx_scalar scCamId = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &cam_warp->camId);
	vx_scalar scLensType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &cam_warp->lens_type);
	vx_scalar scCamW = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &cam_warp->camWidth);
	vx_scalar scCamH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &cam_warp->camHeight);
	vx_scalar scPadCnt = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &cam_warp->paddingPixelCount);
	vx_reference params[] = {
		(vx_reference)scCamId,
		(vx_reference)scLensType,
		(vx_reference)scCamW,
		(vx_reference)scCamH,
		(vx_reference)scPadCnt,
		(vx_reference)cam_params,
		(vx_reference)valid_map,
		(vx_reference)padding_map,
		(vx_reference)src_coord_map,
		(vx_reference)z_buffer_map
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_INIT_CALC_CAMERA_VALID_MAP,
		params,
		dimof(params));

	vxReleaseScalar(&scCamId);
	vxReleaseScalar(&scLensType);
	vxReleaseScalar(&scCamW);
	vxReleaseScalar(&scCamH);
	vxReleaseScalar(&scPadCnt);

	return node;
}

VX_API_ENTRY vx_node VX_API_CALL stitchInitCalcDefCamIdxNode(vx_graph graph, vx_uint32 numCam, vx_uint32 out_width, vx_uint32 out_height, vx_array z_buffer_map, vx_image cam_idx_image)
{
	vx_scalar scNumCam = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &numCam);
	vx_scalar scOutW = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &out_width);
	vx_scalar scOutH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &out_height);
	vx_reference params[] = {
		(vx_reference)scNumCam,
		(vx_reference)scOutW,
		(vx_reference)scOutH,
		(vx_reference)z_buffer_map,
		(vx_reference)cam_idx_image
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_INIT_COMPUTE_DEFAULT_CAMERA_IDX,
		params,
		dimof(params));

	vxReleaseScalar(&scNumCam);
	vxReleaseScalar(&scOutW);
	vxReleaseScalar(&scOutH);

	return node;
}

VX_API_ENTRY vx_node VX_API_CALL stitchInitExtendPadDilateNode(vx_graph graph, vx_uint32 paddingPixels, vx_image valid_map, vx_image padding_map)
{
	vx_scalar scNum = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &paddingPixels);
	vx_reference params[] = {
		(vx_reference)scNum,
		(vx_reference)valid_map,
		(vx_reference)padding_map
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_INIT_EXTEND_PAD_DILATE,
		params,
		dimof(params));
	vxReleaseScalar(&scNum);
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL stitchInitPadVertNode(vx_graph graph, vx_uint32 paddingPixels, vx_uint32 out_W, vx_uint32 out_H, vx_image valid_map, vx_image padding_map)
{
	vx_scalar scNum = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &paddingPixels);
	vx_scalar scWidth = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &out_W);
	vx_scalar scHeight = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &out_H);
	vx_reference params[] = {
		(vx_reference)scNum,
		(vx_reference)scWidth,
		(vx_reference)scHeight,
		(vx_reference)valid_map,
		(vx_reference)padding_map
	};
	vx_node node = stitchCreateNode(graph,
		AMDOVX_KERNEL_STITCHING_INIT_EXTEND_PAD_VERT,
		params,
		dimof(params));
	vxReleaseScalar(&scNum);
	vxReleaseScalar(&scWidth);
	vxReleaseScalar(&scWidth);
	return node;

}

#if _WIN32
#pragma comment(lib, "OpenCL.lib")
#endif
