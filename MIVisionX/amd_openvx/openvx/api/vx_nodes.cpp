/*
 * Copyright (c) 2012-2013 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

/*!
 * \file
 * \brief The Graph Mode Interface for all Base Kernels.
 * \author Erik Rainey <erik.rainey@ti.com>
 */

#include "ago_internal.h"

static vx_node vxCreateNodeByStructure(vx_graph graph,
	vx_enum kernelenum,
	vx_reference params[],
	vx_uint32 num)
{
	vx_status status = VX_SUCCESS;
	vx_node node = 0;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByEnum(context, kernelenum);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_uint32 p = 0;
			for (p = 0; p < num; p++)
			{
				if (params[p]) {
					status = vxSetParameterByIndex(node,
						p,
						params[p]);
					if (status != VX_SUCCESS)
					{
						vxAddLogEntry((vx_reference)graph, status, "Kernel %d Parameter %u is invalid.\n", kernelenum, p);
						vxReleaseNode(&node);
						node = 0;
						break;
					}
				}
			}
		}
		else
		{
			vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "Failed to create node with kernel enum %d\n", kernelenum);
			status = VX_ERROR_NO_MEMORY;
		}
		vxReleaseKernel(&kernel);
	}
	else
	{
		vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "failed to retrieve kernel enum %d\n", kernelenum);
		status = VX_ERROR_NOT_SUPPORTED;
	}
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxColorConvertNode(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph, VX_KERNEL_COLOR_CONVERT, params, dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxChannelExtractNode(vx_graph graph,
                             vx_image input,
                             vx_enum channelNum,
                             vx_image output)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar scalar = vxCreateScalar(context, VX_TYPE_ENUM, &channelNum);
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)scalar,
        (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_CHANNEL_EXTRACT,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&scalar); // node hold reference
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxChannelCombineNode(vx_graph graph,
                             vx_image plane0,
                             vx_image plane1,
                             vx_image plane2,
                             vx_image plane3,
                             vx_image output)
{
    vx_reference params[] = {
       (vx_reference)plane0,
       (vx_reference)plane1,
       (vx_reference)plane2,
       (vx_reference)plane3,
       (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_CHANNEL_COMBINE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxSobel3x3Node(vx_graph graph, vx_image input, vx_image output_x, vx_image output_y)
{
    vx_reference params[] = {
       (vx_reference)input,
       (vx_reference)output_x,
       (vx_reference)output_y,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_SOBEL_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxMagnitudeNode(vx_graph graph, vx_image grad_x, vx_image grad_y, vx_image mag)
{
    vx_reference params[] = {
       (vx_reference)grad_x,
       (vx_reference)grad_y,
       (vx_reference)mag,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_MAGNITUDE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxPhaseNode(vx_graph graph, vx_image grad_x, vx_image grad_y, vx_image orientation)
{
    vx_reference params[] = {
       (vx_reference)grad_x,
       (vx_reference)grad_y,
       (vx_reference)orientation,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_PHASE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxScaleImageNode(vx_graph graph, vx_image src, vx_image dst, vx_enum type)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar stype = vxCreateScalar(context, VX_TYPE_ENUM, &type);
    vx_reference params[] = {
        (vx_reference)src,
        (vx_reference)dst,
        (vx_reference)stype,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_SCALE_IMAGE,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&stype);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxTableLookupNode(vx_graph graph, vx_image input, vx_lut lut, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)lut,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_TABLE_LOOKUP,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxHistogramNode(vx_graph graph, vx_image input, vx_distribution distribution)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)distribution,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_HISTOGRAM,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxEqualizeHistNode(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_EQUALIZE_HISTOGRAM,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxAbsDiffNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out)
{
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)out,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_ABSDIFF,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxMeanStdDevNode(vx_graph graph, vx_image input, vx_scalar mean, vx_scalar stddev)
{
    vx_reference params[] = {
       (vx_reference)input,
       (vx_reference)mean,
       (vx_reference)stddev,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_MEAN_STDDEV,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxThresholdNode(vx_graph graph, vx_image input, vx_threshold thesh, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)thesh,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_THRESHOLD,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxIntegralImageNode(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_INTEGRAL_IMAGE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxErode3x3Node(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_ERODE_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxDilate3x3Node(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_DILATE_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxMedian3x3Node(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_MEDIAN_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxBox3x3Node(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_BOX_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxGaussian3x3Node(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_GAUSSIAN_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxConvolveNode(vx_graph graph, vx_image input, vx_convolution conv, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)conv,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_CUSTOM_CONVOLUTION,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxGaussianPyramidNode(vx_graph graph, vx_image input, vx_pyramid gaussian)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)gaussian,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_GAUSSIAN_PYRAMID,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxAccumulateImageNode(vx_graph graph, vx_image input, vx_image accum)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)accum,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_ACCUMULATE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxAccumulateWeightedImageNode(vx_graph graph, vx_image input, vx_scalar alpha, vx_image accum)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)alpha,
        (vx_reference)accum,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_ACCUMULATE_WEIGHTED,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxAccumulateSquareImageNode(vx_graph graph, vx_image input, vx_scalar scalar, vx_image accum)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)scalar,
        (vx_reference)accum,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_ACCUMULATE_SQUARE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxMinMaxLocNode(vx_graph graph,
                        vx_image input,
                        vx_scalar minVal, vx_scalar maxVal,
                        vx_array minLoc, vx_array maxLoc,
                        vx_scalar minCount, vx_scalar maxCount)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)minVal,
        (vx_reference)maxVal,
        (vx_reference)minLoc,
        (vx_reference)maxLoc,
        (vx_reference)minCount,
        (vx_reference)maxCount,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_MINMAXLOC,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxConvertDepthNode(vx_graph graph, vx_image input, vx_image output, vx_enum policy, vx_scalar shift)
{
    vx_scalar pol = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &policy);
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)pol,
        (vx_reference)shift,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                   VX_KERNEL_CONVERTDEPTH,
                                   params,
                                   dimof(params));
    vxReleaseScalar(&pol);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxCannyEdgeDetectorNode(vx_graph graph, vx_image input, vx_threshold hyst,
                                vx_int32 gradient_size, vx_enum norm_type,
                                vx_image output)
{
    vx_scalar gs = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &gradient_size);
    vx_scalar nt = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &norm_type);
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)hyst,
        (vx_reference)gs,
        (vx_reference)nt,
        (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_CANNY_EDGE_DETECTOR,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&gs);
    vxReleaseScalar(&nt);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxAndNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out)
{
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)out,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_AND,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxOrNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out)
{
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)out,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OR,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxXorNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out)
{
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)out,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_XOR,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxNotNode(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
       (vx_reference)input,
       (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_NOT,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxMultiplyNode(vx_graph graph, vx_image in1, vx_image in2, vx_scalar scale, vx_enum overflow_policy, vx_enum rounding_policy, vx_image out)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar spolicy = vxCreateScalar(context, VX_TYPE_ENUM, &overflow_policy);
    vx_scalar rpolicy = vxCreateScalar(context, VX_TYPE_ENUM, &rounding_policy);
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)scale,
       (vx_reference)spolicy,
       (vx_reference)rpolicy,
       (vx_reference)out,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_MULTIPLY,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&spolicy);
    vxReleaseScalar(&rpolicy);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxAddNode(vx_graph graph, vx_image in1, vx_image in2, vx_enum policy, vx_image out)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar spolicy = vxCreateScalar(context, VX_TYPE_ENUM, &policy);
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)spolicy,
       (vx_reference)out,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_ADD,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&spolicy);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxSubtractNode(vx_graph graph, vx_image in1, vx_image in2, vx_enum policy, vx_image out)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar spolicy = vxCreateScalar(context, VX_TYPE_ENUM, &policy);
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)spolicy,
       (vx_reference)out,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_SUBTRACT,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&spolicy);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxWarpAffineNode(vx_graph graph, vx_image input, vx_matrix matrix, vx_enum type, vx_image output)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar stype = vxCreateScalar(context, VX_TYPE_ENUM, &type);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)matrix,
            (vx_reference)stype,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_WARP_AFFINE,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&stype);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxWarpPerspectiveNode(vx_graph graph, vx_image input, vx_matrix matrix, vx_enum type, vx_image output)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar stype = vxCreateScalar(context, VX_TYPE_ENUM, &type);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)matrix,
            (vx_reference)stype,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_WARP_PERSPECTIVE,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&stype);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxHarrisCornersNode(vx_graph graph,
                            vx_image input,
                            vx_scalar strength_thresh,
                            vx_scalar min_distance,
                            vx_scalar sensitivity,
                            vx_int32 gradient_size,
                            vx_int32 block_size,
                            vx_array corners,
                            vx_scalar num_corners)
{
    vx_scalar win = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &gradient_size);
    vx_scalar blk = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &block_size);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)strength_thresh,
            (vx_reference)min_distance,
            (vx_reference)sensitivity,
            (vx_reference)win,
            (vx_reference)blk,
            (vx_reference)corners,
            (vx_reference)num_corners,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_HARRIS_CORNERS,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&win);
    vxReleaseScalar(&blk);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxFastCornersNode(vx_graph graph, vx_image input, vx_scalar strength_thresh, vx_bool nonmax_supression, vx_array corners, vx_scalar num_corners)
{
    vx_scalar nonmax = vxCreateScalar(vxGetContext((vx_reference)graph),VX_TYPE_BOOL, &nonmax_supression);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)strength_thresh,
            (vx_reference)nonmax,
            (vx_reference)corners,
            (vx_reference)num_corners,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_FAST_CORNERS,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&nonmax);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxOpticalFlowPyrLKNode(vx_graph graph,
                               vx_pyramid old_images,
                               vx_pyramid new_images,
                               vx_array old_points,
                               vx_array new_points_estimates,
                               vx_array new_points,
                               vx_enum termination,
                               vx_scalar epsilon,
                               vx_scalar num_iterations,
                               vx_scalar use_initial_estimate,
                               vx_size window_dimension)
{
    vx_scalar term = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &termination);
    vx_scalar winsize = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_SIZE, &window_dimension);
    vx_reference params[] = {
            (vx_reference)old_images,
            (vx_reference)new_images,
            (vx_reference)old_points,
            (vx_reference)new_points_estimates,
            (vx_reference)new_points,
            (vx_reference)term,
            (vx_reference)epsilon,
            (vx_reference)num_iterations,
            (vx_reference)use_initial_estimate,
            (vx_reference)winsize,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_OPTICAL_FLOW_PYR_LK,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&term);
    vxReleaseScalar(&winsize);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxRemapNode(vx_graph graph,
                    vx_image input,
                    vx_remap table,
                    vx_enum policy,
                    vx_image output)
{
    vx_scalar spolicy = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &policy);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)table,
            (vx_reference)spolicy,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_REMAP,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&spolicy);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxHalfScaleGaussianNode(vx_graph graph, vx_image input, vx_image output, vx_int32 kernel_size)
{
    vx_scalar ksize = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &kernel_size);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)output,
            (vx_reference)ksize,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_HALFSCALE_GAUSSIAN,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&ksize);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxCopyNode(vx_graph graph, vx_reference input, vx_reference output)
{
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_COPY,
                                           params,
                                           dimof(params));
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxWeightedAverageNode(vx_graph graph, vx_image img1, vx_scalar alpha, vx_image img2, vx_image output)
{
    vx_reference params[] = {
            (vx_reference)img1,
            (vx_reference)alpha,
            (vx_reference)img2,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_WEIGHTED_AVERAGE,
                                           params,
                                           dimof(params));
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxNonLinearFilterNode(vx_graph graph, vx_enum function, vx_image input, vx_matrix mask, vx_image output)
{
    vx_scalar func = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &function);
    vx_reference params[] = {
            (vx_reference)func,
            (vx_reference)input,
            (vx_reference)mask,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_NON_LINEAR_FILTER,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&func);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxLaplacianPyramidNode(vx_graph graph, vx_image input, vx_pyramid laplacian, vx_image output)
{
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)laplacian,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_LAPLACIAN_PYRAMID,
                                           params,
                                           dimof(params));
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxLaplacianReconstructNode(vx_graph graph, vx_pyramid laplacian, vx_image input, vx_image output)
{
    vx_reference params[] = {
            (vx_reference)laplacian,
            (vx_reference)input,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_LAPLACIAN_RECONSTRUCT,
                                           params,
                                           dimof(params));
    return node;
}