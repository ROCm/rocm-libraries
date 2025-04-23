/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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


#include "internal_publishKernels.h"
#include "vx_ext_winml.h"

vx_node vxCreateNodeByStructure(vx_graph graph,
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
                                if (params[p])
                                {
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

/************************************************************************************************************
WinML vxExtWinMLNode_OnnxToMivisionX C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtWinMLNode_OnnxToMivisionX
(
        vx_graph graph,
        vx_scalar modelLocation,
        vx_scalar inputTensorName,
        vx_scalar outputTensorName,
        vx_tensor inputTensor,
	vx_array setupArray,
        vx_tensor outputTensor,
	vx_scalar deviceKind		
)
{
        vx_reference params[] = {
                (vx_reference)modelLocation,
                (vx_reference)inputTensorName,
                (vx_reference)outputTensorName,
                (vx_reference)inputTensor,
		(vx_reference)setupArray,
                (vx_reference)outputTensor,
		(vx_reference)deviceKind				
        };

        return vxCreateNodeByStructure(graph,
                VX_KERNEL_WINML_ONNX_TO_MIVISIONX,
                params,
                dimof(params));
}

/************************************************************************************************************
WinML vxExtWinMLNode_ConvertImageToTensorNode C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtWinMLNode_convertImageToTensor
(
	vx_graph graph, 
	vx_image input, 
	vx_tensor output, 
	vx_scalar a,
	vx_scalar b,
	vx_scalar reverse_channel_order
)
{
		vx_reference params[] = {
			(vx_reference)input,
			(vx_reference)output,
			(vx_reference)a,
			(vx_reference)b,
			(vx_reference)reverse_channel_order
		};

		return vxCreateNodeByStructure(graph,
				VX_KERNEL_WINML_CONVERT_IMAGE_TO_TENSOR,
				params, 
				dimof(params));
}

/************************************************************************************************************
WinML vxExtWinMLNode_getTopKLabels C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtWinMLNode_getTopKLabels
(
	vx_graph graph,
	vx_tensor prob_tensor,
	vx_scalar labelFile,
	vx_scalar output_1,
	vx_scalar output_2,
	vx_scalar output_3,
	vx_scalar output_4,
	vx_scalar output_5
)
{

	vx_reference params[] = {
		(vx_reference)prob_tensor,
		(vx_reference)labelFile,
		(vx_reference)output_1,
		(vx_reference)output_2,
		(vx_reference)output_3,
		(vx_reference)output_4,
		(vx_reference)output_5
	};

	return vxCreateNodeByStructure(graph,
		VX_KERNEL_WINML_GET_TOP_K_LABELS,
		params,
		dimof(params));
}
