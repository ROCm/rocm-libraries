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

#include "internal_publishKernels.h"
#include "vx_ext_pop.h"

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

/*******************************************************************************************************************
Bubble Pop C Function
*******************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtPopNode_bubblePop(vx_graph graph, vx_image input, vx_image output)
{

	vx_reference params[] = {
		(vx_reference)input,
		(vx_reference)output,
	};

	return vxCreateNodeByStructure(graph,
								   VX_KERNEL_EXT_POP_BUBBLE_POP,
								   params,
								   dimof(params));
}

/*******************************************************************************************************************
Donut Pop C Function
*******************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtPopNode_donutPop(vx_graph graph, vx_image input, vx_image output)
{

	vx_reference params[] = {
		(vx_reference)input,
		(vx_reference)output,
	};

	return vxCreateNodeByStructure(graph,
								   VX_KERNEL_EXT_POP_DONUT_POP,
								   params,
								   dimof(params));
}