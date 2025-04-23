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
#include "VX/vx.h"
#include <VX/vx_compatibility.h>
#include "opencv2/opencv.hpp"

#if _WIN32
#define SHARED_PUBLIC __declspec(dllexport)
#else
#define SHARED_PUBLIC __attribute__((visibility("default")))
#endif

#ifndef dimof
#define dimof(x) (sizeof(x) / sizeof(x[0]))
#endif

vx_node vxCreateNodeByStructure(vx_graph graph, vx_enum kernelenum, vx_reference params[], vx_uint32 num);

#ifdef __cplusplus
extern "C"
{
#endif

	/*!***********************************************************************************************************
						VX POP - Bubble Pop VX_API_ENTRY C Function NODE
	*************************************************************************************************************/
	/*! \brief [Graph] Creates a OpenCV blur function node.
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtPopNode_bubblePop(vx_graph graph, vx_image input, vx_image output);

	/*!***********************************************************************************************************
						VX POP - Donut Pop VX_API_ENTRY C Function NODE
	*************************************************************************************************************/
	/*! \brief [Graph] Creates a OpenCV blur function node.
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtPopNode_donutPop(vx_graph graph, vx_image input, vx_image output);

#ifdef __cplusplus
}
#endif
