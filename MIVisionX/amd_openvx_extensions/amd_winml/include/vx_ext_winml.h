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

#ifndef _VX_AMD_EXT_WINML_H_
#define _VX_AMD_EXT_WINML_H_

#include "VX/vx.h"
#include <VX/vx_compatibility.h>
#include <vx_ext_amd.h>

/*!
 * \file
 * \brief The AMD OpenVX WinML Nodes Extension Library.
 *
 * \defgroup group_amd_winml Extension: AMD WinML Extension API
 * \brief AMD OpenVX WinML Interop Nodes Extension [only supported on Windows].
 */

/*! \note [Graph] AMD OpenVX WinML Interop Nodes Extension is only supported on Windows.
 */

#ifndef dimof
/*! \def dimof(x)
 *  \brief A macro to get the number of elements in an array.
 *  \param [in] x The array whose size is to be determined.
 *  \return The number of elements in the array.
 */
#define dimof(x) (sizeof(x) / sizeof(x[0]))
#endif

#if _WIN32
#define SHARED_PUBLIC __declspec(dllexport)
#else
/*! \def SHARED_PUBLIC
 *  \brief A macro to specify public visibility for shared library symbols.
 */
#define SHARED_PUBLIC __attribute__((visibility("default")))
#endif

/*! \brief Creates a node in a graph using a predefined kernel structure.
 *  \param [in] graph The handle to the graph.
 *  \param [in] kernelenum The enum value representing the kernel to be used.
 *  \param [in] params An array of parameter references for the kernel.
 *  \param [in] num The number of parameters in the array.
 *  \return A handle to the created node.
 */
vx_node vxCreateNodeByStructure(vx_graph graph, vx_enum kernelenum, vx_reference params[], vx_uint32 num);

#ifdef __cplusplus
extern "C"
{
#endif

	/*!***********************************************************************************************************
											WinML VX_API_ENTRY C Function NODE
	*************************************************************************************************************/

	/*! \brief [Graph] Creates a WinML import ONNX Model and run function node.
	 * \ingroup group_amd_winml
	 * @note Kernel Name: com.winml.onnx_to_mivisionx
	 * \param [in] graph The reference to the graph.
	 * \param [in] modelLocation The ONNX Model Location in vx_scalar.
	 * \param [in] inputTensorName The ONNX Model Input Tensor Name in vx_scalar.
	 * \param [in] outputTensorName The ONNX Model Output Tensor Name in vx_scalar.
	 * \param [in] inputTensor The Input Tensor.
	 * \param [in] setupArray The setup Array for each model in <tt>\ref VX_TYPE_SIZE</tt> format.
	 * \param [out] outputTensor The output Tensor.
	 * \param [in] deviceKind WinML Deploy Device Kind in vx_scalar [optional] (default: 0).
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtWinMLNode_OnnxToMivisionX(vx_graph graph, vx_scalar modelLocation, vx_scalar inputTensorName, vx_scalar outputTensorName, vx_tensor inputTensor, vx_array setupArray, vx_tensor outputTensor, vx_scalar deviceKind);

	/*! \brief [Graph] Creates a WinML convert image to tensor node.
	 * \ingroup group_amd_winml
	 * @note Kernel Name: com.winml.convert_image_to_tensor
	 * \param [in] graph The reference to the graph.
	 * \param [in] input input image vx_image.
	 * \param [out] output The output Tensor.
	 * \param [in] a The input 'a' in vx_scalar.
	 * \param [in] b The input 'b' in vx_scalar.
	 * \param [in] reverse_channel_order reverse channel order in vx_scalar.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtWinMLNode_convertImageToTensor(vx_graph graph, vx_image input, vx_tensor output, vx_scalar a, vx_scalar b, vx_scalar reverse_channel_order);

	/*! \brief [Graph] Creates a output tensor to Top K labels node.
	 * \ingroup group_amd_winml
	 * @note Kernel Name: com.winml.get_top_k_labels
	 * \param [in] graph The reference to the graph.
	 * \param [in] prob_tensor probability tensor vx_tensor.
	 * \param [in] labelFile label text file location vx_scalar.
	 * \param [out] output_1 Top 1 in vx_scalar.
	 * \param [out] output_2 Top 2 in vx_scalar. [optional]
	 * \param [out] output_3 Top 3 in vx_scalar. [optional]
	 * \param [out] output_4 Top 4 in vx_scalar. [optional]
	 * \param [out] output_5 Top 5 in vx_scalar. [optional]
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtWinMLNode_getTopKLabels(vx_graph graph, vx_tensor prob_tensor, vx_scalar labelFile, vx_scalar output_1, vx_scalar output_2, vx_scalar output_3, vx_scalar output_4, vx_scalar output_5);

#ifdef __cplusplus
}
#endif

#endif
