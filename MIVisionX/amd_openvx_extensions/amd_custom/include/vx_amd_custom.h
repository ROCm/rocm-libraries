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

#ifndef _VX_AMD_CUSTOM_H_
#define _VX_AMD_CUSTOM_H_

#include <VX/vx.h>

/*!
 * \file
 * \brief The AMD OpenVX Custom Node Extension Library.
 *
 * \defgroup group_amd_custom Extension: AMD Custom Extension API
 * \brief AMD OpenVX Custom Node Extension.
 */

/*! \brief The type enumeration lists all NN extension types.
 * \ingroup group_amd_custom
 */
enum vx_amd_custom_type_e
{
	VX_TYPE_CUSTOM_PARAMS = VX_TYPE_USER_STRUCT_START + 0x001, /*!< \brief A <tt>\ref vx_nn_convolution_params_t</tt>. */
};

/*! \brief Input parameters for a convolution operation.
 * \ingroup group_amd_custom
 */
typedef struct _vx_amd_custom_params_t
{
	vx_enum function_name; /*!< \brief A <tt> VX_TYPE_ENUM</tt> of the <tt> function name for custom layer</tt> enumeration. */
} vx_amd_custom_params_t;

/*! \brief [Graph] Creates a Custom Layer Node.
 * \ingroup group_amd_custom
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [in] function custom funtion enum.
 * \param [in] custom_parameters for user specified custom_parameters.
 * \param [out] outputs The output tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxCustomLayer(vx_graph graph, vx_tensor inputs, vx_enum function, vx_array custom_parameters, vx_tensor outputs);

#endif
