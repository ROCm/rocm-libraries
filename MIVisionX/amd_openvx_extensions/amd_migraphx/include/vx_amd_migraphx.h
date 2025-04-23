/*
Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef __VX_AMD_MIGRAPHX_H__
#define __VX_AMD_MIGRAPHX_H__

#include <VX/vx.h>
#include <migraphx/migraphx.hpp>

/*!
 * \file
 * \brief The AMD OpenVX MIGraphX Node Extension Library.
 *
 * \defgroup group_amd_migraph Extension: AMD MIGraphX Extension API
 * \brief AMD OpenVX MIGraphX Interop Node Extension
 */

#ifdef __cplusplus
extern "C"
{
#endif

    /*! \brief [Graph] Creates a MIGrpahX Node.
     * \ingroup group_amd_migraph
     * \param [in] graph The handle to the graph.
     * \param [in] path The path to the onnx file
     * \param [in] input the input tensor
     * \param [out] output the output tensor
     * \param [out] fp16q if true then the fp16 quantization will be appiled to the model
     * \param [out] int8q if true then the int8 quantization will be appiled to the model
     * \return <tt> vx_node</tt>.
     * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
     * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
     */
    VX_API_ENTRY vx_node VX_API_CALL amdMIGraphXnode(vx_graph graph, const vx_char *path, vx_tensor input, vx_tensor output, vx_bool fp16q = false, vx_bool int8q = false);

#ifdef __cplusplus
}
#endif

#endif
