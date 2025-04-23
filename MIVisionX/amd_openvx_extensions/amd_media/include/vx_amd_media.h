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

#ifndef __VX_AMD_MEDIA_H__
#define __VX_AMD_MEDIA_H__

#include <VX/vx.h>

/*!
 * \file
 * \brief The AMD OpenVX Media Nodes Extension Library.
 *
 * \defgroup group_amd_media Extension: AMD Media Extension API
 * \brief AMD OpenVX Media Nodes Extension
 */

#ifdef __cplusplus
extern "C"
{
#endif

    /*! \brief [Graph] Creates a decoder Node.
     * \ingroup group_amd_media
     * @note - TBD
     * \return <tt> vx_node</tt>.
     * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
     * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
     */
    VX_API_ENTRY vx_node VX_API_CALL amdMediaDecoderNode(vx_graph graph, const char *input_str, vx_image output, vx_array aux_data, vx_int32 loop_decode = 0, vx_bool enable_opencl_output = false, vx_int32 device_id = -1);

    /*! \brief [Graph] Creates a encoder Layer Node.
     * \ingroup group_amd_media
     * * @note - TBD
     * \return <tt> vx_node</tt>.
     * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
     * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
     */
    VX_API_ENTRY vx_node VX_API_CALL amdMediaEncoderNode(vx_graph graph, const char *output_str, vx_image input, vx_array aux_data_in, vx_array aux_data_out, vx_bool enable_gpu_input = false);

#ifdef __cplusplus
}
#endif

#endif
