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

#ifndef _VX_AMD_WINML_H_
#define _VX_AMD_WINML_H_

#ifdef  __cplusplus
extern "C" {
#endif

 /*! \brief The AMD extension library for WinML */
#define VX_LIBRARY_WINML 2

        /*!
         * \brief The list of available kernels in the WinML extension library.
         */
        enum vx_kernel_ext_amd_winml_e
        {
                /*!
                 * \brief The WinML kernel. Kernel name is "com.winml.onnx_to_mivisionx".
                 */
                VX_KERNEL_WINML_ONNX_TO_MIVISIONX = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_WINML) + 0x00,

		/*!
		 * \brief The WinML kernel. Kernel name is "com.winml.convert_image_to_tensor".
		 */
		VX_KERNEL_WINML_CONVERT_IMAGE_TO_TENSOR = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_WINML) + 0x01,

		/*!
		* \brief The WinML kernel. Kernel name is "com.winml.get_top_k_labels".
		*/
		VX_KERNEL_WINML_GET_TOP_K_LABELS = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_WINML) + 0x02
        };

#ifdef  __cplusplus
}
#endif

#endif
