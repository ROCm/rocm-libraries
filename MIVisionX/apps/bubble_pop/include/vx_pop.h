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

#ifndef _VX_EXT_AMD_POP_H_
#define _VX_EXT_AMD_POP_H_

#include "vx_ext_pop.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*! \brief The AMD extension library for Pop */
#define VX_LIBRARY_EXT_POP 3

	/*!
	 * \brief The list of available vision kernels in the Pop extension library.
	 */
	enum vx_kernel_ext_amd_pop_e
	{
		/*!
		 * \brief The Bubble Pop kernel. Kernel name is "org.pop.bubble_pop".
		 */
		VX_KERNEL_EXT_POP_BUBBLE_POP = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_EXT_POP) + 0x001,
		/*!
		 * \brief The Donut Pop kernel. Kernel name is "org.pop.donut_pop".
		 */
		VX_KERNEL_EXT_POP_DONUT_POP = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_EXT_POP) + 0x002,
	};

#ifdef __cplusplus
}
#endif

#endif
