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

/**********************************************************************
  PUBLIC FUNCTION for OpenVX/WinML user defined functions
  **********************************************************************/
extern "C"  SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
        vx_status status = VX_SUCCESS;

        STATUS_ERROR_CHECK(get_kernels_to_publish());
        STATUS_ERROR_CHECK(Kernel_List->PUBLISH(context));

        return status;
}

/************************************************************************************************************
Add All Kernels to the Kernel List
*************************************************************************************************************/
vx_status get_kernels_to_publish()
{
        vx_status status = VX_SUCCESS;

        Kernel_List = new Kernellist(MAX_KERNELS);

        STATUS_ERROR_CHECK(ADD_KERENEL(WINML_OnnxToMivisionX_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(WINML_ConvertImageToTensor_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(WINML_getTopKLabels_Register));

        return status;
}

/************************************************************************************************************
Add Kernels to the Kernel List
*************************************************************************************************************/
vx_status ADD_KERENEL(std::function<vx_status(vx_context)> func)
{
        vx_status status = VX_SUCCESS;
        STATUS_ERROR_CHECK(Kernel_List->ADD(func));
        return status;
}
