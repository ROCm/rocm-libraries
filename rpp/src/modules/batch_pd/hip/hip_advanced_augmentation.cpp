/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "hip_declarations.hpp"
#include "rpp_hip_host_decls.hpp"

/******************** non_linear_blend ********************/

RppStatus
non_linear_blend_hip_batch_tensor(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_non_linear_blend_batch(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
non_linear_blend_hip_batch_tensor_fp16(Rpp16f *srcPtr1, Rpp16f *srcPtr2, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_non_linear_blend_batch_fp16(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
non_linear_blend_hip_batch_tensor_fp32(Rpp32f *srcPtr1, Rpp32f *srcPtr2, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_non_linear_blend_batch_fp32(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
non_linear_blend_hip_batch_tensor_int8(Rpp8s *srcPtr1, Rpp8s *srcPtr2, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_non_linear_blend_batch_int8(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** water ********************/

RppStatus
water_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_water_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
water_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_water_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
water_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_water_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
water_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_water_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}
/******************** erase ********************/

RppStatus
erase_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp32u* anchor_box_info, Rpp8u* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_erase_batch(srcPtr, dstPtr, anchor_box_info, colors, handle, box_offset, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

        return RPP_SUCCESS;
}

RppStatus
erase_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, Rpp32u* anchor_box_info, Rpp16f* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_erase_batch_fp16(srcPtr, dstPtr, anchor_box_info, colors, handle, box_offset, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
erase_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, Rpp32u* anchor_box_info, Rpp32f* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_erase_batch_fp32(srcPtr, dstPtr, anchor_box_info, colors, handle, box_offset, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
erase_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, Rpp32u* anchor_box_info, Rpp8s* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_erase_batch_int8(srcPtr, dstPtr, anchor_box_info, colors, handle, box_offset, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** color_cast ********************/

RppStatus
color_cast_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_color_cast_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
color_cast_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_color_cast_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
color_cast_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_color_cast_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
color_cast_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_color_cast_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}
/******************** lut ********************/

RppStatus
lut_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp8u* lut, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_lut_batch(srcPtr, dstPtr, lut, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
lut_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, Rpp8s* lut, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_lut_batch_int8(srcPtr, dstPtr, lut, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** crop_and_patch ********************/

RppStatus
crop_and_patch_hip_batch_tensor(Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_and_patch_batch(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_and_patch_hip_batch_tensor_fp16(Rpp16f *srcPtr1, Rpp16f *srcPtr2, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_and_patch_batch_fp16(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_and_patch_hip_batch_tensor_fp32(Rpp32f *srcPtr1, Rpp32f *srcPtr2, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_and_patch_batch_fp32(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
crop_and_patch_hip_batch_tensor_int8(Rpp8s *srcPtr1, Rpp8s *srcPtr2, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_and_patch_batch_int8(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}


/******************** glitch ********************/

RppStatus
glitch_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_glitch_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
glitch_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_glitch_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
glitch_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_glitch_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

RppStatus
glitch_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_glitch_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}
