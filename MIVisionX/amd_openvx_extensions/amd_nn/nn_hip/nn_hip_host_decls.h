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

#ifndef NN_HIP_HOST_DECLS_H
#define NN_HIP_HOST_DECLS_H
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include <VX/vx.h>

int HipExec_Gather_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, unsigned char* in, uint in_offset,
    uint4 in_stride, unsigned char* ind, uint ind_offset, uint4 ind_stride, unsigned char* out, uint out_offset,
    uint4 out_stride, uint axis);

int HipExec_Tile_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, unsigned char* in,
    uint in_offset, uint4 in_stride, uint4 in_dims, unsigned char* rep, uint rep_offset, uint4 rep_stride, unsigned char* out,
    uint out_offset, uint4 out_stride);

int HipExec_Cast_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum input_type, vx_enum output_type, unsigned char* in,
    uint in_offset, uint4 in_stride, unsigned char* out, uint out_offset, uint4 out_stride);

int HipExec_image_to_tensor_layer(hipStream_t stream, vx_df_image format, vx_enum type, uint width, uint height, uint N, unsigned char* in,
    uint in_offset, uint in_stride, unsigned char* out, uint out_offset, uint4 out_stride, float ka, float kb, uint reverse_channel_order);

int HipExec_tensor_to_image_layer(hipStream_t stream, vx_df_image format, vx_enum type, uint width, uint height, uint N, unsigned char* in,
    uint in_offset, uint4 in_stride, unsigned char* out, uint out_offset, uint out_stride, float sc1, float sc2, uint reverse_channel_order);

int HipExec_copy(hipStream_t stream, vx_enum type, unsigned char* inp, unsigned char* out, uint width, uint height, uint ldi, uint i_offset,
    uint ldc, uint c_offset, bool tI);

int HipExec_permute_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, unsigned char* in, uint in_offset, uint4 in_stride,
    unsigned char* order_buf, uint order_offset, uint order_cap, unsigned char* out, uint out_offset, uint4 out_stride);

int HipExec_tensor_log_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, unsigned char *in, uint in_offset,
    uint4 in_stride, unsigned char *out, uint out_offset, uint4 out_stride);

int HipExec_tensor_exp_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, unsigned char *in, uint in_offset,
    uint4 in_stride, unsigned char *out, uint out_offset, uint4 out_stride);

int HipExec_Prior_Box_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, uint imgWidth, uint imgHeight, uint layerWidth,
    uint layerHeight, float minSize, float maxSize, uint flip, uint clip, float offset, uint output_num, uint output_dims_ch2,
    uint num_bytes_for_each_prior, unsigned char *out, uint out_offset, uint4 out_stride, unsigned char *aspect_ratio_buf,
    uint aspect_ratio_offset, uint aspect_ratio_num, unsigned char *variance_buf, uint variance_offset);

int HipExec_Concat_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, unsigned char *out, uint out_offset, size_t output_dim3,
    unsigned char *in_mem[], size_t *in_offset, size_t *ip_size_per_batch, int axis, size_t work_items, int num_inputs, bool batchsz1, vx_enum type);

int HipExec_Argmax_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, unsigned char *i0_buf, uint i0_offset, uint4 i0_stride, uint4 i0_dims,
    unsigned char *o0_buf, uint o0_offset, uint4 o0_stride, uint o0_image_stride, vx_enum output_data_type, uint top_k, vx_enum output_obj_type);

int HipExec_tensor_compare_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, unsigned char* in,
    uint in_offset, uint4 in_stride, unsigned char* in2, uint in2_offset, uint4 in2_stride, unsigned char* out, uint out_offset,
    uint4 out_stride, uint mode);

int HipExec_Upsample_Nearest_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, unsigned char* in, uint in_offset,
    uint4 in_stride, unsigned char* out, uint out_offset, uint4 out_stride);
    
#endif //NN_HIP_HOST_DECLS_H