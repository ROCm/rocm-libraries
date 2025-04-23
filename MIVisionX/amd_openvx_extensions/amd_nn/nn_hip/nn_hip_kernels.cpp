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

#include "../../../amd_openvx/openvx/hipvx/hip_common_funcs.h"
#include "nn_hip_host_decls.h"

// ----------------------------------------------------------------------------
// Neural Network kernels for hip backend
// ----------------------------------------------------------------------------

typedef struct d_half4 {
  __half data[4];
} d_half4;

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Gather_layer(uchar* in, uint in_offset, uint4 in_stride, uchar* ind, uint ind_offset, uint4 ind_stride,
 uchar* out, uint out_offset, uint4 out_stride, uint axis) {

   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

   int indices = *(int*)&ind[ind_offset + y * ind_stride.x];
   T value;
   uint offset;
   if (axis == 0) {
       value = *(T*)&in[in_offset + x * in_stride.x + indices * in_stride.y + z * in_stride.z];
       offset = out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
   } else if (axis == 1) {
       value = *(T*)&in[in_offset + indices * in_stride.x + z * in_stride.y];
       offset = out_offset + y * out_stride.x + z * out_stride.y;
   } else if (axis == 2) {
       value = *(T*)&in[in_offset + z * in_stride.x];
       offset = out_offset + z * out_stride.x;
   }
   out += offset;
   *(T *)&out[0] = value;
}

int HipExec_Gather_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, uchar* in,
    uint in_offset, uint4 in_stride, uchar* ind, uint ind_offset, uint4 ind_stride, uchar* out, uint out_offset,
    uint4 out_stride, uint axis) {

    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x),
                        ceil((float)globalThreads.y/localThreads.y),
                        ceil((float)globalThreads.z/localThreads.z));

    if (type == VX_TYPE_FLOAT32) {
        hipLaunchKernelGGL(Hip_Gather_layer<float>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
            ind, ind_offset, ind_stride, out, out_offset, out_stride, axis);
    } else {
        hipLaunchKernelGGL(Hip_Gather_layer<__half>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
            ind, ind_offset, ind_stride, out, out_offset, out_stride, axis);
    }

    return VX_SUCCESS;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Tile_layer(uchar* in, uint in_offset, uint4 in_stride, uint4 in_dims, uchar* rep, uint rep_offset,
    uint4 rep_stride, uchar* out, uint out_offset, uint4 out_stride) {

   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

   uint nx = x % in_dims.x;
   uint ny = y % in_dims.y;
   uint nz = z % in_dims.z;

   T value = *(T*)&in[in_offset + nx * in_stride.x + ny * in_stride.y + nz * in_stride.z];
   uint offset = out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
   out += offset;
   *(T *)&out[0] = value;
}

int HipExec_Tile_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, uchar* in,
    uint in_offset, uint4 in_stride, uint4 in_dims, uchar* rep, uint rep_offset, uint4 rep_stride, uchar* out,
    uint out_offset, uint4 out_stride) {

    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x),
                        ceil((float)globalThreads.y/localThreads.y),
                        ceil((float)globalThreads.z/localThreads.z));

    if (type == VX_TYPE_FLOAT32) {
        hipLaunchKernelGGL(Hip_Tile_layer<float>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
            in_dims, rep, rep_offset, rep_stride, out, out_offset, out_stride);
    } else {
        hipLaunchKernelGGL(Hip_Tile_layer<__half>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
            in_dims, rep, rep_offset, rep_stride, out, out_offset, out_stride);
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int32_float_v(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   float4 f = *(float4 *)in;
   *(int4 *)&out[0] = make_int4(__float2int_rd(f.x), __float2int_rd(f.y), __float2int_rd(f.z), __float2int_rd(f.w));
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int64_float_v(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   float4 f = *(float4 *)in;
   *(longlong4 *)&out[0] = make_longlong4(__float2ll_rd(f.x), __float2ll_rd(f.y), __float2ll_rd(f.z), __float2ll_rd(f.w));
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_float_float_v(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   *(float4 *)&out[0] = *(float4 *)in;
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int64_int32_v(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   int4 f = *(int4 *)in;
   *(longlong4 *)&out[0] = make_longlong4(f.x, f.y, f.z, f.w);
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int32_int64_v(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   longlong4 f = *(longlong4 *)in;
   *(int4 *)&out[0] = make_int4(f.x, f.y, f.z, f.w);
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int32_float(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   *(int *)&out[0] = __float2int_rd(*(float *)in);
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int64_float(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   *(long long int*)&out[0] = __float2ll_rd(*(float *)in);
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_float_float(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   *(float *)&out[0] = *(float *)in;
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int64_int32(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   int f = *(int *)in;
   *(long long int*)&out[0] = (long long int)f;
}

__global__ void __attribute__((visibility("default")))
Hip_Cast_layer_int32_int64(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {
   uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
   in += in_offset + z * in_stride.z + y * in_stride.y + x * in_stride.x;
   out += out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x;
   long long int f = *(long long int*)in;
   *(int *)&out[0] = (int)f;
}

int HipExec_Cast_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum input_type, vx_enum output_type, uchar* in, uint in_offset,
    uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {

    bool input_element_count_multiple_of_4 = ((globalThreads.x * globalThreads.y) & 3) ? false : true;
    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x),
                        ceil((float)globalThreads.y/localThreads.y),
                        ceil((float)globalThreads.z/localThreads.z));

    if(input_element_count_multiple_of_4) {
        if(input_type == VX_TYPE_FLOAT32) {
            if(output_type == VX_TYPE_INT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_int32_float_v, gridDim, localThreads, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            } else if(output_type == VX_TYPE_INT64) {
                hipLaunchKernelGGL(Hip_Cast_layer_int64_float_v, gridDim, localThreads, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            } else if(output_type == VX_TYPE_FLOAT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_float_float_v, gridDim, localThreads, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        } else if(input_type == VX_TYPE_INT32) {
            if(output_type == VX_TYPE_INT64) {
                hipLaunchKernelGGL(Hip_Cast_layer_int64_int32_v, gridDim, localThreads, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        } else if(input_type == VX_TYPE_INT64) {
            if(output_type == VX_TYPE_INT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_int32_int64_v, gridDim, localThreads, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        }
    } else {
        if(input_type == VX_TYPE_FLOAT32) {
            if(output_type == VX_TYPE_INT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_int32_float, gridDim, localThreads, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            } else if(output_type == VX_TYPE_INT64) {
                hipLaunchKernelGGL(Hip_Cast_layer_int64_float, gridDim, localThreads, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            } else if(output_type == VX_TYPE_FLOAT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_float_float, gridDim, localThreads, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        } else if(input_type == VX_TYPE_INT32) {
            if(output_type == VX_TYPE_INT64) {
                hipLaunchKernelGGL(Hip_Cast_layer_int64_int32, gridDim, localThreads, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        } else if(input_type == VX_TYPE_INT64) {
            if(output_type == VX_TYPE_INT32) {
                hipLaunchKernelGGL(Hip_Cast_layer_int32_int64, gridDim, localThreads, 0, stream, in, in_offset, in_stride, out, out_offset, out_stride);
            }
        }
    }

    return VX_SUCCESS;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_RGBimage_to_tensor_layer(uint srcWidth, uint srcHeight, uchar* in, uint in_offset, uint in_stride, uchar* out,
    uint out_offset, uint4 out_stride, float sc1, float sc2, uint reverse_channel_order) {

    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if(x < srcWidth && y < srcHeight) {
        uint ioffset = in_offset + (y + z * srcHeight) * in_stride + x * 3;
        uint2 rgb2 = *(uint2 *)&in[ioffset & ~3];
        uint rgb = hip_bytealign(rgb2.y, rgb2.x, ioffset & 3);
        T r = (T)sc1 * (T)hip_unpack0(rgb) + (T)sc2;
        T g = (T)sc1 * (T)hip_unpack1(rgb) + (T)sc2;
        T b = (T)sc1 * (T)hip_unpack2(rgb) + (T)sc2;
        out += out_offset + z * out_stride.w + y * out_stride.y + x * out_stride.x;
        *(T *)&out[0] = reverse_channel_order ? b : r;
        *(T *)&out[out_stride.z] = g;
        *(T *)&out[2 * out_stride.z] = reverse_channel_order ? r : b;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_U8floatImage_to_tensor_layer(uint srcWidth, uint srcHeight, uchar* in, uint in_offset, uint in_stride, uchar* out,
    uint out_offset, uint4 out_stride, float sc1, float sc2) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
    if(x < srcWidth && y < srcHeight) {
        uint u4 = *(uint *)&in[in_offset + (y + z * srcHeight) * in_stride + x];
        float p0 = sc1 * hip_unpack0(u4) + sc2;
        float p1 = sc1 * hip_unpack1(u4) + sc2;
        float p2 = sc1 * hip_unpack2(u4) + sc2;
        float p3 = sc1 * hip_unpack3(u4) + sc2;
        *(float4 *)&out[out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x] = make_float4(p0, p1, p2, p3);
    }
}

__global__ void __attribute__((visibility("default")))
Hip_U8halfImage_to_tensor_layer(uint srcWidth, uint srcHeight, uchar* in, uint in_offset, uint in_stride, uchar* out,
    uint out_offset, uint4 out_stride, float sc1, float sc2) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
    if(x < srcWidth && y < srcHeight) {
        uint u4 = *(uint *)&in[in_offset + (y + z * srcHeight) * in_stride + x];

        __half hsc1 = __float2half(sc1);
        __half hsc2 = __float2half(sc2);

        __half p0 = hsc1 * __float2half(hip_unpack0(u4)) + hsc2;
        __half p1 = hsc1 * __float2half(hip_unpack1(u4)) + hsc2;
        __half p2 = hsc1 * __float2half(hip_unpack2(u4)) + hsc2;
        __half p3 = hsc1 * __float2half(hip_unpack3(u4)) + hsc2;

        d_half4 p;
        p.data[0] = p0;
        p.data[1] = p1;
        p.data[2] = p2;
        p.data[3] = p3;
        *(d_half4 *)&out[out_offset + z * out_stride.z + y * out_stride.y + x * out_stride.x] = p;
    }
}

int HipExec_image_to_tensor_layer(hipStream_t stream, vx_df_image format, vx_enum type, uint width, uint height, uint N, uchar* in,
    uint in_offset, uint in_stride, uchar* out, uint out_offset, uint4 out_stride, float sc1, float sc2, uint reverse_channel_order) {

    dim3 blockDim(8, 8, 1);
    uint newWidth = (format == VX_DF_IMAGE_RGB) ? width : (width + 3) / 4;
    dim3 gridDim = dim3(ceil((float)newWidth / blockDim.x), ceil((float)height / blockDim.y), ceil((float)N / blockDim.z));

    if (format == VX_DF_IMAGE_RGB) {
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(Hip_RGBimage_to_tensor_layer<float>, gridDim, blockDim, 0, stream, width, height, in, in_offset, in_stride,
                out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        } else {
            hipLaunchKernelGGL(Hip_RGBimage_to_tensor_layer<__half>, gridDim, blockDim, 0, stream, width, height, in, in_offset, in_stride,
                out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        }
    } else if (format == VX_DF_IMAGE_U8) {
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(Hip_U8floatImage_to_tensor_layer, gridDim, blockDim, 0, stream, width, height, in, in_offset, in_stride,
                out, out_offset, out_stride, sc1, sc2);
        } else {
            hipLaunchKernelGGL(Hip_U8halfImage_to_tensor_layer, gridDim, blockDim, 0, stream, width, height, in, in_offset, in_stride,
                out, out_offset, out_stride, sc1, sc2);
        }
    } else {
        return VX_ERROR_NOT_SUPPORTED;
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_tensor_to_RGBfloatImage_layer(uchar* in, uint in_offset, uint4 in_stride, uint outWidth, uint outHeight, uchar* out, uint out_offset,
    uint out_stride, float sc1, float sc2, uint reverse_channel_order) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if(x < outWidth && y < outHeight) {
        in += in_offset + z * in_stride.w + y * in_stride.y + x * in_stride.x;
        float4 r = *(float4 *)&in[reverse_channel_order ? 2 * in_stride.z : 0];
        float4 g = *(float4 *)&in[in_stride.z];
        float4 b = *(float4 *)&in[reverse_channel_order ? 0 : 2 * in_stride.z];
        r = r * (float4)sc1 + (float4)sc2;
        g = g * (float4)sc1 + (float4)sc2;
        b = b * (float4)sc1 + (float4)sc2;
        uint3 u3;
        u3.x = hip_pack(make_float4(r.x, g.x, b.x, r.y));
        u3.y = hip_pack(make_float4(g.y, b.y, r.z, g.z));
        u3.z = hip_pack(make_float4(b.z, r.w, g.w, b.w));
        *(uint3 *)&out[out_offset + (y + z * outHeight) * out_stride + x * 3] = u3;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_tensor_to_RGBhalfImage_layer(uchar* in, uint in_offset, uint4 in_stride, uint outWidth, uint outHeight, uchar* out,
    uint out_offset, uint out_stride, float sc1, float sc2, uint reverse_channel_order) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if(x < outWidth && y < outHeight) {
        in += in_offset + z * in_stride.w + y * in_stride.y + x * in_stride.x;
        d_half4 r = *(d_half4 *)&in[reverse_channel_order ? 2 * in_stride.z : 0];
        d_half4 g = *(d_half4 *)&in[in_stride.z];
        d_half4 b = *(d_half4 *)&in[reverse_channel_order ? 0 : 2 * in_stride.z];

        __half hsc1 = __float2half(sc1);
        __half hsc2 = __float2half(sc2);

        r.data[0] = r.data[0] * hsc1 + hsc2;
        r.data[1] = r.data[1] * hsc1 + hsc2;
        r.data[2] = r.data[2] * hsc1 + hsc2;
        r.data[3] = r.data[3] * hsc1 + hsc2;

        g.data[0] = g.data[0] * hsc1 + hsc2;
        g.data[1] = g.data[1] * hsc1 + hsc2;
        g.data[2] = g.data[2] * hsc1 + hsc2;
        g.data[3] = g.data[3] * hsc1 + hsc2;

        b.data[0] = b.data[0] * hsc1 + hsc2;
        b.data[1] = b.data[1] * hsc1 + hsc2;
        b.data[2] = b.data[2] * hsc1 + hsc2;
        b.data[3] = b.data[3] * hsc1 + hsc2;

        uint3 u3;
        u3.x = hip_pack(make_float4(__half2float(r.data[0]), __half2float(g.data[0]), __half2float(b.data[0]), __half2float(r.data[1])));
        u3.y = hip_pack(make_float4(__half2float(g.data[1]), __half2float(b.data[1]), __half2float(r.data[2]), __half2float(g.data[2])));
        u3.z = hip_pack(make_float4(__half2float(b.data[2]), __half2float(r.data[3]), __half2float(g.data[3]), __half2float(b.data[3])));
        *(uint3 *)&out[out_offset + (y + z * outHeight) * out_stride + x * 3] = u3;
    }
}


__global__ void __attribute__((visibility("default")))
Hip_tensor_to_U8floatImage_layer(uchar* in, uint in_offset, uint4 in_stride, uint outWidth, uint outHeight, uchar* out,
    uint out_offset, uint out_stride, float sc1, float sc2, uint reverse_channel_order) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if(x < outWidth && y < outHeight) {
        in += in_offset + z * in_stride.w + y * in_stride.y + x * in_stride.x;
        float4 f = *(float4 *)in;
        f = f * (float4)sc1 + (float4)sc2;
        *(uint *)&out[out_offset + (y + z * outHeight) * out_stride + x] = hip_pack(f);
    }
}

__global__ void __attribute__((visibility("default")))
Hip_tensor_to_U8halfImage_layer(uchar* in, uint in_offset, uint4 in_stride, uint outWidth, uint outHeight, uchar* out,
    uint out_offset, uint out_stride, float sc1, float sc2, uint reverse_channel_order) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if(x < outWidth && y < outHeight) {
        in += in_offset + z * in_stride.w + y * in_stride.y + x * in_stride.x;
        d_half4 h = *(d_half4 *)in;

        float4 f;
        f.x = __half2float(h.data[0]);
        f.y = __half2float(h.data[1]);
        f.z = __half2float(h.data[2]);
        f.w = __half2float(h.data[3]);

        f = f * (float4)sc1 + (float4)sc2;
        *(uint *)&out[out_offset + (y + z * outHeight) * out_stride + x] = hip_pack(f);
    }
}

int HipExec_tensor_to_image_layer(hipStream_t stream, vx_df_image format, vx_enum type, uint width, uint height, uint N, uchar* in,
    uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint out_stride, float sc1, float sc2, uint reverse_channel_order) {

    dim3 blockDim(8, 8, 1);
    dim3 gridDim = dim3(ceil((float)((width + 3) / 4) / blockDim.x), ceil((float)height / blockDim.y), ceil((float)N / blockDim.z));

    if (format == VX_DF_IMAGE_RGB) {
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(Hip_tensor_to_RGBfloatImage_layer, gridDim, blockDim, 0, stream, in, in_offset, in_stride,
                width, height, out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        } else {
            hipLaunchKernelGGL(Hip_tensor_to_RGBhalfImage_layer, gridDim, blockDim, 0, stream, in, in_offset, in_stride,
                width, height, out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        }
    } else if (format == VX_DF_IMAGE_U8) {
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(Hip_tensor_to_U8floatImage_layer, gridDim, blockDim, 0, stream, in, in_offset, in_stride,
                width, height, out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        } else {
            hipLaunchKernelGGL(Hip_tensor_to_U8halfImage_layer, gridDim, blockDim, 0, stream, in, in_offset, in_stride,
                width, height, out, out_offset, out_stride, sc1, sc2, reverse_channel_order);
        }
    } else {
        return VX_ERROR_NOT_SUPPORTED;
    }

    return VX_SUCCESS;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
copy_v1(const T* inp, T* out, uint width, uint height, uint BLKW, uint ldi, uint i_offset, uint ldc, uint c_offset) {
    __shared__ float lbuf[256];
    uint gx = blockIdx.x;
    uint gy = blockIdx.y;
    uint lx = threadIdx.x;
    uint ly = threadIdx.y;
    uint ix = hip_mad24(gx, BLKW, lx);
    uint iy = hip_mad24(gy, BLKW, ly);
    if (ix < width && iy < height) {
        uint iloc = iy * ldi + ix + i_offset;
        lbuf[hip_mad24(ly, BLKW + 1, lx)] = inp[iloc];
    }
    __syncthreads();
    uint ox = hip_mad24(gy, BLKW, lx);
    uint oy = hip_mad24(gx, BLKW, ly);
    if(oy < width && ox < height) {
        uint oloc = oy * ldc + ox + c_offset;
        out[oloc] = lbuf[hip_mad24(lx, BLKW + 1, ly)];
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
copy_v2(const T* inp, T* out, uint width, uint height, uint ldi, uint i_offset, uint ldc, uint c_offset) {
    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if(x < width && y < height) {
        uint i = y * ldi + x + i_offset;
        uint o = y * ldc + x + c_offset;
        out[o] = inp[i];
    }
}

int HipExec_copy(hipStream_t stream, vx_enum type, uchar* inp, uchar* out, uint width, uint height, uint ldi, uint i_offset,
    uint ldc, uint c_offset, bool tI) {
    if(tI) {
        dim3 blockDim(16, 16, 1);
        dim3 gridDim = dim3(ceil((float)width / blockDim.x), ceil((float)height / blockDim.y), 1);
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(copy_v1<float>, gridDim, blockDim, 0, stream, (float*)inp, (float*)out, width, height, blockDim.x, ldi,
                i_offset, ldc, c_offset);
        } else {
            hipLaunchKernelGGL(copy_v1<__half>, gridDim, blockDim, 0, stream, (__half*)inp, (__half*)out, width, height, blockDim.x, ldi,
                i_offset, ldc, c_offset);
        }
    } else {
        dim3 blockDim(64, 1, 1);
        dim3 gridDim = dim3(ceil((float)width / blockDim.x), height, 1);
        if (type == VX_TYPE_FLOAT32) {
            hipLaunchKernelGGL(copy_v2<float>, gridDim, blockDim, 0, stream, (float*)inp, (float*)out, width, height, ldi, i_offset, ldc, c_offset);
        } else {
            hipLaunchKernelGGL(copy_v2<float>, gridDim, blockDim, 0, stream, (float*)inp, (float*)out, width, height, ldi, i_offset, ldc, c_offset);
        }
    }
    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_permute_layer(uchar* in, uint in_offset, uint4 in_stride, uchar* order_buf, uint order_offset, uint order_cap, uchar* out, uint out_offset,
    uint4 out_stride) {
    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;
    int num_axis = order_cap;
    int i = x * out_stride.x + y * out_stride.y + z * out_stride.z;
    int old_idx = 0;
    int idx = i;
    for(int k = num_axis - 1, j = 0; k >= 0; k--, j++) {
        int order = 3 - ((int *)(order_buf + order_offset))[j];
        old_idx += (idx / out_stride.data[k]) * (in_stride.data[order]);
        idx %= (out_stride.data[k]);
    }
    out += out_offset + i;
    in += in_offset + old_idx;
    *(float4 *)&out[0] = *(float4 *)&in[0];
}

int HipExec_permute_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, uchar* in, uint in_offset, uint4 in_stride, uchar* order_buf,
    uint order_offset, uint order_cap, uchar* out, uint out_offset, uint4 out_stride) {

    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x),
                        ceil((float)globalThreads.y/localThreads.y),
                        ceil((float)globalThreads.z/localThreads.z));

    hipLaunchKernelGGL(Hip_permute_layer, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
        order_buf, order_offset, order_cap, out, out_offset, out_stride);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_tensor_log_layer(uchar *in, uint in_offset, uint4 in_stride, uchar *out, uint out_offset, uint4 out_stride) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

     float4 value = *(float4 *)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
     out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
     *(float4 *)&out[0] = make_float4(log(value.x), log(value.y), log(value.z), log(value.w));
 }

__global__ void __attribute__((visibility("default")))
Hip_tensor_log_layer_half(uchar *in, uint in_offset, uint4 in_stride, uchar *out, uint out_offset, uint4 out_stride) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

     d_half4 value = *(d_half4 *)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
     out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
     d_half4 p;
     p.data[0] = hlog(value.data[0]);
     p.data[1] = hlog(value.data[1]);
     p.data[2] = hlog(value.data[2]);
     p.data[3] = hlog(value.data[3]);
     *(d_half4 *)&out[0] = p;
 }

int HipExec_tensor_log_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, uchar *in, uint in_offset, uint4 in_stride, uchar *out,
    uint out_offset, uint4 out_stride) {

    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x),
                        ceil((float)globalThreads.y/localThreads.y),
                        ceil((float)globalThreads.z/localThreads.z));

    if (type == VX_TYPE_FLOAT32) {
        hipLaunchKernelGGL(Hip_tensor_log_layer, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
            out, out_offset, out_stride);
    } else {
        hipLaunchKernelGGL(Hip_tensor_log_layer_half, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
            out, out_offset, out_stride);
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_tensor_exp_layer(uchar *in, uint in_offset, uint4 in_stride, uchar *out, uint out_offset, uint4 out_stride) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

     float4 value = *(float4 *)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
     out += out_offset + x  * out_stride.x + y * out_stride.y + z * out_stride.z;
     *(float4 *)&out[0] = make_float4(exp(value.x), exp(value.y), exp(value.z), exp(value.w));
 }

__global__ void __attribute__((visibility("default")))
Hip_tensor_exp_layer_half(uchar *in, uint in_offset, uint4 in_stride, uchar *out, uint out_offset, uint4 out_stride) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    d_half4 value = *(d_half4 *)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
    out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;

     d_half4 p;
     p.data[0] = hexp(value.data[0]);
     p.data[1] = hexp(value.data[1]);
     p.data[2] = hexp(value.data[2]);
     p.data[3] = hexp(value.data[3]);
     *(d_half4 *)&out[0] = p;
 }

int HipExec_tensor_exp_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, uchar *in, uint in_offset, uint4 in_stride, uchar *out,
    uint out_offset, uint4 out_stride) {

    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x),
                        ceil((float)globalThreads.y/localThreads.y),
                        ceil((float)globalThreads.z/localThreads.z));

    if (type == VX_TYPE_FLOAT32) {
        hipLaunchKernelGGL(Hip_tensor_exp_layer, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
            out, out_offset, out_stride);
    } else {
        hipLaunchKernelGGL(Hip_tensor_exp_layer_half, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
            out, out_offset, out_stride);
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Prior_Box_layer(uint imgWidth, uint imgHeight, uint layerWidth, uint layerHeight, float minSize, float maxSize, uint flip,
    uint clip, float offset, uint output_num, uint output_dims_ch2, uint num_bytes_for_each_prior, uchar *out, uint out_offset,
    uint4 out_stride, uchar *aspect_ratio_buf, uint aspect_ratio_offset, uint aspect_ratio_num, uchar *variance_buf, uint variance_offset) {

    uchar *out_ptr = out;
    const float step_x = (float)imgWidth /(float)layerWidth;
    const float step_y = (float)imgHeight /(float)layerHeight;
    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    float center_x = (x + offset) * step_x;
    float center_y = (y + offset) * step_y;
    float box_width, box_height;
    box_width = minSize;
    box_height = minSize;
    out += out_offset + y * hipGridDim_x * num_bytes_for_each_prior + x * num_bytes_for_each_prior;
    *(float *)&out[0] = (center_x - box_width * .5) / imgWidth;
    out += out_stride.y;
    *(float *)&out[0] = (center_y - box_height * .5) / imgHeight;
    out += out_stride.y;
    *(float *)&out[0] = (center_x + box_width * .5) / imgWidth;
    out += out_stride.y;
    *(float *)&out[0] = (center_y + box_height * .5) / imgHeight;
    if (maxSize > 0) {
        box_width = sqrtf((float)(minSize * maxSize));
        box_height = sqrtf((float)(minSize * maxSize));
        out += out_stride.y;
        *(float *)&out[0] = (center_x - box_width * .5) / imgWidth;
        out += out_stride.y;
        *(float *)&out[0] = (center_y - box_height * .5) / imgHeight;
        out += out_stride.y;
        *(float *)&out[0] = (center_x + box_width * .5) / imgWidth;
        out += out_stride.y;
        *(float *)&out[0] = (center_y + box_height * .5) / imgHeight;
    }
    int r = 0;
    while(r < aspect_ratio_num) {
        float ar = ((float *)(aspect_ratio_buf + aspect_ratio_offset))[r];
        if(ar == 0.0f || fabsf(ar - (float)1.) < 1e-6) {
            r++;
            continue;
        }
        box_width = minSize * sqrtf(ar);
        box_height = minSize / sqrtf(ar);
        out += out_stride.y; 
        *(float *)&out[0] = (center_x - box_width * .5) / imgWidth;
        out += out_stride.y; 
        *(float *)&out[0] = (center_y - box_height * .5) / imgHeight;
        out += out_stride.y; 
        *(float *)&out[0] = (center_x + box_width * .5) / imgWidth;
        out += out_stride.y; 
        *(float *)&out[0] = (center_y + box_height * .5) / imgHeight;
        if(flip == 1) {
            float ar_flip=  1 / ar;
            if(isinf(ar_flip) || fabsf(ar_flip - (float)1.) < 1e-6) {
                r++;
                continue;
            }
            box_width = minSize * sqrtf(ar_flip);
            box_height = minSize / sqrtf(ar_flip);
            out += out_stride.y;
            *(float *)&out[0] = (center_x - box_width * .5) / imgWidth;
            out += out_stride.y;
            *(float *)&out[0] = (center_y - box_height * .5) / imgHeight;
            out += out_stride.y;
            *(float *)&out[0] = (center_x + box_width * .5) / imgWidth;
            out += out_stride.y;
            *(float *)&out[0] = (center_y + box_height * .5) / imgHeight;
        }
        r++;
    }
    if (clip == 1) {
        int idx = 0;
        out = out_ptr;
        while(idx < (output_dims_ch2 - 1)) {
            ((float *)(out + out_offset))[idx] = fminf(fmaxf((float)out[idx], (float)0.), (float)1.);
            idx++;
        }
    }
    int count = output_dims_ch2;
    out = out_ptr;
    while(count < output_num) {
        ((float4 *)(out + out_offset))[count++] = ((float4 *)(variance_buf + variance_offset))[0];
    }

}

int HipExec_Prior_Box_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, uint imgWidth, uint imgHeight, uint layerWidth,
    uint layerHeight, float minSize, float maxSize, uint flip, uint clip, float offset, uint output_num, uint output_dims_ch2,
    uint num_bytes_for_each_prior, uchar *out, uint out_offset, uint4 out_stride, uchar *aspect_ratio_buf, uint aspect_ratio_offset,
    uint aspect_ratio_num, uchar *variance_buf, uint variance_offset) {

    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x),
                        ceil((float)globalThreads.y/localThreads.y),
                        ceil((float)globalThreads.z/localThreads.z));

    hipLaunchKernelGGL(Hip_Prior_Box_layer, gridDim, localThreads, 0, stream, imgWidth, imgHeight,
        layerWidth, layerHeight, minSize, maxSize, flip, clip, offset, output_num, output_dims_ch2 / 4, num_bytes_for_each_prior, out, out_offset,
        out_stride, aspect_ratio_buf, aspect_ratio_offset, aspect_ratio_num, variance_buf, variance_offset);

    return VX_SUCCESS;
}


template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat2_layer(uchar *out, uint out_offset, uchar *in0, uint in0_offset, size_t ip_size_per_batch0, uchar *in1, uint in1_offset, size_t ip_size_per_batch1,
    const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[2] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;

    if (id < work_items) {
        outn += out_offset >> 2;
        if (id < ip_size_per_batch0) {
            in0n += (in0_offset >> 2);
            outn[id] = in0n[id];
        } else if ((id >= ip_buffer_offset[1]) && (id < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2);
            outn[id] = in1n[id - ip_buffer_offset[1]];
        }
    }

}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat3_layer(uchar *out, uint out_offset, uchar *in0, uint in0_offset, size_t ip_size_per_batch0, uchar *in1, uint in1_offset, size_t ip_size_per_batch1,
    uchar *in2, uint in2_offset, size_t ip_size_per_batch2, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[3] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;

    if (id < work_items) {
        outn += out_offset >> 2;
        if (id < ip_size_per_batch0) {
            in0n += (in0_offset >> 2);
            outn[id] = in0n[id];
        } else if ((id >= ip_buffer_offset[1]) && (id < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2);
            outn[id] = in1n[id - ip_buffer_offset[1]];
        } else if ((id >= ip_buffer_offset[2]) && (id < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2);
            outn[id] = in2n[id - ip_buffer_offset[2]];
        }
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat4_layer(uchar *out, uint out_offset, uchar *in0, uint in0_offset, size_t ip_size_per_batch0, uchar *in1, uint in1_offset, size_t ip_size_per_batch1,
    uchar *in2, uint in2_offset, size_t ip_size_per_batch2, uchar *in3, uint in3_offset, size_t ip_size_per_batch3, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[4] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;
    ip_buffer_offset[3] += ip_buffer_offset[2] + ip_size_per_batch2;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;
    T *in3n = (T*)in3;

    if (id < work_items) {
        out += out_offset >> 2;
        if (id < ip_size_per_batch0) {
            in0n += (in0_offset >> 2);
            outn[id] = in0n[id];
        } else if ((id >= ip_buffer_offset[1]) && (id < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2);
            outn[id] = in1n[id - ip_buffer_offset[1]];
        } else if ((id >= ip_buffer_offset[2]) && (id < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2);
            outn[id] = in2n[id - ip_buffer_offset[2]];
        } else if ((id >= ip_buffer_offset[3]) && (id < ip_buffer_offset[3] + ip_size_per_batch3)) {
            in3n += (in3_offset >> 2);
            outn[id] = in3n[id - ip_buffer_offset[3]];
        }
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat5_layer(uchar *out, uint out_offset, uchar *in0, uint in0_offset, size_t ip_size_per_batch0, uchar *in1, uint in1_offset, size_t ip_size_per_batch1,
    uchar *in2, uint in2_offset, size_t ip_size_per_batch2, uchar *in3, uint in3_offset, size_t ip_size_per_batch3, uchar *in4, uint in4_offset,
    size_t ip_size_per_batch4, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[5] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;
    ip_buffer_offset[3] += ip_buffer_offset[2] + ip_size_per_batch2;
    ip_buffer_offset[4] += ip_buffer_offset[3] + ip_size_per_batch3;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;
    T *in3n = (T*)in3;
    T *in4n = (T*)in4;

    if (id < work_items) {
        outn += out_offset >> 2;
        if (id < ip_size_per_batch0) {
            in0n += (in0_offset >> 2);
            outn[id] = in0n[id];
        } else if ((id >= ip_buffer_offset[1]) && (id < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2);
            outn[id] = in1n[id - ip_buffer_offset[1]];
        } else if ((id >= ip_buffer_offset[2]) && (id < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2);
            outn[id] = in2n[id - ip_buffer_offset[2]];
        } else if ((id >= ip_buffer_offset[3]) && (id < ip_buffer_offset[3] + ip_size_per_batch3)) {
            in3n += (in3_offset >> 2);
            outn[id] = in3n[id - ip_buffer_offset[3]];
        }  else if ((id >= ip_buffer_offset[4]) && (id < ip_buffer_offset[4] + ip_size_per_batch4)) {
            in4n += (in4_offset >> 2);
            outn[id] = in4n[id - ip_buffer_offset[4]];
        }
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat6_layer(uchar *out, uint out_offset, uchar *in0, uint in0_offset, size_t ip_size_per_batch0, uchar *in1, uint in1_offset, size_t ip_size_per_batch1,
    uchar *in2, uint in2_offset, size_t ip_size_per_batch2, uchar *in3, uint in3_offset, size_t ip_size_per_batch3, uchar *in4, uint in4_offset,
    size_t ip_size_per_batch4, uchar *in5, uint in5_offset, size_t ip_size_per_batch5, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[6] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;
    ip_buffer_offset[3] += ip_buffer_offset[2] + ip_size_per_batch2;
    ip_buffer_offset[4] += ip_buffer_offset[3] + ip_size_per_batch3;
    ip_buffer_offset[5] += ip_buffer_offset[4] + ip_size_per_batch4;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;
    T *in3n = (T*)in3;
    T *in4n = (T*)in4;
    T *in5n = (T*)in5;

    if (id < work_items) {
        out += out_offset >> 2;
        if (id < ip_size_per_batch0) {
            in0 += (in0_offset >> 2);
            outn[id] = in0n[id];
        } else if ((id >= ip_buffer_offset[1]) && (id < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2);
            outn[id] = in1n[id - ip_buffer_offset[1]];
        } else if ((id >= ip_buffer_offset[2]) && (id < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2);
            outn[id] = in2n[id - ip_buffer_offset[2]];
        } else if ((id >= ip_buffer_offset[3]) && (id < ip_buffer_offset[3] + ip_size_per_batch3)) {
            in3n += (in3_offset >> 2);
            outn[id] = in3n[id - ip_buffer_offset[3]];
        } else if ((id >= ip_buffer_offset[4]) && (id < ip_buffer_offset[4] + ip_size_per_batch4)) {
            in4n += (in4_offset >> 2);
            outn[id] = in4n[id - ip_buffer_offset[4]];
        } else if ((id >= ip_buffer_offset[5]) && (id < ip_buffer_offset[5] + ip_size_per_batch5)) {
            in5n += (in5_offset >> 2);
            outn[id] = in5n[id - ip_buffer_offset[5]];
        }
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat7_layer(uchar *out, uint out_offset, uchar *in0, uint in0_offset, size_t ip_size_per_batch0, uchar *in1, uint in1_offset, size_t ip_size_per_batch1,
    uchar *in2, uint in2_offset, size_t ip_size_per_batch2, uchar *in3, uint in3_offset, size_t ip_size_per_batch3, uchar *in4, uint in4_offset,
    size_t ip_size_per_batch4, uchar *in5, uint in5_offset, size_t ip_size_per_batch5, uchar *in6, uint in6_offset, size_t ip_size_per_batch6,
    const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[7] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;
    ip_buffer_offset[3] += ip_buffer_offset[2] + ip_size_per_batch2;
    ip_buffer_offset[4] += ip_buffer_offset[3] + ip_size_per_batch3;
    ip_buffer_offset[5] += ip_buffer_offset[4] + ip_size_per_batch4;
    ip_buffer_offset[6] += ip_buffer_offset[5] + ip_size_per_batch5;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;
    T *in3n = (T*)in3;
    T *in4n = (T*)in4;
    T *in5n = (T*)in5;
    T *in6n = (T*)in6;

    if (id < work_items) {
        outn += out_offset >> 2;
        if (id < ip_size_per_batch0) {
            in0n += (in0_offset >> 2);
            outn[id] = in0n[id];
        } else if ((id >= ip_buffer_offset[1]) && (id < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2);
            outn[id] = in1n[id - ip_buffer_offset[1]];
        } else if ((id >= ip_buffer_offset[2]) && (id < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2);
            outn[id] = in2n[id - ip_buffer_offset[2]];
        } else if ((id >= ip_buffer_offset[3]) && (id < ip_buffer_offset[3] + ip_size_per_batch3)) {
            in3n += (in3_offset >> 2);
            outn[id] = in3n[id - ip_buffer_offset[3]];
        } else if ((id >= ip_buffer_offset[4]) && (id < ip_buffer_offset[4] + ip_size_per_batch4)) {
            in4n += (in4_offset >> 2);
            outn[id] = in4n[id - ip_buffer_offset[4]];
        } else if ((id >= ip_buffer_offset[5]) && (id < ip_buffer_offset[5] + ip_size_per_batch5)) {
            in5n += (in5_offset >> 2);
            outn[id] = in5n[id - ip_buffer_offset[5]];
        } else if ((id >= ip_buffer_offset[6]) && (id < ip_buffer_offset[6] + ip_size_per_batch6)) {
            in6n += (in6_offset >> 2);
            outn[id] = in6n[id - ip_buffer_offset[6]];
        }

    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat8_layer(uchar *out, uint out_offset, uchar *in0, uint in0_offset, size_t ip_size_per_batch0, uchar *in1, uint in1_offset, size_t ip_size_per_batch1,
    uchar *in2, uint in2_offset, size_t ip_size_per_batch2, uchar *in3, uint in3_offset, size_t ip_size_per_batch3, uchar *in4, uint in4_offset,
    size_t ip_size_per_batch4, uchar *in5, uint in5_offset, size_t ip_size_per_batch5, uchar *in6, uint in6_offset, size_t ip_size_per_batch6,
    uchar *in7, uint in7_offset, size_t ip_size_per_batch7, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[8] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;
    ip_buffer_offset[3] += ip_buffer_offset[2] + ip_size_per_batch2;
    ip_buffer_offset[4] += ip_buffer_offset[3] + ip_size_per_batch3;
    ip_buffer_offset[5] += ip_buffer_offset[4] + ip_size_per_batch4;
    ip_buffer_offset[6] += ip_buffer_offset[5] + ip_size_per_batch5;
    ip_buffer_offset[7] += ip_buffer_offset[6] + ip_size_per_batch6;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;
    T *in3n = (T*)in3;
    T *in4n = (T*)in4;
    T *in5n = (T*)in5;
    T *in6n = (T*)in6;
    T *in7n = (T*)in7;

    if (id < work_items) {
        out += out_offset >> 2;
        if (id < ip_size_per_batch0) {
            in0n += (in0_offset >> 2);
            outn[id] = in0n[id];
        } else if ((id >= ip_buffer_offset[1]) && (id < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2);
            outn[id] = in1n[id - ip_buffer_offset[1]];
        } else if ((id >= ip_buffer_offset[2]) && (id < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2);
            outn[id] = in2n[id - ip_buffer_offset[2]];
        } else if ((id >= ip_buffer_offset[3]) && (id < ip_buffer_offset[3] + ip_size_per_batch3)) {
            in3n += (in3_offset >> 2);
            outn[id] = in3n[id - ip_buffer_offset[3]];
        } else if ((id >= ip_buffer_offset[4]) && (id < ip_buffer_offset[4] + ip_size_per_batch4)) {
            in4n += (in4_offset >> 2);
            outn[id] = in4n[id - ip_buffer_offset[4]];
        } else if ((id >= ip_buffer_offset[5]) && (id < ip_buffer_offset[5] + ip_size_per_batch5)) {
            in5n += (in5_offset >> 2);
            outn[id] = in5n[id - ip_buffer_offset[5]];
        } else if ((id >= ip_buffer_offset[6]) && (id < ip_buffer_offset[6] + ip_size_per_batch6)) {
            in6n += (in6_offset >> 2);
            outn[id] = in6n[id - ip_buffer_offset[6]];
        } else if ((id >= ip_buffer_offset[7]) && (id < ip_buffer_offset[7] + ip_size_per_batch7)) {
            in7n += (in7_offset >> 2);
            outn[id] = in7n[id - ip_buffer_offset[7]];
        }

    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat2_batch_layer(uchar *out, uint out_offset, size_t output_dim3, uchar *in0, uint in0_offset, size_t ip_size_per_batch0, uchar *in1, uint in1_offset,
    size_t ip_size_per_batch1, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[2] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;

    if (id < work_items) {
        size_t batch_id = id / output_dim3;
        size_t id_within_batch = id - batch_id * output_dim3;
        outn += out_offset >> 2;
        if (id_within_batch < ip_size_per_batch0) {
            in0n += (in0_offset >> 2) + (batch_id * ip_size_per_batch0);
            outn[id] = in0n[id_within_batch];
        } else if ((id_within_batch >= ip_buffer_offset[1]) && (id_within_batch < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2) + (batch_id * ip_size_per_batch1);
            outn[id] = in1n[id_within_batch - ip_buffer_offset[1]];
        }
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat3_batch_layer(uchar *out, uint out_offset, size_t output_dim3, uchar *in0, uint in0_offset, size_t ip_size_per_batch0, uchar *in1, uint in1_offset,
    size_t ip_size_per_batch1, uchar *in2, uint in2_offset, size_t ip_size_per_batch2, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[3] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;

    if (id < work_items) {
        size_t batch_id = id / output_dim3;
        size_t id_within_batch = id - batch_id * output_dim3;
        outn += out_offset >> 2;
        if (id_within_batch < ip_size_per_batch0) {
            in0n += (in0_offset >> 2) + (batch_id * ip_size_per_batch0);
            outn[id] = in0n[id_within_batch];
        } else if ((id_within_batch >= ip_buffer_offset[1]) && (id_within_batch < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2) + (batch_id * ip_size_per_batch1);
            outn[id] = in1n[id_within_batch - ip_buffer_offset[1]];
        } else if ((id_within_batch >= ip_buffer_offset[2]) && (id_within_batch < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2) + (batch_id * ip_size_per_batch2);
            outn[id] = in2n[id_within_batch - ip_buffer_offset[2]];
        }
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat4_batch_layer(uchar *out, uint out_offset, size_t output_dim3, uchar *in0, uint in0_offset, size_t ip_size_per_batch0,
    uchar *in1, uint in1_offset, size_t ip_size_per_batch1, uchar *in2, uint in2_offset, size_t ip_size_per_batch2, uchar *in3,
    uint in3_offset, size_t ip_size_per_batch3, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[4] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;
    ip_buffer_offset[3] += ip_buffer_offset[2] + ip_size_per_batch2;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;
    T *in3n = (T*)in3;

    if (id < work_items) {
        size_t batch_id = id / output_dim3;
        size_t id_within_batch = id - batch_id * output_dim3;
        outn += out_offset >> 2;
        if (id_within_batch < ip_size_per_batch0) {
            in0n += (in0_offset >> 2) + (batch_id * ip_size_per_batch0);
            outn[id] = in0n[id_within_batch];
        } else if ((id_within_batch >= ip_buffer_offset[1]) && (id_within_batch < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2) + (batch_id * ip_size_per_batch1);
            outn[id] = in1n[id_within_batch - ip_buffer_offset[1]];
        } else if ((id_within_batch >= ip_buffer_offset[2]) && (id_within_batch < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2) + (batch_id * ip_size_per_batch2);
            outn[id] = in2n[id_within_batch - ip_buffer_offset[2]];
        } else if ((id_within_batch >= ip_buffer_offset[3]) && (id_within_batch < ip_buffer_offset[3] + ip_size_per_batch3)) {
            in3n += (in3_offset >> 2) + (batch_id * ip_size_per_batch3);
            outn[id] = in3n[id_within_batch - ip_buffer_offset[3]];
        }
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat5_batch_layer(uchar *out, uint out_offset, size_t output_dim3, uchar *in0, uint in0_offset, size_t ip_size_per_batch0,
    uchar *in1, uint in1_offset, size_t ip_size_per_batch1, uchar *in2, uint in2_offset, size_t ip_size_per_batch2, uchar *in3,
    uint in3_offset, size_t ip_size_per_batch3, uchar *in4, uint in4_offset, size_t ip_size_per_batch4, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[5] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;
    ip_buffer_offset[3] += ip_buffer_offset[2] + ip_size_per_batch2;
    ip_buffer_offset[4] += ip_buffer_offset[3] + ip_size_per_batch3;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;
    T *in3n = (T*)in3;
    T *in4n = (T*)in4;

    if (id < work_items) {
        size_t batch_id = id / output_dim3;
        size_t id_within_batch = id - batch_id * output_dim3;
        outn += out_offset >> 2;
        if (id_within_batch < ip_size_per_batch0) {
            in0n += (in0_offset >> 2) + (batch_id * ip_size_per_batch0);
            outn[id] = in0n[id_within_batch];
        } else if ((id_within_batch >= ip_buffer_offset[1]) && (id_within_batch < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2) + (batch_id * ip_size_per_batch1);
            outn[id] = in1n[id_within_batch - ip_buffer_offset[1]];
        } else if ((id_within_batch >= ip_buffer_offset[2]) && (id_within_batch < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2) + (batch_id * ip_size_per_batch2);
            outn[id] = in2n[id_within_batch - ip_buffer_offset[2]];
        } else if ((id_within_batch >= ip_buffer_offset[3]) && (id_within_batch < ip_buffer_offset[3] + ip_size_per_batch3)) {
            in3n += (in3_offset >> 2) + (batch_id * ip_size_per_batch3);
            outn[id] = in3n[id_within_batch - ip_buffer_offset[3]];
        } else if ((id_within_batch >= ip_buffer_offset[4]) && (id_within_batch < ip_buffer_offset[4] + ip_size_per_batch4)) {
            in4n += (in4_offset >> 2) + (batch_id * ip_size_per_batch4);
            outn[id] = in4n[id_within_batch - ip_buffer_offset[4]];
        }
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat6_batch_layer(uchar *out, uint out_offset, size_t output_dim3, uchar *in0, uint in0_offset, size_t ip_size_per_batch0,
    uchar *in1, uint in1_offset, size_t ip_size_per_batch1, uchar *in2, uint in2_offset, size_t ip_size_per_batch2, uchar *in3,
    uint in3_offset, size_t ip_size_per_batch3, uchar *in4, uint in4_offset, size_t ip_size_per_batch4, uchar *in5, uint in5_offset,
    size_t ip_size_per_batch5, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[6] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;
    ip_buffer_offset[3] += ip_buffer_offset[2] + ip_size_per_batch2;
    ip_buffer_offset[4] += ip_buffer_offset[3] + ip_size_per_batch3;
    ip_buffer_offset[5] += ip_buffer_offset[4] + ip_size_per_batch4;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;
    T *in3n = (T*)in3;
    T *in4n = (T*)in4;
    T *in5n = (T*)in5;

    if (id < work_items) {
        size_t batch_id = id / output_dim3;
        size_t id_within_batch = id - batch_id * output_dim3;
        outn += out_offset >> 2;
        if (id_within_batch < ip_size_per_batch0) {
            in0n += (in0_offset >> 2) + (batch_id * ip_size_per_batch0);
            outn[id] = in0n[id_within_batch];
        } else if ((id_within_batch >= ip_buffer_offset[1]) && (id_within_batch < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2) + (batch_id * ip_size_per_batch1);
            outn[id] = in1n[id_within_batch - ip_buffer_offset[1]];
        } else if ((id_within_batch >= ip_buffer_offset[2]) && (id_within_batch < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2) + (batch_id * ip_size_per_batch2);
            outn[id] = in2n[id_within_batch - ip_buffer_offset[2]];
        } else if ((id_within_batch >= ip_buffer_offset[3]) && (id_within_batch < ip_buffer_offset[3] + ip_size_per_batch3)) {
            in3n += (in3_offset >> 2) + (batch_id * ip_size_per_batch3);
            outn[id] = in3n[id_within_batch - ip_buffer_offset[3]];
        } else if ((id_within_batch >= ip_buffer_offset[4]) && (id_within_batch < ip_buffer_offset[4] + ip_size_per_batch4)) {
            in4n += (in4_offset >> 2) + (batch_id * ip_size_per_batch4);
            outn[id] = in4n[id_within_batch - ip_buffer_offset[4]];
        } else if ((id_within_batch >= ip_buffer_offset[5]) && (id_within_batch < ip_buffer_offset[5] + ip_size_per_batch5)) {
            in5n += (in5_offset >> 2) + (batch_id * ip_size_per_batch5);
            outn[id] = in5n[id_within_batch - ip_buffer_offset[5]];
        }
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat7_batch_layer(uchar *out, uint out_offset, size_t output_dim3, uchar *in0, uint in0_offset, size_t ip_size_per_batch0,
    uchar *in1, uint in1_offset, size_t ip_size_per_batch1, uchar *in2, uint in2_offset, size_t ip_size_per_batch2, uchar *in3,
    uint in3_offset, size_t ip_size_per_batch3, uchar *in4, uint in4_offset, size_t ip_size_per_batch4, uchar *in5, uint in5_offset,
    size_t ip_size_per_batch5, uchar *in6, uint in6_offset, size_t ip_size_per_batch6, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[7] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;
    ip_buffer_offset[3] += ip_buffer_offset[2] + ip_size_per_batch2;
    ip_buffer_offset[4] += ip_buffer_offset[3] + ip_size_per_batch3;
    ip_buffer_offset[5] += ip_buffer_offset[4] + ip_size_per_batch4;
    ip_buffer_offset[6] += ip_buffer_offset[5] + ip_size_per_batch5;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;
    T *in3n = (T*)in3;
    T *in4n = (T*)in4;
    T *in5n = (T*)in5;
    T *in6n = (T*)in6;

    if (id < work_items) {
        size_t batch_id = id / output_dim3;
        size_t id_within_batch = id - batch_id * output_dim3;
        outn += out_offset >> 2;
        if (id_within_batch < ip_size_per_batch0) {
            in0n += (in0_offset >> 2) + (batch_id * ip_size_per_batch0);
            outn[id] = in0n[id_within_batch];
        } else if ((id_within_batch >= ip_buffer_offset[1]) && (id_within_batch < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2) + (batch_id * ip_size_per_batch1);
            outn[id] = in1n[id_within_batch - ip_buffer_offset[1]];
        } else if ((id_within_batch >= ip_buffer_offset[2]) && (id_within_batch < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2) + (batch_id * ip_size_per_batch2);
            outn[id] = in2n[id_within_batch - ip_buffer_offset[2]];
        } else if ((id_within_batch >= ip_buffer_offset[3]) && (id_within_batch < ip_buffer_offset[3] + ip_size_per_batch3)) {
            in3n += (in3_offset >> 2) + (batch_id * ip_size_per_batch3);
            outn[id] = in3n[id_within_batch - ip_buffer_offset[3]];
        } else if ((id_within_batch >= ip_buffer_offset[4]) && (id_within_batch < ip_buffer_offset[4] + ip_size_per_batch4)) {
            in4n += (in4_offset >> 2) + (batch_id * ip_size_per_batch4);
            outn[id] = in4n[id_within_batch - ip_buffer_offset[4]];
        } else if ((id_within_batch >= ip_buffer_offset[5]) && (id_within_batch < ip_buffer_offset[5] + ip_size_per_batch5)) {
            in5n += (in5_offset >> 2) + (batch_id * ip_size_per_batch5);
            outn[id] = in5n[id_within_batch - ip_buffer_offset[5]];
        } else if ((id_within_batch >= ip_buffer_offset[6]) && (id_within_batch < ip_buffer_offset[6] + ip_size_per_batch6)) {
            in6n += (in6_offset >> 2) + (batch_id * ip_size_per_batch6);
            outn[id] = in6n[id_within_batch - ip_buffer_offset[6]];
        }
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Concat8_batch_layer(uchar *out, uint out_offset, size_t output_dim3, uchar *in0, uint in0_offset, size_t ip_size_per_batch0,
    uchar *in1, uint in1_offset, size_t ip_size_per_batch1, uchar *in2, uint in2_offset, size_t ip_size_per_batch2, uchar *in3,
    uint in3_offset, size_t ip_size_per_batch3, uchar *in4, uint in4_offset, size_t ip_size_per_batch4, uchar *in5, uint in5_offset,
    size_t ip_size_per_batch5, uchar *in6, uint in6_offset, size_t ip_size_per_batch6, uchar *in7, uint in7_offset,
    size_t ip_size_per_batch7, const int axis, size_t work_items) {

    size_t id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    vx_size ip_buffer_offset[8] = {0};
    ip_buffer_offset[1] += ip_size_per_batch0;
    ip_buffer_offset[2] += ip_buffer_offset[1] + ip_size_per_batch1;
    ip_buffer_offset[3] += ip_buffer_offset[2] + ip_size_per_batch2;
    ip_buffer_offset[4] += ip_buffer_offset[3] + ip_size_per_batch3;
    ip_buffer_offset[5] += ip_buffer_offset[4] + ip_size_per_batch4;
    ip_buffer_offset[6] += ip_buffer_offset[5] + ip_size_per_batch5;
    ip_buffer_offset[7] += ip_buffer_offset[6] + ip_size_per_batch6;

    T *outn = (T*)out;
    T *in0n = (T*)in0;
    T *in1n = (T*)in1;
    T *in2n = (T*)in2;
    T *in3n = (T*)in3;
    T *in4n = (T*)in4;
    T *in5n = (T*)in5;
    T *in6n = (T*)in6;
    T *in7n = (T*)in7;

    if (id < work_items) {
        size_t batch_id = id / output_dim3;
        size_t id_within_batch = id - batch_id * output_dim3;
        outn += out_offset >> 2;
        if (id_within_batch < ip_size_per_batch0) {
            in0n += (in0_offset >> 2) + (batch_id * ip_size_per_batch0);
            outn[id] = in0n[id_within_batch];
        } else if ((id_within_batch >= ip_buffer_offset[1]) && (id_within_batch < ip_buffer_offset[1] + ip_size_per_batch1)) {
            in1n += (in1_offset >> 2) + (batch_id * ip_size_per_batch1);
            outn[id] = in1n[id_within_batch - ip_buffer_offset[1]];
        } else if ((id_within_batch >= ip_buffer_offset[2]) && (id_within_batch < ip_buffer_offset[2] + ip_size_per_batch2)) {
            in2n += (in2_offset >> 2) + (batch_id * ip_size_per_batch2);
            outn[id] = in2n[id_within_batch - ip_buffer_offset[2]];
        } else if ((id_within_batch >= ip_buffer_offset[3]) && (id_within_batch < ip_buffer_offset[3] + ip_size_per_batch3)) {
            in3n += (in3_offset >> 2) + (batch_id * ip_size_per_batch3);
            outn[id] = in3n[id_within_batch - ip_buffer_offset[3]];
        } else if ((id_within_batch >= ip_buffer_offset[4]) && (id_within_batch < ip_buffer_offset[4] + ip_size_per_batch4)) {
            in4n += (in4_offset >> 2) + (batch_id * ip_size_per_batch4);
            outn[id] = in4n[id_within_batch - ip_buffer_offset[4]];
        } else if ((id_within_batch >= ip_buffer_offset[5]) && (id_within_batch < ip_buffer_offset[5] + ip_size_per_batch5)) {
            in5n += (in5_offset >> 2) + (batch_id * ip_size_per_batch5);
            outn[id] = in5n[id_within_batch - ip_buffer_offset[5]];
        } else if ((id_within_batch >= ip_buffer_offset[6]) && (id_within_batch < ip_buffer_offset[6] + ip_size_per_batch6)) {
            in6n += (in6_offset >> 2) + (batch_id * ip_size_per_batch6);
            outn[id] = in6n[id_within_batch - ip_buffer_offset[6]];
        } else if ((id_within_batch >= ip_buffer_offset[7]) && (id_within_batch < ip_buffer_offset[7] + ip_size_per_batch7)) {
            in7n += (in7_offset >> 2) + (batch_id * ip_size_per_batch7);
            outn[id] = in7n[id_within_batch - ip_buffer_offset[7]];
        }
    }
}

int HipExec_Concat_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, uchar *out, uint out_offset, size_t output_dim3,
    uchar *in_mem[], size_t in_offset[], size_t ip_size_per_batch[], int axis, size_t work_items, int num_inputs, bool batchsz1, vx_enum type) {

    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x),
                        ceil((float)globalThreads.y/localThreads.y),
                        ceil((float)globalThreads.z/localThreads.z));

    switch (num_inputs) {
        case 2:
            if (batchsz1) {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat2_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat2_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], axis, work_items);
                }
            } else {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat2_batch_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat2_batch_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], axis, work_items);
                }
            }
            break;
        case 3:
            if (batchsz1) {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat3_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat3_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], axis, work_items);
                }
            } else {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat3_batch_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat3_batch_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], axis, work_items);
                }
            }
            break;
        case 4:
            if (batchsz1) {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat4_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat4_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], axis, work_items);
                }
            } else {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat4_batch_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat4_batch_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], axis, work_items);
                }
            }
            break;
        case 5:
            if (batchsz1) {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat5_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat5_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], axis, work_items);
                }
            } else {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat5_batch_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat5_batch_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], axis, work_items);
                }
            }
            break;
        case 6:
            if (batchsz1) {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat6_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat6_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], axis, work_items);
                }
            } else {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat6_batch_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat6_batch_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], axis, work_items);
                }
            }
            break;
        case 7:
            if (batchsz1) {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat7_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], in_mem[6], in_offset[6], ip_size_per_batch[6], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat7_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], in_mem[6], in_offset[6], ip_size_per_batch[6], axis, work_items);
                }
            } else {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat7_batch_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], in_mem[6], in_offset[6], ip_size_per_batch[6], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat7_batch_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], in_mem[6], in_offset[6], ip_size_per_batch[6], axis, work_items);
                }
            }
            break;
        case 8:
            if (batchsz1) {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat8_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], in_mem[6], in_offset[6], ip_size_per_batch[6], in_mem[7], in_offset[7], ip_size_per_batch[7], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat8_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], in_mem[6], in_offset[6], ip_size_per_batch[6], in_mem[7], in_offset[7], ip_size_per_batch[7], axis, work_items);
                }
            } else {
                if (type == VX_TYPE_FLOAT32) {
                    hipLaunchKernelGGL(Hip_Concat8_batch_layer<float>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], in_mem[6], in_offset[6], ip_size_per_batch[6], in_mem[7], in_offset[7], ip_size_per_batch[7], axis, work_items);
                } else {
                    hipLaunchKernelGGL(Hip_Concat8_batch_layer<__half>, gridDim, localThreads, 0, stream, out, out_offset,
                        output_dim3, in_mem[0], in_offset[0], ip_size_per_batch[0], in_mem[1], in_offset[1], ip_size_per_batch[1], in_mem[2], in_offset[2],
                        ip_size_per_batch[2], in_mem[3], in_offset[3], ip_size_per_batch[3], in_mem[4], in_offset[4], ip_size_per_batch[4], in_mem[5],
                        in_offset[5], ip_size_per_batch[5], in_mem[6], in_offset[6], ip_size_per_batch[6], in_mem[7], in_offset[7], ip_size_per_batch[7], axis, work_items);
                }
            }
            break;
        default:
            return VX_ERROR_NOT_SUPPORTED;
    }

    return VX_SUCCESS;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Argmax_topk1_layer(uchar * i0_buf, uint i0_offset, uint4 i0_stride, uint4 i0_dims, uchar *o0_buf, uint o0_offset, uint4 o0_stride,
    uint o0_image_stride, uint m, bool isOutputImage) {

    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if (x < i0_dims.x && y < i0_dims.y) {
        i0_buf += i0_offset + z * i0_stride.w + y * i0_stride.y + x * i0_stride.x;
        uint cmax = 0;
        float fmax = *(float *)i0_buf;
        for (uint c = 1; c < i0_dims.z; c++) {
            i0_buf += i0_stride.z;
            float f = *(float *)i0_buf;
            cmax = (f > fmax) ? c : cmax;
            fmax = (f > fmax) ? f : fmax;
        }
        if (isOutputImage) {
            o0_buf += o0_offset + (z * i0_dims.y + y) * o0_image_stride + x * m;
        }
        else {
            o0_buf += o0_offset + z * o0_stride.w + y * o0_stride.y + x * o0_stride.x;
        }
        *(T *)o0_buf = (T)cmax;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Argmax_topk1_m4_u8_layer(uchar * i0_buf, uint i0_offset, uint4 i0_stride, uint4 i0_dims, uchar *o0_buf, uint o0_offset, uint4 o0_stride,
    uint o0_image_stride, uint m, bool isOutputImage) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x ) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if (x < i0_dims.x && y < i0_dims.y) {
        i0_buf += i0_offset + z * i0_stride.w + y * i0_stride.y + x * i0_stride.x;
        uint4 cmax = (uint4)0;
        float4 fmax = *(float4 *)i0_buf;
        for (uint c = 1; c < i0_dims.z; c++) {
            i0_buf += i0_stride.z;
            float4 f = *(float4 *)i0_buf;
            cmax.x = (f.x > fmax.x) ? c : cmax.x;
            fmax.x = (f.x > fmax.x) ? f.x : fmax.x;
            cmax.y = (f.y > fmax.y) ? c : cmax.y;
            fmax.y = (f.y > fmax.y) ? f.y : fmax.y;
            cmax.z = (f.z > fmax.z) ? c : cmax.z;
            fmax.z = (f.z > fmax.z) ? f.z : fmax.z;
            cmax.w = (f.w > fmax.w) ? c : cmax.w;
            fmax.w = (f.w > fmax.w) ? f.w : fmax.w;
        }
        if (isOutputImage) {
            o0_buf += o0_offset + (z * i0_dims.y + y) * o0_image_stride + x * m;
        }
        else {
            o0_buf += o0_offset + z * o0_stride.w + y * o0_stride.y + x * o0_stride.x;
        }
        uint imax = cmax.x + (cmax.y << 8) + (cmax.z << 16) + (cmax.w << 24);
        *(uint *)o0_buf = imax;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Argmax_topk1_m4_u16_i64_layer(uchar * i0_buf, uint i0_offset, uint4 i0_stride, uint4 i0_dims, uchar *o0_buf, uint o0_offset, uint4 o0_stride,
    uint factor, uint o0_image_stride, uint m, bool isOutputImage) {

   uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
   uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
   uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if (x < i0_dims.x && y < i0_dims.y) {
        i0_buf += i0_offset + z * i0_stride.w + y * i0_stride.y + x * i0_stride.x;
        uint4 cmax = (uint4)0;
        float4 fmax = *(float4 *)i0_buf;
        for (uint c = 1; c < i0_dims.z; c++) {
            i0_buf += i0_stride.z;
            float4 f = *(float4 *)i0_buf;
            cmax.x = (f.x > fmax.x) ? c : cmax.x;
            fmax.x = (f.x > fmax.x) ? f.x : fmax.x;
            cmax.y = (f.y > fmax.y) ? c : cmax.y;
            fmax.y = (f.y > fmax.y) ? f.y : fmax.y;
            cmax.z = (f.z > fmax.z) ? c : cmax.z;
            fmax.z = (f.z > fmax.z) ? f.z : fmax.z;
            cmax.w = (f.w > fmax.w) ? c : cmax.w;
            fmax.w = (f.w > fmax.w) ? f.w : fmax.w;
        }

        if (isOutputImage) {
            o0_buf += o0_offset + (z * i0_dims.y + y) * o0_image_stride + x * m;
        }
        else {
            o0_buf += o0_offset + z * o0_stride.w + y * o0_stride.y + x * o0_stride.x;
        }

        uint2 imax;
        imax.x = cmax.x + (cmax.y << factor);
        imax.y = cmax.z + (cmax.w << factor);
        *(uint2 *)o0_buf = imax;
    }
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Argmax_topk2_layer(uchar * i0_buf, uint i0_offset, uint4 i0_stride, uint4 i0_dims, uchar *o0_buf, uint o0_offset, uint4 o0_stride,
    uint o0_image_stride, uint m, bool isOutputImage) {

    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if (x < i0_dims.x && y < i0_dims.y) {
        i0_buf += i0_offset + z * i0_stride.w + y * i0_stride.y + x * i0_stride.x;
        uint cmax = 0;
        uint cmax1;
        float f, fmax, fmax1;
        fmax = *(float *)i0_buf;
        i0_buf += i0_stride.z; f = *(float *)i0_buf;
        cmax1 = (f > fmax) ? 0 : 1;
        cmax = (f > fmax) ? 1 : 0;
        fmax1 = (f > fmax) ? fmax : f;
        fmax = (f > fmax) ? f : fmax;
        for (uint c = 2; c < i0_dims.z; c++) {
            i0_buf += i0_stride.z; f = *(float *)i0_buf;
            cmax1 = (f > fmax) ? cmax : ((f > fmax1) ? c : cmax1);
            fmax1 = (f > fmax) ? fmax : ((f > fmax1) ? f : fmax1);
            cmax  = (f > fmax) ? c : cmax;
            fmax  = (f > fmax) ? f : fmax;
        }

        if (isOutputImage) {
            o0_buf += o0_offset + (z * i0_dims.y + y) * o0_image_stride + x * m;
        }
        else {
            o0_buf += o0_offset + z * o0_stride.w + y * o0_stride.y + x * o0_stride.x;
        }

        *(T *)o0_buf = (T)cmax;
        *(T *)&o0_buf[o0_stride.z] = (T)cmax1;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Argmax_topk2_m4_u8_layer(uchar * i0_buf, uint i0_offset, uint4 i0_stride, uint4 i0_dims, uchar *o0_buf, uint o0_offset, uint4 o0_stride,
    uint o0_image_stride, uint m, bool isOutputImage) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if (x < i0_dims.x && y < i0_dims.y) {
        i0_buf += i0_offset + z * i0_stride.w + y * i0_stride.y + x * i0_stride.x;
        uint4 cmax = (uint4)0;
        uint4 cmax1;
        float4 f, fmax, fmax1;
        fmax = *(float4 *)i0_buf;
        i0_buf += i0_stride.z; f = *(float4 *)i0_buf;
        cmax1.x = (f.x > fmax.x) ? 0 : 1;
        cmax.x = (f.x > fmax.x) ? 1 : 0;
        fmax1.x = (f.x > fmax.x) ? fmax.x : f.x;
        fmax.x = (f.x > fmax.x) ? f.x : fmax.x;
        cmax1.y = (f.y > fmax.y) ? 0 : 1;
        cmax.y = (f.y > fmax.y) ? 1 : 0;
        fmax1.y = (f.y > fmax.y) ? fmax.y : f.y;
        fmax.y = (f.y > fmax.y) ? f.y : fmax.y;
        cmax1.z = (f.z > fmax.z) ? 0 : 1;
        cmax.z = (f.z > fmax.z) ? 1 : 0;
        fmax1.z = (f.z > fmax.z) ? fmax.z : f.z;
        fmax.z = (f.z > fmax.z) ? f.z : fmax.z;
        cmax1.w = (f.w > fmax.w) ? 0 : 1;
        cmax.w = (f.w > fmax.w) ? 1 : 0;
        fmax1.w = (f.w > fmax.w) ? fmax.w : f.w;
        fmax.w = (f.w > fmax.w) ? f.w : fmax.w;
        for (uint c = 2; c < i0_dims.z; c++) {
            i0_buf += i0_stride.z; f = *(float4 *)i0_buf;
            cmax1.x = (f.x > fmax.x) ? cmax.x : ((f.x > fmax1.x) ? c : cmax1.x);
            fmax1.x = (f.x > fmax.x) ? fmax.x : ((f.x > fmax1.x) ? f.x : fmax1.x);
            cmax.x  = (f.x > fmax.x) ? c    : cmax.x;
            fmax.x  = (f.x > fmax.x) ? f.x : fmax.x;
            cmax1.y = (f.y > fmax.y) ? cmax.y : ((f.y > fmax1.y) ? c : cmax1.y);
            fmax1.y = (f.y > fmax.y) ? fmax.y : ((f.y > fmax1.y) ? f.y : fmax1.y);
            cmax.y  = (f.y > fmax.y) ? c : cmax.y;
            fmax.y  = (f.y > fmax.y) ? f.y : fmax.y;
            cmax1.z = (f.z > fmax.z) ? cmax.z : ((f.z > fmax1.z) ? c : cmax1.z);
            fmax1.z = (f.z > fmax.z) ? fmax.z : ((f.z > fmax1.z) ? f.z : fmax1.z);
            cmax.z  = (f.z > fmax.z) ? c : cmax.z;
            fmax.z  = (f.z > fmax.z) ? f.z : fmax.z;
            cmax1.w = (f.w > fmax.w) ? cmax.w : ((f.w > fmax1.w) ? c : cmax1.w);
            fmax1.w = (f.w > fmax.w) ? fmax.w : ((f.w > fmax1.w) ? f.w : fmax1.w);
            cmax.w  = (f.w > fmax.w) ? c : cmax.w;
            fmax.w  = (f.w > fmax.w) ? f.w : fmax.w;
        }

        if (isOutputImage) {
            o0_buf += o0_offset + (z * i0_dims.y + y) * o0_image_stride + x * m;
        }
        else {
            o0_buf += o0_offset + z * o0_stride.w + y * o0_stride.y + x * o0_stride.x;
        }

        uint imax = cmax.x + (cmax.y << 8) + (cmax.z << 16) + (cmax.w << 24);
        *(uint *)o0_buf = imax;
        uint imax1 = cmax1.x + (cmax1.y << 8) + (cmax1.z << 16) + (cmax1.w << 24);
        *(uint *)&o0_buf[o0_stride.z] = imax1;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Argmax_topk2_m4_u16_i64_layer(uchar * i0_buf, uint i0_offset, uint4 i0_stride, uint4 i0_dims, uchar *o0_buf, uint o0_offset, uint4 o0_stride, uint factor,
    uint o0_image_stride, uint m, bool isOutputImage) {

    uint x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    if (x < i0_dims.x && y < i0_dims.y) {
        i0_buf += i0_offset + z * i0_stride.w + y * i0_stride.y + x * i0_stride.x;
        uint4 cmax = (uint4)0;
        uint4 cmax1;
        float4 f, fmax, fmax1;
        fmax = *(float4 *)i0_buf;
        i0_buf += i0_stride.z; f = *(float4 *)i0_buf;
        cmax1.x = (f.x > fmax.x) ? 0 : 1;
        cmax.x = (f.x > fmax.x) ? 1 : 0;
        fmax1.x = (f.x > fmax.x) ? fmax.x : f.x;
        fmax.x = (f.x > fmax.x) ? f.x : fmax.x;
        cmax1.y = (f.y > fmax.y) ? 0 : 1;
        cmax.y = (f.y > fmax.y) ? 1 : 0;
        fmax1.y = (f.y > fmax.y) ? fmax.y : f.y;
        fmax.y = (f.y > fmax.y) ? f.y : fmax.y;
        cmax1.z = (f.z > fmax.z) ? 0 : 1;
        cmax.z = (f.z > fmax.z) ? 1 : 0;
        fmax1.z = (f.z > fmax.z) ? fmax.z : f.z;
        fmax.z = (f.z > fmax.z) ? f.z : fmax.z;
        cmax1.w = (f.w > fmax.w) ? 0 : 1;
        cmax.w = (f.w > fmax.w) ? 1 : 0;
        fmax1.w = (f.w > fmax.w) ? fmax.w : f.w;
        fmax.w = (f.w > fmax.w) ? f.w : fmax.w;
        for (uint c = 2; c < i0_dims.z; c++) {
            i0_buf += i0_stride.z; f = *(float4 *)i0_buf;
            cmax1.x = (f.x > fmax.x) ? cmax.x : ((f.x > fmax1.x) ? c : cmax1.x);
            fmax1.x = (f.x > fmax.x) ? fmax.x : ((f.x > fmax1.x) ? f.x : fmax1.x);
            cmax.x  = (f.x > fmax.x) ? c    : cmax.x;
            fmax.x  = (f.x > fmax.x) ? f.x : fmax.x;
            cmax1.y = (f.y > fmax.y) ? cmax.y : ((f.y > fmax1.y) ? c : cmax1.y);
            fmax1.y = (f.y > fmax.y) ? fmax.y : ((f.y > fmax1.y) ? f.y : fmax1.y);
            cmax.y  = (f.y > fmax.y) ? c : cmax.y;
            fmax.y  = (f.y > fmax.y) ? f.y : fmax.y;
            cmax1.z = (f.z > fmax.z) ? cmax.z : ((f.z > fmax1.z) ? c : cmax1.z);
            fmax1.z = (f.z > fmax.z) ? fmax.z : ((f.z > fmax1.z) ? f.z : fmax1.z);
            cmax.z  = (f.z > fmax.z) ? c : cmax.z;
            fmax.z  = (f.z > fmax.z) ? f.z : fmax.z;
            cmax1.w = (f.w > fmax.w) ? cmax.w : ((f.w > fmax1.w) ? c : cmax1.w);
            fmax1.w = (f.w > fmax.w) ? fmax.w : ((f.w > fmax1.w) ? f.w : fmax1.w);
            cmax.w  = (f.w > fmax.w) ? c : cmax.w;
            fmax.w  = (f.w > fmax.w) ? f.w : fmax.w;
        }

        if (isOutputImage) {
            o0_buf += o0_offset + (z * i0_dims.y + y) * o0_image_stride + x * m;
        }
        else {
            o0_buf += o0_offset + z * o0_stride.w + y * o0_stride.y + x * o0_stride.x;
        }

        uint2 imax;
        imax.x = cmax.x + (cmax.y << factor);
        imax.y = cmax.z + (cmax.w << factor);
        *(uint2 *)o0_buf = imax;
        uint2 imax1;
        imax1.x = cmax1.x + (cmax1.y << factor);
        imax1.y = cmax1.z + (cmax1.w << factor);
        *(uint2 *)&o0_buf[o0_stride.z] = imax1;
    }
}

int HipExec_Argmax_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, uchar *i0_buf, uint i0_offset, uint4 i0_stride, uint4 i0_dims, uchar *o0_buf,
    uint o0_offset, uint4 o0_stride, uint o0_image_stride, vx_enum output_data_type, uint top_k, vx_enum output_obj_type) {

    bool input_width_multiple_of_4 = (i0_dims.x & 3) ? false : true;
    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x), ceil((float)globalThreads.y/localThreads.y), ceil((float)globalThreads.z/localThreads.z));

    if (output_data_type == VX_TYPE_UINT8) {
        if (output_obj_type == VX_TYPE_IMAGE) {
            if(input_width_multiple_of_4) {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_m4_u8_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, o0_image_stride, 1, 1);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_m4_u8_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, o0_image_stride, 1, 1);
                }
            } else {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_layer<uchar>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, o0_image_stride, 1, 1);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_layer<uchar>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, o0_image_stride, 1, 1);
                }
            }
        } else {
            if(input_width_multiple_of_4) {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_m4_u8_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 0, 0, 0);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_m4_u8_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 0, 0, 0);
                }
            } else {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_layer<uchar>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 0, 0, 0);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_layer<uchar>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 0, 0, 0);
                }
            }
        }
    } else if (output_data_type == VX_TYPE_UINT16) {
        if (output_obj_type == VX_TYPE_IMAGE) {
            if(input_width_multiple_of_4) {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_m4_u16_i64_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 16, o0_image_stride, 2, 1);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_m4_u16_i64_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 16, o0_image_stride, 2, 1);
                }
            } else {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_layer<short>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, o0_image_stride, 2, 1);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_layer<short>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, o0_image_stride, 2, 1);
                }
            }
        } else {
            if(input_width_multiple_of_4) {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_m4_u16_i64_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 16, 0, 0, 0);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_m4_u16_i64_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 16, 0, 0, 0);
                }
            } else {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_layer<short>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 0, 0, 0);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_layer<short>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 0, 0, 0);
                }
            }
        }
    } else if (output_data_type == VX_TYPE_INT64) {
        if (output_obj_type == VX_TYPE_IMAGE) {
            if(input_width_multiple_of_4) {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_m4_u16_i64_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 64, o0_image_stride, 2, 1);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_m4_u16_i64_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 64, o0_image_stride, 2, 1);
                }
            } else {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_layer<short>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, o0_image_stride, 2, 1);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_layer<short>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, o0_image_stride, 2, 1);
                }
            }
        } else {
            if(input_width_multiple_of_4) {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_m4_u16_i64_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 64, 0, 0, 0);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_m4_u16_i64_layer, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 64, 0, 0, 0);
                }
            } else {
                if(top_k == 2) {
                    hipLaunchKernelGGL(Hip_Argmax_topk2_layer<short>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 0, 0, 0);
                } else {
                    hipLaunchKernelGGL(Hip_Argmax_topk1_layer<short>, gridDim, localThreads, 0, stream, i0_buf, i0_offset, i0_stride, i0_dims,
                        o0_buf, o0_offset, o0_stride, 0, 0, 0);
                }
            }
        }
    } else {
        return VX_ERROR_NOT_SUPPORTED;
    }

    return VX_SUCCESS;

    }

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_tensor_compare_less_layer(uchar* in, uint in_offset, uint4 in_stride, uchar* in2, uint in2_offset, uint4 in2_stride,
 uchar* out, uint out_offset, uint4 out_stride) {

    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    T value = *(T*)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
    T value2 = *(T*)&in2[in2_offset + x * in2_stride.x + y * in2_stride.y + z * in2_stride.z];
    out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
   
    // compare the values and write to the output\n"
    bool result = (value < value2);
   *(int *)&out[0] = result;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_tensor_compare_greater_layer(uchar* in, uint in_offset, uint4 in_stride, uchar* in2, uint in2_offset, uint4 in2_stride,
 uchar* out, uint out_offset, uint4 out_stride) {

    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    T value = *(T*)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
    T value2 = *(T*)&in2[in2_offset + x * in2_stride.x + y * in2_stride.y + z * in2_stride.z];
    out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
   
    // compare the values and write to the output\n"
    bool result = (value < value2);
   *(int *)&out[0] = result;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_tensor_compare_less_than_layer(uchar* in, uint in_offset, uint4 in_stride, uchar* in2, uint in2_offset, uint4 in2_stride,
 uchar* out, uint out_offset, uint4 out_stride) {

    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    T value = *(T*)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
    T value2 = *(T*)&in2[in2_offset + x * in2_stride.x + y * in2_stride.y + z * in2_stride.z];
    out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
   
    // compare the values and write to the output\n"
    bool result = (value <= value2);
   *(int *)&out[0] = result;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_tensor_compare_greater_than_layer(uchar* in, uint in_offset, uint4 in_stride, uchar* in2, uint in2_offset, uint4 in2_stride,
 uchar* out, uint out_offset, uint4 out_stride) {

    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    T value = *(T*)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
    T value2 = *(T*)&in2[in2_offset + x * in2_stride.x + y * in2_stride.y + z * in2_stride.z];
    out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
   
    // compare the values and write to the output\n"
    bool result = (value >= value2);
   *(int *)&out[0] = result;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_tensor_compare_equal_layer(uchar* in, uint in_offset, uint4 in_stride, uchar* in2, uint in2_offset, uint4 in2_stride,
 uchar* out, uint out_offset, uint4 out_stride) {

    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    T value = *(T*)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
    T value2 = *(T*)&in2[in2_offset + x * in2_stride.x + y * in2_stride.y + z * in2_stride.z];
    out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
   
    // compare the values and write to the output\n"
    bool result = (value == value2);
   *(int *)&out[0] = result;
}

template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_tensor_compare_not_equal_layer(uchar* in, uint in_offset, uint4 in_stride, uchar* in2, uint in2_offset, uint4 in2_stride,
 uchar* out, uint out_offset, uint4 out_stride) {

    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    T value = *(T*)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
    T value2 = *(T*)&in2[in2_offset + x * in2_stride.x + y * in2_stride.y + z * in2_stride.z];
    out += out_offset + x * out_stride.x + y * out_stride.y + z * out_stride.z;
   
    // compare the values and write to the output\n"
    bool result = (value != value2);
   *(int *)&out[0] = result;
}

int HipExec_tensor_compare_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, uchar* in,
    uint in_offset, uint4 in_stride, uchar* in2, uint in2_offset, uint4 in2_stride, uchar* out, uint out_offset,
    uint4 out_stride, uint mode) {

    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x),
                        ceil((float)globalThreads.y/localThreads.y),
                        ceil((float)globalThreads.z/localThreads.z));
    if (type == VX_TYPE_FLOAT32) {
        switch (mode) {
            case 0:
                hipLaunchKernelGGL(Hip_tensor_compare_less_layer<float>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
            case 1:
                hipLaunchKernelGGL(Hip_tensor_compare_greater_layer<float>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
            case 2:
                hipLaunchKernelGGL(Hip_tensor_compare_less_than_layer<float>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
            case 3:
                hipLaunchKernelGGL(Hip_tensor_compare_greater_than_layer<float>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
            case 4:
                hipLaunchKernelGGL(Hip_tensor_compare_equal_layer<float>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
            case 5:
                hipLaunchKernelGGL(Hip_tensor_compare_not_equal_layer<float>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
        }
    }
    else if (type == VX_TYPE_FLOAT16) {
        switch (mode) {
            case 0:
                hipLaunchKernelGGL(Hip_tensor_compare_less_layer<__half>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
            case 1:
                hipLaunchKernelGGL(Hip_tensor_compare_greater_layer<__half>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
            case 2:
                hipLaunchKernelGGL(Hip_tensor_compare_less_than_layer<__half>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
            case 3:
                hipLaunchKernelGGL(Hip_tensor_compare_greater_than_layer<__half>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
            case 4:
                hipLaunchKernelGGL(Hip_tensor_compare_equal_layer<__half>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
            case 5:
                hipLaunchKernelGGL(Hip_tensor_compare_not_equal_layer<__half>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
                    in2, in2_offset, in2_stride, out, out_offset, out_stride);
                break;
        }
    }
    return VX_SUCCESS;
}


template <typename T>
__global__ void __attribute__((visibility("default")))
Hip_Upsample_Nearest_layer(uchar* in, uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {

    uint x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    uint z = hipBlockDim_z * hipBlockIdx_z + hipThreadIdx_z;

    T value = *(T*)&in[in_offset + x * in_stride.x + y * in_stride.y + z * in_stride.z];
    out += out_offset + (x << 1) * out_stride.x + (y << 1) * out_stride.y + z * out_stride.z;
    // read 1 value and write 2x2 output
    *(T *)&out[0] = value;
    *(T *)&out[out_stride.x] = value;
    *(T *)&out[out_stride.y] = value;
    *(T *)&out[out_stride.y + out_stride.x] = value;
}

int HipExec_Upsample_Nearest_layer(hipStream_t stream, dim3 globalThreads, dim3 localThreads, vx_enum type, uchar* in,
    uint in_offset, uint4 in_stride, uchar* out, uint out_offset, uint4 out_stride) {

    dim3 gridDim = dim3(ceil((float)globalThreads.x/localThreads.x),
                        ceil((float)globalThreads.y/localThreads.y),
                        ceil((float)globalThreads.z/localThreads.z));

    if (type == VX_TYPE_FLOAT32) {
        hipLaunchKernelGGL(Hip_Upsample_Nearest_layer<float>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
            out, out_offset, out_stride);
    } else {
        hipLaunchKernelGGL(Hip_Upsample_Nearest_layer<__half>, gridDim, localThreads, 0, stream, in, in_offset, in_stride,
            out, out_offset, out_stride);
    }
    return VX_SUCCESS;
}