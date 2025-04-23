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

#ifndef RPP_HIP_LOAD_STORE_HPP
#define RPP_HIP_LOAD_STORE_HPP

#include <hip/hip_runtime.h>

#include "rppdefs.h"
#include "handle.hpp"
#include "rpp_hip_roi_conversion.hpp"

typedef unsigned char uchar;
typedef signed char schar;
typedef struct { uint   data[ 6]; } d_uint6_s;
typedef struct { float  data[ 6]; } d_float6_s;
typedef struct { float  data[ 8]; } d_float8_s;
typedef struct { float  data[24]; } d_float24_s;
typedef struct { half   data[24]; } d_half24_s;
typedef struct { uchar  data[24]; } d_uchar24_s;
typedef struct { schar  data[24]; } d_schar24sc1s_s;
typedef struct { uchar  data[ 8]; } d_uchar8_s;
typedef struct { uint   data[24]; } d_uint24_s;
typedef struct { int    data[24]; } d_int24_s;
typedef struct { uint   data[ 8]; } d_uint8_s;
typedef struct { int    data[ 8]; } d_int8_s;

// float
typedef union { float f1[5];                                                                    }   d_float5;
typedef union { float f1[6];    float2 f2[3];                                                   }   d_float6;
typedef union { float f1[7];                                                                    }   d_float7;
typedef union { float f1[8];    float2 f2[4];   float4 f4[2];                                   }   d_float8;
typedef union { float f1[9];    float3 f3[3];                                                   }   d_float9;
typedef union { float f1[12];   float4 f4[3];                                                   }   d_float12;
typedef union { float f1[16];   float4 f4[4];   d_float8 f8[2];                                 }   d_float16;
typedef union { float f1[24];   float2 f2[12];  float3 f3[8];   float4 f4[6];   d_float8 f8[3]; }   d_float24;

// uint
typedef union { uint ui1[6];    uint2 ui2[3];                                                   }   d_uint6;
typedef union { uint ui1[8];    uint4 ui4[2];                                                   }   d_uint8;
typedef union { uint ui1[24];   uint4 ui4[6];   d_uint8 ui8[3];                                 }   d_uint24;

// int
typedef union { int i1[6];      int2 i2[3];                                                     }   d_int6;
typedef union { int i1[8];      int4 i4[2];                                                     }   d_int8;
typedef union { int i1[24];     int4 i4[6];     d_int8 i8[3];                                   }   d_int24;

// half
typedef struct { half h1[3];                                                                    }   d_half3_s;
typedef struct { half2 h2[3];                                                                   }   d_half6_s;
typedef union { half h1[8];     half2 h2[4];                                                    }   d_half8;
typedef union { half h1[12];    half2 h2[6];    d_half3_s h3[4];                                }   d_half12;
typedef union { half h1[24];    half2 h2[12];   d_half3_s h3[8];  d_half8 h8[3];                }   d_half24;

// uchar
typedef union { uchar uc1[8];   uchar4 uc4[2];                                                  }   d_uchar8;
typedef union { uchar uc1[24];  uchar4 uc4[6];  uchar3 uc3[8];    d_uchar8 uc8[3];              }   d_uchar24;

// schar
typedef struct { schar sc1[3];                                                                  }   d_schar3_s;
typedef struct { schar sc1[8];                                                                  }   d_schar8_s;
typedef struct { d_schar8_s sc8[3];                                                             }   d_schar24_s;

enum class RPPTensorDataType
{
    U8 = 0,
    FP32,
    FP16,
    I8,
};

struct RPPTensorFunctionMetaData
{
    RPPTensorDataType _in_type = RPPTensorDataType::U8;
    RPPTensorDataType _out_type = RPPTensorDataType::U8;
    RppiChnFormat _in_format = RppiChnFormat::RPPI_CHN_PACKED;
    RppiChnFormat _out_format = RppiChnFormat::RPPI_CHN_PLANAR;
    Rpp32u _in_channels = 3;

    RPPTensorFunctionMetaData(RppiChnFormat in_chn_format, RPPTensorDataType in_tensor_type,
                              RPPTensorDataType out_tensor_type, Rpp32u in_channels,
                              bool out_format_change) : _in_format(in_chn_format), _in_type(in_tensor_type),
                                                        _out_type(out_tensor_type), _in_channels(in_channels)
    {
        if (out_format_change)
        {
            if (_in_format == RPPI_CHN_PLANAR)
                _out_format = RppiChnFormat::RPPI_CHN_PACKED;
            else
                _out_format = RppiChnFormat::RPPI_CHN_PLANAR;
        }
        else
            _out_format = _in_format;
    }
};

#define LOCAL_THREADS_X                 16                  // default rpp hip thread launch config - local threads x = 16
#define LOCAL_THREADS_Y                 16                  // default rpp hip thread launch config - local threads y = 16
#define LOCAL_THREADS_Z                 1                   // default rpp hip thread launch config - local threads z = 1
#define LOCAL_THREADS_X_1DIM            256                 // single dimension kernels rpp thread launch config - local threads x = 256
#define LOCAL_THREADS_Y_1DIM            1                   // single dimension kernels rpp thread launch config - local threads x = 1
#define LOCAL_THREADS_Z_1DIM            1                   // single dimension kernels rpp thread launch config - local threads x = 1
#define ONE_OVER_255                    0.00392156862745f
#define ONE_OVER_256                    0.00390625f
#define SIX_OVER_360                    0.01666667f
#define PI                              3.14159265
#define TWO_PI                          6.2831853
#define RGB_TO_GREY_WEIGHT_RED          0.299f
#define RGB_TO_GREY_WEIGHT_GREEN        0.587f
#define RGB_TO_GREY_WEIGHT_BLUE         0.114f
#define NEWTON_METHOD_INITIAL_GUESS     0x5f3759df          // Initial guess for Newton Raphson Inverse Square Root
#define RPP_255_OVER_1PT57              162.3380757272f     // (255 / 1.570796) - multiplier used in phase computation
#define ONE_OVER_1PT57                  0.6366199048f       // (1 / 1.570796) i.e. 2/pi - multiplier used in phase computation
#define SMEM_LENGTH_X                   128                 // Shared memory length of 128 cols to efficiently utilize all 16 LOCAL_THREADS_X as 16 * 8-byte vectorized global read/writes per thread = 128 bytes, fitting in 32 banks 4 byte wide
#define SMEM_LENGTH_Y_1C                16                  // Shared memory length of 16 rows to efficiently utilize all 16 LOCAL_THREADS_Y as 1 128-byte-long row per thread (single channel greyscale)
#define SMEM_LENGTH_Y_3C                48                  // Shared memory length of 48 rows to efficiently utilize all 16 LOCAL_THREADS_Y as 3 128-byte-long rows per thread (three channel rgb)

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0')

// float4 floor

#define FLOOR4(src, dst) \
dst = make_int4(floorf(src.x), floorf(src.y), floorf(src.z), floorf(src.w));

/******************** HOST FUNCTIONS ********************/

inline int getplnpkdind(RppiChnFormat &format)
{
    return format == RPPI_CHN_PLANAR ? 1 : 3;
}

inline void generate_gaussian_kernel_gpu(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSize)
{
    Rpp32f s, sum = 0.0, multiplier;
    int bound = ((kernelSize - 1) / 2);
    Rpp32u c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -bound; i <= bound; i++)
    {
        for (int j = -bound; j <= bound; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        kernel[i] /= sum;
    }
}

// Retrieve Min and Max given a datatype

inline void getImageBitDepthMinMax(uchar *srcPtr, float2 *bitDepthMinMax_f2) { *bitDepthMinMax_f2 = make_float2(0, 255); }
inline void getImageBitDepthMinMax(float *srcPtr, float2 *bitDepthMinMax_f2) { *bitDepthMinMax_f2 = make_float2(0, 255); }
inline void getImageBitDepthMinMax(half *srcPtr, float2 *bitDepthMinMax_f2) { *bitDepthMinMax_f2 = make_float2(0, 255); }
inline void getImageBitDepthMinMax(schar *srcPtr, float2 *bitDepthMinMax_f2) { *bitDepthMinMax_f2 = make_float2(-128, 127); }

/******************** DEVICE FUNCTIONS ********************/

// -------------------- Set 0 - Range checks and Range adjustment --------------------

// float pixel check for 0-255 range

__device__ __forceinline__ void rpp_hip_pixel_check_and_store(float pixel, uchar* dst)
{
    pixel = fmax(fminf(pixel, 255), 0);
    *dst = (uchar)pixel;
}

// float pixel check for -128-127 range

__device__ __forceinline__ void rpp_hip_pixel_check_and_store(float pixel, schar* dst)
{
    pixel = fmax(fminf(pixel, 127), -128);
    *dst = (schar)pixel;
}

// float pixel check for 0-1 range

__device__ __forceinline__ void rpp_hip_pixel_check_and_store(float pixel, float* dst)
{
    pixel = fmax(fminf(pixel, 1), 0);
    *dst = pixel;
}

__device__ __forceinline__ void rpp_hip_pixel_check_and_store(float pixel, half* dst)
{
    pixel = fmax(fminf(pixel, 1), 0);
    *dst = (half)pixel;
}

__device__ __forceinline__ float rpp_hip_pixel_check_0to255(float src_f1)
{
    return fminf(fmaxf(src_f1, 0), 255);
}

__device__ __forceinline__ float4 rpp_hip_pixel_check_0to255(float4 src_f4)
{
    return make_float4(fminf(fmaxf(src_f4.x, 0.0f), 255.0f),
                       fminf(fmaxf(src_f4.y, 0.0f), 255.0f),
                       fminf(fmaxf(src_f4.z, 0.0f), 255.0f),
                       fminf(fmaxf(src_f4.w, 0.0f), 255.0f));
}

// float4 pixel check for 0-1 range

__device__ __forceinline__ float4 rpp_hip_pixel_check_0to1(float4 src_f4)
{
    return make_float4(fminf(fmaxf(src_f4.x, 0), 1),
                       fminf(fmaxf(src_f4.y, 0), 1),
                       fminf(fmaxf(src_f4.z, 0), 1),
                       fminf(fmaxf(src_f4.w, 0), 1));
}

__device__ __forceinline__ float rpp_hip_pixel_check_0to1(float src_f1)
{
    return fminf(fmaxf(src_f1, 0), 1);
}
// d_float8 pixel check for 0-255 range

__device__ __forceinline__ void rpp_hip_pixel_check_0to255(d_float8 *pix_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to255(pix_f8->f4[0]);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to255(pix_f8->f4[1]);
}

// d_float8 pixel check for 0-1 range

__device__ __forceinline__ void rpp_hip_pixel_check_0to1(d_float8 *pix_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to1(pix_f8->f4[0]);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to1(pix_f8->f4[1]);
}

// d_float24 pixel check for 0-255 range

__device__ __forceinline__ void rpp_hip_pixel_check_0to255(d_float24 *pix_f24)
{
    pix_f24->f4[0] = rpp_hip_pixel_check_0to255(pix_f24->f4[0]);
    pix_f24->f4[1] = rpp_hip_pixel_check_0to255(pix_f24->f4[1]);
    pix_f24->f4[2] = rpp_hip_pixel_check_0to255(pix_f24->f4[2]);
    pix_f24->f4[3] = rpp_hip_pixel_check_0to255(pix_f24->f4[3]);
    pix_f24->f4[4] = rpp_hip_pixel_check_0to255(pix_f24->f4[4]);
    pix_f24->f4[5] = rpp_hip_pixel_check_0to255(pix_f24->f4[5]);
}

// d_float24 pixel check for 0-1 range

__device__ __forceinline__ void rpp_hip_pixel_check_0to1(d_float24 *pix_f24)
{
    pix_f24->f4[0] = rpp_hip_pixel_check_0to1(pix_f24->f4[0]);
    pix_f24->f4[1] = rpp_hip_pixel_check_0to1(pix_f24->f4[1]);
    pix_f24->f4[2] = rpp_hip_pixel_check_0to1(pix_f24->f4[2]);
    pix_f24->f4[3] = rpp_hip_pixel_check_0to1(pix_f24->f4[3]);
    pix_f24->f4[4] = rpp_hip_pixel_check_0to1(pix_f24->f4[4]);
    pix_f24->f4[5] = rpp_hip_pixel_check_0to1(pix_f24->f4[5]);
}

// d_float8 adjust pixel range for different bit depths

__device__ __forceinline__ void rpp_hip_adjust_range(uchar *dstPtr, d_float8 *sum_f8){}

__device__ __forceinline__ void rpp_hip_adjust_range(float *dstPtr, d_float8 *sum_f8)
{
    sum_f8->f4[0] = sum_f8->f4[0] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f8->f4[1] = sum_f8->f4[1] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
}

__device__ __forceinline__ void rpp_hip_adjust_range(schar *dstPtr, d_float8 *sum_f8)
{
    sum_f8->f4[0] = sum_f8->f4[0] - (float4) 128;    // Subtract 128 for schar image data
    sum_f8->f4[1] = sum_f8->f4[1] - (float4) 128;    // Subtract 128 for schar image data
}

__device__ __forceinline__ void rpp_hip_adjust_range(half *dstPtr, d_float8 *sum_f8)
{
    sum_f8->f4[0] = sum_f8->f4[0] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f8->f4[1] = sum_f8->f4[1] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
}

// d_float24 adjust pixel range for different bit depths

__device__ __forceinline__ void rpp_hip_adjust_range(uchar *dstPtr, d_float24 *sum_f24){}

__device__ __forceinline__ void rpp_hip_adjust_range(float *dstPtr, d_float24 *sum_f24)
{
    sum_f24->f4[0] = sum_f24->f4[0] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f24->f4[1] = sum_f24->f4[1] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f24->f4[2] = sum_f24->f4[2] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f24->f4[3] = sum_f24->f4[3] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f24->f4[4] = sum_f24->f4[4] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f24->f4[5] = sum_f24->f4[5] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
}

__device__ __forceinline__ void rpp_hip_adjust_range(schar *dstPtr, d_float24 *sum_f24)
{
    sum_f24->f4[0] = sum_f24->f4[0] - (float4) 128;    // Subtract 128 for schar image data
    sum_f24->f4[1] = sum_f24->f4[1] - (float4) 128;    // Subtract 128 for schar image data
    sum_f24->f4[2] = sum_f24->f4[2] - (float4) 128;    // Subtract 128 for schar image data
    sum_f24->f4[3] = sum_f24->f4[3] - (float4) 128;    // Subtract 128 for schar image data
    sum_f24->f4[4] = sum_f24->f4[4] - (float4) 128;    // Subtract 128 for schar image data
    sum_f24->f4[5] = sum_f24->f4[5] - (float4) 128;    // Subtract 128 for schar image data
}

__device__ __forceinline__ void rpp_hip_adjust_range(half *dstPtr, d_float24 *sum_f24)
{
    sum_f24->f4[0] = sum_f24->f4[0] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f24->f4[1] = sum_f24->f4[1] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f24->f4[2] = sum_f24->f4[2] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f24->f4[3] = sum_f24->f4[3] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f24->f4[4] = sum_f24->f4[4] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f24->f4[5] = sum_f24->f4[5] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
}

// -------------------- Set 1 - Packing --------------------

// Packing to U8s

__device__ __forceinline__ uint rpp_hip_pack(float4 src)
{
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

// Packing to I8s

__device__ __forceinline__ uint rpp_hip_pack_i8(float4 src)
{
    char4 dst_c4;
    dst_c4.w = (schar)(src.w);
    dst_c4.z = (schar)(src.z);
    dst_c4.y = (schar)(src.y);
    dst_c4.x = (schar)(src.x);

    return *(uint *)&dst_c4;
}

// Packing to Uints

__device__ __forceinline__ uint4 rpp_hip_pack_uint4(uchar4 src)
{
    uint4 dst_ui4;
    dst_ui4.w = (uint)(src.w);
    dst_ui4.z = (uint)(src.z);
    dst_ui4.y = (uint)(src.y);
    dst_ui4.x = (uint)(src.x);

    return *(uint4 *)&dst_ui4;
}

// Packing to Ints

__device__ __forceinline__ void rpp_hip_pack_int8(d_schar8_s *src_sc8, d_int8 *srcPtr_i8)
{
    srcPtr_i8->i1[0] = int(src_sc8->sc1[0]);
    srcPtr_i8->i1[1] = int(src_sc8->sc1[1]);
    srcPtr_i8->i1[2] = int(src_sc8->sc1[2]);
    srcPtr_i8->i1[3] = int(src_sc8->sc1[3]);
    srcPtr_i8->i1[4] = int(src_sc8->sc1[4]);
    srcPtr_i8->i1[5] = int(src_sc8->sc1[5]);
    srcPtr_i8->i1[6] = int(src_sc8->sc1[6]);
    srcPtr_i8->i1[7] = int(src_sc8->sc1[7]);
}

// -------------------- Set 2 - Un-Packing --------------------

// Un-Packing from U8s

__device__ __forceinline__ float rpp_hip_unpack0(uint src)
{
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack1(uint src)
{
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack2(uint src)
{
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack3(uint src)
{
    return (float)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 rpp_hip_unpack(uint src)
{
    return make_float4(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src), rpp_hip_unpack3(src));
}

__device__ __forceinline__ float4 rpp_hip_unpack_mirror(uint src)
{
    return make_float4(rpp_hip_unpack3(src), rpp_hip_unpack2(src), rpp_hip_unpack1(src), rpp_hip_unpack0(src));
}

// Un-Packing from I8s

__device__ __forceinline__ float rpp_hip_unpack0(int src)
{
    return (float)(schar)(src & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack1(int src)
{
    return (float)(schar)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack2(int src)
{
    return (float)(schar)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack3(int src)
{
    return (float)(schar)((src >> 24) & 0xFF);
}

// Un-Packing from I16s

__device__ __forceinline__ float rpp_hip_unpack0_(int src)
{
    return (float)((short)(src & 0xFFFF)); 
}

__device__ __forceinline__ float rpp_hip_unpack2_(int src)
{
    return (float)((short)((src >> 16) & 0xFFFF)); 
}

__device__ __forceinline__ float4 rpp_hip_unpack_from_i8(int src)
{
    return make_float4(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src), rpp_hip_unpack3(src));
}

__device__ __forceinline__ float4 rpp_hip_unpack_from_i8_mirror(int src)
{
    return make_float4(rpp_hip_unpack3(src), rpp_hip_unpack2(src), rpp_hip_unpack1(src), rpp_hip_unpack0(src));
}

// Un-Packing from F32s

__device__ __forceinline__ float4 rpp_hip_unpack_mirror(float4 src)
{
    return make_float4(src.w, src.z, src.y, src.x);
}

// -------------------- Set 3 - Bit Depth Conversions --------------------

// I8 to U8 conversions (8 pixels)

__device__ __forceinline__ void rpp_hip_convert8_i8_to_u8(schar *srcPtr, uchar *dstPtr)
{
    int2 *srcPtr_i2;
    srcPtr_i2 = (int2 *)srcPtr;

    uint2 *dstPtr_ui2;
    dstPtr_ui2 = (uint2 *)dstPtr;

    dstPtr_ui2->x = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_i2->x) + (float4) 128);
    dstPtr_ui2->y = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_i2->y) + (float4) 128);
}

// I8 to U8 conversions (24 pixels)

__device__ __forceinline__ void rpp_hip_convert24_i8_to_u8(schar *srcPtr, uchar *dstPtr)
{
    d_int6 *srcPtr_i6;
    srcPtr_i6 = (d_int6 *)srcPtr;

    d_uint6 *dstPtr_ui6;
    dstPtr_ui6 = (d_uint6 *)dstPtr;

    dstPtr_ui6->ui1[0] = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_i6->i1[0]) + (float4) 128);
    dstPtr_ui6->ui1[1] = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_i6->i1[1]) + (float4) 128);
    dstPtr_ui6->ui1[2] = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_i6->i1[2]) + (float4) 128);
    dstPtr_ui6->ui1[3] = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_i6->i1[3]) + (float4) 128);
    dstPtr_ui6->ui1[4] = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_i6->i1[4]) + (float4) 128);
    dstPtr_ui6->ui1[5] = rpp_hip_pack(rpp_hip_unpack_from_i8(srcPtr_i6->i1[5]) + (float4) 128);
}

// -------------------- Set 4 - Loads to float --------------------

// WITHOUT LAYOUT TOGGLE

// U8 loads without layout toggle (8 U8 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(uchar *srcPtr, d_float8 *srcPtr_f8)
{
    uint2 src_ui2 = *(uint2 *)srcPtr;
    srcPtr_f8->f4[0] = rpp_hip_unpack(src_ui2.x);    // write 00-03
    srcPtr_f8->f4[1] = rpp_hip_unpack(src_ui2.y);    // write 04-07
}

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8_mirror(uchar *srcPtr, d_float8 *srcPtr_f8)
{
    uint2 src_ui2 = *(uint2 *)srcPtr;
    srcPtr_f8->f4[0] = rpp_hip_unpack_mirror(src_ui2.y);    // write 07-04
    srcPtr_f8->f4[1] = rpp_hip_unpack_mirror(src_ui2.x);    // write 03-00
}

// F32 loads without layout toggle (8 F32 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(float *srcPtr, d_float8 *srcPtr_f8)
{
    *(d_float8_s *)srcPtr_f8 = *(d_float8_s *)srcPtr;    // write 00-07
}

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8_mirror(float *srcPtr, d_float8 *srcPtr_f8)
{
    d_float8 src_f8;
    *(d_float8_s *)&src_f8 = *(d_float8_s *)srcPtr;
    srcPtr_f8->f4[0] = rpp_hip_unpack_mirror(src_f8.f4[1]);    // write 07-04
    srcPtr_f8->f4[1] = rpp_hip_unpack_mirror(src_f8.f4[0]);    // write 03-00
}

// I8 loads without layout toggle (8 I8 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(schar *srcPtr, d_float8 *srcPtr_f8)
{
    int2 src_i2 = *(int2 *)srcPtr;
    srcPtr_f8->f4[0] = rpp_hip_unpack_from_i8(src_i2.x);    // write 00-03
    srcPtr_f8->f4[1] = rpp_hip_unpack_from_i8(src_i2.y);    // write 04-07
}

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8_mirror(schar *srcPtr, d_float8 *srcPtr_f8)
{
    int2 src_i2 = *(int2 *)srcPtr;
    srcPtr_f8->f4[0] = rpp_hip_unpack_from_i8_mirror(src_i2.y);    // write 07-04
    srcPtr_f8->f4[1] = rpp_hip_unpack_from_i8_mirror(src_i2.x);    // write 03-00
}

// F16 loads without layout toggle (8 F16 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(half *srcPtr, d_float8 *srcPtr_f8)
{
    d_half8 src_h8;
    src_h8 = *(d_half8 *)srcPtr;

    float2 src1_f2, src2_f2;

    src1_f2 = __half22float2(src_h8.h2[0]);
    src2_f2 = __half22float2(src_h8.h2[1]);
    srcPtr_f8->f4[0] = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write 00-03

    src1_f2 = __half22float2(src_h8.h2[2]);
    src2_f2 = __half22float2(src_h8.h2[3]);
    srcPtr_f8->f4[1] = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write 04-07
}

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8_mirror(half *srcPtr, d_float8 *srcPtr_f8)
{
    d_half8 src_h8;
    src_h8 = *(d_half8 *)srcPtr;

    float2 src1_f2, src2_f2;

    src1_f2 = __half22float2(src_h8.h2[3]);
    src2_f2 = __half22float2(src_h8.h2[2]);
    srcPtr_f8->f4[0] = make_float4(src1_f2.y, src1_f2.x, src2_f2.y, src2_f2.x);    // write 07-04

    src1_f2 = __half22float2(src_h8.h2[1]);
    src2_f2 = __half22float2(src_h8.h2[0]);
    srcPtr_f8->f4[1] = make_float4(src1_f2.y, src1_f2.x, src2_f2.y, src2_f2.x);    // write 03-00
}

// I16 loads without layout toggle (8 I16 pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(short *srcPtr, d_float8 *srcPtr_f8)
{
    int4 src_i4 = *(int4 *)srcPtr;
    srcPtr_f8->f4[0] = make_float4(rpp_hip_unpack0_(src_i4.x), rpp_hip_unpack2_(src_i4.x), rpp_hip_unpack0_(src_i4.y), rpp_hip_unpack2_(src_i4.y));
    srcPtr_f8->f4[1] = make_float4(rpp_hip_unpack0_(src_i4.z), rpp_hip_unpack2_(src_i4.z), rpp_hip_unpack0_(src_i4.w), rpp_hip_unpack2_(src_i4.w));
}

// UINT loads without layout toggle (8 UINT pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(uint *srcPtr, d_float8 *srcPtr_f8)
{
    d_uint8 src_ui8 = *(d_uint8 *)srcPtr;
    srcPtr_f8->f4[0] = make_float4((float)src_ui8.ui4[0].x, (float)src_ui8.ui4[0].y, (float)src_ui8.ui4[0].z, (float)src_ui8.ui4[0].w);
    srcPtr_f8->f4[1] = make_float4((float)src_ui8.ui4[1].x, (float)src_ui8.ui4[1].y, (float)src_ui8.ui4[1].z, (float)src_ui8.ui4[1].w);
}

// INT loads without layout toggle (8 INT pixels)

__device__ __forceinline__ void rpp_hip_load8_and_unpack_to_float8(int *srcPtr, d_float8 *srcPtr_f8)
{
    d_int8 src_i8 = *(d_int8 *)srcPtr;
    srcPtr_f8->f4[0] = make_float4((float)src_i8.i4[0].x, (float)src_i8.i4[0].y, (float)src_i8.i4[0].z, (float)src_i8.i4[0].w);
    srcPtr_f8->f4[1] = make_float4((float)src_i8.i4[1].x, (float)src_i8.i4[1].y, (float)src_i8.i4[1].z, (float)src_i8.i4[1].w);
}

// U8 loads without layout toggle PLN3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(uchar *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    d_uint6 src_ui6;

    src_ui6.ui2[0] = *(uint2 *)srcPtr;
    srcPtr += increment;
    src_ui6.ui2[1] = *(uint2 *)srcPtr;
    srcPtr += increment;
    src_ui6.ui2[2] = *(uint2 *)srcPtr;

    srcPtr_f24->f4[0] = rpp_hip_unpack(src_ui6.ui1[0]);    // write R00-R03
    srcPtr_f24->f4[1] = rpp_hip_unpack(src_ui6.ui1[1]);    // write R04-R07
    srcPtr_f24->f4[2] = rpp_hip_unpack(src_ui6.ui1[2]);    // write G00-G03
    srcPtr_f24->f4[3] = rpp_hip_unpack(src_ui6.ui1[3]);    // write G04-G07
    srcPtr_f24->f4[4] = rpp_hip_unpack(src_ui6.ui1[4]);    // write B00-B03
    srcPtr_f24->f4[5] = rpp_hip_unpack(src_ui6.ui1[5]);    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(uchar *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    d_uint6 src_ui6;

    src_ui6.ui2[0] = *(uint2 *)srcPtr;
    srcPtr += increment;
    src_ui6.ui2[1] = *(uint2 *)srcPtr;
    srcPtr += increment;
    src_ui6.ui2[2] = *(uint2 *)srcPtr;

    srcPtr_f24->f4[0] = rpp_hip_unpack_mirror(src_ui6.ui1[1]);    // write R07-R04 (mirrored load)
    srcPtr_f24->f4[1] = rpp_hip_unpack_mirror(src_ui6.ui1[0]);    // write R03-R00 (mirrored load)
    srcPtr_f24->f4[2] = rpp_hip_unpack_mirror(src_ui6.ui1[3]);    // write G07-G04 (mirrored load)
    srcPtr_f24->f4[3] = rpp_hip_unpack_mirror(src_ui6.ui1[2]);    // write G03-G00 (mirrored load)
    srcPtr_f24->f4[4] = rpp_hip_unpack_mirror(src_ui6.ui1[5]);    // write B07-B04 (mirrored load)
    srcPtr_f24->f4[5] = rpp_hip_unpack_mirror(src_ui6.ui1[4]);    // write B03-B00 (mirrored load)
}

// F32 loads without layout toggle PLN3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(float *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    *(d_float8_s *)&srcPtr_f24->f8[0] = *(d_float8_s *)srcPtr;    // write R00-R07
    srcPtr += increment;
    *(d_float8_s *)&srcPtr_f24->f8[1] = *(d_float8_s *)srcPtr;    // write G00-G07
    srcPtr += increment;
    *(d_float8_s *)&srcPtr_f24->f8[2] = *(d_float8_s *)srcPtr;    // write B00-B07
}

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(float *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    d_float24 src_f24;

    *(d_float8_s *)&src_f24.f8[0] = *(d_float8_s *)srcPtr;    // write R00-R07
    srcPtr += increment;
    *(d_float8_s *)&src_f24.f8[1] = *(d_float8_s *)srcPtr;    // write G00-G07
    srcPtr += increment;
    *(d_float8_s *)&src_f24.f8[2] = *(d_float8_s *)srcPtr;    // write B00-B07

    srcPtr_f24->f4[0] = rpp_hip_unpack_mirror(src_f24.f4[1]);    // write R07-R04 (mirrored load)
    srcPtr_f24->f4[1] = rpp_hip_unpack_mirror(src_f24.f4[0]);    // write R03-R00 (mirrored load)
    srcPtr_f24->f4[2] = rpp_hip_unpack_mirror(src_f24.f4[3]);    // write G07-G04 (mirrored load)
    srcPtr_f24->f4[3] = rpp_hip_unpack_mirror(src_f24.f4[2]);    // write G03-G00 (mirrored load)
    srcPtr_f24->f4[4] = rpp_hip_unpack_mirror(src_f24.f4[5]);    // write B07-B04 (mirrored load)
    srcPtr_f24->f4[5] = rpp_hip_unpack_mirror(src_f24.f4[4]);    // write B03-B00 (mirrored load)
}

// I8 loads without layout toggle PLN3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(schar *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    d_int6 src_i6;

    src_i6.i2[0] = *(int2 *)srcPtr;
    srcPtr += increment;
    src_i6.i2[1] = *(int2 *)srcPtr;
    srcPtr += increment;
    src_i6.i2[2] = *(int2 *)srcPtr;

    srcPtr_f24->f4[0] = rpp_hip_unpack_from_i8(src_i6.i1[0]);    // write R00-R03
    srcPtr_f24->f4[1] = rpp_hip_unpack_from_i8(src_i6.i1[1]);    // write R04-R07
    srcPtr_f24->f4[2] = rpp_hip_unpack_from_i8(src_i6.i1[2]);    // write G00-G03
    srcPtr_f24->f4[3] = rpp_hip_unpack_from_i8(src_i6.i1[3]);    // write G04-G07
    srcPtr_f24->f4[4] = rpp_hip_unpack_from_i8(src_i6.i1[4]);    // write B00-B03
    srcPtr_f24->f4[5] = rpp_hip_unpack_from_i8(src_i6.i1[5]);    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(schar *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    d_int6 src_i6;

    src_i6.i2[0] = *(int2 *)srcPtr;
    srcPtr += increment;
    src_i6.i2[1] = *(int2 *)srcPtr;
    srcPtr += increment;
    src_i6.i2[2] = *(int2 *)srcPtr;

    srcPtr_f24->f4[0] = rpp_hip_unpack_from_i8_mirror(src_i6.i1[1]);    // write R07-R04 (mirrored load)
    srcPtr_f24->f4[1] = rpp_hip_unpack_from_i8_mirror(src_i6.i1[0]);    // write R03-R00 (mirrored load)
    srcPtr_f24->f4[2] = rpp_hip_unpack_from_i8_mirror(src_i6.i1[3]);    // write G07-G04 (mirrored load)
    srcPtr_f24->f4[3] = rpp_hip_unpack_from_i8_mirror(src_i6.i1[2]);    // write G03-G00 (mirrored load)
    srcPtr_f24->f4[4] = rpp_hip_unpack_from_i8_mirror(src_i6.i1[5]);    // write B07-B04 (mirrored load)
    srcPtr_f24->f4[5] = rpp_hip_unpack_from_i8_mirror(src_i6.i1[4]);    // write B03-B00 (mirrored load)
}

// F16 loads without layout toggle PLN3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3(half *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    d_half24 src_h24;

    src_h24.h8[0] = *(d_half8 *)srcPtr;
    srcPtr += increment;
    src_h24.h8[1] = *(d_half8 *)srcPtr;
    srcPtr += increment;
    src_h24.h8[2] = *(d_half8 *)srcPtr;

    srcPtr_f24->f2[ 0] = __half22float2(src_h24.h2[ 0]);    // write R00R01
    srcPtr_f24->f2[ 1] = __half22float2(src_h24.h2[ 1]);    // write R02R03
    srcPtr_f24->f2[ 2] = __half22float2(src_h24.h2[ 2]);    // write R04R05
    srcPtr_f24->f2[ 3] = __half22float2(src_h24.h2[ 3]);    // write R06R07
    srcPtr_f24->f2[ 4] = __half22float2(src_h24.h2[ 4]);    // write G00G01
    srcPtr_f24->f2[ 5] = __half22float2(src_h24.h2[ 5]);    // write G02G03
    srcPtr_f24->f2[ 6] = __half22float2(src_h24.h2[ 6]);    // write G04G05
    srcPtr_f24->f2[ 7] = __half22float2(src_h24.h2[ 7]);    // write G06G07
    srcPtr_f24->f2[ 8] = __half22float2(src_h24.h2[ 8]);    // write B00B01
    srcPtr_f24->f2[ 9] = __half22float2(src_h24.h2[ 9]);    // write B02B03
    srcPtr_f24->f2[10] = __half22float2(src_h24.h2[10]);    // write B04B05
    srcPtr_f24->f2[11] = __half22float2(src_h24.h2[11]);    // write B06B07
}

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(half *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    d_half24 src_h24;

    src_h24.h8[0] = *(d_half8 *)srcPtr;
    srcPtr += increment;
    src_h24.h8[1] = *(d_half8 *)srcPtr;
    srcPtr += increment;
    src_h24.h8[2] = *(d_half8 *)srcPtr;

    srcPtr_f24->f2[ 0] = __half22float2(__halves2half2(src_h24.h1[ 7], src_h24.h1[ 6]));    // write R07R06
    srcPtr_f24->f2[ 1] = __half22float2(__halves2half2(src_h24.h1[ 5], src_h24.h1[ 4]));    // write R05R04
    srcPtr_f24->f2[ 2] = __half22float2(__halves2half2(src_h24.h1[ 3], src_h24.h1[ 2]));    // write R03R02
    srcPtr_f24->f2[ 3] = __half22float2(__halves2half2(src_h24.h1[ 1], src_h24.h1[ 0]));    // write R01R00
    srcPtr_f24->f2[ 4] = __half22float2(__halves2half2(src_h24.h1[15], src_h24.h1[14]));    // write G07G06
    srcPtr_f24->f2[ 5] = __half22float2(__halves2half2(src_h24.h1[13], src_h24.h1[12]));    // write G05G04
    srcPtr_f24->f2[ 6] = __half22float2(__halves2half2(src_h24.h1[11], src_h24.h1[10]));    // write G03G02
    srcPtr_f24->f2[ 7] = __half22float2(__halves2half2(src_h24.h1[ 9], src_h24.h1[ 8]));    // write G01G00
    srcPtr_f24->f2[ 8] = __half22float2(__halves2half2(src_h24.h1[23], src_h24.h1[22]));    // write B07B06
    srcPtr_f24->f2[ 9] = __half22float2(__halves2half2(src_h24.h1[21], src_h24.h1[20]));    // write B05B04
    srcPtr_f24->f2[10] = __half22float2(__halves2half2(src_h24.h1[19], src_h24.h1[18]));    // write B03B02
    srcPtr_f24->f2[11] = __half22float2(__halves2half2(src_h24.h1[17], src_h24.h1[16]));    // write B01B00
}

// WITH LAYOUT TOGGLE

// U8 loads with layout toggle PKD3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(uchar *srcPtr, d_float24 *srcPtr_f24)
{
    d_uint6 src_ui6 = *(d_uint6 *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(rpp_hip_unpack0(src_ui6.ui1[0]), rpp_hip_unpack3(src_ui6.ui1[0]), rpp_hip_unpack2(src_ui6.ui1[1]), rpp_hip_unpack1(src_ui6.ui1[2]));    // write R00-R03
    srcPtr_f24->f4[1] = make_float4(rpp_hip_unpack0(src_ui6.ui1[3]), rpp_hip_unpack3(src_ui6.ui1[3]), rpp_hip_unpack2(src_ui6.ui1[4]), rpp_hip_unpack1(src_ui6.ui1[5]));    // write R04-R07
    srcPtr_f24->f4[2] = make_float4(rpp_hip_unpack1(src_ui6.ui1[0]), rpp_hip_unpack0(src_ui6.ui1[1]), rpp_hip_unpack3(src_ui6.ui1[1]), rpp_hip_unpack2(src_ui6.ui1[2]));    // write G00-G03
    srcPtr_f24->f4[3] = make_float4(rpp_hip_unpack1(src_ui6.ui1[3]), rpp_hip_unpack0(src_ui6.ui1[4]), rpp_hip_unpack3(src_ui6.ui1[4]), rpp_hip_unpack2(src_ui6.ui1[5]));    // write G04-G07
    srcPtr_f24->f4[4] = make_float4(rpp_hip_unpack2(src_ui6.ui1[0]), rpp_hip_unpack1(src_ui6.ui1[1]), rpp_hip_unpack0(src_ui6.ui1[2]), rpp_hip_unpack3(src_ui6.ui1[2]));    // write B00-B03
    srcPtr_f24->f4[5] = make_float4(rpp_hip_unpack2(src_ui6.ui1[3]), rpp_hip_unpack1(src_ui6.ui1[4]), rpp_hip_unpack0(src_ui6.ui1[5]), rpp_hip_unpack3(src_ui6.ui1[5]));    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(uchar *srcPtr, d_float24 *srcPtr_f24)
{
    d_uint6 src_ui6 = *(d_uint6 *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(rpp_hip_unpack1(src_ui6.ui1[5]), rpp_hip_unpack2(src_ui6.ui1[4]), rpp_hip_unpack3(src_ui6.ui1[3]), rpp_hip_unpack0(src_ui6.ui1[3]));    // write R07-R04 (mirrored load)
    srcPtr_f24->f4[1] = make_float4(rpp_hip_unpack1(src_ui6.ui1[2]), rpp_hip_unpack2(src_ui6.ui1[1]), rpp_hip_unpack3(src_ui6.ui1[0]), rpp_hip_unpack0(src_ui6.ui1[0]));    // write R03-R00 (mirrored load)
    srcPtr_f24->f4[2] = make_float4(rpp_hip_unpack2(src_ui6.ui1[5]), rpp_hip_unpack3(src_ui6.ui1[4]), rpp_hip_unpack0(src_ui6.ui1[4]), rpp_hip_unpack1(src_ui6.ui1[3]));    // write G07-G04 (mirrored load)
    srcPtr_f24->f4[3] = make_float4(rpp_hip_unpack2(src_ui6.ui1[2]), rpp_hip_unpack3(src_ui6.ui1[1]), rpp_hip_unpack0(src_ui6.ui1[1]), rpp_hip_unpack1(src_ui6.ui1[0]));    // write G03-G00 (mirrored load)
    srcPtr_f24->f4[4] = make_float4(rpp_hip_unpack3(src_ui6.ui1[5]), rpp_hip_unpack0(src_ui6.ui1[5]), rpp_hip_unpack1(src_ui6.ui1[4]), rpp_hip_unpack2(src_ui6.ui1[3]));    // write B07-B04 (mirrored load)
    srcPtr_f24->f4[5] = make_float4(rpp_hip_unpack3(src_ui6.ui1[2]), rpp_hip_unpack0(src_ui6.ui1[2]), rpp_hip_unpack1(src_ui6.ui1[1]), rpp_hip_unpack2(src_ui6.ui1[0]));    // write B03-B00 (mirrored load)
}

// F32 loads with layout toggle PKD3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(float *srcPtr, d_float24 *srcPtr_f24)
{
    d_float24 src_f24;
    *(d_float24_s *)&src_f24 = *(d_float24_s *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(src_f24.f1[ 0], src_f24.f1[ 3], src_f24.f1[ 6], src_f24.f1[ 9]);    // write R00-R03
    srcPtr_f24->f4[1] = make_float4(src_f24.f1[12], src_f24.f1[15], src_f24.f1[18], src_f24.f1[21]);    // write R04-R07
    srcPtr_f24->f4[2] = make_float4(src_f24.f1[ 1], src_f24.f1[ 4], src_f24.f1[ 7], src_f24.f1[10]);    // write G00-G03
    srcPtr_f24->f4[3] = make_float4(src_f24.f1[13], src_f24.f1[16], src_f24.f1[19], src_f24.f1[22]);    // write G04-G07
    srcPtr_f24->f4[4] = make_float4(src_f24.f1[ 2], src_f24.f1[ 5], src_f24.f1[ 8], src_f24.f1[11]);    // write B00-B03
    srcPtr_f24->f4[5] = make_float4(src_f24.f1[14], src_f24.f1[17], src_f24.f1[20], src_f24.f1[23]);    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(float *srcPtr, d_float24 *srcPtr_f24)
{
    d_float24 src_f24;
    *(d_float24_s *)&src_f24 = *(d_float24_s *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(src_f24.f1[21], src_f24.f1[18], src_f24.f1[15], src_f24.f1[12]);    // write R07-R04 (mirrored load)
    srcPtr_f24->f4[1] = make_float4(src_f24.f1[ 9], src_f24.f1[ 6], src_f24.f1[ 3], src_f24.f1[ 0]);    // write R03-R00 (mirrored load)
    srcPtr_f24->f4[2] = make_float4(src_f24.f1[22], src_f24.f1[19], src_f24.f1[16], src_f24.f1[13]);    // write G07-G04 (mirrored load)
    srcPtr_f24->f4[3] = make_float4(src_f24.f1[10], src_f24.f1[ 7], src_f24.f1[ 4], src_f24.f1[ 1]);    // write G03-G00 (mirrored load)
    srcPtr_f24->f4[4] = make_float4(src_f24.f1[23], src_f24.f1[20], src_f24.f1[17], src_f24.f1[14]);    // write B07-B04 (mirrored load)
    srcPtr_f24->f4[5] = make_float4(src_f24.f1[11], src_f24.f1[ 8], src_f24.f1[ 5], src_f24.f1[ 2]);    // write B03-B00 (mirrored load)
}

// I8 loads with layout toggle PKD3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(schar *srcPtr, d_float24 *srcPtr_f24)
{
    d_int6 src_i6 = *(d_int6 *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(rpp_hip_unpack0(src_i6.i1[0]), rpp_hip_unpack3(src_i6.i1[0]), rpp_hip_unpack2(src_i6.i1[1]), rpp_hip_unpack1(src_i6.i1[2]));    // write R00-R03
    srcPtr_f24->f4[1] = make_float4(rpp_hip_unpack0(src_i6.i1[3]), rpp_hip_unpack3(src_i6.i1[3]), rpp_hip_unpack2(src_i6.i1[4]), rpp_hip_unpack1(src_i6.i1[5]));    // write R04-R07
    srcPtr_f24->f4[2] = make_float4(rpp_hip_unpack1(src_i6.i1[0]), rpp_hip_unpack0(src_i6.i1[1]), rpp_hip_unpack3(src_i6.i1[1]), rpp_hip_unpack2(src_i6.i1[2]));    // write G00-G03
    srcPtr_f24->f4[3] = make_float4(rpp_hip_unpack1(src_i6.i1[3]), rpp_hip_unpack0(src_i6.i1[4]), rpp_hip_unpack3(src_i6.i1[4]), rpp_hip_unpack2(src_i6.i1[5]));    // write G04-G07
    srcPtr_f24->f4[4] = make_float4(rpp_hip_unpack2(src_i6.i1[0]), rpp_hip_unpack1(src_i6.i1[1]), rpp_hip_unpack0(src_i6.i1[2]), rpp_hip_unpack3(src_i6.i1[2]));    // write B00-B03
    srcPtr_f24->f4[5] = make_float4(rpp_hip_unpack2(src_i6.i1[3]), rpp_hip_unpack1(src_i6.i1[4]), rpp_hip_unpack0(src_i6.i1[5]), rpp_hip_unpack3(src_i6.i1[5]));    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(schar *srcPtr, d_float24 *srcPtr_f24)
{
    d_int6 src_i6 = *(d_int6 *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(rpp_hip_unpack1(src_i6.i1[5]), rpp_hip_unpack2(src_i6.i1[4]), rpp_hip_unpack3(src_i6.i1[3]), rpp_hip_unpack0(src_i6.i1[3]));    // write R07-R04 (mirrored load)
    srcPtr_f24->f4[1] = make_float4(rpp_hip_unpack1(src_i6.i1[2]), rpp_hip_unpack2(src_i6.i1[1]), rpp_hip_unpack3(src_i6.i1[0]), rpp_hip_unpack0(src_i6.i1[0]));    // write R03-R00 (mirrored load)
    srcPtr_f24->f4[2] = make_float4(rpp_hip_unpack2(src_i6.i1[5]), rpp_hip_unpack3(src_i6.i1[4]), rpp_hip_unpack0(src_i6.i1[4]), rpp_hip_unpack1(src_i6.i1[3]));    // write G07-G04 (mirrored load)
    srcPtr_f24->f4[3] = make_float4(rpp_hip_unpack2(src_i6.i1[2]), rpp_hip_unpack3(src_i6.i1[1]), rpp_hip_unpack0(src_i6.i1[1]), rpp_hip_unpack1(src_i6.i1[0]));    // write G03-G00 (mirrored load)
    srcPtr_f24->f4[4] = make_float4(rpp_hip_unpack3(src_i6.i1[5]), rpp_hip_unpack0(src_i6.i1[5]), rpp_hip_unpack1(src_i6.i1[4]), rpp_hip_unpack2(src_i6.i1[3]));    // write B07-B04 (mirrored load)
    srcPtr_f24->f4[5] = make_float4(rpp_hip_unpack3(src_i6.i1[2]), rpp_hip_unpack0(src_i6.i1[2]), rpp_hip_unpack1(src_i6.i1[1]), rpp_hip_unpack2(src_i6.i1[0]));    // write B03-B00 (mirrored load)
}

// F16 loads with layout toggle PKD3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(half *srcPtr, d_float24 *srcPtr_f24)
{
    d_half24 src_h24;
    src_h24 = *(d_half24 *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(__half2float(src_h24.h1[ 0]), __half2float(src_h24.h1[ 3]), __half2float(src_h24.h1[ 6]), __half2float(src_h24.h1[ 9]));    // write R00-R03
    srcPtr_f24->f4[1] = make_float4(__half2float(src_h24.h1[12]), __half2float(src_h24.h1[15]), __half2float(src_h24.h1[18]), __half2float(src_h24.h1[21]));    // write R04-R07
    srcPtr_f24->f4[2] = make_float4(__half2float(src_h24.h1[ 1]), __half2float(src_h24.h1[ 4]), __half2float(src_h24.h1[ 7]), __half2float(src_h24.h1[10]));    // write G00-G03
    srcPtr_f24->f4[3] = make_float4(__half2float(src_h24.h1[13]), __half2float(src_h24.h1[16]), __half2float(src_h24.h1[19]), __half2float(src_h24.h1[22]));    // write G04-G07
    srcPtr_f24->f4[4] = make_float4(__half2float(src_h24.h1[ 2]), __half2float(src_h24.h1[ 5]), __half2float(src_h24.h1[ 8]), __half2float(src_h24.h1[11]));    // write B00-B03
    srcPtr_f24->f4[5] = make_float4(__half2float(src_h24.h1[14]), __half2float(src_h24.h1[17]), __half2float(src_h24.h1[20]), __half2float(src_h24.h1[23]));    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(half *srcPtr, d_float24 *srcPtr_f24)
{
    d_half24 src_h24;
    src_h24 = *(d_half24 *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(__half2float(src_h24.h1[21]), __half2float(src_h24.h1[18]), __half2float(src_h24.h1[15]), __half2float(src_h24.h1[12]));    // write R07-R04 (mirrored load)
    srcPtr_f24->f4[1] = make_float4(__half2float(src_h24.h1[ 9]), __half2float(src_h24.h1[ 6]), __half2float(src_h24.h1[ 3]), __half2float(src_h24.h1[ 0]));    // write R03-R00 (mirrored load)
    srcPtr_f24->f4[2] = make_float4(__half2float(src_h24.h1[22]), __half2float(src_h24.h1[19]), __half2float(src_h24.h1[16]), __half2float(src_h24.h1[13]));    // write G07-G04 (mirrored load)
    srcPtr_f24->f4[3] = make_float4(__half2float(src_h24.h1[10]), __half2float(src_h24.h1[ 7]), __half2float(src_h24.h1[ 4]), __half2float(src_h24.h1[ 1]));    // write G03-G00 (mirrored load)
    srcPtr_f24->f4[4] = make_float4(__half2float(src_h24.h1[23]), __half2float(src_h24.h1[20]), __half2float(src_h24.h1[17]), __half2float(src_h24.h1[14]));    // write B07-B04 (mirrored load)
    srcPtr_f24->f4[5] = make_float4(__half2float(src_h24.h1[11]), __half2float(src_h24.h1[ 8]), __half2float(src_h24.h1[ 5]), __half2float(src_h24.h1[ 2]));    // write B03-B00 (mirrored load)
}

// UINT loads with layout toggle PLN3 to PKD3 (24 UINT pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(uint *srcPtr, d_float24 *srcPtr_f24)
{
    d_uint24 src_ui24;
    *(d_uint24_s *)&src_ui24 = *(d_uint24_s *)srcPtr;

    srcPtr_f24->f4[0] = make_float4((float)src_ui24.ui1[ 0], (float)src_ui24.ui1[ 3], (float)src_ui24.ui1[ 6], (float)src_ui24.ui1[ 9]);    // write R00-R03
    srcPtr_f24->f4[1] = make_float4((float)src_ui24.ui1[12], (float)src_ui24.ui1[15], (float)src_ui24.ui1[18], (float)src_ui24.ui1[21]);    // write R04-R07
    srcPtr_f24->f4[2] = make_float4((float)src_ui24.ui1[ 1], (float)src_ui24.ui1[ 4], (float)src_ui24.ui1[ 7], (float)src_ui24.ui1[10]);    // write G00-G03
    srcPtr_f24->f4[3] = make_float4((float)src_ui24.ui1[13], (float)src_ui24.ui1[16], (float)src_ui24.ui1[19], (float)src_ui24.ui1[22]);    // write G04-G07
    srcPtr_f24->f4[4] = make_float4((float)src_ui24.ui1[ 2], (float)src_ui24.ui1[ 5], (float)src_ui24.ui1[ 8], (float)src_ui24.ui1[11]);    // write B00-B03
    srcPtr_f24->f4[5] = make_float4((float)src_ui24.ui1[14], (float)src_ui24.ui1[17], (float)src_ui24.ui1[20], (float)src_ui24.ui1[23]);    // write B04-B07
}

// INT loads with layout toggle PLN3 to PKD3 (24 INT pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(int *srcPtr, d_float24 *srcPtr_f24)
{
    d_int24 src_i24;
    *(d_int24_s *)&src_i24 = *(d_int24_s *)srcPtr;

    srcPtr_f24->f4[0] = make_float4((float)src_i24.i1[ 0], (float)src_i24.i1[ 3], (float)src_i24.i1[ 6], (float)src_i24.i1[ 9]);    // write R00-R03
    srcPtr_f24->f4[1] = make_float4((float)src_i24.i1[12], (float)src_i24.i1[15], (float)src_i24.i1[18], (float)src_i24.i1[21]);    // write R04-R07
    srcPtr_f24->f4[2] = make_float4((float)src_i24.i1[ 1], (float)src_i24.i1[ 4], (float)src_i24.i1[ 7], (float)src_i24.i1[10]);    // write G00-G03
    srcPtr_f24->f4[3] = make_float4((float)src_i24.i1[13], (float)src_i24.i1[16], (float)src_i24.i1[19], (float)src_i24.i1[22]);    // write G04-G07
    srcPtr_f24->f4[4] = make_float4((float)src_i24.i1[ 2], (float)src_i24.i1[ 5], (float)src_i24.i1[ 8], (float)src_i24.i1[11]);    // write B00-B03
    srcPtr_f24->f4[5] = make_float4((float)src_i24.i1[14], (float)src_i24.i1[17], (float)src_i24.i1[20], (float)src_i24.i1[23]);    // write B04-B07
}

// U8 loads with layout toggle PLN3 to PKD3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(uchar *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    d_uint6 src_ui6;

    src_ui6.ui2[0] = *(uint2 *)srcPtr;
    srcPtr += increment;
    src_ui6.ui2[1] = *(uint2 *)srcPtr;
    srcPtr += increment;
    src_ui6.ui2[2] = *(uint2 *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(rpp_hip_unpack0(src_ui6.ui1[0]), rpp_hip_unpack0(src_ui6.ui1[2]), rpp_hip_unpack0(src_ui6.ui1[4]), rpp_hip_unpack1(src_ui6.ui1[0]));    // write R00G00B00R01
    srcPtr_f24->f4[1] = make_float4(rpp_hip_unpack1(src_ui6.ui1[2]), rpp_hip_unpack1(src_ui6.ui1[4]), rpp_hip_unpack2(src_ui6.ui1[0]), rpp_hip_unpack2(src_ui6.ui1[2]));    // write G01B01R02G02
    srcPtr_f24->f4[2] = make_float4(rpp_hip_unpack2(src_ui6.ui1[4]), rpp_hip_unpack3(src_ui6.ui1[0]), rpp_hip_unpack3(src_ui6.ui1[2]), rpp_hip_unpack3(src_ui6.ui1[4]));    // write B02R03G03B03
    srcPtr_f24->f4[3] = make_float4(rpp_hip_unpack0(src_ui6.ui1[1]), rpp_hip_unpack0(src_ui6.ui1[3]), rpp_hip_unpack0(src_ui6.ui1[5]), rpp_hip_unpack1(src_ui6.ui1[1]));    // write R04G04B04R05
    srcPtr_f24->f4[4] = make_float4(rpp_hip_unpack1(src_ui6.ui1[3]), rpp_hip_unpack1(src_ui6.ui1[5]), rpp_hip_unpack2(src_ui6.ui1[1]), rpp_hip_unpack2(src_ui6.ui1[3]));    // write G05B05R06G06
    srcPtr_f24->f4[5] = make_float4(rpp_hip_unpack2(src_ui6.ui1[5]), rpp_hip_unpack3(src_ui6.ui1[1]), rpp_hip_unpack3(src_ui6.ui1[3]), rpp_hip_unpack3(src_ui6.ui1[5]));    // write B06R07G07B07
}

// F32 loads with layout toggle PLN3 to PKD3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(float *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    d_float24 src_f24;

    *(d_float8_s *)&(src_f24.f8[0]) = *(d_float8_s *)srcPtr;
    srcPtr += increment;
    *(d_float8_s *)&(src_f24.f8[1]) = *(d_float8_s *)srcPtr;
    srcPtr += increment;
    *(d_float8_s *)&(src_f24.f8[2]) = *(d_float8_s *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(src_f24.f1[ 0], src_f24.f1[ 8], src_f24.f1[16], src_f24.f1[ 1]);    // write R00G00B00R01
    srcPtr_f24->f4[1] = make_float4(src_f24.f1[ 9], src_f24.f1[17], src_f24.f1[ 2], src_f24.f1[10]);    // write G01B01R02G02
    srcPtr_f24->f4[2] = make_float4(src_f24.f1[18], src_f24.f1[ 3], src_f24.f1[11], src_f24.f1[19]);    // write B02R03G03B03
    srcPtr_f24->f4[3] = make_float4(src_f24.f1[ 4], src_f24.f1[12], src_f24.f1[20], src_f24.f1[ 5]);    // write R04G04B04R05
    srcPtr_f24->f4[4] = make_float4(src_f24.f1[13], src_f24.f1[21], src_f24.f1[ 6], src_f24.f1[14]);    // write G05B05R06G06
    srcPtr_f24->f4[5] = make_float4(src_f24.f1[22], src_f24.f1[ 7], src_f24.f1[15], src_f24.f1[23]);    // write B06R07G07B07
}

// I8 loads with layout toggle PLN3 to PKD3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(schar *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    d_int6 src_i6;

    src_i6.i2[0] = *(int2 *)srcPtr;
    srcPtr += increment;
    src_i6.i2[1] = *(int2 *)srcPtr;
    srcPtr += increment;
    src_i6.i2[2] = *(int2 *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(rpp_hip_unpack0(src_i6.i1[0]), rpp_hip_unpack0(src_i6.i1[2]), rpp_hip_unpack0(src_i6.i1[4]), rpp_hip_unpack1(src_i6.i1[0]));    // write R00G00B00R01
    srcPtr_f24->f4[1] = make_float4(rpp_hip_unpack1(src_i6.i1[2]), rpp_hip_unpack1(src_i6.i1[4]), rpp_hip_unpack2(src_i6.i1[0]), rpp_hip_unpack2(src_i6.i1[2]));    // write G01B01R02G02
    srcPtr_f24->f4[2] = make_float4(rpp_hip_unpack2(src_i6.i1[4]), rpp_hip_unpack3(src_i6.i1[0]), rpp_hip_unpack3(src_i6.i1[2]), rpp_hip_unpack3(src_i6.i1[4]));    // write B02R03G03B03
    srcPtr_f24->f4[3] = make_float4(rpp_hip_unpack0(src_i6.i1[1]), rpp_hip_unpack0(src_i6.i1[3]), rpp_hip_unpack0(src_i6.i1[5]), rpp_hip_unpack1(src_i6.i1[1]));    // write R04G04B04R05
    srcPtr_f24->f4[4] = make_float4(rpp_hip_unpack1(src_i6.i1[3]), rpp_hip_unpack1(src_i6.i1[5]), rpp_hip_unpack2(src_i6.i1[1]), rpp_hip_unpack2(src_i6.i1[3]));    // write G05B05R06G06
    srcPtr_f24->f4[5] = make_float4(rpp_hip_unpack2(src_i6.i1[5]), rpp_hip_unpack3(src_i6.i1[1]), rpp_hip_unpack3(src_i6.i1[3]), rpp_hip_unpack3(src_i6.i1[5]));    // write B06R07G07B07
}

// F16 loads with layout toggle PLN3 to PKD3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(half *srcPtr, uint increment, d_float24 *srcPtr_f24)
{
    d_half24 src_h24;

    src_h24.h8[0] = *(d_half8 *)srcPtr;
    srcPtr += increment;
    src_h24.h8[1] = *(d_half8 *)srcPtr;
    srcPtr += increment;
    src_h24.h8[2] = *(d_half8 *)srcPtr;

    srcPtr_f24->f4[0] = make_float4(__half2float(src_h24.h1[ 0]), __half2float(src_h24.h1[ 8]), __half2float(src_h24.h1[16]), __half2float(src_h24.h1[ 1]));    // write R00G00B00R01
    srcPtr_f24->f4[1] = make_float4(__half2float(src_h24.h1[ 9]), __half2float(src_h24.h1[17]), __half2float(src_h24.h1[ 2]), __half2float(src_h24.h1[10]));    // write G01B01R02G02
    srcPtr_f24->f4[2] = make_float4(__half2float(src_h24.h1[18]), __half2float(src_h24.h1[ 3]), __half2float(src_h24.h1[11]), __half2float(src_h24.h1[19]));    // write B02R03G03B03
    srcPtr_f24->f4[3] = make_float4(__half2float(src_h24.h1[ 4]), __half2float(src_h24.h1[12]), __half2float(src_h24.h1[20]), __half2float(src_h24.h1[ 5]));    // write R04G04B04R05
    srcPtr_f24->f4[4] = make_float4(__half2float(src_h24.h1[13]), __half2float(src_h24.h1[21]), __half2float(src_h24.h1[ 6]), __half2float(src_h24.h1[14]));    // write G05B05R06G06
    srcPtr_f24->f4[5] = make_float4(__half2float(src_h24.h1[22]), __half2float(src_h24.h1[ 7]), __half2float(src_h24.h1[15]), __half2float(src_h24.h1[23]));    // write B06R07G07B07
}

// -------------------- Set 5 - Stores from float --------------------

// WITHOUT LAYOUT TOGGLE

// U8 stores without layout toggle (8 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(uchar *dstPtr, d_float8 *dstPtr_f8)
{
    uint2 dst_ui2;
    dst_ui2.x = rpp_hip_pack(dstPtr_f8->f4[0]);
    dst_ui2.y = rpp_hip_pack(dstPtr_f8->f4[1]);
    *(uint2 *)dstPtr = dst_ui2;
}

// F32 stores without layout toggle (8 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(float *dstPtr, d_float8 *dstPtr_f8)
{
    *(d_float8_s *)dstPtr = *(d_float8_s *)dstPtr_f8;
}

// I8 stores without layout toggle (8 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(schar *dstPtr, d_float8 *dstPtr_f8)
{
    uint2 dst_ui2;
    dst_ui2.x = rpp_hip_pack_i8(dstPtr_f8->f4[0]);
    dst_ui2.y = rpp_hip_pack_i8(dstPtr_f8->f4[1]);
    *(uint2 *)dstPtr = dst_ui2;
}

// F16 stores without layout toggle (8 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store8(half *dstPtr, d_float8 *dst_f8)
{
    d_half8 dst_h8;

    dst_h8.h2[0] = __float22half2_rn(make_float2(dst_f8->f1[0], dst_f8->f1[1]));
    dst_h8.h2[1] = __float22half2_rn(make_float2(dst_f8->f1[2], dst_f8->f1[3]));
    dst_h8.h2[2] = __float22half2_rn(make_float2(dst_f8->f1[4], dst_f8->f1[5]));
    dst_h8.h2[3] = __float22half2_rn(make_float2(dst_f8->f1[6], dst_f8->f1[7]));

    *(d_half8 *)dstPtr = dst_h8;
}

// U8 stores without layout toggle PKD3 to PKD3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(uchar *dstPtr, d_float24 *dstPtr_f24)
{
    d_uint6 dst_ui6;

    dst_ui6.ui1[0] = rpp_hip_pack(dstPtr_f24->f4[0]);    // write R00G00B00R01
    dst_ui6.ui1[1] = rpp_hip_pack(dstPtr_f24->f4[1]);    // write G01B01R02G02
    dst_ui6.ui1[2] = rpp_hip_pack(dstPtr_f24->f4[2]);    // write B02R03G03B03
    dst_ui6.ui1[3] = rpp_hip_pack(dstPtr_f24->f4[3]);    // write R04G04B04R05
    dst_ui6.ui1[4] = rpp_hip_pack(dstPtr_f24->f4[4]);    // write G05B05R06G06
    dst_ui6.ui1[5] = rpp_hip_pack(dstPtr_f24->f4[5]);    // write B06R07G07B07

    *(d_uint6_s *)dstPtr = *(d_uint6_s *)&dst_ui6;
}

// F32 stores without layout toggle PKD3 to PKD3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(float *dstPtr, d_float24 *dstPtr_f24)
{
    *(d_float24_s *)dstPtr = *(d_float24_s *)dstPtr_f24;
}

// I8 stores without layout toggle PKD3 to PKD3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(schar *dstPtr, d_float24 *dstPtr_f24)
{
    d_uint6 dst_ui6;

    dst_ui6.ui1[0] = rpp_hip_pack_i8(dstPtr_f24->f4[0]);    // write R00G00B00R01
    dst_ui6.ui1[1] = rpp_hip_pack_i8(dstPtr_f24->f4[1]);    // write G01B01R02G02
    dst_ui6.ui1[2] = rpp_hip_pack_i8(dstPtr_f24->f4[2]);    // write B02R03G03B03
    dst_ui6.ui1[3] = rpp_hip_pack_i8(dstPtr_f24->f4[3]);    // write R04G04B04R05
    dst_ui6.ui1[4] = rpp_hip_pack_i8(dstPtr_f24->f4[4]);    // write G05B05R06G06
    dst_ui6.ui1[5] = rpp_hip_pack_i8(dstPtr_f24->f4[5]);    // write B06R07G07B07

    *(d_uint6_s *)dstPtr = *(d_uint6_s *)&dst_ui6;
}

// F16 stores without layout toggle PKD3 to PKD3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(half *dstPtr, d_float24 *dstPtr_f24)
{
    d_half24 dst_h24;

    dst_h24.h2[ 0] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 0], dstPtr_f24->f1[ 1]));    // write R00G00
    dst_h24.h2[ 1] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 2], dstPtr_f24->f1[ 3]));    // write B00R01
    dst_h24.h2[ 2] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 4], dstPtr_f24->f1[ 5]));    // write G01B01
    dst_h24.h2[ 3] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 6], dstPtr_f24->f1[ 7]));    // write R02G02
    dst_h24.h2[ 4] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 8], dstPtr_f24->f1[ 9]));    // write B02R03
    dst_h24.h2[ 5] = __float22half2_rn(make_float2(dstPtr_f24->f1[10], dstPtr_f24->f1[11]));    // write G03B03
    dst_h24.h2[ 6] = __float22half2_rn(make_float2(dstPtr_f24->f1[12], dstPtr_f24->f1[13]));    // write R04G04
    dst_h24.h2[ 7] = __float22half2_rn(make_float2(dstPtr_f24->f1[14], dstPtr_f24->f1[15]));    // write B04R05
    dst_h24.h2[ 8] = __float22half2_rn(make_float2(dstPtr_f24->f1[16], dstPtr_f24->f1[17]));    // write G05B05
    dst_h24.h2[ 9] = __float22half2_rn(make_float2(dstPtr_f24->f1[18], dstPtr_f24->f1[19]));    // write R06G06
    dst_h24.h2[10] = __float22half2_rn(make_float2(dstPtr_f24->f1[20], dstPtr_f24->f1[21]));    // write B06R07
    dst_h24.h2[11] = __float22half2_rn(make_float2(dstPtr_f24->f1[22], dstPtr_f24->f1[23]));    // write G07B07

    *(d_half24 *)dstPtr = dst_h24;
}

// U8 stores without layout toggle PLN3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(uchar *dstPtr, uint increment, d_float24 *dstPtr_f24)
{
    d_uint6 dst_ui6;

    dst_ui6.ui1[0] = rpp_hip_pack(dstPtr_f24->f4[0]);    // write R00-R03
    dst_ui6.ui1[1] = rpp_hip_pack(dstPtr_f24->f4[1]);    // write R04-R07
    dst_ui6.ui1[2] = rpp_hip_pack(dstPtr_f24->f4[2]);    // write G00-G03
    dst_ui6.ui1[3] = rpp_hip_pack(dstPtr_f24->f4[3]);    // write G04-G07
    dst_ui6.ui1[4] = rpp_hip_pack(dstPtr_f24->f4[4]);    // write B00-B03
    dst_ui6.ui1[5] = rpp_hip_pack(dstPtr_f24->f4[5]);    // write B04-B07

    *(uint2 *)dstPtr = dst_ui6.ui2[0];
    dstPtr += increment;
    *(uint2 *)dstPtr = dst_ui6.ui2[1];
    dstPtr += increment;
    *(uint2 *)dstPtr = dst_ui6.ui2[2];
}

// F32 stores without layout toggle PLN3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(float *dstPtr, uint increment, d_float24 *dstPtr_f24)
{
    *(d_float8_s *)dstPtr = *(d_float8_s *)&(dstPtr_f24->f8[0]);
    dstPtr += increment;
    *(d_float8_s *)dstPtr = *(d_float8_s *)&(dstPtr_f24->f8[1]);
    dstPtr += increment;
    *(d_float8_s *)dstPtr = *(d_float8_s *)&(dstPtr_f24->f8[2]);
}

// I8 stores without layout toggle PLN3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(schar *dstPtr, uint increment, d_float24 *dstPtr_f24)
{
    d_uint6 dst_ui6;

    dst_ui6.ui1[0] = rpp_hip_pack_i8(dstPtr_f24->f4[0]);    // write R00-R03
    dst_ui6.ui1[1] = rpp_hip_pack_i8(dstPtr_f24->f4[1]);    // write R04-R07
    dst_ui6.ui1[2] = rpp_hip_pack_i8(dstPtr_f24->f4[2]);    // write G00-G03
    dst_ui6.ui1[3] = rpp_hip_pack_i8(dstPtr_f24->f4[3]);    // write G04-G07
    dst_ui6.ui1[4] = rpp_hip_pack_i8(dstPtr_f24->f4[4]);    // write B00-B03
    dst_ui6.ui1[5] = rpp_hip_pack_i8(dstPtr_f24->f4[5]);    // write B04-B07

    *(uint2 *)dstPtr = dst_ui6.ui2[0];
    dstPtr += increment;
    *(uint2 *)dstPtr = dst_ui6.ui2[1];
    dstPtr += increment;
    *(uint2 *)dstPtr = dst_ui6.ui2[2];
}

// F16 stores without layout toggle PLN3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pln3(half *dstPtr, uint increment, d_float24 *dstPtr_f24)
{
    d_half24 dst_h24;

    dst_h24.h2[ 0] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 0], dstPtr_f24->f1[ 1]));    // write R00R01
    dst_h24.h2[ 1] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 2], dstPtr_f24->f1[ 3]));    // write R02R03
    dst_h24.h2[ 2] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 4], dstPtr_f24->f1[ 5]));    // write R04R05
    dst_h24.h2[ 3] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 6], dstPtr_f24->f1[ 7]));    // write R06R07
    dst_h24.h2[ 4] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 8], dstPtr_f24->f1[ 9]));    // write G00G01
    dst_h24.h2[ 5] = __float22half2_rn(make_float2(dstPtr_f24->f1[10], dstPtr_f24->f1[11]));    // write G02G03
    dst_h24.h2[ 6] = __float22half2_rn(make_float2(dstPtr_f24->f1[12], dstPtr_f24->f1[13]));    // write G04G05
    dst_h24.h2[ 7] = __float22half2_rn(make_float2(dstPtr_f24->f1[14], dstPtr_f24->f1[15]));    // write G06G07
    dst_h24.h2[ 8] = __float22half2_rn(make_float2(dstPtr_f24->f1[16], dstPtr_f24->f1[17]));    // write B00B01
    dst_h24.h2[ 9] = __float22half2_rn(make_float2(dstPtr_f24->f1[18], dstPtr_f24->f1[19]));    // write B02B03
    dst_h24.h2[10] = __float22half2_rn(make_float2(dstPtr_f24->f1[20], dstPtr_f24->f1[21]));    // write B04B05
    dst_h24.h2[11] = __float22half2_rn(make_float2(dstPtr_f24->f1[22], dstPtr_f24->f1[23]));    // write B06B07

    *(d_half8 *)dstPtr = dst_h24.h8[0];
    dstPtr += increment;
    *(d_half8 *)dstPtr = dst_h24.h8[1];
    dstPtr += increment;
    *(d_half8 *)dstPtr = dst_h24.h8[2];
}

// WITH LAYOUT TOGGLE

// U8 stores with layout toggle PLN3 to PKD3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(uchar *dstPtr, d_float24 *dstPtr_f24)
{
    d_uint6 dst_ui6;

    dst_ui6.ui1[0] = rpp_hip_pack(make_float4(dstPtr_f24->f1[ 0], dstPtr_f24->f1[ 8], dstPtr_f24->f1[16], dstPtr_f24->f1[ 1]));    // write R00G00B00R01
    dst_ui6.ui1[1] = rpp_hip_pack(make_float4(dstPtr_f24->f1[ 9], dstPtr_f24->f1[17], dstPtr_f24->f1[ 2], dstPtr_f24->f1[10]));    // write G01B01R02G02
    dst_ui6.ui1[2] = rpp_hip_pack(make_float4(dstPtr_f24->f1[18], dstPtr_f24->f1[ 3], dstPtr_f24->f1[11], dstPtr_f24->f1[19]));    // write B02R03G03B03
    dst_ui6.ui1[3] = rpp_hip_pack(make_float4(dstPtr_f24->f1[ 4], dstPtr_f24->f1[12], dstPtr_f24->f1[20], dstPtr_f24->f1[ 5]));    // write R04G04B04R05
    dst_ui6.ui1[4] = rpp_hip_pack(make_float4(dstPtr_f24->f1[13], dstPtr_f24->f1[21], dstPtr_f24->f1[ 6], dstPtr_f24->f1[14]));    // write G05B05R06G06
    dst_ui6.ui1[5] = rpp_hip_pack(make_float4(dstPtr_f24->f1[22], dstPtr_f24->f1[ 7], dstPtr_f24->f1[15], dstPtr_f24->f1[23]));    // write B06R07G07B07

    *(d_uint6_s *)dstPtr = *(d_uint6_s *)&dst_ui6;
}

// F32 stores with layout toggle PLN3 to PKD3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(float *dstPtr, d_float24 *dstPtr_f24)
{
    d_float24 dst_f24;

    dst_f24.f4[0] = make_float4(dstPtr_f24->f1[ 0], dstPtr_f24->f1[ 8], dstPtr_f24->f1[16], dstPtr_f24->f1[ 1]);    // write R00G00B00R01
    dst_f24.f4[1] = make_float4(dstPtr_f24->f1[ 9], dstPtr_f24->f1[17], dstPtr_f24->f1[ 2], dstPtr_f24->f1[10]);    // write G01B01R02G02
    dst_f24.f4[2] = make_float4(dstPtr_f24->f1[18], dstPtr_f24->f1[ 3], dstPtr_f24->f1[11], dstPtr_f24->f1[19]);    // write B02R03G03B03
    dst_f24.f4[3] = make_float4(dstPtr_f24->f1[ 4], dstPtr_f24->f1[12], dstPtr_f24->f1[20], dstPtr_f24->f1[ 5]);    // write R04G04B04R05
    dst_f24.f4[4] = make_float4(dstPtr_f24->f1[13], dstPtr_f24->f1[21], dstPtr_f24->f1[ 6], dstPtr_f24->f1[14]);    // write G05B05R06G06
    dst_f24.f4[5] = make_float4(dstPtr_f24->f1[22], dstPtr_f24->f1[ 7], dstPtr_f24->f1[15], dstPtr_f24->f1[23]);    // write B06R07G07B07

    *(d_float24_s *)dstPtr = *(d_float24_s *)&dst_f24;
}

// I8 stores with layout toggle PLN3 to PKD3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(schar *dstPtr, d_float24 *dstPtr_f24)
{
    d_uint6 dst_ui6;

    dst_ui6.ui1[0] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[ 0], dstPtr_f24->f1[ 8], dstPtr_f24->f1[16], dstPtr_f24->f1[ 1]));    // write R00G00B00R01
    dst_ui6.ui1[1] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[ 9], dstPtr_f24->f1[17], dstPtr_f24->f1[ 2], dstPtr_f24->f1[10]));    // write G01B01R02G02
    dst_ui6.ui1[2] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[18], dstPtr_f24->f1[ 3], dstPtr_f24->f1[11], dstPtr_f24->f1[19]));    // write B02R03G03B03
    dst_ui6.ui1[3] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[ 4], dstPtr_f24->f1[12], dstPtr_f24->f1[20], dstPtr_f24->f1[ 5]));    // write R04G04B04R05
    dst_ui6.ui1[4] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[13], dstPtr_f24->f1[21], dstPtr_f24->f1[ 6], dstPtr_f24->f1[14]));    // write G05B05R06G06
    dst_ui6.ui1[5] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[22], dstPtr_f24->f1[ 7], dstPtr_f24->f1[15], dstPtr_f24->f1[23]));    // write B06R07G07B07

    *(d_uint6_s *)dstPtr = *(d_uint6_s *)&dst_ui6;
}

// F16 stores with layout toggle PLN3 to PKD3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pln3_and_store24_pkd3(half *dstPtr, d_float24 *dstPtr_f24)
{
    d_half24 dst_h24;

    dst_h24.h2[ 0] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 0], dstPtr_f24->f1[ 8]));    // write R00G00
    dst_h24.h2[ 1] = __float22half2_rn(make_float2(dstPtr_f24->f1[16], dstPtr_f24->f1[ 1]));    // write B00R01
    dst_h24.h2[ 2] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 9], dstPtr_f24->f1[17]));    // write G01B01
    dst_h24.h2[ 3] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 2], dstPtr_f24->f1[10]));    // write R02G02
    dst_h24.h2[ 4] = __float22half2_rn(make_float2(dstPtr_f24->f1[18], dstPtr_f24->f1[ 3]));    // write B02R03
    dst_h24.h2[ 5] = __float22half2_rn(make_float2(dstPtr_f24->f1[11], dstPtr_f24->f1[19]));    // write G03B03
    dst_h24.h2[ 6] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 4], dstPtr_f24->f1[12]));    // write R04G04
    dst_h24.h2[ 7] = __float22half2_rn(make_float2(dstPtr_f24->f1[20], dstPtr_f24->f1[ 5]));    // write B04R05
    dst_h24.h2[ 8] = __float22half2_rn(make_float2(dstPtr_f24->f1[13], dstPtr_f24->f1[21]));    // write G05B05
    dst_h24.h2[ 9] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 6], dstPtr_f24->f1[14]));    // write R06G06
    dst_h24.h2[10] = __float22half2_rn(make_float2(dstPtr_f24->f1[22], dstPtr_f24->f1[ 7]));    // write B06R07
    dst_h24.h2[11] = __float22half2_rn(make_float2(dstPtr_f24->f1[15], dstPtr_f24->f1[23]));    // write G07B07

    *(d_half24 *)dstPtr = dst_h24;
}

// U8 stores with layout toggle PKD3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pln3(uchar *dstPtr, uint increment, d_float24 *dstPtr_f24)
{
    d_uint6 dst_ui6;

    dst_ui6.ui1[0] = rpp_hip_pack(make_float4(dstPtr_f24->f1[ 0], dstPtr_f24->f1[ 3], dstPtr_f24->f1[ 6], dstPtr_f24->f1[ 9]));    // write R00-R03
    dst_ui6.ui1[1] = rpp_hip_pack(make_float4(dstPtr_f24->f1[12], dstPtr_f24->f1[15], dstPtr_f24->f1[18], dstPtr_f24->f1[21]));    // write R04-R07
    dst_ui6.ui1[2] = rpp_hip_pack(make_float4(dstPtr_f24->f1[ 1], dstPtr_f24->f1[ 4], dstPtr_f24->f1[ 7], dstPtr_f24->f1[10]));    // write G00-G03
    dst_ui6.ui1[3] = rpp_hip_pack(make_float4(dstPtr_f24->f1[13], dstPtr_f24->f1[16], dstPtr_f24->f1[19], dstPtr_f24->f1[22]));    // write G04-G07
    dst_ui6.ui1[4] = rpp_hip_pack(make_float4(dstPtr_f24->f1[ 2], dstPtr_f24->f1[ 5], dstPtr_f24->f1[ 8], dstPtr_f24->f1[11]));    // write B00-B03
    dst_ui6.ui1[5] = rpp_hip_pack(make_float4(dstPtr_f24->f1[14], dstPtr_f24->f1[17], dstPtr_f24->f1[20], dstPtr_f24->f1[23]));    // write B04-B07

    *(uint2 *)dstPtr = dst_ui6.ui2[0];
    dstPtr += increment;
    *(uint2 *)dstPtr = dst_ui6.ui2[1];
    dstPtr += increment;
    *(uint2 *)dstPtr = dst_ui6.ui2[2];
}

// F32 stores with layout toggle PKD3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pln3(float *dstPtr, uint increment, d_float24 *dstPtr_f24)
{
    d_float24 dst_f24;

    dst_f24.f4[0] = make_float4(dstPtr_f24->f1[ 0], dstPtr_f24->f1[ 3], dstPtr_f24->f1[ 6], dstPtr_f24->f1[ 9]);    // write R00-R03
    dst_f24.f4[1] = make_float4(dstPtr_f24->f1[12], dstPtr_f24->f1[15], dstPtr_f24->f1[18], dstPtr_f24->f1[21]);    // write R04-R07
    dst_f24.f4[2] = make_float4(dstPtr_f24->f1[ 1], dstPtr_f24->f1[ 4], dstPtr_f24->f1[ 7], dstPtr_f24->f1[10]);    // write G00-G03
    dst_f24.f4[3] = make_float4(dstPtr_f24->f1[13], dstPtr_f24->f1[16], dstPtr_f24->f1[19], dstPtr_f24->f1[22]);    // write G04-G07
    dst_f24.f4[4] = make_float4(dstPtr_f24->f1[ 2], dstPtr_f24->f1[ 5], dstPtr_f24->f1[ 8], dstPtr_f24->f1[11]);    // write B00-B03
    dst_f24.f4[5] = make_float4(dstPtr_f24->f1[14], dstPtr_f24->f1[17], dstPtr_f24->f1[20], dstPtr_f24->f1[23]);    // write B04-B07

    *(d_float8_s *)dstPtr = *(d_float8_s *)&dst_f24.f8[0];
    dstPtr += increment;
    *(d_float8_s *)dstPtr = *(d_float8_s *)&dst_f24.f8[1];
    dstPtr += increment;
    *(d_float8_s *)dstPtr = *(d_float8_s *)&dst_f24.f8[2];
}

// I8 stores with layout toggle PKD3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pln3(schar *dstPtr, uint increment, d_float24 *dstPtr_f24)
{
    d_uint6 dst_ui6;

    dst_ui6.ui1[0] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[ 0], dstPtr_f24->f1[ 3], dstPtr_f24->f1[ 6], dstPtr_f24->f1[ 9]));    // write R00-R03
    dst_ui6.ui1[1] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[12], dstPtr_f24->f1[15], dstPtr_f24->f1[18], dstPtr_f24->f1[21]));    // write R04-R07
    dst_ui6.ui1[2] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[ 1], dstPtr_f24->f1[ 4], dstPtr_f24->f1[ 7], dstPtr_f24->f1[10]));    // write G00-G03
    dst_ui6.ui1[3] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[13], dstPtr_f24->f1[16], dstPtr_f24->f1[19], dstPtr_f24->f1[22]));    // write G04-G07
    dst_ui6.ui1[4] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[ 2], dstPtr_f24->f1[ 5], dstPtr_f24->f1[ 8], dstPtr_f24->f1[11]));    // write B00-B03
    dst_ui6.ui1[5] = rpp_hip_pack_i8(make_float4(dstPtr_f24->f1[14], dstPtr_f24->f1[17], dstPtr_f24->f1[20], dstPtr_f24->f1[23]));    // write B04-B07

    *(uint2 *)dstPtr = dst_ui6.ui2[0];
    dstPtr += increment;
    *(uint2 *)dstPtr = dst_ui6.ui2[1];
    dstPtr += increment;
    *(uint2 *)dstPtr = dst_ui6.ui2[2];
}

// F16 stores with layout toggle PKD3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pln3(half *dstPtr, uint increment, d_float24 *dstPtr_f24)
{
    d_half24 dst_h24;

    dst_h24.h2[ 0] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 0], dstPtr_f24->f1[ 3]));    // write R00R01
    dst_h24.h2[ 1] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 6], dstPtr_f24->f1[ 9]));    // write R02R03
    dst_h24.h2[ 2] = __float22half2_rn(make_float2(dstPtr_f24->f1[12], dstPtr_f24->f1[15]));    // write R04R05
    dst_h24.h2[ 3] = __float22half2_rn(make_float2(dstPtr_f24->f1[18], dstPtr_f24->f1[21]));    // write R06R07
    dst_h24.h2[ 4] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 1], dstPtr_f24->f1[ 4]));    // write G00G01
    dst_h24.h2[ 5] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 7], dstPtr_f24->f1[10]));    // write G02G03
    dst_h24.h2[ 6] = __float22half2_rn(make_float2(dstPtr_f24->f1[13], dstPtr_f24->f1[16]));    // write G04G05
    dst_h24.h2[ 7] = __float22half2_rn(make_float2(dstPtr_f24->f1[19], dstPtr_f24->f1[22]));    // write G06G07
    dst_h24.h2[ 8] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 2], dstPtr_f24->f1[ 5]));    // write B00B01
    dst_h24.h2[ 9] = __float22half2_rn(make_float2(dstPtr_f24->f1[ 8], dstPtr_f24->f1[11]));    // write B02B03
    dst_h24.h2[10] = __float22half2_rn(make_float2(dstPtr_f24->f1[14], dstPtr_f24->f1[17]));    // write B04B05
    dst_h24.h2[11] = __float22half2_rn(make_float2(dstPtr_f24->f1[20], dstPtr_f24->f1[23]));    // write B06B07

    *(d_half8 *)dstPtr = dst_h24.h8[0];
    dstPtr += increment;
    *(d_half8 *)dstPtr = dst_h24.h8[1];
    dstPtr += increment;
    *(d_half8 *)dstPtr = dst_h24.h8[2];
}

// -------------------- Set 6 - Loads to uchar --------------------

// WITHOUT LAYOUT TOGGLE

// U8 loads without layout toggle (8 U8 pixels)

__device__ __forceinline__ void rpp_hip_load8_to_uchar8(uchar *srcPtr, uchar *srcPtr_uc8)
{
    *(uint2 *)srcPtr_uc8 = *(uint2 *)srcPtr;
}

// F32 loads without layout toggle (8 F32 pixels)

__device__ __forceinline__ void rpp_hip_load8_to_uchar8(float *srcPtr, uchar *srcPtr_uc8)
{
    d_float8 src_f8 = {0};
    *(d_float8_s *)&src_f8 = *(d_float8_s *)srcPtr;

    uint2 *srcPtr_ui2;
    srcPtr_ui2 = (uint2 *)srcPtr_uc8;
    srcPtr_ui2->x = rpp_hip_pack(rpp_hip_pixel_check_0to255(src_f8.f4[0] * (float4) 255.0));
    srcPtr_ui2->y = rpp_hip_pack(rpp_hip_pixel_check_0to255(src_f8.f4[1] * (float4) 255.0));
}

// I8 loads without layout toggle (8 I8 pixels)

__device__ __forceinline__ void rpp_hip_load8_to_uchar8(schar *srcPtr, uchar *srcPtr_uc8)
{
    rpp_hip_convert8_i8_to_u8(srcPtr, srcPtr_uc8);
}

// F16 loads without layout toggle (8 F16 pixels)

__device__ __forceinline__ void rpp_hip_load8_to_uchar8(half *srcPtr, uchar *srcPtr_uc8)
{
    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr, &src_f8);
    rpp_hip_load8_to_uchar8((float *)&src_f8, srcPtr_uc8);
}

// WITH LAYOUT TOGGLE

// U8 loads with layout toggle PKD3 to PLN3 (24 U8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_to_uchar8_pln3(uchar *srcPtr, uchar **srcPtrs_uc8)
{
    d_uchar24 src_uc24;
    *(d_uchar24_s *)&src_uc24 = *(d_uchar24_s *)srcPtr;

    d_uchar8 *srcPtrR_uc8, *srcPtrG_uc8, *srcPtrB_uc8;
    srcPtrR_uc8 = (d_uchar8 *)srcPtrs_uc8[0];
    srcPtrG_uc8 = (d_uchar8 *)srcPtrs_uc8[1];
    srcPtrB_uc8 = (d_uchar8 *)srcPtrs_uc8[2];

    srcPtrR_uc8->uc4[0] = make_uchar4(src_uc24.uc1[ 0], src_uc24.uc1[ 3], src_uc24.uc1[ 6], src_uc24.uc1[ 9]);    // write R00-R03
    srcPtrR_uc8->uc4[1] = make_uchar4(src_uc24.uc1[12], src_uc24.uc1[15], src_uc24.uc1[18], src_uc24.uc1[21]);    // write R04-R07
    srcPtrG_uc8->uc4[0] = make_uchar4(src_uc24.uc1[ 1], src_uc24.uc1[ 4], src_uc24.uc1[ 7], src_uc24.uc1[10]);    // write G00-G03
    srcPtrG_uc8->uc4[1] = make_uchar4(src_uc24.uc1[13], src_uc24.uc1[16], src_uc24.uc1[19], src_uc24.uc1[22]);    // write G04-G07
    srcPtrB_uc8->uc4[0] = make_uchar4(src_uc24.uc1[ 2], src_uc24.uc1[ 5], src_uc24.uc1[ 8], src_uc24.uc1[11]);    // write B00-B03
    srcPtrB_uc8->uc4[1] = make_uchar4(src_uc24.uc1[14], src_uc24.uc1[17], src_uc24.uc1[20], src_uc24.uc1[23]);    // write B04-B07
}

// F32 loads with layout toggle PKD3 to PLN3 (24 F32 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_to_uchar8_pln3(float *srcPtr, uchar **srcPtrs_uc8)
{
    d_float24 src_f24 = {0};
    *(d_float24_s *)&src_f24 = *(d_float24_s *)srcPtr;

    d_uint6 src_ui6;
    src_ui6.ui1[0] = rpp_hip_pack(rpp_hip_pixel_check_0to255(src_f24.f4[0] * (float4) 255.0));    // write R00G00B00R01
    src_ui6.ui1[1] = rpp_hip_pack(rpp_hip_pixel_check_0to255(src_f24.f4[1] * (float4) 255.0));    // write G01B01R02G02
    src_ui6.ui1[2] = rpp_hip_pack(rpp_hip_pixel_check_0to255(src_f24.f4[2] * (float4) 255.0));    // write B02R03G03B03
    src_ui6.ui1[3] = rpp_hip_pack(rpp_hip_pixel_check_0to255(src_f24.f4[3] * (float4) 255.0));    // write R04G04B04R05
    src_ui6.ui1[4] = rpp_hip_pack(rpp_hip_pixel_check_0to255(src_f24.f4[4] * (float4) 255.0));    // write G05B05R06G06
    src_ui6.ui1[5] = rpp_hip_pack(rpp_hip_pixel_check_0to255(src_f24.f4[5] * (float4) 255.0));    // write B06R07G07B07

    d_uchar8 *srcPtrR_uc8, *srcPtrG_uc8, *srcPtrB_uc8;
    srcPtrR_uc8 = (d_uchar8 *)srcPtrs_uc8[0];
    srcPtrG_uc8 = (d_uchar8 *)srcPtrs_uc8[1];
    srcPtrB_uc8 = (d_uchar8 *)srcPtrs_uc8[2];

    d_uchar24 *srcPtr_uc24;
    srcPtr_uc24 = (d_uchar24 *)&src_ui6;

    srcPtrR_uc8->uc4[0] = make_uchar4(srcPtr_uc24->uc1[ 0], srcPtr_uc24->uc1[ 3], srcPtr_uc24->uc1[ 6], srcPtr_uc24->uc1[ 9]);    // write R00-R03
    srcPtrR_uc8->uc4[1] = make_uchar4(srcPtr_uc24->uc1[12], srcPtr_uc24->uc1[15], srcPtr_uc24->uc1[18], srcPtr_uc24->uc1[21]);    // write R04-R07
    srcPtrG_uc8->uc4[0] = make_uchar4(srcPtr_uc24->uc1[ 1], srcPtr_uc24->uc1[ 4], srcPtr_uc24->uc1[ 7], srcPtr_uc24->uc1[10]);    // write G00-G03
    srcPtrG_uc8->uc4[1] = make_uchar4(srcPtr_uc24->uc1[13], srcPtr_uc24->uc1[16], srcPtr_uc24->uc1[19], srcPtr_uc24->uc1[22]);    // write G04-G07
    srcPtrB_uc8->uc4[0] = make_uchar4(srcPtr_uc24->uc1[ 2], srcPtr_uc24->uc1[ 5], srcPtr_uc24->uc1[ 8], srcPtr_uc24->uc1[11]);    // write B00-B03
    srcPtrB_uc8->uc4[1] = make_uchar4(srcPtr_uc24->uc1[14], srcPtr_uc24->uc1[17], srcPtr_uc24->uc1[20], srcPtr_uc24->uc1[23]);    // write B04-B07
}

// I8 loads with layout toggle PKD3 to PLN3 (24 I8 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_to_uchar8_pln3(schar *srcPtr, uchar **srcPtrs_uc8)
{
    d_uchar24 src_uc24;
    rpp_hip_convert24_i8_to_u8(srcPtr, (uchar *)&src_uc24);
    rpp_hip_load24_pkd3_to_uchar8_pln3((uchar *)&src_uc24, srcPtrs_uc8);
}

// F16 loads with layout toggle PKD3 to PLN3 (24 F16 pixels)

__device__ __forceinline__ void rpp_hip_load24_pkd3_to_uchar8_pln3(half *srcPtr, uchar **srcPtrs_uchar8)
{
    d_half24 src_h24;
    src_h24 = *(d_half24 *)srcPtr;

    d_float24 src_f24;
    float2 src1_f2, src2_f2;

    src1_f2 = __half22float2(src_h24.h2[0]);
    src2_f2 = __half22float2(src_h24.h2[1]);
    src_f24.f4[0] = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write R00-R03
    src1_f2 = __half22float2(src_h24.h2[2]);
    src2_f2 = __half22float2(src_h24.h2[3]);
    src_f24.f4[1] = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write R04-R07
    src1_f2 = __half22float2(src_h24.h2[4]);
    src2_f2 = __half22float2(src_h24.h2[5]);
    src_f24.f4[2] = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write G00-G03
    src1_f2 = __half22float2(src_h24.h2[6]);
    src2_f2 = __half22float2(src_h24.h2[7]);
    src_f24.f4[3] = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write G04-G07
    src1_f2 = __half22float2(src_h24.h2[8]);
    src2_f2 = __half22float2(src_h24.h2[9]);
    src_f24.f4[4] = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write B00-B03
    src1_f2 = __half22float2(src_h24.h2[10]);
    src2_f2 = __half22float2(src_h24.h2[11]);
    src_f24.f4[5] = make_float4(src1_f2.x, src1_f2.y, src2_f2.x, src2_f2.y);    // write B04-B07

    rpp_hip_load24_pkd3_to_uchar8_pln3((float *)&src_f24, srcPtrs_uchar8);
}

// U8 loads with layout toggle PKD3 to PLN3 (24 U8 pixels) into d_uchar24

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_uchar24_pln3(uchar *srcPtr, d_uchar24 *srcPtr_uc24)
{
    d_uchar24 src_uc24;
    *(d_uchar24_s *)&src_uc24 = *(d_uchar24_s *)srcPtr;

    srcPtr_uc24->uc4[0] = make_uchar4(src_uc24.uc1[ 0], src_uc24.uc1[ 3], src_uc24.uc1[ 6], src_uc24.uc1[ 9]);    // write R00-R03
    srcPtr_uc24->uc4[1] = make_uchar4(src_uc24.uc1[12], src_uc24.uc1[15], src_uc24.uc1[18], src_uc24.uc1[21]);    // write R04-R07
    srcPtr_uc24->uc4[2] = make_uchar4(src_uc24.uc1[ 1], src_uc24.uc1[ 4], src_uc24.uc1[ 7], src_uc24.uc1[10]);    // write G00-G03
    srcPtr_uc24->uc4[3] = make_uchar4(src_uc24.uc1[13], src_uc24.uc1[16], src_uc24.uc1[19], src_uc24.uc1[22]);    // write G04-G07
    srcPtr_uc24->uc4[4] = make_uchar4(src_uc24.uc1[ 2], src_uc24.uc1[ 5], src_uc24.uc1[ 8], src_uc24.uc1[11]);    // write B00-B03
    srcPtr_uc24->uc4[5] = make_uchar4(src_uc24.uc1[14], src_uc24.uc1[17], src_uc24.uc1[20], src_uc24.uc1[23]);    // write B00-B03
}

// U8 loads with layout non toggle PKD3 to PKD3 (24 U8 pixels) into d_uchar24

__device__ __forceinline__ void rpp_hip_load24_pkd3_and_unpack_to_uchar24_pkd3(uchar *srcPtr, d_uchar24 *srcPtr_uc24)
{
    *(d_uchar24_s *)srcPtr_uc24 = *(d_uchar24_s *)srcPtr;

}

// U8 loads with layout non toggle PLN3 to PKD3 (24 U8 pixels) into d_uchar24

__device__ __forceinline__ void rpp_hip_load24_pln3_and_unpack_to_uchar24_pkd3(uchar *srcPtr, uint increment, d_uchar24 *srcPtr_uc24)
{
    d_uchar24 src_uc24;

    *(d_uchar8_s *)&(src_uc24.uc8[0]) = *(d_uchar8_s *)srcPtr;
    srcPtr += increment;
    *(d_uchar8_s *)&(src_uc24.uc8[1]) = *(d_uchar8_s *)srcPtr;
    srcPtr += increment;
    *(d_uchar8_s *)&(src_uc24.uc8[2]) = *(d_uchar8_s *)srcPtr;

    srcPtr_uc24->uc4[0] = make_uchar4(src_uc24.uc1[ 0], src_uc24.uc1[ 8], src_uc24.uc1[16], src_uc24.uc1[ 1]);    // write R00G00B00R01
    srcPtr_uc24->uc4[1] = make_uchar4(src_uc24.uc1[ 9], src_uc24.uc1[17], src_uc24.uc1[ 2], src_uc24.uc1[10]);    // write G01B01R02G02
    srcPtr_uc24->uc4[2] = make_uchar4(src_uc24.uc1[18], src_uc24.uc1[ 3], src_uc24.uc1[11], src_uc24.uc1[19]);    // write B02R03G03B03
    srcPtr_uc24->uc4[3] = make_uchar4(src_uc24.uc1[ 4], src_uc24.uc1[12], src_uc24.uc1[20], src_uc24.uc1[ 5]);    // write R04G04B04R05
    srcPtr_uc24->uc4[4] = make_uchar4(src_uc24.uc1[13], src_uc24.uc1[21], src_uc24.uc1[ 6], src_uc24.uc1[14]);    // write G05B05R06G06
    srcPtr_uc24->uc4[5] = make_uchar4(src_uc24.uc1[22], src_uc24.uc1[ 7], src_uc24.uc1[15], src_uc24.uc1[23]);    // write B06R07G07B07
}

// -------------------- Set 7 - Templated layout toggles --------------------

// PKD3 to PLN3

template <typename T>
__device__ __forceinline__ void rpp_hip_layouttoggle24_pkd3_to_pln3(T *pixpkd3Ptr_T24)
{
    T pixpln3_T24;

    pixpln3_T24.data[ 0] = pixpkd3Ptr_T24->data[ 0];
    pixpln3_T24.data[ 1] = pixpkd3Ptr_T24->data[ 3];
    pixpln3_T24.data[ 2] = pixpkd3Ptr_T24->data[ 6];
    pixpln3_T24.data[ 3] = pixpkd3Ptr_T24->data[ 9];
    pixpln3_T24.data[ 4] = pixpkd3Ptr_T24->data[12];
    pixpln3_T24.data[ 5] = pixpkd3Ptr_T24->data[15];
    pixpln3_T24.data[ 6] = pixpkd3Ptr_T24->data[18];
    pixpln3_T24.data[ 7] = pixpkd3Ptr_T24->data[21];
    pixpln3_T24.data[ 8] = pixpkd3Ptr_T24->data[ 1];
    pixpln3_T24.data[ 9] = pixpkd3Ptr_T24->data[ 4];
    pixpln3_T24.data[10] = pixpkd3Ptr_T24->data[ 7];
    pixpln3_T24.data[11] = pixpkd3Ptr_T24->data[10];
    pixpln3_T24.data[12] = pixpkd3Ptr_T24->data[13];
    pixpln3_T24.data[13] = pixpkd3Ptr_T24->data[16];
    pixpln3_T24.data[14] = pixpkd3Ptr_T24->data[19];
    pixpln3_T24.data[15] = pixpkd3Ptr_T24->data[22];
    pixpln3_T24.data[16] = pixpkd3Ptr_T24->data[ 2];
    pixpln3_T24.data[17] = pixpkd3Ptr_T24->data[ 5];
    pixpln3_T24.data[18] = pixpkd3Ptr_T24->data[ 8];
    pixpln3_T24.data[19] = pixpkd3Ptr_T24->data[11];
    pixpln3_T24.data[20] = pixpkd3Ptr_T24->data[14];
    pixpln3_T24.data[21] = pixpkd3Ptr_T24->data[17];
    pixpln3_T24.data[22] = pixpkd3Ptr_T24->data[20];
    pixpln3_T24.data[23] = pixpkd3Ptr_T24->data[23];

    *pixpkd3Ptr_T24 = pixpln3_T24;
}

// PLN3 to PKD3

template <typename T>
__device__ __forceinline__ void rpp_hip_layouttoggle24_pln3_to_pkd3(T *pixpln3Ptr_T24)
{
    T pixpkd3_T24;

    pixpkd3_T24.data[ 0] = pixpln3Ptr_T24->data[ 0];
    pixpkd3_T24.data[ 1] = pixpln3Ptr_T24->data[ 8];
    pixpkd3_T24.data[ 2] = pixpln3Ptr_T24->data[16];
    pixpkd3_T24.data[ 3] = pixpln3Ptr_T24->data[ 1];
    pixpkd3_T24.data[ 4] = pixpln3Ptr_T24->data[ 9];
    pixpkd3_T24.data[ 5] = pixpln3Ptr_T24->data[17];
    pixpkd3_T24.data[ 6] = pixpln3Ptr_T24->data[ 2];
    pixpkd3_T24.data[ 7] = pixpln3Ptr_T24->data[10];
    pixpkd3_T24.data[ 8] = pixpln3Ptr_T24->data[18];
    pixpkd3_T24.data[ 9] = pixpln3Ptr_T24->data[ 3];
    pixpkd3_T24.data[10] = pixpln3Ptr_T24->data[11];
    pixpkd3_T24.data[11] = pixpln3Ptr_T24->data[19];
    pixpkd3_T24.data[12] = pixpln3Ptr_T24->data[ 4];
    pixpkd3_T24.data[13] = pixpln3Ptr_T24->data[12];
    pixpkd3_T24.data[14] = pixpln3Ptr_T24->data[20];
    pixpkd3_T24.data[15] = pixpln3Ptr_T24->data[ 5];
    pixpkd3_T24.data[16] = pixpln3Ptr_T24->data[13];
    pixpkd3_T24.data[17] = pixpln3Ptr_T24->data[21];
    pixpkd3_T24.data[18] = pixpln3Ptr_T24->data[ 6];
    pixpkd3_T24.data[19] = pixpln3Ptr_T24->data[14];
    pixpkd3_T24.data[20] = pixpln3Ptr_T24->data[22];
    pixpkd3_T24.data[21] = pixpln3Ptr_T24->data[ 7];
    pixpkd3_T24.data[22] = pixpln3Ptr_T24->data[15];
    pixpkd3_T24.data[23] = pixpln3Ptr_T24->data[23];

    *pixpln3Ptr_T24 = pixpkd3_T24;
}

// ------------------------- Set 8 - Loads to uint / int --------------------------

__device__ __forceinline__ void rpp_hip_load8_to_uint8(uchar *srcPtr, d_uint8 *srcPtr_ui8)
{
    d_uchar8 src_uc8;
    *(d_uchar8_s *)&src_uc8 = *(d_uchar8_s *)srcPtr;

    srcPtr_ui8->ui4[0] = rpp_hip_pack_uint4(src_uc8.uc4[0]);
    srcPtr_ui8->ui4[1] = rpp_hip_pack_uint4(src_uc8.uc4[1]);
}

__device__ __forceinline__ void rpp_hip_load8_to_int8(schar *srcPtr, d_int8 *srcPtr_i8)
{
    d_schar8_s src_sc8;
    *reinterpret_cast<d_schar8_s *>(&src_sc8) = *reinterpret_cast<d_schar8_s *>(srcPtr);

    rpp_hip_pack_int8(&src_sc8, srcPtr_i8);
}

__device__ __forceinline__ void rpp_hip_load24_pln3_to_uint24_pln3(uchar *srcPtr, uint increment, d_uint24 *srcPtr_ui24)
{
    d_uchar24 src_uc24;
    *(d_uchar8_s *)&src_uc24.uc8[0] = *(d_uchar8_s *)srcPtr;
    srcPtr += increment;
    *(d_uchar8_s *)&src_uc24.uc8[1] = *(d_uchar8_s *)srcPtr;
    srcPtr += increment;
    *(d_uchar8_s *)&src_uc24.uc8[2] = *(d_uchar8_s *)srcPtr;

    srcPtr_ui24->ui4[0] = rpp_hip_pack_uint4(src_uc24.uc4[0]);    // write R00-R03
    srcPtr_ui24->ui4[1] = rpp_hip_pack_uint4(src_uc24.uc4[1]);    // write R04-R07
    srcPtr_ui24->ui4[2] = rpp_hip_pack_uint4(src_uc24.uc4[2]);    // write G00-G03
    srcPtr_ui24->ui4[3] = rpp_hip_pack_uint4(src_uc24.uc4[3]);    // write G04-G07
    srcPtr_ui24->ui4[4] = rpp_hip_pack_uint4(src_uc24.uc4[4]);    // write B00-B03
    srcPtr_ui24->ui4[5] = rpp_hip_pack_uint4(src_uc24.uc4[5]);    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pln3_to_int24_pln3(schar *srcPtr, uint increment, d_int24 *srcPtr_i24)
{
    d_schar24_s src_sc24;
    *(d_schar8_s *)&src_sc24.sc8[0] = *(d_schar8_s *)srcPtr;
    srcPtr += increment;
    *(d_schar8_s *)&src_sc24.sc8[1] = *(d_schar8_s *)srcPtr;
    srcPtr += increment;
    *(d_schar8_s *)&src_sc24.sc8[2] = *(d_schar8_s *)srcPtr;

    rpp_hip_pack_int8(&src_sc24.sc8[0], &srcPtr_i24->i8[0]);     // write R00-R07
    rpp_hip_pack_int8(&src_sc24.sc8[1], &srcPtr_i24->i8[1]);     // write G00-G07
    rpp_hip_pack_int8(&src_sc24.sc8[2], &srcPtr_i24->i8[2]);     // write B00-B07
}

__device__ __forceinline__ void rpp_hip_load24_pkd3_to_uint24_pln3(uchar *srcPtr, d_uint24 *srcPtr_ui24)
{
    d_uchar24 src_uc24;
    *(d_uchar24_s *)&src_uc24 = *(d_uchar24_s *)srcPtr;
    rpp_hip_layouttoggle24_pkd3_to_pln3((d_uchar24_s *)&src_uc24);

    srcPtr_ui24->ui4[0] = rpp_hip_pack_uint4(src_uc24.uc4[0]);    // write R00-R03
    srcPtr_ui24->ui4[1] = rpp_hip_pack_uint4(src_uc24.uc4[1]);    // write R04-R07
    srcPtr_ui24->ui4[2] = rpp_hip_pack_uint4(src_uc24.uc4[2]);    // write G00-G03
    srcPtr_ui24->ui4[3] = rpp_hip_pack_uint4(src_uc24.uc4[3]);    // write G04-G07
    srcPtr_ui24->ui4[4] = rpp_hip_pack_uint4(src_uc24.uc4[4]);    // write B00-B03
    srcPtr_ui24->ui4[5] = rpp_hip_pack_uint4(src_uc24.uc4[5]);    // write B04-B07
}

__device__ __forceinline__ void rpp_hip_load24_pkd3_to_int24_pln3(schar *srcPtr, d_int24 *srcPtr_i24)
{
    d_schar24_s src_sc24;
    src_sc24 = *(d_schar24_s *)srcPtr;
    rpp_hip_layouttoggle24_pkd3_to_pln3((d_schar24sc1s_s *)&src_sc24);

    rpp_hip_pack_int8(&src_sc24.sc8[0], &srcPtr_i24->i8[0]);     // write R00-R07
    rpp_hip_pack_int8(&src_sc24.sc8[1], &srcPtr_i24->i8[1]);     // write G00-G07
    rpp_hip_pack_int8(&src_sc24.sc8[2], &srcPtr_i24->i8[2]);     // write B00-B07
}

// ------------------------- Set 9 - Stores from uchar8 --------------------------

__device__ __forceinline__ void rpp_hip_pack_uchar8_and_store8(uchar *dstPtr, d_uchar8 *dstPtr_f8)
{
    *(d_uchar8_s *)dstPtr = *(d_uchar8_s *)dstPtr_f8;
}

__device__ __forceinline__ void rpp_hip_pack_uchar24_pkd3_and_store24_pkd3(uchar *dstPtr, d_uchar24 *dstPtr_f24)
{
    *(d_uchar24_s *)dstPtr = *(d_uchar24_s *)dstPtr_f24;
}

__device__ __forceinline__ void rpp_hip_pack_uchar24_pln3_and_store24_pln3(uchar *dstPtr, uint increment, d_uchar24 *dstPtr_f24)
{
    *(d_uchar8_s *)dstPtr = *(d_uchar8_s *)&(dstPtr_f24->uc8[0]);
    dstPtr += increment;
    *(d_uchar8_s *)dstPtr = *(d_uchar8_s *)&(dstPtr_f24->uc8[1]);
    dstPtr += increment;
    *(d_uchar8_s *)dstPtr = *(d_uchar8_s *)&(dstPtr_f24->uc8[2]);
}

// copy ROI region from input to output for NCDHW layout tensors

template<typename T>
__global__ void copy_ncdhw_hip_tensor(T *srcPtr,
                                      uint3 srcStridesCDH,
                                      T *dstPtr,
                                      uint3 dstStridesCDH,
                                      int channels,
                                      RpptROI3DPtr roiGenericSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    uint srcIdx = ((id_z + roiGenericSrc->xyzwhdROI.xyz.z) * srcStridesCDH.y) + ((id_y + roiGenericSrc->xyzwhdROI.xyz.y) * srcStridesCDH.z) + (id_x + roiGenericSrc->xyzwhdROI.xyz.x);
    uint dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;

    d_float8 val_f8;
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &val_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
        srcIdx += srcStridesCDH.x;
        dstIdx += dstStridesCDH.x;
    }
}

// copy ROI region from input to output for NDHWC layout tensors

template<typename T>
__global__ void copy_ndhwc_hip_tensor(T *srcPtr,
                                      uint2 srcStridesDH,
                                      T *dstPtr,
                                      uint2 dstStridesDH,
                                      RpptROI3DPtr roiGenericSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // WC - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    uint srcIdx = ((id_z + roiGenericSrc->xyzwhdROI.xyz.z) * srcStridesDH.x) + ((id_y + roiGenericSrc->xyzwhdROI.xyz.y) * srcStridesDH.y) + (id_x + roiGenericSrc->xyzwhdROI.xyz.x) * 3;
    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x * 3;

    d_float24 val_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &val_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &val_f24);
}

// PKD3->PLN3 layout toggle kernel

template <typename T>
__global__ void convert_pkd3_pln3_hip_tensor(T *srcPtr,
                                            uint2 srcStridesNH,
                                            T *dstPtr,
                                            uint3 dstStridesNCH,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 dst_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

// PLN3->PKD3 layout toggle kernel

template <typename T>
__global__ void convert_pln3_pkd3_hip_tensor(T *srcPtr,
                                             uint3 srcStridesNCH,
                                             T *dstPtr,
                                             uint2 dstStridesNH,
                                             RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 dst_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, srcStridesNCH.y, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

#endif // RPP_HIP_LOAD_STORE_HPP
