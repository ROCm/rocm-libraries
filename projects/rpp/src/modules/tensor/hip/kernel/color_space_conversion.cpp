/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

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

#include "hip_tensor_executors.hpp"

#include <type_traits>

// YUV->RGB: matrix
__constant__ float rpp_nv12_yuv_to_rgb_mat[3][3];
// okluma black level (0 full range, 16 studio) — must match host black in rpp_nv12_set_mat_yuv2rgb
__constant__ int rpp_nv12_y_bias;

namespace {

template <typename T>
__device__ static T rpp_clamp(T x, T lower, T upper)
{
    return x < lower ? lower : (x > upper ? upper : x);
}

// YUV to RGB for one pixel using __constant__ matrix.
template <typename T>
__device__ static void rpp_yuv_to_rgb_pixel(T y, T u, T v, T *r, T *g, T *b)
{
    constexpr int kBits = (int)(sizeof(T) * 8);
    const int mid = 1 << (kBits - 1);
    const float fmax = (float)((1 << kBits) - 1);
    float fy = (float)((int)y - rpp_nv12_y_bias);
    float fu = (int)u - mid;
    float fv = (int)v - mid;
    float fr = rpp_clamp(rpp_nv12_yuv_to_rgb_mat[0][0] * fy + rpp_nv12_yuv_to_rgb_mat[0][1] * fu + rpp_nv12_yuv_to_rgb_mat[0][2] * fv, 0.0f, fmax);
    float fg = rpp_clamp(rpp_nv12_yuv_to_rgb_mat[1][0] * fy + rpp_nv12_yuv_to_rgb_mat[1][1] * fu + rpp_nv12_yuv_to_rgb_mat[1][2] * fv, 0.0f, fmax);
    float fb = rpp_clamp(rpp_nv12_yuv_to_rgb_mat[2][0] * fy + rpp_nv12_yuv_to_rgb_mat[2][1] * fu + rpp_nv12_yuv_to_rgb_mat[2][2] * fv, 0.0f, fmax);
    *r = (T)(fr + 0.5f);
    *g = (T)(fg + 0.5f);
    *b = (T)(fb + 0.5f);
}

// NV12 → packed RGB; T = Rpp8u. Y and UV are separate planes
template <typename T>
__global__ void yuv_to_rgb_hip_kernel(uint8_t *__restrict__ dp_y,
                                      int y_pitch,
                                      uint8_t *__restrict__ dp_uv,
                                      int uv_pitch,
                                      uint8_t *__restrict__ dp_rgb,
                                      int rgb_pitch,
                                      int width,
                                      int height)
{
    constexpr int rgb_pp = (int)(sizeof(T) * 3);  // 3 components per pixel in bytes
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= width || y + 1 >= height)
        return;

    T *p_y = (T *)(dp_y + x * sizeof(T) + y * y_pitch);
    T *p_dst = (T *)(dp_rgb + x * rgb_pp + y * rgb_pitch);
    T *p_dst1 = (T *)((uint8_t *)p_dst + rgb_pitch);

    T y00 = p_y[0];
    T y01 = p_y[1];
    T y10 = p_y[y_pitch / sizeof(T)];
    T y11 = p_y[y_pitch / sizeof(T) + 1];

    T *p_ch = (T *)(dp_uv + (y / 2) * uv_pitch + x * sizeof(T));
    T u = p_ch[0];
    T v = p_ch[1];

    T r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11;
    rpp_yuv_to_rgb_pixel<T>(y00, u, v, &r00, &g00, &b00);
    rpp_yuv_to_rgb_pixel<T>(y01, u, v, &r01, &g01, &b01);
    rpp_yuv_to_rgb_pixel<T>(y10, u, v, &r10, &g10, &b10);
    rpp_yuv_to_rgb_pixel<T>(y11, u, v, &r11, &g11, &b11);

    p_dst[0] = r00; p_dst[1] = g00; p_dst[2] = b00;
    p_dst[3] = r01; p_dst[4] = g01; p_dst[5] = b01;
    p_dst1[0] = r10; p_dst1[1] = g10; p_dst1[2] = b10;
    p_dst1[3] = r11; p_dst1[4] = g11; p_dst1[5] = b11;
}

// NV12 → packed RGB with cubic vertical chroma upsampling.
// One thread per output pixel. Horizontal chroma is nearest-neighbor.
// Uses two alternating 4-tap phases (B=0, C=0.6):
//   Even y (frac=0.75): cr_base = (y>>1) - 2, weights = rpp_cubic_v_weights_even
//   Odd  y (frac=0.25): cr_base = (y>>1) - 1, weights = rpp_cubic_v_weights_odd
template <typename T>
__global__ void yuv_to_rgb_cubic_v_hip_kernel(uint8_t *__restrict__ dp_y,
                                                int y_pitch,
                                                uint8_t *__restrict__ dp_uv,
                                                int uv_pitch,
                                                uint8_t *__restrict__ dp_rgb,
                                                int rgb_pitch,
                                                int width,
                                                int height)
{
    constexpr int rgb_pp = (int)(sizeof(T) * 3);
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height)
        return;

    // Read luma
    T luma = *(T *)(dp_y + y * y_pitch + x * (int)sizeof(T));

    // Horizontal byte offset (nearest-neighbor horizontal)
    int uv_x_offset = (x & ~1) * (int)sizeof(T);
    int chroma_height = height / 2;

    // 4-tap cubic vertical chroma interpolation (B=0, C=0.6).
    // Even y (frac=0.75): base = (y>>1)-2, weights = {-0.028125, 0.240625, 0.871875, -0.084375}
    // Odd  y (frac=0.25): base = (y>>1)-1, weights = {-0.084375, 0.871875, 0.240625, -0.028125}
    int cr_base = (y & 1) ? (y >> 1) - 1 : (y >> 1) - 2;
    int r0 = rpp_clamp(cr_base,     0, chroma_height - 1);
    int r1 = rpp_clamp(cr_base + 1, 0, chroma_height - 1);
    int r2 = rpp_clamp(cr_base + 2, 0, chroma_height - 1);
    int r3 = rpp_clamp(cr_base + 3, 0, chroma_height - 1);
    T *p0 = (T *)(dp_uv + r0 * uv_pitch + uv_x_offset);
    T *p1 = (T *)(dp_uv + r1 * uv_pitch + uv_x_offset);
    T *p2 = (T *)(dp_uv + r2 * uv_pitch + uv_x_offset);
    T *p3 = (T *)(dp_uv + r3 * uv_pitch + uv_x_offset);
    float w0, w1, w2, w3;
    if (y & 1) { w0 = -0.084375f; w1 = 0.871875f; w2 = 0.240625f; w3 = -0.028125f; }
    else       { w0 = -0.028125f; w1 = 0.240625f; w2 = 0.871875f; w3 = -0.084375f; }
    float u_acc = w0 * (float)p0[0] + w1 * (float)p1[0] + w2 * (float)p2[0] + w3 * (float)p3[0];
    float v_acc = w0 * (float)p0[1] + w1 * (float)p1[1] + w2 * (float)p2[1] + w3 * (float)p3[1];
    T u_val = (T)rpp_clamp(u_acc + 0.5f, 0.0f, 255.0f);
    T v_val = (T)rpp_clamp(v_acc + 0.5f, 0.0f, 255.0f);

    // YUV → RGB conversion
    T r_out, g_out, b_out;
    rpp_yuv_to_rgb_pixel<T>(luma, u_val, v_val, &r_out, &g_out, &b_out);

    T *p_dst = (T *)(dp_rgb + y * rgb_pitch + x * rgb_pp);
    p_dst[0] = r_out;
    p_dst[1] = g_out;
    p_dst[2] = b_out;
}

// NV12 → packed RGB with linear vertical chroma upsampling; T = Rpp8u.
// One thread per output pixel. Horizontal chroma is nearest-neighbor.
// Coordinate model: odd luma rows → identity chroma passthrough,
// even luma rows → average of two nearest chroma rows (linear at frac=0.5).
template <typename T>
__global__ void yuv_to_rgb_linear_v_hip_kernel(uint8_t *__restrict__ dp_y,
                                                 int y_pitch,
                                                 uint8_t *__restrict__ dp_uv,
                                                 int uv_pitch,
                                                 uint8_t *__restrict__ dp_rgb,
                                                 int rgb_pitch,
                                                 int width,
                                                 int height)
{
    constexpr int rgb_pp = (int)(sizeof(T) * 3);
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height)
        return;

    // Read luma
    T luma = *(T *)(dp_y + y * y_pitch + x * (int)sizeof(T));

    // Horizontal byte offset (nearest-neighbor horizontal)
    int uv_x_offset = (x & ~1) * (int)sizeof(T);
    int chroma_height = height / 2;

    // Linear vertical chroma interpolation (branch-free).
    // cr0 and cr1 are the two nearest chroma rows: for odd y they are equal
    // (identity passthrough), for even y they differ by 1 (average at frac=0.5).
    int cr0 = rpp_clamp((y - 1) / 2, 0, chroma_height - 1);
    int cr1 = rpp_clamp(y / 2,       0, chroma_height - 1);
    T *p_uv0 = (T *)(dp_uv + cr0 * uv_pitch + uv_x_offset);
    T *p_uv1 = (T *)(dp_uv + cr1 * uv_pitch + uv_x_offset);
    T u_val = (T)(((float)p_uv0[0] + (float)p_uv1[0]) * 0.5f + 0.5f);
    T v_val = (T)(((float)p_uv0[1] + (float)p_uv1[1]) * 0.5f + 0.5f);

    // YUV → RGB conversion
    T r_out, g_out, b_out;
    rpp_yuv_to_rgb_pixel<T>(luma, u_val, v_val, &r_out, &g_out, &b_out);

    T *p_dst = (T *)(dp_rgb + y * rgb_pitch + x * rgb_pp);
    p_dst[0] = r_out;
    p_dst[1] = g_out;
    p_dst[2] = b_out;
}

} // namespace

// Build YUV->RGB 3x3 matrix and copy to device constant
static hipError_t rpp_nv12_set_mat_yuv2rgb(RpptColorStandard col_standard, RpptColorRange color_range)
{
    float wr = 0.2126f, wb = 0.0722f;
    // Luma and chroma have different ranges in studio/TV mode:
    // Y:  [16, 235] → span 219,  Cb/Cr: [16, 240] → span 224
    // Full range uses [0, 255] (span 255) for both.
    int black = 16, luma_range = 219, chroma_range = 224;
    if (color_range == RpptColorRange_FULL) {
        black = 0;
        luma_range = 255;
        chroma_range = 255;
    }
    switch (col_standard)
    {
    case RpptColorStandard_FCC:
        wr = 0.30f;
        wb = 0.11f;
        break;
    case RpptColorStandard_BT470BG:
    case RpptColorStandard_BT601:
        wr = 0.2990f;
        wb = 0.1140f;
        break;
    case RpptColorStandard_SMPTE240M:
        wr = 0.212f;
        wb = 0.087f;
        break;
    case RpptColorStandard_BT2020_NCL:
    case RpptColorStandard_BT2020_CL:
        wr = 0.2627f;
        wb = 0.0593f;
        break;
    case RpptColorStandard_BT709:
    default:
        break;
    }
    float mat[3][3] = {
        {1.0f, 0.0f, (1.0f - wr) / 0.5f},
        {1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr)},
        {1.0f, (1.0f - wb) / 0.5f, 0.0f},
    };
    float y_scale = 255.0f / luma_range;
    float c_scale = 255.0f / chroma_range;
    for (int i = 0; i < 3; i++) {
        mat[i][0] *= y_scale;
        mat[i][1] *= c_scale;
        mat[i][2] *= c_scale;
    }
    hipError_t status = hipMemcpyToSymbol(rpp_nv12_yuv_to_rgb_mat, mat, sizeof(mat));
    if(status != hipSuccess)
        return status;
    return hipMemcpyToSymbol(rpp_nv12_y_bias, &black, sizeof(black));
}

template <typename T>
RppStatus hip_exec_yuv_to_rgb(T *srcYPtr,
                              Rpp32u src_y_pitch,
                              T *srcUVPtr,
                              Rpp32u src_uv_pitch,
                              T *dstPtr,
                              Rpp32u dst_pitch,
                              Rpp32u width,
                              Rpp32u height,
                              RpptColorStandard col_standard,
                              RpptColorRange color_range,
                              rpp::Handle &handle)
{
    static_assert(sizeof(T) == 1 && std::is_same<typename std::remove_cv<T>::type, Rpp8u>::value,
                  "hip_exec_yuv_to_rgb is only supported for Rpp8u (NV12 8-bit)");
    RPP_HIP_RETURN_IF_ERROR(rpp_nv12_set_mat_yuv2rgb(col_standard, color_range));
    hipLaunchKernelGGL(yuv_to_rgb_hip_kernel<T>,
                       dim3((width + 63) / 32 / 2, (height + 3) / 2 / 2, 1),
                       dim3(32, 2, 1),
                       0,
                       handle.GetStream(),
                       (uint8_t *)srcYPtr,
                       (int)src_y_pitch,
                       (uint8_t *)srcUVPtr,
                       (int)src_uv_pitch,
                       (uint8_t *)dstPtr,
                       (int)dst_pitch,
                       (int)width,
                       (int)height);
    return RPP_SUCCESS;
}

template RppStatus hip_exec_yuv_to_rgb<Rpp8u>(Rpp8u *srcYPtr,
                                              Rpp32u src_y_pitch,
                                              Rpp8u *srcUVPtr,
                                              Rpp32u src_uv_pitch,
                                              Rpp8u *dstPtr,
                                              Rpp32u dst_pitch,
                                              Rpp32u width,
                                              Rpp32u height,
                                              RpptColorStandard col_standard,
                                              RpptColorRange color_range,
                                              rpp::Handle &handle);

template <typename T>
RppStatus hip_exec_yuv_to_rgb_cubic_v(T *srcYPtr,
                                        Rpp32u src_y_pitch,
                                        T *srcUVPtr,
                                        Rpp32u src_uv_pitch,
                                        T *dstPtr,
                                        Rpp32u dst_pitch,
                                        Rpp32u width,
                                        Rpp32u height,
                                        RpptColorStandard col_standard,
                                        RpptColorRange color_range,
                                        rpp::Handle &handle)
{
    static_assert(sizeof(T) == 1 && std::is_same<typename std::remove_cv<T>::type, Rpp8u>::value,
                  "hip_exec_yuv_to_rgb_cubic_v is only supported for Rpp8u (NV12 8-bit)");
    RPP_HIP_RETURN_IF_ERROR(rpp_nv12_set_mat_yuv2rgb(col_standard, color_range));
    hipLaunchKernelGGL(yuv_to_rgb_cubic_v_hip_kernel<T>,
                       dim3((width + 31) / 32, (height + 7) / 8, 1),
                       dim3(32, 8, 1),
                       0,
                       handle.GetStream(),
                       (uint8_t *)srcYPtr,
                       (int)src_y_pitch,
                       (uint8_t *)srcUVPtr,
                       (int)src_uv_pitch,
                       (uint8_t *)dstPtr,
                       (int)dst_pitch,
                       (int)width,
                       (int)height);
    return RPP_SUCCESS;
}

template RppStatus hip_exec_yuv_to_rgb_cubic_v<Rpp8u>(Rpp8u *srcYPtr,
                                                        Rpp32u src_y_pitch,
                                                        Rpp8u *srcUVPtr,
                                                        Rpp32u src_uv_pitch,
                                                        Rpp8u *dstPtr,
                                                        Rpp32u dst_pitch,
                                                        Rpp32u width,
                                                        Rpp32u height,
                                                        RpptColorStandard col_standard,
                                                        RpptColorRange color_range,
                                                        rpp::Handle &handle);

template <typename T>
RppStatus hip_exec_yuv_to_rgb_linear_v(T *srcYPtr,
                                         Rpp32u src_y_pitch,
                                         T *srcUVPtr,
                                         Rpp32u src_uv_pitch,
                                         T *dstPtr,
                                         Rpp32u dst_pitch,
                                         Rpp32u width,
                                         Rpp32u height,
                                         RpptColorStandard col_standard,
                                         RpptColorRange color_range,
                                         rpp::Handle &handle)
{
    static_assert(sizeof(T) == 1 && std::is_same<typename std::remove_cv<T>::type, Rpp8u>::value,
                  "hip_exec_yuv_to_rgb_linear_v is only supported for Rpp8u (NV12 8-bit)");
    RPP_HIP_RETURN_IF_ERROR(rpp_nv12_set_mat_yuv2rgb(col_standard, color_range));
    hipLaunchKernelGGL(yuv_to_rgb_linear_v_hip_kernel<T>,
                       dim3((width + 31) / 32, (height + 7) / 8, 1),
                       dim3(32, 8, 1),
                       0,
                       handle.GetStream(),
                       (uint8_t *)srcYPtr,
                       (int)src_y_pitch,
                       (uint8_t *)srcUVPtr,
                       (int)src_uv_pitch,
                       (uint8_t *)dstPtr,
                       (int)dst_pitch,
                       (int)width,
                       (int)height);
    return RPP_SUCCESS;
}

template RppStatus hip_exec_yuv_to_rgb_linear_v<Rpp8u>(Rpp8u *srcYPtr,
                                                          Rpp32u src_y_pitch,
                                                          Rpp8u *srcUVPtr,
                                                          Rpp32u src_uv_pitch,
                                                          Rpp8u *dstPtr,
                                                          Rpp32u dst_pitch,
                                                          Rpp32u width,
                                                          Rpp32u height,
                                                          RpptColorStandard col_standard,
                                                          RpptColorRange color_range,
                                                          rpp::Handle &handle);
                                                          
