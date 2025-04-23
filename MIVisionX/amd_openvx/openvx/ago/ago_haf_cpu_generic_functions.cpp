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

#include <VX/vx.h>
#include <VX/vxu.h>
#include "ago_internal.h"

/*! \brief The largest nonlinear filter matrix the specification requires support for is 9x9.
*/
#define C_MAX_NONLINEAR_DIM (9)

int HafCpu_WeightedAverage_U8_U8U8
    (
        vx_image img1, 
        vx_float32 alpha, 
        vx_image img2, 
        vx_image output
    )
{
    vx_uint32 y, x, width = 0, height = 0;
    void *dst_base = NULL;
    void *src_base[2] = { NULL, NULL };
    vx_imagepatch_addressing_t dst_addr, src_addr[2];
    vx_rectangle_t rect;
    vx_df_image img1_format = 0;
    vx_df_image img2_format = 0;
    vx_df_image out_format = 0;
    vx_status status = VX_SUCCESS;
    vx_map_id src_map_id[2];
    vx_map_id dst_map_id;
    vxQueryImage(output, VX_IMAGE_FORMAT, &out_format, sizeof(out_format));
    vxQueryImage(img1, VX_IMAGE_FORMAT, &img1_format, sizeof(img1_format));
    vxQueryImage(img2, VX_IMAGE_FORMAT, &img2_format, sizeof(img2_format));

    status = vxGetValidRegionImage(img1, &rect);
    status |= vxMapImagePatch(img1, &rect, 0, &src_map_id[0], &src_addr[0], (void **)&src_base[0],
                              VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    status |= vxMapImagePatch(img2, &rect, 0, &src_map_id[1], &src_addr[1], (void **)&src_base[1],
                              VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    status |= vxMapImagePatch(output, &rect, 0, &dst_map_id, &dst_addr, (void **)&dst_base,
                              VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    width = src_addr[0].dim_x;
    height = src_addr[0].dim_y;
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            void *src0p = vxFormatImagePatchAddress2d(src_base[0], x, y, &src_addr[0]);
            void *src1p = vxFormatImagePatchAddress2d(src_base[1], x, y, &src_addr[1]);
            void *dstp = vxFormatImagePatchAddress2d(dst_base, x, y, &dst_addr);
            vx_int32 src0 = *(vx_uint8 *)src0p;
            vx_int32 src1 = *(vx_uint8 *)src1p;
            vx_int32 result = (vx_int32)((1 - alpha) * (vx_float32)src1 + alpha * (vx_float32)src0);
            *(vx_uint8 *)dstp = (vx_uint8)result;
        }
    }
    status |= vxUnmapImagePatch(img1, src_map_id[0]);
    status |= vxUnmapImagePatch(img2, src_map_id[1]);
    status |= vxUnmapImagePatch(output, dst_map_id);
    return status;
}

// helpers
static int vx_uint8_compare(const void *p1, const void *p2)
{
    vx_uint8 a = *(vx_uint8 *)p1;
    vx_uint8 b = *(vx_uint8 *)p2;
    if (a > b)
        return 1;
    else if (a == b)
        return 0;
    else
        return -1;
}

static vx_uint32 readMaskedRectangle(const void *base,
    const vx_imagepatch_addressing_t *addr,
    const vx_border_t *borders,
    vx_df_image type,
    vx_uint32 center_x,
    vx_uint32 center_y,
    vx_uint32 left,
    vx_uint32 top,
    vx_uint32 right,
    vx_uint32 bottom,
    vx_uint8 *mask,
    vx_uint8 *destination,
    vx_uint32 border_x_start)
{
    vx_int32 width = (vx_int32)addr->dim_x, height = (vx_int32)addr->dim_y;
    vx_int32 stride_y = addr->stride_y;
    vx_int32 stride_x = addr->stride_x;
    vx_uint16 stride_x_bits = addr->stride_x_bits;
    const vx_uint8 *ptr = (const vx_uint8 *)base;
    vx_int32 ky, kx;
    vx_uint32 mask_index = 0;
    vx_uint32 dest_index = 0;

    // kx, ky - kernel x and y
    if (borders->mode == VX_BORDER_REPLICATE || borders->mode == VX_BORDER_UNDEFINED)
    {
        for (ky = -(int32_t)top; ky <= (int32_t)bottom; ++ky)
        {
            vx_int32 y = (vx_int32)(center_y + ky);
            y = y < 0 ? 0 : y >= height ? height - 1 : y;

            for (kx = -(int32_t)left; kx <= (int32_t)right; ++kx, ++mask_index)
            {
                vx_int32 x = (int32_t)(center_x + kx);
                x = x < (int32_t)border_x_start ? (int32_t)border_x_start : x >= width ? width - 1 : x;
                if (mask[mask_index])
                {
                    if (type == VX_DF_IMAGE_U1)
                        ((vx_uint8*)destination)[dest_index++] =
                            ( *(vx_uint8*)(ptr + y*stride_y + (x*stride_x_bits) / 8) & (1 << (x % 8)) ) >> (x % 8);
                    else    // VX_DF_IMAGE_U8
                        ((vx_uint8*)destination)[dest_index++] = *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                }
            }
        }
    }
    else if (borders->mode == VX_BORDER_CONSTANT)
    {
        vx_pixel_value_t cval = borders->constant_value;
        for (ky = -(int32_t)top; ky <= (int32_t)bottom; ++ky)
        {
            vx_int32 y = (vx_int32)(center_y + ky);
            int ccase_y = y < 0 || y >= height;

            for (kx = -(int32_t)left; kx <= (int32_t)right; ++kx, ++mask_index)
            {
                if (mask[mask_index])
                {
                    vx_int32 x = (int32_t)(center_x + kx);
                    int ccase = ccase_y || x < (int32_t)border_x_start || x >= width;
                    if (type == VX_DF_IMAGE_U1)
                        ((vx_uint8*)destination)[dest_index++] = ccase ? ( (vx_uint8)cval.U1 ? 1 : 0 ) :
                            ( *(vx_uint8*)(ptr + y*stride_y + (x*stride_x_bits) / 8) & (1 << (x % 8)) ) >> (x % 8);
                    else    // VX_DF_IMAGE_U8
                        ((vx_uint8*)destination)[dest_index++] = ccase ? (vx_uint8)cval.U8 : *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                }
            }
        }
    }

    return dest_index;
}

vx_status vxAlterRectangle(vx_rectangle_t *rect,
                           vx_int32 dsx,
                           vx_int32 dsy,
                           vx_int32 dex,
                           vx_int32 dey)
{
    if (rect)
    {
        rect->start_x += dsx;
        rect->start_y += dsy;
        rect->end_x += dex;
        rect->end_y += dey;
        return VX_SUCCESS;
    }
    return VX_ERROR_INVALID_REFERENCE;
}

// nodeless version of NonLinearFilter kernel
int HafCpu_NonLinearFilter_DATA_DATADATA
    (
        vx_int32 function,     
        vx_image src, 
        vx_matrix mask, 
        vx_image dst, 
        vx_border_t *border
    )
{
    vx_uint32 y, x;
    void *src_base = NULL;
    void *dst_base = NULL;
    vx_df_image format = 0;
    vx_imagepatch_addressing_t src_addr, dst_addr;
    vx_rectangle_t rect;
    vx_uint32 low_x = 0, low_y = 0, high_x, high_y, shift_x_u1;

    vx_uint8 m[C_MAX_NONLINEAR_DIM * C_MAX_NONLINEAR_DIM];
    vx_uint8 v[C_MAX_NONLINEAR_DIM * C_MAX_NONLINEAR_DIM];
    vx_uint8 res_val = 0;

    vx_status status = vxGetValidRegionImage(src, &rect);
    status |= vxQueryImage(src, VX_IMAGE_FORMAT, &format, sizeof(format));
    status |= vxAccessImagePatch(src, &rect, 0, &src_addr, &src_base, VX_READ_ONLY);
    status |= vxAccessImagePatch(dst, &rect, 0, &dst_addr, &dst_base, VX_WRITE_ONLY);

    vx_enum func = function;

    vx_size mrows, mcols;
    vx_enum mtype = 0;
    status |= vxQueryMatrix(mask, VX_MATRIX_ROWS, &mrows, sizeof(mrows));
    status |= vxQueryMatrix(mask, VX_MATRIX_COLUMNS, &mcols, sizeof(mcols));
    status |= vxQueryMatrix(mask, VX_MATRIX_TYPE, &mtype, sizeof(mtype));

    vx_coordinates2d_t origin;
    status |= vxQueryMatrix(mask, VX_MATRIX_ORIGIN, &origin, sizeof(origin));

    if ((mtype != VX_TYPE_UINT8) || (sizeof(m) < mrows * mcols))
        status = VX_ERROR_INVALID_PARAMETERS;

    status |= vxCopyMatrix(mask, m, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if (status == VX_SUCCESS)
    {
        vx_size rx0 = origin.x;
        vx_size ry0 = origin.y;
        vx_size rx1 = mcols - origin.x - 1;
        vx_size ry1 = mrows - origin.y - 1;

        shift_x_u1 = (format == VX_DF_IMAGE_U1) ? rect.start_x % 8 : 0;
        high_x = src_addr.dim_x - shift_x_u1;   // U1 addressing rounds down imagepatch start_x to nearest byte boundary
        high_y = src_addr.dim_y;

        if (border->mode == VX_BORDER_UNDEFINED)
        {
            low_x  += (vx_uint32)rx0;
            low_y  += (vx_uint32)ry0;
            high_x -= (vx_uint32)rx1;
            high_y -= (vx_uint32)ry1;
            vxAlterRectangle(&rect, (vx_int32)rx0, (vx_int32)ry0, -(vx_int32)rx1, -(vx_int32)ry1);
        }

        for (y = low_y; y < high_y; y++)
        {
            for (x = low_x; x < high_x; x++)
            {
                vx_uint32 xShftd = x + shift_x_u1;      // Bit-shift for U1 valid region start
                vx_uint8 *dst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(dst_base, xShftd, y, &dst_addr);
                vx_int32 count = (vx_int32)readMaskedRectangle(src_base, &src_addr, border, format, xShftd, y, (vx_uint32)rx0, (vx_uint32)ry0, (vx_uint32)rx1, (vx_uint32)ry1, m, v, shift_x_u1);

                qsort(v, count, sizeof(vx_uint8), vx_uint8_compare);

                switch (func)
                {
                case VX_NONLINEAR_FILTER_MIN:    res_val = v[0];         break; /* minimal value */
                case VX_NONLINEAR_FILTER_MAX:    res_val = v[count - 1]; break; /* maximum value */
                case VX_NONLINEAR_FILTER_MEDIAN: res_val = v[count / 2]; break; /* pick the middle value */
                }
                if (format == VX_DF_IMAGE_U1)
                {
                    *dst_ptr = (*dst_ptr & ~(1 << (xShftd % 8))) | (res_val << (xShftd % 8));
                }
                else
                    *dst_ptr = res_val;
            }
        }
    }

    status |= vxCommitImagePatch(src, NULL, 0, &src_addr, src_base);
    status |= vxCommitImagePatch(dst, &rect, 0, &dst_addr, dst_base);
    
    return status;
}

#define C_MAX_CONVOLUTION_DIM (15)

static const vx_int16 gaussian5x5[5][5] =
{
    {1,  4,  6,  4, 1},
    {4, 16, 24, 16, 4},
    {6, 24, 36, 24, 6},
    {4, 16, 24, 16, 4},
    {1,  4,  6,  4, 1}
};

static const vx_uint32 gaussian5x5scale = 256;

static vx_convolution vxCreateGaussian5x5Convolution(vx_context context)
{
    vx_convolution conv = vxCreateConvolution(context, 5, 5);
    vx_status status = vxCopyConvolutionCoefficients(conv, (vx_int16 *)gaussian5x5, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if (status != VX_SUCCESS)
    {
        vxReleaseConvolution(&conv);
        return NULL;
    }

    status = vxSetConvolutionAttribute(conv, VX_CONVOLUTION_SCALE, (void *)&gaussian5x5scale, sizeof(vx_uint32));
    if (status != VX_SUCCESS)
    {
        vxReleaseConvolution(&conv);
        return NULL;
    }
    return conv;
}

static vx_status ownCopyImage(vx_image input, vx_image output)
{
    vx_status status = VX_SUCCESS; // assume success until an error occurs.
    vx_uint32 p = 0;
    vx_uint32 y = 0, x = 0;
    vx_size planes = 0;

    void* src;
    void* dst;
    vx_imagepatch_addressing_t src_addr;
    vx_imagepatch_addressing_t dst_addr;
    vx_rectangle_t src_rect, dst_rect;
    vx_map_id map_id1;
    vx_map_id map_id2;
    vx_df_image src_format = 0;
    vx_df_image out_format = 0;

    status |= vxQueryImage(input, VX_IMAGE_PLANES, &planes, sizeof(planes));
    vxQueryImage(output, VX_IMAGE_FORMAT, &out_format, sizeof(out_format));
    vxQueryImage(input, VX_IMAGE_FORMAT, &src_format, sizeof(src_format));
    status |= vxGetValidRegionImage(input, &src_rect);
    status |= vxGetValidRegionImage(output, &dst_rect);
    for (p = 0; p < planes && status == VX_SUCCESS; p++)
    {
        status = VX_SUCCESS;
        src = NULL;
        dst = NULL;

        status |= vxMapImagePatch(input, &src_rect, p, &map_id1, &src_addr, &src, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
        status |= vxMapImagePatch(output, &dst_rect, p, &map_id2, &dst_addr, &dst, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
        for (y = 0; y < src_addr.dim_y && status == VX_SUCCESS; y += src_addr.step_y)
        {
            for (x = 0; x < src_addr.dim_x && status == VX_SUCCESS; x += src_addr.step_x)
            {
                void* srcp = vxFormatImagePatchAddress2d(src, x, y, &src_addr);
                void* dstp = vxFormatImagePatchAddress2d(dst, x, y, &dst_addr);
                vx_int32 out0 = src_format == VX_DF_IMAGE_U8 ? *(vx_uint8 *)srcp : *(vx_int16 *)srcp;

                if (out_format == VX_DF_IMAGE_U8)
                {
                    if (out0 > UINT8_MAX)
                        out0 = UINT8_MAX;
                    else if (out0 < 0)
                        out0 = 0;
                    *(vx_uint8 *)dstp = (vx_uint8)out0;
                }
                else
                {
                    if (out0 > INT16_MAX)
                        out0 = INT16_MAX;
                    else if (out0 < INT16_MIN)
                        out0 = INT16_MIN;
                    *(vx_int16 *)dstp = (vx_int16)out0;
                }
            }
        }

        if (status == VX_SUCCESS)
        {
            status |= vxUnmapImagePatch(input, map_id1);
            status |= vxUnmapImagePatch(output, map_id2);
        }
    }

    return status;
}

static void auxReadRect(const void *base, const vx_imagepatch_addressing_t *addr, const vx_border_t *borders, vx_df_image type,
    vx_uint32 center_x, vx_uint32 center_y, vx_uint32 radius_x, vx_uint32 radius_y, void *destination)
{
    vx_int32 width = (vx_int32)addr->dim_x, height = (vx_int32)addr->dim_y;
    vx_int32 stride_y = addr->stride_y;
    vx_int32 stride_x = addr->stride_x;
    const vx_uint8 *ptr = (const vx_uint8 *)base;
    vx_int32 ky, kx;
    vx_uint32 dest_index = 0;
    // kx, kx - kernel x and y
    if (borders->mode == VX_BORDER_REPLICATE || borders->mode == VX_BORDER_UNDEFINED)
    {
        for (ky = -(int32_t)radius_y; ky <= (int32_t)radius_y; ++ky)
        {
            vx_int32 y = (vx_int32)(center_y + ky);
            y = y < 0 ? 0 : (y >= height ? height - 1 : y);

            for (kx = -(int32_t)radius_x; kx <= (int32_t)radius_x; ++kx, ++dest_index)
            {
                vx_int32 x = (int32_t)(center_x + kx);
                x = x < 0 ? 0 : (x >= width ? width - 1 : x);

                switch (type)
                {
                case VX_DF_IMAGE_U8:
                    ((vx_uint8*)destination)[dest_index] = *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                    break;
                case VX_DF_IMAGE_S16:
                case VX_DF_IMAGE_U16:
                    ((vx_uint16*)destination)[dest_index] = *(vx_uint16*)(ptr + y*stride_y + x*stride_x);
                    break;
                case VX_DF_IMAGE_S32:
                case VX_DF_IMAGE_U32:
                    ((vx_uint32*)destination)[dest_index] = *(vx_uint32*)(ptr + y*stride_y + x*stride_x);
                    break;
                default:
                    abort();
                }
            }
        }
    }
    else if (borders->mode == VX_BORDER_CONSTANT)
    {
        vx_pixel_value_t cval = borders->constant_value;
        for (ky = -(int32_t)radius_y; ky <= (int32_t)radius_y; ++ky)
        {
            vx_int32 y = (vx_int32)(center_y + ky);
            int ccase_y = y < 0 || y >= height;

            for (kx = -(int32_t)radius_x; kx <= (int32_t)radius_x; ++kx, ++dest_index)
            {
                vx_int32 x = (int32_t)(center_x + kx);
                int ccase = ccase_y || x < 0 || x >= width;

                switch (type)
                {
                case VX_DF_IMAGE_U8:
                    if (!ccase)
                        ((vx_uint8*)destination)[dest_index] = *(vx_uint8*)(ptr + y*stride_y + x*stride_x);
                    else
                        ((vx_uint8*)destination)[dest_index] = (vx_uint8)cval.U8;
                    break;
                case VX_DF_IMAGE_S16:
                case VX_DF_IMAGE_U16:
                    if (!ccase)
                        ((vx_uint16*)destination)[dest_index] = *(vx_uint16*)(ptr + y*stride_y + x*stride_x);
                    else
                        ((vx_uint16*)destination)[dest_index] = (vx_uint16)cval.U16;
                    break;
                case VX_DF_IMAGE_S32:
                case VX_DF_IMAGE_U32:
                    if (!ccase)
                        ((vx_uint32*)destination)[dest_index] = *(vx_uint32*)(ptr + y*stride_y + x*stride_x);
                    else
                        ((vx_uint32*)destination)[dest_index] = (vx_uint32)cval.U32;
                    break;
                default:
                    abort();
                }
            }
        }
    }
    else
        abort();
}

#define CONV_DIM 5
#define CONV_DIM_HALF CONV_DIM / 2

#define INSERT_ZERO_Y(slice, y) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*(1-y)+i] = 0;
#define INSERT_VALUES_Y(slice, y) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*(high_y-y)+i+CONV_DIM_HALF*CONV_DIM] = slice[CONV_DIM*(high_y-y)+i];
#define INSERT_ZERO_X(slice, x) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*i+1-x] = 0;
#define INSERT_VALUES_X(slice, x) for (int i=0; i<CONV_DIM; i++) slice[CONV_DIM*i+(high_x-x)+CONV_DIM_HALF] = slice[CONV_DIM*i+(high_x-x)];

static vx_status replicateConvolve(vx_image src, vx_convolution conv, vx_image dst, vx_border_t *bordermode)
{
    vx_int32 y, x, i;
    void *src_base = NULL;
    void *dst_base = NULL;
    vx_imagepatch_addressing_t src_addr, dst_addr;
    vx_rectangle_t rect;

    vx_size conv_width, conv_height;
    vx_int32 conv_radius_x, conv_radius_y;
    vx_int16 conv_mat[C_MAX_CONVOLUTION_DIM * C_MAX_CONVOLUTION_DIM] = { 0 };
    vx_int32 sum = 0, value = 0;
    vx_uint32 scale = 1;
    vx_df_image src_format = 0;
    vx_df_image dst_format = 0;
    vx_status status = VX_SUCCESS;
    vx_int32 low_x, low_y, high_x, high_y;

    status |= vxQueryImage(src, VX_IMAGE_FORMAT, &src_format, sizeof(src_format));
    status |= vxQueryImage(dst, VX_IMAGE_FORMAT, &dst_format, sizeof(dst_format));
    status |= vxQueryConvolution(conv, VX_CONVOLUTION_COLUMNS, &conv_width, sizeof(conv_width));
    status |= vxQueryConvolution(conv, VX_CONVOLUTION_ROWS, &conv_height, sizeof(conv_height));
    status |= vxQueryConvolution(conv, VX_CONVOLUTION_SCALE, &scale, sizeof(scale));
    conv_radius_x = (vx_int32)conv_width / 2;
    conv_radius_y = (vx_int32)conv_height / 2;
    status |= vxCopyConvolutionCoefficients(conv, conv_mat, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxGetValidRegionImage(src, &rect);
    status |= vxAccessImagePatch(src, &rect, 0, &src_addr, &src_base, VX_READ_ONLY);
    status |= vxAccessImagePatch(dst, &rect, 0, &dst_addr, &dst_base, VX_WRITE_ONLY);

    low_x = 0;
    high_x = src_addr.dim_x;
    low_y = 0;
    high_y = src_addr.dim_y;

    for (y = low_y; y < high_y; ++y)
    {
        for (x = low_x; x < high_x; ++x)
        {
            sum = 0;

            if (src_format == VX_DF_IMAGE_U8)
            {
                vx_uint8 slice[C_MAX_CONVOLUTION_DIM * C_MAX_CONVOLUTION_DIM] = { 0 };

                auxReadRect(src_base, &src_addr, bordermode, src_format, x, y, conv_radius_x, conv_radius_y, slice);

                // purpose of this section is to compensate extra terms caused by replicate border mode (it is the only one allowed)

                if (y < CONV_DIM_HALF)
                {
                    INSERT_ZERO_Y(slice, y)
                }
                else if (y >= high_y - CONV_DIM_HALF)
                {
                    INSERT_VALUES_Y(slice, y)
                }

                if (x < CONV_DIM_HALF)
                {
                    INSERT_ZERO_X(slice, x)
                }
                else if (x >= high_x - CONV_DIM_HALF)
                {
                    INSERT_VALUES_X(slice, x)
                }

                for (i = 0; i < (vx_int32)(conv_width * conv_height); ++i)
                    sum += conv_mat[conv_width * conv_height - 1 - i] * slice[i];
            }
            else if (src_format == VX_DF_IMAGE_S16)
            {
                vx_int16 slice[C_MAX_CONVOLUTION_DIM * C_MAX_CONVOLUTION_DIM] = { 0 };

                auxReadRect(src_base, &src_addr, bordermode, src_format, x, y, conv_radius_x, conv_radius_y, slice);

                // purpose of this section is to compensate extra terms caused by replicate border mode (it is the only one allowed)

                if (y < CONV_DIM_HALF)
                {
                    INSERT_ZERO_Y(slice, y)
                }
                else if (y >= high_y - CONV_DIM_HALF)
                {
                    INSERT_VALUES_Y(slice, y)
                }

                if (x < CONV_DIM_HALF)
                {
                    INSERT_ZERO_X(slice, x)
                }
                else if (x >= high_x - CONV_DIM_HALF)
                {
                    INSERT_VALUES_X(slice, x)
                }

                for (i = 0; i < (vx_int32)(conv_width * conv_height); ++i)
                    sum += conv_mat[conv_width * conv_height - 1 - i] * slice[i];
            }

            value = sum / (vx_int32)scale;

            if (dst_format == VX_DF_IMAGE_U8)
            {
                vx_uint8 *dstp = (vx_uint8 *)vxFormatImagePatchAddress2d(dst_base, x, y, &dst_addr);
                if (value < 0) *dstp = 0;
                else if (value > UINT8_MAX) *dstp = UINT8_MAX;
                else *dstp = value;
            }
            else if (dst_format == VX_DF_IMAGE_S16)
            {
                vx_int16 *dstp = (vx_int16 *)vxFormatImagePatchAddress2d(dst_base, x, y, &dst_addr);
                if (value < INT16_MIN) *dstp = INT16_MIN;
                else if (value > INT16_MAX) *dstp = INT16_MAX;
                else *dstp = value;
            }
        }
    }

    status |= vxCommitImagePatch(src, NULL, 0, &src_addr, src_base);
    status |= vxCommitImagePatch(dst, &rect, 0, &dst_addr, dst_base);

    return status;
}

static vx_status upsampleImage(vx_context context, vx_uint32 width, vx_uint32 height, vx_image filling, vx_convolution conv, vx_image upsample, vx_border_t *border)
{
    vx_status status = VX_SUCCESS;
    vx_df_image format, filling_format;

    format = VX_DF_IMAGE_U8;
    vx_image tmp = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    status |= vxQueryImage(filling, VX_IMAGE_FORMAT, &filling_format, sizeof(filling_format));

    vx_rectangle_t tmp_rect, filling_rect;
    vx_imagepatch_addressing_t tmp_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_imagepatch_addressing_t filling_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_map_id tmp_map_id, filling_map_id;
    void *tmp_base = NULL;
    void *filling_base = NULL;

    status = vxGetValidRegionImage(tmp, &tmp_rect);
    status |= vxMapImagePatch(tmp, &tmp_rect, 0, &tmp_map_id, &tmp_addr, (void **)&tmp_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
    status = vxGetValidRegionImage(filling, &filling_rect);
    status |= vxMapImagePatch(filling, &filling_rect, 0, &filling_map_id, &filling_addr, (void **)&filling_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);

    for (int ix = 0; ix < (int)width; ix++)
    {
        for (int iy = 0; iy < (int)height; iy++)
        {

            void* tmp_datap = vxFormatImagePatchAddress2d(tmp_base, ix, iy, &tmp_addr);

            if (iy % 2 != 0 || ix % 2 != 0)
            {
                if (format == VX_DF_IMAGE_U8)
                    *(vx_uint8 *)tmp_datap = (vx_uint8)0;
                else
                    *(vx_int16 *)tmp_datap = (vx_int16)0;
            }
            else
            {
                void* filling_tmp = vxFormatImagePatchAddress2d(filling_base, ix / 2, iy / 2, &filling_addr);
                vx_int32 filling_data = filling_format == VX_DF_IMAGE_U8 ? *(vx_uint8 *)filling_tmp : *(vx_int16 *)filling_tmp;
                if (format == VX_DF_IMAGE_U8)
                {
                    if (filling_data > UINT8_MAX)
                        filling_data = UINT8_MAX;
                    else if (filling_data < 0)
                        filling_data = 0;
                    *(vx_uint8 *)tmp_datap = (vx_uint8)filling_data;
                }
                else
                {
                    if (filling_data > INT16_MAX)
                        filling_data = INT16_MAX;
                    else if (filling_data < INT16_MIN)
                        filling_data = INT16_MIN;
                    *(vx_int16 *)tmp_datap = (vx_int16)filling_data;
                }
            }
        }
    }

    status |= vxUnmapImagePatch(tmp, tmp_map_id);
    status |= vxUnmapImagePatch(filling, filling_map_id);

    status |=replicateConvolve(tmp, conv, upsample, border);

    vx_rectangle_t upsample_rect;
    vx_imagepatch_addressing_t upsample_addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_map_id upsample_map_id;
    void * upsample_base = NULL;
    vx_df_image upsample_format;

    status |= vxQueryImage(upsample, VX_IMAGE_FORMAT, &upsample_format, sizeof(upsample_format));
    status = vxGetValidRegionImage(upsample, &upsample_rect);
    status |= vxMapImagePatch(upsample, &upsample_rect, 0, &upsample_map_id, &upsample_addr, (void **)&upsample_base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);

    for (int ix = 0; ix < (int)width; ix++)
    {
        for (int iy = 0; iy < (int)height; iy++)
        {
            void* upsample_p = vxFormatImagePatchAddress2d(upsample_base, ix, iy, &upsample_addr);
            vx_int32 upsample_data = upsample_format == VX_DF_IMAGE_U8 ? *(vx_uint8 *)upsample_p : *(vx_int16 *)upsample_p;
            upsample_data *= 4;
            if (upsample_format == VX_DF_IMAGE_U8)
            {
                if (upsample_data > UINT8_MAX)
                    upsample_data = UINT8_MAX;
                else if (upsample_data < 0)
                    upsample_data = 0;
                *(vx_uint8 *)upsample_p = (vx_uint8)upsample_data;
            }
            else
            {
                if (upsample_data > INT16_MAX)
                    upsample_data = INT16_MAX;
                else if (upsample_data < INT16_MIN)
                    upsample_data = INT16_MIN;
                *(vx_int16 *)upsample_p = (vx_int16)upsample_data;
            }
        }
    }
    status |= vxUnmapImagePatch(upsample, upsample_map_id);
    status |= vxReleaseImage(&tmp);
    return status;
}

int HafCpu_LaplacianPyramid_DATA_DATA_DATA
    (
        vx_node node,
        vx_image input, 
        vx_pyramid laplacian,
        vx_image output
    )
{
    vx_status status = VX_SUCCESS;
    
    vx_context context = vxGetContext((vx_reference)node);
    
    vx_size lev;
    vx_size levels = 1;
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    vx_uint32 level_width = 0;
    vx_uint32 level_height = 0;
    vx_df_image format;
    vx_enum policy = VX_CONVERT_POLICY_SATURATE;
    vx_border_t border;
    vx_convolution conv = 0;
    vx_image pyr_gauss_curr_level_filtered = 0;
    vx_image pyr_laplacian_curr_level = 0;
    vx_pyramid gaussian = 0;
    vx_image gauss_cur = 0;
    vx_image gauss_next = 0;

    status |= vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width));
    status |= vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height));
    status |= vxQueryImage(input, VX_IMAGE_FORMAT, &format, sizeof(format));

    status |= vxQueryPyramid(laplacian, VX_PYRAMID_LEVELS, &levels, sizeof(levels));

    status |= vxQueryNode(node, VX_NODE_BORDER, &border, sizeof(border));
    
    border.mode = VX_BORDER_REPLICATE;

    vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border, sizeof(border));

    gaussian = vxCreatePyramid(context, levels + 1, VX_SCALE_PYRAMID_HALF, width, height, VX_DF_IMAGE_U8);
    vxuGaussianPyramid(context, input, gaussian);

    conv = vxCreateGaussian5x5Convolution(context);

    level_width = width;
    level_height = height;
    gauss_cur = vxGetPyramidLevel(gaussian, 0);
    gauss_next = vxGetPyramidLevel(gaussian, 1);
    for (lev = 0; lev < levels; lev++)
    {
        pyr_gauss_curr_level_filtered = vxCreateImage(context, level_width, level_height, VX_DF_IMAGE_S16);
        upsampleImage(context, level_width, level_height, gauss_next, conv, pyr_gauss_curr_level_filtered, &border);

        pyr_laplacian_curr_level = vxGetPyramidLevel(laplacian, (vx_uint32)lev);
        status |= vxuSubtract(context, gauss_cur, pyr_gauss_curr_level_filtered, policy, pyr_laplacian_curr_level);

        if (lev == levels - 1)
        {
            vx_image tmp = vxGetPyramidLevel(gaussian, (vx_uint32) levels);
            ownCopyImage(tmp, output);
            vxReleaseImage(&tmp);
            vxReleaseImage(&gauss_next);
            vxReleaseImage(&gauss_cur);
        }
        else
        {
            /* compute dimensions for the next level */
            level_width = (vx_uint32)ceilf(level_width * VX_SCALE_PYRAMID_HALF);
            level_height = (vx_uint32)ceilf(level_height * VX_SCALE_PYRAMID_HALF);
            /* prepare to the next iteration */
            /* make the next level of gaussian pyramid the current level */
            vxReleaseImage(&gauss_next);
            vxReleaseImage(&gauss_cur);
            gauss_cur = vxGetPyramidLevel(gaussian, (vx_uint32)lev + 1);
            gauss_next = vxGetPyramidLevel(gaussian, (vx_uint32)lev + 2);

        }

        /* decrements the references */

        status |= vxReleaseImage(&pyr_gauss_curr_level_filtered);
        status |= vxReleaseImage(&pyr_laplacian_curr_level);
    }

    status |= vxReleasePyramid(&gaussian);
    status |= vxReleaseConvolution(&conv);

    return status;
}

#define VX_SCALE_PYRAMID_DOUBLE (2.0f)

int HafCpu_LaplacianReconstruct_DATA_DATA_DATA
    (
        vx_node node, 
        vx_pyramid laplacian,
        vx_image input,
        vx_image output
    )
{
    vx_status status = VX_SUCCESS;

    vx_context context = vxGetContext((vx_reference)node);

    vx_size lev;
    vx_size levels = 1;
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    vx_uint32 level_width = 0;
    vx_uint32 level_height = 0;
    vx_df_image format = VX_DF_IMAGE_S16;
    vx_enum policy = VX_CONVERT_POLICY_SATURATE;
    vx_border_t border;
    vx_image filling = 0;
    vx_image pyr_level = 0;
    vx_image filter = 0;
    vx_image out = 0;
    vx_convolution conv;

    vx_scalar spolicy = vxCreateScalar(context, VX_TYPE_ENUM, &policy);

    status |= vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width));
    status |= vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height));
    
    status |= vxQueryPyramid(laplacian, VX_PYRAMID_LEVELS, &levels, sizeof(levels));

    status |= vxQueryNode(node, VX_NODE_BORDER, &border, sizeof(border));
    border.mode = VX_BORDER_REPLICATE;
    conv = vxCreateGaussian5x5Convolution(context);

    level_width = (vx_uint32)ceilf(width  * VX_SCALE_PYRAMID_DOUBLE);
    level_height = (vx_uint32)ceilf(height * VX_SCALE_PYRAMID_DOUBLE);
    filling = vxCreateImage(context, width, height, format);
    for (lev = 0; lev < levels; lev++)
    {
        out = vxCreateImage(context, level_width, level_height, format);
        filter = vxCreateImage(context, level_width, level_height, format);

        pyr_level = vxGetPyramidLevel(laplacian, (vx_uint32)((levels - 1) - lev));

        if (lev == 0)
        {
            ownCopyImage(input, filling);
        }
        upsampleImage(context, level_width, level_height, filling, conv, filter, &border);
        vxuAdd(context, filter, pyr_level, policy, out);
        //vxAddition(filter, pyr_level, spolicy, out);

        status |= vxReleaseImage(&pyr_level);

        if ((levels - 1) - lev == 0)
        {
            ownCopyImage(out, output);
            status |= vxReleaseImage(&filling);
        }
        else
        {
            /* compute dimensions for the next level */
            status |= vxReleaseImage(&filling);
            filling = vxCreateImage(context, level_width, level_height, format);
            ownCopyImage(out, filling);

            level_width = (vx_uint32)ceilf(level_width  * VX_SCALE_PYRAMID_DOUBLE);
            level_height = (vx_uint32)ceilf(level_height * VX_SCALE_PYRAMID_DOUBLE);


        }
        status |= vxReleaseImage(&out);
        status |= vxReleaseImage(&filter);

    }
    status |= vxReleaseConvolution(&conv);
    status |= vxReleaseScalar(&spolicy);
    
    return status;
}
