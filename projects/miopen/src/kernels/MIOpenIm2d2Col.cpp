/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "miopen_cstdint.hpp"

#ifndef LAYOUT_NHWC
#define LAYOUT_NHWC 0
#endif

#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

#ifndef MIOPEN_USE_BFP16
#define MIOPEN_USE_BFP16 0
#endif

#ifndef MIOPEN_USE_INT8
#define MIOPEN_USE_INT8 0
#endif

#ifndef MIOPEN_USE_INT32
#define MIOPEN_USE_INT32 0
#endif

#ifndef MIOPEN_USE_FP8
#define MIOPEN_USE_FP8 0
#endif

#ifndef MIOPEN_USE_BFP8
#define MIOPEN_USE_BFP8 0
#endif

#if MIOPEN_USE_INT8 || MIOPEN_USE_FP8 || MIOPEN_USE_BFP8
typedef char data_t;
#elif MIOPEN_USE_INT32
typedef int data_t;
#elif(MIOPEN_USE_FP16 || MIOPEN_USE_BFP16)
// As the half type degrades the performance, use short instead of half in the
// im2col, which has no match op. May change back to half when compile can
// deliver equal performance as short
typedef short data_t;
#elif MIOPEN_USE_FP32
typedef float data_t;
#endif

/* Simple GPU implementation - number of threads launced == sizeof im2col buffer
 * Each thread writes one pixel of output. First (out_h*out_w) threads write to
 * the first line (row) of the im2col output.
 *
 * kernel void Im2Col(global data_t* im, int im_offset,
 * 		const int h, const int w,
 * 		const int wei_h, const int wei_w,
 * 		const int out_h, const int out_w,
 * 		const int pad_h, const int pad_w,
 * 		const int stride_h, const int stride_w,
 * 		global data_t* col)
 * {
 * 	int tid = get_global_id(0);
 *  // which row of the output to write to
 * 	int col_row = tid / (out_h * out_w);
 *
 *  // which pixel from the image and which channel to read from
 * 	int im_x = col_row % wei_w; // used to compute im_off_w
 * 	int im_y = (col_row / wei_w) % wei_h; // used to compute im_off_y
 * 	int im_c = col_row / (wei_w * wei_h); // im_c is the img channel
 *
 * 	int out_x = tid % out_w;
 * 	int out_y = (tid / out_w) % out_h;
 *
 *  // take the strides and padding into account while reading from the image
 * 	int im_off_h = out_y * stride_h - pad_h + im_y;
 * 	int im_off_w = out_x * stride_w - pad_w + im_x;
 *
 * 	global data_t *im_off = (global data_t *)&im[im_offset];
 *
 * 	if(im_off_h >= 0 && im_off_h < h && im_off_w >= 0 && im_off_w < w) {
 * 		col[col_row*out_h*out_w + out_y*out_w + out_x] = im_off[im_c*h*w + im_off_h*w +
 * im_off_w];
 * 	}
 * 	else {
 * 		col[col_row*out_h*out_w + out_y*out_w + out_x] = 0.;
 * 	}
 * }
 */

#ifdef USE_LARGE_BUFFER_INDEX
using index_t = int64_t;
#else
using index_t = int32_t;
#endif

#if(LAYOUT_NHWC == 1)

#if defined(USE_CHANNEL_BASED) && \
    !defined(EXTREME_LARGE) // Channel based execution with tiling, shared memory

#ifndef LOCAL_MEM_SIZE
#define LOCAL_MEM_SIZE 65536
#endif

extern "C" __global__ void Im2d2Col_v2(const int data_size_off,
                                       data_t* im,
                                       const uint64_t im_offset,
                                       const int h,
                                       const int w,
                                       const int wei_h,
                                       const int wei_w,
                                       const int out_h,
                                       const int out_w,
                                       const int pad_h,
                                       const int pad_w,
                                       const int stride_h,
                                       const int stride_w,
                                       const int dilation_h,
                                       const int dilation_w,
                                       data_t* col)
{
    const int lid    = threadIdx.x;
    const int lsize  = blockDim.x;
    const int chan   = blockIdx.x; // one work-group per channel
    const int tile_h = blockIdx.y;
    const int tile_w = blockIdx.z;

    const int num_groups         = GROUPS;
    const int channels_per_group = CHANNELS / GROUPS;
    const int current_group      = chan / channels_per_group;

    data_t* im_off = im + im_offset;

    // One column per output pixel: rows = wei_h * wei_w * num_ch
    const int patch_size           = WEI_H * WEI_W * channels_per_group;
    const int col_group_total_size = patch_size * out_h * out_w;

    const int base_oh = tile_h * TILE_OUT_H;
    const int base_ow = tile_w * TILE_OUT_W;

    // LDS buffer for the single-channel input region required by a tile.
    // We index it as a 2D [im_rows_wg][im_cols_wg] flattened into 1D.
    __shared__ data_t lds[LOCAL_MEM_SIZE];

    const int out_rows_wg = min((int)TILE_OUT_H, out_h - base_oh);
    const int out_cols_wg = min((int)TILE_OUT_W, out_w - base_ow);

    // Input region needed for this output tile (including halo from stride/dilation).
    const int im_rows_wg = (out_rows_wg - 1) * stride_h + (WEI_H - 1) * dilation_h + 1;
    const int im_cols_wg = (out_cols_wg - 1) * stride_w + (WEI_W - 1) * dilation_w + 1;

    for(int idx = lid; idx < im_rows_wg * im_cols_wg; idx += lsize)
    {
        const int r = idx / im_cols_wg; // 0..im_rows_wg-1  (input space, unit steps)
        const int c = idx % im_cols_wg; // 0..im_cols_wg-1

        // Input coords in "padded" space (pad accounted by later subtraction).
        const int im_y_off = base_oh * stride_h + r;
        const int im_x_off = base_ow * stride_w + c;

        data_t v = (data_t)0;
        if(im_y_off >= pad_h && im_y_off < h + pad_h && im_x_off >= pad_w && im_x_off < w + pad_w)
        {
            const int ih            = im_y_off - pad_h; // 0..h-1
            const int iw            = im_x_off - pad_w; // 0..w-1
            const index_t input_idx = ((index_t)ih * w + iw) * CHANNELS + chan;
            v                       = im_off[input_idx];
        }
        lds[r * im_cols_wg + c] = v;
    }
    __syncthreads();

// --- Produce im2col entries for all outputs in this tile ---
#pragma clang loop unroll(full)
    for(int i = 0; i < TILE_OUT_H * TILE_OUT_W; i += lsize)
    {
        const int t  = i + lid;
        const int oy = t / TILE_OUT_W; // 0..out_rows_wg-1
        const int ox = t % TILE_OUT_W; // 0..out_cols_wg-1
        if(oy < out_rows_wg && ox < out_cols_wg)
        {
            const int out_index        = (base_oh + oy) * out_w + (base_ow + ox);
            const index_t patch_offset = (index_t)out_index * patch_size;

// For each kernel position, pick the corresponding element from LDS
// at (oy*stride + kh*dilation, ox*stride + kw*dilation) in the staged tile.
#pragma clang loop unroll(full)
            for(int kh = 0; kh < WEI_H; ++kh)
            {
                const int im_r = oy * stride_h + kh * dilation_h;
#pragma clang loop unroll(full)
                for(int kw = 0; kw < WEI_W; ++kw)
                {
                    const int im_c = ox * stride_w + kw * dilation_w;

                    const data_t v                   = lds[im_r * im_cols_wg + im_c];
                    const int group_relative_channel = chan - (current_group * channels_per_group);
                    const index_t col_idx            = patch_offset +
                                            ((index_t)kh * WEI_W + kw) * channels_per_group +
                                            group_relative_channel;
                    col[(col_group_total_size * current_group) + col_idx] = v;
                }
            }
        }
    }
}

#elif defined(MANY_CHANNELS) && !defined(EXTREME_LARGE)

extern "C" __global__ void Im2d2Col_v2(const int data_size_off,
                                       data_t* im,
                                       const uint64_t im_offset,
                                       const int h,
                                       const int w,
                                       const int wei_h,
                                       const int wei_w,
                                       const int out_h,
                                       const int out_w,
                                       const int pad_h,
                                       const int pad_w,
                                       const int stride_h,
                                       const int stride_w,
                                       const int dilation_h,
                                       const int dilation_w,
                                       data_t* col)
{
    data_t* im_off = im + im_offset;

    const index_t idx    = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    const index_t grp_id = idx / THREADS_PER_CH;
    const int base_c     = idx % THREADS_PER_CH * ITEMS_PER_THREAD;
    const int out_x      = grp_id % out_w;
    const int out_y      = grp_id / out_w;

    if(grp_id >= (index_t)out_w * out_h)
    {
        return;
    }

    const index_t patch_size   = (index_t)WEI_H * WEI_W * CHANNELS;
    const index_t patch_offset = grp_id * patch_size;

#ifdef FLATTEN_WEI_H
    const int k_y = blockIdx.z;
    if(k_y < WEI_H)
#else
#pragma clang loop unroll(full)
    for(int k_y = 0; k_y < WEI_H; ++k_y)
#endif
    {
        const int src_y    = out_y * stride_h + k_y * dilation_h - pad_h;
        const int src_y_ok = (src_y >= 0) & (src_y < h);

#ifdef FLATTEN_WEI_W
        const int k_x = blockIdx.y;
        if(k_x < WEI_W)
#else
#pragma clang loop unroll(full)
        for(int k_x = 0; k_x < WEI_W; ++k_x)
#endif
        {
            const int src_x    = out_x * stride_w + k_x * dilation_w - pad_w;
            const int src_x_ok = (src_x >= 0) & (src_x < w);
            data_t channel_data[ITEMS_PER_THREAD];
            if(src_x_ok & src_y_ok)
            {
                const index_t base_input = ((index_t)src_y * w + src_x) * CHANNELS + base_c;

#pragma clang loop unroll(full)
                for(int i = 0; i < ITEMS_PER_THREAD; ++i)
                {
                    channel_data[i] = im_off[base_input + i];
                }
            }
            else
            {
#pragma clang loop unroll(full)
                for(int i = 0; i < ITEMS_PER_THREAD; ++i)
                {
                    channel_data[i] = (data_t)0;
                }
            }

            const index_t base_col =
                patch_offset + ((index_t)k_y * WEI_W + k_x) * CHANNELS + base_c;

#pragma clang loop unroll(full)
            for(int i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                col[base_col + i] = channel_data[i];
            }
        }
    }
}

#else // output-pixel based implementation

extern "C" __global__ void Im2d2Col_v2(const int data_size_off,
                                       data_t* im,
                                       const uint64_t im_offset,
                                       const int h,
                                       const int w,
                                       const int wei_h,
                                       const int wei_w,
                                       const int out_h,
                                       const int out_w,
                                       const int pad_h,
                                       const int pad_w,
                                       const int stride_h,
                                       const int stride_w,
                                       const int dilation_h,
                                       const int dilation_w,
                                       data_t* col)
{
    const int lid        = threadIdx.x;
    const int grp_id     = blockIdx.x; // patch id (one output pixel)
    const int local_size = blockDim.x;

    data_t* im_off = im + im_offset;

    const int output_size = out_h * out_w;
    if(grp_id >= output_size)
        return;

    // this workgroup's output pixel
    const int oh = grp_id / out_w;
    const int ow = grp_id % out_w;

    const int num_groups         = GROUPS;
    const int channels_per_group = CHANNELS / GROUPS;

    // The size of the "K" dimension in the GEMM for a single group
    const int patch_size_per_group = WEI_H * WEI_W * channels_per_group;
    const int patch_size           = WEI_H * WEI_W * CHANNELS;

    for(int patch_idx = lid; patch_idx < patch_size; patch_idx += local_size)
    {
        const int c  = patch_idx % CHANNELS;
        const int kw = (patch_idx / CHANNELS) % WEI_W;
        const int kh = patch_idx / (CHANNELS * WEI_W);

        const int current_group            = c / channels_per_group;
        const int channel_in_current_group = c % channels_per_group;
        const int group_offset             = current_group * (output_size * patch_size_per_group);
        const int patch_offset             = grp_id * patch_size_per_group;

        const int src_h = oh * stride_h + kh * dilation_h - pad_h;
        const int src_w = ow * stride_w + kw * dilation_w - pad_w;
        const int ok    = (src_h >= 0) & (src_h < h) & (src_w >= 0) & (src_w < w);

        data_t v = (data_t)0;
        if(ok)
        {
            const int input_idx = ((src_h * w + src_w) * CHANNELS) + c;
            v                   = im_off[input_idx];
        }

        const int col_idx = group_offset + patch_offset + (kh * WEI_W + kw) * channels_per_group +
                            channel_in_current_group;
        col[col_idx] = v;
    }
}

#endif // output pixel based version

#else // LAYOUT_NHWC == 0

extern "C" __global__ void Im2d2Col_v2(const int data_size_off,
                                       data_t* im,
                                       const uint64_t im_offset,
                                       const int h,
                                       const int w,
                                       const int wei_h,
                                       const int wei_w,
                                       const int out_h,
                                       const int out_w,
                                       const int pad_h,
                                       const int pad_w,
                                       const int stride_h,
                                       const int stride_w,
                                       const int dilation_h,
                                       const int dilation_w,
                                       data_t* col,
                                       const int num_ch_per_wg,
                                       const int num_im_blks_x,
                                       const int num_im_blks,
                                       const int tile_sz_x,
                                       const int tile_sz_y)
{
    /// NUM_CH_PER_WG {1;4}
    /// THREADS_PER_CH {256; 64}
    (void)num_ch_per_wg;
    (void)num_im_blks_x;
    (void)num_im_blks;
    (void)tile_sz_x;
    (void)tile_sz_y;

#if USE_IM_OFF_GUARD
#define IM_OFF_GUARD(idx) (idx) < data_size_off ? im_off[(idx)] : 0
#else
#define IM_OFF_GUARD(idx) im_off[idx]
#endif

    data_t* im_off = im + im_offset;

#ifndef EXTREME_LARGE

    int lid = threadIdx.x;
    /// tile_sz_x = {32,16,8,4,2,1}, tile_sz_y = {8,4,2,1}
    /// NUM_IM_BLKS_X = out_w / tile_sz_x
    /// NUM_IM_BLKS = NUM_IM_BLKS_X * out_h / tile_sz_y => out_w * out_h
    /// c * NUM_IM_BLKS => c * out_w * out_h
    index_t gid = blockIdx.x;

#if NUM_IM_BLKS_EQ_1 == 1 && STRIDE_GT_1 == 0
    // This does not need to be a division and should be a right shift
    const int threads_per_ch = 256 / num_ch_per_wg;

    // Load image into LDS
    /// max (LOCAL_MEM_SIZE) = 65536
    __shared__ data_t local_im[LOCAL_MEM_SIZE];

    /// witem_ch [0;4)
    int witem_ch = lid / threads_per_ch;

    int im_lid = lid;
    /// h*w < LOCAL_MEM_SIZE/witem_ch
    int gid_stride = num_ch_per_wg * h * w;
    while(im_lid < gid_stride)
    {
        /// gid = max(1, (c_pack / NUM_CH_PER_WG)) => c
        /// max (c * LOCAL_MEM_SIZE) => 65536 * c
        index_t im_off_id = gid * gid_stride + im_lid;
        local_im[im_lid]  = IM_OFF_GUARD(im_off_id);
        im_lid += 256;
    }
    __syncthreads();

    // where will each thread to col
    /// should fit in LDS size => witem_ch_offset < LOCAL_MEM_SIZE
    /// h*w < LOCAL_MEM_SIZE/witem_ch
    int witem_ch_offset = witem_ch * h * w;
    /// if (NUM_IM_BLKS == 1) => (out_h < 8 && out_w < 32)
    ///      => out_hw_stride < 256
    int out_hw_stride = out_h * out_w;
    if(lid % threads_per_ch < out_hw_stride)
    {
        /// lid[0, 255] % THREADS_PER_CH {256; 64} =>
        /// max(inner_lid)=255; max(out_x)=max(out_y)=255
        int inner_lid = lid % threads_per_ch;
        int out_x     = inner_lid % out_w;
        int out_y     = inner_lid / out_w;

        /// out_w < 32; out_y < 255; out_x < 255
        /// col_x < 2 080 800
        int col_x = out_y * out_w + out_x;
        /// gid = c = group_cnt-1; NUM_CH_PER_WG{1,4}; out_hw_stride < 256;
        /// EXTREME_LARGE==0
        /// => wei_h * wei_w * type_size * NUM_CH_PER_WG < max (LOCAL_MEM_SIZE)
        /// gid * out_hw_stride * LOCAL_MEM_SIZE => c * 256 * 65536
        index_t col_y = ((index_t)gid * num_ch_per_wg + witem_ch) * out_hw_stride * wei_h * wei_w;

        for(int y = 0; y < wei_h; y++)
        {
            for(int x = 0; x < wei_w; x++)
            {
                /// max(im_off_h)*w <= max(LOCAL_MEM_SIZE); max(im_off_w) <= max(LOCAL_MEM_SIZE);
                int im_off_h = out_y * stride_h - pad_h + y * dilation_h;
                int im_off_w = out_x * stride_w - pad_w + x * dilation_w;
                /// y * wei_w * type_size * NUM_CH_PER_WG < max (LOCAL_MEM_SIZE)
                int im_off_wei_hw = y * wei_w + x;
                // col_x + (im_off_wei_hw * out_hw_stride) => 2 080 800 + 65536 * 255
                index_t col_off = col_y + col_x + im_off_wei_hw * out_hw_stride;
                if(im_off_h >= 0 && im_off_h < h && im_off_w >= 0 && im_off_w < w)
                    col[col_off] = local_im[witem_ch_offset + (im_off_h)*w + im_off_w];
                else
                    col[col_off] = 0;
            }
        }
    }

#else  // NUM_IM_BLKS > 1 || STRIDE_GT_1 1

    __shared__ data_t local_im[LOCAL_MEM_SIZE];

    int wg_ch = gid / num_im_blks;
    /// TILE_SZ_X = 32, TILE_SZ_Y = 8;
    /// gid = c * NUM_IM_BLKS => im_x = NUM_IM_BLKS*TILE_SZ_X = NUM_IM_BLKS*32
    /// = NUM_IM_BLKS*32 = out_w * out_h / 8
    int im_x = ((gid % num_im_blks) % num_im_blks_x) * tile_sz_x; /// < out_w
    int im_y = ((gid % num_im_blks) / num_im_blks_x) * tile_sz_y; /// < out_h

    int out_cols_wg = (im_x + tile_sz_x) <= out_w ? tile_sz_x : (out_w - im_x); /// < out_w
    int out_rows_wg = (im_y + tile_sz_y) <= out_h ? tile_sz_y : (out_h - im_y); /// < out_h

    int im_cols_wg = (tile_sz_x - 1) * stride_w + (wei_w - 1) * dilation_w + 1;

    int inner_lid = lid;

    while(inner_lid < LOCAL_MEM_SIZE)
    {
        /// < 256
        int row_to_use = inner_lid / im_cols_wg;
        int col_to_use = inner_lid % im_cols_wg;
        /// max = LOCAL_MEM_SIZE + im_cols_wg
        int lm_offset = row_to_use * im_cols_wg + col_to_use;

        /// out_h*stride_h+256
        int im_y_off = im_y * stride_h + row_to_use;
        /// out_w*stride_w+256
        int im_x_off = im_x * stride_w + col_to_use;

        if(im_y_off >= pad_h && im_y_off < h + pad_h && im_x_off >= pad_w && im_x_off < w + pad_w)
        {
            int im_off_h        = im_y_off - pad_h;
            int im_off_w        = im_x_off - pad_w;
            index_t im_off_id   = (index_t)wg_ch * h * w + im_off_h * w + im_off_w;
            local_im[lm_offset] = IM_OFF_GUARD(im_off_id);
        }
        else
            local_im[lm_offset] = 0;

        inner_lid += 256;
    }
    __syncthreads();

    inner_lid = lid;
    while(inner_lid < out_cols_wg * out_rows_wg)
    {
        int out_x = inner_lid % out_cols_wg; /// < 256
        int out_y = inner_lid / out_cols_wg; /// < 256

        index_t col_x = (index_t)(im_y + out_y) * out_w + im_x + out_x; /// out_h * out_w
        /// c * out_h * out_w * wei_h * wei_w
        index_t col_y = (gid / num_im_blks) * out_h * out_w * wei_h * wei_w;

        for(int y = 0; y < wei_h; y++)
        {
            for(int x = 0; x < wei_w; x++)
            {
                int im_off_h    = out_y * stride_h + y * dilation_h;
                int im_off_w    = out_x * stride_w + x * dilation_w;
                index_t col_off = col_y + col_x + ((index_t)y * wei_w + x) * out_h * out_w;
                col[col_off]    = local_im[(im_off_h)*im_cols_wg + im_off_w];
            }
        }
        inner_lid += 256;
    }
#endif // NUM_IM_BLKS && STRIDE_GT_1
#else  // Very large support

    index_t tid = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < (index_t)out_h * out_w * wei_w * wei_h * NUM_CH_TOTAL)
    {
        // which row of the output to write to
        index_t col_row = tid / ((index_t)out_h * out_w); // wei_w * wei_h * NUM_CH_TOTAL

        // which pixel from the image and which channel to read from
        int im_x = col_row % wei_w;                    // used to compute im_off_w
        int im_y = (col_row / wei_w) % wei_h;          // used to compute im_off_y
        int im_c = col_row / ((index_t)wei_w * wei_h); // im_c is the img channel

        int out_x = tid % out_w;
        int out_y = (tid / out_w) % out_h;

        // take the strides and padding into account while reading from the image
        int im_off_h = out_y * stride_h - pad_h + im_y * dilation_h;
        int im_off_w = out_x * stride_w - pad_w + im_x * dilation_w;

        index_t col_off = col_row * out_h * out_w + (index_t)out_y * out_w + out_x;

        if(im_off_h >= 0 && im_off_h < h && im_off_w >= 0 && im_off_w < w)
        {
            col[col_off] = IM_OFF_GUARD((index_t)im_c * h * w + im_off_h * w + im_off_w);
        }
        else
        {
            col[col_off] = 0.;
        }
        tid += (index_t)gridDim.x * blockDim.x;
    }
#endif
}
#endif // LAYOUT_NHWC else
