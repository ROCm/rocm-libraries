/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#ifndef MLO_POOLINGHOST_H_
#define MLO_POOLINGHOST_H_

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <type_traits>

#include "calcerr.hpp"
#include <miopen/bfloat16.hpp>

#if 0
template<typename _T>
double CalcErr( _T c_val, _T g_val)
{
    double err = 0;
    if (sizeof(_T) == 4)
    {
        int * c_uval = (int *)&c_val;
        int * g_uval = (int *)&g_val;
        err = (double)std::abs(*c_uval - *g_uval);
    }
    else if (sizeof(_T) == 8)
    {
        int64_t * c_uval = (int64_t *)&c_val;
        int64_t * g_uval = (int64_t *)&g_val;
        err = (double)std::abs(*c_uval - *g_uval);

    }

    //		double delta = abs(c_val - g_val);
    //	double nextafter_delta = nextafterf(min(abs(c_val), abs(g_val)), (_T)INFINITY) - min(abs(c_val), abs(g_val));
    //		err = delta / nextafter_delta;
    return err;
}
#endif

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

#ifndef MLO_POOLING_OP_MAX
#define MLO_POOLING_OP_MAX 0
#define MLO_POOLING_OP_AVE 1
#define MLO_POOLING_OP_STC 2
#define MLO_POOLING_OP_AVE_INCLUSIVE 3
#endif

// Non-zero value N skips each Nth computation, thus leading to validation failure.
// For FP16, 100 works well.
#define MLO_POOLING_EMULATE_VALIDATION_FAILURE 0

constexpr int N = 0;
constexpr int C = 1;
constexpr int D = 2;
constexpr int H = 3;
constexpr int W = 4;

struct pooling_math_stats
{
    double max_error          = 0.0;
    int max_num_flops_per_res = 0;
};

struct TensorDimsStrides
{
    int n_batchs;
    int n_outputs;
    int depth;
    int height;
    int width;

    int n_stride;
    int c_stride;
    int d_stride;
    int h_stride;
    int w_stride;

    TensorDimsStrides() {}

    TensorDimsStrides(int spatial_dim, const miopen::TensorDescriptor& desc)
    {
        std::tie(n_batchs, n_outputs, depth, height, width) =
            miopen::GetNCDHW(spatial_dim, desc.GetLengths());

        std::tie(n_stride, c_stride, d_stride, h_stride, w_stride) =
            miopen::GetNCDHW(spatial_dim, desc.GetStrides());
    }
};

struct PoolingConfig
{
    int method;
    int pad_d;
    int stride_d;
    int size_d;
    int pad_h;
    int stride_h;
    int size_h;
    int pad_w;
    int stride_w;
    int size_w;

    PoolingConfig() {}

    PoolingConfig(const int method_,
                  const int pad_d_,
                  const int stride_d_,
                  const int size_d_,
                  const int pad_h_,
                  const int stride_h_,
                  const int size_h_,
                  const int pad_w_,
                  const int stride_w_,
                  const int size_w_)
    {
        method   = method_;
        pad_d    = pad_d_;
        stride_d = stride_d_;
        size_d   = size_d_;
        pad_h    = pad_h_;
        stride_h = stride_h_;
        size_h   = size_h_;
        pad_w    = pad_w_;
        stride_w = stride_w_;
        size_w   = size_w_;
    }
};

template <bool is_mt = false, typename Tgpu_, typename Tcheck_, typename Index, typename Tmatch_>
void fwd_pooling_compute_verify(const TensorDimsStrides& bot_dims_strides,
                                const TensorDimsStrides& top_dims_strides,
                                const PoolingConfig& poolConf,
                                Tmatch_& match,
                                const Tgpu_* bot_ptr,
                                const Tgpu_* top_ptr,
                                const size_t b,
                                const size_t o,
                                const std::array<int, 5>& mask_strides,
                                const bool do_backward,
                                const int index_position,
                                const Tcheck_ MAX_VAL,
                                const Tgpu_ G_MAX_VAL,
                                size_t* mask_ptr,
                                Index* mask_gpu,
                                const Tcheck_ allowedEps,
                                pooling_math_stats& stats,
                                std::optional<std::reference_wrapper<std::mutex>> stats_mutex)
{
    if constexpr(is_mt) // checks when running with multiple threads
    {
        static_assert(std::is_same<Tmatch_, std::atomic_bool>::value == true,
                      "'match' has to be std::atomic_bool when running in multi-threaded mode");
        if(!stats_mutex.has_value())
        {
            MIOPEN_THROW(
                "'std::mutex' is needed as last parameter when running in multi-threaded mode");
        }
    }
    Tcheck_ res_init = (poolConf.method == MLO_POOLING_OP_MAX) ? -MAX_VAL : static_cast<Tcheck_>(0);
    Tcheck_ res      = static_cast<Tcheck_>(0);

    size_t bot_idx_off      = b * bot_dims_strides.n_stride + o * bot_dims_strides.c_stride;
    size_t top_idx_off      = b * top_dims_strides.n_stride + o * top_dims_strides.c_stride;
    size_t mask_gpu_idx_off = b * mask_strides[N] + o * mask_strides[C];

    size_t total_pool_size = poolConf.size_w * poolConf.size_h * poolConf.size_d;

    double max_err            = 0.0;
    int max_num_flops_per_res = 0;

    for(int k = 0; k < top_dims_strides.depth; k++)
    {
        for(int j = 0; j < top_dims_strides.height; j++)
        {
            for(int i = 0; i < top_dims_strides.width; i++)
            {
                // c-emulator
                res                   = res_init;
                int num_flops_per_res = 0;

                int dstart = k * poolConf.stride_d - poolConf.pad_d;
                int hstart = j * poolConf.stride_h - poolConf.pad_h;
                int wstart = i * poolConf.stride_w - poolConf.pad_w;
                int dend   = std::min(dstart + poolConf.size_d, bot_dims_strides.depth);
                int hend   = std::min(hstart + poolConf.size_h, bot_dims_strides.height);
                int wend   = std::min(wstart + poolConf.size_w, bot_dims_strides.width);
                dstart     = std::max(dstart, 0);
                hstart     = std::max(hstart, 0);
                wstart     = std::max(wstart, 0);

                int pool_size        = (poolConf.method == MLO_POOLING_OP_AVE)
                                           ? ((dend - dstart) * (hend - hstart) * (wend - wstart))
                                           : total_pool_size;
                pool_size            = (pool_size == 0) ? 1 : pool_size;
                size_t res_index     = 0;
                size_t res_index_gpu = 0;
                bool found           = false;
                for(int d = dstart; d < dend; ++d)
                {
                    for(int h = hstart; h < hend; ++h)
                    {
                        for(int w = wstart; w < wend; ++w)
                        {
                            size_t bot_index = bot_idx_off + d * bot_dims_strides.d_stride +
                                               h * bot_dims_strides.h_stride +
                                               w * bot_dims_strides.w_stride;
                            if(poolConf.method == MLO_POOLING_OP_MAX)
                            {
                                if(static_cast<Tcheck_>(bot_ptr[bot_index]) > res)
                                {
                                    res       = static_cast<Tcheck_>(bot_ptr[bot_index]);
                                    res_index = bot_index;
                                    res_index_gpu =
                                        index_position == 1
                                            ? (d * bot_dims_strides.height *
                                                   bot_dims_strides.width +
                                               h * bot_dims_strides.width + w)
                                            : ((d - k * poolConf.stride_d + poolConf.pad_d) *
                                               poolConf.size_w * poolConf.size_h) +
                                                  ((h - j * poolConf.stride_h + poolConf.pad_h) *
                                                   poolConf.size_w) +
                                                  (w - i * poolConf.stride_w + poolConf.pad_w);
                                    found = true;
                                }
                            }
                            else if(poolConf.method == MLO_POOLING_OP_AVE ||
                                    poolConf.method == MLO_POOLING_OP_AVE_INCLUSIVE)
                            {
#if MLO_POOLING_EMULATE_VALIDATION_FAILURE
                                if(num_flops_per_res % MLO_POOLING_EMULATE_VALIDATION_FAILURE != 0)
#endif
                                    res += static_cast<Tcheck_>(bot_ptr[bot_index]);
                                ++num_flops_per_res;
                            }
                            else
                            {
                                std::cout << "ERROR: unknown operator : layer: pooling."
                                          << std::endl;
                                match = false;
                                return;
                            }
                        }
                    }
                }
                // special index value is used to mark top points which has no associated
                // bottom
                // points
                if(!found)
                {
                    res_index     = std::numeric_limits<size_t>::max();
                    res_index_gpu = std::numeric_limits<uint8_t>::max();
                }

                size_t top_index = top_idx_off + k * top_dims_strides.d_stride +
                                   j * top_dims_strides.h_stride + i * top_dims_strides.w_stride;
                size_t mask_gpu_index = mask_gpu_idx_off + k * mask_strides[D] +
                                        j * mask_strides[H] + i * mask_strides[W];
                if(poolConf.method == MLO_POOLING_OP_MAX)
                {
                    // the case with the odd input, the even kernel size and 2*pad == kernel
                    // size
                    mask_ptr[top_index] = res_index;
                    if(do_backward)
                    {
                        size_t mg = mask_gpu[mask_gpu_index];
                        if(mg != res_index_gpu)
                        {
                            std::cout << "Mask mismatch, gpu " << mg << " cpu " << res_index_gpu
                                      << "(" << res_index << ")" << std::endl;
                            match = false;
                        }
                    }
                }
                if(poolConf.method == MLO_POOLING_OP_AVE ||
                   poolConf.method == MLO_POOLING_OP_AVE_INCLUSIVE)
                {
                    res /= pool_size;
                    ++num_flops_per_res;
                }
                Tcheck_ c_val = (res == -MAX_VAL) ? 0 : res;

                Tgpu_ gg_val = (top_ptr[top_index]);

                gg_val = (gg_val == (-G_MAX_VAL)) ? static_cast<Tgpu_>(0) : gg_val;

                Tcheck_ g_val(gg_val);

                double err = std::abs(c_val - g_val);

                if(err > allowedEps || std::isnan(c_val) || std::isnan(g_val) ||
                   !std::isfinite(c_val) || !std::isfinite(g_val))
                {
                    std::cout << "Difference " << err << " too large (> " << allowedEps << ") at {"
                              << b << ',' << o << ',' << j << ',' << i << "}, cpu_val = " << c_val
                              << " vs gpu_val = " << g_val << std::endl;
                    std::cout << "Number of flops used: " << num_flops_per_res
                              << ", pool_size: " << pool_size << std::endl;
                    match = false;
                }

                if constexpr(is_mt)
                {
                    max_err               = err > max_err ? err : max_err;
                    max_num_flops_per_res = num_flops_per_res > max_num_flops_per_res
                                                ? num_flops_per_res
                                                : max_num_flops_per_res;
                }
                else
                {
                    if(err > stats.max_error)
                        stats.max_error = err;
                    if(num_flops_per_res > stats.max_num_flops_per_res)
                        stats.max_num_flops_per_res = num_flops_per_res;
                }
            }
        }
    }
    if constexpr(is_mt)
    {
        const std::lock_guard<std::mutex> lock(stats_mutex.value());
        if(max_err > stats.max_error)
        {
            stats.max_error = max_err;
        }

        if(max_num_flops_per_res > stats.max_num_flops_per_res)
        {
            stats.max_num_flops_per_res = max_num_flops_per_res;
        }
    }
};

template <typename Tgpu_ /* the data type used in GPU computations (usually half) */,
          typename Tcheck_ /* the data type used in CPU checkings (usually double) */,
          typename Index>
bool mloPoolingForwardRunHostAndVerify(PoolingConfig poolConf,
                                       const miopenTensorDescriptor_t& bot_,
                                       const miopenTensorDescriptor_t& top_,
                                       const Tgpu_* bot_ptr,
                                       const Tgpu_* top_ptr,
                                       bool do_backward,
                                       size_t* mask_ptr,
                                       Index* mask_gpu,
                                       Tcheck_ allowedEps,
                                       pooling_math_stats& stats,
                                       int index_position = 1)
{
    const miopen::TensorDescriptor& bot = miopen::deref(bot_);
    const miopen::TensorDescriptor& top = miopen::deref(top_);

    const auto spatial_dim = bot.GetLengths().size() - 2;
    TensorDimsStrides bot_dims_strides(spatial_dim, bot);
    TensorDimsStrides top_dims_strides(spatial_dim, top);

    // Mask data is always NCDHW
    std::array<int, 5> mask_strides;
    mask_strides[W] = 1;
    mask_strides[H] = mask_strides[W] * top_dims_strides.width;
    mask_strides[D] = mask_strides[H] * top_dims_strides.height;
    mask_strides[C] = mask_strides[D] * top_dims_strides.depth;
    mask_strides[N] = mask_strides[C] * bot_dims_strides.n_outputs;

    bool match = true;
    Tcheck_ MAX_VAL(3.402823466e+38);
    Tgpu_ G_MAX_VAL = std::numeric_limits<Tgpu_>::max();

    for(int b = 0; b < bot_dims_strides.n_batchs && match; b++)
    {
        for(int o = 0; o < bot_dims_strides.n_outputs && match; o++)
        {
            fwd_pooling_compute_verify(bot_dims_strides,
                                       top_dims_strides,
                                       poolConf,
                                       match,
                                       bot_ptr,
                                       top_ptr,
                                       b,
                                       o,
                                       mask_strides,
                                       do_backward,
                                       index_position,
                                       MAX_VAL,
                                       G_MAX_VAL,
                                       mask_ptr,
                                       mask_gpu,
                                       allowedEps,
                                       stats,
                                       std::nullopt); // for mutex used in mt implementation
        }
    }

    return (match);
}

template <typename Tgpu_ /* the data type used in GPU computations (usually half) */,
          typename Tcheck_ /* the data type used in CPU checkings (usually double) */,
          typename Index>
bool mloPoolingForwardRunHostAndVerify_mt(PoolingConfig poolConf,
                                          const miopenTensorDescriptor_t& bot_,
                                          const miopenTensorDescriptor_t& top_,
                                          const Tgpu_* bot_ptr,
                                          const Tgpu_* top_ptr,
                                          bool do_backward,
                                          size_t* mask_ptr,
                                          Index* mask_gpu,
                                          Tcheck_ allowedEps,
                                          pooling_math_stats& stats,
                                          int index_position = 1)
{
    const miopen::TensorDescriptor& bot = miopen::deref(bot_);
    const miopen::TensorDescriptor& top = miopen::deref(top_);

    const auto spatial_dim = bot.GetLengths().size() - 2;
    TensorDimsStrides bot_dims_strides(spatial_dim, bot);
    TensorDimsStrides top_dims_strides(spatial_dim, top);

    // Mask data is always NCDHW
    std::array<int, 5> mask_strides;
    mask_strides[W] = 1;
    mask_strides[H] = mask_strides[W] * top_dims_strides.width;
    mask_strides[D] = mask_strides[H] * top_dims_strides.height;
    mask_strides[C] = mask_strides[D] * top_dims_strides.depth;
    mask_strides[N] = mask_strides[C] * bot_dims_strides.n_outputs;

    std::atomic_bool match = true;
    Tcheck_ MAX_VAL(3.402823466e+38);
    Tgpu_ G_MAX_VAL = std::numeric_limits<Tgpu_>::max();
    std::mutex stats_mutex;
    miopen::par_ford(bot_dims_strides.n_batchs, bot_dims_strides.n_outputs)([&](int b, int o) {
        fwd_pooling_compute_verify<true>(bot_dims_strides,
                                         top_dims_strides,
                                         poolConf,
                                         match,
                                         bot_ptr,
                                         top_ptr,
                                         b,
                                         o,
                                         mask_strides,
                                         do_backward,
                                         index_position,
                                         MAX_VAL,
                                         G_MAX_VAL,
                                         mask_ptr,
                                         mask_gpu,
                                         allowedEps,
                                         stats,
                                         stats_mutex);
    });

    return (match);
}

template <bool is_mt = false, typename Tgpu_, typename Tcheck_>
void bwd_pooling_compute(const TensorDimsStrides& bot_dims_strides,
                         const TensorDimsStrides& top_dims_strides,
                         const PoolingConfig& poolConf,
                         Tcheck_* bot_df_v_ptr, // the code assumes that bot_df_v_ptr was zeroed
                         const Tgpu_* top_df_ptr,
                         size_t b,
                         size_t o,
                         const size_t* mask_ptr,
                         std::vector<int>& num_flops)
{
    int bot_df_v_off = b * bot_dims_strides.n_stride + o * bot_dims_strides.c_stride;
    int top_df_off   = b * top_dims_strides.n_stride + o * top_dims_strides.c_stride;

    if(poolConf.method == MLO_POOLING_OP_MAX)
    {
        for(int k = 0; k < top_dims_strides.depth; k++)
        {
            for(int j = 0; j < top_dims_strides.height; j++)
            {
                for(int i = 0; i < top_dims_strides.width; i++)
                {
                    size_t top_idx = top_df_off + k * top_dims_strides.d_stride +
                                     j * top_dims_strides.h_stride + i * top_dims_strides.w_stride;
                    size_t bot_idx = mask_ptr[top_idx];
                    // skip top points that don't have associated bottom points
                    if(bot_idx == std::numeric_limits<size_t>::max())
                        continue;
                    if constexpr(is_mt)
                    {
                        std::atomic_ref<Tcheck_>(bot_df_v_ptr[bot_idx])
                            .fetch_add(static_cast<Tcheck_>(top_df_ptr[top_idx]));
                        std::atomic_ref<int>(num_flops[bot_idx])++;
                    }
                    else
                    {
                        bot_df_v_ptr[bot_idx] += static_cast<Tcheck_>(top_df_ptr[top_idx]);
                        ++num_flops[bot_idx];
                    }
                }
            }
        }
    }
    else if(poolConf.method == MLO_POOLING_OP_AVE ||
            poolConf.method == MLO_POOLING_OP_AVE_INCLUSIVE)
    {

        for(int k = 0; k < bot_dims_strides.depth; k++)
        {
            for(int j = 0; j < bot_dims_strides.height; j++)
            {
                for(int i = 0; i < bot_dims_strides.width; i++)
                {
                    // c-emulator
                    const auto bot_idx = bot_df_v_off + k * bot_dims_strides.d_stride +
                                         j * bot_dims_strides.h_stride +
                                         i * bot_dims_strides.w_stride;
                    bot_df_v_ptr[bot_idx] = static_cast<Tcheck_>(0);
                    num_flops[bot_idx]    = 0;

                    int d = k + poolConf.pad_d;
                    int h = j + poolConf.pad_h;
                    int w = i + poolConf.pad_w;
                    int pdstart =
                        (d < poolConf.size_d) ? 0 : (d - poolConf.size_d) / poolConf.stride_d + 1;
                    int pdend = std::min(d / poolConf.stride_d + 1, top_dims_strides.depth);
                    int phstart =
                        (h < poolConf.size_h) ? 0 : (h - poolConf.size_h) / poolConf.stride_h + 1;
                    int phend = std::min(h / poolConf.stride_h + 1, top_dims_strides.height);
                    int pwstart =
                        (w < poolConf.size_w) ? 0 : (w - poolConf.size_w) / poolConf.stride_w + 1;
                    int pwend        = std::min(w / poolConf.stride_w + 1, top_dims_strides.width);
                    Tcheck_ gradient = static_cast<Tcheck_>(0);
                    int gradient_n_flops = 0;
                    for(int pd = pdstart; pd < pdend; ++pd)
                    {
                        for(int ph = phstart; ph < phend; ++ph)
                        {
                            for(int pw = pwstart; pw < pwend; ++pw)
                            {
                                // figure out the pooling size
                                int dstart = pd * poolConf.stride_d - poolConf.pad_d;
                                int hstart = ph * poolConf.stride_h - poolConf.pad_h;
                                int wstart = pw * poolConf.stride_w - poolConf.pad_w;
                                int dend =
                                    std::min(dstart + poolConf.size_d, bot_dims_strides.depth);
                                int hend =
                                    std::min(hstart + poolConf.size_h, bot_dims_strides.height);
                                int wend =
                                    std::min(wstart + poolConf.size_w, bot_dims_strides.width);
                                dstart = std::max(dstart, 0);
                                hstart = std::max(hstart, 0);
                                wstart = std::max(wstart, 0);

                                int pool_size;
                                if(poolConf.method == MLO_POOLING_OP_AVE)
                                    pool_size =
                                        ((dend - dstart) * (hend - hstart) * (wend - wstart) == 0)
                                            ? 1
                                            : (dend - dstart) * (hend - hstart) * (wend - wstart);
                                else
                                    pool_size =
                                        (poolConf.size_w * poolConf.size_h * poolConf.size_d == 0)
                                            ? 1
                                            : poolConf.size_w * poolConf.size_h * poolConf.size_d;

                                const auto top_idx = top_df_off + pd * top_dims_strides.d_stride +
                                                     ph * top_dims_strides.h_stride +
                                                     pw * top_dims_strides.w_stride;

                                gradient += static_cast<Tcheck_>(top_df_ptr[top_idx]) /
                                            static_cast<Tcheck_>(pool_size);
                                gradient_n_flops += 2; // pool_size is computed using
                                                       // integer ops, do not count those.
                            }
                        }
                    }
                    bot_df_v_ptr[bot_idx] = gradient;
                    num_flops[bot_idx]    = gradient_n_flops;
                }
            }
        }
    }
    else
    {
        std::cout << "ERROR: unknown operator : layer: pooling back-propagation." << std::endl;
        return;
    }
}

template <typename Tgpu_ /* the data type used in GPU computations (usually half) */,
          typename Tcheck_ /* the data type used in CPU checkings (usually double) */>
void mloPoolingBackwardRunHost(
    PoolingConfig poolConf,
    const miopenTensorDescriptor_t& bot_df_,
    const miopenTensorDescriptor_t& top_df_,
    Tcheck_* bot_df_v_ptr, // the code assumes that bot_df_v_ptr was zeroed
    const Tgpu_* top_df_ptr,
    const size_t* mask_ptr,
    pooling_math_stats& stats)
{
    const miopen::TensorDescriptor& bot_df = miopen::deref(bot_df_);
    const miopen::TensorDescriptor& top_df = miopen::deref(top_df_);

    const auto spatial_dim = bot_df.GetLengths().size() - 2;
    TensorDimsStrides bot_dims_strides(spatial_dim, bot_df);
    TensorDimsStrides top_dims_strides(spatial_dim, top_df);
    std::vector<int> num_flops(bot_df.GetElementSize(), 0);

    for(int b = 0; b < bot_dims_strides.n_batchs; b++)
    {
        for(int o = 0; o < bot_dims_strides.n_outputs; o++)
        {
            bwd_pooling_compute(bot_dims_strides,
                                top_dims_strides,
                                poolConf,
                                bot_df_v_ptr,
                                top_df_ptr,
                                b,
                                o,
                                mask_ptr,
                                num_flops);
        }
    }
    stats.max_num_flops_per_res = *(std::max_element(num_flops.begin(), num_flops.end()));
}

template <typename Tgpu_ /* the data type used in GPU computations (usually half) */,
          typename Tcheck_ /* the data type used in CPU checkings (usually double) */>
void mloPoolingBackwardRunHost_mt(
    PoolingConfig poolConf,
    const miopenTensorDescriptor_t& bot_df_,
    const miopenTensorDescriptor_t& top_df_,
    Tcheck_* bot_df_v_ptr, // the code assumes that bot_df_v_ptr was zeroed
    const Tgpu_* top_df_ptr,
    const size_t* mask_ptr,
    pooling_math_stats& stats)
{
    const miopen::TensorDescriptor& bot_df = miopen::deref(bot_df_);
    const miopen::TensorDescriptor& top_df = miopen::deref(top_df_);

    const auto spatial_dim = bot_df.GetLengths().size() - 2;
    TensorDimsStrides bot_dims_strides(spatial_dim, bot_df);
    TensorDimsStrides top_dims_strides(spatial_dim, top_df);

    std::vector<int> num_flops(bot_df.GetElementSize(), 0);

    miopen::par_ford(bot_dims_strides.n_batchs, bot_dims_strides.n_outputs)([&](int b, int o) {
        bwd_pooling_compute<true>(bot_dims_strides,
                                  top_dims_strides,
                                  poolConf,
                                  bot_df_v_ptr,
                                  top_df_ptr,
                                  b,
                                  o,
                                  mask_ptr,
                                  num_flops);
    });

    stats.max_num_flops_per_res = *(std::max_element(num_flops.begin(), num_flops.end()));
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
