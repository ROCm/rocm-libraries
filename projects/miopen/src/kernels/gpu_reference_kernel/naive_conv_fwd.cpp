// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "naive_conv.hpp"

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_fwd_nchw(const src_data_t* __restrict__ p_in,
                                           const src_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           dst_data_t* __restrict__ p_out,
                                           Strides5D in_strides,
                                           Strides5D wei_strides,
                                           Strides5D out_strides,
                                           int hi,
                                           int wi,
                                           int n,
                                           int k_per_group,
                                           int c_per_group,
                                           int ho,
                                           int wo,
                                           int sy,
                                           int sx,
                                           int dy,
                                           int dx,
                                           int py,
                                           int px,
                                           int fy,
                                           int fx,
                                           int group)
{
    /*
     *  gridDim.x = group * n * k_per_group (channel slices)
     *  gridDim.y = ceil(ho * wo / blockDim.x) (spatial tiles)
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    if constexpr(ASSUME_PACKED)
    {
        p_in +=
            static_cast<size_t>(in) * c * hi * wi + static_cast<size_t>(ig) * c_per_group * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fy * fx +
                 static_cast<size_t>(ik) * c_per_group * fy * fx;

        p_out += static_cast<size_t>(in) * k * ho * wo +
                 static_cast<size_t>(ig) * k_per_group * ho * wo +
                 static_cast<size_t>(ik) * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[4] + static_cast<size_t>(ig) * in_strides[3];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[4] + static_cast<size_t>(ik) * wei_strides[3];

        p_out += static_cast<size_t>(in) * out_strides[4] +
                 static_cast<size_t>(ig) * out_strides[3] +
                 static_cast<size_t>(ik) * out_strides[2];
    }

    for(int tid = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
        tid < thread_length;
        tid += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.y))
    {
        int iho = tid / wo;
        int iwo = tid % wo;

        acc_data_t value = 0;

        // Precompute valid filter ranges
        int iy_start, iy_end;
        fwd_filter_range(iho, py, dy, sy, fy, hi, iy_start, iy_end);
        int ix_start, ix_end;
        fwd_filter_range(iwo, px, dx, sx, fx, wi, ix_start, ix_end);

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iy = iy_start; iy <= iy_end; iy++)
            {
                int cur_h = sy * iho - py + dy * iy;
                for(int ix = ix_start; ix <= ix_end; ix++)
                {
                    int cur_w = sx * iwo - px + dx * ix;

                    if constexpr(ASSUME_PACKED)
                    {
                        size_t i_idx = static_cast<size_t>(ic) * hi * wi +
                                       static_cast<size_t>(cur_h) * wi + static_cast<size_t>(cur_w);

                        size_t f_idx = static_cast<size_t>(ic) * fy * fx +
                                       static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);

                        value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                 cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                    }
                    else
                    {
                        size_t i_idx = static_cast<size_t>(ic) * in_strides[2] +
                                       static_cast<size_t>(cur_h) * in_strides[1] +
                                       static_cast<size_t>(cur_w) * in_strides[0];

                        size_t f_idx = static_cast<size_t>(ic) * wei_strides[2] +
                                       static_cast<size_t>(iy) * wei_strides[1] +
                                       static_cast<size_t>(ix) * wei_strides[0];

                        value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                 cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                    }
                }
            }
        }
        if constexpr(ASSUME_PACKED)
        {
            size_t o_idx = static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
        else
        {
            size_t o_idx = static_cast<size_t>(iho) * out_strides[1] +
                           static_cast<size_t>(iwo) * out_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
    }
}

// design block_size 256

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_fwd_ncdhw(const src_data_t* __restrict__ p_in,
                                            const src_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            dst_data_t* __restrict__ p_out,
                                            Strides6D in_strides,
                                            Strides6D wei_strides,
                                            Strides6D out_strides,
                                            int di,
                                            int hi,
                                            int wi,
                                            int n,
                                            int k_per_group,
                                            int c_per_group,
                                            int do_,
                                            int ho,
                                            int wo,
                                            int sz,
                                            int sy,
                                            int sx,
                                            int dz,
                                            int dy,
                                            int dx,
                                            int pz,
                                            int py,
                                            int px,
                                            int fz,
                                            int fy,
                                            int fx,
                                            int group)
{
    /*
     *  gridDim.x = group * n * k_per_group (channel slices)
     *  gridDim.y = ceil(do_ * ho * wo / blockDim.x) (spatial tiles)
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = do_ * ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * c * di * hi * wi +
                static_cast<size_t>(ig) * c_per_group * di * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
                 static_cast<size_t>(ik) * c_per_group * fz * fy * fx;

        p_out += static_cast<size_t>(in) * k * do_ * ho * wo +
                 static_cast<size_t>(ig) * k_per_group * do_ * ho * wo +
                 static_cast<size_t>(ik) * do_ * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[5] + static_cast<size_t>(ig) * in_strides[4];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[5] + static_cast<size_t>(ik) * wei_strides[4];

        p_out += static_cast<size_t>(in) * out_strides[5] +
                 static_cast<size_t>(ig) * out_strides[4] +
                 static_cast<size_t>(ik) * out_strides[3];
    }

    for(int tid = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
        tid < thread_length;
        tid += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.y))
    {
        int iwo = tid % wo;
        int iho = (tid / wo) % ho;
        int ido = tid / (ho * wo);

        acc_data_t value = 0;

        // Precompute valid filter ranges
        int iz_start, iz_end;
        fwd_filter_range(ido, pz, dz, sz, fz, di, iz_start, iz_end);
        int iy_start, iy_end;
        fwd_filter_range(iho, py, dy, sy, fy, hi, iy_start, iy_end);
        int ix_start, ix_end;
        fwd_filter_range(iwo, px, dx, sx, fx, wi, ix_start, ix_end);

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iz = iz_start; iz <= iz_end; iz++)
            {
                int cur_d = sz * ido - pz + dz * iz;
                for(int iy = iy_start; iy <= iy_end; iy++)
                {
                    int cur_h = sy * iho - py + dy * iy;
                    for(int ix = ix_start; ix <= ix_end; ix++)
                    {
                        int cur_w = sx * iwo - px + dx * ix;

                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(ic) * di * hi * wi +
                                           static_cast<size_t>(cur_d) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);

                            size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx +
                                           static_cast<size_t>(iz) * fy * fx +
                                           static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(ic) * in_strides[3] +
                                           static_cast<size_t>(cur_d) * in_strides[2] +
                                           static_cast<size_t>(cur_h) * in_strides[1] +
                                           static_cast<size_t>(cur_w) * in_strides[0];

                            size_t f_idx = static_cast<size_t>(ic) * wei_strides[3] +
                                           static_cast<size_t>(iz) * wei_strides[2] +
                                           static_cast<size_t>(iy) * wei_strides[1] +
                                           static_cast<size_t>(ix) * wei_strides[0];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t o_idx = static_cast<size_t>(ido) * ho * wo + static_cast<size_t>(iho) * wo +
                           static_cast<size_t>(iwo);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
        else
        {
            size_t o_idx = static_cast<size_t>(ido) * out_strides[2] +
                           static_cast<size_t>(iho) * out_strides[1] +
                           static_cast<size_t>(iwo) * out_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
    }
}

/***************************** nhwc *****************************/
// design block_size 256

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_fwd_nhwc(const src_data_t* __restrict__ p_in,
                                           const src_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           dst_data_t* __restrict__ p_out,
                                           Strides5D in_strides,
                                           Strides5D wei_strides,
                                           Strides5D out_strides,
                                           int hi,
                                           int wi,
                                           int n,
                                           int k_per_group,
                                           int c_per_group,
                                           int ho,
                                           int wo,
                                           int sy,
                                           int sx,
                                           int dy,
                                           int dx,
                                           int py,
                                           int px,
                                           int fy,
                                           int fx,
                                           int group)
{
    /*
     *  gridDim.x = n * ho (batch * height slices)
     *  gridDim.y = ceil(wo * k / blockDim.x) (spatial tiles)
     */

    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = wo * k;
    int bid           = blockIdx.x;
    int iho           = bid % ho;
    int in            = (bid / ho) % n;

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * hi * wi * c;

        p_out += static_cast<size_t>(in) * ho * wo * k + static_cast<size_t>(iho) * wo * k;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[4];

        p_out +=
            static_cast<size_t>(in) * out_strides[4] + static_cast<size_t>(iho) * out_strides[3];
    }

    for(int tid = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
        tid < thread_length;
        tid += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.y))
    {
        // We want to compute
        //      iwo = tid / k
        //      ik  = tid % k
        // , but
        //      tid = tid / k * k + tid % k = iwo * k + ik
        // so we can avoid the % operation.
        int iwo       = tid / k;
        int global_ik = tid - iwo * k;
        int ig        = global_ik / k_per_group;
        int ik        = global_ik - ig * k_per_group;

        acc_data_t value = 0;

        // Precompute valid filter ranges
        int iy_start, iy_end;
        fwd_filter_range(iho, py, dy, sy, fy, hi, iy_start, iy_end);
        int ix_start, ix_end;
        fwd_filter_range(iwo, px, dx, sx, fx, wi, ix_start, ix_end);

        for(int iy = iy_start; iy <= iy_end; iy++)
        {
            int cur_h = sy * iho - py + dy * iy;
            for(int ix = ix_start; ix <= ix_end; ix++)
            {
                int cur_w = sx * iwo - px + dx * ix;
                for(int ic = 0; ic < c_per_group; ic++)
                {
                    if constexpr(ASSUME_PACKED)
                    {
                        size_t i_idx =
                            static_cast<size_t>(cur_h) * wi * c + static_cast<size_t>(cur_w) * c +
                            static_cast<size_t>(ig) * c_per_group + static_cast<size_t>(ic);

                        size_t f_idx =
                            static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group +
                            static_cast<size_t>(ik) * fy * fx * c_per_group +
                            static_cast<size_t>(iy) * fx * c_per_group +
                            static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);

                        value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                 cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                    }
                    else
                    {
                        size_t i_idx = static_cast<size_t>(cur_h) * in_strides[3] +
                                       static_cast<size_t>(cur_w) * in_strides[2] +
                                       static_cast<size_t>(ig) * in_strides[1] +
                                       static_cast<size_t>(ic) * in_strides[0];

                        size_t f_idx = static_cast<size_t>(ig) * wei_strides[4] +
                                       static_cast<size_t>(ik) * wei_strides[3] +
                                       static_cast<size_t>(iy) * wei_strides[2] +
                                       static_cast<size_t>(ix) * wei_strides[1] +
                                       static_cast<size_t>(ic) * wei_strides[0];

                        value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                 cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t o_idx = static_cast<size_t>(iwo) * k + static_cast<size_t>(ig) * k_per_group +
                           static_cast<size_t>(ik);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
        else
        {
            size_t o_idx = static_cast<size_t>(iwo) * out_strides[2] +
                           static_cast<size_t>(ig) * out_strides[1] +
                           static_cast<size_t>(ik) * out_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
    }
}

// design block_size 256

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_fwd_ndhwc(const src_data_t* __restrict__ p_in,
                                            const src_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            dst_data_t* __restrict__ p_out,
                                            Strides6D in_strides,
                                            Strides6D wei_strides,
                                            Strides6D out_strides,
                                            int di,
                                            int hi,
                                            int wi,
                                            int n,
                                            int k_per_group,
                                            int c_per_group,
                                            int do_,
                                            int ho,
                                            int wo,
                                            int sz,
                                            int sy,
                                            int sx,
                                            int dz,
                                            int dy,
                                            int dx,
                                            int pz,
                                            int py,
                                            int px,
                                            int fz,
                                            int fy,
                                            int fx,
                                            int group)
{
    /*
     *  gridDim.x = group * n * do_ (batch * depth slices)
     *  gridDim.y = ceil(ho * wo * k_per_group / blockDim.x) (spatial tiles)
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = ho * wo * k_per_group;
    int bid           = blockIdx.x;
    int ido           = bid % do_;
    int in            = (bid / do_) % n;
    int ig            = bid / (n * do_);

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(in) * di * hi * wi * c + static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group;

        p_out += static_cast<size_t>(in) * do_ * ho * wo * k +
                 static_cast<size_t>(ido) * ho * wo * k + static_cast<size_t>(ig) * k_per_group;
    }
    else
    {
        // dim order NDHWGC
        // replace C and K with G * C_per_G and G * K_per_G
        p_in += static_cast<size_t>(in) * in_strides[5] + static_cast<size_t>(ig) * in_strides[1];

        // Assumes that group G is the highest dimension in the layout
        p_wei += static_cast<size_t>(ig) * wei_strides[5];

        p_out += static_cast<size_t>(in) * out_strides[5] +
                 static_cast<size_t>(ido) * out_strides[4] +
                 static_cast<size_t>(ig) * out_strides[1];
    }

    for(int tid = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
        tid < thread_length;
        tid += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.y))
    {
        int ik  = tid % k_per_group;
        int iwo = (tid / k_per_group) % wo;
        int iho = tid / (k_per_group * wo);

        acc_data_t value = 0;

        // Precompute valid filter ranges
        int iz_start, iz_end;
        fwd_filter_range(ido, pz, dz, sz, fz, di, iz_start, iz_end);
        int iy_start, iy_end;
        fwd_filter_range(iho, py, dy, sy, fy, hi, iy_start, iy_end);
        int ix_start, ix_end;
        fwd_filter_range(iwo, px, dx, sx, fx, wi, ix_start, ix_end);

        for(int iz = iz_start; iz <= iz_end; iz++)
        {
            int cur_d = sz * ido - pz + dz * iz;
            for(int iy = iy_start; iy <= iy_end; iy++)
            {
                int cur_h = sy * iho - py + dy * iy;
                for(int ix = ix_start; ix <= ix_end; ix++)
                {
                    int cur_w = sx * iwo - px + dx * ix;
                    for(int ic = 0; ic < c_per_group; ic++)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(cur_d) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);

                            size_t f_idx = static_cast<size_t>(ik) * fz * fy * fx * c_per_group +
                                           static_cast<size_t>(iz) * fy * fx * c_per_group +
                                           static_cast<size_t>(iy) * fx * c_per_group +
                                           static_cast<size_t>(ix) * c_per_group +
                                           static_cast<size_t>(ic);

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(cur_d) * in_strides[4] +
                                           static_cast<size_t>(cur_h) * in_strides[3] +
                                           static_cast<size_t>(cur_w) * in_strides[2] +
                                           static_cast<size_t>(ic) * in_strides[0];

                            size_t f_idx = static_cast<size_t>(ik) * wei_strides[4] +
                                           static_cast<size_t>(iz) * wei_strides[3] +
                                           static_cast<size_t>(iy) * wei_strides[2] +
                                           static_cast<size_t>(ix) * wei_strides[1] +
                                           static_cast<size_t>(ic) * wei_strides[0];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }

        if constexpr(ASSUME_PACKED)
        {
            size_t o_idx = static_cast<size_t>(iho) * wo * k + static_cast<size_t>(iwo) * k +
                           static_cast<size_t>(ik);
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
        else
        {
            size_t o_idx = static_cast<size_t>(iho) * out_strides[3] +
                           static_cast<size_t>(iwo) * out_strides[2] +
                           static_cast<size_t>(ik) * out_strides[0];
            applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_out, value, alpha, beta, o_idx);
        }
    }
}

// double accumulator (GPU reference mode)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, float, double, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, float, double, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, half, double, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, hip_bfloat16, double, hip_bfloat16, 0)
// float accumulator (normal solver mode)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, float, float, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, float, float, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, half, float, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, hip_bfloat16, float, hip_bfloat16, 0)
// int8
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, int8_t, int32_t, int8_t, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, int8_t, int32_t, int32_t, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nchw, int8_t, int32_t, float, 0)

// double accumulator (GPU reference mode)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, float, double, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, float, double, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, half, double, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, hip_bfloat16, double, hip_bfloat16, 0)
// float accumulator (normal solver mode)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, float, float, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, float, float, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, half, float, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, hip_bfloat16, float, hip_bfloat16, 0)
// int8
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, int8_t, int32_t, int8_t, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, int8_t, int32_t, int32_t, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(fwd, nhwc, int8_t, int32_t, float, 0)

// double accumulator (GPU reference mode)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, float, double, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, float, double, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, half, double, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, hip_bfloat16, double, hip_bfloat16, 0)
// float accumulator (normal solver mode)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, float, float, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, float, float, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, half, float, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, hip_bfloat16, float, hip_bfloat16, 0)
// int8
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, int8_t, int32_t, int8_t, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, int8_t, int32_t, int32_t, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ncdhw, int8_t, int32_t, float, 0)

// double accumulator (GPU reference mode)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, float, double, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, float, double, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, half, double, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, hip_bfloat16, double, hip_bfloat16, 0)
// float accumulator (normal solver mode)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, float, float, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, float, float, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, half, float, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, hip_bfloat16, float, hip_bfloat16, 0)
// int8
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, int8_t, int32_t, int8_t, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, int8_t, int32_t, int32_t, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(fwd, ndhwc, int8_t, int32_t, float, 0)
