// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "naive_conv.hpp"

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_wrw_nchw(const src_data_t* __restrict__ p_in,
                                           dst_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           const src_data_t* __restrict__ p_out,
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
     *  need to compute total filter pixel: `group * k_per_group * c_per_group *
     * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fy
     * * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(ig) * c_per_group * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fy * fx +
                 static_cast<size_t>(ik) * c_per_group * fy * fx;

        p_out +=
            static_cast<size_t>(ig) * k_per_group * ho * wo + static_cast<size_t>(ik) * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(ig) * in_strides[3];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[4] + static_cast<size_t>(ik) * wei_strides[3];

        p_out +=
            static_cast<size_t>(ig) * out_strides[3] + static_cast<size_t>(ik) * out_strides[2];
    }

    int num_weights    = c_per_group * fy * fx;
    int spatial_length = n * ho * wo;
    int bdx            = static_cast<int>(blockDim.x);
    int tx             = static_cast<int>(threadIdx.x);
    // Invariant: the solver must only set gridDim.y > 1 when dst_data_t has
    // naive_atomic_add (has_native_atomic_add<dst_data_t>::value). The if
    // constexpr guard takes the non-atomic fallback path for unsupported types.
    bool use_atomic = (gridDim.y > 1);

    // Spatial tile range for cross-block tiling (gridDim.y > 1)
    int tile_start = static_cast<int>(blockIdx.y) * bdx;
    int tile_end   = min(tile_start + bdx, spatial_length);

    if(num_weights <= bdx / 2)
    {
        // Fast path: multiple workers per weight element, shared-memory reduction
        __shared__ acc_data_t smem[NAIVE_CONV_BLOCK_SIZE];

        int workers    = bdx / num_weights;
        int weight_idx = tx % num_weights;
        int worker_id  = tx / num_weights;

        int ix = weight_idx % fx;
        int iy = (weight_idx / fx) % fy;
        int ic = weight_idx / (fx * fy);

        acc_data_t value = 0;
        if(worker_id < workers)
        {
            if(use_atomic)
            {
                // Cross-block tiling: iterate over this block's spatial tile
                for(int s = tile_start + worker_id; s < tile_end; s += workers)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int in  = s / (ho * wo);

                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_h >= 0 && cur_h < hi && cur_w >= 0 && cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * c * hi * wi +
                                           static_cast<size_t>(ic) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);

                            size_t o_idx = static_cast<size_t>(in) * k * ho * wo +
                                           static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[4] +
                                           static_cast<size_t>(ic) * in_strides[2] +
                                           static_cast<size_t>(cur_h) * in_strides[1] +
                                           static_cast<size_t>(cur_w) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[1] +
                                           static_cast<size_t>(iwo) * out_strides[0];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
            else
            {
                // Single-block: iterate over all spatial positions
                for(int s = worker_id; s < spatial_length; s += workers)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int in  = s / (ho * wo);

                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_h >= 0 && cur_h < hi && cur_w >= 0 && cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * c * hi * wi +
                                           static_cast<size_t>(ic) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);

                            size_t o_idx = static_cast<size_t>(in) * k * ho * wo +
                                           static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[4] +
                                           static_cast<size_t>(ic) * in_strides[2] +
                                           static_cast<size_t>(cur_h) * in_strides[1] +
                                           static_cast<size_t>(cur_w) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[1] +
                                           static_cast<size_t>(iwo) * out_strides[0];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
        }

        smem[tx] = value;
        __syncthreads();

        if(worker_id == 0)
        {
            acc_data_t sum = smem[tx];
            for(int w = 1; w < workers; w++)
            {
                int idx = tx + w * num_weights;
                if(idx < bdx)
                    sum += smem[idx];
            }

            if constexpr(ASSUME_PACKED)
            {
                size_t f_idx = static_cast<size_t>(ic) * fy * fx + static_cast<size_t>(iy) * fx +
                               static_cast<size_t>(ix);
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(sum));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, sum, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, sum, alpha, beta, f_idx);
                }
            }
            else
            {
                size_t f_idx = static_cast<size_t>(ic) * wei_strides[2] +
                               static_cast<size_t>(iy) * wei_strides[1] +
                               static_cast<size_t>(ix) * wei_strides[0];
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(sum));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, sum, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, sum, alpha, beta, f_idx);
                }
            }
        }
    }
    else
    {
        // Fallback path: one thread per weight element
        for(int tid = tx; tid < thread_length; tid += bdx)
        {
            int ix = tid % fx;
            int iy = (tid / fx) % fy;
            int ic = tid / (fx * fy);

            acc_data_t value = 0;

            if(use_atomic)
            {
                // Cross-block tiling: iterate over this block's spatial tile
                for(int s = tile_start; s < tile_end; s++)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int in  = s / (ho * wo);

                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_h >= 0 && cur_h < hi && cur_w >= 0 && cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * c * hi * wi +
                                           static_cast<size_t>(ic) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);

                            size_t o_idx = static_cast<size_t>(in) * k * ho * wo +
                                           static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[4] +
                                           static_cast<size_t>(ic) * in_strides[2] +
                                           static_cast<size_t>(cur_h) * in_strides[1] +
                                           static_cast<size_t>(cur_w) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[1] +
                                           static_cast<size_t>(iwo) * out_strides[0];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
            else
            {
                // Single-block: iterate over all spatial positions
                for(int in = 0; in < n; in++)
                {
                    for(int iho = 0; iho < ho; iho++)
                    {
                        int valid_h = 1;
                        int cur_h   = sy * iho - py + dy * iy;
                        if(cur_h < 0 || cur_h >= hi)
                            valid_h &= 0;
                        for(int iwo = 0; iwo < wo; iwo++)
                        {
                            int valid_w = 1;
                            int cur_w   = sx * iwo - px + dx * ix;
                            if(cur_w < 0 || cur_w >= wi)
                                valid_w &= 0;

                            if(valid_h & valid_w)
                            {
                                if constexpr(ASSUME_PACKED)
                                {
                                    size_t i_idx = static_cast<size_t>(in) * c * hi * wi +
                                                   static_cast<size_t>(ic) * hi * wi +
                                                   static_cast<size_t>(cur_h) * wi +
                                                   static_cast<size_t>(cur_w);

                                    size_t o_idx = static_cast<size_t>(in) * k * ho * wo +
                                                   static_cast<size_t>(iho) * wo +
                                                   static_cast<size_t>(iwo);

                                    value +=
                                        cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                        cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                                }
                                else
                                {
                                    size_t i_idx = static_cast<size_t>(in) * in_strides[4] +
                                                   static_cast<size_t>(ic) * in_strides[2] +
                                                   static_cast<size_t>(cur_h) * in_strides[1] +
                                                   static_cast<size_t>(cur_w) * in_strides[0];

                                    size_t o_idx = static_cast<size_t>(in) * out_strides[4] +
                                                   static_cast<size_t>(iho) * out_strides[1] +
                                                   static_cast<size_t>(iwo) * out_strides[0];

                                    value +=
                                        cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                        cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                                }
                            }
                        }
                    }
                }
            }

            if constexpr(ASSUME_PACKED)
            {
                size_t f_idx = static_cast<size_t>(ic) * fy * fx + static_cast<size_t>(iy) * fx +
                               static_cast<size_t>(ix);
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(value));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, value, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
                }
            }
            else
            {
                size_t f_idx = static_cast<size_t>(ic) * wei_strides[2] +
                               static_cast<size_t>(iy) * wei_strides[1] +
                               static_cast<size_t>(ix) * wei_strides[0];
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(value));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, value, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
                }
            }
        }
    }
}

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_wrw_ncdhw(const src_data_t* __restrict__ p_in,
                                            dst_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            const src_data_t* __restrict__ p_out,
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
     *  need to compute total filter pixel: `group * k_per_group * c_per_group *
     * fz * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fz
     * * fy * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fz * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(ig) * c_per_group * di * hi * wi;

        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fz * fy * fx +
                 static_cast<size_t>(ik) * c_per_group * fz * fy * fx;

        p_out += static_cast<size_t>(ig) * k_per_group * do_ * ho * wo +
                 static_cast<size_t>(ik) * do_ * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(ig) * in_strides[4];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[5] + static_cast<size_t>(ik) * wei_strides[4];

        p_out +=
            static_cast<size_t>(ig) * out_strides[4] + static_cast<size_t>(ik) * out_strides[3];
    }

    int num_weights    = c_per_group * fz * fy * fx;
    int spatial_length = n * do_ * ho * wo;
    int bdx            = static_cast<int>(blockDim.x);
    int tx             = static_cast<int>(threadIdx.x);
    // Invariant: the solver must only set gridDim.y > 1 when dst_data_t has
    // naive_atomic_add (has_native_atomic_add<dst_data_t>::value). The if
    // constexpr guard takes the non-atomic fallback path for unsupported types.
    bool use_atomic = (gridDim.y > 1);

    // Spatial tile range for cross-block tiling (gridDim.y > 1)
    int tile_start = static_cast<int>(blockIdx.y) * bdx;
    int tile_end   = min(tile_start + bdx, spatial_length);

    if(num_weights <= bdx / 2)
    {
        __shared__ acc_data_t smem_ncdhw[NAIVE_CONV_BLOCK_SIZE];

        int workers    = bdx / num_weights;
        int weight_idx = tx % num_weights;
        int worker_id  = tx / num_weights;

        int ix = weight_idx % fx;
        int iy = (weight_idx / fx) % fy;
        int iz = (weight_idx / (fx * fy)) % fz;
        int ic = weight_idx / (fx * fy * fz);

        acc_data_t value = 0;
        if(worker_id < workers)
        {
            if(use_atomic)
            {
                // Cross-block tiling: iterate over this block's spatial tile
                for(int s = tile_start + worker_id; s < tile_end; s += workers)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int ido = (s / (wo * ho)) % do_;
                    int in  = s / (do_ * ho * wo);

                    int cur_d = sz * ido - pz + dz * iz;
                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_d >= 0 && cur_d < di && cur_h >= 0 && cur_h < hi && cur_w >= 0 &&
                       cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * c * di * hi * wi +
                                           static_cast<size_t>(ic) * di * hi * wi +
                                           static_cast<size_t>(cur_d) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);

                            size_t o_idx = static_cast<size_t>(in) * k * do_ * ho * wo +
                                           static_cast<size_t>(ido) * ho * wo +
                                           static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                           static_cast<size_t>(ic) * in_strides[3] +
                                           static_cast<size_t>(cur_d) * in_strides[2] +
                                           static_cast<size_t>(cur_h) * in_strides[1] +
                                           static_cast<size_t>(cur_w) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                           static_cast<size_t>(ido) * out_strides[2] +
                                           static_cast<size_t>(iho) * out_strides[1] +
                                           static_cast<size_t>(iwo) * out_strides[0];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
            else
            {
                // Single-block: iterate over all spatial positions
                for(int s = worker_id; s < spatial_length; s += workers)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int ido = (s / (wo * ho)) % do_;
                    int in  = s / (do_ * ho * wo);

                    int cur_d = sz * ido - pz + dz * iz;
                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_d >= 0 && cur_d < di && cur_h >= 0 && cur_h < hi && cur_w >= 0 &&
                       cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * c * di * hi * wi +
                                           static_cast<size_t>(ic) * di * hi * wi +
                                           static_cast<size_t>(cur_d) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);

                            size_t o_idx = static_cast<size_t>(in) * k * do_ * ho * wo +
                                           static_cast<size_t>(ido) * ho * wo +
                                           static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                           static_cast<size_t>(ic) * in_strides[3] +
                                           static_cast<size_t>(cur_d) * in_strides[2] +
                                           static_cast<size_t>(cur_h) * in_strides[1] +
                                           static_cast<size_t>(cur_w) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                           static_cast<size_t>(ido) * out_strides[2] +
                                           static_cast<size_t>(iho) * out_strides[1] +
                                           static_cast<size_t>(iwo) * out_strides[0];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
        }

        smem_ncdhw[tx] = value;
        __syncthreads();

        if(worker_id == 0)
        {
            acc_data_t sum = smem_ncdhw[tx];
            for(int w = 1; w < workers; w++)
            {
                int idx = tx + w * num_weights;
                if(idx < bdx)
                    sum += smem_ncdhw[idx];
            }

            if constexpr(ASSUME_PACKED)
            {
                size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx +
                               static_cast<size_t>(iz) * fy * fx + static_cast<size_t>(iy) * fx +
                               static_cast<size_t>(ix);
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(sum));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, sum, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, sum, alpha, beta, f_idx);
                }
            }
            else
            {
                size_t f_idx = static_cast<size_t>(ic) * wei_strides[3] +
                               static_cast<size_t>(iz) * wei_strides[2] +
                               static_cast<size_t>(iy) * wei_strides[1] +
                               static_cast<size_t>(ix) * wei_strides[0];
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(sum));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, sum, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, sum, alpha, beta, f_idx);
                }
            }
        }
    }
    else
    {
        // Fallback path: one thread per weight element
        for(int tid = tx; tid < thread_length; tid += bdx)
        {
            int ix = tid % fx;
            int iy = (tid / fx) % fy;
            int iz = (tid / (fx * fy)) % fz;
            int ic = tid / (fx * fy * fz);

            acc_data_t value = 0;

            if(use_atomic)
            {
                // Cross-block tiling: iterate over this block's spatial tile
                for(int s = tile_start; s < tile_end; s++)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int ido = (s / (wo * ho)) % do_;
                    int in  = s / (do_ * ho * wo);

                    int cur_d = sz * ido - pz + dz * iz;
                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_d >= 0 && cur_d < di && cur_h >= 0 && cur_h < hi && cur_w >= 0 &&
                       cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * c * di * hi * wi +
                                           static_cast<size_t>(ic) * di * hi * wi +
                                           static_cast<size_t>(cur_d) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);

                            size_t o_idx = static_cast<size_t>(in) * k * do_ * ho * wo +
                                           static_cast<size_t>(ido) * ho * wo +
                                           static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                           static_cast<size_t>(ic) * in_strides[3] +
                                           static_cast<size_t>(cur_d) * in_strides[2] +
                                           static_cast<size_t>(cur_h) * in_strides[1] +
                                           static_cast<size_t>(cur_w) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                           static_cast<size_t>(ido) * out_strides[2] +
                                           static_cast<size_t>(iho) * out_strides[1] +
                                           static_cast<size_t>(iwo) * out_strides[0];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
            else
            {
                // Single-block: iterate over all spatial positions
                for(int in = 0; in < n; in++)
                {
                    for(int ido = 0; ido < do_; ido++)
                    {
                        int valid_d = 1;
                        int cur_d   = sz * ido - pz + dz * iz;
                        if(cur_d < 0 || cur_d >= di)
                            valid_d &= 0;
                        for(int iho = 0; iho < ho; iho++)
                        {
                            int valid_h = 1;
                            int cur_h   = sy * iho - py + dy * iy;
                            if(cur_h < 0 || cur_h >= hi)
                                valid_h &= 0;
                            for(int iwo = 0; iwo < wo; iwo++)
                            {
                                int valid_w = 1;
                                int cur_w   = sx * iwo - px + dx * ix;
                                if(cur_w < 0 || cur_w >= wi)
                                    valid_w &= 0;

                                if(valid_d & valid_h & valid_w)
                                {
                                    if constexpr(ASSUME_PACKED)
                                    {
                                        size_t i_idx = static_cast<size_t>(in) * c * di * hi * wi +
                                                       static_cast<size_t>(ic) * di * hi * wi +
                                                       static_cast<size_t>(cur_d) * hi * wi +
                                                       static_cast<size_t>(cur_h) * wi +
                                                       static_cast<size_t>(cur_w);

                                        size_t o_idx = static_cast<size_t>(in) * k * do_ * ho * wo +
                                                       static_cast<size_t>(ido) * ho * wo +
                                                       static_cast<size_t>(iho) * wo +
                                                       static_cast<size_t>(iwo);

                                        value +=
                                            cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                            cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                                    }
                                    else
                                    {
                                        size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                                       static_cast<size_t>(ic) * in_strides[3] +
                                                       static_cast<size_t>(cur_d) * in_strides[2] +
                                                       static_cast<size_t>(cur_h) * in_strides[1] +
                                                       static_cast<size_t>(cur_w) * in_strides[0];

                                        size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                                       static_cast<size_t>(ido) * out_strides[2] +
                                                       static_cast<size_t>(iho) * out_strides[1] +
                                                       static_cast<size_t>(iwo) * out_strides[0];

                                        value +=
                                            cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                            cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if constexpr(ASSUME_PACKED)
            {
                size_t f_idx = static_cast<size_t>(ic) * fz * fy * fx +
                               static_cast<size_t>(iz) * fy * fx + static_cast<size_t>(iy) * fx +
                               static_cast<size_t>(ix);
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(value));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, value, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
                }
            }
            else
            {
                size_t f_idx = static_cast<size_t>(ic) * wei_strides[3] +
                               static_cast<size_t>(iz) * wei_strides[2] +
                               static_cast<size_t>(iy) * wei_strides[1] +
                               static_cast<size_t>(ix) * wei_strides[0];
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(value));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, value, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
                }
            }
        }
    }
}

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_wrw_nhwc(const src_data_t* __restrict__ p_in,
                                           dst_data_t* __restrict__ p_wei,
                                           const double alpha,
                                           const double beta,
                                           const src_data_t* __restrict__ p_out,
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
     *  need to compute total filter pixel: `group * k_per_group * fy * fx *
     * c_per_group`.
     *  to distribute this workload, let one workgroup compute `fy * fx *
     * c_per_group` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fy * fx * c_per_group +
                 static_cast<size_t>(ik) * fy * fx * c_per_group;

        p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);
    }
    else
    {
        p_in += static_cast<size_t>(ig) * in_strides[1];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[4] + static_cast<size_t>(ik) * wei_strides[3];

        p_out +=
            static_cast<size_t>(ig) * out_strides[1] + static_cast<size_t>(ik) * out_strides[0];
    }

    int num_weights    = c_per_group * fy * fx;
    int spatial_length = n * ho * wo;
    int bdx            = static_cast<int>(blockDim.x);
    int tx             = static_cast<int>(threadIdx.x);
    // Invariant: the solver must only set gridDim.y > 1 when dst_data_t has
    // naive_atomic_add (has_native_atomic_add<dst_data_t>::value). The if
    // constexpr guard takes the non-atomic fallback path for unsupported types.
    bool use_atomic = (gridDim.y > 1);

    // Spatial tile range for cross-block tiling (gridDim.y > 1)
    int tile_start = static_cast<int>(blockIdx.y) * bdx;
    int tile_end   = min(tile_start + bdx, spatial_length);

    if(num_weights <= bdx / 2)
    {
        __shared__ acc_data_t smem_nhwc[NAIVE_CONV_BLOCK_SIZE];

        int workers    = bdx / num_weights;
        int weight_idx = tx % num_weights;
        int worker_id  = tx / num_weights;

        int ic = weight_idx % c_per_group;
        int ix = (weight_idx / c_per_group) % fx;
        int iy = weight_idx / (c_per_group * fx);

        acc_data_t value = 0;
        if(worker_id < workers)
        {
            if(use_atomic)
            {
                // Cross-block tiling: iterate over this block's spatial tile
                for(int s = tile_start + worker_id; s < tile_end; s += workers)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int in  = s / (ho * wo);

                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_h >= 0 && cur_h < hi && cur_w >= 0 && cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);

                            size_t o_idx = static_cast<size_t>(in) * ho * wo * k +
                                           static_cast<size_t>(iho) * wo * k +
                                           static_cast<size_t>(iwo) * k;

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[4] +
                                           static_cast<size_t>(cur_h) * in_strides[3] +
                                           static_cast<size_t>(cur_w) * in_strides[2] +
                                           static_cast<size_t>(ic) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[3] +
                                           static_cast<size_t>(iwo) * out_strides[2];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
            else
            {
                // Single-block: iterate over all spatial positions
                for(int s = worker_id; s < spatial_length; s += workers)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int in  = s / (ho * wo);

                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_h >= 0 && cur_h < hi && cur_w >= 0 && cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);

                            size_t o_idx = static_cast<size_t>(in) * ho * wo * k +
                                           static_cast<size_t>(iho) * wo * k +
                                           static_cast<size_t>(iwo) * k;

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[4] +
                                           static_cast<size_t>(cur_h) * in_strides[3] +
                                           static_cast<size_t>(cur_w) * in_strides[2] +
                                           static_cast<size_t>(ic) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[3] +
                                           static_cast<size_t>(iwo) * out_strides[2];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
        }

        smem_nhwc[tx] = value;
        __syncthreads();

        if(worker_id == 0)
        {
            acc_data_t sum = smem_nhwc[tx];
            for(int w = 1; w < workers; w++)
            {
                int idx = tx + w * num_weights;
                if(idx < bdx)
                    sum += smem_nhwc[idx];
            }

            if constexpr(ASSUME_PACKED)
            {
                size_t f_idx = static_cast<size_t>(iy) * fx * c_per_group +
                               static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(sum));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, sum, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, sum, alpha, beta, f_idx);
                }
            }
            else
            {
                size_t f_idx = static_cast<size_t>(iy) * wei_strides[2] +
                               static_cast<size_t>(ix) * wei_strides[1] +
                               static_cast<size_t>(ic) * wei_strides[0];
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(sum));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, sum, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, sum, alpha, beta, f_idx);
                }
            }
        }
    }
    else
    {
        // Fallback path: one thread per weight element
        for(int tid = tx; tid < thread_length; tid += bdx)
        {
            int ic = tid % c_per_group;
            int ix = (tid / c_per_group) % fx;
            int iy = tid / (c_per_group * fx);

            acc_data_t value = 0;

            if(use_atomic)
            {
                // Cross-block tiling: iterate over this block's spatial tile
                for(int s = tile_start; s < tile_end; s++)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int in  = s / (ho * wo);

                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_h >= 0 && cur_h < hi && cur_w >= 0 && cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);

                            size_t o_idx = static_cast<size_t>(in) * ho * wo * k +
                                           static_cast<size_t>(iho) * wo * k +
                                           static_cast<size_t>(iwo) * k;

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[4] +
                                           static_cast<size_t>(cur_h) * in_strides[3] +
                                           static_cast<size_t>(cur_w) * in_strides[2] +
                                           static_cast<size_t>(ic) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[3] +
                                           static_cast<size_t>(iwo) * out_strides[2];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
            else
            {
                // Single-block: iterate over all spatial positions
                for(int in = 0; in < n; in++)
                {
                    for(int iho = 0; iho < ho; iho++)
                    {
                        int valid_h = 1;
                        int cur_h   = sy * iho - py + dy * iy;
                        if(cur_h < 0 || cur_h >= hi)
                            valid_h &= 0;
                        for(int iwo = 0; iwo < wo; iwo++)
                        {
                            int valid_w = 1;
                            int cur_w   = sx * iwo - px + dx * ix;
                            if(cur_w < 0 || cur_w >= wi)
                                valid_w &= 0;

                            if(valid_h & valid_w)
                            {
                                if constexpr(ASSUME_PACKED)
                                {
                                    size_t i_idx = static_cast<size_t>(in) * hi * wi * c +
                                                   static_cast<size_t>(cur_h) * wi * c +
                                                   static_cast<size_t>(cur_w) * c +
                                                   static_cast<size_t>(ic);

                                    size_t o_idx = static_cast<size_t>(in) * ho * wo * k +
                                                   static_cast<size_t>(iho) * wo * k +
                                                   static_cast<size_t>(iwo) * k;

                                    value +=
                                        cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                        cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                                }
                                else
                                {
                                    size_t i_idx = static_cast<size_t>(in) * in_strides[4] +
                                                   static_cast<size_t>(cur_h) * in_strides[3] +
                                                   static_cast<size_t>(cur_w) * in_strides[2] +
                                                   static_cast<size_t>(ic) * in_strides[0];

                                    size_t o_idx = static_cast<size_t>(in) * out_strides[4] +
                                                   static_cast<size_t>(iho) * out_strides[3] +
                                                   static_cast<size_t>(iwo) * out_strides[2];

                                    value +=
                                        cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                        cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                                }
                            }
                        }
                    }
                }
            }

            if constexpr(ASSUME_PACKED)
            {
                size_t f_idx = static_cast<size_t>(iy) * fx * c_per_group +
                               static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(value));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, value, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
                }
            }
            else
            {
                size_t f_idx = static_cast<size_t>(iy) * wei_strides[2] +
                               static_cast<size_t>(ix) * wei_strides[1] +
                               static_cast<size_t>(ic) * wei_strides[0];
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(value));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, value, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
                }
            }
        }
    }
}

template <bool ASSUME_PACKED,
          typename src_data_t,
          typename acc_data_t,
          typename dst_data_t,
          int use_tf32 = 0>
inline __device__ void naive_conv_wrw_ndhwc(const src_data_t* __restrict__ p_in,
                                            dst_data_t* __restrict__ p_wei,
                                            const double alpha,
                                            const double beta,
                                            const src_data_t* __restrict__ p_out,
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
     *  need to compute total filter pixel: `group * k_per_group * fz * fy * fx
     * * c_per_group`.
     *  to distribute this workload, let one workgroup compute `fz * fy * fx *
     * c_per_group` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = fz * fy * fx * c_per_group;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    if constexpr(ASSUME_PACKED)
    {
        p_in += static_cast<size_t>(ig) * c_per_group;

        p_wei += static_cast<size_t>(ig) * k_per_group * fz * fy * fx * c_per_group +
                 static_cast<size_t>(ik) * fz * fy * fx * c_per_group;

        p_out += static_cast<size_t>(ig) * k_per_group + static_cast<size_t>(ik);
    }
    else
    {
        p_in += static_cast<size_t>(ig) * in_strides[1];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[5] + static_cast<size_t>(ik) * wei_strides[4];

        p_out +=
            static_cast<size_t>(ig) * out_strides[1] + static_cast<size_t>(ik) * out_strides[0];
    }

    int num_weights    = fz * fy * fx * c_per_group;
    int spatial_length = n * do_ * ho * wo;
    int bdx            = static_cast<int>(blockDim.x);
    int tx             = static_cast<int>(threadIdx.x);
    // Invariant: the solver must only set gridDim.y > 1 when dst_data_t has
    // naive_atomic_add (has_native_atomic_add<dst_data_t>::value). The if
    // constexpr guard takes the non-atomic fallback path for unsupported types.
    bool use_atomic = (gridDim.y > 1);

    // Spatial tile range for cross-block tiling (gridDim.y > 1)
    int tile_start = static_cast<int>(blockIdx.y) * bdx;
    int tile_end   = min(tile_start + bdx, spatial_length);

    if(num_weights <= bdx / 2)
    {
        __shared__ acc_data_t smem_ndhwc[NAIVE_CONV_BLOCK_SIZE];

        int workers    = bdx / num_weights;
        int weight_idx = tx % num_weights;
        int worker_id  = tx / num_weights;

        int ic = weight_idx % c_per_group;
        int ix = (weight_idx / c_per_group) % fx;
        int iy = (weight_idx / (c_per_group * fx)) % fy;
        int iz = weight_idx / (c_per_group * fx * fy);

        acc_data_t value = 0;
        if(worker_id < workers)
        {
            if(use_atomic)
            {
                // Cross-block tiling: iterate over this block's spatial tile
                for(int s = tile_start + worker_id; s < tile_end; s += workers)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int ido = (s / (wo * ho)) % do_;
                    int in  = s / (do_ * ho * wo);

                    int cur_d = sz * ido - pz + dz * iz;
                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_d >= 0 && cur_d < di && cur_h >= 0 && cur_h < hi && cur_w >= 0 &&
                       cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * di * hi * wi * c +
                                           static_cast<size_t>(cur_d) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);

                            size_t o_idx = static_cast<size_t>(in) * do_ * ho * wo * k +
                                           static_cast<size_t>(ido) * ho * wo * k +
                                           static_cast<size_t>(iho) * wo * k +
                                           static_cast<size_t>(iwo) * k;

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                           static_cast<size_t>(cur_d) * in_strides[4] +
                                           static_cast<size_t>(cur_h) * in_strides[3] +
                                           static_cast<size_t>(cur_w) * in_strides[2] +
                                           static_cast<size_t>(ic) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                           static_cast<size_t>(ido) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[3] +
                                           static_cast<size_t>(iwo) * out_strides[2];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
            else
            {
                // Single-block: iterate over all spatial positions
                for(int s = worker_id; s < spatial_length; s += workers)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int ido = (s / (wo * ho)) % do_;
                    int in  = s / (do_ * ho * wo);

                    int cur_d = sz * ido - pz + dz * iz;
                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_d >= 0 && cur_d < di && cur_h >= 0 && cur_h < hi && cur_w >= 0 &&
                       cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * di * hi * wi * c +
                                           static_cast<size_t>(cur_d) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);

                            size_t o_idx = static_cast<size_t>(in) * do_ * ho * wo * k +
                                           static_cast<size_t>(ido) * ho * wo * k +
                                           static_cast<size_t>(iho) * wo * k +
                                           static_cast<size_t>(iwo) * k;

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                           static_cast<size_t>(cur_d) * in_strides[4] +
                                           static_cast<size_t>(cur_h) * in_strides[3] +
                                           static_cast<size_t>(cur_w) * in_strides[2] +
                                           static_cast<size_t>(ic) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                           static_cast<size_t>(ido) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[3] +
                                           static_cast<size_t>(iwo) * out_strides[2];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
        }

        smem_ndhwc[tx] = value;
        __syncthreads();

        if(worker_id == 0)
        {
            acc_data_t sum = smem_ndhwc[tx];
            for(int w = 1; w < workers; w++)
            {
                int idx = tx + w * num_weights;
                if(idx < bdx)
                    sum += smem_ndhwc[idx];
            }

            if constexpr(ASSUME_PACKED)
            {
                size_t f_idx = static_cast<size_t>(iz) * fy * fx * c_per_group +
                               static_cast<size_t>(iy) * fx * c_per_group +
                               static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(sum));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, sum, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, sum, alpha, beta, f_idx);
                }
            }
            else
            {
                size_t f_idx = static_cast<size_t>(iz) * wei_strides[3] +
                               static_cast<size_t>(iy) * wei_strides[2] +
                               static_cast<size_t>(ix) * wei_strides[1] +
                               static_cast<size_t>(ic) * wei_strides[0];
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(sum));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, sum, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, sum, alpha, beta, f_idx);
                }
            }
        }
    }
    else
    {
        // Fallback path: one thread per weight element
        for(int tid = tx; tid < thread_length; tid += bdx)
        {
            int ic = tid % c_per_group;
            int ix = (tid / c_per_group) % fx;
            int iy = (tid / (c_per_group * fx)) % fy;
            int iz = tid / (c_per_group * fx * fy);

            acc_data_t value = 0;

            if(use_atomic)
            {
                // Cross-block tiling: iterate over this block's spatial tile
                for(int s = tile_start; s < tile_end; s++)
                {
                    int iwo = s % wo;
                    int iho = (s / wo) % ho;
                    int ido = (s / (wo * ho)) % do_;
                    int in  = s / (do_ * ho * wo);

                    int cur_d = sz * ido - pz + dz * iz;
                    int cur_h = sy * iho - py + dy * iy;
                    int cur_w = sx * iwo - px + dx * ix;

                    if(cur_d >= 0 && cur_d < di && cur_h >= 0 && cur_h < hi && cur_w >= 0 &&
                       cur_w < wi)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(in) * di * hi * wi * c +
                                           static_cast<size_t>(cur_d) * hi * wi * c +
                                           static_cast<size_t>(cur_h) * wi * c +
                                           static_cast<size_t>(cur_w) * c + static_cast<size_t>(ic);

                            size_t o_idx = static_cast<size_t>(in) * do_ * ho * wo * k +
                                           static_cast<size_t>(ido) * ho * wo * k +
                                           static_cast<size_t>(iho) * wo * k +
                                           static_cast<size_t>(iwo) * k;

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                           static_cast<size_t>(cur_d) * in_strides[4] +
                                           static_cast<size_t>(cur_h) * in_strides[3] +
                                           static_cast<size_t>(cur_w) * in_strides[2] +
                                           static_cast<size_t>(ic) * in_strides[0];

                            size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                           static_cast<size_t>(ido) * out_strides[4] +
                                           static_cast<size_t>(iho) * out_strides[3] +
                                           static_cast<size_t>(iwo) * out_strides[2];

                            value += cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                        }
                    }
                }
            }
            else
            {
                // Single-block: iterate over all spatial positions
                for(int in = 0; in < n; in++)
                {
                    for(int ido = 0; ido < do_; ido++)
                    {
                        int valid_d = 1;
                        int cur_d   = sz * ido - pz + dz * iz;
                        if(cur_d < 0 || cur_d >= di)
                            valid_d &= 0;
                        for(int iho = 0; iho < ho; iho++)
                        {
                            int valid_h = 1;
                            int cur_h   = sy * iho - py + dy * iy;
                            if(cur_h < 0 || cur_h >= hi)
                                valid_h &= 0;
                            for(int iwo = 0; iwo < wo; iwo++)
                            {
                                int valid_w = 1;
                                int cur_w   = sx * iwo - px + dx * ix;
                                if(cur_w < 0 || cur_w >= wi)
                                    valid_w &= 0;

                                if(valid_d & valid_h & valid_w)
                                {
                                    if constexpr(ASSUME_PACKED)
                                    {
                                        size_t i_idx = static_cast<size_t>(in) * di * hi * wi * c +
                                                       static_cast<size_t>(cur_d) * hi * wi * c +
                                                       static_cast<size_t>(cur_h) * wi * c +
                                                       static_cast<size_t>(cur_w) * c +
                                                       static_cast<size_t>(ic);

                                        size_t o_idx = static_cast<size_t>(in) * do_ * ho * wo * k +
                                                       static_cast<size_t>(ido) * ho * wo * k +
                                                       static_cast<size_t>(iho) * wo * k +
                                                       static_cast<size_t>(iwo) * k;

                                        value +=
                                            cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                            cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                                    }
                                    else
                                    {
                                        size_t i_idx = static_cast<size_t>(in) * in_strides[5] +
                                                       static_cast<size_t>(cur_d) * in_strides[4] +
                                                       static_cast<size_t>(cur_h) * in_strides[3] +
                                                       static_cast<size_t>(cur_w) * in_strides[2] +
                                                       static_cast<size_t>(ic) * in_strides[0];

                                        size_t o_idx = static_cast<size_t>(in) * out_strides[5] +
                                                       static_cast<size_t>(ido) * out_strides[4] +
                                                       static_cast<size_t>(iho) * out_strides[3] +
                                                       static_cast<size_t>(iwo) * out_strides[2];

                                        value +=
                                            cast_to<src_data_t, acc_data_t, use_tf32>(p_in[i_idx]) *
                                            cast_to<src_data_t, acc_data_t, use_tf32>(p_out[o_idx]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if constexpr(ASSUME_PACKED)
            {
                size_t f_idx = static_cast<size_t>(iz) * fy * fx * c_per_group +
                               static_cast<size_t>(iy) * fx * c_per_group +
                               static_cast<size_t>(ix) * c_per_group + static_cast<size_t>(ic);
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(value));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, value, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
                }
            }
            else
            {
                size_t f_idx = static_cast<size_t>(iz) * wei_strides[3] +
                               static_cast<size_t>(iy) * wei_strides[2] +
                               static_cast<size_t>(ix) * wei_strides[1] +
                               static_cast<size_t>(ic) * wei_strides[0];
                if constexpr(has_native_atomic_add<dst_data_t>::value)
                {
                    if(use_atomic)
                        naive_atomic_add(&p_wei[f_idx], cast_to<acc_data_t, dst_data_t>(value));
                    else
                        applyalphaBetaUpdate<dst_data_t, acc_data_t>(
                            p_wei, value, alpha, beta, f_idx);
                }
                else
                {
                    applyalphaBetaUpdate<dst_data_t, acc_data_t>(p_wei, value, alpha, beta, f_idx);
                }
            }
        }
    }
}

// double accumulator (GPU reference mode)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, float, double, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, float, double, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, half, double, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, hip_bfloat16, double, hip_bfloat16, 0)
// float accumulator (normal solver mode)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, float, float, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, float, float, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, half, float, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nchw, hip_bfloat16, float, hip_bfloat16, 0)

// double accumulator (GPU reference mode)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, float, double, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, float, double, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, half, double, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, hip_bfloat16, double, hip_bfloat16, 0)
// float accumulator (normal solver mode)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, float, float, float, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, float, float, float, 1)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, half, float, half, 0)
DEFINE_2D_NAIVE_CONV_KERNEL(wrw, nhwc, hip_bfloat16, float, hip_bfloat16, 0)

// double accumulator (GPU reference mode)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, float, double, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, float, double, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, half, double, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, hip_bfloat16, double, hip_bfloat16, 0)
// float accumulator (normal solver mode)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, float, float, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, float, float, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, half, float, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ncdhw, hip_bfloat16, float, hip_bfloat16, 0)

// double accumulator (GPU reference mode)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, float, double, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, float, double, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, half, double, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, hip_bfloat16, double, hip_bfloat16, 0)
// float accumulator (normal solver mode)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, float, float, float, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, float, float, float, 1)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, half, float, half, 0)
DEFINE_3D_NAIVE_CONV_KERNEL(wrw, ndhwc, hip_bfloat16, float, hip_bfloat16, 0)
