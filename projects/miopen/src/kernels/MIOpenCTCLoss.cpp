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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#endif
#include "float_types.h"

constexpr static FLOAT negative_cutoff_val = -1e20;

inline __device__ FLOAT LogAddExp(const FLOAT* x, const FLOAT* y)
{
    FLOAT a = max(*x, *y);
    FLOAT b = min(*x, *y);
    FLOAT c = b - a;

    return c <= negative_cutoff_val ? max(a, negative_cutoff_val)
                                    // We don't need the extra precision of log1pf() and it adds
                                    // performance overhead.
                                    // cppcheck-suppress unpreciseMathCall
                                    : max(a + logf(expf(b - a) + 1), negative_cutoff_val);
}

template <class T>
inline __device__ void NonAtomicLogAddExp(const unsigned int& lid,
                                          const int& label_length,
                                          const int* label_prime,
                                          const int& j1,
                                          const unsigned int& batch_id,
                                          const int& j,
                                          T* beta_buff0,
                                          T* beta_buff1,
                                          const unsigned int& label_prime_len,
                                          T* alpha_log,
                                          T* gradients)
{
    __syncthreads();

    if(lid == 0 || lid == 1)
    {
        for(int k = 0; k < label_length; k++)
        {
            int klid       = 2 * k + lid;
            int lb_cur     = lid == 0 ? BLANK_LB : *(label_prime + klid);
            size_t gidx    = j1 * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + lb_cur;
            T beta_temp    = j % 2 == 0 ? *(beta_buff0 + klid) : *(beta_buff1 + klid);
            size_t bidx_ts = j1 * label_prime_len + klid;

            beta_temp += *(alpha_log + bidx_ts);
            T grad_temp = gradients[gidx];

            gradients[gidx] = LogAddExp(&grad_temp, &beta_temp);
        }
    }
    if(lid == 0)
    {
        int k = 2 * label_length;

        size_t gidx    = j1 * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + BLANK_LB;
        T beta_temp    = j % 2 == 0 ? *(beta_buff0 + k) : *(beta_buff1 + k);
        size_t bidx_ts = j1 * label_prime_len + k;

        beta_temp += *(alpha_log + bidx_ts);
        T grad_temp = gradients[gidx];

        gradients[gidx] = LogAddExp(&grad_temp, &beta_temp);
    }
}

template <class T>
inline __device__ void AtomicLogAddExp(const int& j1,
                                       const unsigned int& batch_id,
                                       const unsigned int& lb_cur,
                                       const unsigned int& label_prime_len,
                                       const int& k1,
                                       const T* alpha_log,
                                       T* gradients,
                                       T& beta_temp)
{
    static_assert(false && "Method only implemented for FP32");
}

template <>
inline __device__ void AtomicLogAddExp(const int& j1,
                                       const unsigned int& batch_id,
                                       const unsigned int& lb_cur,
                                       const unsigned int& label_prime_len,
                                       const int& k1,
                                       const float* alpha_log,
                                       float* gradients,
                                       float& beta_temp)
{
    size_t gidx    = j1 * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + lb_cur;
    size_t bidx_ts = j1 * label_prime_len + k1;
    beta_temp += *(alpha_log + bidx_ts);

    float* addr = gradients + gidx;

    unsigned int prev_val_int, cur_val_int;
    memcpy(&cur_val_int, addr, sizeof(cur_val_int));

    do
    {
        prev_val_int = cur_val_int;
        float prev_val;
        memcpy(&prev_val, &prev_val_int, sizeof(prev_val));

        float a       = max(prev_val, beta_temp);
        float b       = min(prev_val, beta_temp);
        float c       = b - a;
        float new_val = c <= negative_cutoff_val
                            ? max(a, negative_cutoff_val)
                            // We don't need the extra precision of log1pf() and it adds performance
                            // overhead.
                            // cppcheck-suppress unpreciseMathCall
                            : max(a + logf(expf(b - a) + 1), negative_cutoff_val);

        unsigned int new_val_int;
        memcpy(&new_val_int, &new_val, sizeof(new_val));

        // We want atomicCAS operates on global memory, so use a reinterpret cast rather
        // then using memcpy to a threads private storage.
        // cppcheck-suppress invalidPointerCast
        cur_val_int = atomicCAS(reinterpret_cast<unsigned int*>(addr), prev_val_int, new_val_int);
    } while(cur_val_int != prev_val_int);
}

inline __device__ void CTCAlpha(const FLOAT* probs_logits,
                                const int* label_prime,
                                const unsigned int label_length,
                                const unsigned int input_length,
                                const unsigned int batch_id,
                                const unsigned int label_repeat,
                                FLOAT* alpha,
                                FLOAT* loss)
{
    unsigned int label_prime_len = 2 * label_length + 1;
    const unsigned int lid       = threadIdx.x;

    unsigned int aidx0 = label_length + label_repeat < input_length ? 0 : 1;
    unsigned int aidx1 = 1;
    for(unsigned int i = aidx0 + lid; i <= aidx1; i += WORK_PER_GRP)
    {
        unsigned int lb_cur = i % 2 == 0 ? BLANK_LB : *(label_prime + i);
        unsigned int pidx   = batch_id * PROBS_STRIDE1 + lb_cur;
        *(alpha + i)        = *(probs_logits + pidx);
    }
    __syncthreads();

    for(unsigned int j = 1; j < input_length; j++)
    {

        for(unsigned int i = lid; i <= label_prime_len - 1; i += WORK_PER_GRP)
        {
            unsigned int lb_cur = i % 2 == 0 ? BLANK_LB : *(label_prime + i);
            unsigned int lb_pre = i % 2 == 0 ? BLANK_LB : *(label_prime + i - 2);
            size_t pidx         = j * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + lb_cur;
            size_t aidx_ts      = j * label_prime_len + i;
            size_t aidx_t1s     = (j - 1) * label_prime_len + i;

            FLOAT alpha_t1s1 = *(alpha + aidx_t1s - 1);
            FLOAT alpha_t1s  = *(alpha + aidx_t1s);

            FLOAT alpha_ts = i == 0 ? alpha_t1s : LogAddExp(&alpha_t1s, &alpha_t1s1);
            if(i >= 2)
            {
                if(lb_cur != BLANK_LB && lb_cur != lb_pre)
                {
                    FLOAT alpha_t1s2 = *(alpha + aidx_t1s - 2);
                    alpha_ts         = LogAddExp(&alpha_ts, &alpha_t1s2);
                }
            }

            alpha_ts += *(probs_logits + pidx);
            *(alpha + aidx_ts) = max(alpha_ts, negative_cutoff_val);
        }
        __syncthreads();
    }

    if(lid == 0)
    {
        unsigned int alpha_size = input_length * label_prime_len;
        FLOAT alp0              = *(alpha + alpha_size - 1);
        FLOAT alp1              = *(alpha + alpha_size - 2);
        *loss                   = -LogAddExp(&alp0, &alp1);
    }
}

inline __device__ void CTCGradient(const FLOAT* probs_logits,
                                   const int* label_prime,
                                   const unsigned int label_length,
                                   const unsigned int input_length,
                                   const unsigned int batch_id,
                                   const unsigned int label_repeat,
                                   FLOAT* alpha_log,
                                   FLOAT* beta_buff0,
                                   FLOAT* beta_buff1,
                                   const FLOAT* loss,
                                   FLOAT* gradients)
{
    unsigned int label_prime_len = 2 * label_length + 1;

    FLOAT prob_lx_log = -*(loss);

    const unsigned int lid = threadIdx.x;

    unsigned int aidx0 = 1;
    unsigned int aidx1 = label_length + label_repeat < input_length ? 0 : 1;

    for(unsigned int j = 0; j < input_length; j++)
    {
        for(unsigned int i = lid; i < CLASS_SZ; i += WORK_PER_GRP)
        {
            *(gradients + j * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i) = negative_cutoff_val;
        }
    }
    __syncthreads();

    for(unsigned int k = aidx1 + lid; k <= aidx0; k += WORK_PER_GRP)
    {
        unsigned int k1     = label_prime_len - 1 - k;
        unsigned int lb_cur = k1 % 2 == 0 ? BLANK_LB : *(label_prime + k1);
        unsigned int pidx = (input_length - 1) * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + lb_cur;
        unsigned int gidx = (input_length - 1) * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + lb_cur;
        unsigned int bidx_ts = (input_length - 1) * label_prime_len + k1;

        FLOAT probs_logits_pidx = *(probs_logits + pidx);
        *(beta_buff0 + k1)      = probs_logits_pidx;

        FLOAT alpha_temp = *(alpha_log + bidx_ts);
        alpha_temp += probs_logits_pidx;
        FLOAT grad_temp = negative_cutoff_val;

        gradients[gidx] = LogAddExp(&grad_temp, &alpha_temp);
    }
    __syncthreads();

    for(int i = lid; i < CLASS_SZ; i += WORK_PER_GRP)
    {
        unsigned int pidx       = (input_length - 1) * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + i;
        unsigned int gidx       = (input_length - 1) * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i;
        FLOAT probs_logits_pidx = *(probs_logits + pidx);
        FLOAT grad_temp         = gradients[gidx];
        if constexpr(SOFTMAX_APPLIED == 0)
        {
            grad_temp -= probs_logits_pidx * 2;
        }
        else
        {
            grad_temp -= probs_logits_pidx;
        }
        grad_temp -= prob_lx_log;
        grad_temp = grad_temp <= negative_cutoff_val ? 0 : expf(grad_temp);

        if constexpr(SOFTMAX_APPLIED == 1)
        {
            *(gradients + gidx) = expf(probs_logits_pidx) - grad_temp;
        }
        else
        {
            *(gradients + gidx) = -grad_temp;
        }
    }
    __syncthreads();

    for(unsigned int j = 1; j < input_length; j++)
    {
        int j1 = input_length - 1 - j;

        for(int k = lid; k <= label_prime_len - 1; k += WORK_PER_GRP)
        {
            int k1     = label_prime_len - 1 - k;
            int lb_cur = k1 % 2 == 0 ? BLANK_LB : *(label_prime + k1);
            int lb_pre = k1 % 2 == 0 ? BLANK_LB : *(label_prime + k1 + 2);

            size_t pidx = j1 * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + lb_cur;

            FLOAT beta_temp = j % 2 == 0 ? *(beta_buff1 + k1) : *(beta_buff0 + k1);

            if(k1 <= label_prime_len - 2)
            {
                FLOAT beta_temp1 = j % 2 == 0 ? *(beta_buff1 + k1 + 1) : *(beta_buff0 + k1 + 1);
                beta_temp        = LogAddExp(&beta_temp, &beta_temp1);
            }
            if(k1 <= label_prime_len - 3)
            {
                if(lb_cur != BLANK_LB && lb_cur != lb_pre)
                {
                    FLOAT beta_temp2 = j % 2 == 0 ? *(beta_buff1 + k1 + 2) : *(beta_buff0 + k1 + 2);
                    beta_temp        = LogAddExp(&beta_temp, &beta_temp2);
                }
            }

            beta_temp += *(probs_logits + pidx);
            beta_temp = max(beta_temp, negative_cutoff_val);
            if(j % 2 == 0)
            {
                *(beta_buff0 + k1) = beta_temp;
            }
            else
            {
                *(beta_buff1 + k1) = beta_temp;
            }

            if constexpr(OPT_ATOMIC_LOGADDEXP == 1)
            {
                AtomicLogAddExp(
                    j1, batch_id, lb_cur, label_prime_len, k1, alpha_log, gradients, beta_temp);
            }
        }

        if constexpr(OPT_ATOMIC_LOGADDEXP == 0)
        {
            NonAtomicLogAddExp(lid,
                               label_length,
                               label_prime,
                               j1,
                               batch_id,
                               j,
                               beta_buff0,
                               beta_buff1,
                               label_prime_len,
                               alpha_log,
                               gradients);
        }

        __syncthreads();

        for(int i = lid; i < CLASS_SZ; i += WORK_PER_GRP)
        {
            size_t pidx = j1 * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + i;
            size_t gidx = j1 * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i;

            FLOAT probs_logits_pidx = *(probs_logits + pidx);

            FLOAT grad_temp = gradients[gidx];

            if constexpr(SOFTMAX_APPLIED == 0)
            {
                grad_temp -= probs_logits_pidx * 2;
            }
            else
            {
                grad_temp -= probs_logits_pidx;
            }

            grad_temp -= prob_lx_log;
            grad_temp = grad_temp <= negative_cutoff_val ? 0 : expf(grad_temp);

            if constexpr(SOFTMAX_APPLIED == 1)
            {
                *(gradients + gidx) = expf(probs_logits_pidx) - grad_temp;
            }
            else
            {
                *(gradients + gidx) = -grad_temp;
            }
        }
        __syncthreads();
    }
}

template <bool OptLclMemBeta, bool OptLclMemLb>
__forceinline__ __device__ void CTCLoss(const unsigned int lid,
                                        const unsigned int gid,
                                        const unsigned int grp_id,
                                        const FLOAT* probs_logits,
                                        FLOAT* workSpace,
                                        int* dim_data,
                                        FLOAT* losses,
                                        FLOAT* gradients);

template <>
__forceinline__ __device__ void CTCLoss<false, false>(const unsigned int lid,
                                                      const unsigned int gid,
                                                      const unsigned int grp_id,
                                                      const FLOAT* probs_logits,
                                                      FLOAT* workSpace,
                                                      int* dim_data,
                                                      FLOAT* losses,
                                                      FLOAT* gradients)
{
    for(unsigned int bid = grp_id; bid < BATCH_SZ; bid += GRP_NUM)
    {
        unsigned int input_len     = *(dim_data + bid);
        unsigned int label_len     = *(dim_data + BATCH_SZ + bid);
        unsigned int label_offsets = *(dim_data + 2 * BATCH_SZ + bid);
        unsigned int label_repeat  = *(dim_data + 3 * BATCH_SZ + bid);

        for(unsigned int i = lid; i < label_len; i += WORK_PER_GRP)
        {
            dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN + 2 * i + 1] =
                dim_data[4 * BATCH_SZ + label_offsets + i];
        }

        for(unsigned int i = lid; i < MAX_TSTEP * MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(workSpace + ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN + i) = negative_cutoff_val;
        }

        for(unsigned int i = lid; i < 2 * MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(workSpace + BETA_OFFSET + bid * 2 * MAX_S_LEN + i) = negative_cutoff_val;
        }

        __syncthreads();

        CTCAlpha(probs_logits,
                 &dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN],
                 label_len,
                 input_len,
                 bid,
                 label_repeat,
                 &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
                 &losses[bid]);

        __syncthreads();

        CTCGradient(probs_logits,
                    &dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN],
                    label_len,
                    input_len,
                    bid,
                    label_repeat,
                    &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
                    &workSpace[BETA_OFFSET + bid * 2 * MAX_S_LEN],
                    &workSpace[BETA_OFFSET + (bid * 2 + 1) * MAX_S_LEN],
                    &losses[bid],
                    gradients);
    }
}

template <>
__forceinline__ __device__ void CTCLoss<true, false>(const unsigned int lid,
                                                     const unsigned int gid,
                                                     const unsigned int grp_id,
                                                     const FLOAT* probs_logits,
                                                     FLOAT* workSpace,
                                                     int* dim_data,
                                                     FLOAT* losses,
                                                     FLOAT* gradients)
{
    __shared__ FLOAT beta0[MAX_S_LEN];
    __shared__ FLOAT beta1[MAX_S_LEN];

    for(unsigned int bid = grp_id; bid < BATCH_SZ; bid += GRP_NUM)
    {
        unsigned int input_len     = *(dim_data + bid);
        unsigned int label_len     = *(dim_data + BATCH_SZ + bid);
        unsigned int label_offsets = *(dim_data + 2 * BATCH_SZ + bid);
        unsigned int label_repeat  = *(dim_data + 3 * BATCH_SZ + bid);

        for(unsigned int i = lid; i < label_len; i += WORK_PER_GRP)
        {
            dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN + 2 * i + 1] =
                dim_data[4 * BATCH_SZ + label_offsets + i];
        }

        for(unsigned int i = lid; i < MAX_TSTEP * MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(workSpace + ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN + i) = negative_cutoff_val;
        }

        __syncthreads();

        CTCAlpha(probs_logits,
                 &dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN],
                 label_len,
                 input_len,
                 bid,
                 label_repeat,
                 &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
                 &losses[bid]);

        for(unsigned int i = lid; i < MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(beta0 + i) = negative_cutoff_val;
            *(beta1 + i) = negative_cutoff_val;
        }

        __syncthreads();

        CTCGradient(probs_logits,
                    &dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN],
                    label_len,
                    input_len,
                    bid,
                    label_repeat,
                    &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
                    &beta0[0],
                    &beta1[0],
                    &losses[bid],
                    gradients);
    }
}

template <>
__forceinline__ __device__ void CTCLoss<false, true>(const unsigned int lid,
                                                     const unsigned int gid,
                                                     const unsigned int grp_id,
                                                     const FLOAT* probs_logits,
                                                     FLOAT* workSpace,
                                                     int* dim_data,
                                                     FLOAT* losses,
                                                     FLOAT* gradients)
{
    __shared__ int lb_prime[MAX_S_LEN];

    for(unsigned int bid = grp_id; bid < BATCH_SZ; bid += GRP_NUM)
    {
        unsigned int input_len     = *(dim_data + bid);
        unsigned int label_len     = *(dim_data + BATCH_SZ + bid);
        unsigned int label_offsets = *(dim_data + 2 * BATCH_SZ + bid);
        unsigned int label_repeat  = *(dim_data + 3 * BATCH_SZ + bid);

        for(unsigned int i = lid; i < label_len; i += WORK_PER_GRP)
        {
            lb_prime[2 * i + 1] = dim_data[4 * BATCH_SZ + label_offsets + i];
        }

        for(unsigned int i = lid; i < MAX_TSTEP * MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(workSpace + ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN + i) = negative_cutoff_val;
        }

        for(unsigned int i = lid; i < 2 * MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(workSpace + BETA_OFFSET + bid * 2 * MAX_S_LEN + i) = negative_cutoff_val;
        }

        __syncthreads();

        CTCAlpha(probs_logits,
                 &lb_prime[0],
                 label_len,
                 input_len,
                 bid,
                 label_repeat,
                 &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
                 &losses[bid]);

        __syncthreads();

        CTCGradient(probs_logits,
                    &lb_prime[0],
                    label_len,
                    input_len,
                    bid,
                    label_repeat,
                    &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
                    &workSpace[BETA_OFFSET + bid * 2 * MAX_S_LEN],
                    &workSpace[BETA_OFFSET + (bid * 2 + 1) * MAX_S_LEN],
                    &losses[bid],
                    gradients);
    }
}

template <>
__forceinline__ __device__ void CTCLoss<true, true>(const unsigned int lid,
                                                    const unsigned int gid,
                                                    const unsigned int grp_id,
                                                    const FLOAT* probs_logits,
                                                    FLOAT* workSpace,
                                                    int* dim_data,
                                                    FLOAT* losses,
                                                    FLOAT* gradients)
{
    __shared__ FLOAT beta0[MAX_S_LEN];
    __shared__ FLOAT beta1[MAX_S_LEN];
    __shared__ int lb_prime[MAX_S_LEN];

    for(unsigned int bid = grp_id; bid < BATCH_SZ; bid += GRP_NUM)
    {
        unsigned int input_len     = *(dim_data + bid);
        unsigned int label_len     = *(dim_data + BATCH_SZ + bid);
        unsigned int label_offsets = *(dim_data + 2 * BATCH_SZ + bid);
        unsigned int label_repeat  = *(dim_data + 3 * BATCH_SZ + bid);

        for(unsigned int i = lid; i < label_len; i += WORK_PER_GRP)
        {
            lb_prime[2 * i + 1] = dim_data[4 * BATCH_SZ + label_offsets + i];
        }

        for(unsigned int i = lid; i < MAX_TSTEP * MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(workSpace + ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN + i) = negative_cutoff_val;
        }

        __syncthreads();

        CTCAlpha(probs_logits,
                 &lb_prime[0],
                 label_len,
                 input_len,
                 bid,
                 label_repeat,
                 &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
                 &losses[bid]);

        for(unsigned int i = lid; i < MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(beta0 + i) = negative_cutoff_val;
            *(beta1 + i) = negative_cutoff_val;
        }

        __syncthreads();

        CTCGradient(probs_logits,
                    &lb_prime[0],
                    label_len,
                    input_len,
                    bid,
                    label_repeat,
                    &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
                    &beta0[0],
                    &beta1[0],
                    &losses[bid],
                    gradients);
    }
}

extern "C" __global__ void CTCLossGPU([[maybe_unused]] const FLOAT* probs,
                                      FLOAT* workSpace,
                                      int* dim_data,
                                      FLOAT* losses,
                                      FLOAT* gradients)
{
    const unsigned int lid    = threadIdx.x;
    const unsigned int gid    = blockIdx.x * blockDim.x + lid;
    const unsigned int grp_id = gid / WORK_PER_GRP;

    const FLOAT* probs_logits;
    if constexpr(SOFTMAX_APPLIED == 1)
    {
        probs_logits = &workSpace[PROBLOG_OFFSET];
    }
    else
    {
        probs_logits = probs;
    }

    CTCLoss<OPT_LCL_MEM_BETA, OPT_LCL_MEM_LB>(
        lid, gid, grp_id, probs_logits, workSpace, dim_data, losses, gradients);
}
