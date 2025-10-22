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
#include <hip/hip_runtime.h>
#include "float_types.h"

#ifndef NEGATIVE_CUTOFF_VAL
#define NEGATIVE_CUTOFF_VAL (FLOAT)(-1e20)
#endif

#ifndef SOFTMAX_LEN
#define SOFTMAX_LEN 1
#endif

#ifndef SOFTMAX_APPLIED
#define SOFTMAX_APPLIED 1
#endif

#ifndef PROBS_STRIDE0
#define PROBS_STRIDE0 (BATCH_SZ * CLASS_SZ)
#endif
#ifndef PROBS_STRIDE1
#define PROBS_STRIDE1 CLASS_SZ
#endif

#if SOFTMAX_APPLIED == 1
#define USE_PROBS_STRIDE0 (BATCH_SZ * CLASS_SZ)
#define USE_PROBS_STRIDE1 CLASS_SZ
#else
#define USE_PROBS_STRIDE0 PROBS_STRIDE0
#define USE_PROBS_STRIDE1 PROBS_STRIDE1
#endif

#ifndef GRADS_STRIDE0
#define GRADS_STRIDE0 (BATCH_SZ * CLASS_SZ)
#endif
#ifndef GRADS_STRIDE1
#define GRADS_STRIDE1 CLASS_SZ
#endif

#ifndef BLANK_LB_ID
#define BLANK_LB 0
#elif BLANK_LB_ID < 0
#define BLANK_LB 0
#elif BLANK_LB_ID >= CLASS_SZ
#define BLANK_LB (CLASS_SZ - 1)
#else
#define BLANK_LB BLANK_LB_ID
#endif

#ifdef OPT_LCL_MEM_LB
#define ADDRSPACE_LB __shared__
#else
#define ADDRSPACE_LB
#endif

#ifdef OPT_LCL_MEM_BETA
#define ADDRSPACE_BETA __shared__
#else
#define ADDRSPACE_BETA
#endif

inline __device__ FLOAT LogAddExp(const FLOAT* x, const FLOAT* y)
{
    FLOAT a = max(*x, *y);
    FLOAT b = min(*x, *y);
    FLOAT c = b - a;

    return c <= NEGATIVE_CUTOFF_VAL ? max(a, NEGATIVE_CUTOFF_VAL)
                                    : max(a + log1pf(expf(b - a)), NEGATIVE_CUTOFF_VAL);
}

#ifdef OPT_ATOMIC_LOGADDEXP
template <class T>
struct TypeTraits
{
};

template <>
struct TypeTraits<float>
{
    using UnsignedIntType = unsigned int;
};

inline __device__ void AtomicLogAddExp(FLOAT* addr, const float operand)
{
    using UINT = TypeTraits<FLOAT>::UnsignedIntType;
    UINT prev_val_int, cur_val_int;
    memcpy(&cur_val_int, addr, sizeof(cur_val_int));

    do
    {
        prev_val_int = cur_val_int;
        FLOAT prev_val;
        memcpy(&prev_val, &prev_val_int, sizeof(prev_val));

        FLOAT a       = max(prev_val, operand);
        FLOAT b       = min(prev_val, operand);
        FLOAT c       = b - a;
        FLOAT new_val = c <= NEGATIVE_CUTOFF_VAL
                            ? max(a, NEGATIVE_CUTOFF_VAL)
                            : max(a + log1pf(expf(b - a)), NEGATIVE_CUTOFF_VAL);

        UINT new_val_int;
        memcpy(&new_val_int, &new_val, sizeof(new_val));

        // We want atomicCAS operates on global memory, so use a reinterpret cast rather
        // then using memcpy to a threads private storage.
        // cppcheck-suppress invalidPointerCast
        cur_val_int = atomicCAS(reinterpret_cast<UINT*>(addr), prev_val_int, new_val_int);
    } while(cur_val_int != prev_val_int);
}
#endif

inline __device__ void CTCAlpha(const FLOAT* probs_logits,
                                const ADDRSPACE_LB int* label_prime,
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
        unsigned int pidx   = batch_id * USE_PROBS_STRIDE1 + lb_cur;
        *(alpha + i)        = *(probs_logits + pidx);
    }
    __syncthreads();

    for(unsigned int j = 1; j < input_length; j++)
    {

        for(unsigned int i = lid; i <= label_prime_len - 1; i += WORK_PER_GRP)
        {
            unsigned int lb_cur = i % 2 == 0 ? BLANK_LB : *(label_prime + i);
            unsigned int lb_pre = i % 2 == 0 ? BLANK_LB : *(label_prime + i - 2);
            size_t pidx         = j * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + lb_cur;
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
            *(alpha + aidx_ts) = max(alpha_ts, NEGATIVE_CUTOFF_VAL);
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
                                   const ADDRSPACE_LB int* label_prime,
                                   const unsigned int label_length,
                                   const unsigned int input_length,
                                   const unsigned int batch_id,
                                   const unsigned int label_repeat,
                                   FLOAT* alpha_log,
                                   ADDRSPACE_BETA FLOAT* beta_buff0,
                                   ADDRSPACE_BETA FLOAT* beta_buff1,
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
            *(gradients + j * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i) = NEGATIVE_CUTOFF_VAL;
        }
    }
    __syncthreads();

    for(unsigned int k = aidx1 + lid; k <= aidx0; k += WORK_PER_GRP)
    {
        unsigned int k1     = label_prime_len - 1 - k;
        unsigned int lb_cur = k1 % 2 == 0 ? BLANK_LB : *(label_prime + k1);
        unsigned int pidx =
            (input_length - 1) * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + lb_cur;
        unsigned int gidx = (input_length - 1) * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + lb_cur;
        unsigned int bidx_ts = (input_length - 1) * label_prime_len + k1;

        FLOAT probs_logits_pidx = *(probs_logits + pidx);
        *(beta_buff0 + k1)      = probs_logits_pidx;

        FLOAT alpha_temp = *(alpha_log + bidx_ts);
        alpha_temp += probs_logits_pidx;
        FLOAT grad_temp = NEGATIVE_CUTOFF_VAL;

        gradients[gidx] = LogAddExp(&grad_temp, &alpha_temp);
    }
    __syncthreads();

    for(int i = lid; i < CLASS_SZ; i += WORK_PER_GRP)
    {
        unsigned int pidx =
            (input_length - 1) * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + i;
        unsigned int gidx       = (input_length - 1) * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i;
        FLOAT probs_logits_pidx = *(probs_logits + pidx);
        FLOAT grad_temp         = gradients[gidx];
        grad_temp -= probs_logits_pidx
#if SOFTMAX_APPLIED == 0
                     * 2
#endif
            ;
        grad_temp -= prob_lx_log;
        grad_temp = grad_temp <= NEGATIVE_CUTOFF_VAL ? 0 : expf(grad_temp);

        *(gradients + gidx) =
#if SOFTMAX_APPLIED == 1
            expf(probs_logits_pidx)
#endif
            - grad_temp;
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

            size_t pidx = j1 * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + lb_cur;

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
            beta_temp = max(beta_temp, NEGATIVE_CUTOFF_VAL);
            if(j % 2 == 0)
            {
                *(beta_buff0 + k1) = beta_temp;
            }
            else
            {
                *(beta_buff1 + k1) = beta_temp;
            }

#ifdef OPT_ATOMIC_LOGADDEXP
            size_t gidx    = j1 * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + lb_cur;
            size_t bidx_ts = j1 * label_prime_len + k1;
            beta_temp += *(alpha_log + bidx_ts);

            AtomicLogAddExp(gradients + gidx, beta_temp);
#else
        }
        __syncthreads();

        if(lid == 0 || lid == 1)
        {
            for(int k = 0; k < label_length; k++)
            {
                int klid        = 2 * k + lid;
                int lb_cur      = lid == 0 ? BLANK_LB : *(label_prime + klid);
                size_t gidx     = j1 * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + lb_cur;
                FLOAT beta_temp = j % 2 == 0 ? *(beta_buff0 + klid) : *(beta_buff1 + klid);
                size_t bidx_ts  = j1 * label_prime_len + klid;

                beta_temp += *(alpha_log + bidx_ts);
                FLOAT grad_temp = gradients[gidx];

                gradients[gidx] = LogAddExp(&grad_temp, &beta_temp);
            }
        }
        if(lid == 0)
        {
            int k = 2 * label_length;

            size_t gidx     = j1 * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + BLANK_LB;
            FLOAT beta_temp = j % 2 == 0 ? *(beta_buff0 + k) : *(beta_buff1 + k);
            size_t bidx_ts  = j1 * label_prime_len + k;

            beta_temp += *(alpha_log + bidx_ts);
            FLOAT grad_temp = gradients[gidx];

            gradients[gidx] = LogAddExp(&grad_temp, &beta_temp);
#endif
        }
        __syncthreads();

        for(int i = lid; i < CLASS_SZ; i += WORK_PER_GRP)
        {
            size_t pidx = j1 * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + i;
            size_t gidx = j1 * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i;

            FLOAT probs_logits_pidx = *(probs_logits + pidx);

            FLOAT grad_temp = gradients[gidx];

            grad_temp -= probs_logits_pidx
#if SOFTMAX_APPLIED == 0
                         * 2
#endif
                ;
            grad_temp -= prob_lx_log;
            grad_temp = grad_temp <= NEGATIVE_CUTOFF_VAL ? 0 : expf(grad_temp);

            *(gradients + gidx) =
#if SOFTMAX_APPLIED == 1
                expf(probs_logits_pidx)
#endif
                - grad_temp;
        }
        __syncthreads();
    }
}

extern "C" __global__ void
CTCLossGPU(const FLOAT* probs, FLOAT* workSpace, int* dim_data, FLOAT* losses, FLOAT* gradients)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + lid;

    unsigned int grp_id = gid / WORK_PER_GRP;

#ifdef OPT_LCL_MEM_BETA
    __shared__ FLOAT beta0[MAX_S_LEN];
    __shared__ FLOAT beta1[MAX_S_LEN];
#endif

#ifdef OPT_LCL_MEM_LB
    __shared__ int lb_prime[MAX_S_LEN];
#endif

    for(unsigned int bid = grp_id; bid < BATCH_SZ; bid += GRP_NUM)
    {
        unsigned int input_len     = *(dim_data + bid);
        unsigned int label_len     = *(dim_data + BATCH_SZ + bid);
        unsigned int label_offsets = *(dim_data + 2 * BATCH_SZ + bid);
        unsigned int label_repeat  = *(dim_data + 3 * BATCH_SZ + bid);

        for(unsigned int i = lid; i < label_len; i += WORK_PER_GRP)
        {
#ifdef OPT_LCL_MEM_LB
            lb_prime[
#else
            dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN +
#endif
                2 * i + 1] = dim_data[4 * BATCH_SZ + label_offsets + i];
        }

        for(unsigned int i = lid; i < MAX_TSTEP * MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(workSpace + ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN + i) = NEGATIVE_CUTOFF_VAL;
        }

#ifndef OPT_LCL_MEM_BETA
        for(unsigned int i = lid; i < 2 * MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(workSpace + BETA_OFFSET + bid * 2 * MAX_S_LEN + i) = NEGATIVE_CUTOFF_VAL;
        }
#endif

        __syncthreads();

        CTCAlpha(
#if SOFTMAX_APPLIED == 1
            &workSpace[PROBLOG_OFFSET]
#else
            probs
#endif
            ,
#ifdef OPT_LCL_MEM_LB
            &lb_prime[0]
#else
            &dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN]
#endif
            ,
            label_len,
            input_len,
            bid,
            label_repeat,
            &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
            &losses[bid]);

#ifdef OPT_LCL_MEM_BETA
        for(unsigned int i = lid; i < MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(beta0 + i) = NEGATIVE_CUTOFF_VAL;
            *(beta1 + i) = NEGATIVE_CUTOFF_VAL;
        }
#endif

        __syncthreads();

        CTCGradient(
#if SOFTMAX_APPLIED == 1
            &workSpace[PROBLOG_OFFSET]
#else
            probs
#endif
            ,
#ifdef OPT_LCL_MEM_LB
            &lb_prime[0]
#else
            &dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN]
#endif
            ,
            label_len,
            input_len,
            bid,
            label_repeat,
            &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
#ifdef OPT_LCL_MEM_BETA
            &beta0[0],
            &beta1[0]
#else
            &workSpace[BETA_OFFSET + bid * 2 * MAX_S_LEN],
            &workSpace[BETA_OFFSET + (bid * 2 + 1) * MAX_S_LEN]
#endif
            ,
            &losses[bid],
            gradients);
    }

    (void)probs;
}
