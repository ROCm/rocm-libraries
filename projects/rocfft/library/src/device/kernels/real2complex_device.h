// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef REAL_TO_COMPLEX_DEVICE_H
#define REAL_TO_COMPLEX_DEVICE_H

// The even-length real to complex post process device kernel
template <typename Tcomplex, bool Ndiv4, CallbackType cbtype, bool SCALE = false>
__device__ inline void post_process_interleaved(const size_t    idx_p,
                                                const size_t    idx_q,
                                                const size_t    half_N,
                                                const size_t    quarter_N,
                                                const Tcomplex* input,
                                                Tcomplex*       output,
                                                size_t          output_base,
                                                const Tcomplex* twiddles,
                                                void* __restrict__ load_cb_fn,
                                                void* __restrict__ load_cb_data,
                                                uint32_t load_cb_lds_bytes,
                                                void* __restrict__ store_cb_fn,
                                                void* __restrict__ store_cb_data,
                                                const real_type_t<Tcomplex> scale_factor = 0.0)
{
    // post process can't be the first kernel, so don't bother
    // going through the load cb to read global memory
    auto store_cb = get_store_cb<Tcomplex, cbtype>(store_cb_fn);

    Tcomplex outval;

    if(idx_p == 0)
    {
        outval.x = input[0].x - input[0].y;
        outval.y = 0;
        store_cb(output,
                 output_base + half_N,
                 SCALE ? (outval * scale_factor) : outval,
                 store_cb_data,
                 nullptr);

        outval.x = input[0].x + input[0].y;
        outval.y = 0;
        store_cb(output,
                 output_base + 0,
                 SCALE ? (outval * scale_factor) : outval,
                 store_cb_data,
                 nullptr);

        if(Ndiv4)
        {
            outval.x = input[quarter_N].x;
            outval.y = -input[quarter_N].y;

            store_cb(output,
                     output_base + quarter_N,
                     SCALE ? (outval * scale_factor) : outval,
                     store_cb_data,
                     nullptr);
        }
    }
    else
    {
        const Tcomplex p = input[idx_p];
        const Tcomplex q = input[idx_q];
        const Tcomplex u = 0.5 * (p + q);
        const Tcomplex v = 0.5 * (p - q);

        const Tcomplex twd_p = twiddles[idx_p];
        // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

        outval.x = u.x + v.x * twd_p.y + u.y * twd_p.x;
        outval.y = v.y + u.y * twd_p.y - v.x * twd_p.x;
        store_cb(output,
                 output_base + idx_p,
                 SCALE ? (outval * scale_factor) : outval,
                 store_cb_data,
                 nullptr);

        outval.x = u.x - v.x * twd_p.y - u.y * twd_p.x;
        outval.y = -v.y + u.y * twd_p.y - v.x * twd_p.x;
        store_cb(output,
                 output_base + idx_q,
                 SCALE ? (outval * scale_factor) : outval,
                 store_cb_data,
                 nullptr);
    }
}

// TODO: rework pre/post processing
template <typename T, bool Ndiv4, CallbackType cbtype>
__device__ inline void post_process_interleaved_inplace(const size_t idx_p,
                                                        const size_t idx_q,
                                                        const size_t half_N,
                                                        const size_t quarter_N,
                                                        T*           inout,
                                                        size_t       offset_base,
                                                        const T*     twiddles,
                                                        void* __restrict__ load_cb_fn,
                                                        void* __restrict__ load_cb_data,
                                                        uint32_t load_cb_lds_bytes,
                                                        void* __restrict__ store_cb_fn,
                                                        void* __restrict__ store_cb_data)
{
    // post process can't be the first kernel, so don't bother
    // going through the load cb to read global memory
    auto store_cb = get_store_cb<T, cbtype>(store_cb_fn);

    T p, q, outval;
    if(idx_p < quarter_N)
    {
        p = inout[offset_base + idx_p];
        q = inout[offset_base + idx_q];
    }

    __syncthreads();

    if(idx_p == 0)
    {
        outval.x = p.x + p.y;
        outval.y = 0;
        store_cb(inout, offset_base + idx_p, outval, store_cb_data, nullptr);

        outval.x = p.x - p.y;
        outval.y = 0;
        store_cb(inout, offset_base + idx_q, outval, store_cb_data, nullptr);

        if(Ndiv4)
        {
            outval   = inout[offset_base + quarter_N];
            outval.y = -outval.y;
            store_cb(inout, offset_base + quarter_N, outval, store_cb_data, nullptr);
        }
    }
    else if(idx_p < quarter_N)
    {
        const T u = 0.5 * (p + q);
        const T v = 0.5 * (p - q);

        const T twd_p = twiddles[idx_p];
        // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

        outval.x = u.x + v.x * twd_p.y + u.y * twd_p.x;
        outval.y = v.y + u.y * twd_p.y - v.x * twd_p.x;
        store_cb(inout, offset_base + idx_p, outval, store_cb_data, nullptr);

        outval.x = u.x - v.x * twd_p.y - u.y * twd_p.x;
        outval.y = -v.y + u.y * twd_p.y - v.x * twd_p.x;
        store_cb(inout, offset_base + idx_q, outval, store_cb_data, nullptr);
    }
}

// The below 2 functions are only for inplace in lds. So no callback.
template <typename Tcomplex, bool Ndiv4>
__device__ inline void real_post_process_kernel_inplace(const size_t    idx_p,
                                                        const size_t    idx_q,
                                                        const size_t    quarter_N,
                                                        Tcomplex*       inout,
                                                        size_t          offset_base,
                                                        const Tcomplex* twiddles)
{
    if(idx_p < quarter_N)
    {
        Tcomplex p = inout[offset_base + idx_p];
        Tcomplex q = inout[offset_base + idx_q];

        if(idx_p == 0)
        {
            inout[offset_base + idx_p].x = p.x + p.y;
            inout[offset_base + idx_p].y = 0;

            inout[offset_base + idx_q].x = p.x - p.y;
            inout[offset_base + idx_q].y = 0;

            if(Ndiv4)
            {
                inout[offset_base + quarter_N].y = -inout[offset_base + quarter_N].y;
            }
        }
        else
        {
            const Tcomplex u = 0.5 * (p + q);
            const Tcomplex v = 0.5 * (p - q);

            const Tcomplex twd_p = twiddles[idx_p];
            // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

            inout[offset_base + idx_p].x = u.x + v.x * twd_p.y + u.y * twd_p.x;
            inout[offset_base + idx_p].y = v.y + u.y * twd_p.y - v.x * twd_p.x;

            inout[offset_base + idx_q].x = u.x - v.x * twd_p.y - u.y * twd_p.x;
            inout[offset_base + idx_q].y = -v.y + u.y * twd_p.y - v.x * twd_p.x;
        }
    }
}

template <typename Tcomplex, bool Ndiv4>
__device__ inline void real_pre_process_kernel_inplace(const size_t    idx_p,
                                                       const size_t    idx_q,
                                                       const size_t    quarter_N,
                                                       Tcomplex*       inout,
                                                       size_t          offset_base,
                                                       const Tcomplex* twiddles)
{
    if(idx_p < quarter_N)
    {
        Tcomplex p = inout[offset_base + idx_p];
        Tcomplex q = inout[offset_base + idx_q];

        if(idx_p == 0)
        {
            // NB: multi-dimensional transforms may have non-zero
            // imaginary part at index 0 or at the Nyquist frequency.
            inout[offset_base + idx_p].x = p.x + q.x;
            inout[offset_base + idx_p].y = p.x - q.x;

            if(Ndiv4)
            {
                auto quarter_elem                = inout[offset_base + quarter_N];
                inout[offset_base + quarter_N].x = 2.0 * quarter_elem.x;
                inout[offset_base + quarter_N].y = -2.0 * quarter_elem.y;
            }
        }
        else
        {
            const Tcomplex u = p + q;
            const Tcomplex v = p - q;

            const Tcomplex twd_p = twiddles[idx_p];
            // NB: twd_q = -conj(twd_p);

            inout[offset_base + idx_p].x = u.x + v.x * twd_p.y - u.y * twd_p.x;
            inout[offset_base + idx_p].y = v.y + u.y * twd_p.y + v.x * twd_p.x;

            inout[offset_base + idx_q].x = u.x - v.x * twd_p.y + u.y * twd_p.x;
            inout[offset_base + idx_q].y = -v.y + u.y * twd_p.y + v.x * twd_p.x;
        }
    }
}

#endif
