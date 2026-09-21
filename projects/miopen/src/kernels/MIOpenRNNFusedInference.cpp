// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Fused single-layer LSTM forward-INFERENCE recurrent loop (one cooperative
// launch per direction). Consumes MIOpen's existing workSpace layout: the
// batched input-projection GEMM + bias have already written the i,f,o,c gate
// pre-activations (from x) into workSpace. This kernel adds the recurrent
// contribution W_hh * h_{t-1} and applies the LSTM cell update, walking the
// whole sequence on-device with a grid-wide barrier per timestep — replacing
// the host-side per-timestep loop (~2 launches/step) that otherwise dominates
// at small batch.
//
// Supports UNIFORM batch (all sequences same length, max_batch rows per timestep).
// The recurrence is independent across batch, so a single grid barrier per timestep
// still suffices; work is mapped over (gate_row, batch) and (hidden, batch).
//
// GATED OFF BY DEFAULT in host code (MIOPEN_DEBUG_RNN_FUSED_INFERENCE). Only
// engaged for: single layer, LSTM, miopenRNNdefault, fp32, no dropout, packed,
// uniform batch across time. Falls back otherwise.
//
// workSpace layout. Within timestep t the max_batch rows are contiguous at
// row index (t*max_batch + b); each row is hy_stride floats. For batch b,
// direction ri (gate order matches LSTMForwardHiddenStateUpdate: i,f,o,c):
//   rb     = (t*max_batch + b)*hy_stride
//   base   = rb + ri*wei_len
//   i/f/o gate = base + {0,1,2}*hy_h (sigmoid) ; c gate = base + 3*hy_h (tanh)
//   hidden = rb + hid_off + ri*hy_h   (hid_off = bi*hy_h*5)
//   cell   = rb + bi*wei_len + ri*hy_h
//   hx/cx (initial hidden/cell) and hy/cy (final) = hcx_offset + b*hy_h
// hy_stride = hy_h*bi*6 ; wei_len = 4*hy_h ; W_hh packed in w at wei_shift_dir.

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#endif
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#ifndef MIO_RNN_FINF_WAVE
#define MIO_RNN_FINF_WAVE 32
#endif

__device__ __forceinline__ float finf_sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

// One cooperative grid; handles uniform batch (max_batch rows/timestep).
//   workSpace : MIOpen RNN workspace (gates pre-filled with input projection+bias)
//   w         : packed weights; W_hh for this direction begins at wei_shift_dir
//   hx/cx     : initial hidden/cell (h_{-1}/c_{-1}); skipped when use_hx/use_cx==0
//   hy/cy     : final hidden/cell out (h_{T-1}/c_{T-1}); written when use_hy/use_cy
//   max_batch : rows per timestep (B); recurrence is independent across batch
//   hy_h      : hidden size H
//   hy_stride : workspace row stride (= H*bi*6)
//   wei_len   : 4*H
//   hid_off   : bi*H*5
//   uni_stride: H  (W_hh row stride)
//   bi        : 1 or 2
//   ri        : direction index (0 fwd, 1 rev) for column placement
extern "C" __global__ void RNNFusedLSTMInfer(float* __restrict__ workSpace,
                                             const float* __restrict__ w,
                                             const float* __restrict__ cx,
                                             const float* __restrict__ hx,
                                             float* __restrict__ hy,
                                             float* __restrict__ cy,
                                             int seqLen,
                                             int max_batch,
                                             int hy_h,
                                             int hy_stride,
                                             int wei_len,
                                             int hid_off,
                                             int uni_stride,
                                             int bi,
                                             int ri,
                                             int reverse,
                                             long long wei_shift_dir,
                                             long long hcx_offset,
                                             int use_cx,
                                             int use_hx,
                                             int use_hy,
                                             int use_cy)
{
    cg::grid_group grid       = cg::this_grid();
    const int lane            = threadIdx.x % MIO_RNN_FINF_WAVE;
    const int wave_in_block   = threadIdx.x / MIO_RNN_FINF_WAVE;
    const int waves_per_block = blockDim.x / MIO_RNN_FINF_WAVE;
    const int global_wave     = blockIdx.x * waves_per_block + wave_in_block;
    const int total_waves     = gridDim.x * waves_per_block;
    const int gtid            = blockIdx.x * blockDim.x + threadIdx.x;
    const int gnth            = gridDim.x * blockDim.x;
    const long long wh_dir    = wei_shift_dir + (long long)ri * wei_len * uni_stride;

    // h/c carried in the workspace hidden/cell slots. Rows for timestep t are
    // (t*max_batch + b) for b in [0,max_batch). At s==0 the previous hidden
    // state is the caller-provided hx (h_{-1}); for s>0 it is the workspace
    // hidden slot of the previously processed timestep.
    for(int s = 0; s < seqLen; ++s)
    {
        const int t      = reverse ? (seqLen - 1 - s) : s;
        const int t_prev = reverse ? (t + 1) : (t - 1);

        // recurrent: for each (gate row g, batch b): gates[t,b,g] += W_hh[g,:].h_prev[b,:]
        // One wave per (g,b) pair, coalesced lane reduction over k, shuffle-combine.
        // h_prev is hx at s==0 (skipped when hx is absent, i.e. h_{-1}=0) and the
        // prior timestep's workspace hidden otherwise. The guard is uniform across
        // the grid (s and use_hx are the same for every thread) so the grid.sync()
        // is reached collectively.
        const bool has_prev = (s > 0) || use_hx;
        if(has_prev)
        {
            const int ng = 4 * hy_h;
            for(int gb = global_wave; gb < ng * max_batch; gb += total_waves)
            {
                const int g          = gb % ng;
                const int b          = gb / ng;
                const long long base = (long long)(t * max_batch + b) * hy_stride + ri * wei_len;
                const float* hprev =
                    (s == 0) ? (hx + hcx_offset + (long long)b * hy_h)
                             : (workSpace + (long long)(t_prev * max_batch + b) * hy_stride +
                                hid_off + ri * hy_h);
                const float* wh = w + wh_dir + (long long)g * uni_stride;
                float acc       = 0.f;
                for(int k = lane; k < hy_h; k += MIO_RNN_FINF_WAVE)
                    acc += wh[k] * hprev[k];
                for(int o = MIO_RNN_FINF_WAVE / 2; o > 0; o >>= 1)
                    acc += __shfl_down(acc, o);
                if(lane == 0)
                    workSpace[base + g] += acc;
            }
            grid.sync();
        }

        // cell update + activations; one thread per (hidden unit j, batch b).
        // MIOpen gate order is i,f,o,c: +0=i +1=f +2=o (sigmoid), +3=c (tanh).
        for(int jb = gtid; jb < hy_h * max_batch; jb += gnth)
        {
            const int j          = jb % hy_h;
            const int b          = jb / hy_h;
            const long long rb   = (long long)(t * max_batch + b) * hy_stride;
            const long long base = rb + ri * wei_len;
            const long long hid  = rb + hid_off + ri * hy_h;
            const long long cell = rb + (long long)bi * wei_len + ri * hy_h;

            float ig     = finf_sigmoid(workSpace[base + 0 * hy_h + j]);
            float fg     = finf_sigmoid(workSpace[base + 1 * hy_h + j]);
            float og     = finf_sigmoid(workSpace[base + 2 * hy_h + j]);
            float cg_    = tanhf(workSpace[base + 3 * hy_h + j]);
            float c_prev = 0.f;
            if(s == 0)
            {
                if(use_cx)
                    c_prev = cx[hcx_offset + (long long)b * hy_h + j];
            }
            else
            {
                c_prev = workSpace[(long long)(t_prev * max_batch + b) * hy_stride +
                                   (long long)bi * wei_len + ri * hy_h + j];
            }
            float c_new         = fg * c_prev + ig * cg_;
            float h_new         = og * tanhf(c_new);
            workSpace[cell + j] = c_new;
            workSpace[hid + j]  = h_new;

            // Final processed timestep for this direction carries the recurrent
            // state out to hy/cy (offset matches the stock copy-out for uniform
            // batch: hcx_offset + b*hy_h).
            if(s == seqLen - 1)
            {
                if(use_hy)
                    hy[hcx_offset + (long long)b * hy_h + j] = h_new;
                if(use_cy)
                    cy[hcx_offset + (long long)b * hy_h + j] = c_new;
            }
        }
        grid.sync();
    }
}
