// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of the host-side facts of library/kernels/common/_attention_bwd_ring.py
 * (DmaBatch, RingWaits, dma_wait_counts). See
 * rocke/helper_rocke.instances.common._attention_bwd_ring.h for the mapping.
 *
 * Every Python raise is one ckc::Error throw with the same message; the C
 * entry points catch it. Python's // and % floor; fdiv / fmod do the same.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_ring.h"

#include <string>

#include "rocke/error.hpp"

namespace rocke_attn_bwd
{
namespace
{

long long fdiv(long long a, long long b)
{
    if(b == 0)
        throw ckc::ValueError("integer division or modulo by zero");
    long long q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

long long fmod_(long long a, long long b)
{
    return a - fdiv(a, b) * b;
}

void dma_batch_check(const rocke_attn_bwd_dma_batch_t& d)
{
    long long nbytes = (long long)d.block_m * d.head_size * d.elem_bytes;
    if(d.wave_size != 64)
        throw ckc::ValueError("the LDS DMA ring is wave64 only");
    if(fmod_(nbytes, ROCKE_ATTN_BWD_DMA_PIECE_BYTES))
        throw ckc::ValueError("a " + std::to_string(d.block_m) + " x " + std::to_string(d.head_size)
                              + " tile is not whole 1 KiB DMA pieces");
    if(d.block_m > d.wave_size || fmod_(d.block_m, 16))
        throw ckc::ValueError("the stats DMA needs block_m <= 64, a multiple of 16");
    if(d.waves < 1)
        throw ckc::ValueError("waves must be positive");
}

int of_wave_count(const rocke_attn_bwd_dma_batch_t& d, int wave, int* out, int cap)
{
    int n = 0;
    int total = rocke_attn_bwd_dma_batch_n_instructions(&d);
    for(int k = 0; k < total; ++k)
        if(fmod_(k, d.waves) == wave)
        {
            if(out != NULL && n < cap)
                out[n] = k;
            ++n;
        }
    return n;
}

} // namespace
} // namespace rocke_attn_bwd

using namespace rocke_attn_bwd;

extern "C" {

rocke_status_t rocke_attn_bwd_dma_batch_init(rocke_attn_bwd_dma_batch_t* out,
                                             int block_m,
                                             int head_size,
                                             int waves,
                                             int wave_size,
                                             int elem_bytes,
                                             rocke_attn_bwd_err_t* err)
{
    store_err(err, ROCKE_OK, "");
    try
    {
        rocke_attn_bwd_dma_batch_t d = {block_m, head_size, waves, wave_size, elem_bytes};
        dma_batch_check(d);
        if(out != NULL)
            *out = d;
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        return store_err(err, e.code(), e.what());
    }
}

int rocke_attn_bwd_dma_batch_pieces(const rocke_attn_bwd_dma_batch_t* d)
{
    return (int)fdiv((long long)d->block_m * d->head_size * d->elem_bytes,
                     ROCKE_ATTN_BWD_DMA_PIECE_BYTES);
}

int rocke_attn_bwd_dma_batch_n_instructions(const rocke_attn_bwd_dma_batch_t* d)
{
    return 2 * rocke_attn_bwd_dma_batch_pieces(d) + 2;
}

bool rocke_attn_bwd_dma_batch_instruction(const rocke_attn_bwd_dma_batch_t* d,
                                          int k,
                                          const char** tensor,
                                          int* index)
{
    int pieces = rocke_attn_bwd_dma_batch_pieces(d);
    const char* t;
    int j;
    if(k < 0 || k >= 2 * pieces + 2)
        return false;
    if(k < pieces)
        t = "q", j = k;
    else if(k < 2 * pieces)
        t = "do", j = k - pieces;
    else
        t = k == 2 * pieces ? "lse2" : "dsum", j = 0;
    if(tensor != NULL)
        *tensor = t;
    if(index != NULL)
        *index = j;
    return true;
}

int rocke_attn_bwd_dma_batch_of_wave(const rocke_attn_bwd_dma_batch_t* d,
                                     int wave,
                                     int* out,
                                     int cap)
{
    return of_wave_count(*d, wave, out, cap);
}

int rocke_attn_bwd_dma_batch_min_per_wave(const rocke_attn_bwd_dma_batch_t* d)
{
    int best = 0;
    for(int w = 0; w < d->waves; ++w)
    {
        int n = of_wave_count(*d, w, NULL, 0);
        if(w == 0 || n < best)
            best = n;
    }
    return best;
}

rocke_status_t rocke_attn_bwd_dma_wait_counts(
    int ring_depth, int n_qd, int n_a, rocke_attn_bwd_ring_waits_t* out, rocke_attn_bwd_err_t* err)
{
    store_err(err, ROCKE_OK, "");
    if(!(ring_depth == 1 || ring_depth == 2 || ring_depth == 3) || n_qd < 0 || n_a < 0)
    {
        std::string msg = "bad ring wait inputs (" + std::to_string(ring_depth) + ", "
                          + std::to_string(n_qd) + ", " + std::to_string(n_a) + ")";
        return store_err(err, ROCKE_ERR_VALUE, msg.c_str());
    }
    const int none = ROCKE_ATTN_BWD_NONE;
    const int vmax = ROCKE_ATTN_BWD_RING_VMCNT_MAX;
    rocke_attn_bwd_ring_waits_t w;
    if(ring_depth == 1)
        w = {none, 0, none};
    else if(ring_depth == 2)
        w = {0, none, n_a < vmax ? n_a : vmax};
    else
    {
        long long s = (long long)n_qd + n_a;
        w = {n_qd < vmax ? n_qd : vmax, none, (int)(s < vmax ? s : vmax)};
    }
    if(out != NULL)
        *out = w;
    return ROCKE_OK;
}

} // extern "C"
