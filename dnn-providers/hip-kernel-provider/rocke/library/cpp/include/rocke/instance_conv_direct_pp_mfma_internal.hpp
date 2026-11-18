// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_direct_pp_mfma_internal.hpp -- C++ engine port of
 * library/kernels/common/conv_direct_pp_mfma.py: the gfx950 LDS-DMA ping-pong
 * MFMA direct pipeline that the forward (conv_direct_fwd_pp_mfma.cpp) and the
 * dgrad (conv_direct_dgrad_pp_mfma.cpp) share. A family supplies its weight
 * half as a PpWeights; the pipeline calls each hook once, at the point the
 * Python builder does, so every IR op is emitted in the Python order (see the
 * contract in rocke/instance_conv_direct_wgrad_internal.hpp).
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_PP_MFMA_INTERNAL_HPP
#define ROCKE_INSTANCE_CONV_DIRECT_PP_MFMA_INTERNAL_HPP

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <map>
#include <string>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/instance_conv_direct_wgrad_internal.hpp" /* Em, V */

namespace rocke_pp
{

using rocke_dwg::Em;
using rocke_dwg::V;

constexpr int kWave = 64;
constexpr int kCK = 64; /* channels per chunk */
constexpr int kGran = 8; /* halves per 16-byte granule */
constexpr int kTH = 16; /* output rows per block */
constexpr int kTW = 16; /* output columns per block */
constexpr int kWaves = 8;
constexpr int kRankWaves = 4;
constexpr int kAtom = 16;
constexpr int kPartsC = kCK / 32; /* 32-channel MFMA steps per chunk and tap */
constexpr int64_t kOob = ((int64_t)1 << 31) - 1;

/* The fields both families' specs share (PpMfmaSpec). */
struct PpSpec
{
    const rocke_direct_conv_problem_t* problem;
    const char* name;
    int block_k;
    int wave_size;
};

/* What pp_mfma_reason takes besides the spec: the family's name in the
 * reasons, the block_k values its weight staging takes (block_k_reason
 * refuses the rest) and whether it needs the transposing LDS read. */
struct PpFamily
{
    const char* family;
    const int* block_ks;
    int n_block_ks;
    const char* block_k_reason;
    bool needs_ds_read_tr;
};

long pp_lds_bytes(const PpSpec& s);
int pp_threads_per_block(const PpSpec& s);
void pp_grid(const PpSpec& s, int grid[3]);
rocke_status_t pp_kernel_name(const PpSpec& s, char* out, size_t out_cap);
/* pp_mfma_reason: true (reason "ok") when the spec builds on arch. */
bool pp_is_valid_spec(
    const PpSpec& s, const PpFamily& f, const char* arch, char* reason, size_t reason_cap);

/* PpCtx: what the pipeline has emitted that a PpWeights builds on. */
struct PpCtx
{
    rocke_ir_builder_t* b;
    Em& e;
    const PpSpec& spec;
    const rocke_type_t* io_type;
    V lane, wave_k, wave_g, k0, C, K, c0, c2, oob, w_rsrc;
    V w_bufs[2], w_addr[2];
    std::function<V(V, V)> lds_at;
    /* Lane values a PpWeights computed and the pipeline reuses: l_row,
     * l_gran, lane_c (dma_geometry), lane_hi, l16 (a_geometry). */
    std::map<std::string, V> shared;
};

/* The family half of a ping-pong kernel (PpWeights in the Python builder). */
struct PpWeights
{
    virtual ~PpWeights() = default;
    /* The rank-0 weight DMA geometry; may set shared l_row, l_gran, lane_c. */
    virtual void dma_geometry(PpCtx& c) = 0;
    /* DMA the weights of tap for the chunk at c_base into buffer buf. */
    virtual void issue(PpCtx& c, int buf, int tap, V c_base) = 0;
    /* The A read geometry; must set shared lane_hi, may set l16. */
    virtual void a_geometry(PpCtx& c) = 0;
    /* A fragment kf of 32-channel half part from buffer buf. */
    virtual V read_a(PpCtx& c, int buf, int kf, int part) = 0;
};

/* emit_pp_mfma(spec, weights): the kernel of an already validated spec. */
rocke_kernel_def_t* emit_pp_mfma(rocke_ir_builder_t* b, const PpSpec& s, PpWeights& w);

} // namespace rocke_pp

#endif /* ROCKE_INSTANCE_CONV_DIRECT_PP_MFMA_INTERNAL_HPP */
