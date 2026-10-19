// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <string>
#include <cstdlib>

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "mask.hpp"
#include "bias.hpp"

#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

// Function pointer type for workspace packing (no captures, points to code segment)
using PrepareWorkspaceHostFunc = size_t (*)(void*,                   // host_ws
                                            ck_tile::index_t,        // batch
                                            ck_tile::index_t,        // hdim_q
                                            ck_tile::index_t,        // nhead_q
                                            ck_tile::index_t,        // seqlen_q
                                            ck_tile::index_t,        // seqlen_k
                                            const ck_tile::index_t*, // seqstart_q
                                            const ck_tile::index_t*  // seqstart_k
);

// Function pointer type for the on-device workspace-metadata kernel launch. Launches a
// single-thread kernel that writes nsplits[]/offsets[] straight into the device workspace,
// so no host callback, pinned staging, or metadata D2H/H2D is needed (graph-capturable).
using PrepareWorkspaceDeviceFunc = void (*)(void*,                   // device_ws
                                            ck_tile::index_t,        // batch
                                            ck_tile::index_t,        // hdim_q
                                            ck_tile::index_t,        // nhead_q
                                            ck_tile::index_t,        // seqlen_q
                                            ck_tile::index_t,        // seqlen_k
                                            ck_tile::index_t,        // num_cus
                                            const ck_tile::index_t*, // seqstart_q (device)
                                            const ck_tile::index_t*, // seqstart_k (device)
                                            const ck_tile::stream_config&);

struct FmhaBwdFp32
{
};

struct FmhaBwdFp16
{
};

struct FmhaBwdBf16
{
};

template <typename DataType>
struct FmhaBwdTypeConfig;

template <>
struct FmhaBwdTypeConfig<FmhaBwdFp32>
{
    using QDataType             = float;
    using KDataType             = float;
    using VDataType             = float;
    using GemmDataType          = float;
    using BiasDataType          = float;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = float;
    using OGradDataType         = float;
    using QGradDataType         = float;
    using KGradDataType         = float;
    using VGradDataType         = float;
    using BiasGradDataType      = float;
};

template <>
struct FmhaBwdTypeConfig<FmhaBwdFp16>
{
    using QDataType             = ck_tile::half_t;
    using KDataType             = ck_tile::half_t;
    using VDataType             = ck_tile::half_t;
    using GemmDataType          = ck_tile::half_t;
    using BiasDataType          = ck_tile::half_t;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = ck_tile::half_t;
    using OGradDataType         = ck_tile::half_t;
    using QGradDataType         = ck_tile::half_t;
    using KGradDataType         = ck_tile::half_t;
    using VGradDataType         = ck_tile::half_t;
    using BiasGradDataType      = ck_tile::half_t;
};

template <>
struct FmhaBwdTypeConfig<FmhaBwdBf16>
{
    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using GemmDataType          = ck_tile::bf16_t;
    using BiasDataType          = ck_tile::bf16_t;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = ck_tile::bf16_t;
    using OGradDataType         = ck_tile::bf16_t;
    using QGradDataType         = ck_tile::bf16_t;
    using KGradDataType         = ck_tile::bf16_t;
    using VGradDataType         = ck_tile::bf16_t;
    using BiasGradDataType      = ck_tile::bf16_t;
};

struct FmhaMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};

// runtime args, some will passed to karg, some will used to compute grids/blocks
struct fmha_bwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    const void* o_ptr;
    const void* lse_ptr;
    const void* do_ptr;
    void* d_ptr;
    void* rand_val_ptr;
    void* dq_ptr;
    void* dk_ptr;
    void* dv_ptr;
    void* dbias_ptr;
    void* workspace_ptr;
    const void* sink_ptr; // sink scores [nhead] in log-space (LSEDataType); nullptr disables sink
    void* d_sink_ptr; // sink gradient output [nhead] (LSEDataType); nullptr disables sink gradient

    // Usage notes for sequence length pointer parameters:
    //
    // [Note: Define "Group mode" vs "Batch mode" here if possible, e.g., "Group mode handles
    // MQA/GQA..."]
    //
    // With padding:
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical (including padding) sequence
    //       lengths. [array size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr: Records logical (excluding padding) length for each
    //       sequence. [array size: batch]
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //       sequence lengths. [array size: batch + 1]
    //     - seqlen_q_ptr (per-sequence) and cu_seqlen_q_ptr (cumulative logical) are mutually
    //       exclusive. Use one set, not both.
    //
    //   Batch mode:
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqstart_* and seqlen_* pointers must be nullptr.
    //
    // Without padding:
    //   (Note: Physical length equals logical length)
    //
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical sequence lengths. [array
    //     size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr and cu_seqlen_q_ptr/cu_seqlen_k_ptr must be nullptr.
    //
    //   Batch mode:
    //     - All sequence length pointers (seqstart_*, seqlen_*, cu_seqlen_*) must be nullptr.
    //
    const void* seqstart_q_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqstart_k_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqlen_q_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* seqlen_k_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* cu_seqlen_q_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    const void* cu_seqlen_k_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t max_seqlen_k;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    float scale;
    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_o;
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_do;
    ck_tile::index_t stride_dq;
    ck_tile::index_t stride_dk;
    ck_tile::index_t stride_dv;
    ck_tile::index_t stride_dbias;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t nhead_stride_lsed;
    ck_tile::index_t nhead_stride_dq;
    ck_tile::index_t nhead_stride_dk;
    ck_tile::index_t nhead_stride_dv;
    ck_tile::index_t nhead_stride_dbias;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_do;
    ck_tile::index_t batch_stride_lsed;
    ck_tile::index_t batch_stride_dq;
    ck_tile::index_t batch_stride_dk;
    ck_tile::index_t batch_stride_dv;
    ck_tile::index_t batch_stride_dbias;
    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
    float p_drop;
    float p_undrop;
    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;
#if CK_TILE_FMHA_BWD_DET_FAST
    // Head chunking. Appended at the end so existing positional aggregate
    // initialisers keep working; both defaults mean "not chunked":
    //   nhead_q_offset    global index of this call's first q head
    //   nhead_ratio_qk_in global nhead_q/nhead_k; 0 = derive it from nhead_q/nhead_k
    // The ratio has to be passed explicitly because a chunk's own nhead_q/nhead_k no
    // longer carries it once chunks are allowed to start mid kv-group.
    ck_tile::index_t nhead_q_offset    = 0;
    ck_tile::index_t nhead_ratio_qk_in = 0;
#endif
};

template <typename FmhaBwdDQDKDVKernel>
auto fmha_bwd_dq_dk_dv_create_kargs_and_grids(fmha_bwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdDQDKDVKernel::kIsGroupMode)
        {
            return FmhaBwdDQDKDVKernel::MakeKargsImpl(args.q_ptr,
                                                      args.k_ptr,
                                                      args.v_ptr,
                                                      args.bias_ptr,
                                                      args.lse_ptr,
                                                      args.do_ptr,
                                                      args.d_ptr,
                                                      args.rand_val_ptr,
                                                      args.dq_ptr,
                                                      args.dk_ptr,
                                                      args.dv_ptr,
                                                      args.dbias_ptr,
                                                      args.workspace_ptr,
                                                      args.seqstart_q_ptr,
                                                      args.seqstart_k_ptr,
                                                      args.seqlen_q_ptr,
                                                      args.seqlen_k_ptr,
                                                      args.cu_seqlen_q_ptr,
                                                      args.cu_seqlen_k_ptr,
                                                      args.batch,
                                                      args.hdim_q,
                                                      args.hdim_v,
                                                      args.nhead_q,
#if CK_TILE_FMHA_BWD_DET_FAST
                                                      args.nhead_ratio_qk_in > 0
                                                          ? args.nhead_ratio_qk_in
                                                          : (args.nhead_q / args.nhead_k),
#else
                                                      args.nhead_q / args.nhead_k,
#endif
                                                      args.scale,
                                                      args.stride_q,
                                                      args.stride_k,
                                                      args.stride_v,
                                                      args.stride_bias,
                                                      args.stride_randval,
                                                      args.stride_do,
                                                      args.stride_dq,
                                                      args.stride_dk,
                                                      args.stride_dv,
                                                      args.stride_dbias,
                                                      args.nhead_stride_q,
                                                      args.nhead_stride_k,
                                                      args.nhead_stride_v,
                                                      args.nhead_stride_bias,
                                                      args.nhead_stride_randval,
                                                      args.nhead_stride_do,
                                                      args.nhead_stride_lsed,
                                                      args.nhead_stride_dq,
                                                      args.nhead_stride_dk,
                                                      args.nhead_stride_dv,
                                                      args.nhead_stride_dbias,
                                                      args.window_size_left,
                                                      args.window_size_right,
                                                      args.mask_type,
                                                      args.p_drop,
                                                      args.drop_seed_offset
#if CK_TILE_FMHA_BWD_DET_FAST
                                                      ,
                                                      args.nhead_q_offset
#endif
    );
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdDQDKDVKernel::MakeKargsImpl(args.q_ptr,
                                                      args.k_ptr,
                                                      args.v_ptr,
                                                      args.bias_ptr,
                                                      args.lse_ptr,
                                                      args.do_ptr,
                                                      args.d_ptr,
                                                      args.rand_val_ptr,
                                                      args.dq_ptr,
                                                      args.dk_ptr,
                                                      args.dv_ptr,
                                                      args.dbias_ptr,
                                                      args.workspace_ptr,
                                                      args.seqlen_q,
                                                      args.seqlen_k,
                                                      args.batch,
                                                      args.hdim_q,
                                                      args.hdim_v,
                                                      args.nhead_q,
#if CK_TILE_FMHA_BWD_DET_FAST
                                                      args.nhead_ratio_qk_in > 0
                                                          ? args.nhead_ratio_qk_in
                                                          : (args.nhead_q / args.nhead_k),
#else
                                                      args.nhead_q / args.nhead_k,
#endif
                                                      args.scale,
                                                      args.stride_q,
                                                      args.stride_k,
                                                      args.stride_v,
                                                      args.stride_bias,
                                                      args.stride_randval,
                                                      args.stride_do,
                                                      args.stride_dq,
                                                      args.stride_dk,
                                                      args.stride_dv,
                                                      args.stride_dbias,
                                                      args.nhead_stride_q,
                                                      args.nhead_stride_k,
                                                      args.nhead_stride_v,
                                                      args.nhead_stride_bias,
                                                      args.nhead_stride_randval,
                                                      args.nhead_stride_do,
                                                      args.nhead_stride_lsed,
                                                      args.nhead_stride_dq,
                                                      args.nhead_stride_dk,
                                                      args.nhead_stride_dv,
                                                      args.nhead_stride_dbias,
                                                      args.batch_stride_q,
                                                      args.batch_stride_k,
                                                      args.batch_stride_v,
                                                      args.batch_stride_bias,
                                                      args.batch_stride_randval,
                                                      args.batch_stride_do,
                                                      args.batch_stride_lsed,
                                                      args.batch_stride_dq,
                                                      args.batch_stride_dk,
                                                      args.batch_stride_dv,
                                                      args.batch_stride_dbias,
                                                      args.window_size_left,
                                                      args.window_size_right,
                                                      args.mask_type,
                                                      args.p_drop,
                                                      args.drop_seed_offset
#if CK_TILE_FMHA_BWD_DET_FAST
                                                      ,
                                                      args.nhead_q_offset
#endif
    );
        }
    }();

    const auto grid_seqlen_k =
        FmhaBwdDQDKDVKernel::kIsGroupMode ? args.max_seqlen_k : args.seqlen_k;
    dim3 grids = FmhaBwdDQDKDVKernel::GridSize(args.batch, args.nhead_q, grid_seqlen_k);
    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaBwdOGradDotOKernel>
auto fmha_bwd_dot_do_o_create_kargs_and_grids(fmha_bwd_args args)
{
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdOGradDotOKernel::kIsGroupMode)
        {
            return FmhaBwdOGradDotOKernel::MakeKargs(args.o_ptr,
                                                     args.do_ptr,
                                                     args.d_ptr,
                                                     args.lse_ptr,
                                                     args.sink_ptr,
                                                     args.d_sink_ptr,
                                                     args.p_undrop,
                                                     args.seqstart_q_ptr,
                                                     args.seqlen_q_ptr,
                                                     args.cu_seqlen_q_ptr,
                                                     args.hdim_v,
                                                     args.nhead_q,
                                                     args.stride_do,
                                                     args.stride_o,
                                                     args.nhead_stride_do,
                                                     args.nhead_stride_o,
                                                     args.nhead_stride_lsed);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdOGradDotOKernel::MakeKargs(args.o_ptr,
                                                     args.do_ptr,
                                                     args.d_ptr,
                                                     args.lse_ptr,
                                                     args.sink_ptr,
                                                     args.d_sink_ptr,
                                                     args.p_undrop,
                                                     args.seqlen_q,
                                                     args.hdim_v,
                                                     args.nhead_q,
                                                     args.stride_do,
                                                     args.stride_o,
                                                     args.nhead_stride_do,
                                                     args.nhead_stride_o,
                                                     args.nhead_stride_lsed,
                                                     args.batch_stride_do,
                                                     args.batch_stride_o,
                                                     args.batch_stride_lsed);
        }
    }();

    dim3 grids = FmhaBwdOGradDotOKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q);
    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaBwdConvertQGradKernel>
auto fmha_bwd_convert_dq_create_kargs_and_grids(fmha_bwd_args args)
{
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdConvertQGradKernel::kIsGroupMode)
        {
            return FmhaBwdConvertQGradKernel::MakeKargs(args.workspace_ptr,
                                                        args.dq_ptr,
                                                        args.batch,
                                                        args.nhead_q,
                                                        args.seqstart_q_ptr,
                                                        args.seqstart_k_ptr,
                                                        args.seqlen_q_ptr,
                                                        args.seqlen_k_ptr,
                                                        args.cu_seqlen_q_ptr,
                                                        args.cu_seqlen_k_ptr,
                                                        args.hdim_q,
                                                        args.stride_dq,
                                                        args.nhead_stride_dq);
        }
        else
        { // create batch mode kernel arguments
#if CK_TILE_FMHA_BWD_DET_FAST
            // Right reach of the mask, as k_max(i) = min(seqlen_k - 1, i + off).
            // convert_dq uses it to stop the split reduction early; seqlen_k disables it.
            const ck_tile::index_t mask_k_end_off = [&]() -> ck_tile::index_t {
                if(args.mask_type == 0 || args.window_size_right < 0)
                    return args.seqlen_k; // no right limit
                ck_tile::index_t off = args.window_size_right;
                if(args.mask_type == 2) // bottom-right aligned
                    off += args.seqlen_k - args.seqlen_q;
                return off;
            }();
#endif
            return FmhaBwdConvertQGradKernel::MakeKargs(args.workspace_ptr,
                                                        args.dq_ptr,
                                                        args.batch,
                                                        args.nhead_q,
                                                        args.seqlen_q,
                                                        args.seqlen_k,
                                                        args.hdim_q,
                                                        args.stride_dq,
                                                        args.nhead_stride_dq,
                                                        args.batch_stride_dq
#if CK_TILE_FMHA_BWD_DET_FAST
                                                        ,
                                                        mask_k_end_off
#endif
            );
        }
    }();

    dim3 grids = FmhaBwdConvertQGradKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q);
    return ck_tile::make_tuple(kargs, grids);
}

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          typename FmhaMask_,
          typename FmhaDropout_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          ck_tile::index_t kPadD_,
          ck_tile::index_t kPadDv_,
          bool kIsDeterministic_,
          bool kUseTrLoad_,
          ck_tile::index_t MaxSeqLenQ_,
          ck_tile::index_t kN0>
struct fmha_bwd_dq_dk_dv_traits_
{
};

template <typename Traits_, typename Arch = void>
float fmha_bwd_dq_dk_dv_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
void fmha_bwd_dq_dk_dv_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
std::string fmha_bwd_dq_dk_dv_get_name_();
template <typename Traits_, typename Arch = void>
int fmha_bwd_dq_dk_dv_maxq_();
struct fmha_bwd_traits;
template <typename Traits_, typename Arch = void>
size_t fmha_bwd_dq_dk_dv_dq_ws_host_size_(int batch_size);
// `total_seqlen_q_padded` is total q tokens across all batches (incl. per-batch padding):
//   - batch mode: max_batch * seqlen_q
//   - group mode: seqstart_q[batch] (== varlen q tensor's first dim)
template <typename Traits_, typename Arch = void>
size_t fmha_bwd_dq_dk_dv_dq_ws_device_upper_bound_(ck_tile::index_t max_batch,
                                                   ck_tile::index_t hdim_q,
                                                   ck_tile::index_t nhead_q,
                                                   ck_tile::index_t total_seqlen_q_padded,
                                                   ck_tile::index_t max_seqlen_k);
template <typename Traits_, typename Arch = void>
size_t fmha_bwd_dq_dk_dv_dq_prepare_ws_host_(void* cpu_ws,
                                             ck_tile::index_t batch_size,
                                             ck_tile::index_t hdim_q,
                                             ck_tile::index_t nhead_q,
                                             ck_tile::index_t seqlen_q,
                                             ck_tile::index_t seqlen_k,
                                             const ck_tile::index_t* seqstart_qs,
                                             const ck_tile::index_t* seqstart_ks);
// On-device metadata preparation: launches a kernel that writes nsplits[]/offsets[] into
// the device workspace on stream `s`. Replaces the host-callback + H2D staging path.
template <typename Traits_, typename Arch = void>
void fmha_bwd_dq_dk_dv_dq_prepare_ws_device_(void* device_ws,
                                             ck_tile::index_t batch_size,
                                             ck_tile::index_t hdim_q,
                                             ck_tile::index_t nhead_q,
                                             ck_tile::index_t seqlen_q,
                                             ck_tile::index_t seqlen_k,
                                             ck_tile::index_t num_cus,
                                             const ck_tile::index_t* seqstart_qs,
                                             const ck_tile::index_t* seqstart_ks,
                                             const ck_tile::stream_config& s);
template <typename Traits_, typename Arch = void>
bool fmha_bwd_dq_dk_dv_needs_zero_dq_acc_();

template <ck_tile::index_t HDim_, typename DataType_, bool kIsGroupMode_, bool kPadS_, bool kPadDv_>
struct fmha_bwd_dot_do_o_traits_
{
    static constexpr ck_tile::index_t HDim = HDim_;
    using DataType                         = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode     = kIsGroupMode_;
    static constexpr bool kPadS            = kPadS_;
    static constexpr bool kPadDv           = kPadDv_;
};

template <typename Traits_, typename Arch = void>
float fmha_bwd_dot_do_o_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
void fmha_bwd_dot_do_o_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
std::string fmha_bwd_dot_do_o_get_name_();

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          bool kPadS_,
          bool kPadD_,
          bool kIsDeterministic_>
struct fmha_bwd_convert_dq_traits_
{
};

template <typename Traits_, typename Arch = void>
float fmha_bwd_convert_dq_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
void fmha_bwd_convert_dq_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
std::string fmha_bwd_convert_dq_get_name_();

// Traits that are used to dispatch different kernel implementations for fmha backward
struct fmha_bwd_traits
{
    int seqlen_q;
    int seqlen_k;
    int batch;
    int max_seqlen_q;
    int max_seqlen_k;
    int hdim_q;
    int hdim_v;
    int nhead_q;
    int nhead_k;
    std::string data_type;
    bool is_group_mode;
    mask_enum mask_type;
    bias_enum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_dbias;
    bool has_dropout;
    bool is_store_randval;
    bool is_deterministic;
    // TODO: padding check is inside this api
};

template <typename T0 /*dot_do_o_trait*/,
          typename T1 /*dq_dk_dv_trait*/,
          typename T2 /*convert_dq_trait*/,
          typename Arch>
float fmha_bwd_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if constexpr(!std::is_same_v<T2, void>)
    {
        if(s.log_level_ > 0)
            std::cout << ", " << fmha_bwd_dot_do_o_get_name_<T0, Arch>() << "@"
                      << fmha_bwd_convert_dq_get_name_<T2, Arch>() << "@"
                      << fmha_bwd_dq_dk_dv_get_name_<T1, Arch>() << std::flush;
        return ck_tile::launch_kernel(
            s,
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dot_do_o_oneshot_<T0, Arch>(s_, a); },
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dq_dk_dv_oneshot_<T1, Arch>(s_, a); },
            [=](const ck_tile::stream_config& s_) {
                fmha_bwd_convert_dq_oneshot_<T2, Arch>(s_, a);
            });
    }
    else
    {
        if(s.log_level_ > 0)
            std::cout << ", " << fmha_bwd_dot_do_o_get_name_<T0, Arch>() << "@"
                      << fmha_bwd_dq_dk_dv_get_name_<T1, Arch>() << std::flush;
        return ck_tile::launch_kernel(
            s,
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dot_do_o_oneshot_<T0, Arch>(s_, a); },
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dq_dk_dv_oneshot_<T1, Arch>(s_, a); });
    }
}

template <int Version = 2>
float fmha_bwd(const fmha_bwd_traits&, fmha_bwd_args, const ck_tile::stream_config&);

struct fmha_bwd_launcher
{
    std::function<float(fmha_bwd_args, const ck_tile::stream_config&)> run{
        [](fmha_bwd_args, const ck_tile::stream_config&) {
            std::cerr << "fmha_bwd: no kernel found for given traits, skipping run\n";
            return -1.0f;
        }};
    // Layout: [host_ws_size_ bytes (host-prepared metadata)][dq_acc region]
    size_t workspace_size = 0;

    fmha_bwd_launcher(const fmha_bwd_traits&);
    fmha_bwd_launcher(fmha_bwd_launcher&&)            = delete;
    fmha_bwd_launcher& operator=(fmha_bwd_launcher&&) = delete;

    ~fmha_bwd_launcher() = default;

    ck_tile::index_t selected_max_seqlen_q() const { return selected_max_seqlen_q_; }

    // Stream-async workspace preparation. Zeroes the dq_acc region (if the kernel
    // accumulates) and launches a single-thread device kernel that writes the
    // nsplits[]/offsets[] metadata straight into the device workspace. Every op is
    // stream-ordered and capturable -- there is no host callback, no pinned host staging,
    // and no metadata D2H/H2D. seqstart_q_dev/seqstart_k_dev are read on-device in group
    // mode and must already be resident on the device.
    void prepare_workspace_async( //
        void* device_ws_ptr,
        const int* seqstart_q_dev,
        const int* seqstart_k_dev,
        const ck_tile::stream_config& s)
    {
        hipStream_t stream = s.stream_id_;

        // Fast path: no metadata region; just zero dq_acc if needed.
        if(host_ws_size_ == 0)
        {
            if(needs_zero_dq_acc_ && workspace_size > 0)
                HIP_CHECK_ERROR(hipMemsetAsync(device_ws_ptr, 0, workspace_size, stream));
            return;
        }

        if(traits_.is_group_mode && (!seqstart_q_dev || !seqstart_k_dev))
            throw std::runtime_error("fmha_bwd_launcher::prepare_workspace_async: "
                                     "seqstart_q_dev and seqstart_k_dev are required in "
                                     "group mode");

        // Zero only the dq_acc region that follows the metadata region; the metadata
        // region is fully overwritten by the prepare kernel below.
        if(needs_zero_dq_acc_ && workspace_size > host_ws_size_)
            HIP_CHECK_ERROR(hipMemsetAsync(static_cast<char*>(device_ws_ptr) + host_ws_size_,
                                           0,
                                           workspace_size - host_ws_size_,
                                           stream));

        prepare_ws_dev_func_(device_ws_ptr,
                             batch_,
                             hdim_q_,
                             nhead_q_,
                             seqlen_q_,
                             seqlen_k_,
                             num_cus_,
                             reinterpret_cast<const ck_tile::index_t*>(seqstart_q_dev),
                             reinterpret_cast<const ck_tile::index_t*>(seqstart_k_dev),
                             s);
    }

    // Backward-compatible overload for callers written against the previous API,
    // which passed a pinned-host-buffer allocator for staging workspace metadata.
    // Metadata is now produced by an on-device kernel, so the allocator is unused.
    template <typename HostAllocFn>
    void prepare_workspace_async( //
        void* device_ws_ptr,
        const int* seqstart_q_dev,
        const int* seqstart_k_dev,
        const ck_tile::stream_config& s,
        const HostAllocFn&)
    {
        prepare_workspace_async(device_ws_ptr, seqstart_q_dev, seqstart_k_dev, s);
    }

    private:
    fmha_bwd_traits traits_{};
    ck_tile::index_t selected_max_seqlen_q_ = 0;
    size_t host_ws_size_                    = 0;
    bool needs_zero_dq_acc_                 = false;

#if CK_TILE_FMHA_BWD_DET_FAST
    // Head chunking. The non-persistent deterministic path gives every kv tile its own
    // dq_acc slot, so the workspace is nsplits x larger than the persistent path's. A
    // chunk owns *every* kv tile of its heads, so it computes those heads' dq completely
    // and needs no cross-chunk accumulator -- the chunks are independent, and the result
    // is bit-identical to running them in one go. The persistent path's dq_acc is already
    // small, so none of this is compiled into that build.
    int head_chunk_  = 0; // 0 = all heads in one launch
    int batch_chunk_ = 0; // 0 = all batches in one launch
    int elem_bytes_  = 0; // q/k/v/o/do/dq/dk/dv element size; lse and d are float

    static int fmha_bwd_elem_bytes_(const std::string& dt)
    {
        if(dt == "fp16" || dt == "bf16")
            return 2;
        if(dt == "fp32")
            return 4;
        return 0; // unknown: leave chunking off rather than guess a stride
    }
#endif

    // Function pointer (points to code segment, survives launcher destruction)
    PrepareWorkspaceDeviceFunc prepare_ws_dev_func_ = nullptr;

    // Captured data passed to the prepare kernel at dispatch time.
    ck_tile::index_t batch_    = 0;
    ck_tile::index_t hdim_q_   = 0;
    ck_tile::index_t nhead_q_  = 0;
    ck_tile::index_t seqlen_q_ = 0;
    ck_tile::index_t seqlen_k_ = 0;
    // CU count queried on the host (persistent deterministic nsplits depends on it, and
    // must match the value used when sizing the device workspace upper bound).
    ck_tile::index_t num_cus_ = 0;

    template <typename T0 /*dot_do_o_trait*/,
              typename T1 /*dq_dk_dv_trait*/,
              typename T2 /*convert_dq_trait*/,
              typename Arch>
    void init(const fmha_bwd_traits& t)
    {
        traits_                = t;
        selected_max_seqlen_q_ = fmha_bwd_dq_dk_dv_maxq_<T1, Arch>();
        run                    = [](fmha_bwd_args a, const ck_tile::stream_config& s) {
            return fmha_bwd_<T0, T1, T2, Arch>(s, a);
        };
        host_ws_size_         = fmha_bwd_dq_dk_dv_dq_ws_host_size_<T1, Arch>(t.batch);
        size_t device_ws_size = 0;
        if(host_ws_size_ > 0)
        {
            // In group mode t.seqlen_q is already the padded total (== seqstart_q[batch]);
            // in batch mode it's per-batch and the total is batch * seqlen_q.
            const ck_tile::index_t total_seqlen_q_padded =
                t.is_group_mode ? t.seqlen_q : t.batch * t.seqlen_q;
            device_ws_size = fmha_bwd_dq_dk_dv_dq_ws_device_upper_bound_<T1, Arch>(
                t.batch, t.hdim_q, t.nhead_q, total_seqlen_q_padded, t.max_seqlen_k);

            // Store function pointer (directly assign template-instantiated function)
            prepare_ws_dev_func_ = &fmha_bwd_dq_dk_dv_dq_prepare_ws_device_<T1, Arch>;

            // Store captured data as member variables
            batch_    = t.batch;
            hdim_q_   = t.hdim_q;
            nhead_q_  = t.nhead_q;
            seqlen_q_ = t.seqlen_q;
            seqlen_k_ = t.seqlen_k;
            num_cus_  = ck_tile::get_num_cus();
        }
        workspace_size     = host_ws_size_ + device_ws_size;
        needs_zero_dq_acc_ = fmha_bwd_dq_dk_dv_needs_zero_dq_acc_<T1, Arch>();

#if CK_TILE_FMHA_BWD_DET_FAST
        elem_bytes_ = fmha_bwd_elem_bytes_(t.data_type);
        head_chunk_ = pick_chunks_(t, device_ws_size, batch_chunk_);
        if(head_chunk_ > 0 || batch_chunk_ > 0)
        {
            // device_ws_size is sizeof(float) * nhead_q * nsplits * (batch * seqlen_q) *
            // hdim_q, and both chunk sizes divide their dimension, so this is exact.
            const size_t hc = (head_chunk_ > 0) ? head_chunk_ : t.nhead_q;
            const size_t bc = (batch_chunk_ > 0) ? batch_chunk_ : t.batch;
            workspace_size  = host_ws_size_ + device_ws_size /
                                                 static_cast<size_t>(t.nhead_q) /
                                                 static_cast<size_t>(t.batch) * hc * bc;
        }
#endif
    }

#if CK_TILE_FMHA_BWD_DET_FAST
    // Pick the (batch, head) chunk that minimises workspace without paying for it in
    // time. Everything is counted in "rounds", one round being a full sweep of cus
    // workgroups, so no throughput or FLOP calibration is needed:
    //
    //   cost = nchunks * ceil(grid/cus) + kChunkTail * (nchunks - 1)
    //
    // The first term is exact integer arithmetic and catches the trap of a grid that
    // merely reaches the CU count -- 320 workgroups on 256 CUs still costs two full
    // rounds, i.e. 62% efficiency. The second is the per-chunk cost, dominated by the
    // exposed prologue and drain of a chunk's last round and therefore proportional to
    // a round, not to wall-clock: 0.18-0.63 rounds over twelve configurations across
    // four shapes, median 0.3. In absolute microseconds the same data spreads 75-447,
    // which is why the first attempt at this constant was unusable.
    //
    // Both dimensions are chunked. Heads alone leave a floor of
    // batch * ceil(seqlen_k/kN0) * seqlen_q * hdim * 4 -- nhead cancels out -- so a
    // long-sequence shape can sit far above the budget however finely its heads are
    // split. Batch chunks are even simpler than head chunks: separate batches share
    // nothing at all.
    int pick_chunks_(const fmha_bwd_traits& t, size_t device_ws_size, int& batch_chunk)
    {
        constexpr double kChunkTail = 0.3;
        // Budget baseline, in MiB, for a 128-wide head. One dq_acc split costs
        // batch * nhead * seqlen_q * hdim_q * 4 bytes, so the bytes a given split count
        // needs scale with hdim_q: a flat budget buys only half as many splits at d=256
        // as at d=128 and forces twice the chunking for the same work. Measured on the
        // d=256 shape (-b=2 -h=16 -h_k=8 -s=8192 -d=256 -mask=1): -2.0% at 4 GiB vs
        // -7.6% at 8 GiB vs -11.6% at 16 GiB, against the persistent path.
        // Scale the default up with hdim_q so the budget buys a consistent amount of
        // work. Never scale it down: a narrow head does not make VRAM any less scarce,
        // and d<=128 then keeps exactly the 4 GiB behaviour this was validated with.
        constexpr size_t kBudgetMiBPer128 = 4096;

        batch_chunk = 0;

        if(!t.is_deterministic || t.is_group_mode || device_ws_size == 0 || elem_bytes_ == 0)
            return 0;
        if(t.has_dbias || t.has_dropout || t.is_store_randval ||
           t.bias_type != bias_enum::no_bias)
            return 0;
        if(t.nhead_q <= 0 || t.nhead_k <= 0 || t.nhead_q % t.nhead_k != 0 || t.batch <= 0)
            return 0;

        // CK_DET_HEAD_CHUNK: "auto"/unset = cost model, 0 = never chunk, N = force N heads.
        int forced = -1;
        if(const char* e = std::getenv("CK_DET_HEAD_CHUNK"))
        {
            if(std::string(e) != "auto")
            {
                forced = std::atoi(e);
                if(forced == 0)
                    return 0;
            }
        }
        if(forced > 0)
            return (forced < t.nhead_q && t.nhead_q % forced == 0) ? forced : 0;

        const long long denom = 4LL * t.batch * t.nhead_q * t.seqlen_q * t.hdim_q;
        if(denom <= 0)
            return 0;
        // jobs_per_head == ceil(seqlen_k / kN0), recovered from the unchunked workspace:
        // that is nhead_q * jph * batch * seqlen_q * hdim_q * sizeof(float).
        const int jph = static_cast<int>(device_ws_size / denom);
        const int cus = static_cast<int>(ck_tile::get_num_cus());
        if(jph <= 0 || cus <= 0)
            return 0;

        const bool pairs = (t.mask_type != mask_enum::no_mask) && (jph <= 160);
        const int xdim   = pairs ? ((jph + 1) / 2) : jph;
        if(xdim <= 0)
            return 0;

        const int hdim_mult = (t.hdim_q > 128) ? (t.hdim_q / 128) : 1;
        size_t budget_mib   = kBudgetMiBPer128 * static_cast<size_t>(hdim_mult);
        if(const char* bs = std::getenv("CK_DET_WS_BUDGET_MB"))
            budget_mib = static_cast<size_t>(std::atoll(bs)); // explicit override wins
        const size_t budget_bytes = budget_mib * size_t{1024} * 1024;

        const size_t ws_unit = device_ws_size / static_cast<size_t>(t.batch) /
                               static_cast<size_t>(t.nhead_q); // one batch, one head
        auto ws_of   = [&](int bc, int hc) { return ws_unit * static_cast<size_t>(bc) *
                                                    static_cast<size_t>(hc); };
        auto cost_of = [&](int bc, int hc) {
            const long long nch = static_cast<long long>(t.batch / bc) * (t.nhead_q / hc);
            const long long g   = static_cast<long long>(xdim) * hc * bc;
            return static_cast<double>(nch * ((g + cus - 1) / cus)) +
                   kChunkTail * static_cast<double>(nch - 1);
        };
        int best_b = 0, best_h = 0;
        double best_cost = -1.0;
        int small_b = 0, small_h = 0;
        size_t small_ws = 0;
        for(int bc = 1; bc <= t.batch; ++bc)
        {
            if(t.batch % bc != 0)
                continue;
            for(int hc = 1; hc <= t.nhead_q; ++hc)
            {
                if(t.nhead_q % hc != 0)
                    continue;
                const size_t w = ws_of(bc, hc);
                const double c = cost_of(bc, hc);
                if(small_b == 0 || w < small_ws)
                {
                    small_b = bc; small_h = hc; small_ws = w;
                }
                if(budget_bytes > 0 && w > budget_bytes)
                    continue;
                if(best_cost < 0.0 || c < best_cost)
                {
                    best_b = bc; best_h = hc; best_cost = c;
                }
            }
        }

        if(best_b == 0)
        {
            // Nothing fits. A budget is a VRAM limit, so honour it as closely as we can
            // rather than falling back to the largest workspace of all -- overshooting by
            // nhead*batch can OOM the caller outright.
            best_b = small_b;
            best_h = small_h;
        }
        batch_chunk = (best_b == t.batch) ? 0 : best_b;
        return (best_h == t.nhead_q) ? 0 : best_h;
    }
#endif // CK_TILE_FMHA_BWD_DET_FAST

    public:
#if !CK_TILE_FMHA_BWD_DET_FAST
    template <typename... Args>
    float operator()(Args&&... args) const
    {
        return run(std::forward<Args>(args)...);
    }
#else
    float operator()(fmha_bwd_args a, const ck_tile::stream_config& s) const
    {
        if(head_chunk_ <= 0 && batch_chunk_ <= 0)
            return run(a, s);

        const int hc    = (head_chunk_ > 0) ? head_chunk_ : traits_.nhead_q;
        const int bc    = (batch_chunk_ > 0) ? batch_chunk_ : traits_.batch;
        const int ratio = traits_.nhead_q / traits_.nhead_k;
        const int eb    = elem_bytes_;
        float total     = 0.f;

        auto adv  = [](const void* p, long long n, ck_tile::index_t st, int by) {
            return static_cast<const void*>(static_cast<const char*>(p) + n * st * by);
        };
        auto advw = [](void* p, long long n, ck_tile::index_t st, int by) {
            return static_cast<void*>(static_cast<char*>(p) + n * st * by);
        };

        for(int b0 = 0; b0 < traits_.batch; b0 += bc)
        {
            for(int h0 = 0; h0 < traits_.nhead_q; h0 += hc)
            {
                const int hk0 = h0 / ratio; // first kv head this chunk touches
                fmha_bwd_args c = a;
                c.batch   = bc;
                c.nhead_q = hc;
                // nhead_k only feeds the derived ratio and an assert; the kernel takes
                // the real ratio from nhead_ratio_qk_in, so keep the global value.
                c.nhead_k           = traits_.nhead_k;
                c.nhead_ratio_qk_in = ratio;
                c.nhead_q_offset    = h0;

                c.q_ptr   = adv(a.q_ptr, h0, a.nhead_stride_q, eb);
                c.k_ptr   = adv(a.k_ptr, hk0, a.nhead_stride_k, eb);
                c.v_ptr   = adv(a.v_ptr, hk0, a.nhead_stride_v, eb);
                c.o_ptr   = adv(a.o_ptr, h0, a.nhead_stride_o, eb);
                c.do_ptr  = adv(a.do_ptr, h0, a.nhead_stride_do, eb);
                c.lse_ptr = adv(a.lse_ptr, h0, a.nhead_stride_lsed, 4); // LSE is float
                c.d_ptr   = advw(a.d_ptr, h0, a.nhead_stride_lsed, 4);  // D is float
                c.dq_ptr  = advw(a.dq_ptr, h0, a.nhead_stride_dq, eb);
                // dk/dv are the expanded per-q-head outputs: the kernel addresses them
                // with i_nhead, not i_nhead_k, so they advance by h0. With MHA the two
                // coincide; only GQA tells them apart.
                c.dk_ptr = advw(a.dk_ptr, h0, a.nhead_stride_dk, eb);
                c.dv_ptr = advw(a.dv_ptr, h0, a.nhead_stride_dv, eb);

                if(b0 != 0)
                {
                    c.q_ptr   = adv(c.q_ptr, b0, a.batch_stride_q, eb);
                    c.k_ptr   = adv(c.k_ptr, b0, a.batch_stride_k, eb);
                    c.v_ptr   = adv(c.v_ptr, b0, a.batch_stride_v, eb);
                    c.o_ptr   = adv(c.o_ptr, b0, a.batch_stride_o, eb);
                    c.do_ptr  = adv(c.do_ptr, b0, a.batch_stride_do, eb);
                    c.lse_ptr = adv(c.lse_ptr, b0, a.batch_stride_lsed, 4);
                    c.d_ptr   = advw(const_cast<void*>(static_cast<const void*>(c.d_ptr)),
                                     b0, a.batch_stride_lsed, 4);
                    c.dq_ptr  = advw(c.dq_ptr, b0, a.batch_stride_dq, eb);
                    c.dk_ptr  = advw(c.dk_ptr, b0, a.batch_stride_dk, eb);
                    c.dv_ptr  = advw(c.dv_ptr, b0, a.batch_stride_dv, eb);
                }

                // dq_acc is reused by every chunk, so re-prepare before each one.
                const_cast<fmha_bwd_launcher*>(this)->prepare_workspace_async(
                    a.workspace_ptr, nullptr, nullptr, s);
                const float t = run(c, s);
                if(t < 0)
                    return t;
                total += t;
            }
        }
        return total;
    }
#endif // CK_TILE_FMHA_BWD_DET_FAST
};
