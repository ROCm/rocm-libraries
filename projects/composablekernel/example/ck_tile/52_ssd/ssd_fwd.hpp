// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Mamba-2 SSD forward — ck_tile host-side API.
// Orchestrates the multi-kernel SSD forward pass using:
//   - ck_tile BatchedGemmKernel for the four batched GEMMs
//   - Custom HIP kernels for element-wise / scan operations
//
// GEMM layout strategy:
//   ck_tile BatchedGemmKernel computes C(Row) = A(Row) @ B(Col)
//   where B(Col) uses BLAS column-major convention:
//     B(k, j) = B_ptr[j * stride_b + k]   (stride_b >= K)
//   i.e. passing a row-major matrix M as B effectively gives A @ M^T.
//
//   Therefore, to compute A @ X, pass X^T as the B data (so the implicit
//   transpose yields X). Equivalently, pack the source WITHOUT transposing
//   when the GEMM formula already contains a transpose (e.g., A @ X^T),
//   so the column-major transpose produces X^T directly.
//
//   All sub-matrices with non-standard strides (B_mat, C_mat with stride C*L)
//   are first packed into contiguous temporary buffers.

#pragma once

#include <algorithm>
#include <cstdio>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

#include "ssd_problem.hpp"
#include "ssd_kernels.hpp"

namespace ck_tile {

// ---------------------------------------------------------------------------
// GemmConfig for rectangular GEMMs (M=128,N=64 or M=64,N=128 etc.)
// ---------------------------------------------------------------------------
struct SsdGemmConfig
{
    static constexpr index_t M_Tile = 128;
    static constexpr index_t N_Tile = 64;
    static constexpr index_t K_Tile = 32;

    static constexpr index_t M_Warp = 2;
    static constexpr index_t N_Warp = 2;
    static constexpr index_t K_Warp = 1;

    static constexpr index_t M_Warp_Tile = 32;
    static constexpr index_t N_Warp_Tile = 32;
    static constexpr index_t K_Warp_Tile = 8;

    static constexpr bool DoubleSmemBuffer = false;
    static constexpr GemmPipeline Pipeline = GemmPipeline::MEMORY;
    static constexpr auto Scheduler        = GemmPipelineScheduler::Interwave;
};

// Square 128x128 variant for IntraBMM1
struct SsdGemmConfigSquare
{
    static constexpr index_t M_Tile = 128;
    static constexpr index_t N_Tile = 128;
    static constexpr index_t K_Tile = 32;

    static constexpr index_t M_Warp = 2;
    static constexpr index_t N_Warp = 2;
    static constexpr index_t K_Warp = 1;

    static constexpr index_t M_Warp_Tile = 32;
    static constexpr index_t N_Warp_Tile = 32;
    static constexpr index_t K_Warp_Tile = 8;

    static constexpr bool DoubleSmemBuffer = false;
    static constexpr GemmPipeline Pipeline = GemmPipeline::MEMORY;
    static constexpr auto Scheduler        = GemmPipelineScheduler::Interwave;
};

// ---------------------------------------------------------------------------
// Pipeline type traits
// ---------------------------------------------------------------------------
template <GemmPipeline PipelineId>
struct SsdPipelineTraits;

template <>
struct SsdPipelineTraits<GemmPipeline::MEMORY>
{
    template <typename P>
    using type = GemmPipelineAgBgCrMem<P>;
};

template <>
struct SsdPipelineTraits<GemmPipeline::COMPUTE_V3>
{
    template <typename P>
    using type = GemmPipelineAgBgCrCompV3<P>;
};

// ---------------------------------------------------------------------------
// launch_batched_gemm: C(Row) = A(Row) @ B(Col)
//
// BLAS column-major convention for B[K, N]:
//   B(k, j) = B_ptr[j * stride_b + k]    (stride_b >= K)
//
// This computes: C[i][j] = sum_k A[i * stride_a + k] * B[j * stride_b + k]
//
// Passing a row-major matrix M[K,N] as ColumnMajor B effectively computes
// A @ M^T (the column-major read transposes the row-major data).
// To compute A @ M, store M^T in B (pre-transpose to cancel the implicit one).
// ---------------------------------------------------------------------------
template <typename GemmCfg>
inline float launch_batched_gemm(const void* a_ptr,
                                 index_t stride_a,
                                 index_t batch_stride_a,
                                 const void* b_ptr,
                                 index_t stride_b,
                                 index_t batch_stride_b,
                                 void* c_ptr,
                                 index_t stride_c,
                                 index_t batch_stride_c,
                                 index_t M,
                                 index_t N,
                                 index_t K,
                                 index_t batch_count,
                                 hipStream_t stream)
{
    using ADataType   = float;
    using BDataType   = float;
    using AccDataType = float;
    using CDataType   = float;
    using ALayout     = tensor_layout::gemm::RowMajor;
    using BLayout     = tensor_layout::gemm::ColumnMajor;
    using CLayout     = tensor_layout::gemm::RowMajor;
    using DsDataType  = tuple<>;
    using DsLayout    = tuple<>;
    using PassThrough = element_wise::PassThrough;

    constexpr bool kPadM          = true;
    constexpr bool kPadN          = true;
    constexpr bool kPadK          = true;
    constexpr bool TransposeC     = false;
    constexpr bool DoubleSmemBuf  = GemmCfg::DoubleSmemBuffer;
    constexpr index_t kBlockPerCu = 1;
    constexpr index_t TilePartGrp = 8;
    constexpr index_t TilePartM01 = 4;

    using GemmShape =
        TileGemmShape<sequence<GemmCfg::M_Tile, GemmCfg::N_Tile, GemmCfg::K_Tile>,
                      sequence<GemmCfg::M_Warp, GemmCfg::N_Warp, GemmCfg::K_Warp>,
                      sequence<GemmCfg::M_Warp_Tile, GemmCfg::N_Warp_Tile, GemmCfg::K_Warp_Tile>>;

    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<GemmShape, TilePartGrp, TilePartM01>;

    using Traits = TileGemmUniversalTraits<kPadM,
                                           kPadN,
                                           kPadK,
                                           DoubleSmemBuf,
                                           ALayout,
                                           BLayout,
                                           CLayout,
                                           TransposeC>;

    using PipelineProblem = UniversalGemmPipelineProblem<ADataType,
                                                         BDataType,
                                                         AccDataType,
                                                         GemmShape,
                                                         Traits,
                                                         GemmCfg::Scheduler>;

    using Pipeline = typename SsdPipelineTraits<GemmCfg::Pipeline>::template type<PipelineProblem>;

    using Epilogue = CShuffleEpilogue<CShuffleEpilogueProblem<ADataType,
                                                              BDataType,
                                                              DsDataType,
                                                              AccDataType,
                                                              CDataType,
                                                              DsLayout,
                                                              CLayout,
                                                              PassThrough,
                                                              TilePartitioner::MPerBlock,
                                                              TilePartitioner::NPerBlock,
                                                              GemmCfg::M_Warp,
                                                              GemmCfg::N_Warp,
                                                              GemmCfg::M_Warp_Tile,
                                                              GemmCfg::N_Warp_Tile,
                                                              GemmCfg::K_Warp_Tile,
                                                              PipelineProblem::TransposeC>>;

    using Kernel = BatchedGemmKernel<TilePartitioner, Pipeline, Epilogue>;

    // IsSupportedArgument requires batch_stride >= matrix_size even for batch_count=1.
    const index_t safe_batch_stride_a = (batch_stride_a > 0) ? batch_stride_a : M * K;
    const index_t safe_batch_stride_b = (batch_stride_b > 0) ? batch_stride_b : K * N;
    const index_t safe_batch_stride_c = (batch_stride_c > 0) ? batch_stride_c : M * N;

    BatchedGemmHostArgs args{a_ptr,
                             b_ptr,
                             const_cast<void*>(static_cast<const void*>(c_ptr)),
                             /*k_batch=*/1,
                             M,
                             N,
                             K,
                             stride_a,
                             stride_b,
                             stride_c,
                             safe_batch_stride_a,
                             safe_batch_stride_b,
                             safe_batch_stride_c,
                             batch_count};

    auto kargs = Kernel::MakeKernelArgs(args);
    if(!Kernel::IsSupportedArgument(kargs))
    {
        printf("[SSD] GEMM unsupported args: M=%d N=%d K=%d stride_a=%d stride_b=%d stride_c=%d "
               "bs_a=%d bs_b=%d bs_c=%d batch=%d\n",
               static_cast<int>(M),
               static_cast<int>(N),
               static_cast<int>(K),
               static_cast<int>(stride_a),
               static_cast<int>(stride_b),
               static_cast<int>(stride_c),
               static_cast<int>(safe_batch_stride_a),
               static_cast<int>(safe_batch_stride_b),
               static_cast<int>(safe_batch_stride_c),
               static_cast<int>(batch_count));
        return -1.0f;
    }

    const dim3 grids  = Kernel::GridSize(M, N, 1, batch_count);
    const dim3 blocks = Kernel::BlockSize();

    return launch_kernel(stream_config{stream, false},
                         make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}

// ---------------------------------------------------------------------------
// Helper kernel: pack a strided [rows, cols] sub-matrix (stride between rows
// is src_stride) into a contiguous [rows, cols] buffer (stride = cols).
// Also supports transpose: if do_transpose, output is [cols, rows].
// Grid: (1,1,1)  Block: (256,1,1)  — simple, for small matrices.
// ---------------------------------------------------------------------------
__global__ void pack_matrix_kernel(const float* __restrict__ src,
                                   float* __restrict__ dst,
                                   int rows,
                                   int cols,
                                   int src_stride,
                                   bool do_transpose)
{
    int total = rows * cols;
    for(int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < total; idx += blockDim.x * gridDim.x)
    {
        int r     = idx / cols;
        int c     = idx % cols;
        float val = src[r * src_stride + c];
        if(do_transpose)
            dst[c * rows + r] = val; // [cols, rows] row-major
        else
            dst[r * cols + c] = val; // [rows, cols] row-major
    }
}

// ---------------------------------------------------------------------------
// SSD Forward: full orchestration (batched version)
//
// Key optimizations over the naive serial implementation:
//   1) Single workspace allocation — avoids 12x hipMalloc/hipFree per call.
//   2) Batched pack kernels — one launch packs all (beh, ci) slices at once.
//   3) Batched GEMM — batch_count = BEH*C instead of serial loops with 1.
// ---------------------------------------------------------------------------
inline float ssd_fwd(const SsdHostArgs& args, hipStream_t stream = nullptr)
{
    const int B   = args.batch;
    const int G   = args.groups;
    const int EH  = args.exp_heads;
    const int C   = args.chunks;
    const int L   = args.chunk_len;
    const int D   = args.head_dim;
    const int N   = args.state_dim;
    const int BEH = B * EH;
    const int grp = args.grp_ratio();
    const int BC  = BEH * C; // total batch count for GEMM

    // --- Single workspace allocation ---
    const size_t sz_cumsum = static_cast<size_t>(BEH) * C * L;
    const size_t sz_ibmm1  = static_cast<size_t>(BEH) * C * L * L;
    const size_t sz_pi2    = static_cast<size_t>(BEH) * C * L * L;
    const size_t sz_abmm2  = static_cast<size_t>(BEH) * C * L * D;
    const size_t sz_pi1    = static_cast<size_t>(BEH) * C * N * L;
    const size_t sz_rbmm1  = static_cast<size_t>(BEH) * C * N * D;
    const size_t sz_state  = static_cast<size_t>(BEH) * C * N * D;
    const size_t sz_rbmm2  = static_cast<size_t>(BEH) * C * L * D;
    const size_t sz_ce     = static_cast<size_t>(BEH) * C * L;
    const size_t sz_lv     = static_cast<size_t>(BEH) * C;
    const size_t sz_pack_a =
        static_cast<size_t>(BC) * std::max(static_cast<size_t>(N * L), static_cast<size_t>(D * L));
    const size_t sz_pack_b =
        static_cast<size_t>(BC) * std::max(static_cast<size_t>(N * L), static_cast<size_t>(D * N));

    const size_t total_floats = sz_cumsum + sz_ibmm1 + sz_pi2 + sz_abmm2 + sz_pi1 + sz_rbmm1 +
                                sz_state + sz_rbmm2 + sz_ce + sz_lv + sz_pack_a + sz_pack_b;
    DeviceMem d_workspace(total_floats * sizeof(float));
    float* ws = static_cast<float*>(d_workspace.GetDeviceBuffer());

    // Carve up workspace
    float* cum = ws;
    ws += sz_cumsum;
    float* ibmm1 = ws;
    ws += sz_ibmm1;
    float* pi2 = ws;
    ws += sz_pi2;
    float* abmm2 = ws;
    ws += sz_abmm2;
    float* pi1 = ws;
    ws += sz_pi1;
    float* rbmm1 = ws;
    ws += sz_rbmm1;
    float* st = ws;
    ws += sz_state;
    float* rbmm2 = ws;
    ws += sz_rbmm2;
    float* ce = ws;
    ws += sz_ce;
    float* lv = ws;
    ws += sz_lv;
    float* pa = ws;
    ws += sz_pack_a;
    float* pb = ws;
    ws += sz_pack_b;

    // Zero entire workspace (async, on the same stream)
    (void)hipMemsetAsync(d_workspace.GetDeviceBuffer(), 0, total_floats * sizeof(float), stream);

    const float* p_x   = static_cast<const float*>(args.p_x);
    const float* p_da  = static_cast<const float*>(args.p_delta_a);
    const float* p_dlt = static_cast<const float*>(args.p_delta);
    const float* p_bm  = static_cast<const float*>(args.p_b_mat);
    const float* p_cm  = static_cast<const float*>(args.p_c_mat);
    const float* p_dp  = static_cast<const float*>(args.p_d_param);
    const float* p_z   = static_cast<const float*>(args.p_z);
    float* p_y         = static_cast<float*>(args.p_y);
    float* p_fs        = static_cast<float*>(args.p_fstate);

    // ====== Step 1: Cumsum ======
    ssd_cumsum_kernel<<<BEH, C, 0, stream>>>(p_da, cum, C, L);

    // ====== Step 2: IntraBMM1 — batched C^T @ B -> [L, L] ======
    // Pack all (beh, ci) slices of C_mat and B_mat, then one batched GEMM.
    {
        const int nl_blocks = (N * L + 255) / 256;

        // Pack C_mat[N,L] stride=C*L -> transpose -> pa[BC, L, N] (ld=N)
        pack_bcmat_batched_kernel<<<dim3(nl_blocks, BC), 256, 0, stream>>>(
            p_cm, pa, EH, G, grp, N, C, L, /*transpose=*/true);

        // Pack B_mat[N,L] stride=C*L -> transpose -> pb[BC, L, N] (ld=N)
        // ColumnMajor B implicitly transposes, so pre-transpose to cancel.
        pack_bcmat_batched_kernel<<<dim3(nl_blocks, BC), 256, 0, stream>>>(
            p_bm, pb, EH, G, grp, N, C, L, /*transpose=*/true);

        // Batched GEMM: Out[L,L] = pa(Row)[L,N] @ pb(Col)[L,N]
        launch_batched_gemm<SsdGemmConfigSquare>(
            pa,
            N,
            static_cast<index_t>(L * N), // A: [L,N], batch_stride=L*N
            pb,
            N,
            static_cast<index_t>(L * N), // B: [L,N] col-major, batch_stride=L*N
            ibmm1,
            L,
            static_cast<index_t>(L * L), // C: [L,L], batch_stride=L*L
            L,
            L,
            N,
            BC,
            stream);
    }

    // ====== Step 3: SegSum + Pre_IntraBMM2 ======
    ssd_segsum_pre_intra2_kernel<<<dim3(BEH, C), 256, 0, stream>>>(cum, p_dlt, ibmm1, pi2, C, L);

    // ====== Step 4: IntraBMM2 — batched PreIntra2 @ X^T -> [L, D] ======
    // Pack all X slices, then one batched GEMM.
    {
        const int dl_blocks = (D * L + 255) / 256;

        // Pack X_chunk[D,L] stride=C*L -> pa[BC, D, L] (ld=L, no transpose)
        // ColumnMajor B implicitly transposes, giving us X^T for the GEMM.
        pack_x_batched_kernel<<<dim3(dl_blocks, BC), 256, 0, stream>>>(p_x, pa, D, C, L);

        // Batched GEMM: Out[L,D] = PI2(Row)[L,L] @ pa(Col)[D,L]
        launch_batched_gemm<SsdGemmConfig>(
            pi2,
            L,
            static_cast<index_t>(L * L), // A: [L,L], batch_stride=L*L
            pa,
            L,
            static_cast<index_t>(D * L), // B: [D,L] col-major, batch_stride=D*L
            abmm2,
            D,
            static_cast<index_t>(L * D), // C: [L,D], batch_stride=L*D
            L,
            D,
            L,
            BC,
            stream);
    }

    // ====== Step 5: Pre_InterBMM1 + CumsumExp + LastVals ======
    ssd_pre_inter1_kernel<<<dim3(BEH, C), 256, 0, stream>>>(
        cum, p_dlt, p_bm, pi1, ce, lv, C, L, N, G, grp);

    // ====== Step 6: InterBMM1 — batched PreInter1 @ X^T -> [N, D] ======
    {
        const int dl_blocks = (D * L + 255) / 256;

        // Pack X again (same layout as Step 4)
        pack_x_batched_kernel<<<dim3(dl_blocks, BC), 256, 0, stream>>>(p_x, pa, D, C, L);

        // Batched GEMM: Out[N,D] = PI1(Row)[N,L] @ pa(Col)[D,L]
        launch_batched_gemm<SsdGemmConfig>(
            pi1,
            L,
            static_cast<index_t>(N * L), // A: [N,L], batch_stride=N*L
            pa,
            L,
            static_cast<index_t>(D * L), // B: [D,L] col-major, batch_stride=D*L
            rbmm1,
            D,
            static_cast<index_t>(N * D), // C: [N,D], batch_stride=N*D
            N,
            D,
            L,
            BC,
            stream);
    }

    // ====== Step 7: State propagation ======
    ssd_state_propagation_kernel<<<BEH, 256, 0, stream>>>(rbmm1, lv, st, C, N, D);

    // ====== Step 8: InterBMM2 — batched C^T @ State -> [L, D] ======
    {
        const int nl_blocks = (N * L + 255) / 256;
        const int nd_blocks = (N * D + 255) / 256;

        // Pack C_mat transpose -> pa[BC, L, N] (ld=N), same as Step 2
        pack_bcmat_batched_kernel<<<dim3(nl_blocks, BC), 256, 0, stream>>>(
            p_cm, pa, EH, G, grp, N, C, L, /*transpose=*/true);

        // Pack State[N,D] -> transpose -> pb[BC, D, N] (ld=N)
        // ColumnMajor B implicitly transposes pb back to State.
        pack_state_batched_kernel<<<dim3(nd_blocks, BC), 256, 0, stream>>>(st, pb, N, D);

        // Batched GEMM: Out[L,D] = pa(Row)[L,N] @ pb(Col)[D,N]
        launch_batched_gemm<SsdGemmConfig>(
            pa,
            N,
            static_cast<index_t>(L * N), // A: [L,N], batch_stride=L*N
            pb,
            N,
            static_cast<index_t>(D * N), // B: [D,N] col-major, batch_stride=D*N
            rbmm2,
            D,
            static_cast<index_t>(L * D), // C: [L,D], batch_stride=L*D
            L,
            D,
            N,
            BC,
            stream);
    }

    // ====== Step 9: Epilogue ======
    ssd_epilogue_kernel<<<dim3(BEH, C), 256, 0, stream>>>(
        rbmm2, abmm2, ce, p_x, p_dp, p_z, p_y, C, L, D, EH);

    // ====== Step 10: Final state ======
    ssd_final_state_kernel<<<BEH, 256, 0, stream>>>(rbmm1, st, lv, p_fs, C, N, D);

    (void)hipDeviceSynchronize();
    return 0.0f;
}

} // namespace ck_tile
