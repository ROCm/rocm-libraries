// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/config.hpp>

#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/miopen.h>
#include <miopen/performance_config.hpp>
#include <miopen/solver.hpp>
#include <miopen/utility/transposing_solver.hpp>
#include <miopen/conv/data_invoke_params.hpp>

#include <initializer_list>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace miopen {

namespace debug {

/// If set to true, then always enable ConvDirectNaive* solver, regardless of environment value
/// MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_* that control enable/disable of these solvers.
/// Currently used during driver using naive kernel as gpu reference.
MIOPEN_EXPORT extern bool
    AlwaysEnableConvDirectNaive; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)

} // namespace debug

struct AnyInvokeParams;

namespace solver {
/// \todo Move wave_size into abstraction wich represent GPU information
const int wave_size = 64;

namespace conv {

/// Common interface for convolution tunable and non-tunable solvers
using ConvSolverInterface = SolverInterface<ExecutionContext, miopen::conv::ProblemDescription>;

/// Common interface for convolution non-tunable solvers
using ConvSolverInterfaceNonTunable =
    SolverInterfaceNonTunable<ExecutionContext, miopen::conv::ProblemDescription>;

/// Common interface for convolution tunable solvers
using ConvSolverInterfaceTunable =
    SolverInterfaceTunable<ExecutionContext, miopen::conv::ProblemDescription>;

/// Typedef for convolution non-tunable solvers
using ConvSolver = SolverBaseNonTunable<ExecutionContext, miopen::conv::ProblemDescription>;

/// Typedef for convolution tunable solvers
template <class PerformanceConfig>
using ConvTunableSolver =
    SolverBaseTunable<ExecutionContext, miopen::conv::ProblemDescription, PerformanceConfig>;

struct PerformanceConfigConvAsm3x3U : PerfConfigBase<PerformanceConfigConvAsm3x3U>
{
    int limit_wave_cnt;        // [0..9]
    int filters_per_wave;      // [1..8]
    int output_lines_per_wave; // [1..8]

    PerformanceConfigConvAsm3x3U(int lwc, int fpw, int olpw);
    PerformanceConfigConvAsm3x3U() : PerformanceConfigConvAsm3x3U(-1, -1, -1) {}
    PerformanceConfigConvAsm3x3U(bool) : PerformanceConfigConvAsm3x3U(0, 1, 1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.limit_wave_cnt, "limit_wave_cnt");
        f(self.filters_per_wave, "filters_per_wave");
        f(self.output_lines_per_wave, "output_lines_per_wave");
    }

    void HeuristicInit(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConfigConvAsm3x3U& other) const;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsm3x3U final : ConvTunableSolver<PerformanceConfigConvAsm3x3U>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm3x3U>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    PerformanceConfigConvAsm3x3U
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConfigConvAsm3x3U&) const override;
    PerformanceConfigConvAsm3x3U Search(const ExecutionContext&,
                                        const miopen::conv::ProblemDescription&,
                                        const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvAsm3x3U&) const override;
};

struct PerformanceConfigConvAsm1x1U : PerfConfigBase<PerformanceConfigConvAsm1x1U>
{
    // ----------------- // Full set          Optimized       Spare
    // ----------------------------------------------------------------------------
    int read_size;        // [1..4]            <same>          <same>
    int k_mult;           // 1,[4,8,12..32]    2^n[8..32]      1,4
    int chunks_per_wave;  // [1..16]           [1..8]          <same>
    int chunk_size;       // 2^n[1..64]        2^n[16..64]     1,4
    int n_mult;           // [1..8]            [1..4]          <same>
    int c_mult;           // 2^n[1..32]        2^n[1..4]       <same>
    int waves_c_in_group; // [1..8]            [1..4]          <same>
    int waves_k_in_group; // 1,[2,4,8]         1,[2,4,8]       <same>
    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT
    PerformanceConfigConvAsm1x1U(int, int, int, int, int, int, int, int, bool);
    PerformanceConfigConvAsm1x1U()
        : PerformanceConfigConvAsm1x1U(-1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsm1x1U(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.read_size, "read_size");
        f(self.k_mult, "k_mult");
        f(self.chunks_per_wave, "chunks_per_wave");
        f(self.chunk_size, "chunk_size");
        f(self.n_mult, "n_mult");
        f(self.c_mult, "c_mult");
        f(self.waves_c_in_group, "waves_c_in_group");
        f(self.waves_k_in_group, "waves_k_in_group");
    }

    int GetReadSize() const { return read_size; }
    int GetKMult() const { return k_mult; }
    int GetChunksPerWave() const { return chunks_per_wave; }
    int GetChunkSize() const { return chunk_size; }
    int GetNMult() const { return n_mult; }
    int GetCMult() const { return c_mult; }
    int GetWavesCInGroup() const { return waves_c_in_group; }
    int GetWavesKInGroup() const { return waves_k_in_group; }
    int GetNPerGpr() const
    {
        assert(chunk_size);
        return 64 / chunk_size;
    }

    void StaticHeuristic(const miopen::conv::ProblemDescription& problem);
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool
    IsModelApplicable(const ExecutionContext& ctx,
                      const miopen::conv::ProblemDescription& problem) const;
    bool IsValidValue() const { return IsValidValueImpl(8); }
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription& problem) const
    {
        return IsValidImpl(problem, 8);
    }
    bool operator==(const PerformanceConfigConvAsm1x1U& other) const;

private:
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    bool IsPartiallyValid(const miopen::conv::ProblemDescription& problem,
                          int sequence_length) const
    {
        return IsValidImpl(problem, sequence_length);
    }
    bool IsPartiallyValidValue(int sequence_length) const
    {
        return IsValidValueImpl(sequence_length);
    }
    bool RunParameterPredictionModel(const ExecutionContext&,
                                     const miopen::conv::ProblemDescription&);
    bool ModelApplyToken(int index, std::string value, const miopen::conv::ProblemDescription&);
#endif
    bool IsValidImpl(const miopen::conv::ProblemDescription& problem, int sequence_length) const;
    bool IsValidValueImpl(int sequence_length) const;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsm1x1U final : ConvTunableSolver<PerformanceConfigConvAsm1x1U>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm1x1U>(); }

    PerformanceConfigConvAsm1x1U
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConfigConvAsm1x1U&) const override;
    PerformanceConfigConvAsm1x1U Search(const ExecutionContext&,
                                        const miopen::conv::ProblemDescription&,
                                        const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvAsm1x1U&) const override;
};

struct PerformanceConfigConvAsm1x1UV2 : PerfConfigBase<PerformanceConfigConvAsm1x1UV2>
{
    // ----------------- // Full set          Optimized       Spare
    // ----------------------------------------------------------------------------
    int chunk_size;       // 2^n[1..64]        2^n[16..64]     <same>
    int dwords_per_ld;    // [1..4]            1,2,3           <same>
    int k_mult;           // [1..32]           8,16            1,2,3,4
    int c_mult;           // [1..32]           2^n[1..4]       <same>
    int n_mult;           // [1..32]           1,2             <same>
    int w_mult;           // [1..32]           1,2             <same>
    int h_mult;           // [1..32]           1,2             <same>
    int h_per_chunk;      // 2^n[1..64]        [2,4,8]         <same>
    int waves_k_in_group; // [1..8]            2,4             <same>
    int waves_c_in_group; // [1..8]            1,2             <same>
    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT
    PerformanceConfigConvAsm1x1UV2(int, int, int, int, int, int, int, int, int, int, bool);
    PerformanceConfigConvAsm1x1UV2()
        : PerformanceConfigConvAsm1x1UV2(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsm1x1UV2(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.chunk_size, "chunk_size");
        f(self.dwords_per_ld, "dwords_per_ld");
        f(self.k_mult, "k_mult");
        f(self.c_mult, "c_mult");
        f(self.n_mult, "n_mult");
        f(self.w_mult, "w_mult");
        f(self.h_mult, "h_mult");
        f(self.h_per_chunk, "h_per_chunk");
        f(self.waves_k_in_group, "waves_k_in_group");
        f(self.waves_c_in_group, "waves_c_in_group");
    }

    int GetChunkSize() const { return chunk_size; }
    int GetDwordsPerLd() const { return dwords_per_ld; }
    int GetCMult() const { return c_mult; }
    int GetKMult() const { return k_mult; }
    int GetNMult() const { return n_mult; }
    int GetWMult() const { return w_mult; }
    int GetHMult() const { return h_mult; }
    int GetHPerChunk() const { return h_per_chunk; }
    int GetWavesCInGroup() const { return waves_c_in_group; }
    int GetWavesKInGroup() const { return waves_k_in_group; }
    int GetNPerGpr() const
    {
        assert(chunk_size);
        return 64 / chunk_size;
    }

    void HeuristicInit(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConfigConvAsm1x1UV2& other) const;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsm1x1UV2 final
    : ConvTunableSolver<PerformanceConfigConvAsm1x1UV2>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm1x1UV2>(); }

    PerformanceConfigConvAsm1x1UV2
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConfigConvAsm1x1UV2&) const override;
    PerformanceConfigConvAsm1x1UV2 Search(const ExecutionContext&,
                                          const miopen::conv::ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvAsm1x1UV2&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsm5x10u2v2f1 final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm5x10u2v2f1>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsm5x10u2v2b1 final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsm5x10u2v2b1>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsm7x7c3h224w224k64u2v2p3q3f1 final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsm7x7c3h224w224k64u2v2p3q3f1>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipDirectFwd11x11 final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipDirectFwd11x11>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct PerformanceImplicitGemm : PerfConfigBase<PerformanceImplicitGemm>
{
    int BPerBlock; // 2^n[8..16]
    int KPerBlock; // 2^n[32..128]
    int EPerBlock; // 2^n[4..16]

    int GemmNRepeat; // == 2

    int GemmMPerThreadSubC; // 2^n[2..4]
    int GemmNPerThreadSubC; // 2^n[2..4]

    int GemmMLevel0Cluster; // 2^n[1..4]
    int GemmNLevel0Cluster; // 2^n[1..4]
    int GemmMLevel1Cluster; // 2^n[1..4]
    int GemmNLevel1Cluster; // 2^n[1..4]

    int InBlockCopyClusterLengths_E;  // 2^n[4..16]
    int InBlockCopyClusterLengths_B;  // 2^n[8..16]
    int InBlockCopyClusterLengths_N1; // 2^n[1..2]
    int InBlockCopyClusterLengths_N2; // 2^n[1..4]

    int WeiBlockCopyClusterLengths_E; // 2^n[1..4]
    int WeiBlockCopyClusterLengths_K; // 2^n[16..128]

    bool use_spare_set;

    PerformanceImplicitGemm(
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, bool);

    PerformanceImplicitGemm()
        : PerformanceImplicitGemm(
              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemm(bool spare);

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BPerBlock, "BPerBlock");
        f(self.KPerBlock, "KPerBlock");
        f(self.EPerBlock, "EPerBlock");
        f(self.GemmNRepeat, "GemmNRepeat");
        f(self.GemmMPerThreadSubC, "GemmMPerThreadSubC");
        f(self.GemmNPerThreadSubC, "GemmNPerThreadSubC");
        f(self.GemmMLevel0Cluster, "GemmMLevel0Cluster");
        f(self.GemmNLevel0Cluster, "GemmNLevel0Cluster");
        f(self.GemmMLevel1Cluster, "GemmMLevel1Cluster");
        f(self.GemmNLevel1Cluster, "GemmNLevel1Cluster");
        f(self.InBlockCopyClusterLengths_E, "InBlockCopyClusterLengths_E");
        f(self.InBlockCopyClusterLengths_N1, "InBlockCopyClusterLengths_N1");
        f(self.InBlockCopyClusterLengths_B, "InBlockCopyClusterLengths_B");
        f(self.InBlockCopyClusterLengths_N2, "InBlockCopyClusterLengths_N2");
        f(self.WeiBlockCopyClusterLengths_E, "WeiBlockCopyClusterLengths_E");
        f(self.WeiBlockCopyClusterLengths_K, "WeiBlockCopyClusterLengths_K");
    }

    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceImplicitGemm& other) const;
};

struct PerformanceImplicitGemmV4R1 : public PerformanceImplicitGemm
{
    PerformanceImplicitGemmV4R1(int a,
                                int b,
                                int c,
                                int d,
                                int e,
                                int f,
                                int g,
                                int h,
                                int i,
                                int j,
                                int k,
                                int l,
                                int m,
                                int n,
                                int o,
                                int p,
                                bool q)
        : PerformanceImplicitGemm(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)
    {
    }

    PerformanceImplicitGemmV4R1()
        : PerformanceImplicitGemmV4R1(
              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmV4R1(bool spare) : PerformanceImplicitGemm(spare) {}

    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
};

struct PerformanceImplicitGemmV4R4Fwd : PerfConfigBase<PerformanceImplicitGemmV4R4Fwd>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    PerformanceImplicitGemmV4R4Fwd(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmV4R4Fwd(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmV4R4Fwd(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmV4R4Fwd() : PerformanceImplicitGemmV4R4Fwd(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmV4R4Fwd(bool spare);

    bool operator==(const PerformanceImplicitGemmV4R4Fwd& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    std::tuple<int, bool> CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, bool> CalculateBlockGemmPerformanceParameters() const;
    std::tuple<int, int, int, int, bool> CalculateGemmABlockCopyPerformanceParameters() const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
};

struct PerformanceImplicitGemmV4R4WrW : PerfConfigBase<PerformanceImplicitGemmV4R4WrW>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    PerformanceImplicitGemmV4R4WrW(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmV4R4WrW(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmV4R4WrW(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmV4R4WrW() : PerformanceImplicitGemmV4R4WrW(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmV4R4WrW(bool spare);

    bool operator==(const PerformanceImplicitGemmV4R4WrW& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    std::tuple<int, bool> CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, bool> CalculateBlockGemmPerformanceParameters() const;
    std::tuple<int, int, int, int, bool> CalculateGemmABlockCopyPerformanceParameters() const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
};

struct PerformanceImplicitGemmBwdDataV1R1 : PerfConfigBase<PerformanceImplicitGemmBwdDataV1R1>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    PerformanceImplicitGemmBwdDataV1R1(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmBwdDataV1R1()
        : PerformanceImplicitGemmBwdDataV1R1(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmBwdDataV1R1(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmBwdDataV1R1(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmBwdDataV1R1(bool spare);

    bool operator==(const PerformanceImplicitGemmBwdDataV1R1& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    std::tuple<int, bool> CalculateGridSize(const ExecutionContext&,
                                            const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, bool> CalculateBlockGemmPerformanceParameters() const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const ExecutionContext&,
                                                 const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const ExecutionContext&,
                                                 const miopen::conv::ProblemDescription&) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const;
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
};

struct PerformanceImplicitGemmBwdDataV4R1 : PerfConfigBase<PerformanceImplicitGemmBwdDataV4R1>
{
    int BlockSize;

    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;

    int GemmMPerThread;
    int GemmNPerThread;

    bool use_spare_set;

    PerformanceImplicitGemmBwdDataV4R1(int, int, int, int, int, int, bool);

    PerformanceImplicitGemmBwdDataV4R1()
        : PerformanceImplicitGemmBwdDataV4R1(-1, -1, -1, -1, -1, -1, false)
    {
    }

    PerformanceImplicitGemmBwdDataV4R1(int a, int b, int c, int d, int e, int f)
        : PerformanceImplicitGemmBwdDataV4R1(a, b, c, d, e, f, false)
    {
    }

    PerformanceImplicitGemmBwdDataV4R1(bool spare);

    bool operator==(const PerformanceImplicitGemmBwdDataV4R1& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    std::tuple<int, bool> CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, bool> CalculateBlockGemmPerformanceParameters() const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, bool>
    CalculateGemmCThreadCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<std::size_t, bool> MIOPEN_INTERNALS_EXPORT
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
};

struct PerformanceImplicitGemmBwdDataV4R1Xdlops
    : PerfConfigBase<PerformanceImplicitGemmBwdDataV4R1Xdlops>
{
    int GemmNPerBlock; // 2^n[8..16]
    int GemmMPerBlock; // 2^n[32..128]
    int GemmKPerBlock; // 2^n[4..16]

    int GemmKPACKSize; // 2^[1..4]

    int GemmMPerWave;
    int GemmNPerWave;

    // GemmAThreadCopyMoreGemmK is currently a fix value, is untunable
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;

    bool use_spare_set;
    MIOPEN_INTERNALS_EXPORT
    PerformanceImplicitGemmBwdDataV4R1Xdlops(int, int, int, int, int, int, bool, bool, bool);

    PerformanceImplicitGemmBwdDataV4R1Xdlops();
    PerformanceImplicitGemmBwdDataV4R1Xdlops(bool spare);
    PerformanceImplicitGemmBwdDataV4R1Xdlops(
        int a, int b, int c, int d, int e, int f, bool g, bool h)
        : PerformanceImplicitGemmBwdDataV4R1Xdlops(a, b, c, d, e, f, g, h, false)
    {
    }

    bool operator==(const PerformanceImplicitGemmBwdDataV4R1Xdlops& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmKPACKSize, "GemmKPACKSize");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
    }

    std::tuple<int, bool> CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool IsReallyValid(const miopen::conv::ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ExecutionContext&,
                                 const miopen::conv::ProblemDescription&) const;
    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmV4R1Fwd final
    : ConvTunableSolver<PerformanceImplicitGemmV4R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R1Fwd>();
    }

    PerformanceImplicitGemmV4R1
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceImplicitGemmV4R1&) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmV4R1&) const override;
    PerformanceImplicitGemmV4R1 Search(const ExecutionContext&,
                                       const miopen::conv::ProblemDescription&,
                                       const AnyInvokeParams& invoke_ctx) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmV4R4Fwd final
    : ConvTunableSolver<PerformanceImplicitGemmV4R4Fwd>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R4Fwd>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    PerformanceImplicitGemmV4R4Fwd
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceImplicitGemmV4R4Fwd&) const override;
    PerformanceImplicitGemmV4R4Fwd Search(const ExecutionContext&,
                                          const miopen::conv::ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmV4R4Fwd&) const override;

private:
    static std::tuple<int, int, int> CalculateGemmSize(const miopen::conv::ProblemDescription&);

    friend struct PerformanceImplicitGemmV4R4Fwd;
};

struct PerformanceConvMlirIgemm : PerfConfigBase<PerformanceConvMlirIgemm>
{
    int BlockSize;
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerThread;
    int GemmNPerThread;
    bool use_spare_set;

    /// \ref https://github.com/ROCm/MIOpen/issues/1154
    static PerformanceConvMlirIgemm& MlirHeuristicInitRequest()
    {
        static PerformanceConvMlirIgemm heur;
        heur.SetMlirHeuristicInitRequest();
        return heur;
    }

    PerformanceConvMlirIgemm(int, int, int, int, int, int, bool);

    PerformanceConvMlirIgemm(int a, int b, int c, int d, int e, int f)
        : PerformanceConvMlirIgemm(a, b, c, d, e, f, false)
    {
    }

    PerformanceConvMlirIgemm() : PerformanceConvMlirIgemm(-1, -1, -1, -1, -1, -1, false) {}

    PerformanceConvMlirIgemm(bool spare);

    bool operator==(const PerformanceConvMlirIgemm& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.BlockSize, "BlockSize");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerThread, "GemmMPerThread");
        f(self.GemmNPerThread, "GemmNPerThread");
    }

    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool SetNextValue(const miopen::conv::ProblemDescription&);

private:
    void SetMlirHeuristicInitRequest();
};

struct ConvMlirIgemmFwd final : ConvTunableSolver<PerformanceConvMlirIgemm>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvMlirIgemmFwd>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    PerformanceConvMlirIgemm
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConvMlirIgemm&) const override;
    PerformanceConvMlirIgemm Search(const ExecutionContext&,
                                    const miopen::conv::ProblemDescription&,
                                    const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemm&) const override;
};

struct PerformanceConvMlirIgemmXdlops : PerfConfigBase<PerformanceConvMlirIgemmXdlops>
{
    int GemmMPerBlock; // 2^n[32..128]
    int GemmNPerBlock; // 2^n[8..16]
    int GemmKPerBlock; // 2^n[4..16]
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPACKSize; // 2^[1..4]

    // GemmAThreadCopyMoreGemmK is currently a fix value, is untunable
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;

    bool use_spare_set;

    /// \ref https://github.com/ROCm/MIOpen/issues/1154
    static PerformanceConvMlirIgemmXdlops& MlirHeuristicInitRequest()
    {
        static PerformanceConvMlirIgemmXdlops heur;
        heur.SetMlirHeuristicInitRequest();
        return heur;
    }

    MIOPEN_INTERNALS_EXPORT
    PerformanceConvMlirIgemmXdlops(int, int, int, int, int, int, bool, bool, bool);

    PerformanceConvMlirIgemmXdlops();
    PerformanceConvMlirIgemmXdlops(bool spare);
    PerformanceConvMlirIgemmXdlops(int a, int b, int c, int d, int e, int f, bool g, bool h)
        : PerformanceConvMlirIgemmXdlops(a, b, c, d, e, f, g, h, false)
    {
    }

    bool operator==(const PerformanceConvMlirIgemmXdlops& other) const;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPACKSize, "GemmKPACKSize");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
    }

    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool SetNextValue(const miopen::conv::ProblemDescription&);

private:
    void SetMlirHeuristicInitRequest();
};

struct ConvMlirIgemmFwdXdlops final : ConvTunableSolver<PerformanceConvMlirIgemmXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvMlirIgemmFwdXdlops>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    PerformanceConvMlirIgemmXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConvMlirIgemmXdlops&) const override;
    PerformanceConvMlirIgemmXdlops Search(const ExecutionContext&,
                                          const miopen::conv::ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemmXdlops&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmV4R4WrW final
    : ConvTunableSolver<PerformanceImplicitGemmV4R4WrW>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R4WrW>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    PerformanceImplicitGemmV4R4WrW
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceImplicitGemmV4R4WrW&) const override;
    PerformanceImplicitGemmV4R4WrW Search(const ExecutionContext&,
                                          const miopen::conv::ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmV4R4WrW&) const override;

private:
    static std::tuple<int, int, int> CalculateGemmSize(const miopen::conv::ProblemDescription&);

    friend struct PerformanceImplicitGemmV4R4WrW;
};

struct ConvMlirIgemmWrW final : ConvTunableSolver<PerformanceConvMlirIgemm>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvMlirIgemmWrW>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    PerformanceConvMlirIgemm
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConvMlirIgemm&) const override;
    PerformanceConvMlirIgemm Search(const ExecutionContext&,
                                    const miopen::conv::ProblemDescription&,
                                    const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemm&) const override;
};

struct ConvMlirIgemmWrWXdlops final : ConvTunableSolver<PerformanceConvMlirIgemmXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvMlirIgemmWrWXdlops>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    PerformanceConvMlirIgemmXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConvMlirIgemmXdlops&) const override;
    PerformanceConvMlirIgemmXdlops Search(const ExecutionContext&,
                                          const miopen::conv::ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemmXdlops&) const override;
};

struct PerformanceImplicitGemmForwardV4R4Xdlops
    : PerfConfigBase<PerformanceImplicitGemmForwardV4R4Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;
    int GemmBThreadDataPerRead_GemmN;

    MIOPEN_INTERNALS_EXPORT
    PerformanceImplicitGemmForwardV4R4Xdlops(int, int, int, int, int, int, bool, bool, int);
    PerformanceImplicitGemmForwardV4R4Xdlops();
    PerformanceImplicitGemmForwardV4R4Xdlops(bool) : PerformanceImplicitGemmForwardV4R4Xdlops() {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
        f(self.GemmBThreadDataPerRead_GemmN, "GemmBThreadDataPerRead_GemmN");
    }

    bool operator==(const PerformanceImplicitGemmForwardV4R4Xdlops& other) const;

    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool IsReallyValid(const miopen::conv::ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ExecutionContext&,
                                 const miopen::conv::ProblemDescription&) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceImplicitGemmForwardV4R5Xdlops
    : PerfConfigBase<PerformanceImplicitGemmForwardV4R5Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;
    int GemmBThreadDataPerRead_GemmN;

    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT
    PerformanceImplicitGemmForwardV4R5Xdlops(int, int, int, int, int, int, bool, bool, int, bool);
    PerformanceImplicitGemmForwardV4R5Xdlops();
    PerformanceImplicitGemmForwardV4R5Xdlops(bool spare);

    PerformanceImplicitGemmForwardV4R5Xdlops(
        int a, int b, int c, int d, int e, int f, bool g, bool h, int i)
        : PerformanceImplicitGemmForwardV4R5Xdlops(a, b, c, d, e, f, g, h, i, false)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
        f(self.GemmBThreadDataPerRead_GemmN, "GemmBThreadDataPerRead_GemmN");
    }

    bool operator==(const PerformanceImplicitGemmForwardV4R5Xdlops& other) const;

    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool IsReallyValid(const miopen::conv::ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ExecutionContext&,
                                 const miopen::conv::ProblemDescription&) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm
    : PerfConfigBase<PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    int GemmMFactor;
    int GemmNFactor;
    int GemmKFactor;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;
    int GemmBThreadDataPerRead_GemmN;

    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm(
        int, int, int, int, int, int, int, int, int, bool, bool, int);
    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm();
    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm(bool)
        : PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm()
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmMFactor, "GemmMFactor");
        f(self.GemmNFactor, "GemmNFactor");
        f(self.GemmKFactor, "GemmKFactor");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
        f(self.GemmBThreadDataPerRead_GemmN, "GemmBThreadDataPerRead_GemmN");
    }

    bool operator==(const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm& other) const;

    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool IsReallyValid(const miopen::conv::ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ExecutionContext&,
                                 const miopen::conv::ProblemDescription&) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct PerformanceImplicitGemmBwdV1R1Xdlops : PerfConfigBase<PerformanceImplicitGemmBwdV1R1Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmKPack;

    MIOPEN_INTERNALS_EXPORT
    PerformanceImplicitGemmBwdV1R1Xdlops(int, int, int, int, int, int, bool, bool);
    PerformanceImplicitGemmBwdV1R1Xdlops();
    PerformanceImplicitGemmBwdV1R1Xdlops(bool) : PerformanceImplicitGemmBwdV1R1Xdlops() {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmKPack, "GemmBThreadCopyMoreGemmKPack");
    }

    bool operator==(const PerformanceImplicitGemmBwdV1R1Xdlops& other) const;

    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool IsReallyValid(const miopen::conv::ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ExecutionContext&,
                                 const miopen::conv::ProblemDescription&) const;

    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmForwardV4R4Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmForwardV4R4Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmForwardV4R4Xdlops>();
    }

    PerformanceImplicitGemmForwardV4R4Xdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceImplicitGemmForwardV4R4Xdlops&) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmForwardV4R4Xdlops&) const override;
    PerformanceImplicitGemmForwardV4R4Xdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;

private:
    static std::tuple<int, int, int, int>
    CalculateGemmSize(const miopen::conv::ProblemDescription&);

    friend struct PerformanceImplicitGemmForwardV4R4Xdlops;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm final
    : ConvTunableSolver<PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm>();
    }

    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm&) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm&) const override;
    PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;

private:
    static std::tuple<int, int, int, int, int, int, int> CalculateGemmSize(
        const miopen::conv::ProblemDescription&, int GemmMFactor, int GemmNFactor, int GemmKFactor);

    friend struct PerformanceImplicitGemmForwardV4R4Xdlops_Padded_Gemm;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmForwardV4R5Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmForwardV4R5Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmForwardV4R5Xdlops>();
    }

    PerformanceImplicitGemmForwardV4R5Xdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceImplicitGemmForwardV4R5Xdlops&) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmForwardV4R5Xdlops&) const override;
    PerformanceImplicitGemmForwardV4R5Xdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmV4R1WrW final
    : ConvTunableSolver<PerformanceImplicitGemmV4R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmV4R1WrW>();
    }

    PerformanceImplicitGemmV4R1
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceImplicitGemmV4R1&) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmV4R1&) const override;
    PerformanceImplicitGemmV4R1 Search(const ExecutionContext&,
                                       const miopen::conv::ProblemDescription&,
                                       const AnyInvokeParams& invoke_ctx) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmBwdDataV1R1 final
    : ConvTunableSolver<PerformanceImplicitGemmBwdDataV1R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV1R1>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    PerformanceImplicitGemmBwdDataV1R1
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceImplicitGemmBwdDataV1R1&) const override;
    PerformanceImplicitGemmBwdDataV1R1 Search(const ExecutionContext&,
                                              const miopen::conv::ProblemDescription&,
                                              const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmBwdDataV1R1&) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

private:
    static std::tuple<int, int, int> CalculateGemmSize(const ExecutionContext&,
                                                       const miopen::conv::ProblemDescription&);

    friend struct PerformanceImplicitGemmBwdDataV1R1;
};

struct ConvMlirIgemmBwd final : ConvTunableSolver<PerformanceConvMlirIgemm>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvMlirIgemmBwd>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    PerformanceConvMlirIgemm
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConvMlirIgemm&) const override;
    PerformanceConvMlirIgemm Search(const ExecutionContext&,
                                    const miopen::conv::ProblemDescription&,
                                    const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemm&) const override;
};

struct ConvMlirIgemmBwdXdlops final : ConvTunableSolver<PerformanceConvMlirIgemmXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvMlirIgemmBwdXdlops>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    PerformanceConvMlirIgemmXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConvMlirIgemmXdlops&) const override;
    PerformanceConvMlirIgemmXdlops Search(const ExecutionContext&,
                                          const miopen::conv::ProblemDescription&,
                                          const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvMlirIgemmXdlops&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmBwdDataV4R1 final
    : ConvTunableSolver<PerformanceImplicitGemmBwdDataV4R1>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV4R1>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    PerformanceImplicitGemmBwdDataV4R1
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceImplicitGemmBwdDataV4R1&) const override;
    PerformanceImplicitGemmBwdDataV4R1 Search(const ExecutionContext&,
                                              const miopen::conv::ProblemDescription&,
                                              const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmBwdDataV4R1&) const override;

private:
    static int CalculateNumberOfGemm(const miopen::conv::ProblemDescription&);
    static std::tuple<int, int, int> CalculateGemmSize(const miopen::conv::ProblemDescription&,
                                                       int gemm_id);

    friend struct PerformanceImplicitGemmBwdDataV4R1;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmBwdDataV4R1Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmBwdDataV4R1Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV4R1Xdlops>();
    }

    PerformanceImplicitGemmBwdDataV4R1Xdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceImplicitGemmBwdDataV4R1Xdlops&) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmBwdDataV4R1Xdlops&) const override;
    PerformanceImplicitGemmBwdDataV4R1Xdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;

private:
    static int CalculateNumberOfGemm(const miopen::conv::ProblemDescription&);
    static std::tuple<int, int, int, int> CalculateGemmSize(const miopen::conv::ProblemDescription&,
                                                            int gemm_id);

    friend struct PerformanceImplicitGemmBwdDataV4R1Xdlops;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmBwdDataV1R1Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmBwdV1R1Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmBwdDataV1R1Xdlops>();
    }

    PerformanceImplicitGemmBwdV1R1Xdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceImplicitGemmBwdV1R1Xdlops&) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    PerformanceImplicitGemmBwdV1R1Xdlops Search(const ExecutionContext&,
                                                const miopen::conv::ProblemDescription&,
                                                const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmBwdV1R1Xdlops&) const override;

private:
    static std::tuple<int, int, int, int>
    CalculateGemmSize(const miopen::conv::ProblemDescription&);

    friend struct PerformanceImplicitGemmBwdV1R1Xdlops;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsmImplicitGemmV4R1DynamicFwd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicFwd>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsmImplicitGemmV4R1DynamicFwd_1x1 final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicFwd_1x1>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsmImplicitGemmV4R1DynamicWrw final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicWrw>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsmImplicitGemmGTCDynamicWrwXdlops final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicWrwXdlops>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsmImplicitGemmV4R1DynamicBwd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmV4R1DynamicBwd>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsmImplicitGemmGTCDynamicFwdXdlops final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicFwdXdlops>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsmImplicitGemmGTCDynamicBwdXdlops final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicBwdXdlops>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

/// Holds common member functions for the Solvers which share the same
/// "legacy exhaustive search" machinery.
struct MIOPEN_INTERNALS_EXPORT ConvOclDirectFwdLegacyExhaustiveSearch
    : ConvTunableSolver<LegacyPerformanceConfig>
{
    LegacyPerformanceConfig
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    LegacyPerformanceConfig Search(const ExecutionContext&,
                                   const miopen::conv::ProblemDescription&,
                                   const AnyInvokeParams& invoke_ctx) const override;

private:
    template <typename Tgpu>
    LegacyPerformanceConfig SearchImpl(const ExecutionContext&,
                                       const miopen::conv::ProblemDescription&,
                                       const AnyInvokeParams& invoke_ctx) const;
};

struct MIOPEN_INTERNALS_EXPORT ConvOclDirectFwd final : ConvOclDirectFwdLegacyExhaustiveSearch
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclDirectFwd>(); }

    static ConvSolution BaseGetSolution(const ExecutionContext&,
                                        const miopen::conv::ProblemDescription&,
                                        const LegacyPerformanceConfig&);

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const LegacyPerformanceConfig&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const LegacyPerformanceConfig&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvBinWinograd3x3U final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvBinWinograd3x3U>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvBinWinogradRxS final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvBinWinogradRxS>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct PerformanceConfigConvBinWinogradRxS : PerfConfigBase<PerformanceConfigConvBinWinogradRxS>
{
    int n_groups;
    PerformanceConfigConvBinWinogradRxS(int n_groups_);
    PerformanceConfigConvBinWinogradRxS() : PerformanceConfigConvBinWinogradRxS(-1) {}
    PerformanceConfigConvBinWinogradRxS(bool) : PerformanceConfigConvBinWinogradRxS(1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.n_groups, "n_groups");
    }
    int GetNGroups() const { return n_groups; }

    template <int Winodata, int Winofilter>
    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext& ctx, const miopen::conv::ProblemDescription&) const
    {
        return IsValid(ctx);
    }
    bool IsValid(const ExecutionContext&) const;
    bool operator==(const PerformanceConfigConvBinWinogradRxS& other) const;
};

template <int Winodata, int Winofilter>
struct ConvBinWinoRxS final : ConvTunableSolver<PerformanceConfigConvBinWinogradRxS>
{
    const std::string& SolverDbId() const override { return GetSolverDbId(); }

    static const std::string& GetSolverDbId()
    {
        static const std::string dbId = std::string("ConvBinWinogradRxSf")
                                            .append(std::to_string(Winodata))
                                            .append("x")
                                            .append(std::to_string(Winofilter));
        return dbId;
    }

    PerformanceConfigConvBinWinogradRxS
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConfigConvBinWinogradRxS&) const override;
    PerformanceConfigConvBinWinogradRxS Search(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&,
                                               const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvBinWinogradRxS&) const override;

private:
    static size_t GetNGroups(const size_t group_conv, const size_t grid_group_size)
    {
        assert(group_conv != 0);
        return grid_group_size / group_conv;
    }
};

// Suppress misleading clang warnings
#ifndef CONV_BIN_WINO_RXS_CPP

extern template struct ConvBinWinoRxS<2, 3>;
extern template struct ConvBinWinoRxS<3, 2>;

#endif

struct MIOPEN_INTERNALS_EXPORT ConvBinWinogradRxSf2x3g1 final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvBinWinogradRxSf2x3g1>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct ConvMPBidirectWinograd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<
            ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;

    // kernel_file_name for solver identification
    static fs::path GetSolverFileNames(int id)
    {
        static const fs::path names[3] = {"xform_bidirect_winograd_data.s",
                                          "xform_bidirect_winograd_filter.s",
                                          "xform_bidirect_winograd_out.s"};
        return names[id];
    }

    static std::string GetSolverKernelNames(int id)
    {
        static const std::string name_suffix =
            '_' + std::to_string(WinoDataH) + '_' + std::to_string(WinoDataW) + '_' +
            std::to_string(WinoFilterH) + '_' + std::to_string(WinoFilterW);
        static const std::string names[3] = {
            "miopenGcnAsmMPBidirectWinogradXformData" + name_suffix,
            "miopenGcnAsmMPBidirectWinogradXformFilter" + name_suffix,
            "miopenGcnAsmMPBidirectWinogradXformOut" + name_suffix};
        return names[id];
    }

    static int GetSolverWinoXformHWSize() { return WinoDataH + WinoFilterH - 1; }
};

#ifndef CONV_MP_BIDIRECTIONAL_WINOGRAD_CPP
extern template struct ConvMPBidirectWinograd<2, 3>;
extern template struct ConvMPBidirectWinograd<3, 3>;
extern template struct ConvMPBidirectWinograd<4, 3>;
extern template struct ConvMPBidirectWinograd<5, 3>;
extern template struct ConvMPBidirectWinograd<6, 3>;
#endif

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct ConvMPBidirectWinograd_xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmForwardV4R4Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<
            ConvMPBidirectWinograd_xdlops<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override
    {
        return ConvHipImplicitGemmForwardV4R4Xdlops{}.IsDynamic() &&
               ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>{}
                   .IsDynamic() &&
               IsThisSolverDynamic();
    }

    PerformanceImplicitGemmForwardV4R4Xdlops
    GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                const miopen::conv::ProblemDescription& problem) const override
    {
        const auto xdlops_problem = GetTransformedProblem(problem);
        const auto xdlops_ctx     = GetTransformedConvContext(ctx, xdlops_problem);

        return ConvHipImplicitGemmForwardV4R4Xdlops{}.GetDefaultPerformanceConfig(xdlops_ctx,
                                                                                  xdlops_problem);
    }

    bool
    IsValidPerformanceConfig(const ExecutionContext& ctx,
                             const miopen::conv::ProblemDescription& problem,
                             const PerformanceImplicitGemmForwardV4R4Xdlops& config) const override
    {
        const auto xdlops_problem = GetTransformedProblem(problem);
        const auto xdlops_ctx     = GetTransformedConvContext(ctx, xdlops_problem);

        return ConvHipImplicitGemmForwardV4R4Xdlops{}.IsValidPerformanceConfig(
            xdlops_ctx, xdlops_problem, config);
    }

    size_t GetWorkspaceSize(const ExecutionContext& ctx,
                            const miopen::conv::ProblemDescription& problem) const override
    {
        const auto xdlops_problem = GetTransformedProblem(problem);
        const auto xdlops_ctx     = GetTransformedConvContext(ctx, xdlops_problem);

        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>()
                   .GetWorkspaceSize(ctx, problem) +
               ConvHipImplicitGemmForwardV4R4Xdlops{}.GetWorkspaceSize(xdlops_ctx, xdlops_problem);
    }

    bool MayNeedWorkspace() const override { return true; }

    PerformanceImplicitGemmForwardV4R4Xdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmForwardV4R4Xdlops&) const override;

private:
    ExecutionContext
    GetTransformedConvContext(const ExecutionContext& ctx,
                              const miopen::conv::ProblemDescription& transformed_problem) const;
    miopen::conv::ProblemDescription
    GetTransformedProblem(const miopen::conv::ProblemDescription& problem) const;

    // kernel_file_name for solver identification
    static fs::path GetSolverFileNames(int id)
    {
        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverFileNames(id);
    }

    static std::string GetSolverKernelNames(int id)
    {
        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverKernelNames(id);
    }

    static int GetSolverWinoXformHWSize()
    {
        return ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverWinoXformHWSize();
    }

    bool IsThisSolverDynamic() const { return true; }
};

// To suppress misleading clang warnings
#if defined(__clang__) && defined(CONV_MP_BIDIRECTIONAL_WINOGRAD_CPP)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-template-vtables"
#endif

extern template struct ConvMPBidirectWinograd_xdlops<2, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<3, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<4, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<5, 3>;
extern template struct ConvMPBidirectWinograd_xdlops<6, 3>;

#if defined(__clang__) && defined(CONV_MP_BIDIRECTIONAL_WINOGRAD_CPP)
#pragma clang diagnostic pop
#endif

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct ConvWinograd3x3MultipassWrW final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<
            ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    bool IsDynamic() const override { return true; }

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;

    // kernel_file_name for solver identification
    static fs::path GetSolverFileNames(int id)
    {
        static const fs::path names[3] = {"xform_data.s", "xform_filter.s", "xform_out.s"};
        return names[id];
    }

    static std::string GetSolverKernelNames(int id)
    {
        static const std::string name_suffix =
            '_' + std::to_string(WinoDataH) + '_' + std::to_string(WinoDataW) + '_' +
            std::to_string(WinoFilterH) + '_' + std::to_string(WinoFilterW);
        static const std::string names[3] = {"miopenGcnAsmWinogradXformData" + name_suffix,
                                             "miopenGcnAsmWinogradXformFilter" + name_suffix,
                                             "miopenGcnAsmWinogradXformOut" + name_suffix};

        return names[id];
    }

    static int GetGroupCountMult() { return 4; }

    static int GetSolverWinoXformHWSize(const miopen::conv::ProblemDescription& problem, int id)
    {
        if(id == 0)
        {
            return WinoDataH +
                   (WinoFilterH - 1) * (WinoDataH == 7 ? 2 : problem.GetKernelStrideH());
        }
        else
        {
            return WinoDataW +
                   (WinoFilterW - 1) * (WinoDataW == 7 ? 2 : problem.GetKernelStrideW());
        }
    }

private:
    InvokerFactory PrepareInvokerFactory(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription&,
                                         std::size_t ws_sz) const;
};

#ifndef CONV_MULTIPASS_WINO3X3WRW_CPP
extern template struct ConvWinograd3x3MultipassWrW<3, 2>;
extern template struct ConvWinograd3x3MultipassWrW<3, 3>;
extern template struct ConvWinograd3x3MultipassWrW<3, 4>;
extern template struct ConvWinograd3x3MultipassWrW<3, 5>;
extern template struct ConvWinograd3x3MultipassWrW<3, 6>;
extern template struct ConvWinograd3x3MultipassWrW<7, 2>;
extern template struct ConvWinograd3x3MultipassWrW<7, 3>;
extern template struct ConvWinograd3x3MultipassWrW<1, 1, 7, 2>;
extern template struct ConvWinograd3x3MultipassWrW<1, 1, 7, 3>;
extern template struct ConvWinograd3x3MultipassWrW<7, 2, 1, 1>;
extern template struct ConvWinograd3x3MultipassWrW<7, 3, 1, 1>;
extern template struct ConvWinograd3x3MultipassWrW<5, 3>;
extern template struct ConvWinograd3x3MultipassWrW<5, 4>;
#endif

struct PerformanceConfigAsmDirect3x3WrW : PerfConfigBase<PerformanceConfigAsmDirect3x3WrW>
{
    int limit_wave_cnt;   // [0..9]
    int reverse_inout;    // [0..1], 1 is allowed for stride=1x1 only.
    int chunk_size;       // {16,8}, Smaller values increase register pressure.
    int k_per_wave;       // {1,2,4,8} && ((chunk_size * k_per_wave) <= 64).
                          // Higher values increase register pressure.
    int pipe_lines_depth; // [1..16] && (pipe_lines_depth <= img_h).
                          // Higher values increase register pressure.
    int n_per_group;      // [1..8] && (n_per_group <= batch_size).

    PerformanceConfigAsmDirect3x3WrW(int lwc, int rio, int csz, int kpw, int pld, int npg);
    PerformanceConfigAsmDirect3x3WrW() : PerformanceConfigAsmDirect3x3WrW(-1, -1, -1, -1, -1, -1) {}
    PerformanceConfigAsmDirect3x3WrW(bool) : PerformanceConfigAsmDirect3x3WrW(0, 0, 8, 1, 1, 1) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.limit_wave_cnt, "limit_wave_cnt");
        f(self.reverse_inout, "reverse_inout");
        f(self.chunk_size, "chunk_size");
        f(self.k_per_wave, "k_per_wave");
        f(self.pipe_lines_depth, "pipe_lines_depth");
        f(self.n_per_group, "n_per_group");
    }

    int GetLimitWaveCnt() const { return limit_wave_cnt; }
    int GetReverseInout() const { return reverse_inout; }
    int GetChunkSize() const { return chunk_size; }
    int GetKPerWave() const { return k_per_wave; }
    int GetPipeLinesDepth() const { return pipe_lines_depth; }
    int GetNPerGroup() const { return n_per_group; }
    int GetCPerWave() const
    {
        assert(chunk_size);
        return 64 / chunk_size;
    }

    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConfigAsmDirect3x3WrW& other) const;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsmBwdWrW3x3 final
    : ConvTunableSolver<PerformanceConfigAsmDirect3x3WrW>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsmBwdWrW3x3>(); }

    PerformanceConfigAsmDirect3x3WrW
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConfigAsmDirect3x3WrW&) const override;
    PerformanceConfigAsmDirect3x3WrW Search(const ExecutionContext&,
                                            const miopen::conv::ProblemDescription&,
                                            const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigAsmDirect3x3WrW& config) const override;
};

template <uint32_t Winodata, uint32_t Winofilter>
struct ConvWinoFuryRxS final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvWinoFuryRxS<Winodata, Winofilter>>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

#ifndef CONV_WINO_FURY_RXS_CPP
extern template struct ConvWinoFuryRxS<2, 3>;
// extern template struct ConvWinoFuryRxS<3, 2>;
#endif

template <uint32_t Winodata, uint32_t Winofilter>
struct ConvWinoRageRxS final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvWinoRageRxS<Winodata, Winofilter>>();
    }
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

#ifndef CONV_WINO_RAGE_RXS_CPP
extern template struct ConvWinoRageRxS<2, 3>;
#endif

struct PerformanceConfigConvAsmBwdWrW1x1 : PerfConfigBase<PerformanceConfigConvAsmBwdWrW1x1>
{

    int chunk_size;    // {1,2,4,8,16}
    int c_per_gpr;     // {1,2,4,8,16}
    int c_mult;        // {1,2,4,8,16}
    int k_per_gpr;     // {1,2,4,8,16}
    int k_mult;        // {1,2,4,8,16}
    int n_per_gpr;     // {1,2,4}
    int n_part_cnt;    // [1..8]
    int read_size;     // [1..4]
    int short_store;   // {0,1}
    int data_prefetch; // [0..4]
    bool use_spare_set;

    /// The following conditions must be met.
    ///
    /// Shader design-related constraints:
    /// - (A) (chunk_size * c_per_gpr) == 16
    /// - (B) k_per_gpr <= c_per_gpr
    /// - (C) (c_mult > 1 || k_mult > 1)
    ///         ? ((fwd_C % (c_per_gpr * c_mult) == 0) && (fwd_K % (k_per_gpr * k_mult) == 0))
    ///         : (true)
    ///
    /// Resource-related constraints:
    /// - (D) c_mult * k_mult * k_per_gpr + 9 + (c_mult + k_mult) * read_size * pipe_depth <= 256
    ///
    /// Where:
    /// - fwd_C := Num input channels for forward convolution (-c).
    ///   For backward, this is actually n_outputs.
    /// - fwd_K := Num output channels for forward convolution (-k).
    ///   For backward, this is actually n_inputs.

    PerformanceConfigConvAsmBwdWrW1x1(int chunk_size_,
                                      int c_per_gpr_,
                                      int c_mult_,
                                      int k_per_gpr_,
                                      int k_mult_,
                                      int n_per_gpr_,
                                      int n_part_cnt_,
                                      int read_size_,
                                      int short_store_,
                                      int data_prefetch_,
                                      bool);
    PerformanceConfigConvAsmBwdWrW1x1()
        : PerformanceConfigConvAsmBwdWrW1x1(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, false)
    {
    }
    PerformanceConfigConvAsmBwdWrW1x1(bool spare)
        : PerformanceConfigConvAsmBwdWrW1x1(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, spare)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.chunk_size, "chunk_size");
        f(self.c_per_gpr, "c_per_gpr");
        f(self.c_mult, "c_mult");
        f(self.k_per_gpr, "k_per_gpr");
        f(self.k_mult, "k_mult");
        f(self.n_per_gpr, "n_per_gpr");
        f(self.n_part_cnt, "n_part_cnt");
        f(self.read_size, "read_size");
        f(self.short_store, "short_store");
        f(self.data_prefetch, "data_prefetch");
    }

    int GetChunkSize() const { return chunk_size; }
    int GetCPerGpr() const { return c_per_gpr; }
    int GetCMult() const { return c_mult; }
    int GetKPerGpr() const { return k_per_gpr; }
    int GetKMult() const { return k_mult; }
    int GetNPerGpr() const { return n_per_gpr; }
    int GetNPartCnt() const { return n_part_cnt; }
    // "hw" stands for "height-and-width".
    int GetHWPerGpr() const
    {
        assert(c_per_gpr);
        assert(n_per_gpr);
        assert(chunk_size);
        return wave_size / (c_per_gpr * n_per_gpr * chunk_size);
    }
    int GetReadSize() const { return read_size; }
    int GetShortStore() const { return short_store; }
    int GetDataPrefetch() const { return data_prefetch; }

    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConfigConvAsmBwdWrW1x1& other) const;
};

struct ConvAsmBwdWrW1x1 final : ConvTunableSolver<PerformanceConfigConvAsmBwdWrW1x1>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvAsmBwdWrW1x1>(); }

    PerformanceConfigConvAsmBwdWrW1x1 MIOPEN_INTERNALS_EXPORT GetDefaultPerformanceConfig(
        const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConfigConvAsmBwdWrW1x1&) const override;
    PerformanceConfigConvAsmBwdWrW1x1 Search(const ExecutionContext&,
                                             const miopen::conv::ProblemDescription&,
                                             const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvAsmBwdWrW1x1&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvOclBwdWrW53 final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvOclBwdWrW53>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT fft final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<fft>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct PerformanceImplicitGemmWrwV4R4Xdlops : PerfConfigBase<PerformanceImplicitGemmWrwV4R4Xdlops>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmK;
    bool use_spare_set;

    MIOPEN_INTERNALS_EXPORT
    PerformanceImplicitGemmWrwV4R4Xdlops(int, int, int, int, int, int, bool, bool, bool);
    PerformanceImplicitGemmWrwV4R4Xdlops();
    PerformanceImplicitGemmWrwV4R4Xdlops(bool spare);
    PerformanceImplicitGemmWrwV4R4Xdlops(int a, int b, int c, int d, int e, int f, bool g, bool h)
        : PerformanceImplicitGemmWrwV4R4Xdlops(a, b, c, d, e, f, g, h, false)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmK, "GemmBThreadCopyMoreGemmK");
    }

    bool operator==(const PerformanceImplicitGemmWrwV4R4Xdlops& other) const;

    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool IsReallyValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ExecutionContext&,
                                 const miopen::conv::ProblemDescription&) const;

    std::tuple<int, int, int, int, int, bool>
    CalculateGemmSizeAndGemmKBlock(const ExecutionContext&,
                                   const miopen::conv::ProblemDescription&) const;
    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ExecutionContext&,
                                            const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmWrwV4R4Xdlops final
    : ConvTunableSolver<PerformanceImplicitGemmWrwV4R4Xdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmWrwV4R4Xdlops>();
    }

    PerformanceImplicitGemmWrwV4R4Xdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceImplicitGemmWrwV4R4Xdlops&) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceImplicitGemmWrwV4R4Xdlops&) const override;
    PerformanceImplicitGemmWrwV4R4Xdlops Search(const ExecutionContext&,
                                                const miopen::conv::ProblemDescription&,
                                                const AnyInvokeParams& invoke_ctx) const override;
};

struct PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm
    : PerfConfigBase<PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm>
{
    int GemmMPerBlock;
    int GemmNPerBlock;
    int GemmKPerBlock;
    int GemmMPerWave;
    int GemmNPerWave;
    int GemmKPack;
    int GemmMFactor;
    int GemmNFactor;
    int GemmKTotalFactor;
    bool GemmAThreadCopyMoreGemmK;
    bool GemmBThreadCopyMoreGemmK;

    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm(
        int, int, int, int, int, int, int, int, int, bool, bool);
    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm();
    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm(bool)
        : PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm()
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.GemmMPerBlock, "GemmMPerBlock");
        f(self.GemmNPerBlock, "GemmNPerBlock");
        f(self.GemmKPerBlock, "GemmKPerBlock");
        f(self.GemmMPerWave, "GemmMPerWave");
        f(self.GemmNPerWave, "GemmNPerWave");
        f(self.GemmKPack, "GemmKPack");
        f(self.GemmMFactor, "GemmMFactor");
        f(self.GemmNFactor, "GemmNFactor");
        f(self.GemmKTotalFactor, "GemmKTotalFactor");
        f(self.GemmAThreadCopyMoreGemmK, "GemmAThreadCopyMoreGemmK");
        f(self.GemmBThreadCopyMoreGemmK, "GemmBThreadCopyMoreGemmK");
    }

    bool operator==(const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm& other) const;

    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool IsReallyValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const;
    bool IsFastToBeUsedForTuning(const ExecutionContext&,
                                 const miopen::conv::ProblemDescription&) const;

    std::tuple<int, int, int, int, int, int, int, int, bool>
    CalculateGemmSizeAndGemmKBlock(const ExecutionContext&,
                                   const miopen::conv::ProblemDescription&) const;
    std::tuple<int, bool> CalculateBlockSize() const;
    std::tuple<int, bool> CalculateGridSize(const ExecutionContext&,
                                            const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmABlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<int, int, int, int, int, bool>
    CalculateGemmBBlockCopyPerformanceParameters(const miopen::conv::ProblemDescription&) const;
    std::tuple<std::size_t, bool>
    CalculateLdsNumberOfByte(const miopen::conv::ProblemDescription&) const;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm final
    : ConvTunableSolver<PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm>();
    }

    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm&) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm&) const override;
    PerformanceImplicitGemmWrwV4R4Xdlops_Padded_Gemm
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
};

struct PerformanceConvCkIgemmFwdV6r1DlopsNchw
    : PerfConfigBase<PerformanceConvCkIgemmFwdV6r1DlopsNchw>
{
    int ck_tunable_list_id;

    PerformanceConvCkIgemmFwdV6r1DlopsNchw(int a) : ck_tunable_list_id(a) {}

    PerformanceConvCkIgemmFwdV6r1DlopsNchw() : PerformanceConvCkIgemmFwdV6r1DlopsNchw(-1) {}

    PerformanceConvCkIgemmFwdV6r1DlopsNchw(bool) : PerformanceConvCkIgemmFwdV6r1DlopsNchw(0) {}

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.ck_tunable_list_id, "ck_tunable_list_id");
    }

    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConvCkIgemmFwdV6r1DlopsNchw& config) const
    {
        return ck_tunable_list_id == config.ck_tunable_list_id;
    }
};

struct ConvCkIgemmFwdV6r1DlopsNchw final : ConvTunableSolver<PerformanceConvCkIgemmFwdV6r1DlopsNchw>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvCkIgemmFwdV6r1DlopsNchw>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    PerformanceConvCkIgemmFwdV6r1DlopsNchw
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConvCkIgemmFwdV6r1DlopsNchw&) const override;
    PerformanceConvCkIgemmFwdV6r1DlopsNchw Search(const ExecutionContext&,
                                                  const miopen::conv::ProblemDescription&,
                                                  const AnyInvokeParams& invoke_ctx) const override;
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConvCkIgemmFwdV6r1DlopsNchw&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipConv final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvHipConv>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.5f;
    }
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvDirectNaiveConvFwd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvDirectNaiveConvFwd>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled.
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.01f;
    }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvDirectNaiveConvBwd final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvDirectNaiveConvBwd>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled.
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.01f;
    }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT ConvDirectNaiveConvWrw final : ConvSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvDirectNaiveConvWrw>();
    }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled.
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.01f;
    }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT GemmFwdBase : ConvSolver
{
    bool IsDynamic() const override { return true; }
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

private:
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    friend struct GemmFwd1x1_0_2;
    friend struct GemmFwd1x1_0_1_int8;
    friend struct GemmFwd1x1_0_1;
    friend struct GemmFwdRest;
};

struct MIOPEN_INTERNALS_EXPORT GemmFwd1x1_0_2 final : GemmFwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwd1x1_0_2>(); }

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    bool IsSlow(const ExecutionContext& context,
                const miopen::conv::ProblemDescription& problem) const override;

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;

    friend struct GemmFwdRest;
};

struct MIOPEN_INTERNALS_EXPORT GemmFwd1x1_0_1_int8 final : GemmFwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwd1x1_0_1_int8>(); }

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;

    friend struct GemmFwdRest;
};

struct MIOPEN_INTERNALS_EXPORT GemmFwd1x1_0_1 final : GemmFwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwd1x1_0_1>(); }

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    bool IsSlow(const ExecutionContext& context,
                const miopen::conv::ProblemDescription& problem) const override;

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;

    friend struct GemmFwdRest;
};

struct MIOPEN_INTERNALS_EXPORT GemmFwdRest final : GemmFwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmFwdRest>(); }

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    bool IsSlow(const ExecutionContext& context,
                const miopen::conv::ProblemDescription& problem) const override;

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT GemmBwdBase : ConvSolver
{
    bool IsDynamic() const override { return true; }
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

private:
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    friend struct GemmBwd1x1_stride2;
    friend struct GemmBwd1x1_stride1;
    friend struct GemmBwdRest;
};

struct MIOPEN_INTERNALS_EXPORT GemmBwd1x1_stride2 final : GemmBwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmBwd1x1_stride2>(); }

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    bool IsSlow(const ExecutionContext& context,
                const miopen::conv::ProblemDescription& problem) const override;

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;

    friend struct GemmBwdRest;
};

struct MIOPEN_INTERNALS_EXPORT GemmBwd1x1_stride1 final : GemmBwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmBwd1x1_stride1>(); }

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    bool IsSlow(const ExecutionContext& context,
                const miopen::conv::ProblemDescription& problem) const override;

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription& problem) const override;

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription& problem) const override;

    friend struct GemmBwdRest;
};

struct MIOPEN_INTERNALS_EXPORT GemmBwdRest final : GemmBwdBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmBwdRest>(); }

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    bool IsSlow(const ExecutionContext& context,
                const miopen::conv::ProblemDescription& problem) const override;

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT GemmWrwBase : ConvSolver
{
    bool IsDynamic() const override { return true; }
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

private:
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    friend struct GemmWrw1x1_stride1;
    friend struct GemmWrwUniversal;
};

struct MIOPEN_INTERNALS_EXPORT GemmWrw1x1_stride1 final : GemmWrwBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmWrw1x1_stride1>(); }

    bool IsSlow(const ExecutionContext& context,
                const miopen::conv::ProblemDescription& problem) const override;

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;

    friend struct GemmWrwUniversal;
};

struct MIOPEN_INTERNALS_EXPORT GemmWrwUniversal final : GemmWrwBase
{
    const std::string& SolverDbId() const override { return GetSolverDbId<GemmWrwUniversal>(); }

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;

    bool MayNeedWorkspace() const override { return true; }

    bool IsSlow(const ExecutionContext& context,
                const miopen::conv::ProblemDescription& problem) const override;

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

struct MIOPEN_INTERNALS_EXPORT PerformanceConfigAsmImplicitGemmGTC
    : PerfConfigBase<PerformanceConfigAsmImplicitGemmGTC>
{
    std::string direction;
    std::string tensor_layout;
    std::string precision;
    int nxb;
    int nxe;

    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block;

    int wave_tile_m;
    int wave_tile_n;
    int wave_tile_k;
    int wave_step_m;
    int wave_step_n;
    int wave_repeat_m;
    int wave_repeat_n;

    int multihead;
    int vector_store;
    int gemm_k_global_split;
    int merge_e;
    int tensor_a_pass_through;

    std::vector<int> tensor_a_thread_lengths;
    std::vector<int> tensor_a_cluster_lengths;
    std::vector<int> tensor_b_thread_lengths;
    std::vector<int> tensor_b_cluster_lengths;

    bool use_spare_set;
    int index;

    PerformanceConfigAsmImplicitGemmGTC(std::string dir,
                                        std::string layout,
                                        std::string prec,
                                        int b,
                                        int e,
                                        int mpb,
                                        int npb,
                                        int kpb,
                                        int wtm,
                                        int wtn,
                                        int wtk,
                                        int wsm,
                                        int wsn,
                                        int wrm,
                                        int wrn,
                                        int mh,
                                        int vs,
                                        int gks,
                                        int me,
                                        int pta,
                                        std::initializer_list<int> ta_t,
                                        std::initializer_list<int> ta_c,
                                        std::initializer_list<int> tb_t,
                                        std::initializer_list<int> tb_c,
                                        bool spare = false);
    PerformanceConfigAsmImplicitGemmGTC(std::string dir,
                                        std::string layout,
                                        miopenDataType_t prec,
                                        int b,
                                        int e,
                                        int mpb,
                                        int npb,
                                        int kpb,
                                        int wtm,
                                        int wtn,
                                        int wtk,
                                        int wsm,
                                        int wsn,
                                        int wrm,
                                        int wrn,
                                        int mh,
                                        int vs,
                                        int gks,
                                        int me,
                                        int pta,
                                        std::initializer_list<int> ta_t,
                                        std::initializer_list<int> ta_c,
                                        std::initializer_list<int> tb_t,
                                        std::initializer_list<int> tb_c,
                                        bool spare = false);
    PerformanceConfigAsmImplicitGemmGTC()
        : PerformanceConfigAsmImplicitGemmGTC("fwd",
                                              "nchw",
                                              "fp32",
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTC("fwd",
                                              "nchw",
                                              "fp32",
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              {1, 1, 1, 1},
                                              spare)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.direction, "dir");
        f(self.tensor_layout, "lyt");
        f(self.precision, "pre");
        f(self.nxb, "nxb");
        f(self.nxe, "nxe");
        f(self.gemm_m_per_block, "mpb");
        f(self.gemm_n_per_block, "npb");
        f(self.gemm_k_per_block, "kpb");

        f(self.wave_tile_m, "wtm");
        f(self.wave_tile_n, "wtn");
        f(self.wave_tile_k, "wtk");
        f(self.wave_step_m, "wsm");
        f(self.wave_step_n, "wsn");
        f(self.wave_repeat_m, "wrm");
        f(self.wave_repeat_n, "wrn");

        f(self.multihead, "mh");
        f(self.vector_store, "vs");
        f(self.gemm_k_global_split, "gks");
        f(self.merge_e, "me");
        f(self.tensor_a_pass_through, "pta");

        f(self.tensor_a_thread_lengths[0], "ta0");
        f(self.tensor_a_thread_lengths[1], "ta1");
        f(self.tensor_a_thread_lengths[2], "ta2");
        f(self.tensor_a_thread_lengths[3], "ta3");

        f(self.tensor_a_cluster_lengths[0], "ca0");
        f(self.tensor_a_cluster_lengths[1], "ca1");
        f(self.tensor_a_cluster_lengths[2], "ca2");
        f(self.tensor_a_cluster_lengths[3], "ca3");

        f(self.tensor_b_thread_lengths[0], "tb0");
        f(self.tensor_b_thread_lengths[1], "tb1");
        f(self.tensor_b_thread_lengths[2], "tb2");
        f(self.tensor_b_thread_lengths[3], "tb3");

        f(self.tensor_b_cluster_lengths[0], "cb0");
        f(self.tensor_b_cluster_lengths[1], "cb1");
        f(self.tensor_b_cluster_lengths[2], "cb2");
        f(self.tensor_b_cluster_lengths[3], "cb3");
        f(self.index, "index");
    }

    // Chilrden must provide support for ComputedContainer.
    void HeuristicInit(const ExecutionContext&)                                          = delete;
    bool SetNextValue(const miopen::conv::ProblemDescription&)                           = delete;
    bool IsValidValue() const                                                            = delete;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const = delete;

    bool IsDefaultConstructed() const;
    bool operator==(const PerformanceConfigAsmImplicitGemmGTC& other) const;
    void CopyParameters(const PerformanceConfigAsmImplicitGemmGTC& other);
    std::string ToString() const override;
    std::string ToKernelName(const ExecutionContext&) const;
    int BlockSize() const;
};

struct PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC : PerformanceConfigAsmImplicitGemmGTC
{
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     std::string prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     miopenDataType_t prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC()
        : PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           spare)
    {
    }

    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription& config);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC>();
    }

    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC&) const override;
    PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigAsmImplicitGemmGTCFwdXdlopsNHWC&) const override;
};

struct PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC : PerformanceConfigAsmImplicitGemmGTC
{
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     std::string prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     miopenDataType_t prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC()
        : PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           spare)
    {
    }
    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
};

struct MIOPEN_INTERNALS_EXPORT ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC>();
    }

    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC&) const override;
    PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigAsmImplicitGemmGTCBwdXdlopsNHWC&) const override;
};

struct PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC : PerformanceConfigAsmImplicitGemmGTC
{
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     std::string prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC(std::string dir,
                                                     std::string layout,
                                                     miopenDataType_t prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int wtm,
                                                     int wtn,
                                                     int wtk,
                                                     int wsm,
                                                     int wsn,
                                                     int wrm,
                                                     int wrn,
                                                     int mh,
                                                     int vs,
                                                     int gks,
                                                     int me,
                                                     int pta,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTC(dir,
                                              layout,
                                              prec,
                                              b,
                                              e,
                                              mpb,
                                              npb,
                                              kpb,
                                              wtm,
                                              wtn,
                                              wtk,
                                              wsm,
                                              wsn,
                                              wrm,
                                              wrn,
                                              mh,
                                              vs,
                                              gks,
                                              me,
                                              pta,
                                              ta_t,
                                              ta_c,
                                              tb_t,
                                              tb_c,
                                              spare)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC()
        : PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC("fwd",
                                                           "nchw",
                                                           "fp32",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           spare)
    {
    }

    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
    size_t ComputeKernelOccupancy() const;

private:
    void SetParamsForKSplit(const miopen::conv::ProblemDescription& problem,
                            const size_t& occupancy);
};

struct MIOPEN_INTERNALS_EXPORT ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC>();
    }

    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC&) const override;
    PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigAsmImplicitGemmGTCWrwXdlopsNHWC&) const override;
};

struct PerformanceConfigAsmImplicitGemmGTCvector
    : PerfConfigBase<PerformanceConfigAsmImplicitGemmGTCvector>
{
    std::string direction;
    std::string tensor_layout;
    std::string precision;
    int nxb;
    int nxe;

    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block;

    int lanegroup_tile_m;
    int lanegroup_tile_n;
    int lanegroup_wave_m;
    int lanegroup_wave_n;
    int lanegroup_repeat_m;
    int lanegroup_repeat_n;

    int vector_c;

    std::vector<int> tensor_a_thread_lengths;
    std::vector<int> tensor_a_cluster_lengths;
    std::vector<int> tensor_b_thread_lengths;
    std::vector<int> tensor_b_cluster_lengths;

    bool use_spare_set;
    int index;

    MIOPEN_INTERNALS_EXPORT
    PerformanceConfigAsmImplicitGemmGTCvector(std::string dir,
                                              std::string layout,
                                              std::string prec,
                                              int b,
                                              int e,
                                              int mpb,
                                              int npb,
                                              int kpb,
                                              int lgtm,
                                              int lgtn,
                                              int lgpwm,
                                              int lgpwn,
                                              int lgrm,
                                              int lgrn,
                                              int vec_c,
                                              std::initializer_list<int> ta_t,
                                              std::initializer_list<int> ta_c,
                                              std::initializer_list<int> tb_t,
                                              std::initializer_list<int> tb_c,
                                              bool spare = false);

    MIOPEN_INTERNALS_EXPORT
    PerformanceConfigAsmImplicitGemmGTCvector(std::string dir,
                                              std::string layout,
                                              miopenDataType_t prec,
                                              int b,
                                              int e,
                                              int mpb,
                                              int npb,
                                              int kpb,
                                              int lgtm,
                                              int lgtn,
                                              int lgpwm,
                                              int lgpwn,
                                              int lgrm,
                                              int lgrn,
                                              int vec_c,
                                              std::initializer_list<int> ta_t,
                                              std::initializer_list<int> ta_c,
                                              std::initializer_list<int> tb_t,
                                              std::initializer_list<int> tb_c,
                                              bool spare = false);

    PerformanceConfigAsmImplicitGemmGTCvector()
        : PerformanceConfigAsmImplicitGemmGTCvector("fwd",
                                                    "nchwc_kcyxc",
                                                    "Half",
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCvector(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCvector("fwd",
                                                    "nchwc_kcyxc",
                                                    "Half",
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    {1, 1, 1, 1},
                                                    spare)
    {
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.direction, "dir");
        f(self.tensor_layout, "lyt");
        f(self.precision, "pre");
        f(self.nxb, "nxb");
        f(self.nxe, "nxe");
        f(self.gemm_m_per_block, "mpb");
        f(self.gemm_n_per_block, "npb");
        f(self.gemm_k_per_block, "kpb");

        f(self.lanegroup_tile_m, "lgtm");
        f(self.lanegroup_tile_n, "lgtn");
        f(self.lanegroup_wave_m, "lgpwm");
        f(self.lanegroup_wave_n, "lgpwn");
        f(self.lanegroup_repeat_m, "lgrm");
        f(self.lanegroup_repeat_n, "lgrn");

        f(self.vector_c, "vec_c");

        f(self.tensor_a_thread_lengths[0], "ta0");
        f(self.tensor_a_thread_lengths[1], "ta1");
        f(self.tensor_a_thread_lengths[2], "ta2");
        f(self.tensor_a_thread_lengths[3], "ta3");

        f(self.tensor_a_cluster_lengths[0], "ca0");
        f(self.tensor_a_cluster_lengths[1], "ca1");
        f(self.tensor_a_cluster_lengths[2], "ca2");
        f(self.tensor_a_cluster_lengths[3], "ca3");

        f(self.tensor_b_thread_lengths[0], "tb0");
        f(self.tensor_b_thread_lengths[1], "tb1");
        f(self.tensor_b_thread_lengths[2], "tb2");
        f(self.tensor_b_thread_lengths[3], "tb3");

        f(self.tensor_b_cluster_lengths[0], "cb0");
        f(self.tensor_b_cluster_lengths[1], "cb1");
        f(self.tensor_b_cluster_lengths[2], "cb2");
        f(self.tensor_b_cluster_lengths[3], "cb3");
        f(self.index, "index");
    }

    // Chilrden must provide support for ComputedContainer.
    void HeuristicInit(const ExecutionContext&)                                          = delete;
    bool SetNextValue(const miopen::conv::ProblemDescription&)                           = delete;
    bool IsValidValue() const                                                            = delete;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription&) const = delete;

    bool IsDefaultConstructed() const;
    bool operator==(const PerformanceConfigAsmImplicitGemmGTCvector& other) const;
    void CopyParameters(const PerformanceConfigAsmImplicitGemmGTCvector& other);
    std::string ToString() const override;
    std::string ToKernelName(const ExecutionContext&) const;
    int BlockSize() const;
};
struct PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC : PerformanceConfigAsmImplicitGemmGTCvector
{

    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC(std::string dir,
                                                     std::string layout,
                                                     std::string prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int lgtm,
                                                     int lgtn,
                                                     int lgpwm,
                                                     int lgpwn,
                                                     int lgrm,
                                                     int lgrn,
                                                     int vec_c,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTCvector(dir,
                                                    layout,
                                                    prec,
                                                    b,
                                                    e,
                                                    mpb,
                                                    npb,
                                                    kpb,
                                                    lgtm,
                                                    lgtn,
                                                    lgpwm,
                                                    lgpwn,
                                                    lgrm,
                                                    lgrn,
                                                    vec_c,
                                                    ta_t,
                                                    ta_c,
                                                    tb_t,
                                                    tb_c,
                                                    spare)
    {
    }

    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC(std::string dir,
                                                     std::string layout,
                                                     miopenDataType_t prec,
                                                     int b,
                                                     int e,
                                                     int mpb,
                                                     int npb,
                                                     int kpb,
                                                     int lgtm,
                                                     int lgtn,
                                                     int lgpwm,
                                                     int lgpwn,
                                                     int lgrm,
                                                     int lgrn,
                                                     int vec_c,
                                                     std::initializer_list<int> ta_t,
                                                     std::initializer_list<int> ta_c,
                                                     std::initializer_list<int> tb_t,
                                                     std::initializer_list<int> tb_c,
                                                     bool spare = false)
        : PerformanceConfigAsmImplicitGemmGTCvector(dir,
                                                    layout,
                                                    prec,
                                                    b,
                                                    e,
                                                    mpb,
                                                    npb,
                                                    kpb,
                                                    lgtm,
                                                    lgtn,
                                                    lgpwm,
                                                    lgpwn,
                                                    lgrm,
                                                    lgrn,
                                                    vec_c,
                                                    ta_t,
                                                    ta_c,
                                                    tb_t,
                                                    tb_c,
                                                    spare)
    {
    }

    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC()
        : PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC("fwd",
                                                           "nchwc_kcyxc",
                                                           "Half",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           false)
    {
    }
    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC(bool spare)
        : PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC("fwd",
                                                           "nchwc_kcyxc",
                                                           "Half",
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           1,
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           {1, 1, 1, 1},
                                                           spare)
    {
    }

    void HeuristicInit(const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
};

struct ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC final
    : ConvTunableSolver<PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC>();
    }
    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC&) const override;
    PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigAsmImplicitGemmGTCFwdDlopsNCHWC&) const override;
};

struct PerformanceConfigHipImplicitGemmGroupFwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemmGroupFwdXdlops>
{
    int index             = 0;
    int split_k           = 0; // not used for FWD, but required for AI heuristics interface
    std::string kernel_id = "";
    std::vector<std::string> valid_kernels;

    PerformanceConfigHipImplicitGemmGroupFwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }

    PerformanceConfigHipImplicitGemmGroupFwdXdlops() = default;

    explicit PerformanceConfigHipImplicitGemmGroupFwdXdlops(bool)
        : PerformanceConfigHipImplicitGemmGroupFwdXdlops(0, "")
    {
    }

    void DefaultKernelFromList(const ExecutionContext& ctx);
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConfigHipImplicitGemmGroupFwdXdlops& other) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsModelApplicable(const ExecutionContext& ctx,
                      const miopen::conv::ProblemDescription& problem) const;
    bool UseTF32() const { return use_tf32; }

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    // OLD KTN functions (for gfx90a) - Public for use by RunKTNGeneric template
    template <typename DataType>
    bool RunParameterPredictionModelKTN(const ExecutionContext& ctx,
                                        const miopen::conv::ProblemDescription& problem);
#endif

private:
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    std::vector<int> heuristic_indexes;
    std::unordered_map<int, std::vector<std::string>> heuristic_kernels;

    void InitHeuristicKernelIDsKTN(const std::string& type);
    bool ModelApplyTokenKTN(int idx, std::string value, const std::string& arch);
#endif

    template <typename DataType, typename ComputeType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    void InitValidKernels(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
    mutable bool use_tf32 = false;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmGroupFwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmGroupFwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmGroupFwdXdlops>();
    }

    PerformanceConfigHipImplicitGemmGroupFwdXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmGroupFwdXdlops&) const override;
    PerformanceConfigHipImplicitGemmGroupFwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmGroupFwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
};

struct PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemm3DGroupFwdXdlops>
{
    int index             = 0;
    int split_k           = 0; // not used for this solver, but required for AI heuristics interface
    std::string kernel_id = "";
    std::vector<std::string> valid_kernels;

    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }

    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops() = default;

    explicit PerformanceConfigHipImplicitGemm3DGroupFwdXdlops(bool)
        : PerformanceConfigHipImplicitGemm3DGroupFwdXdlops(0, "")
    {
    }
    void DefaultKernelFromList(const ExecutionContext& ctx);
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops& other) const;
    bool UseTF32() const { return use_tf32; }

private:
    void InitValidKernels(const miopen::conv::ProblemDescription& problem);
    mutable bool use_tf32 = false;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemm3DGroupFwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemm3DGroupFwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemm3DGroupFwdXdlops>();
    }

    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops&) const override;
    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops&) const override;
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override;

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
};

struct PerformanceConfigHipImplicitGemm3DGroupWrwXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemm3DGroupWrwXdlops>
{
    int index;
    int split_k;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemm3DGroupWrwXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemm3DGroupWrwXdlops()
        : PerformanceConfigHipImplicitGemm3DGroupWrwXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemm3DGroupWrwXdlops(bool)
        : PerformanceConfigHipImplicitGemm3DGroupWrwXdlops(0, "")
    {
    }
    void DefaultKernelFromList(const ExecutionContext& ctx);
    void HeuristicInit(const ExecutionContext&, const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops& other) const;
    bool UseTF32() const { return use_tf32; }

private:
    void InitValidKernels(const miopen::conv::ProblemDescription& problem);
    mutable bool use_tf32 = false;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemm3DGroupWrwXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemm3DGroupWrwXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemm3DGroupWrwXdlops>();
    }

    PerformanceConfigHipImplicitGemm3DGroupWrwXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops&) const override;
    PerformanceConfigHipImplicitGemm3DGroupWrwXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

private:
    size_t GetCKMaxWorkspaceSize(const miopen::conv::ProblemDescription& problem) const;
};

struct PerformanceConfigHipImplicitGemm3DGroupBwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemm3DGroupBwdXdlops>
{
    int index;
    int split_k = 0; // not used for this solver, but required for AI heuristics interface
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemm3DGroupBwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemm3DGroupBwdXdlops()
        : PerformanceConfigHipImplicitGemm3DGroupBwdXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemm3DGroupBwdXdlops(bool)
        : PerformanceConfigHipImplicitGemm3DGroupBwdXdlops(0, "")
    {
    }
    void DefaultKernelFromList(const ExecutionContext& ctx);
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops& other) const;
    bool UseTF32() const { return use_tf32; }

private:
    void InitValidKernels(const miopen::conv::ProblemDescription& problem);
    mutable bool use_tf32 = false;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemm3DGroupBwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemm3DGroupBwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemm3DGroupBwdXdlops>();
    }

    PerformanceConfigHipImplicitGemm3DGroupBwdXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(
        const ExecutionContext&,
        const miopen::conv::ProblemDescription&,
        const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops&) const override;
    PerformanceConfigHipImplicitGemm3DGroupBwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution
    GetSolution(const ExecutionContext&,
                const miopen::conv::ProblemDescription&,
                const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }
};

struct PerformanceConfigHipImplicitGemmGroupBwdXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemmGroupBwdXdlops>
{
    int index   = 0;
    int split_k = 1;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemmGroupBwdXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemmGroupBwdXdlops()
        : PerformanceConfigHipImplicitGemmGroupBwdXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemmGroupBwdXdlops(bool)
        : PerformanceConfigHipImplicitGemmGroupBwdXdlops(0, "")
    {
    }

    void DefaultKernelFromList(const ExecutionContext& ctx);
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConfigHipImplicitGemmGroupBwdXdlops& other) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsModelApplicable(const ExecutionContext& ctx,
                      const miopen::conv::ProblemDescription& problem) const;
    bool UseTF32() const { return use_tf32; }

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    // OLD KTN functions (for gfx90a) - Public for use by RunKTNGeneric template
    template <typename DataType>
    bool RunParameterPredictionModelKTN(const ExecutionContext& ctx,
                                        const miopen::conv::ProblemDescription& problem);
#endif

private:
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    std::vector<int> heuristic_indexes;
    std::unordered_map<int, std::vector<std::string>> heuristic_kernels;

    void InitHeuristicKernelIDsKTN();
    bool ModelApplyTokenKTN(int idx,
                            std::string value,
                            const std::string& arch,
                            const miopen::conv::ProblemDescription& problem);
#endif

    template <typename DataType, typename ComputeType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    void InitValidKernels(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
    mutable bool use_tf32 = false;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmGroupBwdXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmGroupBwdXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmGroupBwdXdlops>();
    }

    PerformanceConfigHipImplicitGemmGroupBwdXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmGroupBwdXdlops&) const override;
    PerformanceConfigHipImplicitGemmGroupBwdXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmGroupBwdXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

private:
    size_t GetCKMaxWorkspaceSize(const miopen::conv::ProblemDescription& problem) const;
};

struct PerformanceConfigHipImplicitGemmGroupWrwXdlops
    : PerfConfigBaseCK<PerformanceConfigHipImplicitGemmGroupWrwXdlops>
{
    int index   = 0;
    int split_k = 1;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigHipImplicitGemmGroupWrwXdlops(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigHipImplicitGemmGroupWrwXdlops()
        : PerformanceConfigHipImplicitGemmGroupWrwXdlops(0, "")
    {
    }
    PerformanceConfigHipImplicitGemmGroupWrwXdlops(bool)
        : PerformanceConfigHipImplicitGemmGroupWrwXdlops(0, "")
    {
    }

    void DefaultKernelFromList(const ExecutionContext& ctx);
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    bool SetNextValue(const miopen::conv::ProblemDescription&);
    bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    bool IsValid(const miopen::conv::ProblemDescription&) const;
    bool operator==(const PerformanceConfigHipImplicitGemmGroupWrwXdlops& other) const;
    MIOPEN_INTERNALS_EXPORT bool
    IsModelApplicable(const ExecutionContext& ctx,
                      const miopen::conv::ProblemDescription& problem) const;
    bool UseTF32() const { return use_tf32; }

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    // OLD KTN functions (for gfx90a) - Public for use by RunKTNGeneric template
    template <typename DataType>
    bool RunParameterPredictionModelKTN(const ExecutionContext& ctx,
                                        const miopen::conv::ProblemDescription& problem);
#endif

private:
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    std::vector<int> heuristic_indexes;
    std::unordered_map<int, std::vector<std::string>> heuristic_kernels;

    void InitHeuristicKernelIDsKTN(const std::string& type);
    bool ModelApplyTokenKTN(int idx,
                            std::string value,
                            const std::string& arch,
                            const miopen::conv::ProblemDescription& problem);
#endif

    template <typename DataType, typename ComputeType>
    void Init(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    void Init(const miopen::conv::ProblemDescription&);
    void InitValidKernels(const miopen::conv::ProblemDescription&);
    template <typename DataType>
    bool CheckIsSupportCKArgs(const miopen::conv::ProblemDescription&) const;
    mutable bool use_tf32 = false;
};

struct MIOPEN_INTERNALS_EXPORT ConvHipImplicitGemmGroupWrwXdlops final
    : ConvTunableSolver<PerformanceConfigHipImplicitGemmGroupWrwXdlops>
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<ConvHipImplicitGemmGroupWrwXdlops>();
    }

    PerformanceConfigHipImplicitGemmGroupWrwXdlops
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool
    IsValidPerformanceConfig(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmGroupWrwXdlops&) const override;
    PerformanceConfigHipImplicitGemmGroupWrwXdlops
    Search(const ExecutionContext&,
           const miopen::conv::ProblemDescription&,
           const AnyInvokeParams& invoke_ctx) const override;
    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigHipImplicitGemmGroupWrwXdlops&) const override;
    /// \ref igemm_get_wti_magic_number
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    };

    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override;
    bool MayNeedWorkspace() const override { return true; }

private:
    size_t GetCKMaxWorkspaceSize(const miopen::conv::ProblemDescription& problem) const;
};

struct PerformanceConfigConvDepthwiseFwd2D : PerfConfigBaseCK<PerformanceConfigConvDepthwiseFwd2D>
{
    int index;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;
    PerformanceConfigConvDepthwiseFwd2D(int idx, std::string kernl_id)
        : index(idx), kernel_id(kernl_id)
    {
    }
    PerformanceConfigConvDepthwiseFwd2D() : PerformanceConfigConvDepthwiseFwd2D(0, "") {}
    PerformanceConfigConvDepthwiseFwd2D(bool) : PerformanceConfigConvDepthwiseFwd2D(0, "") {}
    MIOPEN_INTERNALS_EXPORT void HeuristicInit(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool SetNextValue(const miopen::conv::ProblemDescription&);
    MIOPEN_INTERNALS_EXPORT bool IsValidValue() const;
    bool IsValid(const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
    {
        return IsValid(problem);
    }
    MIOPEN_INTERNALS_EXPORT bool IsValid(const miopen::conv::ProblemDescription&) const;
    MIOPEN_INTERNALS_EXPORT bool operator==(const PerformanceConfigConvDepthwiseFwd2D& other) const;

private:
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    std::vector<int> heuristic_indexes;
    std::unordered_map<int, std::vector<std::string>> heuristic_kernels;
#endif
};

struct MIOPEN_INTERNALS_EXPORT ConvDepthwiseFwd2D final
    : ConvTunableSolver<PerformanceConfigConvDepthwiseFwd2D>
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvDepthwiseFwd2D>(); }

    PerformanceConfigConvDepthwiseFwd2D
    GetDefaultPerformanceConfig(const ExecutionContext&,
                                const miopen::conv::ProblemDescription&) const override;
    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const PerformanceConfigConvDepthwiseFwd2D&) const override;
    PerformanceConfigConvDepthwiseFwd2D Search(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&,
                                               const AnyInvokeParams& invoke_ctx) const override;

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return true; }
    /// Use very small fixed value enough to backup GEMM for cases when
    /// GEMM is disabled.
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    }
    bool MayNeedWorkspace() const override { return false; }
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override
    {
        return 0;
    }

    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&,
                             const PerformanceConfigConvDepthwiseFwd2D&) const override;
};
/// Common base for ConvWinogradNHWC transposing solvers (tunable and non-tunable).
/// Provides ConvertFromApiParams, ConvertForInnerSolver, Transpose, and GetTransposes.
/// Template params:
///   Derived    - CRTP derived class
///   Inner      - the inner (NCHW) winograd solver
///   SolverBase - ConvSolver or ConvTunableSolver<PerformanceConfigType>
template <class Derived, class Inner, class SolverBase>
struct ConvWinogradNHWCTransposingBase : TransposingSolver<Derived,
                                                           SolverBase,
                                                           miopen::conv::ProblemDescription,
                                                           miopen::conv::TransposeConvInvokeParams,
                                                           Inner>
{
    using Problem      = miopen::conv::ProblemDescription;
    using InvokeParams = miopen::conv::TransposeConvInvokeParams;
    using Base         = TransposingSolver<Derived, SolverBase, Problem, InvokeParams, Inner>;

    /// Convert from API params to TransposeConvInvokeParams.
    /// The API passes DataInvokeParams for Fwd/Bwd and WrWInvokeParams for WrW.
    static InvokeParams ConvertFromApiParams(const AnyInvokeParams& any_params)
    {
        if(any_params.IsOfType<miopen::conv::WrWInvokeParams>())
        {
            const auto& wrw_params = any_params.CastTo<miopen::conv::WrWInvokeParams>();
            return InvokeParams{wrw_params};
        }
        const auto& data_params = any_params.CastTo<miopen::conv::DataInvokeParams>();
        return InvokeParams{data_params};
    }

    /// Convert TransposeConvInvokeParams back to the correct type for inner solver.
    /// Inner Winograd solvers expect DataInvokeParams for Fwd/Bwd and WrWInvokeParams for WrW.
    static AnyInvokeParams ConvertForInnerSolver(const InvokeParams& params)
    {
        if(params.is_wrw)
            return params.ToWrWInvokeParams();
        return params.ToDataInvokeParams();
    }

    /// Override Transpose to recompute layout strings after transposing tensors.
    /// This is needed because conv::ProblemDescription caches layout strings at construction,
    /// and they must be updated to reflect the new NCHW-like strides after transposition.
    inline static Problem Transpose(const Problem& problem)
    {
        auto transposed_problem = Base::Transpose(problem);
        // CRITICAL: Attempt to update cached layout strings to match transposed strides.
        // Some inner solvers (e.g., RxSf2x3g1) check IsLayoutDefault() which validates
        // cached layout strings. Note: For degenerate dimensions (N=1, C=1, etc.), the cached
        // string may not be updated since strides satisfy multiple layouts. Solvers using
        // IsPossibleLayout4D5D() will work correctly regardless as they check actual strides.
        transposed_problem.HeuristicUpdateLayouts();
        return transposed_problem;
    }

    inline static auto GetTransposes(const Problem& problem)
    {
        const bool is_wrw = problem.IsDirectionBackwardWrW();

        // Layout string "NCDHW" supports both 4D (NCHW) and 5D (NCDHW) tensors:
        // - For 4D tensors: Automatically interpreted as NCHW (D dimension is implicit/1)
        // - For 5D tensors: Full NCDHW layout for 3D convolutions
        // This makes the transposing solver future-proof for both 2D and 3D convolutions.
        return std::array<ProblemTensorTransposeDescriptor<Problem, InvokeParams>, 3>{{
            {
                &Problem::GetIn,
                &InvokeParams::inDesc,
                &InvokeParams::in, // in (dy for WrW): always an input
                nullptr,
                "NCDHW", // transpose NHWC/NDHWC->NCHW/NCDHW
                true,
            },
            {
                &Problem::GetWeights,
                &InvokeParams::wDesc,
                is_wrw ? nullptr : &InvokeParams::w,           // Fwd/Bwd: w is input
                is_wrw ? &InvokeParams::w_as_output : nullptr, // WrW: dw is output
                "NCDHW", // weights: layout adapts to tensor dimensionality
                !is_wrw, // Fwd/Bwd: input; WrW: output
            },
            {
                &Problem::GetOut,
                &InvokeParams::outDesc,
                is_wrw ? &InvokeParams::out_as_input : nullptr, // WrW: x is input
                is_wrw ? nullptr : &InvokeParams::out,          // Fwd/Bwd: out is output
                "NCDHW", // out: layout adapts to tensor dimensionality
                is_wrw,  // Fwd/Bwd: output; WrW: input
            },
        }};
    }
};

/// Non-tunable transposing wrapper for NHWC winograd solvers.
template <class Inner>
struct ConvWinogradNHWCTransposingSolver
    : ConvWinogradNHWCTransposingBase<ConvWinogradNHWCTransposingSolver<Inner>, Inner, ConvSolver>
{
};

/// Tunable transposing wrapper for NHWC winograd solvers.
/// Uses ConvTunableSolver<Inner::PerformanceConfigType> as base so that the transposed
/// solver properly exposes tuning methods (GetDefaultPerformanceConfig, IsValidPerformanceConfig,
/// Search, GetSolution with config). The TransposingSolverGetSolution<..., true> specialization
/// handles delegation of all tunable methods to the inner solver with transposed problems.
template <class Inner>
struct ConvWinogradNHWCTransposingTunableSolver
    : ConvWinogradNHWCTransposingBase<ConvWinogradNHWCTransposingTunableSolver<Inner>,
                                      Inner,
                                      ConvTunableSolver<typename Inner::PerformanceConfigType>>
{
};

struct TransposedConvBinWinograd3x3U final : ConvWinogradNHWCTransposingSolver<ConvBinWinograd3x3U>
{
    const std::string& SolverDbId() const { return GetSolverDbId<TransposedConvBinWinograd3x3U>(); }
};

struct TransposedConvBinWinogradRxS final : ConvWinogradNHWCTransposingSolver<ConvBinWinogradRxS>
{
    const std::string& SolverDbId() const { return GetSolverDbId<TransposedConvBinWinogradRxS>(); }
};

struct TransposedConvBinWinogradRxSf2x3g1 final
    : ConvWinogradNHWCTransposingSolver<ConvBinWinogradRxSf2x3g1>
{
    const std::string& SolverDbId() const
    {
        return GetSolverDbId<TransposedConvBinWinogradRxSf2x3g1>();
    }
};
template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct TransposedConvMPBidirectWinograd final
    : ConvWinogradNHWCTransposingSolver<
          ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>
{
    const std::string& SolverDbId() const
    {
        return this->template GetSolverDbId<
            TransposedConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>();
    }
};

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct TransposedConvWinograd3x3MultipassWrW final
    : ConvWinogradNHWCTransposingSolver<
          ConvWinograd3x3MultipassWrW<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>
{
    const std::string& SolverDbId() const
    {
        return this->template GetSolverDbId<TransposedConvWinograd3x3MultipassWrW<WinoDataH,
                                                                                  WinoFilterH,
                                                                                  WinoDataW,
                                                                                  WinoFilterW>>();
    }
};

template <uint32_t Winodata, uint32_t Winofilter>
struct TransposedConvWinoFuryRxS final
    : ConvWinogradNHWCTransposingSolver<ConvWinoFuryRxS<Winodata, Winofilter>>
{
    const std::string& SolverDbId() const
    {
        return this->template GetSolverDbId<TransposedConvWinoFuryRxS<Winodata, Winofilter>>();
    }
};

template <uint32_t Winodata, uint32_t Winofilter>
struct TransposedConvWinoRageRxS final
    : ConvWinogradNHWCTransposingSolver<ConvWinoRageRxS<Winodata, Winofilter>>
{
    const std::string& SolverDbId() const
    {
        return this->template GetSolverDbId<TransposedConvWinoRageRxS<Winodata, Winofilter>>();
    }
};

// Tunable transposed Winograd solvers for NHWC layout support
template <int Winodata, int Winofilter>
struct TransposedConvBinWinoRxS final
    : ConvWinogradNHWCTransposingTunableSolver<ConvBinWinoRxS<Winodata, Winofilter>>
{
    const std::string& SolverDbId() const override
    {
        return this->template GetSolverDbId<TransposedConvBinWinoRxS<Winodata, Winofilter>>();
    }
};

template <int WinoDataH, int WinoFilterH, int WinoDataW = WinoDataH, int WinoFilterW = WinoFilterH>
struct TransposedConvMPBidirectWinograd_xdlops final
    : ConvWinogradNHWCTransposingTunableSolver<
          ConvMPBidirectWinograd_xdlops<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>>
{
    const std::string& SolverDbId() const
    {
        return this->template GetSolverDbId<TransposedConvMPBidirectWinograd_xdlops<WinoDataH,
                                                                                    WinoFilterH,
                                                                                    WinoDataW,
                                                                                    WinoFilterW>>();
    }
};

// To suppress misleading clang warnings
#if defined(__clang__) && defined(CONV_MP_BIDIRECTIONAL_WINOGRAD_CPP)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-template-vtables"
#endif
extern template struct TransposedConvMPBidirectWinograd_xdlops<2, 3>;
extern template struct TransposedConvMPBidirectWinograd_xdlops<3, 3>;
extern template struct TransposedConvMPBidirectWinograd_xdlops<4, 3>;
extern template struct TransposedConvMPBidirectWinograd_xdlops<5, 3>;
extern template struct TransposedConvMPBidirectWinograd_xdlops<6, 3>;
#if defined(__clang__) && defined(CONV_MP_BIDIRECTIONAL_WINOGRAD_CPP)
#pragma clang diagnostic pop
#endif

/// Placeholder forward 3D depthwise convolution (FP16/BF16, default/NCDHW layout); stub kernel.
struct MIOPEN_INTERNALS_EXPORT ConvDepthwiseFwd3D final : ConvSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<ConvDepthwiseFwd3D>(); }

    bool IsApplicable(const ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override;
    bool IsDynamic() const override { return false; }
    float GetWti(const ExecutionContext&, const miopen::conv::ProblemDescription&) const override
    {
        return 0.02f;
    }
    size_t GetWorkspaceSize(const ExecutionContext&,
                            const miopen::conv::ProblemDescription&) const override
    {
        return 0;
    }
    ConvSolution GetSolution(const ExecutionContext&,
                             const miopen::conv::ProblemDescription&) const override;
};

// Test helper functions for metadata validation
// These functions return all CK kernel TypeStrings without problem-based filtering
// Declared here but implemented in the respective solver .cpp files
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#endif

} // namespace conv
} // namespace solver
} // namespace miopen
