// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pick.hpp>
#include <miopen/conv/heuristics/lgbm_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_predict.hpp>
#include <miopen/conv/heuristics/lgbm_forest.hpp>
#include <miopen/conv/heuristics/lgbm_common.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp> // common::EngineeredConvFeatures, ConvDirection

#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>

// Force-disable unseen-architecture routing even when the model supports it,
// reverting to abstain-on-unknown-arch. Enabled by default when the loaded
// model was trained for it (LgbmMetadata::AllowUnseenArch()).
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_LGBM_DISABLE_OOD_ARCH)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace miopen {
namespace ai {
namespace lgbm {

namespace {

// Indices into the 61-feature row, matching model_meta.json rank.feature_order:
// 41 base features + 20 derived (13 tn_* GEMM-geometry + 7 al_* tile-alignment)
// at 28..47, then the categorical and GPU blocks. GPU inputs are the six
// hipDeviceProp_t fields + gfx_id; derived features come from conv dims +
// cu_count.
constexpr int kIdxNMiniBatchSize = 0;
constexpr int kIdxChannels       = 1;
constexpr int kIdxDepth          = 2;
constexpr int kIdxHeight         = 3;
constexpr int kIdxWidth          = 4;
constexpr int kIdxOutputChannels = 5;
constexpr int kIdxDepthOutput    = 6;
constexpr int kIdxSpatialDim     = 7;
constexpr int kIdxNumDimensions  = 8;
constexpr int kIdxFilterHeightY  = 9;
constexpr int kIdxFilterWidthX   = 10;
constexpr int kIdxFilterDepthZ   = 11;
constexpr int kIdxPadHeight      = 12;
constexpr int kIdxPadWidth       = 13;
constexpr int kIdxPadDepth       = 14;
constexpr int kIdxStrideHeight   = 15;
constexpr int kIdxStrideWidth    = 16;
constexpr int kIdxStrideDepth    = 17;
constexpr int kIdxDilationHeight = 18;
constexpr int kIdxDilationWidth  = 19;
constexpr int kIdxDilationDepth  = 20;
constexpr int kIdxGroups         = 21;
constexpr int kIdxFlopCnt        = 22;
constexpr int kIdxBytesRead      = 23;
constexpr int kIdxBytesWritten   = 24;
constexpr int kIdxBytesProcessed = 25;
constexpr int kIdxGflops         = 26;
constexpr int kIdxBandwidthGbps  = 27;
// Derived TunaNet GEMM-geometry features occupy indices [28..40] in
// EngineeredConvFeatures output order (tn_log_flops, tn_log_M/N/K,
// tn_M_over_N/M_over_K/N_over_K, tn_log_gemm_size, tn_log_work_per_cu,
// tn_spatial_reduction, tn_filter_coverage, tn_channel_ratio, tn_group_density).
// They are written as a contiguous block from this base index.
constexpr int kIdxTnBlockBegin = 28;
constexpr int kNumTnFeatures   = 13;
// Derived tile-alignment features (41..47).
constexpr int kIdxAlC64     = 41;
constexpr int kIdxAlC32     = 42;
constexpr int kIdxAlOc64    = 43;
constexpr int kIdxAlOc32    = 44;
constexpr int kIdxAlN8      = 45;
constexpr int kIdxAlCRem64  = 46;
constexpr int kIdxAlOcRem64 = 47;
// Categorical problem features (48..52).
constexpr int kIdxDataType  = 48;
constexpr int kIdxDirection = 49;
constexpr int kIdxInLayout  = 50;
constexpr int kIdxFilLayout = 51;
constexpr int kIdxOutLayout = 52;
// GPU numeric features (53..58), all hipDeviceProp_t-backed.
constexpr int kIdxCuCount               = 53;
constexpr int kIdxWaveSize              = 54;
constexpr int kIdxLdsSizePerWorkgroupKb = 55;
constexpr int kIdxL2CacheTotalKb        = 56;
constexpr int kIdxBoostClockMhz         = 57;
constexpr int kIdxVramBytes             = 58;
constexpr int kIdxGfxId                 = 59;
constexpr int kIdxSolverName            = 60;

// SetNumeric, DirectionPerfDbCode, DataTypeName are shared with the perf-config
// picker; see lgbm_common.hpp. SetCategorical is layer-1-only (the solver_name
// and gfx_id categoricals live only in the rank model's feature row).
inline void SetCategorical(LgbmEntry& e, int code)
{
    if(code < 0)
        e.missing = -1;
    else
    {
        e.missing = 0;
        e.fvalue  = static_cast<double>(code);
    }
}

common::ConvDirection ToEngineeredDirection(conv::Direction d)
{
    switch(d)
    {
    case conv::Direction::Forward: return common::ConvDirection::Forward;
    case conv::Direction::BackwardData: return common::ConvDirection::BackwardData;
    case conv::Direction::BackwardWeights: return common::ConvDirection::BackwardWeights;
    }
    return common::ConvDirection::Forward;
}

// Fill the base problem feature block (indices 0..27). The 6 workload features
// (flop_cnt/bytes_*/gflops/bandwidth_gbps) are fed as NaN because they cannot be
// reproduced at runtime; LightGBM routes NaN through the trained missing branch.
void FillProblemFeatures(LgbmEntry* row, const conv::ProblemDescription& p)
{
    const double nan_v = std::numeric_limits<double>::quiet_NaN();

    SetNumeric(row[kIdxNMiniBatchSize], static_cast<double>(p.GetBatchSize()));
    SetNumeric(row[kIdxChannels], static_cast<double>(p.GetInChannels()));
    SetNumeric(row[kIdxDepth], static_cast<double>(p.GetInDepth()));
    SetNumeric(row[kIdxHeight], static_cast<double>(p.GetInHeight()));
    SetNumeric(row[kIdxWidth], static_cast<double>(p.GetInWidth()));
    SetNumeric(row[kIdxOutputChannels], static_cast<double>(p.GetOutChannels()));
    SetNumeric(row[kIdxDepthOutput], static_cast<double>(p.GetOutDepth()));
    SetNumeric(row[kIdxSpatialDim], static_cast<double>(p.GetSpatialDims()));
    SetNumeric(row[kIdxNumDimensions], static_cast<double>(p.GetSpatialDims()) + 2.0);
    SetNumeric(row[kIdxFilterHeightY], static_cast<double>(p.GetWeightsHeight()));
    SetNumeric(row[kIdxFilterWidthX], static_cast<double>(p.GetWeightsWidth()));
    SetNumeric(row[kIdxFilterDepthZ], static_cast<double>(p.GetWeightsDepth()));
    SetNumeric(row[kIdxPadHeight], static_cast<double>(p.GetPadH()));
    SetNumeric(row[kIdxPadWidth], static_cast<double>(p.GetPadW()));
    SetNumeric(row[kIdxPadDepth], static_cast<double>(p.GetPadD()));
    SetNumeric(row[kIdxStrideHeight], static_cast<double>(p.GetKernelStrideH()));
    SetNumeric(row[kIdxStrideWidth], static_cast<double>(p.GetKernelStrideW()));
    SetNumeric(row[kIdxStrideDepth], static_cast<double>(p.GetKernelStrideD()));
    SetNumeric(row[kIdxDilationHeight], static_cast<double>(p.GetDilationH()));
    SetNumeric(row[kIdxDilationWidth], static_cast<double>(p.GetDilationW()));
    SetNumeric(row[kIdxDilationDepth], static_cast<double>(p.GetDilationD()));
    SetNumeric(row[kIdxGroups], static_cast<double>(p.GetGroupCount()));

    SetNumeric(row[kIdxFlopCnt], nan_v);
    SetNumeric(row[kIdxBytesRead], nan_v);
    SetNumeric(row[kIdxBytesWritten], nan_v);
    SetNumeric(row[kIdxBytesProcessed], nan_v);
    SetNumeric(row[kIdxGflops], nan_v);
    SetNumeric(row[kIdxBandwidthGbps], nan_v);
}

// Fill the 13 tn_* GEMM-geometry features (28..40) via common::EngineeredConvFeatures
// (shared with the TunaNet/candidate-selection encoders). H_out/W_out come from
// the output descriptor; 2D geometry is used for all convs, with 3D extent carried
// by the base depth/spatial features.
void FillTunaNetFeatures(LgbmEntry* row, const conv::ProblemDescription& p, std::size_t num_cu)
{
    const auto feats = common::EngineeredConvFeatures(p.GetBatchSize(),
                                                      p.GetInChannels(),
                                                      p.GetOutChannels(),
                                                      p.GetInHeight(),
                                                      p.GetInWidth(),
                                                      p.GetOutHeight(),
                                                      p.GetOutWidth(),
                                                      p.GetWeightsHeight(),
                                                      p.GetWeightsWidth(),
                                                      p.GetGroupCount(),
                                                      num_cu,
                                                      ToEngineeredDirection(p.GetDirection()));
    // EngineeredConvFeatures emits >= 13 values; we consume the first 13 in the
    // model's tn_* order.
    for(int i = 0; i < kNumTnFeatures; ++i)
        SetNumeric(row[kIdxTnBlockBegin + i], static_cast<double>(feats[i]));
}

// Fill the 7 al_* tile-alignment features (41..47). Integer divisibility /
// last-64-tile under-fill of the GEMM-contracting channel dims (see
// deploy/README_CPP_DERIVED.md).
void FillAlignFeatures(LgbmEntry* row, const conv::ProblemDescription& p)
{
    const std::size_t c_in  = p.GetInChannels();
    const std::size_t c_out = p.GetOutChannels();
    const std::size_t n     = p.GetBatchSize();

    SetNumeric(row[kIdxAlC64], (c_in % 64 == 0) ? 1.0 : 0.0);
    SetNumeric(row[kIdxAlC32], (c_in % 32 == 0) ? 1.0 : 0.0);
    SetNumeric(row[kIdxAlOc64], (c_out % 64 == 0) ? 1.0 : 0.0);
    SetNumeric(row[kIdxAlOc32], (c_out % 32 == 0) ? 1.0 : 0.0);
    SetNumeric(row[kIdxAlN8], (n % 8 == 0) ? 1.0 : 0.0);
    // Last-64-tile under-fill fraction ((-x) mod 64)/64, unsigned-safe form.
    SetNumeric(row[kIdxAlCRem64], static_cast<double>((64 - (c_in % 64)) % 64) / 64.0);
    SetNumeric(row[kIdxAlOcRem64], static_cast<double>((64 - (c_out % 64)) % 64) / 64.0);
}

// Fill the categorical problem features (48..52).
void FillProblemCategoricals(LgbmEntry* row,
                             const conv::ProblemDescription& p,
                             const LgbmMetadata& meta)
{
    SetCategorical(row[kIdxDataType],
                   meta.CategoricalCode("data_type", DataTypeName(p.GetInDataType())));
    SetCategorical(
        row[kIdxDirection],
        meta.CategoricalCode("direction", std::to_string(DirectionPerfDbCode(p.GetDirection()))));
    SetCategorical(row[kIdxInLayout], meta.CategoricalCode("in_layout", p.GetInLayout()));
    SetCategorical(row[kIdxFilLayout], meta.CategoricalCode("fil_layout", p.GetWeightsLayout()));
    SetCategorical(row[kIdxOutLayout], meta.CategoricalCode("out_layout", p.GetOutLayout()));
}

// Fill the GPU feature block (53..59): six hipDeviceProp_t fields + gfx_id.
void FillGpuFeatures(LgbmEntry* row,
                     const Handle& handle,
                     const std::string& gfx_id,
                     const LgbmMetadata& meta)
{
    SetNumeric(row[kIdxCuCount], static_cast<double>(handle.GetMaxComputeUnits()));
    SetNumeric(row[kIdxWaveSize], static_cast<double>(handle.GetWavefrontWidth()));
    SetNumeric(row[kIdxLdsSizePerWorkgroupKb],
               static_cast<double>(handle.GetLocalMemorySize()) / 1024.0);
    SetNumeric(row[kIdxL2CacheTotalKb], static_cast<double>(handle.GetL2CacheSize()) / 1024.0);
    SetNumeric(row[kIdxBoostClockMhz], static_cast<double>(handle.GetClockRateKhz()) / 1000.0);
    SetNumeric(row[kIdxVramBytes], static_cast<double>(handle.GetGlobalMemorySize()));

    SetCategorical(row[kIdxGfxId], meta.CategoricalCode("gfx_id", gfx_id));
}

} // namespace

std::vector<uint64_t> PickSolverRanked(const conv::ProblemDescription& problem,
                                       const Handle& handle)
{
    const auto& meta = LgbmMetadata::Get();
    if(!meta.IsReady())
    {
        MIOPEN_LOG_I2("lgbm: abstain (metadata not ready; lgbm_model_meta.json "
                      "missing or failed to load)");
        return {};
    }

    // GetDeviceName() already returns the normalized gfx_id (no
    // :sramecc+:xnack- suffix).
    const std::string gfx_id = handle.GetDeviceName();

    // Architecture gating. A gfx_id in the model's vocab is scored normally. An
    // unknown arch is scored only when the model declares unseen-arch support
    // (AllowUnseenArch(), i.e. trained with gfx_id feature-dropout): gfx_id is
    // routed through the missing branch so the continuous GPU-numeric features
    // carry the arch signal. FillGpuFeatures already encodes an out-of-vocab
    // gfx_id as the missing marker (-1), so no special filling is needed here.
    // Otherwise the picker abstains and falls through to TunaNet. Set
    // MIOPEN_DEBUG_LGBM_DISABLE_OOD_ARCH=1 to force abstain on unknown arch.
    const int gfx_code = meta.CategoricalCode("gfx_id", gfx_id);
    MIOPEN_LOG_I2("lgbm: engaged for gfx_id=\"" << gfx_id << "\" (vocab code " << gfx_code
                                                << "), groups=" << problem.GetGroupCount());
    if(gfx_code < 0)
    {
        const bool ood_ok =
            meta.AllowUnseenArch() && !env::enabled(MIOPEN_DEBUG_LGBM_DISABLE_OOD_ARCH);
        if(!ood_ok)
        {
            MIOPEN_LOG_I2("lgbm: abstain (gfx_id \"" << gfx_id << "\" not in model vocab)");
            return {};
        }
        MIOPEN_LOG_I2("lgbm: unseen arch \"" << gfx_id << "\"; scoring via gfx_id missing branch");
    }

    // Build the constant problem + derived + GPU prefix once; only solver_name
    // (index 60) varies per candidate.
    std::array<LgbmEntry, kNumFeatures> row{};
    FillProblemFeatures(row.data(), problem);
    FillTunaNetFeatures(row.data(), problem, handle.GetMaxComputeUnits());
    FillAlignFeatures(row.data(), problem);
    FillProblemCategoricals(row.data(), problem, meta);
    FillGpuFeatures(row.data(), handle, gfx_id, meta);

    // Score every solver in the vocabulary (lambdarank: higher = predicted
    // faster) and return the solver IDs sorted by score. The downstream walk
    // applies IsApplicable, so no masking is done here.
    //
    // The always-applicable ConvDirectNaiveConv* fallbacks are demoted below all
    // non-naive solvers for low-group convs, so a naive solver is reachable only
    // when nothing else applies. The demotion is gated by group count: at
    // groups >= naive_guard_max_groups naive is often genuinely fastest, so the
    // raw score order is kept there.
    const bool guard_naive =
        problem.GetGroupCount() < static_cast<unsigned>(meta.NaiveGuardMaxGroups());
    const auto& solvers = meta.Solvers();
    struct Scored
    {
        double score;
        std::size_t idx;
        bool demote; // naive fallback that should sink below non-naive solvers
    };
    const auto& forest = LgbmForest::GetRank();
    if(!forest.IsReady())
    {
        MIOPEN_LOG_I2("lgbm: abstain (rank model unavailable)");
        return {};
    }
    std::vector<Scored> scored;
    scored.reserve(solvers.size());
    for(std::size_t i = 0; i < solvers.size(); ++i)
    {
        SetCategorical(row[kIdxSolverName], meta.SolverCode(solvers[i]));
        const double s = forest.Score(row.data(), row.size());
        scored.push_back({s, i, guard_naive && meta.IsNaiveFallback(solvers[i])});
    }
    std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
        if(a.demote != b.demote)
            return !a.demote; // non-demoted solvers rank ahead of demoted naive
        return a.score > b.score;
    });

    std::vector<uint64_t> ranked;
    ranked.reserve(scored.size());
    std::size_t dropped = 0;
    for(const auto& entry : scored)
    {
        // Map the model's solver name to this build's solver Id. Names unknown
        // to this MIOpen version are skipped (model/runtime vocab drift).
        if(const solver::Id id{solvers[entry.idx].c_str()}; id.IsValid())
            ranked.push_back(id.Value());
        else
            ++dropped;
    }
    // An empty result (every scored name unknown to this build) makes the caller
    // fall through to TunaNet/WTI, indistinguishable from an abstain; log the
    // counts so the two can be told apart.
    MIOPEN_LOG_I2("lgbm: scored " << scored.size() << " solvers, " << ranked.size()
                                  << " valid in this build, " << dropped
                                  << " dropped (unknown name), guard_naive=" << guard_naive);
    if(ranked.empty())
        MIOPEN_LOG_I2("lgbm: abstain (no scored solver is known to this MIOpen build)");
    return ranked;
}

int ScoreCandidateMatrixForTest(const std::vector<std::vector<double>>& candidate_rows)
{
    const auto& meta = LgbmMetadata::Get();
    if(!meta.IsReady() || candidate_rows.empty())
        return -1;

    const auto& forest = LgbmForest::GetRank();
    if(!forest.IsReady())
        return -1;

    double best_score = -std::numeric_limits<double>::infinity();
    int best          = -1;
    std::array<LgbmEntry, kNumFeatures> row{};
    for(int c = 0; c < static_cast<int>(candidate_rows.size()); ++c)
    {
        if(candidate_rows[c].size() != static_cast<std::size_t>(kNumFeatures))
            return -1;
        for(int i = 0; i < kNumFeatures; ++i)
            SetNumeric(row[i], candidate_rows[c][i]);
        const double s = forest.Score(row.data(), row.size());
        if(s > best_score)
        {
            best_score = s;
            best       = c;
        }
    }
    return best;
}

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
