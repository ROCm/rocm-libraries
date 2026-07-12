#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pick.hpp>
#include <miopen/conv/heuristics/lgbm_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_predict.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp> // common::EngineeredConvFeatures, ConvDirection

#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>

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

// Indices into the v18 TunaNet+Align 61-feature row, matching model_meta.json
// rank.feature_order. The 41 HIP-only base features are unchanged from v16/v17;
// the 20 derived features (13 tn_* GEMM-geometry + 7 al_* tile-alignment) are
// computed in C++ and inserted at 28..47, shifting the categorical + GPU blocks
// down. No embedded GPU data: base GPU inputs are six hipDeviceProp_t fields +
// gfx_id, and the derived features come from conv dims + cu_count only.
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
constexpr int kIdxAlC64    = 41;
constexpr int kIdxAlC32    = 42;
constexpr int kIdxAlOc64   = 43;
constexpr int kIdxAlOc32   = 44;
constexpr int kIdxAlN8     = 45;
constexpr int kIdxAlCRem64 = 46;
constexpr int kIdxAlOcRem64= 47;
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
constexpr int kIdxGfxId      = 59;
constexpr int kIdxSolverName = 60;

// Treelite missing-marker. Generated header sets missing = -1 to indicate
// "present"; we mirror that in our LgbmEntry union.
inline void SetNumeric(LgbmEntry& e, double v)
{
    if(std::isnan(v))
        e.missing = -1;
    else
    {
        e.missing = 0;
        e.fvalue  = v;
    }
}

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

// Map MIOpen's conv::Direction enum to the perf-DB convention used to train
// the model: 1=Forward, 2=BackwardData, 4=BackwardWeights.
int DirectionPerfDbCode(conv::Direction d)
{
    switch(d)
    {
    case conv::Direction::Forward: return 1;
    case conv::Direction::BackwardData: return 2;
    case conv::Direction::BackwardWeights: return 4;
    }
    return 1;
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

std::string DataTypeName(miopenDataType_t t)
{
    // Only the four dtypes in the model's data_type vocab are named; anything
    // else returns "" (encoded as the missing category). An if-chain avoids
    // -Wswitch-enum, which would require listing every miopenDataType_t value.
    if(t == miopenHalf)
        return "fp16";
    if(t == miopenFloat)
        return "fp32";
    if(t == miopenBFloat16)
        return "bf16";
    if(t == miopenInt8)
        return "int8";
    return "";
}

// Fill the base problem feature block (indices 0..27). The 6 derived workload
// features (flop_cnt/bytes_*/gflops/bandwidth_gbps) are fed as NaN: the perf-DB
// instrumentation that produced them at training time is direction-aware in a
// way a runtime textbook estimate cannot reproduce, and a prior validation
// showed NaN reproduces more reference picks. LightGBM treats NaN as a real
// branch direction.
void FillProblemFeatures(LgbmEntry* row, const conv::ProblemDescription& p)
{
    const double nan_v = std::numeric_limits<double>::quiet_NaN();

    SetNumeric(row[kIdxNMiniBatchSize], static_cast<double>(p.GetBatchSize()));
    SetNumeric(row[kIdxChannels],       static_cast<double>(p.GetInChannels()));
    SetNumeric(row[kIdxDepth],          static_cast<double>(p.GetInDepth()));
    SetNumeric(row[kIdxHeight],         static_cast<double>(p.GetInHeight()));
    SetNumeric(row[kIdxWidth],          static_cast<double>(p.GetInWidth()));
    SetNumeric(row[kIdxOutputChannels], static_cast<double>(p.GetOutChannels()));
    SetNumeric(row[kIdxDepthOutput],    static_cast<double>(p.GetOutDepth()));
    SetNumeric(row[kIdxSpatialDim],     static_cast<double>(p.GetSpatialDims()));
    SetNumeric(row[kIdxNumDimensions],  static_cast<double>(p.GetSpatialDims()) + 2.0);
    SetNumeric(row[kIdxFilterHeightY],  static_cast<double>(p.GetWeightsHeight()));
    SetNumeric(row[kIdxFilterWidthX],   static_cast<double>(p.GetWeightsWidth()));
    SetNumeric(row[kIdxFilterDepthZ],   static_cast<double>(p.GetWeightsDepth()));
    SetNumeric(row[kIdxPadHeight],      static_cast<double>(p.GetPadH()));
    SetNumeric(row[kIdxPadWidth],       static_cast<double>(p.GetPadW()));
    SetNumeric(row[kIdxPadDepth],       static_cast<double>(p.GetPadD()));
    SetNumeric(row[kIdxStrideHeight],   static_cast<double>(p.GetKernelStrideH()));
    SetNumeric(row[kIdxStrideWidth],    static_cast<double>(p.GetKernelStrideW()));
    SetNumeric(row[kIdxStrideDepth],    static_cast<double>(p.GetKernelStrideD()));
    SetNumeric(row[kIdxDilationHeight], static_cast<double>(p.GetDilationH()));
    SetNumeric(row[kIdxDilationWidth],  static_cast<double>(p.GetDilationW()));
    SetNumeric(row[kIdxDilationDepth],  static_cast<double>(p.GetDilationD()));
    SetNumeric(row[kIdxGroups],         static_cast<double>(p.GetGroupCount()));

    SetNumeric(row[kIdxFlopCnt],        nan_v);
    SetNumeric(row[kIdxBytesRead],      nan_v);
    SetNumeric(row[kIdxBytesWritten],   nan_v);
    SetNumeric(row[kIdxBytesProcessed], nan_v);
    SetNumeric(row[kIdxGflops],         nan_v);
    SetNumeric(row[kIdxBandwidthGbps],  nan_v);
}

// Fill the 13 tn_* TunaNet GEMM-geometry features (28..40) by reusing MIOpen's
// production EngineeredConvFeatures verbatim -- this is the single source of
// truth shared with the TunaNet/candidate-selection encoders, so the C++ and
// the Python training pipeline cannot drift. H_out/W_out are taken from the
// output descriptor (exact, incl. transpose/asym). 2D geometry is used for all
// convs (matches the model's training); 3D extent is carried by the base
// depth/spatial features.
void FillTunaNetFeatures(LgbmEntry* row,
                         const conv::ProblemDescription& p,
                         std::size_t num_cu)
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

    SetNumeric(row[kIdxAlC64],  (c_in % 64 == 0) ? 1.0 : 0.0);
    SetNumeric(row[kIdxAlC32],  (c_in % 32 == 0) ? 1.0 : 0.0);
    SetNumeric(row[kIdxAlOc64], (c_out % 64 == 0) ? 1.0 : 0.0);
    SetNumeric(row[kIdxAlOc32], (c_out % 32 == 0) ? 1.0 : 0.0);
    SetNumeric(row[kIdxAlN8],   (n % 8 == 0) ? 1.0 : 0.0);
    // Last-64-tile under-fill fraction ((-x) mod 64)/64, unsigned-safe form.
    SetNumeric(row[kIdxAlCRem64],  static_cast<double>((64 - (c_in % 64)) % 64) / 64.0);
    SetNumeric(row[kIdxAlOcRem64], static_cast<double>((64 - (c_out % 64)) % 64) / 64.0);
}

// Fill the categorical problem features (48..52).
void FillProblemCategoricals(LgbmEntry* row,
                             const conv::ProblemDescription& p,
                             const LgbmMetadata& meta)
{
    SetCategorical(row[kIdxDataType],
                   meta.CategoricalCode("data_type", DataTypeName(p.GetInDataType())));
    SetCategorical(row[kIdxDirection],
                   meta.CategoricalCode("direction",
                                        std::to_string(DirectionPerfDbCode(p.GetDirection()))));
    SetCategorical(row[kIdxInLayout],  meta.CategoricalCode("in_layout", p.GetInLayout()));
    SetCategorical(row[kIdxFilLayout], meta.CategoricalCode("fil_layout", p.GetWeightsLayout()));
    SetCategorical(row[kIdxOutLayout], meta.CategoricalCode("out_layout", p.GetOutLayout()));
}

// Fill the GPU feature block (53..59): six hipDeviceProp_t fields + gfx_id.
void FillGpuFeatures(LgbmEntry* row,
                     const Handle& handle,
                     const std::string& gfx_id,
                     const LgbmMetadata& meta)
{
    SetNumeric(row[kIdxCuCount],   static_cast<double>(handle.GetMaxComputeUnits()));
    SetNumeric(row[kIdxWaveSize],  static_cast<double>(handle.GetWavefrontWidth()));
    SetNumeric(row[kIdxLdsSizePerWorkgroupKb],
               static_cast<double>(handle.GetLocalMemorySize()) / 1024.0);
    SetNumeric(row[kIdxL2CacheTotalKb],
               static_cast<double>(handle.GetL2CacheSize()) / 1024.0);
    SetNumeric(row[kIdxBoostClockMhz],
               static_cast<double>(handle.GetClockRateKhz()) / 1000.0);
    SetNumeric(row[kIdxVramBytes], static_cast<double>(handle.GetGlobalMemorySize()));

    SetCategorical(row[kIdxGfxId], meta.CategoricalCode("gfx_id", gfx_id));
}

} // namespace

std::vector<uint64_t> PickSolverRanked(const conv::ProblemDescription& problem,
                                       const Handle& handle)
{
    const auto& meta = LgbmMetadata::Get();
    if(!meta.IsReady())
        return {};

    // GetDeviceName() already returns the normalized gfx_id (no
    // :sramecc+:xnack- suffix). Architecture gating: only run on gfx_ids the
    // model was trained on; otherwise fall through to TunaNet.
    const std::string gfx_id = handle.GetDeviceName();
    if(meta.CategoricalCode("gfx_id", gfx_id) < 0)
    {
        MIOPEN_LOG_I2("lgbm: abstain (gfx_id " << gfx_id << " not in model vocab)");
        return {};
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
    // faster), then return the solver IDs sorted by score. MIOpen's downstream
    // walk keeps every applicable solver in rank order (best-ranked wins), so
    // there is no candidate masking or applicability check here.
    //
    // The ConvDirectNaiveConv* fallbacks are ALWAYS applicable, and the model
    // over-ranks them on out-of-distribution shapes -- ranking one #1 would make
    // the walk pick it over a much faster real kernel. So for LOW-group convs we
    // demote naive fallbacks below every non-naive solver (relative score order
    // preserved), making a naive solver reachable only when nothing else does.
    //
    // The guard is group-count-aware: at groups >= naive_guard_max_groups naive
    // is genuinely the fastest solver a large fraction of the time (an
    // unconditional guard swaps in the slow grouped-XDLOPS solver and regresses
    // badly), so above the threshold we leave the raw score order and let naive
    // win on merit.
    const bool guard_naive =
        problem.GetGroupCount() < static_cast<unsigned>(meta.NaiveGuardMaxGroups());
    const auto& solvers = meta.Solvers();
    struct Scored
    {
        double score;
        std::size_t idx;
        bool demote; // naive fallback that should sink below non-naive solvers
    };
    std::vector<Scored> scored;
    scored.reserve(solvers.size());
    for(std::size_t i = 0; i < solvers.size(); ++i)
    {
        SetCategorical(row[kIdxSolverName], meta.SolverCode(solvers[i]));
        double s = 0.0;
        lgbm_rank_predict(row.data(), /*pred_margin=*/0, &s);
        scored.push_back({s, i, guard_naive && meta.IsNaiveFallback(solvers[i])});
    }
    std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
        if(a.demote != b.demote)
            return !a.demote; // non-demoted solvers rank ahead of demoted naive
        return a.score > b.score;
    });

    std::vector<uint64_t> ranked;
    ranked.reserve(scored.size());
    for(const auto& entry : scored)
    {
        // Map the model's solver name to this build's solver Id. Names unknown
        // to this MIOpen version are skipped (model/runtime vocab drift).
        if(const solver::Id id{solvers[entry.idx].c_str()}; id.IsValid())
            ranked.push_back(id.Value());
    }
    return ranked;
}

int ScoreCandidateMatrixForTest(const std::vector<std::vector<double>>& candidate_rows)
{
    const auto& meta = LgbmMetadata::Get();
    if(!meta.IsReady() || candidate_rows.empty())
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
        double s = 0.0;
        lgbm_rank_predict(row.data(), /*pred_margin=*/0, &s);
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
