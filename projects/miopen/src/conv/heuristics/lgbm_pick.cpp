#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pick.hpp>
#include <miopen/conv/heuristics/lgbm_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_predict.hpp>

#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace miopen {
namespace ai {
namespace lgbm {

namespace {

// Indices into the v16 41-feature row, matching model_meta.json
// rank.feature_order. v16 is HIP-only: every GPU feature that is not directly
// readable from hipDeviceProp_t was dropped, so there is no embedded per-arch
// table. The only GPU inputs are the six hipDeviceProp_t-backed numerics below
// plus gfx_id.
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
constexpr int kIdxDataType       = 28;
constexpr int kIdxDirection      = 29;
constexpr int kIdxInLayout       = 30;
constexpr int kIdxFilLayout      = 31;
constexpr int kIdxOutLayout      = 32;
constexpr int kIdxCuCount               = 33;
constexpr int kIdxWaveSize              = 34;
constexpr int kIdxLdsSizePerWorkgroupKb = 35;
constexpr int kIdxL2CacheTotalKb        = 36;
constexpr int kIdxBoostClockMhz         = 37;
constexpr int kIdxVramBytes             = 38;
constexpr int kIdxGfxId                 = 39;
constexpr int kIdxSolverName            = 40;

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

// Fill the problem feature block (indices 0..32). The 6 derived workload
// features (flop_cnt/bytes_*/gflops/bandwidth_gbps) are fed as NaN: the
// perf-DB instrumentation that produced them at training time is direction-
// aware in a way a runtime textbook estimate cannot reproduce, and a prior
// validation showed NaN reproduces more reference picks than a textbook
// estimate. LightGBM treats NaN as a real branch direction.
void FillProblemFeatures(LgbmEntry* row,
                         const conv::ProblemDescription& p,
                         const LgbmMetadata& meta)
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

    SetCategorical(row[kIdxDataType],
                   meta.CategoricalCode("data_type", DataTypeName(p.GetInDataType())));
    SetCategorical(row[kIdxDirection],
                   meta.CategoricalCode("direction",
                                        std::to_string(DirectionPerfDbCode(p.GetDirection()))));
    SetCategorical(row[kIdxInLayout],  meta.CategoricalCode("in_layout", p.GetInLayout()));
    SetCategorical(row[kIdxFilLayout], meta.CategoricalCode("fil_layout", p.GetWeightsLayout()));
    SetCategorical(row[kIdxOutLayout], meta.CategoricalCode("out_layout", p.GetOutLayout()));
}

// Fill the GPU feature block (indices 33..39). v16 uses only fields readable
// from the live device via the Handle (hipDeviceProp_t) plus gfx_id; there is
// no curated per-arch data, so the model can project to unseen architectures.
void FillGpuFeatures(LgbmEntry* row, const Handle& handle, const std::string& gfx_id,
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

// Score the full solver vocabulary over a finished problem+GPU prefix and
// return the argmax solver index. `row[kIdxSolverName]` is overwritten per
// candidate. v16 has no candidate masking, margin gate, or applicability VETO.
std::size_t ArgmaxOverVocab(std::array<LgbmEntry, kNumFeatures>& row, const LgbmMetadata& meta)
{
    const auto& solvers = meta.Solvers();
    double best_score   = -std::numeric_limits<double>::infinity();
    std::size_t top     = 0;
    for(std::size_t i = 0; i < solvers.size(); ++i)
    {
        SetCategorical(row[kIdxSolverName], meta.SolverCode(solvers[i]));
        double s = 0.0;
        lgbm_rank_predict(row.data(), /*pred_margin=*/0, &s);
        if(s > best_score)
        {
            best_score = s;
            top        = i;
        }
    }
    return top;
}

} // namespace

solver::Id PickSolver(const conv::ProblemDescription& problem, const Handle& handle)
{
    const auto& meta = LgbmMetadata::Get();
    if(!meta.IsReady())
        return {};

    // GetDeviceName() already returns the normalized gfx_id (no
    // :sramecc+:xnack- suffix).
    const std::string gfx_id = handle.GetDeviceName();

    // Architecture gating: only run on gfx_ids the model was trained on. The
    // feature set is otherwise fully runtime-derived, so the model can project
    // to unseen architectures; this gate keeps it to validated archs until that
    // projection is vetted on new silicon.
    if(meta.CategoricalCode("gfx_id", gfx_id) < 0)
    {
        MIOPEN_LOG_I2("lgbm: abstain (gfx_id " << gfx_id << " not in model vocab)");
        return {};
    }

    std::array<LgbmEntry, kNumFeatures> row{};
    FillProblemFeatures(row.data(), problem, meta);
    FillGpuFeatures(row.data(), handle, gfx_id, meta);

    const std::size_t top = ArgmaxOverVocab(row, meta);

    if(solver::Id id{meta.Solvers()[top].c_str()}; id.IsValid())
        return id;
    MIOPEN_LOG_I2("lgbm: solver \"" << meta.Solvers()[top]
                                     << "\" unknown to this MIOpen build; abstain");
    return {};
}

std::string ScoreRowArgmaxForTest(const std::vector<double>& feature_row)
{
    const auto& meta = LgbmMetadata::Get();
    if(!meta.IsReady() || feature_row.size() != static_cast<std::size_t>(kNumFeatures))
        return "";

    std::array<LgbmEntry, kNumFeatures> row{};
    for(int i = 0; i < kNumFeatures; ++i)
        SetNumeric(row[i], feature_row[i]);

    const std::size_t top = ArgmaxOverVocab(row, meta);
    return meta.Solvers()[top];
}

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
