// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pcfg_pick.hpp>
#include <miopen/conv/heuristics/lgbm_pcfg_hook.hpp>
#include <miopen/conv/heuristics/lgbm_pcfg_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_predict.hpp>
#include <miopen/conv/heuristics/lgbm_common.hpp>

#include <miopen/conv/problem_description.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_LGBM_PCFG)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace miopen {
namespace ai {
namespace lgbm {
namespace pcfg {

namespace {

// SetNumeric, DirectionPerfDbCode, DataTypeName are shared with the layer-1
// solver picker; see lgbm_common.hpp. DataTypeCode below is pcfg-only (the
// perf-config feature row uses an integer dtype code, not the vocab string).
int DataTypeCode(miopenDataType_t t)
{
    // Matches model_fields.build_X: fp32:0, fp16:1, bf16:2, int8:3.
    if(t == miopenFloat)
        return 0;
    if(t == miopenHalf)
        return 1;
    if(t == miopenBFloat16)
        return 2;
    if(t == miopenInt8)
        return 3;
    return -1;
}

// Fixed gfx_code vocabulary for solvers trained with a gfx_code feature
// (SolverModel::has_gfx_code). Unknown arch -> -1 (the missing-category
// sentinel). This ordering is not carried in the metadata and differs from the
// rank model's gfx_id vocab, so it is hardcoded and must match the order the
// pcfg models were trained with.
int GfxCode(const std::string& gfx_id)
{
    static const std::array<const char*, 10> kGfxOrder = {"gfx906",
                                                          "gfx90a",
                                                          "gfx942",
                                                          "gfx950",
                                                          "gfx1100",
                                                          "gfx1101",
                                                          "gfx1102",
                                                          "gfx1105",
                                                          "gfx1151",
                                                          "gfx1201"};
    for(int i = 0; i < static_cast<int>(kGfxOrder.size()); ++i)
        if(gfx_id == kGfxOrder[static_cast<std::size_t>(i)])
            return i;
    return -1;
}

inline double Log1pAbs(double v) { return std::log1p(std::fabs(v)); }

// Build the problem+GPU prefix, matching model_fields.build_X exactly: 14
// log1p(|geom|) + 5 log1p(|derived|) + 6 raw GPU numerics + direction +
// dtype_code, then (only when with_gfx_code) a trailing gfx_code categorical.
// Order must match prob_feat_cols in the metadata.
void FillProblemPrefix(std::vector<double>& prefix,
                       const conv::ProblemDescription& p,
                       const Handle& handle,
                       const std::string& gfx_id,
                       bool with_gfx_code)
{
    const double channels = static_cast<double>(p.GetInChannels());
    const double height   = static_cast<double>(p.GetInHeight());
    const double width    = static_cast<double>(p.GetInWidth());
    const double out_ch   = static_cast<double>(p.GetOutChannels());
    const double fil_y    = static_cast<double>(p.GetWeightsHeight());
    const double fil_x    = static_cast<double>(p.GetWeightsWidth());
    const double pad_h    = static_cast<double>(p.GetPadH());
    const double pad_w    = static_cast<double>(p.GetPadW());
    const double stride_h = static_cast<double>(p.GetKernelStrideH());
    const double stride_w = static_cast<double>(p.GetKernelStrideW());
    const double dil_h    = static_cast<double>(p.GetDilationH());
    const double dil_w    = static_cast<double>(p.GetDilationW());
    const double groups   = static_cast<double>(p.GetGroupCount());
    const double batch    = static_cast<double>(p.GetBatchSize());

    const double g       = groups < 1.0 ? 1.0 : groups;
    const double cpg     = channels / g;
    const double opg     = out_ch / g;
    const double farea   = fil_y * fil_x;
    const double spatial = height * width;
    const double bxs     = batch * height * width;

    prefix.clear();
    prefix.reserve(kNumBaseProbFeatures + 1);
    // 14 log geometry (order: channels,height,width,output_channels,fil_y,fil_x,
    // pad_h,pad_w,stride_h,stride_w,dil_h,dil_w,groups,n_mini_batch_size)
    prefix.push_back(Log1pAbs(channels));
    prefix.push_back(Log1pAbs(height));
    prefix.push_back(Log1pAbs(width));
    prefix.push_back(Log1pAbs(out_ch));
    prefix.push_back(Log1pAbs(fil_y));
    prefix.push_back(Log1pAbs(fil_x));
    prefix.push_back(Log1pAbs(pad_h));
    prefix.push_back(Log1pAbs(pad_w));
    prefix.push_back(Log1pAbs(stride_h));
    prefix.push_back(Log1pAbs(stride_w));
    prefix.push_back(Log1pAbs(dil_h));
    prefix.push_back(Log1pAbs(dil_w));
    prefix.push_back(Log1pAbs(groups));
    prefix.push_back(Log1pAbs(batch));
    // 5 log derived (cpg,opg,farea,spatial,bxs)
    prefix.push_back(Log1pAbs(cpg));
    prefix.push_back(Log1pAbs(opg));
    prefix.push_back(Log1pAbs(farea));
    prefix.push_back(Log1pAbs(spatial));
    prefix.push_back(Log1pAbs(bxs));
    // 6 raw GPU numerics (cu_count,wave_size,lds_size_per_workgroup_kb,
    // l2_cache_total_kb,boost_clock_mhz,vram_bytes)
    prefix.push_back(static_cast<double>(handle.GetMaxComputeUnits()));
    prefix.push_back(static_cast<double>(handle.GetWavefrontWidth()));
    prefix.push_back(static_cast<double>(handle.GetLocalMemorySize()) / 1024.0);
    prefix.push_back(static_cast<double>(handle.GetL2CacheSize()) / 1024.0);
    prefix.push_back(static_cast<double>(handle.GetClockRateKhz()) / 1000.0);
    prefix.push_back(static_cast<double>(handle.GetGlobalMemorySize()));
    // direction, dtype_code (raw integer codes, not log)
    prefix.push_back(static_cast<double>(DirectionPerfDbCode(p.GetDirection())));
    prefix.push_back(static_cast<double>(DataTypeCode(p.GetInDataType())));
    // optional trailing gfx_code categorical (PCFG_GFXID solvers)
    if(with_gfx_code)
        prefix.push_back(static_cast<double>(GfxCode(gfx_id)));
}

// Score every candidate in the bucket and return their descriptors ordered
// best->worst by predicted speed (lambdarank: higher score = faster). The sort
// is stable, so equal scores keep catalog order. A "" element denotes the solver
// default config. The caller walks this order and takes the first descriptor
// that passes IsValidPerformanceConfig.
std::vector<std::string> RankBucket(const LgbmForest& forest,
                                    const SolverModel& model,
                                    const std::vector<double>& prefix,
                                    const std::vector<Candidate>& cands)
{
    if(cands.empty())
        return {};

    std::vector<LgbmEntry> row(static_cast<std::size_t>(model.feat_count));
    // Fill the constant problem+GPU prefix once.
    for(int i = 0; i < model.prob_feat_count; ++i)
        SetNumeric(row[i], prefix[static_cast<std::size_t>(i)]);

    std::vector<double> scores(cands.size());
    for(std::size_t c = 0; c < cands.size(); ++c)
    {
        const auto& args         = cands[c].args;
        const std::size_t prob_n = static_cast<std::size_t>(model.prob_feat_count);
        for(std::size_t a = 0; a < static_cast<std::size_t>(model.arg_count); ++a)
            SetNumeric(row[prob_n + a], args[a]);

        scores[c] = forest.Score(row.data(), row.size());
    }

    std::vector<std::size_t> order(cands.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    // Stable descending sort: ties preserve catalog order so order[0] matches the
    // Python model's argmax (which uses np.argmax = first max on ties).
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return scores[a] > scores[b];
    });

    std::vector<std::string> ranked;
    ranked.reserve(order.size());
    for(const auto idx : order)
        ranked.push_back(cands[idx].desc);
    return ranked;
}

// The grouped-conv XDLOPS solvers ship a two-tower kernel-tuning-net (KTN)
// perf-config predictor (the *_input_encoder / *_kernel_config_encoder models),
// run inside the solver's own GetDefaultPerformanceConfig on gfx90a/gfx942/gfx950
// (see PerformanceConfig...::IsModelApplicable). The two-tower takes precedence
// there; the LGBM perf-config picker is the fallback for solvers/architectures
// the two-tower does not cover. This predicate mirrors that model's coverage so
// the picker defers rather than competing with it.
bool TwoTowerCoversSolver(const std::string& solver_name, const std::string& gfx_id)
{
    const bool arch_covered = gfx_id.starts_with("gfx90a") || gfx_id.starts_with("gfx942") ||
                              gfx_id.starts_with("gfx950");
    if(!arch_covered)
        return false;
    static const std::array<const char*, 6> kTwoTowerSolvers = {
        "ConvHipImplicitGemmGroupFwdXdlops",
        "ConvHipImplicitGemmGroupBwdXdlops",
        "ConvHipImplicitGemmGroupWrwXdlops",
        "ConvHipImplicitGemm3DGroupFwdXdlops",
        "ConvHipImplicitGemm3DGroupBwdXdlops",
        "ConvHipImplicitGemm3DGroupWrwXdlops"};
    return std::find(kTwoTowerSolvers.begin(), kTwoTowerSolvers.end(), solver_name) !=
           kTwoTowerSolvers.end();
}

} // namespace

std::vector<std::string> PickConfig(const std::string& solver_name,
                                    const conv::ProblemDescription& problem,
                                    const Handle& handle)
{
    const auto& meta = LgbmPcfgMetadata::Get();
    if(!meta.IsReady())
        return {};

    const SolverModel* model = meta.Find(solver_name);
    if(model == nullptr || !model->forest)
        return {}; // no perf-config model for this solver

    const std::string gfx_id = handle.GetDeviceName();

    // Two-tower KTN takes precedence where it applies: defer to the solver's own
    // default-config path (which runs it) rather than preempting it here.
    if(TwoTowerCoversSolver(solver_name, gfx_id))
    {
        MIOPEN_LOG_I2("lgbm_pcfg: deferring " << solver_name << " on " << gfx_id
                                              << " to the two-tower KTN model");
        return {};
    }

    // Otherwise the perf-config picker runs on every architecture its per-solver
    // models cover; a bucket lookup below naturally abstains (empty result) for any
    // gfx_id/direction/dtype the models were not trained on.
    const std::string key = gfx_id + "|" +
                            std::to_string(DirectionPerfDbCode(problem.GetDirection())) + "|" +
                            DataTypeName(problem.GetInDataType());

    const auto bit = model->buckets.find(key);
    if(bit == model->buckets.end())
    {
        MIOPEN_LOG_I2("lgbm_pcfg: no bucket " << key << " for " << solver_name << "; abstain");
        return {};
    }

    std::vector<double> prefix;
    FillProblemPrefix(prefix, problem, handle, gfx_id, model->has_gfx_code);

    auto ranked = RankBucket(*model->forest, *model, prefix, bit->second);
    if(!ranked.empty())
        MIOPEN_LOG_I2("lgbm_pcfg: "
                      << solver_name << " ranked " << ranked.size() << " configs, top=\""
                      << (ranked.front().empty() ? "<default>" : ranked.front()) << "\"");
    return ranked;
}

std::vector<std::string> ScorePickForTest(const std::string& solver_name,
                                          const std::vector<double>& prob_feature_prefix,
                                          const std::vector<std::string>& cand_descs,
                                          const std::vector<std::vector<double>>& cand_args)
{
    const auto& meta = LgbmPcfgMetadata::Get();
    if(!meta.IsReady())
        return {};
    const SolverModel* model = meta.Find(solver_name);
    if(model == nullptr || !model->forest)
        return {};
    if(prob_feature_prefix.size() != static_cast<std::size_t>(model->prob_feat_count) ||
       cand_descs.size() != cand_args.size() || cand_descs.empty())
        return {};

    std::vector<Candidate> cands;
    cands.reserve(cand_descs.size());
    for(std::size_t i = 0; i < cand_descs.size(); ++i)
    {
        if(cand_args[i].size() != static_cast<std::size_t>(model->arg_count))
            return {};
        cands.push_back(Candidate{cand_descs[i], cand_args[i]});
    }
    return RankBucket(*model->forest, *model, prob_feature_prefix, cands);
}

std::vector<std::string> MaybePickConfig(const std::string& solver_db_id,
                                         const conv::ProblemDescription& problem,
                                         const Handle& handle)
{
    // Env gate lives here so the MIOPEN_DEBUG_LGBM_PCFG declaration has a single
    // home and the generic FindSolutionImpl template stays env-var agnostic.
    if(env::disabled(MIOPEN_DEBUG_LGBM_PCFG))
        return {};
    return PickConfig(solver_db_id, problem, handle);
}

} // namespace pcfg
} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
