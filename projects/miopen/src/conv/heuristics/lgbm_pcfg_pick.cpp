// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pcfg_pick.hpp>
#include <miopen/conv/heuristics/lgbm_pcfg_hook.hpp>
#include <miopen/conv/heuristics/lgbm_pcfg_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_predict.hpp>
#include <miopen/conv/heuristics/lgbm_common.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp> // common::LgbmOnly

#include <miopen/conv/problem_description.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/timer.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_LGBM_PCFG)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
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

inline double Log1pAbs(double v) { return std::log1p(std::fabs(v)); }

// Build the problem+GPU prefix, matching model_fields.build_X exactly: 14
// log1p(|geom|) + 5 log1p(|derived|) + 6 raw GPU numerics + direction +
// dtype_code, then (only when model.has_gfx_code) a trailing gfx_code categorical.
// Order must match prob_feat_cols in the metadata.
void FillProblemPrefix(std::vector<double>& prefix,
                       const conv::ProblemDescription& p,
                       const Handle& handle,
                       const std::string& device_name,
                       const SolverModel& model)
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
    if(model.has_gfx_code)
        prefix.push_back(static_cast<double>(GfxCode(model, device_name)));
}

// Score every candidate in the bucket and return their descriptors and scores ordered
// best->worst by predicted speed (lambdarank: higher score = faster). The sort
// is stable, so equal scores keep catalog order. A "" element denotes the solver
// default config. The caller walks this order and takes the first descriptor
// that passes IsValidPerformanceConfig.
RankedConfigs RankBucket(const LgbmForest& forest,
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

    RankedConfigs ranked;
    ranked.descs.reserve(order.size());
    ranked.scores.reserve(order.size());
    for(const auto idx : order)
    {
        ranked.descs.push_back(cands[idx].desc);
        ranked.scores.push_back(scores[idx]);
    }
    return ranked;
}

// The grouped-conv XDLOPS solvers ship a two-tower kernel-tuning-net (KTN)
// perf-config predictor (the *_input_encoder model plus a precomputed config-embedding
// table that replaces the former *_kernel_config_encoder),
// run inside the solver's own GetDefaultPerformanceConfig on gfx90a/gfx942/gfx950
// (see PerformanceConfig...::IsModelApplicable). The two-tower takes precedence
// there; the LGBM perf-config picker is the fallback for solvers/architectures
// the two-tower does not cover. This predicate mirrors that model's coverage so
// the picker defers rather than competing with it. Under MIOPEN_DEBUG_LGBM_ONLY
// the two-tower is bypassed, so nothing is deferred.
bool TwoTowerCoversSolver(const std::string& solver_name, const std::string& gfx_id)
{
    if(common::LgbmOnly())
        return false;
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

int GfxCode(const SolverModel& model, const std::string& device_name)
{
    const auto& vocab    = model.gfx_vocab;
    const std::string id = NormalizeArchName(device_name);
    const auto it        = std::find(vocab.begin(), vocab.end(), id);
    return it == vocab.end() ? -1 : static_cast<int>(it - vocab.begin());
}

std::string
BucketKey(const std::string& device_name, conv::Direction direction, miopenDataType_t data_type)
{
    return NormalizeArchName(device_name) + "|" + std::to_string(DirectionPerfDbCode(direction)) +
           "|" + DataTypeName(data_type);
}

// ConvHipConv on gfx1250, every direction: the model retrained on hipconv v0.5 configs was trained
// on gfx950 data only and has no gfx1250 buckets, and no model has been checked on gfx1250
// hardware.
//
// Gated here, rather than by dropping the gfx1250 buckets from lgbm_pcfg.bin, so enabling them is
// a one-line change.
bool HeldOutOfPicker(const std::string& solver_name,
                     const std::string& gfx_id,
                     conv::Direction /*direction*/)
{
    return solver_name == "ConvHipConv" && gfx_id.starts_with("gfx1250");
}

// ConvHipConv on gfx950, backward weights: the pick is used only when its score exceeds the
// default config's score by more than this margin. Otherwise the solver runs its default config.
constexpr double kConvHipConvWrwGfx950Margin = 7.5;

std::optional<double>
DefaultMargin(const std::string& solver_name, const std::string& gfx_id, conv::Direction direction)
{
    if(solver_name == "ConvHipConv" && gfx_id.starts_with("gfx950") &&
       direction == conv::Direction::BackwardWeights)
        return kConvHipConvWrwGfx950Margin;
    return std::nullopt;
}

std::vector<std::string>
KeepAboveDefaultMargin(const RankedConfigs& ranked, const std::string& default_desc, double margin)
{
    const auto it = std::find(ranked.descs.begin(), ranked.descs.end(), default_desc);
    // The default is not among the scored candidates, so there is nothing to compare against.
    if(it == ranked.descs.end())
        return {};
    const double bar = ranked.scores[static_cast<std::size_t>(it - ranked.descs.begin())] + margin;

    std::vector<std::string> kept;
    for(std::size_t i = 0; i < ranked.descs.size() && ranked.scores[i] > bar; ++i)
        kept.push_back(ranked.descs[i]);
    return kept;
}

std::vector<std::string> PickConfig(const std::string& solver_name,
                                    const conv::ProblemDescription& problem,
                                    const Handle& handle,
                                    const std::function<std::string()>& default_desc)
{
    // Hold-out first: it needs only the device name, so a held-out combination never loads the
    // solver's model (Find parses and caches it under the metadata mutex).
    //
    // GetDeviceName() strips the :sramecc+:xnack- feature suffix but not a target-variant suffix
    // (gfx1250 parts report "gfx1250-strict"). The bucket keys and the gfx_code vocab use the
    // base name: gfx_id is the normalized name for the arch predicates below, device_name the raw
    // one handed to BucketKey/GfxCode (which normalize it) and kept for the logs.
    std::string device_name;
    {
        ScopedTimeLogger t("lgbm_pcfg.PickConfig.GetDeviceName");
        device_name = handle.GetDeviceName();
    }
    const std::string gfx_id = NormalizeArchName(device_name);

    if(HeldOutOfPicker(solver_name, gfx_id, problem.GetDirection()))
    {
        MIOPEN_LOG_I2("lgbm_pcfg: " << solver_name << " held out of the picker on " << gfx_id
                                    << ", direction " << problem.GetDirectionStr());
        return {};
    }

    const LgbmPcfgMetadata* meta_ptr = nullptr;
    {
        ScopedTimeLogger t("lgbm_pcfg.PickConfig.MetadataGet");
        meta_ptr = &LgbmPcfgMetadata::Get();
    }
    const auto& meta = *meta_ptr;
    if(!meta.IsReady())
        return {};

    const SolverModel* model = meta.Find(solver_name);
    if(model == nullptr || !model->forest)
        return {}; // no perf-config model for this solver

    // Two-tower KTN takes precedence where it applies: defer to the solver's own
    // default-config path (which runs it) rather than preempting it here.
    if(TwoTowerCoversSolver(solver_name, gfx_id))
    {
        MIOPEN_LOG_I2("lgbm_pcfg: deferring " << solver_name << " on " << gfx_id << " (device \""
                                              << device_name << "\") to the two-tower KTN model");
        return {};
    }

    // Otherwise the perf-config picker runs on every architecture its per-solver
    // models cover; a bucket lookup below naturally abstains (empty result) for any
    // gfx_id/direction/dtype the models were not trained on.
    const std::string key = BucketKey(device_name, problem.GetDirection(), problem.GetInDataType());

    const auto bit = model->buckets.find(key);
    if(bit == model->buckets.end())
    {
        MIOPEN_LOG_I2("lgbm_pcfg: no bucket " << key << " (device \"" << device_name << "\") for "
                                              << solver_name << "; abstain");
        return {};
    }

    std::vector<double> prefix;
    {
        ScopedTimeLogger t("lgbm_pcfg.PickConfig.FillProblemPrefix");
        FillProblemPrefix(prefix, problem, handle, device_name, *model);
    }

    RankedConfigs ranked;
    {
        ScopedTimeLogger t("lgbm_pcfg.PickConfig.RankBucket(" + std::to_string(bit->second.size()) +
                           " cands)");
        ranked = RankBucket(*model->forest, *model, prefix, bit->second);
    }
    if(ranked.descs.empty())
        return {};

    MIOPEN_LOG_I2(
        "lgbm_pcfg: " << solver_name << " ranked " << ranked.descs.size() << " configs, top=\""
                      << (ranked.descs.front().empty() ? "<default>" : ranked.descs.front())
                      << "\"");

    if(const auto margin = DefaultMargin(solver_name, gfx_id, problem.GetDirection()))
    {
        auto kept = KeepAboveDefaultMargin(ranked, default_desc(), *margin);
        MIOPEN_LOG_I2("lgbm_pcfg: " << solver_name << " " << kept.size()
                                    << " configs score above the default by more than " << *margin);
        return kept;
    }
    return std::move(ranked.descs);
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
    return RankBucket(*model->forest, *model, prob_feature_prefix, cands).descs;
}

std::vector<std::string> MaybePickConfig(const std::string& solver_db_id,
                                         const conv::ProblemDescription& problem,
                                         const Handle& handle,
                                         const std::function<std::string()>& default_desc)
{
    // Env gate lives here so the MIOPEN_DEBUG_LGBM_PCFG declaration has a single
    // home and the generic FindSolutionImpl template stays env-var agnostic.
    if(env::disabled(MIOPEN_DEBUG_LGBM_PCFG))
        return {};
    return PickConfig(solver_db_id, problem, handle, default_desc);
}

} // namespace pcfg
} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
