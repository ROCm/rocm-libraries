// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pcfg_metadata.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/db_path.hpp>
#include <miopen/logger.hpp>

#include <nlohmann/json.hpp>

#include <limits>
#include <utility>

namespace miopen {
namespace ai {
namespace lgbm {
namespace pcfg {

const LgbmPcfgMetadata& LgbmPcfgMetadata::Get()
{
    static const LgbmPcfgMetadata instance;
    return instance;
}

LgbmPcfgMetadata::LgbmPcfgMetadata()
{
    const auto meta_path    = GetSystemDbPath() / "lgbm_pcfg_model_meta.json";
    const auto catalog_path = GetSystemDbPath() / "lgbm_pcfg_catalog.json";

    try
    {
        // meta: { "<solver>": { feat_order, prob_feat_cols, arg_cols, ... }, ... }
        auto meta = ai::common::LoadJSON(meta_path);
        for(auto it = meta.begin(); it != meta.end(); ++it)
        {
            const auto& block        = it.value();
            const std::size_t n_feat = block.at("feat_order").size();
            const auto& prob_cols    = block.at("prob_feat_cols");
            const std::size_t n_prob = prob_cols.size();
            const std::size_t n_arg  = block.at("arg_cols").size();

            // The prefix is the base set, optionally with a trailing gfx_code
            // categorical (PCFG_GFXID solvers). Detect it from the last column so
            // the C++ feature builder knows whether to append gfx_code. Reject
            // anything else so a real schema drift fails loudly here rather than
            // silently corrupting predictions downstream.
            const bool has_gfx_code =
                n_prob == static_cast<std::size_t>(kNumBaseProbFeatures) + 1 &&
                prob_cols.back().get<std::string>() == "gfx_code";
            const bool base_ok = n_prob == static_cast<std::size_t>(kNumBaseProbFeatures);
            if(!(base_ok || has_gfx_code) || n_feat != n_prob + n_arg)
            {
                MIOPEN_LOG_W("lgbm_pcfg: skipping "
                             << it.key() << " (feat schema mismatch: prob=" << n_prob
                             << " arg=" << n_arg << " feat=" << n_feat << ")");
                continue;
            }

            // Load the solver's LightGBM forest asset. A solver whose model
            // file is missing/unparseable is dropped here (the picker abstains
            // for it) rather than kept with no way to score.
            const auto model_file = GetSystemDbPath() / ("lgbm_pcfg_" + it.key() + "_model.txt");
            auto forest           = std::make_shared<const LgbmForest>(model_file.string());
            if(!forest->IsReady())
            {
                MIOPEN_LOG_W("lgbm_pcfg: skipping " << it.key() << " (model file "
                                                    << model_file.string() << " unavailable)");
                continue;
            }

            SolverModel m;
            m.feat_count      = static_cast<int>(n_feat);
            m.prob_feat_count = static_cast<int>(n_prob);
            m.arg_count       = static_cast<int>(n_arg);
            m.has_gfx_code    = has_gfx_code;
            m.forest          = std::move(forest);
            models.emplace(it.key(), std::move(m));
        }

        // catalog: { "<solver>": { "buckets": { "<key>": [ {desc, args}, ... ] } } }
        auto catalog = ai::common::LoadJSON(catalog_path);
        for(auto it = catalog.begin(); it != catalog.end(); ++it)
        {
            const auto mit = models.find(it.key());
            if(mit == models.end())
                continue; // catalog entry without a matching model block
            SolverModel& m      = mit->second;
            const auto& buckets = it.value().at("buckets");
            for(auto bit = buckets.begin(); bit != buckets.end(); ++bit)
            {
                auto& dst = m.buckets[bit.key()];
                for(const auto& c : bit.value())
                {
                    const auto& jargs = c.at("args");
                    // The picker indexes args[0..arg_count) positionally when
                    // building the feature row; a candidate with the wrong arg
                    // count would be an out-of-bounds read. Skip (and log) any
                    // mismatch rather than trust file-loaded data blindly. The
                    // exporter's parity gate guarantees a match, so this only
                    // guards against a desynced catalog/meta pair.
                    if(jargs.size() != static_cast<std::size_t>(m.arg_count))
                    {
                        MIOPEN_LOG_W("lgbm_pcfg: skipping candidate in "
                                     << it.key() << " bucket " << bit.key() << " (args="
                                     << jargs.size() << ", expected " << m.arg_count << ")");
                        continue;
                    }
                    Candidate cand;
                    cand.desc = c.at("desc").get<std::string>();
                    cand.args.reserve(jargs.size());
                    for(const auto& a : jargs)
                    {
                        // Exported args are ints, floats, or null (missing).
                        // A missing feature is encoded as NaN downstream, so map
                        // null -> NaN here.
                        cand.args.push_back(a.is_null() ? std::numeric_limits<double>::quiet_NaN()
                                                        : a.get<double>());
                    }
                    dst.push_back(std::move(cand));
                }
            }
        }

        ready = !models.empty();
        if(ready)
            MIOPEN_LOG_I2("lgbm_pcfg metadata loaded: " << models.size() << " solver models");
        else
            MIOPEN_LOG_W("lgbm_pcfg: no usable solver models; picker will abstain");
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_W("lgbm_pcfg metadata load failed (" << e.what() << "); picker will abstain");
        ready = false;
    }
}

const SolverModel* LgbmPcfgMetadata::Find(const std::string& solver_name) const
{
    const auto it = models.find(solver_name);
    return it != models.end() ? &it->second : nullptr;
}

} // namespace pcfg
} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
