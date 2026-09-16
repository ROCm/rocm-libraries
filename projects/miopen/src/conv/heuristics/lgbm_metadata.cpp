// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_metadata.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/db_path.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>

#include <nlohmann/json.hpp>

namespace miopen {
namespace ai {
namespace lgbm {

const LgbmMetadata& LgbmMetadata::Get()
{
    static const LgbmMetadata instance;
    return instance;
}

LgbmMetadata::LgbmMetadata()
{
    const auto meta_path = GetSystemDbPath() / "lgbm_model_meta.json";

    try
    {
        auto meta = ai::common::LoadJSON(meta_path);

        const auto& rank = meta.at("rank");

        for(auto it = rank.at("categorical_vocab").begin();
            it != rank.at("categorical_vocab").end();
            ++it)
        {
            categorical_vocab[it.key()] = it.value().get<std::vector<std::string>>();
        }
        solvers = rank.at("solvers").get<std::vector<std::string>>();
        for(int i = 0; i < static_cast<int>(solvers.size()); ++i)
            solver_index[solvers[i]] = i;

        // Always-applicable naive fallbacks; demoted to the tail of the ranked
        // pick list (for low-group convs only). Optional key (older bundles
        // omit it).
        if(rank.contains("naive_fallback_solvers"))
        {
            for(const auto& name : rank.at("naive_fallback_solvers"))
                naive_fallback.insert(name.get<std::string>());
        }
        // Group-count threshold for the naive demotion; defaults to 64.
        if(rank.contains("naive_guard_max_groups"))
            naive_guard_max_groups = rank.at("naive_guard_max_groups").get<int>();

        // Unseen-architecture support: a model trained with gfx_id feature
        // dropout ships rank.gfx_id_unseen_code (the code to feed for an arch
        // outside the vocab; -1 = the missing branch). Its presence is the
        // signal that routing unknown arches through the model is meaningful
        // rather than an abstain. Absent -> older model, keep abstaining.
        if(rank.contains("gfx_id_unseen_code"))
        {
            allow_unseen_arch = true;
            unseen_arch_code  = rank.at("gfx_id_unseen_code").get<int>();
        }

        ready = true;
        MIOPEN_LOG_I2("LGBM metadata loaded: " << solvers.size() << " solvers");
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_W("LGBM metadata load failed (" << e.what() << "); picker will abstain");
        ready = false;
    }
}

int LgbmMetadata::CategoricalCode(const std::string& column, const std::string& value) const
{
    const auto col_it = categorical_vocab.find(column);
    if(col_it == categorical_vocab.end())
        return -1;
    const auto& vocab = col_it->second;
    for(int i = 0; i < static_cast<int>(vocab.size()); ++i)
    {
        if(vocab[i] == value)
            return i;
    }
    return -1;
}

int LgbmMetadata::SolverCode(const std::string& solver_name) const
{
    const auto it = solver_index.find(solver_name);
    return it != solver_index.end() ? it->second : -1;
}

bool LgbmMetadata::IsNaiveFallback(const std::string& solver_name) const
{
    return naive_fallback.count(solver_name) != 0;
}

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
