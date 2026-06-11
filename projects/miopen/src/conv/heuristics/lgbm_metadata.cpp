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
    const auto meta_path   = GetSystemDbPath() / "lgbm_model_meta.json";
    const auto thresh_path = GetSystemDbPath() / "lgbm_per_spec_thresh.json";

    try
    {
        auto meta   = ai::common::LoadJSON(meta_path);
        auto thresh = ai::common::LoadJSON(thresh_path);

        // model_meta.json holds the rank model's feature schema under "rank"
        // (categorical_vocab + solvers). In v5 triple_vocab is a top-level key;
        // older bundles nested it under "appl". Accept either.
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

        const nlohmann::json* triple = nullptr;
        if(meta.contains("triple_vocab"))
            triple = &meta.at("triple_vocab");
        else if(meta.contains("appl") && meta.at("appl").contains("triple_vocab"))
            triple = &meta.at("appl").at("triple_vocab");
        if(triple != nullptr)
        {
            for(auto it = triple->begin(); it != triple->end(); ++it)
                triple_vocab[it.key()] = it.value().get<std::vector<std::string>>();
        }

        if(thresh.contains("default"))
            default_margin = thresh.at("default").get<double>();
        if(thresh.contains("appl_prob_thresh"))
            default_appl_prob = thresh.at("appl_prob_thresh").get<double>();
        if(thresh.contains("per_spec_thresh"))
        {
            for(auto it = thresh.at("per_spec_thresh").begin();
                it != thresh.at("per_spec_thresh").end();
                ++it)
                per_spec_margin[it.key()] = it.value().get<double>();
        }
        if(thresh.contains("per_spec_appl_thresh"))
        {
            for(auto it = thresh.at("per_spec_appl_thresh").begin();
                it != thresh.at("per_spec_appl_thresh").end();
                ++it)
                per_spec_appl[it.key()] = it.value().get<double>();
        }

        ready = true;
        MIOPEN_LOG_I2("LGBM metadata loaded: " << solvers.size() << " solvers, "
                                               << triple_vocab.size() << " triple_vocab entries");
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_W("LGBM metadata load failed (" << e.what() << "); picker will abstain");
        ready = false;
    }
}

double LgbmMetadata::MarginThresh(const std::string& spec_id) const
{
    const auto it = per_spec_margin.find(spec_id);
    return it != per_spec_margin.end() ? it->second : default_margin;
}

double LgbmMetadata::ApplThresh(const std::string& spec_id) const
{
    const auto it = per_spec_appl.find(spec_id);
    return it != per_spec_appl.end() ? it->second : default_appl_prob;
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

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
