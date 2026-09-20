// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pcfg_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_binary.hpp>
#include <miopen/db_path.hpp>
#include <miopen/load_file.hpp>
#include <miopen/logger.hpp>

#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

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
    // Single self-describing binary bundle (see lgbm_binary.hpp and
    // script/convert_lgbm_binary.py): "MIOPCFG1" + u32 version + u32 num_solvers,
    // a directory of (name, offset, size), then per-solver sections holding the
    // feature counts, a FOREST block, and the candidate buckets. Loaded once
    // (Meyers singleton) via a bulk file slurp + pointer walk -- no JSON/text
    // parse. The verbose export-time metadata (feat_order/vocab/dtype_codes) is
    // not needed at runtime and is not in the bundle.
    std::vector<char> buffer;
    try
    {
        buffer = LoadFile(GetSystemDbPath() / "lgbm_pcfg.bin");
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_W("lgbm_pcfg: cannot load lgbm_pcfg.bin (" << e.what()
                                                             << "); picker will abstain");
        return;
    }

    BinReader reader(buffer.data(), buffer.size());
    if(!reader.ReadMagic("MIOPCFG1", 8) || reader.ReadU32() != kBinaryFormatVersion)
    {
        MIOPEN_LOG_W("lgbm_pcfg: lgbm_pcfg.bin bad magic/version; picker will abstain");
        return;
    }

    struct DirEntry
    {
        std::string name;
        std::uint64_t offset;
    };
    const std::uint32_t num_solvers = reader.ReadU32();
    std::vector<DirEntry> directory;
    directory.reserve(num_solvers);
    for(std::uint32_t i = 0; i < num_solvers; ++i)
    {
        DirEntry entry;
        entry.name   = reader.ReadString();
        entry.offset = reader.ReadU64();
        reader.ReadU64(); // section size (unused; sections are read via offsets)
        directory.push_back(std::move(entry));
    }
    if(!reader.Ok())
    {
        MIOPEN_LOG_W("lgbm_pcfg: lgbm_pcfg.bin directory truncated; picker will abstain");
        return;
    }

    for(const auto& entry : directory)
    {
        // Each section is read from its absolute offset, so a single bad section
        // cannot desync the others.
        reader.SeekTo(entry.offset);
        SolverModel m;
        m.feat_count      = reader.ReadI32();
        m.prob_feat_count = reader.ReadI32();
        m.arg_count       = reader.ReadI32();
        m.has_gfx_code    = reader.ReadU8() != 0;

        auto forest = std::make_shared<const LgbmForest>(reader);
        if(!reader.Ok() || !forest->IsReady())
        {
            MIOPEN_LOG_W("lgbm_pcfg: skipping " << entry.name << " (forest unreadable)");
            continue;
        }
        m.forest = std::move(forest);

        const std::uint32_t num_buckets = reader.ReadU32();
        for(std::uint32_t b = 0; b < num_buckets; ++b)
        {
            const std::string key         = reader.ReadString();
            const std::uint32_t num_cands = reader.ReadU32();
            auto& dst                     = m.buckets[key];
            dst.reserve(num_cands);
            for(std::uint32_t c = 0; c < num_cands; ++c)
            {
                Candidate cand;
                cand.desc = reader.ReadString();
                cand.args = reader.ReadArray<double>(static_cast<std::size_t>(m.arg_count));
                dst.push_back(std::move(cand));
            }
        }
        if(!reader.Ok())
        {
            MIOPEN_LOG_W("lgbm_pcfg: skipping " << entry.name << " (catalog truncated)");
            continue;
        }
        models.emplace(entry.name, std::move(m));
    }

    ready = !models.empty();
    if(ready)
        MIOPEN_LOG_I2("lgbm_pcfg metadata loaded: " << models.size() << " solver models (binary)");
    else
        MIOPEN_LOG_W("lgbm_pcfg: no usable solver models; picker will abstain");
}

const SolverModel* LgbmPcfgMetadata::Find(const std::string& solver_name) const
{
    const auto it = models.find(solver_name);
    return it != models.end() ? &it->second : nullptr;
}

std::vector<std::string> LgbmPcfgMetadata::SolverNames() const
{
    std::vector<std::string> names;
    names.reserve(models.size());
    for(const auto& kv : models)
        names.push_back(kv.first);
    return names;
}

} // namespace pcfg
} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
