// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pcfg_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_binary.hpp>
#include <miopen/db_path.hpp>
#include <miopen/logger.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
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

namespace {

// Cap on the header+directory prefix read at construction. The directory of a
// realistic bundle is a few KB (a dozen solvers); 1 MiB covers thousands. A
// bundle whose directory somehow exceeds this fails to parse and abstains.
constexpr std::uint64_t kDirPrefixCap = 1u << 20;

// Read exactly [offset, offset+len) of a file. Returns empty on any short read
// or open failure (the caller then abstains). Single named-return so NRVO
// applies (the build is -Werror=nrvo).
std::vector<char> ReadRange(const fs::path& path, std::uint64_t offset, std::uint64_t len)
{
    std::vector<char> buf;
    std::ifstream in(path, std::ios::binary);
    if(in.is_open())
    {
        in.seekg(static_cast<std::streamoff>(offset));
        buf.resize(len);
        in.read(buf.data(), static_cast<std::streamsize>(len));
        if(static_cast<std::uint64_t>(in.gcount()) != len)
            buf.clear();
    }
    return buf;
}

} // namespace

LgbmPcfgMetadata::LgbmPcfgMetadata() : bin_path(GetSystemDbPath() / "lgbm_pcfg.bin")
{
    // Single self-describing binary bundle (see lgbm_binary.hpp and
    // script/convert_lgbm_binary.py): "MIOPCFG1" + u32 version + u32 num_solvers,
    // a directory of (name, offset, size), then per-solver sections holding the
    // feature counts, a FOREST block, and the candidate buckets. Only the
    // header+directory is read here (a small prefix); each section is read from
    // disk and parsed lazily in Find().
    std::uint64_t file_size = 0;
    {
        std::ifstream in(bin_path, std::ios::binary | std::ios::ate);
        if(!in.is_open())
        {
            MIOPEN_LOG_W("lgbm_pcfg: cannot open " << bin_path << "; picker will abstain");
            return;
        }
        file_size = static_cast<std::uint64_t>(in.tellg());
    }

    const std::uint64_t prefix_len = std::min<std::uint64_t>(file_size, kDirPrefixCap);
    const auto prefix              = ReadRange(bin_path, 0, prefix_len);
    if(prefix.empty())
    {
        MIOPEN_LOG_W("lgbm_pcfg: cannot read " << bin_path << "; picker will abstain");
        return;
    }

    BinReader reader(prefix.data(), prefix.size());
    if(!reader.ReadMagic("MIOPCFG1", 8) || reader.ReadU32() != kBinaryFormatVersion)
    {
        MIOPEN_LOG_W("lgbm_pcfg: lgbm_pcfg.bin bad magic/version; picker will abstain");
        return;
    }

    const std::uint32_t num_solvers = reader.ReadU32();
    for(std::uint32_t i = 0; i < num_solvers; ++i)
    {
        const std::string name  = reader.ReadString();
        const std::uint64_t off = reader.ReadU64();
        const std::uint64_t sz  = reader.ReadU64();
        directory.emplace(name, Section{off, sz});
    }
    if(!reader.Ok())
    {
        MIOPEN_LOG_W("lgbm_pcfg: directory truncated or exceeds prefix cap; picker will abstain");
        directory.clear();
        return;
    }

    ready = !directory.empty();
    if(ready)
        MIOPEN_LOG_I2("lgbm_pcfg metadata: " << directory.size()
                                             << " solver models available (lazy)");
    else
        MIOPEN_LOG_W("lgbm_pcfg: no usable solver models; picker will abstain");
}

bool LgbmPcfgMetadata::LoadSection(std::uint64_t offset, std::uint64_t size, SolverModel& out) const
{
    // Read just this solver's byte range from disk; the section is self-contained
    // (no references outside [offset, offset+size)), so it parses from a local
    // buffer at position 0.
    const auto buf = ReadRange(bin_path, offset, size);
    if(buf.empty())
        return false;
    BinReader reader(buf.data(), buf.size());
    out.feat_count      = reader.ReadI32();
    out.prob_feat_count = reader.ReadI32();
    out.arg_count       = reader.ReadI32();
    out.has_gfx_code    = reader.ReadU8() != 0;

    auto forest = std::make_shared<const LgbmForest>(reader);
    if(!reader.Ok() || !forest->IsReady())
        return false;
    out.forest = std::move(forest);

    const std::uint32_t num_buckets = reader.ReadU32();
    for(std::uint32_t b = 0; b < num_buckets; ++b)
    {
        const std::string key         = reader.ReadString();
        const std::uint32_t num_cands = reader.ReadU32();
        auto& dst                     = out.buckets[key];
        dst.reserve(num_cands);
        for(std::uint32_t c = 0; c < num_cands; ++c)
        {
            Candidate cand;
            cand.desc = reader.ReadString();
            cand.args = reader.ReadArray<double>(static_cast<std::size_t>(out.arg_count));
            dst.push_back(std::move(cand));
        }
    }
    return reader.Ok();
}

const SolverModel* LgbmPcfgMetadata::Find(const std::string& solver_name) const
{
    const auto dit = directory.find(solver_name);
    if(dit == directory.end())
        return nullptr; // no perf-config model for this solver

    const std::lock_guard<std::mutex> lock(mutex);
    const auto cit = cache.find(solver_name);
    if(cit != cache.end())
        return &cit->second;

    SolverModel m;
    if(!LoadSection(dit->second.offset, dit->second.size, m))
    {
        MIOPEN_LOG_W("lgbm_pcfg: " << solver_name << " section unreadable; abstaining for it");
        return nullptr;
    }
    return &cache.emplace(solver_name, std::move(m)).first->second;
}

std::vector<std::string> LgbmPcfgMetadata::SolverNames() const
{
    std::vector<std::string> names;
    names.reserve(directory.size());
    for(const auto& kv : directory)
        names.push_back(kv.first);
    return names;
}

} // namespace pcfg
} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
