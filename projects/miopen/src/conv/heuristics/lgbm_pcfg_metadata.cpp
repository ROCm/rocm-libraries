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
#include <limits>
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

// Arg columns are referenced from a catalog template by a u8 (1..255, see
// script/convert_lgbm_binary.py), so a section cannot use more than 255.
constexpr int kMaxArgCount = 255;

// Cap on the memory one solver section may decode to (columns, candidate args,
// descriptors, strings, bucket entries). The compressed bytes do not bound this:
// a constant column or a shared string prefix costs a few bytes and expands to
// megabytes. The largest shipped section decodes to about 18 MB; 128 MiB leaves
// room to grow while keeping a corrupt section from exhausting memory.
constexpr std::uint64_t kMaxDecodedBytes = std::uint64_t{1} << 27;

// Remaining decoded-size allowance for the section being parsed.
struct DecodeBudget
{
    std::uint64_t left = kMaxDecodedBytes;

    // Reserve count * unit bytes. False, and nothing reserved, if that exceeds
    // what is left (overflow-safe: the product is never formed before the check).
    bool Charge(std::uint64_t count, std::uint64_t unit)
    {
        if(unit != 0 && count > left / unit)
            return false;
        left -= count * unit;
        return true;
    }
};

// A column of n integers (see script/convert_lgbm_binary.py): d distinct values
// (d == 1: the value; else ascending zigzag deltas), then, for d > 1, n packed
// indices into them.
bool ReadIntColumn(BinReader& reader, std::size_t n, std::vector<std::int64_t>& out)
{
    const std::uint64_t d = reader.ReadVarint();
    if(!reader.Ok() || d == 0 || d > n)
        return false;
    std::vector<std::int64_t> values(d);
    std::uint64_t prev = 0;
    for(auto& v : values)
    {
        prev += static_cast<std::uint64_t>(reader.ReadZigzag());
        v = static_cast<std::int64_t>(prev);
    }
    if(d == 1)
    {
        out.assign(n, values[0]);
        return reader.Ok();
    }
    unsigned width = 0;
    for(std::uint64_t m = d - 1; m != 0; m >>= 1)
        ++width;
    const auto index = reader.ReadPacked(n, width);
    if(!reader.Ok())
        return false;
    out.resize(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        if(index[i] >= d)
            return false;
        out[i] = values[index[i]];
    }
    return true;
}

// `count` strings, each stored as the number of leading bytes shared with the
// previous string, then its remaining bytes.
bool ReadStrings(BinReader& reader,
                 std::uint64_t count,
                 DecodeBudget& budget,
                 std::vector<std::string>& out)
{
    std::string prev;
    for(std::uint64_t i = 0; i < count; ++i)
    {
        const std::uint64_t shared = reader.ReadVarint();
        const std::uint64_t tail   = reader.ReadVarint();
        if(!reader.Ok() || shared > prev.size() || tail > reader.Remaining())
            return false;
        // A string can share its whole prefix with the previous one for two
        // bytes of input, so charge the decoded size, not the bytes read.
        if(!budget.Charge(1, shared + tail + sizeof(std::string)))
            return false;
        prev.resize(shared);
        prev += reader.ReadChars(tail);
        out.push_back(prev);
    }
    return reader.Ok();
}

// Reads one solver's catalog (layout in script/convert_lgbm_binary.py): the
// distinct candidates, then the buckets that index into them. A descriptor is a
// template whose kSlotMark bytes stand for decimal numbers; each number is
// either stored per candidate or equal to one of the candidate's args.
// Args decode to double; INT32_MIN (the exporter's "missing") decodes to NaN, as
// the previous f64 catalog stored it.
// `arg_count` must be at most kMaxArgCount.
bool ReadCatalog(BinReader& reader,
                 std::size_t arg_count,
                 std::unordered_map<std::string, std::vector<Candidate>>& buckets)
{
    constexpr char kSlotMark               = '\x01';
    constexpr std::uint64_t kMaxCandidates = 1u << 16;
    constexpr std::uint64_t kMaxDigits     = 20; // decimal digits of an int64
    constexpr std::int64_t kMissingArg     = std::numeric_limits<std::int32_t>::min();

    DecodeBudget budget;

    const std::uint64_t num_unique    = reader.ReadVarint();
    const std::uint64_t num_templates = reader.ReadVarint();
    if(!reader.Ok() || num_unique > kMaxCandidates || num_templates > num_unique)
        return false;
    std::vector<std::string> templates;
    if(!ReadStrings(reader, num_templates, budget, templates))
        return false;
    std::vector<std::size_t> counts(num_templates);
    std::uint64_t total = 0;
    for(auto& c : counts)
    {
        const std::uint64_t v = reader.ReadVarint();
        if(!reader.Ok() || v == 0 || v > num_unique)
            return false;
        c = static_cast<std::size_t>(v);
        total += v;
    }
    if(total != num_unique)
        return false;

    // Every arg column takes at least two bytes, so this bounds the column count
    // by the input. It does not bound the decoded size (constant columns), which
    // the budget does: the int64 columns and the double args built from them.
    if(num_unique > 0 && arg_count > reader.Remaining() / 2)
        return false;
    if(!budget.Charge(num_unique,
                      sizeof(Candidate) + arg_count * (sizeof(std::int64_t) + sizeof(double))))
        return false;
    std::vector<Candidate> unique(num_unique);
    if(num_unique > 0)
    {
        std::vector<std::vector<std::int64_t>> cols(arg_count);
        for(auto& col : cols)
            if(!ReadIntColumn(reader, num_unique, col))
                return false;

        std::size_t first = 0;
        for(std::size_t t = 0; t < num_templates; ++t)
        {
            const std::string& tpl = templates[t];
            const std::size_t num_slots =
                static_cast<std::size_t>(std::count(tpl.begin(), tpl.end(), kSlotMark));
            // Charge the per-slot columns and descriptors of this template up front
            // (the columns are freed per template, but this keeps one simple bound).
            if(!budget.Charge(counts[t], num_slots * sizeof(std::int64_t)) ||
               !budget.Charge(counts[t], tpl.size() + num_slots * kMaxDigits))
                return false;
            std::vector<std::vector<std::int64_t>> own(num_slots);
            std::vector<const std::int64_t*> slot(num_slots);
            for(std::size_t s = 0; s < num_slots; ++s)
            {
                const std::uint8_t ref = reader.ReadU8();
                if(ref == 0)
                {
                    if(!ReadIntColumn(reader, counts[t], own[s]))
                        return false;
                    slot[s] = own[s].data();
                }
                else if(ref <= arg_count)
                    slot[s] = cols[ref - 1].data() + first;
                else
                    return false;
            }
            for(std::size_t k = 0; k < counts[t]; ++k)
            {
                std::string& desc = unique[first + k].desc;
                std::size_t s     = 0;
                for(const char ch : tpl)
                {
                    if(ch != kSlotMark)
                    {
                        desc += ch;
                        continue;
                    }
                    const std::int64_t v = slot[s++][k];
                    if(v < 0)
                        return false;
                    desc += std::to_string(v);
                }
            }
            first += counts[t];
        }

        for(std::size_t i = 0; i < unique.size(); ++i)
        {
            auto& args = unique[i].args;
            args.reserve(arg_count);
            for(const auto& col : cols)
                args.push_back(col[i] == kMissingArg ? std::numeric_limits<double>::quiet_NaN()
                                                     : static_cast<double>(col[i]));
        }
    }

    const std::uint64_t num_buckets = reader.ReadVarint();
    std::vector<std::string> keys;
    if(!reader.Ok() || !ReadStrings(reader, num_buckets, budget, keys))
        return false;
    const unsigned idx_width = reader.ReadU8();
    for(const auto& key : keys)
    {
        const std::uint64_t n = reader.ReadVarint();
        // The index vector and the bucket's candidate slots.
        if(!reader.Ok() || !budget.Charge(n, sizeof(std::uint32_t) + sizeof(Candidate)))
            return false;
        const auto index = reader.ReadPacked(static_cast<std::size_t>(n), idx_width);
        if(!reader.Ok())
            return false;
        auto& dst = buckets[key];
        dst.reserve(index.size());
        for(const std::uint32_t i : index)
        {
            if(i >= unique.size())
                return false;
            // Each entry is a copy of the candidate's descriptor and args.
            if(!budget.Charge(1, unique[i].desc.size() + unique[i].args.size() * sizeof(double)))
                return false;
            dst.push_back(unique[i]);
        }
    }
    return reader.Ok();
}

// Parse one solver section (feature counts, forest, catalog) from `buf`, which
// holds exactly that section. Returns false on any malformed or oversized field.
bool ParseSection(const std::vector<char>& buf, SolverModel& out)
{
    if(buf.empty())
        return false;
    BinReader reader(buf.data(), buf.size());
    out.feat_count      = reader.ReadI32();
    out.prob_feat_count = reader.ReadI32();
    out.arg_count       = reader.ReadI32();
    out.has_gfx_code    = reader.ReadU8() != 0;

    // The header fixes the schema (see SolverModel): feat_count is the problem
    // prefix plus the arg columns, and the picker sizes its feature row from it.
    // Reject a header that disagrees, or an arg count the format cannot express,
    // before anything is allocated from it.
    if(!reader.Ok() || out.arg_count < 0 || out.arg_count > kMaxArgCount ||
       out.prob_feat_count != kNumBaseProbFeatures + (out.has_gfx_code ? 1 : 0) ||
       out.feat_count != out.prob_feat_count + out.arg_count)
        return false;

    if(out.has_gfx_code)
    {
        // A gfx_code model without its vocab cannot be encoded; reject the
        // section so the picker abstains for this solver.
        const std::uint16_t num_gfx = reader.ReadU16();
        if(num_gfx == 0)
            return false;
        out.gfx_vocab.reserve(num_gfx);
        for(std::uint16_t i = 0; i < num_gfx; ++i)
            out.gfx_vocab.push_back(reader.ReadString());
    }

    auto forest = std::make_shared<const LgbmForest>(reader);
    if(!reader.Ok() || !forest->IsReady())
        return false;
    out.forest = std::move(forest);

    // Catalog (v5): the distinct candidates, then the buckets that index them.
    return ReadCatalog(reader, static_cast<std::size_t>(out.arg_count), out.buckets);
}

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
    if(!reader.ReadMagic("MIOPCFG1", 8) || reader.ReadU32() != kPcfgFormatVersion)
    {
        MIOPEN_LOG_W("lgbm_pcfg: lgbm_pcfg.bin bad magic/version; picker will abstain");
        return;
    }

    const std::uint32_t num_solvers = reader.ReadU32();
    for(std::uint32_t i = 0; i < num_solvers && reader.Ok(); ++i)
    {
        const std::string name  = reader.ReadString();
        const std::uint64_t off = reader.ReadU64();
        const std::uint64_t sz  = reader.ReadU64();
        // A section must lie inside the file; its size is allocated when it is read.
        if(off > file_size || sz > file_size - off)
        {
            MIOPEN_LOG_W("lgbm_pcfg: section " << name << " lies outside the file");
            directory.clear();
            return;
        }
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
    return ParseSection(ReadRange(bin_path, offset, size), out);
}

bool ParseSolverSectionForTest(const char* data, std::size_t size, SolverModel& out)
{
    return ParseSection(std::vector<char>(data, data + size), out);
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
