// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hipdnn_integration_tests
{

// One [[supported.matchers]] entry. Claims the cross-product of
// opChains × (io dtype pairs) × layouts is fully supported by the
// engine on the owning block's (arch, platform). Exact-string matching
// only — see RFC 0012 §11.4 for why wildcards are intentionally rejected.
//
// The dtype dimension supports two equivalent forms in TOML for
// engineer convenience:
//   io_dtypes      = ["fp16", "fp32"]              (symmetric shorthand)
//   io_dtype_pairs = ["fp16->fp32", "bf16->fp32"]  (explicit asymmetric)
// Loader keeps both lists for round-trip readability. Matching checks
// io_dtypes for symmetric pairs and io_dtype_pairs for asymmetric ones;
// a single matcher may use either, both, or neither.
struct SupportMatcher
{
    std::vector<std::string> opChains;
    // Symmetric-shorthand dtypes. "fp16" covers the pair fp16->fp16.
    // May be empty when the matcher exclusively uses io_dtype_pairs.
    std::vector<std::string> ioDtypes;
    // Explicit asymmetric pairs in "in->out" form. May be empty when
    // the matcher is purely symmetric.
    std::vector<std::string> ioDtypePairs;
    std::vector<std::string> layouts;

    // Stable identity used in failure messages so a CI log points back at
    // the exact matcher to edit. Filled in by the loader as
    // "<sidecar>:[[supported]]#<blockIndex>/[[supported.matchers]]#<idx>".
    std::string sourceLocation;

    // Match an observation. outputDtype may be empty — treated as
    // symmetric (output == input).
    bool contains(std::string_view opChain,
                  std::string_view inputDtype,
                  std::string_view outputDtype,
                  std::string_view layout) const
    {
        if(!memberOf(opChains, opChain) || !memberOf(layouts, layout))
        {
            return false;
        }
        const std::string effectiveOutput
            = outputDtype.empty() ? std::string(inputDtype) : std::string(outputDtype);
        return pairMatches(inputDtype, effectiveOutput);
    }

    // Iterate the matcher's full (opChain, inputDtype, outputDtype,
    // layout) cross-product. ioDtypes contributes symmetric pairs
    // (in == out); ioDtypePairs contributes explicit pairs. Visitor
    // returns false to stop early.
    template <typename Fn>
    void forEachTuple(Fn&& fn) const
    {
        for(const auto& op : opChains)
        {
            for(const auto& io : ioDtypes)
            {
                for(const auto& layout : layouts)
                {
                    if(!fn(op, io, io, layout))
                    {
                        return;
                    }
                }
            }
            for(const auto& pair : ioDtypePairs)
            {
                const auto [in, out] = splitDtypePair(pair);
                for(const auto& layout : layouts)
                {
                    if(!fn(op, in, out, layout))
                    {
                        return;
                    }
                }
            }
        }
    }

    // Split "in->out" into (in, out). For malformed strings (no arrow),
    // returns (s, s) so the caller still gets something usable — the
    // loader validates the format up front, so this only matters for
    // defensive code paths.
    static std::pair<std::string, std::string> splitDtypePair(std::string_view pair)
    {
        const auto arrow = pair.find("->");
        if(arrow == std::string_view::npos)
        {
            return {std::string(pair), std::string(pair)};
        }
        return {std::string(pair.substr(0, arrow)), std::string(pair.substr(arrow + 2))};
    }

private:
    static bool memberOf(const std::vector<std::string>& haystack, std::string_view needle)
    {
        return std::any_of(haystack.begin(), haystack.end(), [&](const std::string& candidate) {
            return candidate == needle;
        });
    }

    bool pairMatches(std::string_view in, std::string_view out) const
    {
        // Symmetric shorthand: io_dtypes entry "X" covers pair X->X.
        if(in == out && memberOf(ioDtypes, in))
        {
            return true;
        }
        // Explicit pair: io_dtype_pairs entry "X->Y" covers pair X->Y.
        const std::string needle = std::string(in) + "->" + std::string(out);
        return memberOf(ioDtypePairs, needle);
    }
};

// One [[supported]] block. Scoped to a single (arch, platform).
struct SupportBlock
{
    std::string arch; // required; matched as exact arch-token
    std::optional<std::string> platform; // optional; exact match against "windows"/"linux"
    std::vector<SupportMatcher> matchers;
    std::string sourceLocation; // "<sidecar>:[[supported]]#<index>"
};

// Tokenize a raw gcnArchName by splitting at the first ':'. Returns the
// prefix so callers can exact-match against [[supported]] arch entries
// without substring-matching collisions ("gfx10" vs "gfx1030").
inline std::string archTokenOf(std::string_view rawArch)
{
    auto colon = rawArch.find(':');
    if(colon == std::string_view::npos)
    {
        return std::string(rawArch);
    }
    return std::string(rawArch.substr(0, colon));
}

// Derive the sidecar path that pairs with a main TOML file.
// "MIOPEN_ENGINE.toml" -> "MIOPEN_ENGINE.supported.toml" in the same dir.
inline std::filesystem::path sidecarPathFor(const std::filesystem::path& mainConfig)
{
    auto stem = mainConfig.stem().string(); // "MIOPEN_ENGINE"
    auto extension = mainConfig.extension().string(); // ".toml"
    return mainConfig.parent_path() / (stem + ".supported" + extension);
}

// Parsed contents of a single sidecar file plus the engine name claimed
// in its [meta] block. A SupportClaims instance loads exactly one file;
// callers compose multiple instances (per loaded engine) when needed.
class SupportClaims
{
public:
    SupportClaims() = default;

    // Load the sidecar file at sidecarPath. The file must declare
    // [meta] version = 1 and [meta] engine = <expectedEngineName>; any
    // mismatch throws to make misconfiguration loud. An empty file
    // (sidecar present but with zero [[supported]] blocks) is legal:
    // bring-up state for a new (engine, arch) — the verifier treats it
    // as "not enforced" per RFC 0012 §6.
    SupportClaims(const std::filesystem::path& sidecarPath, std::string_view expectedEngineName);

    const std::filesystem::path& path() const
    {
        return _path;
    }

    const std::string& engineName() const
    {
        return _engineName;
    }

    const std::vector<SupportBlock>& blocks() const
    {
        return _blocks;
    }

    // Return a pointer to the [[supported]] block matching the current
    // (archToken, platform), or nullptr if no block applies. The caller
    // treats nullptr as "not enforced" (RFC 0012 §6).
    const SupportBlock* blockFor(std::string_view archToken, std::string_view platform) const;

    // True iff some matcher in the active block covers the observation.
    // outputDtype may be empty for symmetric I/O (output == input).
    bool isClaimed(std::string_view archToken,
                   std::string_view platform,
                   std::string_view opChain,
                   std::string_view inputDtype,
                   std::string_view outputDtype,
                   std::string_view layout) const;

private:
    std::filesystem::path _path;
    std::string _engineName;
    std::vector<SupportBlock> _blocks;
};

} // namespace hipdnn_integration_tests
