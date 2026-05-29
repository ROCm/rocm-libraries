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
// opChains x ioDtypes x layouts is fully supported by the engine on the
// owning block's (arch, platform). Exact-string matching only — see
// RFC 0012 §11.4 for why wildcards are intentionally rejected.
struct SupportMatcher
{
    std::vector<std::string> opChains;
    std::vector<std::string> ioDtypes;
    std::vector<std::string> layouts;

    // Stable identity used in failure messages so a CI log points back at
    // the exact matcher to edit. Filled in by the loader as
    // "<sidecar>:[[supported]]#<blockIndex>/[[supported.matchers]]#<idx>".
    std::string sourceLocation;

    bool contains(std::string_view opChain, std::string_view ioDtype, std::string_view layout) const
    {
        return memberOf(opChains, opChain) && memberOf(ioDtypes, ioDtype)
               && memberOf(layouts, layout);
    }

    // Iterate the matcher's full (opChain, ioDtype, layout) cross-product.
    // Cheap for hand-written matchers; the auto-gen tool's safety check
    // calls this too. Visitor returns false to stop early. Stays in the
    // header so the verifier's tuple-coverage loop can inline the body.
    template <typename Fn>
    void forEachTuple(Fn&& fn) const
    {
        for(const auto& op : opChains)
        {
            for(const auto& io : ioDtypes)
            {
                for(const auto& layout : layouts)
                {
                    if(!fn(op, io, layout))
                    {
                        return;
                    }
                }
            }
        }
    }

private:
    static bool memberOf(const std::vector<std::string>& haystack, std::string_view needle)
    {
        return std::any_of(haystack.begin(), haystack.end(), [&](const std::string& candidate) {
            return candidate == needle;
        });
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
    bool isClaimed(std::string_view archToken,
                   std::string_view platform,
                   std::string_view opChain,
                   std::string_view ioDtype,
                   std::string_view layout) const;

private:
    std::filesystem::path _path;
    std::string _engineName;
    std::vector<SupportBlock> _blocks;
};

} // namespace hipdnn_integration_tests
