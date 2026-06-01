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

// One entry in a matcher's dtype_combos list. Mirrors the dispatch
// signature MIOpen (and other engines) consider when picking solvers:
// input dtype, output dtype, compute dtype, optional intermediate
// dtype. Inline-table form in TOML; named fields rather than positional
// so future dimensions can be added without breaking existing readers.
//
//   io          required. The input dtype. Also the output dtype when
//               `output` is omitted (the symmetric case, common).
//   output      optional. Set only when output dtype differs from io
//               (mixed-precision graphs).
//   compute     required. The compute/accumulation dtype.
//   intermediate optional. Set only when graph_attributes specifies it.
//
// Equality compares all four fields. Empty `output` is normalized to
// `io` before comparison so {io="fp16"} and {io="fp16", output="fp16"}
// are equivalent.
struct DtypeCombo
{
    std::string io;
    std::string output; // empty == same as io (symmetric)
    std::string compute;
    std::string intermediate; // empty == graph didn't set intermediate

    // Effective output dtype (io if output is empty).
    std::string effectiveOutput() const
    {
        return output.empty() ? io : output;
    }

    bool matches(std::string_view obsIo,
                 std::string_view obsOutput,
                 std::string_view obsCompute,
                 std::string_view obsIntermediate) const
    {
        const std::string effectiveOut = effectiveOutput();
        const std::string obsEffective
            = obsOutput.empty() ? std::string(obsIo) : std::string(obsOutput);
        return io == obsIo && effectiveOut == obsEffective && compute == obsCompute
               && intermediate == obsIntermediate;
    }

    bool operator==(const DtypeCombo& other) const
    {
        return io == other.io && effectiveOutput() == other.effectiveOutput()
               && compute == other.compute && intermediate == other.intermediate;
    }

    bool operator<(const DtypeCombo& other) const
    {
        const auto a = std::make_tuple(io, effectiveOutput(), compute, intermediate);
        const auto b
            = std::make_tuple(other.io, other.effectiveOutput(), other.compute, other.intermediate);
        return a < b;
    }
};

// One [[supported.matchers]] entry. Claims the cross-product of
// opChains × dtypeCombos × layouts is fully supported by the engine on
// the owning block's (arch, platform). Exact-string matching only —
// see RFC 0012 §11.4 for why wildcards are intentionally rejected.
//
// dtype_combos uses TOML inline tables with named fields rather than a
// positional array, so the schema mirrors the support-matrix markdown
// display ("[io=bf16, compute=fp32, intermediate=fp32]") and can in
// fact serve as the source of truth that renders it. Earlier shapes
// — flat io_dtypes list, "in->out" string pairs — collapsed
// information the engine actually dispatches on (compute, intermediate)
// or required parser conventions for what TOML already gives us
// natively as inline tables.
struct SupportMatcher
{
    std::vector<std::string> opChains;
    // All claimed dtype combinations, sorted by (io, output, compute,
    // intermediate) for deterministic emission.
    std::vector<DtypeCombo> dtypeCombos;
    std::vector<std::string> layouts;

    // Stable identity used in failure messages so a CI log points back at
    // the exact matcher to edit. Filled in by the loader as
    // "<sidecar>:[[supported]]#<blockIndex>/[[supported.matchers]]#<idx>".
    std::string sourceLocation;

    // Match an observation. obsOutputDtype may be empty — treated as
    // symmetric (output == input). obsIntermediateDtype is empty when
    // the observed graph didn't set intermediate_data_type.
    bool contains(std::string_view opChain,
                  std::string_view obsIo,
                  std::string_view obsOutput,
                  std::string_view obsCompute,
                  std::string_view obsIntermediate,
                  std::string_view layout) const
    {
        if(!memberOf(opChains, opChain) || !memberOf(layouts, layout))
        {
            return false;
        }
        for(const auto& combo : dtypeCombos)
        {
            if(combo.matches(obsIo, obsOutput, obsCompute, obsIntermediate))
            {
                return true;
            }
        }
        return false;
    }

    // Iterate the matcher's full (opChain, io, output, compute,
    // intermediate, layout) cross-product. Visitor returns false to
    // stop early.
    template <typename Fn>
    void forEachTuple(Fn&& fn) const
    {
        for(const auto& op : opChains)
        {
            for(const auto& combo : dtypeCombos)
            {
                const std::string effectiveOut = combo.effectiveOutput();
                for(const auto& layout : layouts)
                {
                    if(!fn(op, combo.io, effectiveOut, combo.compute, combo.intermediate, layout))
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
    // outputDtype may be empty for symmetric I/O; intermediateDtype is
    // empty when the observed graph didn't set intermediate_data_type.
    bool isClaimed(std::string_view archToken,
                   std::string_view platform,
                   std::string_view opChain,
                   std::string_view inputDtype,
                   std::string_view outputDtype,
                   std::string_view computeDtype,
                   std::string_view intermediateDtype,
                   std::string_view layout) const;

private:
    std::filesystem::path _path;
    std::string _engineName;
    std::vector<SupportBlock> _blocks;
};

} // namespace hipdnn_integration_tests
