// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/SupportClaims.hpp"

#include <set>
#include <stdexcept>
#include <utility>

// HIP/tomlplusplus macro collision workaround — mirrors TestSettings.hpp.
#ifdef __noinline__
#pragma push_macro("__noinline__")
#undef __noinline__
#include <toml.hpp>
#pragma pop_macro("__noinline__")
#else
#include <toml.hpp>
#endif

namespace hipdnn_integration_tests
{

namespace
{

std::vector<std::string>
    parseRequiredStringArray(const toml::table& table, const char* key, const std::string& label)
{
    const auto* arr = table[key].as_array();
    if(arr == nullptr)
    {
        throw std::runtime_error("SupportClaims: " + label + " missing required '"
                                 + std::string(key) + "' array");
    }
    std::vector<std::string> result;
    result.reserve(arr->size());
    std::set<std::string> seen;
    for(const auto& entry : *arr)
    {
        auto value = entry.value<std::string>();
        if(!value.has_value())
        {
            throw std::runtime_error("SupportClaims: " + label + " '" + std::string(key)
                                     + "' must contain only strings");
        }
        if(!seen.insert(*value).second)
        {
            throw std::runtime_error("SupportClaims: " + label + " '" + std::string(key)
                                     + "' contains duplicate '" + *value + "'");
        }
        result.push_back(std::move(*value));
    }
    if(result.empty())
    {
        throw std::runtime_error("SupportClaims: " + label + " '" + std::string(key)
                                 + "' must not be empty");
    }
    return result;
}

void rejectWildcards(const std::vector<std::string>& values,
                     const char* key,
                     const std::string& label)
{
    for(const auto& value : values)
    {
        if(value.find('*') != std::string::npos || value.find('?') != std::string::npos)
        {
            throw std::runtime_error("SupportClaims: " + label + " '" + std::string(key)
                                     + "' value '" + value
                                     + "' contains a wildcard; matchers are exact-string "
                                       "only (RFC 0012 §11.4)");
        }
    }
}

// Parse one dtype_combos inline table:
//   {io="fp16", output="fp32", compute="fp32", intermediate="fp32"}
// `io` and `compute` are required; `output` and `intermediate` are
// optional. Any wildcard character in a value, or any unknown key, is
// rejected — schema drift is loud rather than silent.
DtypeCombo
    parseDtypeCombo(const toml::table& comboTable, const std::string& label, size_t comboIndex)
{
    const std::string entryLabel = label + " dtype_combos[" + std::to_string(comboIndex) + "]";

    const auto readRequired = [&](const char* key) {
        auto value = comboTable[key].value<std::string>();
        if(!value.has_value() || value->empty())
        {
            throw std::runtime_error("SupportClaims: " + entryLabel + " missing required '"
                                     + std::string(key) + "'");
        }
        if(value->find('*') != std::string::npos || value->find('?') != std::string::npos)
        {
            throw std::runtime_error("SupportClaims: " + entryLabel + " '" + std::string(key)
                                     + "' value '" + *value
                                     + "' contains a wildcard (RFC 0012 §11.4)");
        }
        return *value;
    };
    const auto readOptional = [&](const char* key) {
        auto value = comboTable[key].value<std::string>();
        if(!value.has_value())
        {
            return std::string{};
        }
        if(value->find('*') != std::string::npos || value->find('?') != std::string::npos)
        {
            throw std::runtime_error("SupportClaims: " + entryLabel + " '" + std::string(key)
                                     + "' value '" + *value
                                     + "' contains a wildcard (RFC 0012 §11.4)");
        }
        return *value;
    };

    DtypeCombo combo;
    combo.io = readRequired("io");
    combo.compute = readRequired("compute");
    combo.output = readOptional("output");
    combo.intermediate = readOptional("intermediate");

    // Catch typos: any key outside the known set means either the
    // schema is being extended without a version bump or the engineer
    // typo'd a field name. Either way it's not safe to silently ignore.
    for(const auto& [keyNode, _] : comboTable)
    {
        const std::string keyStr(keyNode.str());
        if(keyStr != "io" && keyStr != "output" && keyStr != "compute" && keyStr != "intermediate")
        {
            throw std::runtime_error("SupportClaims: " + entryLabel + " unknown key '" + keyStr
                                     + "' (expected io / output / compute / intermediate)");
        }
    }
    return combo;
}

SupportMatcher parseMatcher(const toml::node& node,
                            const std::filesystem::path& sidecarPath,
                            size_t blockIndex,
                            size_t matcherIndex)
{
    const std::string label = "[[supported]]#" + std::to_string(blockIndex)
                              + "/[[supported.matchers]]#" + std::to_string(matcherIndex) + " in "
                              + sidecarPath.string();

    const auto* table = node.as_table();
    if(table == nullptr)
    {
        throw std::runtime_error("SupportClaims: " + label + " is not a table");
    }

    SupportMatcher matcher;
    matcher.sourceLocation = label;
    matcher.opChains = parseRequiredStringArray(*table, "op_chains", label);
    matcher.layouts = parseRequiredStringArray(*table, "layouts", label);

    // Catch older schemas with their previous fields so the engineer
    // gets a clear "regenerate" message instead of a confusing "missing
    // dtype_combos" one.
    for(const char* obsoleteKey : {"io_dtypes", "io_dtype_pairs"})
    {
        if((*table)[obsoleteKey].as_array() != nullptr)
        {
            throw std::runtime_error(
                "SupportClaims: " + label + " uses the obsolete '" + std::string(obsoleteKey)
                + "' field. The sidecar predates the named-field dtype_combos format — "
                  "regenerate via --write-support-claims to produce dtype_combos inline-table "
                  "entries.");
        }
    }

    const auto* combosArray = (*table)["dtype_combos"].as_array();
    if(combosArray == nullptr)
    {
        throw std::runtime_error("SupportClaims: " + label
                                 + " missing required 'dtype_combos' array of inline tables "
                                   "(e.g. dtype_combos = [{io=\"fp16\", compute=\"fp32\"}, ...])");
    }
    if(combosArray->empty())
    {
        throw std::runtime_error("SupportClaims: " + label + " 'dtype_combos' must not be empty");
    }
    std::set<DtypeCombo> seenCombos;
    for(size_t i = 0; i < combosArray->size(); ++i)
    {
        const auto* comboTable = combosArray->at(i).as_table();
        if(comboTable == nullptr)
        {
            throw std::runtime_error("SupportClaims: " + label + " dtype_combos["
                                     + std::to_string(i) + "] must be an inline table");
        }
        auto combo = parseDtypeCombo(*comboTable, label, i);
        if(!seenCombos.insert(combo).second)
        {
            throw std::runtime_error(
                "SupportClaims: " + label + " dtype_combos[" + std::to_string(i)
                + "] duplicates an earlier entry (compared by io/output/compute/intermediate)");
        }
        matcher.dtypeCombos.push_back(std::move(combo));
    }

    rejectWildcards(matcher.opChains, "op_chains", label);
    rejectWildcards(matcher.layouts, "layouts", label);
    return matcher;
}

SupportBlock
    parseBlock(const toml::node& node, const std::filesystem::path& sidecarPath, size_t blockIndex)
{
    const std::string sectionLabel
        = "[[supported]]#" + std::to_string(blockIndex) + " in " + sidecarPath.string();

    const auto* table = node.as_table();
    if(table == nullptr)
    {
        throw std::runtime_error("SupportClaims: " + sectionLabel + " is not a table");
    }

    SupportBlock block;
    block.sourceLocation = sectionLabel;

    auto arch = (*table)["arch"].value<std::string>();
    if(!arch.has_value() || arch->empty())
    {
        throw std::runtime_error("SupportClaims: " + sectionLabel + " missing required 'arch'");
    }
    if(arch->find('*') != std::string::npos)
    {
        throw std::runtime_error("SupportClaims: " + sectionLabel
                                 + " 'arch' must not contain '*' "
                                   "(use one [[supported]] block per arch)");
    }
    block.arch = std::move(*arch);

    if(auto plat = (*table)["platform"].value<std::string>(); plat.has_value())
    {
        if(*plat != "windows" && *plat != "linux")
        {
            throw std::runtime_error("SupportClaims: " + sectionLabel + " 'platform' = '" + *plat
                                     + "' is not 'windows' or 'linux'");
        }
        block.platform = std::move(*plat);
    }

    const auto* matchers = (*table)["matchers"].as_array();
    if(matchers == nullptr)
    {
        // Block with zero matchers is legal but vacuous — it claims
        // nothing. Allowed so the auto-gen tool can emit a stub for a
        // new asic without a separate "is there anything?" check.
        return block;
    }

    for(size_t m = 0; m < matchers->size(); ++m)
    {
        block.matchers.push_back(parseMatcher(matchers->at(m), sidecarPath, blockIndex, m));
    }
    return block;
}

} // namespace

SupportClaims::SupportClaims(const std::filesystem::path& sidecarPath,
                             std::string_view expectedEngineName)
    : _path(sidecarPath)
{
    auto table = toml::parse_file(sidecarPath.string());

    auto version = table["meta"]["version"].value<int64_t>();
    if(!version.has_value())
    {
        throw std::runtime_error("SupportClaims: missing [meta].version in "
                                 + sidecarPath.string());
    }
    // RFC 0012 §5: sidecar version is decoupled from the main TOML's
    // version. Version history:
    //   v1 — initial schema with op_chain bare node names.
    //   v2 — extended op_chain with per-node :variant tags (e.g.
    //        Pointwise:RELU_FWD[lower_clip]).
    //   v3 — added asymmetric io_dtype_pairs alongside symmetric
    //        io_dtypes shorthand.
    //   v4 — collapsed to io_dtype_pairs string form ("in->out").
    //   v5 — replaced io_dtype_pairs strings with dtype_combos
    //        inline-table arrays carrying named fields:
    //          { io, output?, compute, intermediate? }
    //        Schema mirrors the support-matrix markdown display, can
    //        in fact serve as the source of truth that renders it,
    //        and gives compute/intermediate first-class matcher key
    //        status (the engine actually dispatches on them).
    //   v6 — extended describeNodeVariant() with shape-flag tags on
    //        Conv (1x1, grouped, multi_batch, non_square, padding,
    //        stride, dilation) and Batchnorm-family (multi_batch).
    //        op_chain strings now read e.g. "ConvFprop[1x1,grouped]"
    //        — engines that partition support along these shape axes
    //        (hipblaslt only handling 1x1, hip-kernel skipping
    //        grouped/dilated) can record distinct matcher rectangles
    //        instead of over-claiming via the bare node type.
    // Older readers can't tell that the format changed and would
    // silently miss matchers, so the safe contract is refuse-and-regen.
    if(*version != 6)
    {
        throw std::runtime_error("SupportClaims: unsupported version " + std::to_string(*version)
                                 + " in " + sidecarPath.string()
                                 + " (expected 6; older sidecars predate the conv/batchnorm "
                                   "shape-variant op_chain tags and need regeneration via "
                                   "--write-support-claims)");
    }

    auto engine = table["meta"]["engine"].value<std::string>();
    if(!engine.has_value() || engine->empty())
    {
        throw std::runtime_error("SupportClaims: missing [meta].engine in " + sidecarPath.string()
                                 + " (required when sidecar is in use; see RFC 0012 §5.1)");
    }
    _engineName = std::move(*engine);
    if(_engineName != expectedEngineName)
    {
        throw std::runtime_error("SupportClaims: [meta].engine mismatch in " + sidecarPath.string()
                                 + " (file declares '" + _engineName + "', expected '"
                                 + std::string(expectedEngineName) + "')");
    }

    if(auto* blocks = table["supported"].as_array())
    {
        for(size_t i = 0; i < blocks->size(); ++i)
        {
            _blocks.push_back(parseBlock(blocks->at(i), sidecarPath, i));
        }
    }
}

const SupportBlock* SupportClaims::blockFor(std::string_view archToken,
                                            std::string_view platform) const
{
    for(const auto& block : _blocks)
    {
        if(block.arch != archToken)
        {
            continue;
        }
        if(block.platform.has_value() && *block.platform != platform)
        {
            continue;
        }
        return &block;
    }
    return nullptr;
}

bool SupportClaims::isClaimed(std::string_view archToken,
                              std::string_view platform,
                              std::string_view opChain,
                              std::string_view inputDtype,
                              std::string_view outputDtype,
                              std::string_view computeDtype,
                              std::string_view intermediateDtype,
                              std::string_view layout) const
{
    const auto* block = blockFor(archToken, platform);
    if(block == nullptr)
    {
        return false;
    }
    return std::any_of(
        block->matchers.begin(), block->matchers.end(), [&](const SupportMatcher& matcher) {
            return matcher.contains(
                opChain, inputDtype, outputDtype, computeDtype, intermediateDtype, layout);
        });
}

} // namespace hipdnn_integration_tests
