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
    matcher.ioDtypes = parseRequiredStringArray(*table, "io_dtypes", label);
    matcher.layouts = parseRequiredStringArray(*table, "layouts", label);
    rejectWildcards(matcher.opChains, "op_chains", label);
    rejectWildcards(matcher.ioDtypes, "io_dtypes", label);
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
    // version. v2 bumped the op_chain string format to include per-node
    // variant tags (describeGraph extension) — v1 sidecars are stale
    // because their op_chain strings no longer match what verifyGraph
    // records. Refuse loudly rather than silently mis-evaluate.
    if(*version != 2)
    {
        throw std::runtime_error("SupportClaims: unsupported version " + std::to_string(*version)
                                 + " in " + sidecarPath.string()
                                 + " (expected 2; v1 sidecars predate the op_chain variant tag "
                                   "and need regeneration via --write-support-claims)");
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
                              std::string_view ioDtype,
                              std::string_view layout) const
{
    const auto* block = blockFor(archToken, platform);
    if(block == nullptr)
    {
        return false;
    }
    return std::any_of(
        block->matchers.begin(), block->matchers.end(), [&](const SupportMatcher& matcher) {
            return matcher.contains(opChain, ioDtype, layout);
        });
}

} // namespace hipdnn_integration_tests
