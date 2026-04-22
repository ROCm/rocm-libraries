// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "common/PlatformUtils.hpp"

// HIP's host_defines.h redefines __noinline__ as either
// __attribute__((noinline)) or empty, which collides with tomlplusplus's
// TOML_HAS_ATTR(__noinline__) macro.  Temporarily undefine it around the
// toml.hpp include so the tomlplusplus preprocessor logic compiles cleanly
// under clang-tidy (which processes #if operands before short-circuit).
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

struct ToleranceOverride
{
    float atol;
    float rtol;
};

// Loads a per-engine TOML settings file for integration tests.
// Currently supports tolerance overrides; additional settings (knobs,
// support matrix, etc.) will be added in future versions.
//
// TOML format:
//   [meta]
//   version = 1
//
//   [[tolerance_overrides]]
//   filters = ["*convolution*fp16*"]
//   atol = 1e-3
//   rtol = 1e-2
//
// Filters use GTest-style globs (globMatch with * wildcards).
// Later entries take precedence over earlier ones when multiple filters match.
class TestSettings
{
public:
    // Load config from a TOML file. Throws on parse errors or invalid format.
    explicit TestSettings(const std::filesystem::path& configPath)
    {
        auto table = toml::parse_file(configPath.string());

        // Validate meta.version
        auto version = table["meta"]["version"].value<int64_t>();
        if(!version.has_value())
        {
            throw std::runtime_error("TestSettings: missing [meta].version in "
                                     + configPath.string());
        }
        if(*version != 1)
        {
            throw std::runtime_error("TestSettings: unsupported version " + std::to_string(*version)
                                     + " in " + configPath.string() + " (expected 1)");
        }

        // Parse [[tolerance_overrides]]
        if(auto overrides = table["tolerance_overrides"].as_array())
        {
            for(const auto& entry : *overrides)
            {
                auto* entryTable = entry.as_table();
                if(entryTable == nullptr)
                {
                    throw std::runtime_error(
                        "TestSettings: [[tolerance_overrides]] entry is not a table");
                }

                OverrideEntry parsed;

                // Parse filters array
                auto* filtersArray = (*entryTable)["filters"].as_array();
                if(filtersArray == nullptr || filtersArray->empty())
                {
                    throw std::runtime_error(
                        "TestSettings: [[tolerance_overrides]] entry missing 'filters' array");
                }
                for(const auto& filter : *filtersArray)
                {
                    auto filterStr = filter.value<std::string>();
                    if(!filterStr.has_value())
                    {
                        throw std::runtime_error("TestSettings: filter value must be a string");
                    }
                    parsed.filters.push_back(*filterStr);
                }

                // Parse atol
                auto atol = (*entryTable)["atol"].value<double>();
                if(!atol.has_value())
                {
                    throw std::runtime_error(
                        "TestSettings: [[tolerance_overrides]] entry missing 'atol'");
                }
                parsed.atol = static_cast<float>(*atol);

                // Parse rtol
                auto rtol = (*entryTable)["rtol"].value<double>();
                if(!rtol.has_value())
                {
                    throw std::runtime_error(
                        "TestSettings: [[tolerance_overrides]] entry missing 'rtol'");
                }
                parsed.rtol = static_cast<float>(*rtol);

                _overrides.push_back(std::move(parsed));
            }
        }
    }

    // Find a tolerance override matching the given test name.
    // Returns the last matching override (later entries take precedence).
    // Returns std::nullopt if no filter matches.
    std::optional<ToleranceOverride> findToleranceOverride(std::string_view testName) const
    {
        std::optional<ToleranceOverride> result;
        const std::string testNameStr(testName);

        for(const auto& entry : _overrides)
        {
            for(const auto& filter : entry.filters)
            {
                if(globMatch(filter, testNameStr))
                {
                    result = ToleranceOverride{entry.atol, entry.rtol};
                    break; // matched this entry, continue to next for precedence
                }
            }
        }

        return result;
    }

    size_t toleranceOverrideCount() const
    {
        return _overrides.size();
    }

private:
    struct OverrideEntry
    {
        std::vector<std::string> filters;
        float atol;
        float rtol;
    };

    std::vector<OverrideEntry> _overrides;
};

} // namespace hipdnn_integration_tests
