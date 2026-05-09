// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
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
// Currently supports tolerance overrides and arch-scoped test skips;
// additional settings (knobs, support matrix, etc.) will be added in
// future versions.
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
//   [[test_skips]]
//   archs   = ["gfx1100", "gfx1101"]
//   filters = ["*Activation*Fp16*"]
//   reason  = "missing activation kernel on RDNA3"
//
//   # Global skip (omit 'archs' to skip on every arch)
//   [[test_skips]]
//   filters = ["*KnownBroken*"]
//   reason  = "tracked in #1234"
//
// Filters use GTest-style globs (globMatch with * wildcards).
// For tolerance_overrides, later entries take precedence over earlier
// ones when multiple filters match.
// For test_skips, an entry matches when:
//   - 'archs' is omitted/empty (global rule), OR any 'archs' value is a
//     substring of the device's raw gcnArchName, AND
//   - any 'filters' glob matches the gtest-formatted test name.
// The first matching entry wins.
class TestSettings
{
public:
    // Load config from a TOML file. Throws on parse errors or invalid format.
    explicit TestSettings(const std::filesystem::path& configPath)
    {
        auto table = toml::parse_file(configPath.string());
        validateMetaVersion(table, configPath);

        if(auto* arr = table["tolerance_overrides"].as_array())
        {
            for(const auto& node : *arr)
            {
                _overrides.push_back(parseToleranceOverride(node));
            }
        }

        if(auto* arr = table["test_skips"].as_array())
        {
            for(const auto& node : *arr)
            {
                _skips.push_back(parseTestSkip(node));
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

    // Find a skip rule matching the given test name on the given device.
    // deviceArchRaw is the unparsed gcnArchName string (e.g. "gfx942:sramecc+:xnack-").
    // An entry matches when:
    //   - the entry's 'archs' list is empty (global rule, matches every arch),
    //     OR any 'archs' value is a substring of deviceArchRaw, AND
    //   - any 'filters' glob matches testName.
    // First matching entry wins; returns its reason. Returns nullopt if no
    // entry matches.
    std::optional<std::string> findSkip(std::string_view testName,
                                        std::string_view deviceArchRaw) const
    {
        const std::string testNameStr(testName);
        const std::string deviceArchStr(deviceArchRaw);

        for(const auto& entry : _skips)
        {
            const bool archMatches
                = entry.archs.empty()
                  || std::any_of(
                      entry.archs.begin(), entry.archs.end(), [&](const std::string& candidate) {
                          return deviceArchStr.find(candidate) != std::string::npos;
                      });
            if(!archMatches)
            {
                continue;
            }

            const bool filterMatches
                = std::any_of(entry.filters.begin(),
                              entry.filters.end(),
                              [&](const std::string& f) { return globMatch(f, testNameStr); });
            if(filterMatches)
            {
                return entry.reason;
            }
        }

        return std::nullopt;
    }

    size_t skipEntryCount() const
    {
        return _skips.size();
    }

private:
    struct OverrideEntry
    {
        std::vector<std::string> filters;
        float atol;
        float rtol;
    };

    struct SkipEntry
    {
        std::vector<std::string> archs;
        std::vector<std::string> filters;
        std::string reason;
    };

    static void validateMetaVersion(const toml::table& table,
                                    const std::filesystem::path& configPath)
    {
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
    }

    // Parse a TOML string array. Returns the parsed strings, or an empty
    // vector if the key is missing. Throws if a value is present but is
    // not a string. Callers that require the array to be non-empty must
    // check the result themselves.
    static std::vector<std::string>
        parseStringArray(const toml::table& entry, std::string_view key, const char* sectionName)
    {
        std::vector<std::string> result;
        auto* arr = entry[key].as_array();
        if(arr == nullptr)
        {
            return result;
        }
        for(const auto& node : *arr)
        {
            auto str = node.value<std::string>();
            if(!str.has_value())
            {
                throw std::runtime_error(std::string(sectionName) + ": '" + std::string(key)
                                         + "' value must be a string");
            }
            result.push_back(*str);
        }
        return result;
    }

    static OverrideEntry parseToleranceOverride(const toml::node& node)
    {
        static constexpr const char* kSection = "TestSettings: [[tolerance_overrides]]";

        const auto* table = node.as_table();
        if(table == nullptr)
        {
            throw std::runtime_error(std::string(kSection) + " entry is not a table");
        }

        OverrideEntry parsed;
        parsed.filters = parseStringArray(*table, "filters", kSection);
        if(parsed.filters.empty())
        {
            throw std::runtime_error(std::string(kSection) + " entry missing 'filters' array");
        }

        auto atol = (*table)["atol"].value<double>();
        if(!atol.has_value())
        {
            throw std::runtime_error(std::string(kSection) + " entry missing 'atol'");
        }
        parsed.atol = static_cast<float>(*atol);

        auto rtol = (*table)["rtol"].value<double>();
        if(!rtol.has_value())
        {
            throw std::runtime_error(std::string(kSection) + " entry missing 'rtol'");
        }
        parsed.rtol = static_cast<float>(*rtol);

        return parsed;
    }

    static SkipEntry parseTestSkip(const toml::node& node)
    {
        static constexpr const char* kSection = "TestSettings: [[test_skips]]";

        const auto* table = node.as_table();
        if(table == nullptr)
        {
            throw std::runtime_error(std::string(kSection) + " entry is not a table");
        }

        SkipEntry parsed;
        // 'archs' is optional. Empty list = global rule that matches every arch.
        parsed.archs = parseStringArray(*table, "archs", kSection);
        parsed.filters = parseStringArray(*table, "filters", kSection);
        if(parsed.filters.empty())
        {
            throw std::runtime_error(std::string(kSection) + " entry missing 'filters' array");
        }

        auto reason = (*table)["reason"].value<std::string>();
        if(!reason.has_value() || reason->empty())
        {
            throw std::runtime_error(std::string(kSection) + " entry missing 'reason'");
        }
        parsed.reason = std::move(*reason);

        return parsed;
    }

    std::vector<OverrideEntry> _overrides;
    std::vector<SkipEntry> _skips;
};

} // namespace hipdnn_integration_tests
