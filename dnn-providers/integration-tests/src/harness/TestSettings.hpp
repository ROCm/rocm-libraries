// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_test_sdk/utilities/ArchMatch.hpp>

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

// Which comparison an output tensor is graded by. Mirrors
// bundle::ValidatorKind, kept separate so TestSettings stays free of the
// comparison headers.
enum class ValidatorOverrideKind
{
    ALLCLOSE,
    RMS,
    ALLCLOSE_MATCHING_INFINITIES,
};

struct ValidatorOverride
{
    ValidatorOverrideKind kind;
    // Relative-RMS threshold. Meaningful only when kind == RMS.
    float rmsThreshold;
    // Per-tensor tolerance of an allclose or allclose_matching_infinities entry. Each is
    // present only when the entry spells it; a present value outranks the matching
    // [[tolerance_overrides]] entry and the policy default for that tensor. Always empty for
    // RMS, which the parser enforces.
    std::optional<float> atol = std::nullopt;
    std::optional<float> rtol = std::nullopt;
};

// Loads a per-engine TOML settings file for integration tests.
// Currently supports tolerance overrides, validator overrides and arch-scoped
// test skips; additional settings (knobs, support matrix, etc.) will be added
// in future versions.
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
//   [[validator_overrides]]
//   filters       = ["*LayernormBackward*"]
//   tensors       = ["*::DSCALE", "*::DBIAS"]
//   validator     = "rms"
//   rms_threshold = 1e-4
//
//   [[validator_overrides]]
//   filters   = ["*SdpaFwd*Masked*"]
//   tensors   = ["*::LSE"]
//   validator = "allclose_matching_infinities"
//
//   [[validator_overrides]]
//   filters   = ["*gauntlet*BatchnormBackward*"]
//   tensors   = ["*::DX"]
//   validator = "allclose"
//   atol      = 4e-4      # optional, with or without rtol
//   rtol      = 2e-3
//
//   [[test_skips]]
//   archs     = ["gfx1100", "gfx1101"]
//   platforms = ["windows"]                       # optional
//   filters   = ["*Activation*Fp16*"]
//   reason    = "missing activation kernel on Windows + RDNA3"
//
//   # Global skip (omit 'archs' and 'platforms' to skip everywhere)
//   [[test_skips]]
//   filters = ["*KnownBroken*"]
//   reason  = "tracked in #1234"
//
// Filters use GTest-style globs (globMatch with * wildcards).
// For tolerance_overrides, later entries take precedence over earlier
// ones when multiple filters match.
// For validator_overrides, an entry applies when a 'filters' glob matches the
// gtest-formatted test name AND a 'tensors' glob matches the output tensor's
// label (its name, or "uid=N" when the graph did not name it). Later entries
// take precedence. Absent any match, allclose is used — allclose is the default
// and nothing else can select a validator.
// 'validator' is one of "allclose", "allclose_matching_infinities" or "rms".
// "allclose_matching_infinities" grades exactly as "allclose" does, at the same
// resolved atol/rtol, except that an element that is infinite with the same sign in
// both the reference and the device output compares equal; NaN, opposite-signed
// infinities and finite-versus-infinite disagreements still fail. It is defined for
// float, half, bfloat16 and double outputs only. In a C++ graph test it exists only
// as a host validator: a comparison that runs on the device (--validator gpu, or auto
// when the GPU reference produced the expected values) fails the tensor, so a run
// grading with it needs --validator cpu. A bundle test grades it on either site.
// Validator entries are parsed strictly: 'rms' requires a positive 'rms_threshold',
// and neither "allclose" nor "allclose_matching_infinities" may carry one, because
// an entry that does not say exactly what it means is a load error rather than a
// silent fall-back to allclose.
// "allclose" and "allclose_matching_infinities" entries may also set 'atol', 'rtol' or
// both: a finite, non-negative number that replaces that tolerance for the tensors the
// entry matches. It outranks a [[tolerance_overrides]] entry for the same test, which
// is the one way to relax one output of a graph and leave its siblings on the default
// (a tolerance override applies to every tensor of the test). A key an entry omits
// leaves that tolerance to the tolerance override or the default. 'rms' with 'atol' or
// 'rtol' is a load error, since an RMS comparison reads neither. When several entries
// match one tensor the last one wins as a whole: its omitted keys do not inherit an
// earlier entry's.
// For test_skips, an entry matches when ALL of:
//   - 'archs' is omitted/empty (any arch), OR any 'archs' value is a
//     substring of the device's raw gcnArchName.
//   - 'platforms' is omitted/empty (any platform), OR any 'platforms'
//     value exactly equals the current platform ("windows" or "linux").
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

        if(auto* arr = table["validator_overrides"].as_array())
        {
            for(const auto& node : *arr)
            {
                _validatorOverrides.push_back(parseValidatorOverride(node));
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

    // Find a validator override for one output tensor of one test.
    // Returns the last matching entry (later entries take precedence), or
    // std::nullopt when nothing matches — which means allclose.
    std::optional<ValidatorOverride> findValidatorOverride(std::string_view testName,
                                                           std::string_view tensorLabel) const
    {
        std::optional<ValidatorOverride> result;
        const std::string testNameStr(testName);
        const std::string tensorLabelStr(tensorLabel);

        for(const auto& entry : _validatorOverrides)
        {
            const bool testMatches
                = std::any_of(entry.filters.begin(), entry.filters.end(), [&](const auto& f) {
                      return globMatch(f, testNameStr);
                  });
            if(!testMatches)
            {
                continue;
            }
            const bool tensorMatches
                = std::any_of(entry.tensors.begin(), entry.tensors.end(), [&](const auto& t) {
                      return globMatch(t, tensorLabelStr);
                  });
            if(tensorMatches)
            {
                result = ValidatorOverride{entry.kind, entry.rmsThreshold, entry.atol, entry.rtol};
            }
        }

        return result;
    }

    size_t validatorOverrideCount() const
    {
        return _validatorOverrides.size();
    }

    // Find a skip rule matching the given test name, device, and platform.
    // deviceArchRaw is the unparsed gcnArchName string (e.g. "gfx942:sramecc+:xnack-").
    // platform is the lowercase platform name ("windows" or "linux").
    // An entry matches when:
    //   - the entry's 'archs' list is empty (any arch), OR any 'archs' value
    //     is a substring of deviceArchRaw, AND
    //   - the entry's 'platforms' list is empty (any platform), OR any
    //     'platforms' value exactly equals platform, AND
    //   - any 'filters' glob matches testName.
    // First matching entry wins; returns its reason. Returns nullopt if no
    // entry matches.
    std::optional<std::string> findSkip(std::string_view testName,
                                        std::string_view deviceArchRaw,
                                        std::string_view platform) const
    {
        const std::string testNameStr(testName);
        const std::string deviceArchStr(deviceArchRaw);
        const std::string platformStr(platform);

        for(const auto& entry : _skips)
        {
            // Loose match: a skip entry's arch may be a family prefix (e.g.
            // "gfx10" covers all gfx10xx), matched as a substring of the
            // device's raw gcnArchName.
            const bool archMatched
                = entry.archs.empty()
                  || std::any_of(
                      entry.archs.begin(), entry.archs.end(), [&](const std::string& candidate) {
                          return hipdnn_test_sdk::utilities::archMatches(
                              deviceArchStr,
                              candidate,
                              hipdnn_test_sdk::utilities::ArchMatchMode::SUBSTRING);
                      });
            if(!archMatched)
            {
                continue;
            }

            const bool platformMatches
                = entry.platforms.empty()
                  || std::any_of(entry.platforms.begin(),
                                 entry.platforms.end(),
                                 [&](const std::string& p) { return p == platformStr; });
            if(!platformMatches)
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

    struct ValidatorEntry
    {
        std::vector<std::string> filters;
        std::vector<std::string> tensors;
        ValidatorOverrideKind kind;
        float rmsThreshold;
        std::optional<float> atol;
        std::optional<float> rtol;
    };

    struct SkipEntry
    {
        std::vector<std::string> archs;
        std::vector<std::string> platforms;
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

    // The finite, non-negative float at `key`, or nullopt when the key is absent. Throws when
    // it is present and not a number, NaN, infinite or outside the float range, or negative:
    // a tolerance that cannot be applied is a load error, never a fall-back to the default.
    static std::optional<float>
        parseTolerance(const toml::table& entry, std::string_view key, const char* sectionName)
    {
        const auto node = entry[key];
        if(!node)
        {
            return std::nullopt;
        }
        const auto value = node.value<double>();
        if(!value.has_value())
        {
            throw std::runtime_error(std::string(sectionName) + " entry '" + std::string(key)
                                     + "' must be a number");
        }
        // Checked on the double: narrowing one beyond the float range is undefined behavior.
        if(!std::isfinite(*value) || *value < 0.0
           || *value > static_cast<double>(std::numeric_limits<float>::max()))
        {
            throw std::runtime_error(std::string(sectionName) + " entry '" + std::string(key)
                                     + "' must be finite and non-negative");
        }
        return static_cast<float>(*value);
    }

    static OverrideEntry parseToleranceOverride(const toml::node& node)
    {
        static constexpr const char* K_SECTION = "TestSettings: [[tolerance_overrides]]";

        const auto* table = node.as_table();
        if(table == nullptr)
        {
            throw std::runtime_error(std::string(K_SECTION) + " entry is not a table");
        }

        OverrideEntry parsed;
        parsed.filters = parseStringArray(*table, "filters", K_SECTION);
        if(parsed.filters.empty())
        {
            throw std::runtime_error(std::string(K_SECTION) + " entry missing 'filters' array");
        }

        auto atol = (*table)["atol"].value<double>();
        if(!atol.has_value())
        {
            throw std::runtime_error(std::string(K_SECTION) + " entry missing 'atol'");
        }
        parsed.atol = static_cast<float>(*atol);

        auto rtol = (*table)["rtol"].value<double>();
        if(!rtol.has_value())
        {
            throw std::runtime_error(std::string(K_SECTION) + " entry missing 'rtol'");
        }
        parsed.rtol = static_cast<float>(*rtol);

        return parsed;
    }

    static ValidatorEntry parseValidatorOverride(const toml::node& node)
    {
        static constexpr const char* K_SECTION = "TestSettings: [[validator_overrides]]";

        const auto* table = node.as_table();
        if(table == nullptr)
        {
            throw std::runtime_error(std::string(K_SECTION) + " entry is not a table");
        }

        ValidatorEntry parsed;
        parsed.filters = parseStringArray(*table, "filters", K_SECTION);
        if(parsed.filters.empty())
        {
            throw std::runtime_error(std::string(K_SECTION) + " entry missing 'filters' array");
        }

        parsed.tensors = parseStringArray(*table, "tensors", K_SECTION);
        if(parsed.tensors.empty())
        {
            throw std::runtime_error(std::string(K_SECTION)
                                     + " entry missing 'tensors' array — a validator override "
                                       "names the tensors it applies to, never a whole test");
        }

        auto validator = (*table)["validator"].value<std::string>();
        if(!validator.has_value())
        {
            throw std::runtime_error(std::string(K_SECTION) + " entry missing 'validator'");
        }

        const bool isAllClose = (*validator == "allclose");
        const bool isMatchingInfinities = (*validator == "allclose_matching_infinities");
        const bool isRms = (*validator == "rms");

        // The name is checked before its content. Hoisting the threshold guard above
        // this would answer an unrecognised name with a complaint about
        // 'rms_threshold', sending an operator to fix the line that is not wrong.
        if(!isAllClose && !isMatchingInfinities && !isRms)
        {
            throw std::runtime_error(
                std::string(K_SECTION) + " entry has unknown validator '" + *validator
                + R"~(' (expected "allclose", "allclose_matching_infinities" or "rms"))~");
        }

        if(!isRms)
        {
            // A threshold on an entry whose validator has none means the file does not
            // say what its author meant — either 'validator' was edited and the
            // threshold left behind, or the reverse. Ignoring it silently is how an
            // override quietly stops being the check someone thinks it is.
            if((*table)["rms_threshold"])
            {
                throw std::runtime_error(std::string(K_SECTION) + R"~( entry with validator = ")~"
                                         + *validator + R"~(" also sets 'rms_threshold')~");
            }
            parsed.kind = isAllClose ? ValidatorOverrideKind::ALLCLOSE
                                     : ValidatorOverrideKind::ALLCLOSE_MATCHING_INFINITIES;
            parsed.rmsThreshold = 0.0f;
            parsed.atol = parseTolerance(*table, "atol", K_SECTION);
            parsed.rtol = parseTolerance(*table, "rtol", K_SECTION);
            return parsed;
        }

        // The mirror of the guard above: an RMS comparison reads neither tolerance, so one
        // here is an entry that does not do what it says.
        for(const char* key : {"atol", "rtol"})
        {
            if((*table)[key])
            {
                throw std::runtime_error(std::string(K_SECTION)
                                         + R"~( entry with validator = "rms" also sets ')~" + key
                                         + "'");
            }
        }

        parsed.kind = ValidatorOverrideKind::RMS;
        auto threshold = (*table)["rms_threshold"].value<double>();
        if(!threshold.has_value())
        {
            throw std::runtime_error(
                std::string(K_SECTION)
                + R"~( entry with validator = "rms" missing 'rms_threshold')~");
        }
        if(!(*threshold > 0.0))
        {
            throw std::runtime_error(std::string(K_SECTION) + " entry has non-positive "
                                     + "'rms_threshold'");
        }
        parsed.rmsThreshold = static_cast<float>(*threshold);

        return parsed;
    }

    static SkipEntry parseTestSkip(const toml::node& node)
    {
        static constexpr const char* K_SECTION = "TestSettings: [[test_skips]]";

        const auto* table = node.as_table();
        if(table == nullptr)
        {
            throw std::runtime_error(std::string(K_SECTION) + " entry is not a table");
        }

        SkipEntry parsed;
        // 'archs' and 'platforms' are both optional. Empty = matches any.
        parsed.archs = parseStringArray(*table, "archs", K_SECTION);
        parsed.platforms = parseStringArray(*table, "platforms", K_SECTION);
        parsed.filters = parseStringArray(*table, "filters", K_SECTION);
        if(parsed.filters.empty())
        {
            throw std::runtime_error(std::string(K_SECTION) + " entry missing 'filters' array");
        }

        auto reason = (*table)["reason"].value<std::string>();
        if(!reason.has_value() || reason->empty())
        {
            throw std::runtime_error(std::string(K_SECTION) + " entry missing 'reason'");
        }
        parsed.reason = std::move(*reason);

        return parsed;
    }

    std::vector<OverrideEntry> _overrides;
    std::vector<ValidatorEntry> _validatorOverrides;
    std::vector<SkipEntry> _skips;
};

} // namespace hipdnn_integration_tests
