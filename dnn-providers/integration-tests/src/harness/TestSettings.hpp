// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_test_sdk/utilities/ArchMatch.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp>

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

/// The direction of a convolution; each has its own [conv_<direction>_tolerance] table in an
/// engine's settings file.
enum class ConvToleranceDirection
{
    FWD,
    DGRAD,
    WRW,
};

/// The top-level tables of the settings file, by ConvToleranceDirection.
inline constexpr std::array<std::string_view, 3> K_CONV_TOLERANCE_TABLES
    = {"conv_fwd_tolerance", "conv_dgrad_tolerance", "conv_wrw_tolerance"};

/// The output types a table has a sub-table for.
inline constexpr std::array<std::string_view, 3> K_CONV_TOLERANCE_TYPES
    = {"float", "half", "bfloat16"};

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
//   # Coefficients of the dynamic tolerance of a convolution output, one table per direction
//   # (conv_fwd_tolerance, conv_dgrad_tolerance, conv_wrw_tolerance) and one sub-table per
//   # output type (float, half, bfloat16)
//   [conv_wrw_tolerance.half]
//   lambda                    = 40.0
//   lambda_per_quartic_root_k = 1.5
//
//   # A table of the same keys named by an architecture applies to the devices that
//   # architecture matches, instead of the table above it
//   [conv_wrw_tolerance.half.gfx90a]
//   lambda                    = 60.0
//
//   [conv_fwd_tolerance.float]
//   summation_order = 9.0
//
// The tolerance model has an allowance for kernels that sum the K products in float and round
// once (the SDK's default, see ConvDynamicTolerance.hpp). An engine whose kernels need more says
// so here, for the direction and output type it applies to; every key is optional and an unset
// one keeps the SDK's value:
//   - 'summation_order' multiplies sqrt(K) * u32 in the summation-order term. float sub-table
//     only. A finite positive number.
//   - 'lambda' is the allowance for intermediates and partial results, in units of eps times
//     the root mean square of the output (the means of the inputs included). It replaces the
//     SDK's allowance of the direction and type, it does not add to it. A finite number, not
//     negative.
//   - 'lambda_per_quartic_root_k' is added to 'lambda' per K^(1/4), for kernels that split a
//     longer reduction into more parts. A finite number, not negative, and only together with
//     'lambda'.
// A table named by an architecture, nested under an output type's table, holds the same keys
// for the devices whose raw gcnArchName contains that name (the match of 'archs' in
// [[test_skips]]: "gfx90a" matches "gfx90a:sramecc+:xnack-"). It replaces the coefficients of
// the output type's table whole: a key it leaves unset is the SDK's value, not the output
// type's. A device that no arch table matches takes the output type's table. A device that two
// arch tables of one output type match is an error naming both, raised when the tolerance is
// looked up, since the device is not known before. No table nests under an arch table.
// An output type other than float, half or bfloat16, a key other than these three that is not an
// arch table, a key of the wrong type or value, 'summation_order' outside the float sub-table,
// and 'lambda_per_quartic_root_k' without 'lambda' are load errors. A matching
// [[tolerance_overrides]] entry still replaces the dynamic tolerance as a whole. Only the bundle
// harness reads these tables.
//
// A top-level key other than meta, tolerance_overrides, validator_overrides, test_skips,
// conv_fwd_tolerance, conv_dgrad_tolerance and conv_wrw_tolerance is a load error, so a misspelt
// section name cannot go unnoticed.
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
        rejectUnknownTopLevelKeys(table, configPath);

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

        for(const auto direction : {ConvToleranceDirection::FWD,
                                    ConvToleranceDirection::DGRAD,
                                    ConvToleranceDirection::WRW})
        {
            if(const auto convTolerance
               = table[K_CONV_TOLERANCE_TABLES[static_cast<size_t>(direction)]])
            {
                parseConvTolerance(*convTolerance.node(), direction);
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
                result = ValidatorOverride{entry.kind, entry.rmsThreshold};
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

    // The coefficients of the dynamic tolerance of a convolution of `direction` whose output is
    // of `outputType` ("float", "half" or "bfloat16") on a device whose raw gcnArchName is `arch`
    // (e.g. "gfx942:sramecc+:xnack-") that this engine's file sets; every one unset, which keeps
    // the default of the harness, for another type or when it sets none. The table of an
    // architecture that `arch` contains replaces the output type's own table whole. Throws when
    // two arch tables of the output type both match `arch`: a file cannot be checked for that
    // before the device is known.
    hipdnn_test_sdk::utilities::conv::ConvToleranceCoefficients convToleranceCoefficients(
        ConvToleranceDirection direction, std::string_view outputType, std::string_view arch) const
    {
        const auto typeIt
            = std::find(K_CONV_TOLERANCE_TYPES.begin(), K_CONV_TOLERANCE_TYPES.end(), outputType);
        if(typeIt == K_CONV_TOLERANCE_TYPES.end())
        {
            return {};
        }
        const auto& table
            = _convTolerance[static_cast<size_t>(direction)]
                            [static_cast<size_t>(typeIt - K_CONV_TOLERANCE_TYPES.begin())];
        const std::pair<std::string, hipdnn_test_sdk::utilities::conv::ConvToleranceCoefficients>*
            matched
            = nullptr;
        for(const auto& entry : table.archTables)
        {
            if(!hipdnn_test_sdk::utilities::archMatches(
                   arch, entry.first, hipdnn_test_sdk::utilities::ArchMatchMode::SUBSTRING))
            {
                continue;
            }
            if(matched != nullptr)
            {
                const std::string prefix
                    = "[" + std::string(K_CONV_TOLERANCE_TABLES[static_cast<size_t>(direction)])
                      + "." + std::string(outputType) + ".";
                throw std::runtime_error("TestSettings: " + prefix + matched->first + "] and "
                                         + prefix + entry.first + "] both match the device arch '"
                                         + std::string(arch) + "'; make the arch names exclusive");
            }
            matched = &entry;
        }
        return matched != nullptr ? matched->second : table.coefficients;
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

    // A top-level key the loader does not read is a load error, so that a misspelt section name
    // ([conv_fwd_tolerence.float]) cannot leave its setting silently out of a file.
    static void rejectUnknownTopLevelKeys(const toml::table& table,
                                          const std::filesystem::path& configPath)
    {
        static constexpr std::array<std::string_view, 7> K_KNOWN = {"meta",
                                                                    "tolerance_overrides",
                                                                    "validator_overrides",
                                                                    "test_skips",
                                                                    "conv_fwd_tolerance",
                                                                    "conv_dgrad_tolerance",
                                                                    "conv_wrw_tolerance"};
        for(const auto& entry : table)
        {
            const std::string_view key = entry.first.str();
            if(std::find(K_KNOWN.begin(), K_KNOWN.end(), key) == K_KNOWN.end())
            {
                throw std::runtime_error("TestSettings: unknown top-level key '" + std::string(key)
                                         + "' in " + configPath.string()
                                         + " (expected meta, tolerance_overrides, "
                                           "validator_overrides, test_skips, conv_fwd_tolerance, "
                                           "conv_dgrad_tolerance or conv_wrw_tolerance)");
            }
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
            return parsed;
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

    // The keys of one table of coefficients: [conv_<direction>_tolerance.<output type>], or a
    // table of one architecture under it. Only the keys of ConvToleranceCoefficients exist,
    // 'summation_order' for float only; anything else is a load error, so that a coefficient for
    // a term the harness does not read cannot sit in a file unnoticed. A table-valued key that is
    // not one of those keys is an arch table: it is handed to `archTables` when the caller takes
    // them, and an error when it does not (no table nests under an arch table).
    static hipdnn_test_sdk::utilities::conv::ConvToleranceCoefficients
        parseConvCoefficients(const toml::table& table,
                              const std::string& where,
                              bool isFloat,
                              std::vector<std::pair<std::string, const toml::table*>>* archTables)
    {
        hipdnn_test_sdk::utilities::conv::ConvToleranceCoefficients parsed;
        for(const auto& [key, value] : table)
        {
            const std::string keyName(key.str());
            const bool isCoefficient = keyName == "summation_order" || keyName == "lambda"
                                       || keyName == "lambda_per_quartic_root_k";
            if(!isCoefficient && value.is_table())
            {
                if(archTables == nullptr)
                {
                    throw std::runtime_error(where + " has a nested table '" + keyName
                                             + "': a table named by an arch holds coefficients "
                                               "only");
                }
                if(keyName.empty())
                {
                    throw std::runtime_error(where + " has an arch table with an empty name");
                }
                archTables->emplace_back(keyName, value.as_table());
                continue;
            }
            if(!isCoefficient)
            {
                throw std::runtime_error(where + " has unknown key '" + keyName
                                         + "' (expected 'summation_order', 'lambda', "
                                           "'lambda_per_quartic_root_k' or a table named by an "
                                           "arch)");
            }
            const auto number = value.value<double>();
            if(!number.has_value())
            {
                throw std::runtime_error(where + " '" + keyName + "' must be a number");
            }
            if(keyName == "summation_order")
            {
                if(!isFloat)
                {
                    throw std::runtime_error(where
                                             + " 'summation_order' is for float outputs only");
                }
                if(!std::isfinite(*number) || !(*number > 0.0))
                {
                    throw std::runtime_error(where
                                             + " 'summation_order' must be finite and positive");
                }
                parsed.summationOrder = number;
                continue;
            }
            if(!std::isfinite(*number) || *number < 0.0)
            {
                throw std::runtime_error(where + " '" + keyName
                                         + "' must be finite and not negative");
            }
            (keyName == "lambda" ? parsed.lambda : parsed.lambdaPerQuarticRootK) = number;
        }
        if(parsed.lambdaPerQuarticRootK.has_value() && !parsed.lambda.has_value())
        {
            throw std::runtime_error(
                where + " 'lambda_per_quartic_root_k' needs 'lambda': it adds to it");
        }
        return parsed;
    }

    // [conv_<direction>_tolerance.<output type>] sets the coefficients of the dynamic tolerance
    // of a convolution of that direction whose output is of that type, and the arch tables under
    // it those of the devices of that architecture. Only the three output types exist; anything
    // else is a load error.
    void parseConvTolerance(const toml::node& node, ConvToleranceDirection direction)
    {
        const std::string_view tableName = K_CONV_TOLERANCE_TABLES[static_cast<size_t>(direction)];
        const std::string section = "TestSettings: [" + std::string(tableName) + "]";

        const auto* table = node.as_table();
        if(table == nullptr)
        {
            throw std::runtime_error(section + " is not a table");
        }
        for(const auto& [typeName, typeNode] : *table)
        {
            const auto typeIt = std::find(
                K_CONV_TOLERANCE_TYPES.begin(), K_CONV_TOLERANCE_TYPES.end(), typeName.str());
            if(typeIt == K_CONV_TOLERANCE_TYPES.end())
            {
                throw std::runtime_error(section + " has unknown output type '"
                                         + std::string(typeName.str())
                                         + R"~(' (expected "float", "half" or "bfloat16"))~");
            }
            const std::string where = "TestSettings: [" + std::string(tableName) + "."
                                      + std::string(typeName.str()) + "]";
            const auto* typeTable = typeNode.as_table();
            if(typeTable == nullptr)
            {
                throw std::runtime_error(where + " is not a table");
            }

            const bool isFloat = *typeIt == "float";
            std::vector<std::pair<std::string, const toml::table*>> archTables;
            ConvToleranceTable parsed;
            parsed.coefficients = parseConvCoefficients(*typeTable, where, isFloat, &archTables);
            for(const auto& [arch, archTable] : archTables)
            {
                parsed.archTables.emplace_back(
                    arch,
                    parseConvCoefficients(*archTable,
                                          where.substr(0, where.size() - 1) + "." + arch + "]",
                                          isFloat,
                                          nullptr));
            }
            _convTolerance[static_cast<size_t>(direction)]
                          [static_cast<size_t>(typeIt - K_CONV_TOLERANCE_TYPES.begin())]
                = std::move(parsed);
        }
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
    // One [conv_<direction>_tolerance.<output type>] table: its own coefficients, and the
    // coefficients of each arch table under it by the arch name.
    struct ConvToleranceTable
    {
        hipdnn_test_sdk::utilities::conv::ConvToleranceCoefficients coefficients;
        std::vector<
            std::pair<std::string, hipdnn_test_sdk::utilities::conv::ConvToleranceCoefficients>>
            archTables;
    };

    std::array<std::array<ConvToleranceTable, K_CONV_TOLERANCE_TYPES.size()>,
               K_CONV_TOLERANCE_TABLES.size()>
        _convTolerance{};
};

} // namespace hipdnn_integration_tests
