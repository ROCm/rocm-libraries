// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the client string<->enum helpers declared in
// hipblaslt_datatype2string.hpp. Nothing here touches a device, so this file is
// compiled into both hipblaslt-test (where it runs in the existing CI lane) and
// the standalone hipblaslt-client-unit-tests binary (which needs no GPU).
//
// Suites are prefixed `HostUnit` so a single pattern in
// clients/tests/test_categories.yaml selects all of them for every ctest tier.
//
// Motivated by the PR #6514 review (davidd-amd), observation #2 ("Do we have a
// plan to start adding unit tests for functions that we are changing that can be
// unit tested?"). Tracked as AIHPBLAS-3550.

#include "hipblaslt_datatype2string.hpp"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace
{
    template <typename Enum>
    using NameTable = std::vector<std::pair<std::string, Enum>>;

    // The marker every *_to_string helper in hipblaslt_datatype2string.hpp
    // returns for a value it does not recognize.
    constexpr const char* kUnknownName = "invalid";

    // Every init string the client accepts today, paired with its enum value.
    // Keep this in sync with string2hipblaslt_initialization().
    const NameTable<hipblaslt_initialization>& known_init_modes()
    {
        static const NameTable<hipblaslt_initialization> modes = {
            {"rand_int", hipblaslt_initialization::rand_int},
            {"trig_float", hipblaslt_initialization::trig_float},
            {"hpl", hipblaslt_initialization::hpl},
            {"special", hipblaslt_initialization::special},
            {"zero", hipblaslt_initialization::zero},
            {"norm_dist", hipblaslt_initialization::norm_dist},
            {"uniform_01", hipblaslt_initialization::uniform_01},
            {"integer_exact", hipblaslt_initialization::integer_exact},
            {"fp16_accumulator_probe", hipblaslt_initialization::fp16_accumulator_probe},
            {"inf", hipblaslt_initialization::inf},
            {"neg_zero", hipblaslt_initialization::neg_zero},
            {"neg_inf", hipblaslt_initialization::neg_inf},
            {"nan", hipblaslt_initialization::nan},
            {"norm_dist_one_special", hipblaslt_initialization::norm_dist_one_special},
            {"uniform_low_precision", hipblaslt_initialization::uniform_low_precision},
        };
        return modes;
    }

    // Keep in sync with string_to_hipblaslt_activation_type().
    const NameTable<hipblaslt_activation_type>& known_activations()
    {
        static const NameTable<hipblaslt_activation_type> types = {
            {"none", hipblaslt_activation_type::none},
            {"relu", hipblaslt_activation_type::relu},
            {"gelu", hipblaslt_activation_type::gelu},
            {"swish", hipblaslt_activation_type::swish},
            {"clamp", hipblaslt_activation_type::clamp},
            {"sigmoid", hipblaslt_activation_type::sigmoid},
        };
        return types;
    }

    // Keep in sync with string_to_hipblaslt_bias_source().
    const NameTable<hipblaslt_bias_source>& known_bias_sources()
    {
        static const NameTable<hipblaslt_bias_source> sources = {
            {"a", hipblaslt_bias_source::a},
            {"b", hipblaslt_bias_source::b},
            {"d", hipblaslt_bias_source::d},
        };
        return sources;
    }

    template <typename Enum>
    const std::string* registered_name(const NameTable<Enum>& table, int value)
    {
        for(const auto& entry : table)
        {
            if(static_cast<int>(entry.second) == value)
                return &entry.first;
        }
        return nullptr;
    }

    // Assert `table` describes the enum->string mapping exactly, across the
    // enum's whole integer space. Values absent from the table must report
    // kUnknownName; when one does not, an enumerator was added to the header and
    // to its *_to_string switch without being registered here, so none of the
    // round-trip tests below cover it. That silent drift is what this guard
    // exists to catch.
    //
    // Casting an arbitrary int to these enums is well defined: they are all
    // scoped enums with a fixed underlying type (int), so every int value is
    // representable.
    template <typename Enum, typename ToString>
    void expect_table_is_exhaustive(const NameTable<Enum>& table,
                                    int                    max_value,
                                    ToString               to_string,
                                    const char*            enum_name,
                                    const char*            table_name)
    {
        for(int value = 0; value <= max_value; ++value)
        {
            const std::string  actual   = to_string(static_cast<Enum>(value));
            const std::string* expected = registered_name(table, value);
            if(expected)
            {
                EXPECT_EQ(actual, *expected) << enum_name << " value " << value;
            }
            else
            {
                EXPECT_EQ(actual, kUnknownName)
                    << enum_name << " value " << value << " maps to \"" << actual
                    << "\" but is not registered in " << table_name
                    << "(). Add it there and to the string parser so the round-trip"
                       " tests cover it.";
            }
        }
    }

    // Sweep bounds: high enough to cover every enumerator plus headroom for new
    // ones, small enough to stay instant.
    constexpr int kInitSweepMax       = 1023; // enumerators run to 999
    constexpr int kActivationSweepMax = 63; // enumerators run to 5
    constexpr int kBiasSourceSweepMax = 63; // enumerators run to 3
}

TEST(HostUnitInitParser, KnownStringsMapToExpectedEnum)
{
    for(const auto& [name, expected] : known_init_modes())
    {
        EXPECT_EQ(static_cast<int>(string2hipblaslt_initialization(name)),
                  static_cast<int>(expected))
            << "string2hipblaslt_initialization(\"" << name << "\")";
    }
}

TEST(HostUnitInitParser, RoundTripsThroughInitialization2String)
{
    for(const auto& [name, expected] : known_init_modes())
    {
        EXPECT_STREQ(hipblaslt_initialization2string(string2hipblaslt_initialization(name)),
                     name.c_str());
    }
}

TEST(HostUnitInitParser, EveryEnumeratorIsRegistered)
{
    expect_table_is_exhaustive(
        known_init_modes(),
        kInitSweepMax,
        [](hipblaslt_initialization init) { return hipblaslt_initialization2string(init); },
        "hipblaslt_initialization",
        "known_init_modes");
}

// Current (develop) behavior: an unrecognized init string silently maps to the
// integer sentinel 0, which is NOT a valid hipblaslt_initialization enumerator
// (the enum starts at 111). This is exactly the footgun called out in the PR
// #6514 review, observation #3. AIHPBLAS-3551 replaces this with a
// std::optional-returning, noexcept parser, at which point this expectation is
// meant to flip.
TEST(HostUnitInitParser, UnknownStringMapsToZeroSentinel_PreAIHPBLAS3551)
{
    EXPECT_EQ(static_cast<int>(string2hipblaslt_initialization("not_a_real_init_mode")), 0);
    EXPECT_EQ(static_cast<int>(string2hipblaslt_initialization("")), 0);
}

TEST(HostUnitActivationParser, KnownStringsMapAndRoundTrip)
{
    for(const auto& [name, expected] : known_activations())
    {
        EXPECT_EQ(static_cast<int>(string_to_hipblaslt_activation_type(name)),
                  static_cast<int>(expected))
            << "string_to_hipblaslt_activation_type(\"" << name << "\")";
        EXPECT_STREQ(hipblaslt_activation_type_to_string(expected), name.c_str());
    }
}

TEST(HostUnitActivationParser, EveryEnumeratorIsRegistered)
{
    expect_table_is_exhaustive(
        known_activations(),
        kActivationSweepMax,
        [](hipblaslt_activation_type type) { return hipblaslt_activation_type_to_string(type); },
        "hipblaslt_activation_type",
        "known_activations");
}

// none == 0 is a valid activation, so the unknown sentinel here is -1, not 0.
TEST(HostUnitActivationParser, UnknownStringMapsToNegativeSentinel)
{
    EXPECT_EQ(static_cast<int>(string_to_hipblaslt_activation_type("not_an_activation")), -1);
}

TEST(HostUnitBiasSourceParser, KnownStringsMapAndRoundTrip)
{
    for(const auto& [name, expected] : known_bias_sources())
    {
        EXPECT_EQ(static_cast<int>(string_to_hipblaslt_bias_source(name)),
                  static_cast<int>(expected))
            << "string_to_hipblaslt_bias_source(\"" << name << "\")";
        EXPECT_STREQ(hipblaslt_bias_source_to_string(expected), name.c_str());
    }
}

TEST(HostUnitBiasSourceParser, EveryEnumeratorIsRegistered)
{
    expect_table_is_exhaustive(
        known_bias_sources(),
        kBiasSourceSweepMax,
        [](hipblaslt_bias_source source) { return hipblaslt_bias_source_to_string(source); },
        "hipblaslt_bias_source",
        "known_bias_sources");
}

// bias_source values start at 1, so 0 is the unknown sentinel.
TEST(HostUnitBiasSourceParser, UnknownStringMapsToZeroSentinel)
{
    EXPECT_EQ(static_cast<int>(string_to_hipblaslt_bias_source("not_a_bias_source")), 0);
}
