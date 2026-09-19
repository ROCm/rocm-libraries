// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the client string<->enum helpers declared in
// hipblaslt_datatype2string.hpp. See README.md in this directory for the
// dual-target build, the HostUnit/smoke_ naming, and the X-macro tripwire
// pattern used below.

#include "hipblaslt_datatype2string.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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

    // One list per enum, naming every enumerator exactly once. Each list is
    // expanded twice: into the name table the tests below iterate, and into a
    // tripwire switch that makes the compiler reject a list missing an
    // enumerator. Adding an X() entry therefore updates both.
    //
    // The spelling in each X() is both the enum member and the string the
    // parsers are expected to accept for it, so the two cannot disagree here.
    // Whether the parser really accepts that string is what the round-trip tests
    // check.

    // Keep in sync with string2hipblaslt_initialization().
#define HIPBLASLT_TEST_INIT_MODES(X) \
    X(rand_int)                      \
    X(trig_float)                    \
    X(hpl)                           \
    X(special)                       \
    X(zero)                          \
    X(norm_dist)                     \
    X(uniform_01)                    \
    X(integer_exact)                 \
    X(fp16_accumulator_probe)        \
    X(inf)                           \
    X(neg_zero)                      \
    X(neg_inf)                       \
    X(nan)                           \
    X(norm_dist_one_special)         \
    X(uniform_low_precision)

    // Keep in sync with string_to_hipblaslt_activation_type().
#define HIPBLASLT_TEST_ACTIVATIONS(X) \
    X(none)                           \
    X(relu)                           \
    X(gelu)                           \
    X(swish)                          \
    X(clamp)                          \
    X(sigmoid)

    // Keep in sync with string_to_hipblaslt_bias_source().
#define HIPBLASLT_TEST_BIAS_SOURCES(X) \
    X(a)                               \
    X(b)                               \
    X(d)

    const NameTable<hipblaslt_initialization>& known_init_modes()
    {
#define X(name) {#name, hipblaslt_initialization::name},
        static const NameTable<hipblaslt_initialization> modes = {HIPBLASLT_TEST_INIT_MODES(X)};
#undef X
        return modes;
    }

    const NameTable<hipblaslt_activation_type>& known_activations()
    {
#define X(name) {#name, hipblaslt_activation_type::name},
        static const NameTable<hipblaslt_activation_type> types = {HIPBLASLT_TEST_ACTIVATIONS(X)};
#undef X
        return types;
    }

    const NameTable<hipblaslt_bias_source>& known_bias_sources()
    {
#define X(name) {#name, hipblaslt_bias_source::name},
        static const NameTable<hipblaslt_bias_source> sources = {HIPBLASLT_TEST_BIAS_SOURCES(X)};
#undef X
        return sources;
    }

    // Tripwires. These are never called; they exist so that adding an enumerator
    // to one of these enums without adding it to the matching list above fails to
    // compile, in the developer's own build, at the moment they make the change.
    //
    // Each switch deliberately has no `default:` arm, and -Wswitch is promoted to
    // an error for the block, so an unlisted enumerator is reported as
    // "enumeration value 'foo' not handled in switch". Nothing in
    // hipblaslt_datatype2string.hpp can raise that diagnostic on its own: the
    // *_to_string switches all carry `default: return "invalid"`, and
    // string2hipblaslt_initialization is a ternary chain rather than a switch.
    //
    // This is the only check here that is independent of an enumerator's numeric
    // value, which is why it and not the sweep below is the primary guard.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#endif

    [[maybe_unused]] void assert_init_modes_listed(hipblaslt_initialization value)
    {
        switch(value)
        {
#define X(name) case hipblaslt_initialization::name:
            HIPBLASLT_TEST_INIT_MODES(X)
#undef X
            break;
        }
    }

    [[maybe_unused]] void assert_activations_listed(hipblaslt_activation_type value)
    {
        switch(value)
        {
#define X(name) case hipblaslt_activation_type::name:
            HIPBLASLT_TEST_ACTIVATIONS(X)
#undef X
            break;
        }
    }

    [[maybe_unused]] void assert_bias_sources_listed(hipblaslt_bias_source value)
    {
        switch(value)
        {
#define X(name) case hipblaslt_bias_source::name:
            HIPBLASLT_TEST_BIAS_SOURCES(X)
#undef X
            break;
        }
    }

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

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

    // Upper bound for the sweep below, derived from the table rather than
    // hand-written so it cannot go stale as enumerators are added. The headroom
    // is what lets the sweep also cover values just past the end of the enum.
    template <typename Enum>
    int sweep_max(const NameTable<Enum>& table)
    {
        int max_registered = 0;
        for(const auto& entry : table)
            max_registered = std::max(max_registered, static_cast<int>(entry.second));
        return max_registered + 64;
    }

    // Assert `table` describes the enum->string mapping exactly, over the
    // registered enumerators and a margin past the largest of them. Values absent
    // from the table must report kUnknownName; when one does not, an enumerator
    // was added to the header and to its *_to_string switch without being
    // registered here, so none of the round-trip tests below cover it.
    //
    // This is a secondary guard. It can only see values it sweeps, so on its own
    // it would miss an enumerator numbered past the margin, which is exactly what
    // the compile-time tripwires above are for. What it adds is behavioral: it
    // pins the "unrecognized values report invalid" contract that the tripwires
    // say nothing about.
    //
    // Casting an arbitrary int to these enums is well defined: they are all
    // scoped enums with a fixed underlying type (int), so every int value is
    // representable.
    template <typename Enum, typename ToString>
    void expect_table_is_exhaustive(const NameTable<Enum>& table,
                                    ToString               to_string,
                                    const char*            enum_name,
                                    const char*            table_name)
    {
        const int max_value = sweep_max(table);
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
}

TEST(HostUnitInitParser, smoke_KnownStringsMapToExpectedEnum)
{
    for(const auto& [name, expected] : known_init_modes())
    {
        EXPECT_EQ(static_cast<int>(string2hipblaslt_initialization(name)),
                  static_cast<int>(expected))
            << "string2hipblaslt_initialization(\"" << name << "\")";
    }
}

TEST(HostUnitInitParser, smoke_RoundTripsThroughInitialization2String)
{
    for(const auto& [name, expected] : known_init_modes())
    {
        EXPECT_STREQ(hipblaslt_initialization2string(string2hipblaslt_initialization(name)),
                     name.c_str());
    }
}

TEST(HostUnitInitParser, smoke_EveryEnumeratorIsRegistered)
{
    expect_table_is_exhaustive(
        known_init_modes(),
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
TEST(HostUnitInitParser, smoke_UnknownStringMapsToZeroSentinel_PreAIHPBLAS3551)
{
    EXPECT_EQ(static_cast<int>(string2hipblaslt_initialization("not_a_real_init_mode")), 0);
    EXPECT_EQ(static_cast<int>(string2hipblaslt_initialization("")), 0);
}

TEST(HostUnitActivationParser, smoke_KnownStringsMapAndRoundTrip)
{
    for(const auto& [name, expected] : known_activations())
    {
        EXPECT_EQ(static_cast<int>(string_to_hipblaslt_activation_type(name)),
                  static_cast<int>(expected))
            << "string_to_hipblaslt_activation_type(\"" << name << "\")";
        EXPECT_STREQ(hipblaslt_activation_type_to_string(expected), name.c_str());
    }
}

TEST(HostUnitActivationParser, smoke_EveryEnumeratorIsRegistered)
{
    expect_table_is_exhaustive(
        known_activations(),
        [](hipblaslt_activation_type type) { return hipblaslt_activation_type_to_string(type); },
        "hipblaslt_activation_type",
        "known_activations");
}

// none == 0 is a valid activation, so the unknown sentinel here is -1, not 0.
TEST(HostUnitActivationParser, smoke_UnknownStringMapsToNegativeSentinel)
{
    EXPECT_EQ(static_cast<int>(string_to_hipblaslt_activation_type("not_an_activation")), -1);
}

TEST(HostUnitBiasSourceParser, smoke_KnownStringsMapAndRoundTrip)
{
    for(const auto& [name, expected] : known_bias_sources())
    {
        EXPECT_EQ(static_cast<int>(string_to_hipblaslt_bias_source(name)),
                  static_cast<int>(expected))
            << "string_to_hipblaslt_bias_source(\"" << name << "\")";
        EXPECT_STREQ(hipblaslt_bias_source_to_string(expected), name.c_str());
    }
}

TEST(HostUnitBiasSourceParser, smoke_EveryEnumeratorIsRegistered)
{
    expect_table_is_exhaustive(
        known_bias_sources(),
        [](hipblaslt_bias_source source) { return hipblaslt_bias_source_to_string(source); },
        "hipblaslt_bias_source",
        "known_bias_sources");
}

// bias_source values start at 1, so 0 is the unknown sentinel.
TEST(HostUnitBiasSourceParser, smoke_UnknownStringMapsToZeroSentinel)
{
    EXPECT_EQ(static_cast<int>(string_to_hipblaslt_bias_source("not_a_bias_source")), 0);
}
