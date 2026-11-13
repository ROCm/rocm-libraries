// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The backend and plugin SDK each ship their own copy of the numerical note values; both
// copies share one include guard, so a translation unit normally sees only one of them. Each
// is included here into its own namespace so their values can be compared.
#include <cstdint>

#include <gtest/gtest.h>

#include <type_traits>

namespace backend_header
{
#include <HipdnnBackendNumericalNote.h>
}

#undef HIPDNN_NUMERICAL_NOTE_TYPES_DEFINED

namespace plugin_header
{
#include <hipdnn_plugin_sdk/NumericalNote.h>
}

TEST(TestNumericalNoteHeaders, BackendAndPluginCopiesAgreeOnEveryValue)
{
    // clang-format off
#define EXPECT_NOTE_VALUES_AGREE(NOTE) \
    EXPECT_EQ(static_cast<int64_t>(backend_header::NOTE), static_cast<int64_t>(plugin_header::NOTE)) << #NOTE
    // clang-format on
    EXPECT_NOTE_VALUES_AGREE(HIPDNN_NUMERICAL_NOTE_TENSOR_CORE);
    EXPECT_NOTE_VALUES_AGREE(HIPDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS);
    EXPECT_NOTE_VALUES_AGREE(HIPDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION);
    EXPECT_NOTE_VALUES_AGREE(HIPDNN_NUMERICAL_NOTE_FFT);
    EXPECT_NOTE_VALUES_AGREE(HIPDNN_NUMERICAL_NOTE_NONDETERMINISTIC);
    EXPECT_NOTE_VALUES_AGREE(HIPDNN_NUMERICAL_NOTE_WINOGRAD);
    EXPECT_NOTE_VALUES_AGREE(HIPDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4);
    EXPECT_NOTE_VALUES_AGREE(HIPDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6);
    EXPECT_NOTE_VALUES_AGREE(HIPDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13);
    EXPECT_NOTE_VALUES_AGREE(HIPDNN_NUMERICAL_NOTE_STRICT_NAN_PROP);
    EXPECT_NOTE_VALUES_AGREE(HIPDNN_NUMERICAL_NOTE_TYPE_COUNT);
#undef EXPECT_NOTE_VALUES_AGREE

    // Plugins serialize these values; the first note is 0.
    EXPECT_EQ(static_cast<int64_t>(backend_header::HIPDNN_NUMERICAL_NOTE_TENSOR_CORE), 0);
    EXPECT_EQ(static_cast<int64_t>(plugin_header::HIPDNN_NUMERICAL_NOTE_TENSOR_CORE), 0);
}

TEST(TestNumericalNoteHeaders, BackendAndPluginCopiesAgreeOnTransportType)
{
    static_assert(std::is_same_v<backend_header::hipdnnBackendNumericalNote_t, int32_t>);
    static_assert(std::is_same_v<plugin_header::hipdnnBackendNumericalNote_t, int32_t>);
}
