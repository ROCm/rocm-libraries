// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/engine_api_version.h>
#include <hipdnn_plugin_sdk/version.h>

#include <gtest/gtest.h>

using namespace hipdnn_data_sdk::utilities;

TEST(TestVersion, ParsedSuccessfully)
{
    EXPECT_NO_THROW(Version(std::string_view{HIPDNN_PLUGIN_SDK_VERSION_STRING}));
}

TEST(TestVersion, PositiveVersion)
{
    Version version;
    ASSERT_NO_THROW(version = Version(std::string_view{HIPDNN_PLUGIN_SDK_VERSION_STRING}));

    EXPECT_GE(version.major, 0);
    EXPECT_GE(version.minor, 0);
    EXPECT_GE(version.patch, 0);
}

TEST(TestVersion, OverrideExecuteMinApiVersionParses)
{
    EXPECT_NO_THROW(Version{hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION});
    const Version v{hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION};
    EXPECT_EQ(v.major, 1);
    EXPECT_EQ(v.minor, 1);
    EXPECT_EQ(v.patch, 0);
}

TEST(TestVersion, BaselineVersionLessThanOverrideExecuteMinApiVersion)
{
    const Version baseline{hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE};
    const Version overrideMin{hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION};
    EXPECT_TRUE(baseline < overrideMin);
}

TEST(TestVersion, RaggedTensorMinApiVersionParses)
{
    EXPECT_NO_THROW(Version{hipdnn_plugin_sdk::K_RAGGED_TENSOR_MIN_API_VERSION});
    const Version v{hipdnn_plugin_sdk::K_RAGGED_TENSOR_MIN_API_VERSION};
    EXPECT_EQ(v.major, 1);
    EXPECT_EQ(v.minor, 3);
    EXPECT_EQ(v.patch, 0);
}

TEST(TestVersion, OverrideExecuteMinApiVersionLessThanRaggedTensorMinApiVersion)
{
    const Version overrideMin{hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION};
    const Version raggedMin{hipdnn_plugin_sdk::K_RAGGED_TENSOR_MIN_API_VERSION};
    EXPECT_TRUE(overrideMin < raggedMin);
}

TEST(TestVersion, TensorAlignmentMinApiVersionParses)
{
    EXPECT_NO_THROW(Version{hipdnn_plugin_sdk::K_TENSOR_ATTRIBUTE_ALIGNMENT_MIN_VERSION});
    const Version v{hipdnn_plugin_sdk::K_TENSOR_ATTRIBUTE_ALIGNMENT_MIN_VERSION};
    EXPECT_EQ(v.major, 1);
    EXPECT_EQ(v.minor, 3);
    EXPECT_EQ(v.patch, 0);
}

TEST(TestVersion, TensorAlignmentMinApiVersionEqualsRaggedTensorMinApiVersion)
{
    const Version alignmentMin{hipdnn_plugin_sdk::K_TENSOR_ATTRIBUTE_ALIGNMENT_MIN_VERSION};
    const Version raggedMin{hipdnn_plugin_sdk::K_RAGGED_TENSOR_MIN_API_VERSION};
    EXPECT_TRUE(alignmentMin == raggedMin);
}

TEST(TestVersion, OverrideExecuteMinApiVersionLessThanTensorAlignmentMinApiVersion)
{
    const Version overrideMin{hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION};
    const Version alignmentMin{hipdnn_plugin_sdk::K_TENSOR_ATTRIBUTE_ALIGNMENT_MIN_VERSION};
    EXPECT_TRUE(overrideMin < alignmentMin);
}

TEST(TestVersion, RaggedOffsetMultiplierMinApiVersionParses)
{
    EXPECT_NO_THROW(Version{hipdnn_plugin_sdk::K_RAGGED_OFFSET_MULTIPLIER_MIN_VERSION});
    const Version v{hipdnn_plugin_sdk::K_RAGGED_OFFSET_MULTIPLIER_MIN_VERSION};
    EXPECT_EQ(v.major, 1);
    EXPECT_EQ(v.minor, 4);
    EXPECT_EQ(v.patch, 0);
}

TEST(TestVersion, RaggedOffsetMultiplierMinApiVersionIsBelowMaxSupported)
{
    const Version multiplierMin{hipdnn_plugin_sdk::K_RAGGED_OFFSET_MULTIPLIER_MIN_VERSION};
    const Version maxSupported{hipdnn_plugin_sdk::K_MAX_SUPPORTED_API_VERSION};
    EXPECT_TRUE(multiplierMin < maxSupported);
}

TEST(TestVersion, CuSeqLenMinApiVersionParses)
{
    EXPECT_NO_THROW(Version{hipdnn_plugin_sdk::K_CU_SEQ_LEN_MIN_API_VERSION});
    const Version v{hipdnn_plugin_sdk::K_CU_SEQ_LEN_MIN_API_VERSION};
    EXPECT_EQ(v.major, 1);
    EXPECT_EQ(v.minor, 6);
    EXPECT_EQ(v.patch, 0);
}

// 1.5.0 is the UHD prediction entry points release and has no cu_seq_len support, so such
// plugins must still be rejected for a cu_seq_len graph.
TEST(TestVersion, CuSeqLenMinApiVersionIsAboveUhdApiVersion)
{
    const Version uhdApiVersion{std::string_view{"1.5.0"}};
    const Version cuSeqLenMin{hipdnn_plugin_sdk::K_CU_SEQ_LEN_MIN_API_VERSION};
    EXPECT_TRUE(uhdApiVersion < cuSeqLenMin);
}

// Plugins built from this SDK report HIPDNN_ENGINE_API_VERSION. It must reach the highest
// graph-feature floor, or no plugin from this tree could serve such a graph.
TEST(TestVersion, SdkEngineApiVersionReachesMaxSupportedFloor)
{
    const Version sdkVersion{std::string_view{HIPDNN_ENGINE_API_VERSION}};
    const Version maxSupported{hipdnn_plugin_sdk::K_MAX_SUPPORTED_API_VERSION};
    EXPECT_FALSE(sdkVersion < maxSupported);
}

TEST(TestVersion, CuSeqLenMinApiVersionEqualsMaxSupported)
{
    const Version cuSeqLenMin{hipdnn_plugin_sdk::K_CU_SEQ_LEN_MIN_API_VERSION};
    const Version maxSupported{hipdnn_plugin_sdk::K_MAX_SUPPORTED_API_VERSION};
    EXPECT_TRUE(cuSeqLenMin == maxSupported);
}

TEST(TestVersion, CuSeqLenFloorDominatesEveryOtherFeatureFloor)
{
    const Version cuSeqLenMin{hipdnn_plugin_sdk::K_CU_SEQ_LEN_MIN_API_VERSION};
    EXPECT_EQ(
        hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(true, true, true, true, true, true),
        cuSeqLenMin);
    EXPECT_EQ(hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(
                  false, false, false, false, false, true),
              cuSeqLenMin);
}

TEST(TestVersion, NoCuSeqLenLeavesFloorBelowCuSeqLenMinVersion)
{
    const Version cuSeqLenMin{hipdnn_plugin_sdk::K_CU_SEQ_LEN_MIN_API_VERSION};
    EXPECT_TRUE(
        hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(true, true, true, true, true, false)
        < cuSeqLenMin);
}

TEST(TestVersion, RaggedTensorMinApiVersionLessThanRaggedOffsetMultiplierMinApiVersion)
{
    const Version raggedMin{hipdnn_plugin_sdk::K_RAGGED_TENSOR_MIN_API_VERSION};
    const Version multiplierMin{hipdnn_plugin_sdk::K_RAGGED_OFFSET_MULTIPLIER_MIN_VERSION};
    EXPECT_TRUE(raggedMin < multiplierMin);
}
