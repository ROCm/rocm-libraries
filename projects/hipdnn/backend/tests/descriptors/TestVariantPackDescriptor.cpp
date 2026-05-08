// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/VariantDescriptor.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace hipdnn_backend
{

class TestVariantPackDescriptorWhenInitialized : public ::testing::Test
{
protected:
    VariantDescriptor _descriptor;

    void SetUp() override
    {
        ASSERT_FALSE(_descriptor.isFinalized());
    }
};

namespace
{

void setBaseVariantPackAttributes(VariantDescriptor& descriptor)
{
    std::vector<void*> devPtrs = {reinterpret_cast<void*>(0x1234),
                                  reinterpret_cast<void*>(0x5678),
                                  reinterpret_cast<void*>(0x9abc)};
    std::vector<int64_t> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                            HIPDNN_TYPE_VOID_PTR,
                            static_cast<int64_t>(devPtrs.size()),
                            static_cast<const void*>(devPtrs.data()));

    descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(uids.size()),
                            uids.data());

    descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                            HIPDNN_TYPE_VOID_PTR,
                            1,
                            static_cast<const void*>(&workspace));
}

void setOverrideVariantPackAttributes(VariantDescriptor& descriptor,
                                      const std::vector<int64_t>& overrideUids,
                                      const std::vector<int64_t>& overrideShapes,
                                      const std::vector<int64_t>& overrideStrides,
                                      const std::vector<int64_t>& overrideLengths)
{
    descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(overrideUids.size()),
                            overrideUids.data());

    descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(overrideShapes.size()),
                            overrideShapes.data());

    descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(overrideStrides.size()),
                            overrideStrides.data());

    descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS,
                            HIPDNN_TYPE_INT64,
                            static_cast<int64_t>(overrideLengths.size()),
                            overrideLengths.data());
}

void setBaseAndOverrideVariantPackAttributes(VariantDescriptor& descriptor,
                                             const std::vector<int64_t>& overrideUids,
                                             const std::vector<int64_t>& overrideShapes,
                                             const std::vector<int64_t>& overrideStrides,
                                             const std::vector<int64_t>& overrideLengths)
{
    setBaseVariantPackAttributes(descriptor);
    setOverrideVariantPackAttributes(
        descriptor, overrideUids, overrideShapes, overrideStrides, overrideLengths);
}

void expectQueryThenGetInt64Vector(const VariantDescriptor& descriptor,
                                   hipdnnBackendAttributeName_t attributeName,
                                   const std::vector<int64_t>& expected)
{
    int64_t elementCount = -1;
    ASSERT_NO_THROW(
        descriptor.getAttribute(attributeName, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, static_cast<int64_t>(expected.size()));

    std::vector<int64_t> retrieved(static_cast<size_t>(elementCount));
    elementCount = -1;
    ASSERT_NO_THROW(descriptor.getAttribute(attributeName,
                                            HIPDNN_TYPE_INT64,
                                            static_cast<int64_t>(retrieved.size()),
                                            &elementCount,
                                            retrieved.data()));
    EXPECT_EQ(elementCount, static_cast<int64_t>(expected.size()));
    EXPECT_EQ(retrieved, expected);
}

} // namespace

TEST_F(TestVariantPackDescriptorWhenInitialized, ValidSetAttributes)
{
    std::array<void*, 3> devPtrs = {reinterpret_cast<void*>(0x1234),
                                    reinterpret_cast<void*>(0x5678),
                                    reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                             HIPDNN_TYPE_VOID_PTR,
                             devPtrs.size(),
                             static_cast<const void*>(devPtrs.data()));

    _descriptor.setAttribute(
        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, uids.size(), uids.data());

    _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                             HIPDNN_TYPE_VOID_PTR,
                             1,
                             static_cast<const void*>(&workspace));

    ASSERT_NO_THROW(_descriptor.finalize());
    EXPECT_TRUE(_descriptor.isFinalized());
}

TEST_F(TestVariantPackDescriptorWhenInitialized, ValidSetAndGetBeforeFinalAttributes)
{
    std::array<void*, 3> devPtrs = {reinterpret_cast<void*>(0x1234),
                                    reinterpret_cast<void*>(0x5678),
                                    reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                             HIPDNN_TYPE_VOID_PTR,
                             devPtrs.size(),
                             static_cast<const void*>(devPtrs.data()));

    _descriptor.setAttribute(
        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, uids.size(), uids.data());

    _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                             HIPDNN_TYPE_VOID_PTR,
                             1,
                             static_cast<const void*>(&workspace));
    // getting before finalized
    std::array<void*, 3> retrievedDevPtrs;
    int64_t elementCount = 0;

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                        HIPDNN_TYPE_VOID_PTR,
                                                        retrievedDevPtrs.size(),
                                                        &elementCount,
                                                        retrievedDevPtrs.data()),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestVariantPackDescriptorWhenInitialized, InvalidSetAttributes)
{
    std::array<void*, 3> devPtrs = {reinterpret_cast<void*>(0x1234),
                                    reinterpret_cast<void*>(0x5678),
                                    reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                        HIPDNN_TYPE_INT64,
                                                        devPtrs.size(),
                                                        devPtrs.data()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _descriptor.setAttribute(
            HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_VOID_PTR, uids.size(), uids.data()),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _descriptor.setAttribute(
            HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 2, &workspace),
        HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(
        _descriptor.setAttribute(
            HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS, HIPDNN_TYPE_VOID_PTR, devPtrs.size(), nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestVariantPackDescriptorWhenInitialized, InvalidFinalizeCounts)
{
    std::array<void*, 3> devPtrs = {reinterpret_cast<void*>(0x1234),
                                    reinterpret_cast<void*>(0x5678),
                                    reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 2> uids = {1, 2};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                             HIPDNN_TYPE_VOID_PTR,
                             devPtrs.size(),
                             static_cast<const void*>(devPtrs.data()));

    _descriptor.setAttribute(
        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, uids.size(), uids.data());

    _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                             HIPDNN_TYPE_VOID_PTR,
                             1,
                             static_cast<const void*>(&workspace));

    ASSERT_THROW(_descriptor.finalize(), HipdnnException);
    EXPECT_FALSE(_descriptor.isFinalized());
}

TEST_F(TestVariantPackDescriptorWhenInitialized, InvalidFinalizeUnsetParams)
{
    ASSERT_THROW(_descriptor.finalize(), HipdnnException);
    EXPECT_FALSE(_descriptor.isFinalized());
}

struct VariantPackFinalizeRejectCase
{
    const char* name;
    std::vector<int64_t> overrideUids;
    std::vector<int64_t> overrideShapes;
    std::vector<int64_t> overrideStrides;
    std::vector<int64_t> overrideLengths;
};

class TestVariantPackDescriptorFinalizeRejects
    : public TestVariantPackDescriptorWhenInitialized,
      public ::testing::WithParamInterface<VariantPackFinalizeRejectCase>
{
};

TEST_P(TestVariantPackDescriptorFinalizeRejects, InvalidOverrideAttributes)
{
    const auto& testCase = GetParam();
    setBaseAndOverrideVariantPackAttributes(_descriptor,
                                            testCase.overrideUids,
                                            testCase.overrideShapes,
                                            testCase.overrideStrides,
                                            testCase.overrideLengths);

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.finalize(), HIPDNN_STATUS_BAD_PARAM);
    EXPECT_FALSE(_descriptor.isFinalized());
}

INSTANTIATE_TEST_SUITE_P(
    OverrideAttributes,
    TestVariantPackDescriptorFinalizeRejects,
    ::testing::Values(
        VariantPackFinalizeRejectCase{
            "UidLengthCountMismatch", {1, 2}, {1, 2, 3, 4}, {4, 3, 2, 1}, {4}},
        VariantPackFinalizeRejectCase{"DuplicateOverrideUid",
                                      {1, 1},
                                      {1, 2, 3, 4, 5, 6, 7, 8},
                                      {8, 7, 6, 5, 4, 3, 2, 1},
                                      {4, 4}},
        VariantPackFinalizeRejectCase{
            "OverrideUidAbsentFromBaseUids", {4}, {1, 2, 3, 4}, {4, 3, 2, 1}, {4}},
        VariantPackFinalizeRejectCase{"NegativeOverrideLength", {1}, {1}, {1}, {-1}},
        VariantPackFinalizeRejectCase{"ZeroOverrideLength", {1}, {1}, {1}, {0}},
        VariantPackFinalizeRejectCase{
            "OverrideLengthAboveUint32Max",
            {1},
            {1},
            {1},
            {static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1}},
        VariantPackFinalizeRejectCase{"HugeLengthSumWithoutHugeBuffers",
                                      {1, 2},
                                      {1},
                                      {1},
                                      {static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                                       static_cast<int64_t>(std::numeric_limits<uint32_t>::max())}},
        VariantPackFinalizeRejectCase{"OverrideShapesFlatCountMismatch", {1}, {1}, {1, 1}, {2}},
        VariantPackFinalizeRejectCase{
            "OverrideShapesFlatCountTooLong", {1}, {1, 2, 3}, {1, 1}, {2}},
        VariantPackFinalizeRejectCase{"OverrideStridesFlatCountMismatch", {1}, {1, 1}, {1}, {2}},
        VariantPackFinalizeRejectCase{
            "OverrideStridesFlatCountTooLong", {1}, {1, 1}, {1, 2, 3}, {2}}),
    [](const auto& info) { return std::string(info.param.name); });

enum class SingleOverrideAttribute
{
    UNIQUE_IDS,
    SHAPES,
    STRIDES,
    LENGTHS
};

struct VariantPackPartialOverrideCase
{
    const char* name;
    SingleOverrideAttribute attribute;
};

class TestVariantPackDescriptorPartialOverrideRejects
    : public TestVariantPackDescriptorWhenInitialized,
      public ::testing::WithParamInterface<VariantPackPartialOverrideCase>
{
};

TEST_P(TestVariantPackDescriptorPartialOverrideRejects, FinalizeRejectsPartialOverrideMetadata)
{
    setBaseVariantPackAttributes(_descriptor);

    const auto& testCase = GetParam();
    std::vector<int64_t> values{1};
    hipdnnBackendAttributeName_t attr = HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS;
    switch(testCase.attribute)
    {
    case SingleOverrideAttribute::UNIQUE_IDS:
        attr = HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS;
        break;
    case SingleOverrideAttribute::SHAPES:
        attr = HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES;
        break;
    case SingleOverrideAttribute::STRIDES:
        attr = HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES;
        break;
    case SingleOverrideAttribute::LENGTHS:
        attr = HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS;
        break;
    default:
        break;
    }

    _descriptor.setAttribute(
        attr, HIPDNN_TYPE_INT64, static_cast<int64_t>(values.size()), values.data());

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.finalize(), HIPDNN_STATUS_BAD_PARAM);
    EXPECT_FALSE(_descriptor.isFinalized());
}

INSTANTIATE_TEST_SUITE_P(
    OverrideAttributes,
    TestVariantPackDescriptorPartialOverrideRejects,
    ::testing::Values(
        VariantPackPartialOverrideCase{"OverrideUidsOnly", SingleOverrideAttribute::UNIQUE_IDS},
        VariantPackPartialOverrideCase{"OverrideShapesOnly", SingleOverrideAttribute::SHAPES},
        VariantPackPartialOverrideCase{"OverrideStridesOnly", SingleOverrideAttribute::STRIDES},
        VariantPackPartialOverrideCase{"OverrideLengthsOnly", SingleOverrideAttribute::LENGTHS}),
    [](const auto& info) { return std::string(info.param.name); });

TEST_F(TestVariantPackDescriptorWhenInitialized, FinalizeAcceptsNonPackedNonContiguousStrides)
{
    setBaseAndOverrideVariantPackAttributes(_descriptor, {1}, {2, 3, 4, 5}, {100, 30, 7, 2}, {4});

    ASSERT_NO_THROW(_descriptor.finalize());
    EXPECT_TRUE(_descriptor.isFinalized());
}

TEST_F(TestVariantPackDescriptorWhenInitialized, InvalidGetAttributeNotFinalized)
{
    std::array<void*, 3> retrievedDevPtrs;
    int64_t elementCount = 0;

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                        HIPDNN_TYPE_VOID_PTR,
                                                        retrievedDevPtrs.size(),
                                                        &elementCount,
                                                        retrievedDevPtrs.data()),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

class TestVariantPackDescriptorWhenFinalized : public ::testing::Test
{
protected:
    VariantDescriptor _descriptor;
    std::array<void*, 3> _devPtrs = {reinterpret_cast<void*>(0x1234),
                                     reinterpret_cast<void*>(0x5678),
                                     reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> _uids = {1, 2, 3};
    void* _workspace = reinterpret_cast<void*>(0xdeadbeef);

    void SetUp() override
    {
        _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                 HIPDNN_TYPE_VOID_PTR,
                                 static_cast<int64_t>(_devPtrs.size()),
                                 static_cast<const void*>(_devPtrs.data()));

        _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                 HIPDNN_TYPE_INT64,
                                 static_cast<int64_t>(_uids.size()),
                                 _uids.data());

        _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                 HIPDNN_TYPE_VOID_PTR,
                                 1,
                                 static_cast<const void*>(&_workspace));

        ASSERT_NO_THROW(_descriptor.finalize());
        EXPECT_TRUE(_descriptor.isFinalized());
    }
};

TEST_F(TestVariantPackDescriptorWhenFinalized, InvalidSetAttribute)
{
    ASSERT_THROW_HIPDNN_STATUS(_descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                        HIPDNN_TYPE_VOID_PTR,
                                                        static_cast<int64_t>(_devPtrs.size()),
                                                        _devPtrs.data()),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestVariantPackDescriptorWhenFinalized, ValidGetAttributes)
{
    std::array<void*, 3> retrievedDevPtrs;
    std::array<int64_t, 3> retrievedUids;
    void* retrievedWorkspace = nullptr;
    int64_t elementCount = 0;

    _descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                             HIPDNN_TYPE_VOID_PTR,
                             retrievedDevPtrs.size(),
                             &elementCount,
                             static_cast<void*>(retrievedDevPtrs.data()));
    EXPECT_EQ(elementCount, _devPtrs.size());
    EXPECT_EQ(std::memcmp(static_cast<const void*>(retrievedDevPtrs.data()),
                          static_cast<const void*>(_devPtrs.data()),
                          _devPtrs.size() * sizeof(void*)),
              0);

    _descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                             HIPDNN_TYPE_INT64,
                             retrievedUids.size(),
                             &elementCount,
                             retrievedUids.data());
    EXPECT_EQ(elementCount, _uids.size());
    EXPECT_EQ(std::memcmp(retrievedUids.data(), _uids.data(), _uids.size() * sizeof(int64_t)), 0);

    _descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                             HIPDNN_TYPE_VOID_PTR,
                             1,
                             &elementCount,
                             static_cast<void*>(&retrievedWorkspace));
    EXPECT_EQ(elementCount, 1);
    EXPECT_EQ(retrievedWorkspace, _workspace);
}

TEST_F(TestVariantPackDescriptorWhenFinalized, InvalidGetAttributes)
{
    std::array<void*, 3> retrievedDevPtrs;
    std::array<int64_t, 3> retrievedUids;
    void* retrievedWorkspace = nullptr;
    int64_t elementCount = 0;

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                        HIPDNN_TYPE_INT64,
                                                        retrievedDevPtrs.size(),
                                                        &elementCount,
                                                        retrievedDevPtrs.data()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                                        HIPDNN_TYPE_VOID_PTR,
                                                        retrievedUids.size(),
                                                        &elementCount,
                                                        retrievedUids.data()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                                        HIPDNN_TYPE_VOID_PTR,
                                                        2,
                                                        &elementCount,
                                                        &retrievedWorkspace),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                        HIPDNN_TYPE_VOID_PTR,
                                                        retrievedDevPtrs.size(),
                                                        &elementCount,
                                                        nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                        HIPDNN_TYPE_VOID_PTR,
                                                        -1,
                                                        &elementCount,
                                                        retrievedDevPtrs.data()),
                               HIPDNN_STATUS_BAD_PARAM);

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                        HIPDNN_TYPE_VOID_PTR,
                                                        retrievedDevPtrs.size(),
                                                        nullptr,
                                                        retrievedDevPtrs.data()),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// ============================================================================
// Per-execute override tensor attributes 704-707 (RFC 0008)
// ============================================================================

/// Fixture that finalizes a VariantDescriptor with both base attributes and the
/// override attributes populated from the RFC 0008 §4.3 worked example:
///   - Two override tensors (UID 1 with rank 4, UID 2 with rank 3).
///   - OVERRIDE_SHAPES is the flat concatenation [n,c,h,w | n,c,k] of the two
///     per-tensor shape vectors, sliced via OVERRIDE_LENGTHS = [4, 3].
///   - OVERRIDE_STRIDES uses the same fan-out.
class TestVariantPackDescriptorWithOverrides : public ::testing::Test
{
protected:
    VariantDescriptor _descriptor;
    std::array<void*, 3> _devPtrs = {reinterpret_cast<void*>(0x1234),
                                     reinterpret_cast<void*>(0x5678),
                                     reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> _uids = {1, 2, 3};
    void* _workspace = reinterpret_cast<void*>(0xdeadbeef);

    // Worked example from RFC 0008 §4.3.
    std::array<int64_t, 2> _overrideUids = {1, 2};
    std::array<int64_t, 7> _overrideShapes = {8, 16, 32, 32, 4, 64, 128};
    std::array<int64_t, 7> _overrideStrides = {16384, 1024, 32, 1, 8192, 128, 1};
    std::array<int64_t, 2> _overrideLengths = {4, 3};

    void SetUp() override
    {
        _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                 HIPDNN_TYPE_VOID_PTR,
                                 static_cast<int64_t>(_devPtrs.size()),
                                 static_cast<const void*>(_devPtrs.data()));

        _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                 HIPDNN_TYPE_INT64,
                                 static_cast<int64_t>(_uids.size()),
                                 _uids.data());

        _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                 HIPDNN_TYPE_VOID_PTR,
                                 1,
                                 static_cast<const void*>(&_workspace));

        _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS,
                                 HIPDNN_TYPE_INT64,
                                 static_cast<int64_t>(_overrideUids.size()),
                                 _overrideUids.data());

        _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES,
                                 HIPDNN_TYPE_INT64,
                                 static_cast<int64_t>(_overrideShapes.size()),
                                 _overrideShapes.data());

        _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES,
                                 HIPDNN_TYPE_INT64,
                                 static_cast<int64_t>(_overrideStrides.size()),
                                 _overrideStrides.data());

        _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS,
                                 HIPDNN_TYPE_INT64,
                                 static_cast<int64_t>(_overrideLengths.size()),
                                 _overrideLengths.data());

        ASSERT_NO_THROW(_descriptor.finalize());
        EXPECT_TRUE(_descriptor.isFinalized());
    }
};

TEST_F(TestVariantPackDescriptorWithOverrides, OverrideLengthsRejectsNonInt64ElementType)
{
    // 707 (LENGTHS) requires HIPDNN_TYPE_INT64. Other integer types must be rejected
    // so callers cannot accidentally pass a uint32_t buffer (which is the dispatch-
    // boundary type, not the variant-pack storage type).
    std::array<int64_t, 2> retrieved{};
    int64_t elementCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(_descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS,
                                                        HIPDNN_TYPE_VOID_PTR,
                                                        static_cast<int64_t>(retrieved.size()),
                                                        &elementCount,
                                                        retrieved.data()),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestVariantPackDescriptorWithOverrides, OverrideLengthsAccessorReturnsStoredVector)
{
    // Direct C++ accessor mirrors the C-API getAttribute path.
    const auto& lens = _descriptor.getOverrideLengths();
    ASSERT_EQ(lens.size(), 2u);
    EXPECT_EQ(lens[0], 4);
    EXPECT_EQ(lens[1], 3);

    const auto& uids = _descriptor.getOverrideUniqueIds();
    ASSERT_EQ(uids.size(), 2u);
    EXPECT_EQ(uids[0], 1);
    EXPECT_EQ(uids[1], 2);
}

TEST_F(TestVariantPackDescriptorWithOverrides, QueryThenGetVectorAttributes)
{
    expectQueryThenGetInt64Vector(_descriptor,
                                  HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                  std::vector<int64_t>(_uids.begin(), _uids.end()));
    expectQueryThenGetInt64Vector(_descriptor,
                                  HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS,
                                  std::vector<int64_t>(_overrideUids.begin(), _overrideUids.end()));
    expectQueryThenGetInt64Vector(
        _descriptor,
        HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES,
        std::vector<int64_t>(_overrideShapes.begin(), _overrideShapes.end()));
    expectQueryThenGetInt64Vector(
        _descriptor,
        HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES,
        std::vector<int64_t>(_overrideStrides.begin(), _overrideStrides.end()));
    expectQueryThenGetInt64Vector(
        _descriptor,
        HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS,
        std::vector<int64_t>(_overrideLengths.begin(), _overrideLengths.end()));
}

TEST_F(TestVariantPackDescriptorWithOverrides, WorkspaceAndDataPointersRejectNullOutputQueries)
{
    int64_t elementCount = 0;

    ASSERT_THROW_HIPDNN_STATUS(_descriptor.getAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                        HIPDNN_TYPE_VOID_PTR,
                                                        0,
                                                        &elementCount,
                                                        nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    ASSERT_THROW_HIPDNN_STATUS(
        _descriptor.getAttribute(
            HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &elementCount, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestVariantPackDescriptorWithOverrides, RfcWorkedExampleSliceReconstructionViaPrefixSum)
{
    // Reconstruct each tensor's shape and stride vector from the flat override arrays.
    const auto& uids = _descriptor.getOverrideUniqueIds();
    const auto& shapes = _descriptor.getOverrideShapes();
    const auto& strides = _descriptor.getOverrideStrides();
    const auto& lens = _descriptor.getOverrideLengths();

    ASSERT_EQ(uids.size(), lens.size());

    std::vector<std::vector<int64_t>> reconstructedShapes;
    std::vector<std::vector<int64_t>> reconstructedStrides;
    int64_t offset = 0;
    for(size_t i = 0; i < uids.size(); ++i)
    {
        const auto rank = lens[i];
        ASSERT_GE(rank, 0);
        ASSERT_LE(static_cast<size_t>(offset + rank), shapes.size());
        ASSERT_LE(static_cast<size_t>(offset + rank), strides.size());

        reconstructedShapes.emplace_back(shapes.begin() + offset, shapes.begin() + offset + rank);
        reconstructedStrides.emplace_back(strides.begin() + offset,
                                          strides.begin() + offset + rank);
        offset += rank;
    }
    EXPECT_EQ(static_cast<size_t>(offset), shapes.size());
    EXPECT_EQ(static_cast<size_t>(offset), strides.size());

    ASSERT_EQ(reconstructedShapes.size(), 2u);
    EXPECT_EQ(reconstructedShapes[0], (std::vector<int64_t>{8, 16, 32, 32}));
    EXPECT_EQ(reconstructedShapes[1], (std::vector<int64_t>{4, 64, 128}));

    ASSERT_EQ(reconstructedStrides.size(), 2u);
    EXPECT_EQ(reconstructedStrides[0], (std::vector<int64_t>{16384, 1024, 32, 1}));
    EXPECT_EQ(reconstructedStrides[1], (std::vector<int64_t>{8192, 128, 1}));
}

TEST_F(TestVariantPackDescriptorWhenInitialized, OverrideAttributesEmptyByDefaultBeforeFinalize)
{
    // When no OVERRIDE_* attributes are set, finalize() (after providing the required
    // base attributes) must not require them. Verify the override accessors return
    // empty vectors after finalize.
    std::array<void*, 1> devPtrs = {reinterpret_cast<void*>(0x1234)};
    std::array<int64_t, 1> uids = {1};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                             HIPDNN_TYPE_VOID_PTR,
                             static_cast<int64_t>(devPtrs.size()),
                             static_cast<const void*>(devPtrs.data()));
    _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                             HIPDNN_TYPE_INT64,
                             static_cast<int64_t>(uids.size()),
                             uids.data());
    _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                             HIPDNN_TYPE_VOID_PTR,
                             1,
                             static_cast<const void*>(&workspace));
    ASSERT_NO_THROW(_descriptor.finalize());

    EXPECT_TRUE(_descriptor.getOverrideUniqueIds().empty());
    EXPECT_TRUE(_descriptor.getOverrideShapes().empty());
    EXPECT_TRUE(_descriptor.getOverrideStrides().empty());
    EXPECT_TRUE(_descriptor.getOverrideLengths().empty());

    expectQueryThenGetInt64Vector(_descriptor, HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS, {});
    expectQueryThenGetInt64Vector(_descriptor, HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES, {});
    expectQueryThenGetInt64Vector(_descriptor, HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES, {});
    expectQueryThenGetInt64Vector(_descriptor, HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS, {});
}

} // namespace hipdnn_backend
