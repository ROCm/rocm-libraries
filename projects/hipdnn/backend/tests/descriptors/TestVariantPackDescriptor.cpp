// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/VariantDescriptor.hpp"
#include <gtest/gtest.h>

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
                             devPtrs.data());

    _descriptor.setAttribute(
        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, uids.size(), uids.data());

    _descriptor.setAttribute(
        HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &workspace);

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
                             devPtrs.data());

    _descriptor.setAttribute(
        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, uids.size(), uids.data());

    _descriptor.setAttribute(
        HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &workspace);
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
                             devPtrs.data());

    _descriptor.setAttribute(
        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, uids.size(), uids.data());

    _descriptor.setAttribute(
        HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &workspace);

    ASSERT_THROW(_descriptor.finalize(), HipdnnException);
    EXPECT_FALSE(_descriptor.isFinalized());
}

TEST_F(TestVariantPackDescriptorWhenInitialized, InvalidFinalizeUnsetParams)
{
    ASSERT_THROW(_descriptor.finalize(), HipdnnException);
    EXPECT_FALSE(_descriptor.isFinalized());
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
                                 _devPtrs.data());

        _descriptor.setAttribute(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                 HIPDNN_TYPE_INT64,
                                 static_cast<int64_t>(_uids.size()),
                                 _uids.data());

        _descriptor.setAttribute(
            HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &_workspace);

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
                             retrievedDevPtrs.data());
    EXPECT_EQ(elementCount, _devPtrs.size());
    EXPECT_EQ(
        std::memcmp(retrievedDevPtrs.data(), _devPtrs.data(), _devPtrs.size() * sizeof(void*)), 0);

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
                             &retrievedWorkspace);
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
                                                        retrievedDevPtrs.size(),
                                                        nullptr,
                                                        retrievedDevPtrs.data()),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

} // namespace hipdnn_backend
