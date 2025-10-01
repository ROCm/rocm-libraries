// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"
#include <cstring>
#include <gtest/gtest.h>

class IntegrationVariantPackDescriptorApi : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _varpack;

    void SetUp() override
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_varpack),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_varpack, nullptr);
    }

    void TearDown() override
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(_varpack), HIPDNN_STATUS_SUCCESS);
    }
};

TEST_F(IntegrationVariantPackDescriptorApi, ValidSetAttributesAndFinalize)
{
    std::array<void*, 3> devPtrs = {reinterpret_cast<void*>(0x1234),
                                    reinterpret_cast<void*>(0x5678),
                                    reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        devPtrs.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _varpack, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, 3, uids.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _varpack, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &workspace),
        HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendFinalize(_varpack), HIPDNN_STATUS_SUCCESS);
}

TEST_F(IntegrationVariantPackDescriptorApi, ValidSetAttributesAndGetAttributesBeforeFinalize)
{
    std::array<void*, 3> devPtrs = {reinterpret_cast<void*>(0x1234),
                                    reinterpret_cast<void*>(0x5678),
                                    reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        devPtrs.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetAttribute(
                  _varpack, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, 3, uids.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _varpack, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &workspace),
        HIPDNN_STATUS_SUCCESS);

    std::array<void*, 3> retrievedDevPtrs;
    int64_t elementCount = 0;

    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_INT64,
                                        3,
                                        &elementCount,
                                        retrievedDevPtrs.data()),
              HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(IntegrationVariantPackDescriptorApi, InvalidSetAttributes)
{
    std::array<void*, 3> devPtrs = {reinterpret_cast<void*>(0x1234),
                                    reinterpret_cast<void*>(0x5678),
                                    reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> uids = {1, 2, 3};
    void* workspace = reinterpret_cast<void*>(0xdeadbeef);

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS expects HIPDNN_TYPE_VOID_PTR
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _varpack, HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS, HIPDNN_TYPE_INT64, 3, devPtrs.data()),
        HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS expects HIPDNN_TYPE_INT64
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _varpack, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_VOID_PTR, 3, uids.data()),
        HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_WORKSPACE expects HIPDNN_TYPE_VOID_PTR
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _varpack, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 2, &workspace),
        HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since null ptr passed
    EXPECT_EQ(
        hipdnnBackendSetAttribute(
            _varpack, HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS, HIPDNN_TYPE_VOID_PTR, 3, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationVariantPackDescriptorApi, InvalidGetAttributes)
{
    std::array<void*, 3> retrievedDevPtrs;
    int64_t elementCount = 0;

    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_INT64,
                                        3,
                                        &elementCount,
                                        retrievedDevPtrs.data()),
              HIPDNN_STATUS_NOT_INITIALIZED);
}

// Invalid finalize since no devPtrs, or ids were set
TEST_F(IntegrationVariantPackDescriptorApi, InvalidFinalizeEmpty)
{
    EXPECT_EQ(hipdnnBackendFinalize(_varpack), HIPDNN_STATUS_BAD_PARAM);
}

// Invalid finalize since devPtrs, and ids do not have the same size.
TEST_F(IntegrationVariantPackDescriptorApi, InvalidFinalizeParams)
{
    std::array<void*, 3> devPtrs = {reinterpret_cast<void*>(0x1234),
                                    reinterpret_cast<void*>(0x5678),
                                    reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 2> uids = {1, 2};

    EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        devPtrs.size(),
                                        devPtrs.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                        HIPDNN_TYPE_INT64,
                                        uids.size(),
                                        uids.data()),
              HIPDNN_STATUS_SUCCESS);

    EXPECT_EQ(hipdnnBackendFinalize(_varpack), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationVariantPackDescriptorApi, InvalidNullDescriptor)
{
    hipdnnBackendDescriptor_t nullDescriptor = nullptr;
    int64_t dummyData = 42;
    auto status = hipdnnBackendSetAttribute(
        nullDescriptor, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, 1, &dummyData);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    int64_t retrievedData = 0;
    int64_t elementCount = 0;
    status = hipdnnBackendGetAttribute(nullDescriptor,
                                       HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &elementCount,
                                       &retrievedData);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

class IntegrationVariantPackDescriptorApiFinalized : public ::testing::Test
{
protected:
    hipdnnBackendDescriptor_t _varpack;
    std::array<void*, 3> _devPtrs = {reinterpret_cast<void*>(0x1234),
                                     reinterpret_cast<void*>(0x5678),
                                     reinterpret_cast<void*>(0x9abc)};
    std::array<int64_t, 3> _uids = {1, 2, 3};
    void* _workspace = reinterpret_cast<void*>(0xdeadbeef);

    void SetUp() override
    {
        EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &_varpack),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_varpack, nullptr);

        EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                            HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                            HIPDNN_TYPE_VOID_PTR,
                                            static_cast<int64_t>(_devPtrs.size()),
                                            _devPtrs.data()),
                  HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                            HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                            HIPDNN_TYPE_INT64,
                                            static_cast<int64_t>(_uids.size()),
                                            _uids.data()),
                  HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            hipdnnBackendSetAttribute(
                _varpack, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &_workspace),
            HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendFinalize(_varpack), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        EXPECT_EQ(hipdnnBackendDestroyDescriptor(_varpack), HIPDNN_STATUS_SUCCESS);
    }
};

TEST_F(IntegrationVariantPackDescriptorApiFinalized, ValidGetAttributes)
{
    std::array<void*, 3> retrievedDevPtrs;
    std::array<int64_t, 3> retrievedUids;
    void* retrievedWorkspace = nullptr;
    int64_t elementCount = 0;

    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        &elementCount,
                                        retrievedDevPtrs.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 3);
    EXPECT_EQ(
        std::memcmp(retrievedDevPtrs.data(), _devPtrs.data(), _devPtrs.size() * sizeof(void*)), 0);

    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                        HIPDNN_TYPE_INT64,
                                        3,
                                        &elementCount,
                                        retrievedUids.data()),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 3);
    EXPECT_EQ(std::memcmp(retrievedUids.data(), _uids.data(), _uids.size() * sizeof(int64_t)), 0);

    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                        HIPDNN_TYPE_VOID_PTR,
                                        1,
                                        &elementCount,
                                        &retrievedWorkspace),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(elementCount, 1);
    EXPECT_EQ(retrievedWorkspace, _workspace);
}

TEST_F(IntegrationVariantPackDescriptorApiFinalized, InvalidGetAttributes)
{
    std::array<void*, 3> retrievedDevPtrs;
    std::array<int64_t, 3> retrievedUids;
    void* retrievedWorkspace = nullptr;
    int64_t elementCount = 0;

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS expects HIPDNN_TYPE_VOID_PTR
    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_INT64,
                                        3,
                                        &elementCount,
                                        retrievedDevPtrs.data()),
              HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS expects HIPDNN_TYPE_INT64
    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        &elementCount,
                                        retrievedUids.data()),
              HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since HIPDNN_ATTR_VARIANT_PACK_WORKSPACE expects HIPDNN_TYPE_VOID_PTR
    // but elementCount should be 1
    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                        HIPDNN_TYPE_VOID_PTR,
                                        2,
                                        &elementCount,
                                        &retrievedWorkspace),
              HIPDNN_STATUS_BAD_PARAM);

    // HIPDNN_STATUS_BAD_PARAM since arrayOfElements cannot be nullptr
    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        &elementCount,
                                        nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    // HIPDNN_STATUS_BAD_PARAM since elementCount cannot be nullptr
    EXPECT_EQ(hipdnnBackendGetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        3,
                                        nullptr,
                                        retrievedDevPtrs.data()),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(IntegrationVariantPackDescriptorApiFinalized, InvalidSetAttributes)
{
    EXPECT_EQ(hipdnnBackendSetAttribute(_varpack,
                                        HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                        HIPDNN_TYPE_VOID_PTR,
                                        static_cast<int64_t>(_devPtrs.size()),
                                        _devPtrs.data()),
              HIPDNN_STATUS_NOT_INITIALIZED);
}
