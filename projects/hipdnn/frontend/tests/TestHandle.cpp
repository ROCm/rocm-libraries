// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_frontend/Handle.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::detail;
using namespace ::testing;

class TestHandle : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);

        ON_CALL(*_mockBackend, getLastErrorString(_, _))
            .WillByDefault([](char* errorString, size_t size) {
                const std::string fakeError = "Fake backend error";
                hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
                    errorString, fakeError.c_str(), size - 1);
            });
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

TEST_F(TestHandle, CreateHandlePairReturnSuccess)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x4567);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto [handle, error] = createHipdnnHandle();
    EXPECT_TRUE(error.is_good());
    EXPECT_NE(handle, nullptr);
    EXPECT_EQ(*handle, fakeHandle);
}

TEST_F(TestHandle, CreateHandlePairReturnFailure)
{
    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([](hipdnnHandle_t*) {
        return HIPDNN_STATUS_INTERNAL_ERROR;
    });

    auto [handle, error] = createHipdnnHandle();
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(handle, nullptr);
}

TEST_F(TestHandle, CreateHandlePairReturnWithStreamSuccess)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x5678);
    auto fakeStream = reinterpret_cast<hipStream_t>(0xCDEF);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, setStream(fakeHandle, fakeStream))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto [handle, error] = createHipdnnHandle(fakeStream);
    EXPECT_TRUE(error.is_good());
    EXPECT_NE(handle, nullptr);
    EXPECT_EQ(*handle, fakeHandle);
}

TEST_F(TestHandle, CreateHandlePairReturnWithStreamFailure)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x6789);
    auto fakeStream = reinterpret_cast<hipStream_t>(0xDEF0);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, setStream(fakeHandle, fakeStream))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto [handle, error] = createHipdnnHandle(fakeStream);
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(handle, nullptr);
}

TEST_F(TestHandle, CreateHandleSuccess)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x1234);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    HipdnnHandlePtr handle;
    auto error = createHipdnnHandle(handle);
    EXPECT_TRUE(error.is_good());
    EXPECT_NE(handle, nullptr);
    EXPECT_EQ(*handle, fakeHandle);
}

TEST_F(TestHandle, CreateHandleFailure)
{
    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([](hipdnnHandle_t*) {
        return HIPDNN_STATUS_INTERNAL_ERROR;
    });

    HipdnnHandlePtr handle;
    auto error = createHipdnnHandle(handle);
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(handle, nullptr);
}

TEST_F(TestHandle, CreateHandleWithStreamSuccess)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x2345);
    auto fakeStream = reinterpret_cast<hipStream_t>(0xABCD);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, setStream(fakeHandle, fakeStream))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    HipdnnHandlePtr handle;
    auto error = createHipdnnHandle(handle, fakeStream);
    EXPECT_TRUE(error.is_good());
    EXPECT_NE(handle, nullptr);
    EXPECT_EQ(*handle, fakeHandle);
}

TEST_F(TestHandle, CreateHandleWithStreamFailure)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x3456);
    auto fakeStream = reinterpret_cast<hipStream_t>(0xBCDE);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, setStream(fakeHandle, fakeStream))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    HipdnnHandlePtr handle;
    auto error = createHipdnnHandle(handle, fakeStream);
    EXPECT_TRUE(error.is_bad());
}

TEST_F(TestHandle, HandleDestroyedOnScopeExit)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x7890);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    {
        auto [handle, error] = createHipdnnHandle();
        EXPECT_TRUE(error.is_good());
        EXPECT_NE(handle, nullptr);
    }
    // destroy is verified by the EXPECT_CALL expectation
}

TEST_F(TestHandle, MoveTransfersOwnership)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x8901);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle))
        .Times(1)
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto [handle1, error] = createHipdnnHandle();
    EXPECT_TRUE(error.is_good());
    EXPECT_EQ(*handle1, fakeHandle);

    auto handle2 = std::move(handle1);
    EXPECT_EQ(handle1, nullptr); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(*handle2, fakeHandle);
}

TEST_F(TestHandle, NullHandleDeleterIsNoop)
{
    EXPECT_CALL(*_mockBackend, destroy(_)).Times(0);

    {
        const HipdnnHandlePtr handle{nullptr};
    }
}

TEST_F(TestHandle, DestroyerHandlesFailure)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0xBEEF);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    {
        HipdnnHandlePtr handle;
        auto error = createHipdnnHandle(handle);
        EXPECT_TRUE(error.is_good());
    }
    // Scope exits, deleter runs, destroy fails but logs error without throwing
}

TEST_F(TestHandle, SetStreamOnValidHandle)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0x9012);
    auto fakeStream = reinterpret_cast<hipStream_t>(0xEF01);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, setStream(fakeHandle, fakeStream))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto [handle, error] = createHipdnnHandle();
    EXPECT_TRUE(error.is_good());
    auto streamError = setHipdnnHandleStream(handle, fakeStream);
    EXPECT_TRUE(streamError.is_good());
}

TEST_F(TestHandle, SetStreamOnNullHandle)
{
    const HipdnnHandlePtr handle{nullptr};
    auto fakeStream = reinterpret_cast<hipStream_t>(0xF012);

    auto error = setHipdnnHandleStream(handle, fakeStream);
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), ErrorCode::INVALID_VALUE);
}

TEST_F(TestHandle, GetStreamFromNullHandle)
{
    const HipdnnHandlePtr handle{nullptr};
    hipStream_t stream = nullptr;

    auto error = getHipdnnHandleStream(handle, &stream);
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), ErrorCode::INVALID_VALUE);
}

TEST_F(TestHandle, GetStreamFromValidHandle)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0xA123);
    auto fakeStream = reinterpret_cast<hipStream_t>(0x1234);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, getStream(fakeHandle, _))
        .WillOnce([&fakeStream](hipdnnHandle_t, hipStream_t* out) {
            *out = fakeStream;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto [handle, error] = createHipdnnHandle();
    EXPECT_TRUE(error.is_good());
    hipStream_t stream = nullptr;
    auto streamError = getHipdnnHandleStream(handle, &stream);
    EXPECT_TRUE(streamError.is_good());
    EXPECT_EQ(stream, fakeStream);
}

TEST_F(TestHandle, GetStreamNullPointer)
{
    auto fakeHandle = reinterpret_cast<hipdnnHandle_t>(0xB234);

    EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
        *out = fakeHandle;
        return HIPDNN_STATUS_SUCCESS;
    });
    EXPECT_CALL(*_mockBackend, destroy(fakeHandle)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto [handle, error] = createHipdnnHandle();
    EXPECT_TRUE(error.is_good());
    auto streamError = getHipdnnHandleStream(handle, nullptr);
    EXPECT_TRUE(streamError.is_bad());
    EXPECT_EQ(streamError.get_code(), ErrorCode::INVALID_VALUE);
}
