// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPolicyInfo.cpp
 * @brief Frontend unit tests for getLoadedHeuristicPolicyInfos
 *
 * Drives the two-call size/retrieve protocol through Mock_hipdnn_backend so
 * each branch (count failure, info failure, zero-policy fast path, NUL-strip
 * behavior, multi-policy enumeration) is exercised without a real backend.
 */

#include <cstring>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_frontend/HeuristicPolicyInfo.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::detail;
// using-directive instead of `using ::testing::_;` to avoid
// bugprone-reserved-identifier on the global-namespace `_`. Matches the
// pattern used in TestGraph.cpp.
using namespace ::testing;

namespace
{

// Sentinel handle: getLoadedHeuristicPolicyInfos rejects nullptr early, so any
// non-null pointer suffices for the mock path.
hipdnnHandle_t fakeHandle()
{
    static int s_sentinel = 0;
    return reinterpret_cast<hipdnnHandle_t>(&s_sentinel);
}

// Helper: install a getHeuristicPolicyInfo expectation that returns a single
// policy with the given fields. `nameLen` lets a caller choose whether the
// returned length includes the terminating NUL (matching the backend) or omits
// it (alternate-backend behavior worth exercising for the NUL-strip path).
void expectPolicyInfoReturning(Mock_hipdnn_backend& mock,
                               size_t index,
                               int64_t policyId,
                               const std::string& policyName,
                               const std::string& pluginName,
                               const std::string& pluginVersion,
                               const std::string& apiVersion,
                               bool includeNulInLen)
{
    const size_t lenAdjust = includeNulInLen ? 1u : 0u;

    EXPECT_CALL(mock, getHeuristicPolicyInfo(fakeHandle(), index, _, _, _, _, _, _, _, _, _))
        .WillRepeatedly([=](hipdnnHandle_t,
                            size_t,
                            int64_t* outPolicyId,
                            char* policyNameBuf,
                            size_t* policyNameLen,
                            char* pluginNameBuf,
                            size_t* pluginNameLen,
                            char* pluginVersionBuf,
                            size_t* pluginVersionLen,
                            char* apiVersionBuf,
                            size_t* apiVersionLen) {
            // Size-query mode: any string buffer is nullptr -> only fill lengths.
            const bool sizeQuery = (policyNameBuf == nullptr) || (pluginNameBuf == nullptr)
                                   || (pluginVersionBuf == nullptr) || (apiVersionBuf == nullptr);
            *policyNameLen = policyName.size() + lenAdjust;
            *pluginNameLen = pluginName.size() + lenAdjust;
            *pluginVersionLen = pluginVersion.size() + lenAdjust;
            *apiVersionLen = apiVersion.size() + lenAdjust;
            if(sizeQuery)
            {
                if(outPolicyId != nullptr)
                {
                    *outPolicyId = policyId;
                }
                return HIPDNN_STATUS_SUCCESS;
            }
            if(outPolicyId != nullptr)
            {
                *outPolicyId = policyId;
            }
            std::memcpy(policyNameBuf, policyName.c_str(), policyName.size());
            std::memcpy(pluginNameBuf, pluginName.c_str(), pluginName.size());
            std::memcpy(pluginVersionBuf, pluginVersion.c_str(), pluginVersion.size());
            std::memcpy(apiVersionBuf, apiVersion.c_str(), apiVersion.size());
            // NUL-terminate when the backend reported space for it; this mirrors
            // copyMaxSizeWithNullTerminator on the real path.
            if(includeNulInLen)
            {
                policyNameBuf[policyName.size()] = '\0';
                pluginNameBuf[pluginName.size()] = '\0';
                pluginVersionBuf[pluginVersion.size()] = '\0';
                apiVersionBuf[apiVersion.size()] = '\0';
            }
            return HIPDNN_STATUS_SUCCESS;
        });
}

} // namespace

class TestHeuristicPolicyInfo : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);

        ON_CALL(*_mockBackend, getErrorString(_)).WillByDefault(Return("mock backend error"));
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

TEST_F(TestHeuristicPolicyInfo, NullHandleReturnsErrorWithoutBackendCall)
{
    EXPECT_CALL(*_mockBackend, getHeuristicPolicyCount(_, _)).Times(0);
    EXPECT_CALL(*_mockBackend, getHeuristicPolicyInfo(_, _, _, _, _, _, _, _, _, _, _)).Times(0);

    auto [policies, err] = getLoadedHeuristicPolicyInfos(nullptr);

    EXPECT_TRUE(err.is_bad());
    EXPECT_TRUE(policies.empty());
}

TEST_F(TestHeuristicPolicyInfo, ZeroPoliciesReturnsEmptyVectorAndNoError)
{
    EXPECT_CALL(*_mockBackend, getHeuristicPolicyCount(fakeHandle(), _))
        .WillOnce(DoAll(SetArgPointee<1>(0u), Return(HIPDNN_STATUS_SUCCESS)));
    EXPECT_CALL(*_mockBackend, getHeuristicPolicyInfo(_, _, _, _, _, _, _, _, _, _, _)).Times(0);

    auto [policies, err] = getLoadedHeuristicPolicyInfos(fakeHandle());

    EXPECT_FALSE(err.is_bad());
    EXPECT_TRUE(policies.empty());
}

TEST_F(TestHeuristicPolicyInfo, CountFailurePropagatesAsBackendError)
{
    EXPECT_CALL(*_mockBackend, getHeuristicPolicyCount(fakeHandle(), _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getHeuristicPolicyInfo(_, _, _, _, _, _, _, _, _, _, _)).Times(0);

    auto [policies, err] = getLoadedHeuristicPolicyInfos(fakeHandle());

    EXPECT_TRUE(err.is_bad());
    EXPECT_TRUE(policies.empty());
    EXPECT_NE(err.get_message().find("count"), std::string::npos)
        << "Error message should reference the count call, got: " << err.get_message();
}

TEST_F(TestHeuristicPolicyInfo, InfoSizeQueryFailurePropagatesAsBackendError)
{
    EXPECT_CALL(*_mockBackend, getHeuristicPolicyCount(fakeHandle(), _))
        .WillOnce(DoAll(SetArgPointee<1>(1u), Return(HIPDNN_STATUS_SUCCESS)));
    // First (size-query) call fails immediately.
    EXPECT_CALL(*_mockBackend, getHeuristicPolicyInfo(fakeHandle(), 0u, _, _, _, _, _, _, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    auto [policies, err] = getLoadedHeuristicPolicyInfos(fakeHandle());

    EXPECT_TRUE(err.is_bad());
    EXPECT_TRUE(policies.empty());
    EXPECT_NE(err.get_message().find("index 0"), std::string::npos)
        << "Error message should reference the failing index, got: " << err.get_message();
}

TEST_F(TestHeuristicPolicyInfo, EnumeratesMultiplePoliciesAndStripsTrailingNul)
{
    EXPECT_CALL(*_mockBackend, getHeuristicPolicyCount(fakeHandle(), _))
        .WillOnce(DoAll(SetArgPointee<1>(2u), Return(HIPDNN_STATUS_SUCCESS)));

    expectPolicyInfoReturning(*_mockBackend,
                              /*index=*/0u,
                              /*policyId=*/0x1111,
                              /*policyName=*/"PolicyA",
                              /*pluginName=*/"plugin_a",
                              /*pluginVersion=*/"1.0.0",
                              /*apiVersion=*/"0.1",
                              /*includeNulInLen=*/true);
    expectPolicyInfoReturning(*_mockBackend,
                              /*index=*/1u,
                              /*policyId=*/0x2222,
                              /*policyName=*/"PolicyB",
                              /*pluginName=*/"plugin_b",
                              /*pluginVersion=*/"2.0.0",
                              /*apiVersion=*/"0.2",
                              /*includeNulInLen=*/true);

    auto [policies, err] = getLoadedHeuristicPolicyInfos(fakeHandle());

    ASSERT_FALSE(err.is_bad());
    ASSERT_EQ(policies.size(), 2u);

    EXPECT_EQ(policies[0].policyId, 0x1111);
    EXPECT_EQ(policies[0].policyName, "PolicyA");
    EXPECT_EQ(policies[0].pluginName, "plugin_a");
    EXPECT_EQ(policies[0].pluginVersion, "1.0.0");
    EXPECT_EQ(policies[0].apiVersion, "0.1");

    EXPECT_EQ(policies[1].policyId, 0x2222);
    EXPECT_EQ(policies[1].policyName, "PolicyB");

    // bufferToString must strip the trailing NUL that the backend reports as
    // part of the length, so the std::string never contains it.
    for(const auto& p : policies)
    {
        EXPECT_EQ(p.policyName.find('\0'), std::string::npos);
        EXPECT_EQ(p.pluginName.find('\0'), std::string::npos);
        EXPECT_EQ(p.pluginVersion.find('\0'), std::string::npos);
        EXPECT_EQ(p.apiVersion.find('\0'), std::string::npos);
    }
}

TEST_F(TestHeuristicPolicyInfo, AlternateBackendReportingLengthWithoutNulStillWorks)
{
    // Some hypothetical alternate backend implementation could report length
    // without the NUL terminator. The frontend must handle both reporting
    // conventions correctly because bufferToString only strips a trailing NUL
    // when one is actually present.
    EXPECT_CALL(*_mockBackend, getHeuristicPolicyCount(fakeHandle(), _))
        .WillOnce(DoAll(SetArgPointee<1>(1u), Return(HIPDNN_STATUS_SUCCESS)));
    expectPolicyInfoReturning(*_mockBackend,
                              /*index=*/0u,
                              /*policyId=*/42,
                              /*policyName=*/"PolicyX",
                              /*pluginName=*/"plug_x",
                              /*pluginVersion=*/"1",
                              /*apiVersion=*/"2",
                              /*includeNulInLen=*/false);

    auto [policies, err] = getLoadedHeuristicPolicyInfos(fakeHandle());

    ASSERT_FALSE(err.is_bad());
    ASSERT_EQ(policies.size(), 1u);
    EXPECT_EQ(policies[0].policyName, "PolicyX");
    EXPECT_EQ(policies[0].pluginName, "plug_x");
}

TEST_F(TestHeuristicPolicyInfo, SnakeCaseAliasDelegatesToCamelCase)
{
    EXPECT_CALL(*_mockBackend, getHeuristicPolicyCount(fakeHandle(), _))
        .WillOnce(DoAll(SetArgPointee<1>(0u), Return(HIPDNN_STATUS_SUCCESS)));

    auto [policies, err] = get_loaded_heuristic_policy_infos(fakeHandle());

    EXPECT_FALSE(err.is_bad());
    EXPECT_TRUE(policies.empty());
}
