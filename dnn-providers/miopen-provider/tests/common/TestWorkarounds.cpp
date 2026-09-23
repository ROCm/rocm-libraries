// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestWorkarounds.hpp"

#include <gtest/gtest.h>

TEST(TestWorkaroundsIssue6979, PredicateMatchesGraphNotSupported)
{
    const hipdnn_frontend::Error result(hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED,
                                        "no applicable engine");
    EXPECT_TRUE(IS_WORKAROUND_ISSUE_6979(result));
}

TEST(TestWorkaroundsIssue6979, PredicateRejectsBackendError)
{
    const hipdnn_frontend::Error result(hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                                        "real backend failure");
    EXPECT_FALSE(IS_WORKAROUND_ISSUE_6979(result));
}

TEST(TestWorkaroundsIssue6979, PredicateRejectsOk)
{
    const hipdnn_frontend::Error result;
    EXPECT_FALSE(IS_WORKAROUND_ISSUE_6979(result));
}
