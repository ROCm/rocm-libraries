// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>

// Plugin log capture is set up per-test via hipdnnSetUserLogCallback_ext (see
// ScopedUserLogCallback in TestRockeClientAotLoad.cpp): plugin markers reach the
// process only through the backend logger, so a backend user callback is the
// reliable sink. No process-wide log recording is installed here.
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    return RUN_ALL_TESTS();
}
