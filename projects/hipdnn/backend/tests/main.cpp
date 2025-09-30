/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include "logging/Logging.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

int main(int argc, char** argv)
{
    hipdnn_backend::logging::initialize();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
