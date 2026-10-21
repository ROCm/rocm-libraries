// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <roc/hostnumerics/scalar_type.hpp>
#include <roc/hostnumerics/version.hpp>

TEST(Core, Version) {
    using namespace roc::hostnumerics;
    EXPECT_EQ(version(), "0.1.0");
    EXPECT_EQ(scalarTypeInfo(ScalarType::Float16).storageBits, 16);
}
