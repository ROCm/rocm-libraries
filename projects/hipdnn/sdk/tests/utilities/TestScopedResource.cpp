// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/ScopedResource.hpp>

using namespace hipdnn_sdk;

namespace
{

struct Resource
{
    bool released = false;
};

} // namespace

TEST(TestScopedResource, BasicUsage)
{
    Resource r;

    {
        utilities::ScopedResource sr(&r, [](auto r) { r->released = true; });
        ASSERT_TRUE(!sr.isEmpty());
        ASSERT_EQ(sr.get()->released, r.released);
    }

    ASSERT_TRUE(r.released);
}

TEST(TestScopedResource, CheckMove)
{
    Resource r;

    {
        utilities::ScopedResource sr1(&r, [](auto r) { r->released = true; });

        utilities::ScopedResource sr2(std::move(sr1));
        ASSERT_TRUE(sr1.isEmpty());
        ASSERT_FALSE(sr2.isEmpty());
        ASSERT_FALSE(r.released);
    }

    ASSERT_TRUE(r.released);
}

TEST(TestScopedResource, CheckMoveAssign)
{
    Resource r1;
    Resource r2;

    {
        auto dtor = [](auto r) { r->released = true; };
        utilities::ScopedResource sr1(&r1, dtor);
        utilities::ScopedResource sr2(&r2, dtor);
        sr2 = std::move(sr1);

        ASSERT_FALSE(r1.released);
        ASSERT_TRUE(r2.released);

        ASSERT_TRUE(sr1.isEmpty());
        ASSERT_FALSE(sr2.isEmpty());
    }

    ASSERT_TRUE(r1.released);
}
