// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/Debug.hpp>

#include "DebugInstanceProbe.hpp"

// ROCM-31245 regression guard. This test binary is a separate link unit from
// tensilelite-host, so it stands in for libhipblaslt: if Debug can be
// duplicated across a shared-library boundary, it is duplicated here too.
// With a static tensilelite-host there is only one link unit and the checks
// below cannot fail, so they only run against a shared build.
namespace
{
    class ExcludedLibGuard
    {
    public:
        ExcludedLibGuard()
            : m_saved(TensileLite::Debug::Instance().excludedLibFromGetAll())
        {
        }

        ~ExcludedLibGuard()
        {
            TensileLite::Debug::Instance().setExcludedLibFromGetAll(m_saved);
        }

    private:
        TensileLite::StringSet m_saved;
    };

    class SingletonIdentity : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
#ifndef TENSILELITE_HOST_IS_SHARED
            GTEST_SKIP() << "tensilelite-host is static; there is no library boundary to test.";
#endif
        }
    };

    TEST_F(SingletonIdentity, DebugIsSharedAcrossLinkUnits)
    {
        EXPECT_EQ(static_cast<const void*>(&TensileLite::Debug::Instance()),
                  TensileLite::debugInstanceAddress())
            << "TensileLite::Debug is duplicated across the tensilelite-host boundary. "
               "Debug::Instance() must stay defined out of line in Debug.cpp.";
    }

    TEST_F(SingletonIdentity, DebugStatePropagatesAcrossLinkUnits)
    {
        ExcludedLibGuard restore;

        TensileLite::StringSet probe({"GridBasedMatching", "PredictionMatching"});
        TensileLite::Debug::Instance().setExcludedLibFromGetAll(probe);

        const auto* hostDebug
            = static_cast<const TensileLite::Debug*>(TensileLite::debugInstanceAddress());
        EXPECT_EQ(hostDebug->excludedLibFromGetAll(), probe)
            << "State written through Debug::Instance() here is not visible inside "
               "tensilelite-host.";
    }
} // namespace
