// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/Debug.hpp>

#include "DebugInstanceProbe.hpp"

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
