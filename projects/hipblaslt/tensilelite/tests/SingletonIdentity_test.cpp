// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/Debug.hpp>

// ROCM-31245 regression guard.
//
// TensileLite::Debug is a LazySingleton: `Instance()` is an implicitly-inline
// member of a header-only class template holding a function-local static. Both
// libtensilelite-host and its consumers (libhipblaslt, this test binary) are
// built with -fvisibility=hidden and -fvisibility-inlines-hidden. Unless
// `Instance()` carries an explicit default-visibility attribute, each link unit
// gets its own private copy of the static, and there is no diagnostic of any
// kind: no link error, no warning, no crash.
//
// That is exactly what happened when TensileLite was split out of libhipblaslt
// into its own shared object. libhipblaslt wrote the
// "skip GridBasedMatching/PredictionMatching" set into its copy of Debug;
// ExactLogicLibrary::findAllSolutions read libtensilelite-host's copy, saw an
// empty set, and fully traversed the expensive ranking libraries on every
// hipblasLtMatmulAlgoGetHeuristic call -- a 100x-200x host-side regression.
//
// This test binary is a separate link unit from libtensilelite-host, so it
// stands in for libhipblaslt: if the singleton can duplicate across a library
// boundary at all, it duplicates here too.
namespace
{
    TEST(SingletonIdentity, DebugSingletonIsSharedAcrossLinkUnits)
    {
        const void* fromThisLinkUnit
            = static_cast<const void*>(&TensileLite::Debug::Instance());
        const void* fromTensileLiteHost = TensileLite::debugInstanceAddress();

        EXPECT_EQ(fromThisLinkUnit, fromTensileLiteHost)
            << "TensileLite::Debug has been duplicated across a link-unit boundary. "
               "Check that LazySingleton::Instance() still carries TENSILELITEHOST_EXPORT "
               "(Tensile/Singleton.hpp). See ROCM-31245.";
    }

    // Address equality is necessary but not sufficient on its own to notice a
    // regression -- verify that mutations actually propagate, which is the
    // property callers depend on.
    TEST(SingletonIdentity, DebugSingletonStatePropagatesAcrossLinkUnits)
    {
        const TensileLite::StringSet original
            = TensileLite::Debug::Instance().excludedLibFromGetAll();

        TensileLite::StringSet probe({"GridBasedMatching", "PredictionMatching"});
        TensileLite::Debug::Instance().setExcludedLibFromGetAll(probe);

        const auto* observed = static_cast<const TensileLite::Debug*>(
            TensileLite::debugInstanceAddress());
        EXPECT_EQ(observed->excludedLibFromGetAll().size(), 2u)
            << "State written through Debug::Instance() in this link unit is not visible "
               "to libtensilelite-host. See ROCM-31245.";

        TensileLite::StringSet restore(original);
        TensileLite::Debug::Instance().setExcludedLibFromGetAll(restore);
    }
} // namespace
