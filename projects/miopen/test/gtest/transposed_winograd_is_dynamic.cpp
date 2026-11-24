// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <miopen/conv/solvers.hpp>

// The NHWC Winograd solvers wrap an NCHW Winograd solver with transpose kernels. A wrapper that
// reports IsDynamic() == false is dropped from the Find that runs on a find-db miss in the default
// (dynamic hybrid) find mode, and from immediate mode, so NHWC problems that are missing from the
// system db never get Winograd even though the inner solver is dynamic.
namespace {

template <class Wrapper, class Inner>
void ExpectSameDynamism()
{
    EXPECT_EQ(Wrapper{}.IsDynamic(), Inner{}.IsDynamic())
        << Wrapper{}.SolverDbId() << " must be as dynamic as the solver it wraps ("
        << Inner{}.SolverDbId() << ")";
}

} // namespace

using CPU_TransposedWinogradIsDynamic_NONE = testing::TestWithParam<int>;

TEST_P(CPU_TransposedWinogradIsDynamic_NONE, MatchesInnerSolver)
{
    namespace conv = miopen::solver::conv;

    ExpectSameDynamism<conv::TransposedConvBinWinograd3x3U, conv::ConvBinWinograd3x3U>();
    ExpectSameDynamism<conv::TransposedConvBinWinogradRxS, conv::ConvBinWinogradRxS>();
    ExpectSameDynamism<conv::TransposedConvBinWinogradRxSf2x3g1, conv::ConvBinWinogradRxSf2x3g1>();
    ExpectSameDynamism<conv::TransposedConvWinoFuryRxS<2, 3>, conv::ConvWinoFuryRxS<2, 3>>();
    ExpectSameDynamism<conv::TransposedConvWinoRageRxS<2, 3>, conv::ConvWinoRageRxS<2, 3>>();
    // Tunable wrapper.
    ExpectSameDynamism<conv::TransposedConvBinWinoRxS<2, 3>, conv::ConvBinWinoRxS<2, 3>>();
    ExpectSameDynamism<conv::TransposedConvBinWinoRxS<3, 2>, conv::ConvBinWinoRxS<3, 2>>();
    ExpectSameDynamism<conv::TransposedConvMPBidirectWinograd<2, 3>,
                       conv::ConvMPBidirectWinograd<2, 3>>();
    ExpectSameDynamism<conv::TransposedConvWinograd3x3MultipassWrW<3, 2>,
                       conv::ConvWinograd3x3MultipassWrW<3, 2>>();
    // ConvMPBidirectWinograd_xdlops is not covered: its explicit instantiations are not exported
    // from the Windows DLL, so a test cannot reference them. BinWinoRxS covers the tunable wrapper.
}

// Guards the premise of the test above: these inner solvers are dynamic, so their wrappers must
// be as well.
TEST_P(CPU_TransposedWinogradIsDynamic_NONE, DynamicInnerSolversGiveDynamicWrappers)
{
    namespace conv = miopen::solver::conv;

    EXPECT_TRUE(conv::TransposedConvBinWinogradRxSf2x3g1{}.IsDynamic());
    EXPECT_TRUE((conv::TransposedConvWinoRageRxS<2, 3>{}.IsDynamic()));
    EXPECT_TRUE((conv::TransposedConvBinWinoRxS<3, 2>{}.IsDynamic()));
}

INSTANTIATE_TEST_SUITE_P(Quick, CPU_TransposedWinogradIsDynamic_NONE, testing::Values(0));
