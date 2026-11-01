// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR)

#include <gtest/gtest.h>

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

// This suite is a placeholder: explicit-knob/forced-kernel GPU coverage for
// hipkernel:Gfx950ConvWgradTwoStageAot (its two-stage stage1+stage2 split-K dispatch) has
// not been written yet. See dnn-providers/hip-kernel-provider/rocke/WGRAD_AOT_PROGRESS.md
// for the engine's current coverage and the follow-up this test file tracks. Kept
// compiling and registered (rather than omitted) so the CMake wiring that gates on this
// engine's availability has a real source to build against.
TEST(Gfx950ConvWgradTwoStageAotKnobs, ExplicitKnobCoverageNotYetWritten)
{
    GTEST_SKIP() << "Explicit-knob/forced-kernel GPU cases for "
                    "hipkernel:Gfx950ConvWgradTwoStageAot are not implemented yet; see "
                    "WGRAD_AOT_PROGRESS.md.";
}

}  // namespace

}  // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif  // defined(HIPDNN_ENABLE_KERNEL_INGESTOR)
