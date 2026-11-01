// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR)

#include <gtest/gtest.h>

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

// This suite is a placeholder: explicit-knob/forced-kernel GPU coverage for
// hipkernel:Gfx950ConvWgradAot (mirroring IntegrationGpuGfx950ConvDgrad.cpp's forced-tile
// cases) has not been written yet. See dnn-providers/hip-kernel-provider/rocke/
// WGRAD_AOT_PROGRESS.md for the engine's current coverage and the follow-up this test
// file tracks. Kept compiling and registered (rather than omitted) so the CMake wiring
// that gates on this engine's availability has a real source to build against.
TEST(Gfx950ConvWgradAotKnobs, ExplicitKnobCoverageNotYetWritten)
{
    GTEST_SKIP() << "Explicit-knob/forced-kernel GPU cases for hipkernel:Gfx950ConvWgradAot "
                    "are not implemented yet; see WGRAD_AOT_PROGRESS.md.";
}

}  // namespace

}  // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif  // defined(HIPDNN_ENABLE_KERNEL_INGESTOR)
