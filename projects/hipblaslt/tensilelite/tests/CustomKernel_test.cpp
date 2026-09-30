/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <Tensile/ContractionSolution.hpp>

#include "FallbackTestUtils.hpp"

using namespace TensileLite;
using TensileLite::testing::dummyProblem;
using TensileLite::testing::makeDevice;

// generateKernelCall rejects a solution whose metadata was never populated by
// testing threads.x / macrotile.x against zero. Every field is deserialized with
// mapOptional and vector3 leaves its components uninitialized by default, so that
// guard only means anything if an unpopulated CustomKernel really does hold zeros.
TEST(CustomKernelTest, DefaultsTripTheUninitializedMetadataGuard)
{
    CustomKernel kernel;

    EXPECT_EQ(kernel.threads.x, 0u);
    EXPECT_EQ(kernel.macrotile.x, 0u);
}

namespace
{
    // Just enough metadata to get past the guards in generateCustomCall, carrying a
    // single argument so the emitted buffer holds exactly that value.
    void configureProbeKernel(ContractionSolution& solution, CustomArgDefinition arg)
    {
        solution.sizeMapping.macroTile = TensileLite::dim3(128, 128, 1);
        solution.sizeMapping.depthU    = 64;
        // Pin the work-group mapping: the auto path reads it off a HipAMDGPU, which a
        // plain AMDGPU test device is not.
        solution.sizeMapping.workGroupMapping    = 1;
        solution.sizeMapping.workGroupMappingXCC = 1;

        solution.customKernel.name      = "probe";
        solution.customKernel.macrotile = TensileLite::dim3(128, 128, 64);
        solution.customKernel.threads   = TensileLite::dim3(256, 1, 1);
        solution.customKernel.grid
            = {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::One};
        solution.customKernel.args = {arg};
    }

    AMDGPU probeDevice()
    {
        return makeDevice(TensileLite::testing::_MI350_CHIP_ID,
                          TensileLite::testing::_SPX_CU,
                          "mi350spx");
    }

    uint32_t emittedSplitK(int16_t compiledGsu, int16_t runtimeGsu)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::SplitK});
        solution.sizeMapping.globalSplitU = compiledGsu;

        auto problem = dummyProblem();
        problem.setParams().setGSU(runtimeGsu);

        auto              device = probeDevice();
        ContractionInputs inputs;
        StreamKSettings   sk;

        auto invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);

        EXPECT_EQ(invocation.args.size(), sizeof(uint32_t));
        uint32_t splitK = 0;
        std::memcpy(&splitK, invocation.args.data(), sizeof(splitK));
        return splitK;
    }

    std::vector<uint8_t> emittedScalar(CustomArgSemantic semantic,
                                       rocisa::DataType  computeType,
                                       ConstantVariant   value)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::float32, semantic});
        solution.sizeMapping.globalSplitU = 1;

        auto problem = dummyProblem();
        problem.setAlphaType(computeType);
        problem.setBetaType(computeType);

        auto              device = probeDevice();
        ContractionInputs inputs;
        inputs.alpha = value;
        inputs.beta  = value;
        StreamKSettings sk;

        auto invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);

        auto const* bytes = static_cast<uint8_t const*>(invocation.args.data());
        return std::vector<uint8_t>(bytes, bytes + invocation.args.size());
    }
}

// The kernel wants log2(GSU). Taking it from the solution's compiled-in
// globalSplitU disagrees with the grid and workspace sizing, which are built from
// the runtime-effective GSU, so a user-supplied GSU has to win here too.
TEST(CustomKernelTest, SplitKFollowsTheRuntimeGsu)
{
    EXPECT_EQ(emittedSplitK(/*compiled*/ 1, /*runtime*/ 8), 3u);
    EXPECT_EQ(emittedSplitK(/*compiled*/ 1, /*runtime*/ 4), 2u);
    EXPECT_EQ(emittedSplitK(/*compiled*/ 8, /*runtime*/ 1), 0u);
}

// With no runtime override the effective GSU falls back to the compiled value, so
// the emitted argument is unchanged for kernels that never set one.
TEST(CustomKernelTest, SplitKFallsBackToTheCompiledGsu)
{
    EXPECT_EQ(emittedSplitK(/*compiled*/ 1, /*runtime*/ 0), 0u);
    EXPECT_EQ(emittedSplitK(/*compiled*/ 16, /*runtime*/ 0), 4u);
}

// A custom kernel declares alpha and beta as 32-bit slots, so a narrower compute
// type has to be widened before it is written or every argument after it shifts.
// KernelArguments only widens an argument spelled exactly "alpha" or "beta".
TEST(CustomKernelTest, ScalarsFillTheDeclaredThirtyTwoBitSlot)
{
    for(auto semantic : {CustomArgSemantic::Alpha, CustomArgSemantic::Beta})
    {
        EXPECT_EQ(emittedScalar(semantic, rocisa::DataType::Float, 1.5f).size(),
                  sizeof(float));

        auto const fromHalf
            = emittedScalar(semantic, rocisa::DataType::Half, static_cast<Half>(1.5f));
        ASSERT_EQ(fromHalf.size(), sizeof(float));

        float widened = 0.0f;
        std::memcpy(&widened, fromHalf.data(), sizeof(widened));
        EXPECT_EQ(widened, 1.5f);

        EXPECT_EQ(
            emittedScalar(semantic, rocisa::DataType::BFloat16, static_cast<BFloat16>(1.5f))
                .size(),
            sizeof(float));
    }
}

// Every CustomArgSemantic value, including the new SK4 individual
// (TotalItems/SKTiles/SKSplit/SKItersPerWI) and SK5 combined
// (...Or... entries, Tensile-internal only) semantics, must round-trip
// through toString/fromStringCustomArgSemantic -- this is what lets a
// custom.config YAML's `semantic:` string survive a deserialize/reserialize
// cycle unchanged.
TEST(CustomKernelTest, EveryCustomArgSemanticRoundTripsThroughItsString)
{
    for(int i = 0; i < static_cast<int>(CustomArgSemantic::COUNT); i++)
    {
        auto        semantic = static_cast<CustomArgSemantic>(i);
        std::string str      = toString(semantic);
        EXPECT_EQ(fromStringCustomArgSemantic(str), semantic) << "for '" << str << "'";
    }
}

namespace
{
    // Configures a probe kernel that declares the custom.config fields a
    // handwritten kernel needs to participate in GSU's MultipleBuffer
    // multi-buffer pattern (workspaceType + workspaceSizePerElemC), plus two
    // pointer args -- AddressC then AddressD, back to back with no padding
    // in between -- so the emitted buffer's first two 8-byte slots decode
    // directly to those addresses.
    void configureMultiBufferGsuKernel(ContractionSolution& solution,
                                       int16_t              gsu,
                                       size_t                workspaceSizePerElemC)
    {
        configureProbeKernel(solution, {CustomArgType::address, CustomArgSemantic::AddressC});
        solution.customKernel.args = {
            {CustomArgType::address, CustomArgSemantic::AddressC},
            {CustomArgType::address, CustomArgSemantic::AddressD},
        };
        solution.customKernel.workspaceType         = CustomWorkspaceType::SplitK;
        solution.customKernel.workspaceSizePerElemC = workspaceSizePerElemC;

        solution.sizeMapping.globalSplitU       = gsu;
        solution.sizeMapping.globalAccumulation = 2; // MultipleBuffer
        // Large enough that the conversion kernel's _PostGSU<N> suffix below
        // is not clamped down to something smaller than the requested gsu.
        solution.sizeMapping.globalSplitUPGR = 64;
    }
}

// GFA's multi-kernel path has existed but was never exercised end-to-end for
// a handwritten custom kernel: solve() chains a second (conversion/reduction)
// kernel onto GSU's MultipleBuffer mode identically for custom and
// Tensile-generated kernels, and generateCustomCall() already redirects
// AddressC/AddressD to the workspace for that mode -- but nothing called
// solve() on a custom kernel to confirm the two kernels, the workspace
// redirection, and the workspace sizing all actually agree with each other.
TEST(CustomKernelTest, MultipleBufferGsuChainsWorkspaceRedirectedConversionKernel)
{
    ContractionSolution solution;
    constexpr int16_t gsu                   = 4;
    constexpr size_t  workspaceSizePerElemC = 4;
    configureMultiBufferGsuKernel(solution, gsu, workspaceSizePerElemC);

    auto problem = dummyProblem();
    auto device  = probeDevice();

    int cStorage = 0, dStorage = 0, wsStorage = 0;
    ContractionInputs inputs;
    inputs.c  = &cStorage;
    inputs.d  = &dStorage;
    inputs.ws = &wsStorage;

    auto rv = solution.solve(problem, inputs, device);

    // Exactly two kernels: the main custom kernel and the shared
    // generateOutputConversionCall() reduction/conversion kernel -- no
    // beta-only pre-kernel (excluded for globalAccumulation == 2) and no
    // bias-gradient kernel (the dummy problem has no bias).
    ASSERT_EQ(rv.size(), 2u);

    // The main kernel's AddressC/AddressD args must be redirected to the
    // workspace, not the user's C/D: MultipleBuffer writes untouched partial
    // sums there for the conversion kernel to reduce.
    ASSERT_EQ(rv[0].args.size(), 2 * sizeof(void const*));
    void const* emittedAddressC = nullptr;
    void const* emittedAddressD = nullptr;
    std::memcpy(&emittedAddressC, rv[0].args.data(), sizeof(emittedAddressC));
    std::memcpy(&emittedAddressD,
               static_cast<uint8_t const*>(rv[0].args.data()) + sizeof(emittedAddressC),
               sizeof(emittedAddressD));
    EXPECT_EQ(emittedAddressC, inputs.ws);
    EXPECT_EQ(emittedAddressD, inputs.ws);
    EXPECT_NE(emittedAddressC, inputs.c);
    EXPECT_NE(emittedAddressD, inputs.d);

    // The conversion kernel's name carries a _PostGSU<N> suffix identifying
    // the (power-of-two-rounded, PGR-clamped) split factor it reduces.
    EXPECT_NE(rv[1].kernelName.find("_PostGSU4"), std::string::npos) << rv[1].kernelName;

    // Workspace sizing must follow the custom kernel's own macrotile and
    // workspaceSizePerElemC (what a handwritten kernel actually declares in
    // custom.config), not sizeMapping's -- only generated kernels populate
    // sizeMapping's copies of those fields.
    size_t tiles    = problem.getNumTiles(solution.sizeMapping, gsu) * problem.d().sizes()[2];
    size_t tileSize = solution.customKernel.macrotile.x * solution.customKernel.macrotile.y
                      * workspaceSizePerElemC;
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), tiles * tileSize);
}
