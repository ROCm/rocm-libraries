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

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>
#ifndef TENSILE_YAML
#include <Tensile/msgpack/MessagePack.hpp>
#endif

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

#ifndef TENSILE_YAML
// A kernel's uniformSummationOrder declaration reaches hipBLASLt only through
// this mapping; lost, every declaring kernel would quietly be refused again
// under uniform summation order.
TEST(CustomKernelTest, UniformSummationOrderDeserializes)
{
    for(const bool declared : {false, true})
    {
        msgpack::zone                          zone;
        std::map<std::string, msgpack::object> fields{
            {"name", msgpack::object(std::string("probe"), zone)}};
        if(declared)
            fields["uniformSummationOrder"] = msgpack::object(true, zone);

        msgpack::sbuffer packed;
        msgpack::pack(packed, fields);
        auto unpacked = msgpack::unpack(packed.data(), packed.size());

        CustomKernel                    kernel;
        Serialization::MessagePackInput reader(unpacked.get());
        reader.input(kernel);
        ASSERT_TRUE(reader.error.empty()) << reader.error.front();
        EXPECT_EQ(kernel.name, "probe");
        EXPECT_EQ(kernel.uniformSummationOrder, declared);
    }
}

// hipBLASLt's rocRoller-off selector matches a rocRoller kernel on its declared
// unroll. A kernel that declares none reads back as 0, which it never picks.
TEST(CustomKernelTest, UnrollDeserializes)
{
    for(const int declared : {0, 2, 4})
    {
        msgpack::zone                          zone;
        std::map<std::string, msgpack::object> fields{
            {"name", msgpack::object(std::string("probe"), zone)}};
        if(declared != 0)
            fields["unroll"] = msgpack::object(declared, zone);

        msgpack::sbuffer packed;
        msgpack::pack(packed, fields);
        auto unpacked = msgpack::unpack(packed.data(), packed.size());

        CustomKernel                    kernel;
        Serialization::MessagePackInput reader(unpacked.get());
        reader.input(kernel);
        ASSERT_TRUE(reader.error.empty()) << reader.error.front();
        EXPECT_EQ(kernel.unroll, declared);
    }
}

namespace
{
    // Reads one {type: uint32, semantic: <name>} argument the way a logic file carries it.
    std::vector<std::string> readArgSemantic(std::string const& name, CustomArgSemantic& semantic)
    {
        msgpack::zone                          zone;
        std::map<std::string, msgpack::object> fields{
            {"type", msgpack::object(std::string("uint32"), zone)},
            {"semantic", msgpack::object(name, zone)}};

        msgpack::sbuffer packed;
        msgpack::pack(packed, fields);
        auto unpacked = msgpack::unpack(packed.data(), packed.size());

        CustomArgDefinition             arg;
        Serialization::MessagePackInput reader(unpacked.get());
        reader.input(arg);
        semantic = arg.semantic;
        return reader.error;
    }
}

// rocRoller names its N tile count numTiles1 (dimension 1). The semantics are
// named by N, and the dimension-index spelling is not a semantic.
TEST(CustomKernelTest, StreamKTileCountSemanticsDeserialize)
{
    for(auto const& [name, expected] :
        {std::pair{"RR_SKNumTilesN", CustomArgSemantic::RR_SKNumTilesN},
         std::pair{"RR_SKMagicMultipleNumTilesN", CustomArgSemantic::RR_SKMagicMultipleNumTilesN},
         std::pair{"RR_SKMagicShiftsNumTilesN", CustomArgSemantic::RR_SKMagicShiftsNumTilesN}})
    {
        CustomArgSemantic semantic{};
        auto const        errors = readArgSemantic(name, semantic);
        EXPECT_TRUE(errors.empty()) << name;
        EXPECT_EQ(semantic, expected) << name;
    }

    for(auto const* name :
        {"RR_SKNumTiles1", "RR_SKMagicMultipleNumTiles1", "RR_SKMagicShiftsNumTiles1"})
    {
        CustomArgSemantic semantic{};
        EXPECT_FALSE(readArgSemantic(name, semantic).empty()) << name;
    }
}
#endif

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

    KernelInvocation splitKProbeCall(int16_t        compiledGsu,
                                     int16_t        runtimeGsu,
                                     CustomGridSize gridY = CustomGridSize::TilesYGSU)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::SplitK});
        solution.customKernel.grid.y      = gridY;
        solution.sizeMapping.globalSplitU = compiledGsu;

        auto problem = dummyProblem();
        problem.setParams().setGSU(runtimeGsu);

        auto              device = probeDevice();
        ContractionInputs inputs;
        StreamKSettings   sk;

        return solution.generateCustomCall<false>(problem, inputs, device, sk);
    }

    uint32_t emittedSplitK(int16_t compiledGsu, int16_t runtimeGsu)
    {
        auto invocation = splitKProbeCall(compiledGsu, runtimeGsu);

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

// A split-K kernel reduces into D only once every GSU slice of a tile has arrived,
// so TilesYGSU has to launch one row of work groups per slice, following the
// runtime-effective GSU like the SplitK argument does.
TEST(CustomKernelTest, TilesYGsuLaunchesEveryGsuSlice)
{
    auto const unsplit = splitKProbeCall(/*compiled*/ 1, /*runtime*/ 0).numWorkGroups;

    auto const compiled = splitKProbeCall(/*compiled*/ 16, /*runtime*/ 0).numWorkGroups;
    EXPECT_EQ(compiled.x, unsplit.x);
    EXPECT_EQ(compiled.y, 16 * unsplit.y);
    EXPECT_EQ(compiled.z, unsplit.z);

    EXPECT_EQ(splitKProbeCall(/*compiled*/ 1, /*runtime*/ 4).numWorkGroups.y, 4 * unsplit.y);
    EXPECT_EQ(splitKProbeCall(/*compiled*/ 0, /*runtime*/ 0).numWorkGroups.y, unsplit.y);
    EXPECT_EQ(
        splitKProbeCall(/*compiled*/ 1, /*runtime*/ 0, CustomGridSize::TilesY).numWorkGroups.y,
        unsplit.y);
}

// A grid with no GSU term would launch one slice per tile and report success with
// D unwritten, so a split-K launch on one has to fail instead.
TEST(CustomKernelTest, GridWithoutGsuTermRejectsSplitK)
{
    EXPECT_THROW(splitKProbeCall(/*compiled*/ 16, /*runtime*/ 0, CustomGridSize::TilesY),
                 std::runtime_error);
    EXPECT_THROW(splitKProbeCall(/*compiled*/ 1, /*runtime*/ 4, CustomGridSize::TilesY),
                 std::runtime_error);
    EXPECT_NO_THROW(splitKProbeCall(/*compiled*/ 1, /*runtime*/ 0, CustomGridSize::TilesY));
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

namespace
{
    // hipBLASLt passes compact scales; the TensileLite client uses the padded
    // descriptor. Keep both caller layouts available to the argument probes.
    ContractionProblemGemm mxScaleProblem(size_t M,
                                          size_t N,
                                          size_t K,
                                          bool   transA,
                                          bool   transB,
                                          int    mxBlock      = 32,
                                          bool   packedScales = true)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(transA,
                                                            transB,
                                                            rocisa::DataType::Float4,
                                                            rocisa::DataType::Float4,
                                                            rocisa::DataType::BFloat16,
                                                            rocisa::DataType::BFloat16,
                                                            M,
                                                            N,
                                                            K,
                                                            1,
                                                            transA ? K : M,
                                                            transA ? K * M : M * K,
                                                            transB ? N : K,
                                                            transB ? N * K : K * N,
                                                            M,
                                                            M * N,
                                                            M,
                                                            M * N,
                                                            0.0);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        problem.setMXScaleA(rocisa::DataType::E8, mxBlock);
        problem.setMXScaleB(rocisa::DataType::E8, mxBlock);
        problem.setCustomKernelMXScalesArePacked(packedScales);
        return problem;
    }

    // The int64 this problem emits into a probe kernel's single scale-stride slot.
    int64_t emittedScaleStride(CustomArgSemantic             semantic,
                               ContractionProblemGemm const& problem,
                               int                           argsVersion       = 0,
                               bool                          useInitialStrides = false)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::int64, semantic});
        solution.sizeMapping.globalSplitU                      = 1;
        solution.internalArgsSupport.persistentLoopArgsVersion = argsVersion;
        solution.problemType.useInitialStridesAB               = useInitialStrides;
        if(argsVersion == 1)
        {
            // Version 1 is valid only for a persistent data-parallel solution
            // with KernArgsVersion 3.
            solution.internalArgsSupport.version        = 3;
            solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
            solution.customKernel.generated             = true;
        }

        auto              device = probeDevice();
        ContractionInputs inputs;
        StreamKSettings   sk;

        auto invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);

        EXPECT_EQ(invocation.args.size(), sizeof(int64_t));
        int64_t stride = 0;
        std::memcpy(&stride, invocation.args.data(), sizeof(stride));
        return stride;
    }
}

// A compact buffer needs the unpadded pitch when ceil(K/mxBlock) is not a
// multiple of 8; using the descriptor pitch would misaddress later rows.
TEST(CustomKernelTest, ScaleStrideIsCompactNotPadded)
{
    // TN: K is the fast (bound) dim of both A and B, so the padded K-block count
    // would leak straight into the stride.
    struct
    {
        size_t  K;
        int64_t compact;
        int64_t padded;
    } const tnCases[] = {
        {128, 4, 8}, // ceil(K/32) = 4  -> padded to 8
        {384, 12, 16}, // ceil(K/32) = 12 -> padded to 16
        {256, 8, 8}, // ceil(K/32) = 8  -> already aligned, the two agree
    };

    for(auto const& c : tnCases)
    {
        auto problem = mxScaleProblem(64, 64, c.K, /*transA*/ true, /*transB*/ false);

        // The descriptor really does carry the padding; we are diverging from it.
        EXPECT_EQ(problem.mxsa().strides()[1], static_cast<size_t>(c.padded)) << "K=" << c.K;
        EXPECT_EQ(problem.mxsb().strides()[1], static_cast<size_t>(c.padded)) << "K=" << c.K;

        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem), c.compact)
            << "K=" << c.K;
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem), c.compact)
            << "K=" << c.K;
    }
}

// Both argument versions must preserve the caller's batch layout, including
// padding in the free dimension as well as the K-block dimension.
TEST(CustomKernelTest, ScaleBatchStrideFollowsTheCallerBuffer)
{
    auto packedProblem = mxScaleProblem(60, 60, 128, true, false);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA1, packedProblem), 4 * 60);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB1, packedProblem), 4 * 60);
    for(bool packed : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "packed=" << packed);
        auto          problem  = mxScaleProblem(60, 60, 128, true, false, 32, packed);
        const int64_t expected = packed ? 4 * 60 : 8 * 64;
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA1, problem, 1), expected);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB1, problem, 1), expected);
    }
}

TEST(CustomKernelTest, ScaleStridesIncludeTheInitialStrideForDataParallelArgs)
{
    for(bool packed : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "packed=" << packed);
        auto problem = mxScaleProblem(60, 60, 128, true, false, 32, packed);
        // With initial strides enabled, slots 0 and 1 describe tensor dimensions 0 and 1.
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem, 1, true), 1);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem, 1, true), 1);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA1, problem, 1, true),
                  packed ? 4 : 8);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB1, problem, 1, true),
                  packed ? 4 : 8);
    }
}

TEST(CustomKernelTest, ScaleStrideRoundsUpPartialBlocks)
{
    for(int block : {16, 32})
    {
        SCOPED_TRACE(::testing::Message() << "block=" << block);
        auto          problem  = mxScaleProblem(60, 60, 129, true, false, block);
        const int64_t expected = block == 16 ? 9 : 5;
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem), expected);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem), expected);
    }
}

// The TensileLite client allocates and fills the descriptor's padded layout.
// Compact strides would make some rows read padding instead of their scales.
TEST(CustomKernelTest, ScaleStrideFollowsDescriptorWhenCallerDoesNotPack)
{
    struct
    {
        size_t  K;
        int64_t padded; // what the descriptor says, and what must be emitted
    } const tnCases[] = {
        {128, 8}, // ceil(K/32) = 4  -> padded to 8;  packed would have said 4
        {384, 16}, // ceil(K/32) = 12 -> padded to 16; packed would have said 12
        {256, 8}, // ceil(K/32) = 8  -> already aligned, the two agree
    };

    for(auto const& c : tnCases)
    {
        auto problem = mxScaleProblem(
            64, 64, c.K, /*transA*/ true, /*transB*/ false, /*mxBlock*/ 32, /*packedScales*/ false);

        EXPECT_EQ(problem.mxsa().strides()[1], static_cast<size_t>(c.padded)) << "K=" << c.K;
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem), c.padded)
            << "K=" << c.K;
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem), c.padded)
            << "K=" << c.K;
    }

    // Same for the K-slow orientation the packed path handles specially.
    auto nn = mxScaleProblem(
        60, 60, 128, /*transA*/ false, /*transB*/ false, /*mxBlock*/ 32, /*packedScales*/ false);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, nn), 64);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, nn), 8);
}

// hipBLASLt takes the A and B scale formats independently, so a pre-swizzled
// operand keeps the padded strides while the other operand gets compact ones.
TEST(CustomKernelTest, ScaleStrideFollowsEachOperandsBuffer)
{
    for(bool packedA : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "packedA=" << packedA);
        auto problem = mxScaleProblem(60, 60, 128, /*transA*/ true, /*transB*/ false);
        problem.setCustomKernelMXScaleAIsPacked(packedA);
        problem.setCustomKernelMXScaleBIsPacked(!packedA);

        // ceil(128/32) = 4 K-blocks pad to 8, and M = N = 60 pad to 64.
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem), packedA ? 4 : 8);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem), packedA ? 8 : 4);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA1, problem, 1),
                  packedA ? 4 * 60 : 8 * 64);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB1, problem, 1),
                  packedA ? 8 * 64 : 4 * 60);
    }
}

// NN places K on different stored axes in A and B. Unaligned M and K make
// compact and padded pitches differ for both operands.
TEST(CustomKernelTest, ScaleStrideIsCompactForEitherBoundDimension)
{
    auto problem = mxScaleProblem(60, 60, 128, /*transA*/ false, /*transB*/ false);

    // A: bound dim is index 1, so the stride is A's free dim, padded to 32.
    EXPECT_EQ(problem.mxsa().strides()[1], 64u);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem), 60);

    // B: bound dim is index 0, so the stride is the K-block count, padded to 8.
    EXPECT_EQ(problem.mxsb().strides()[1], 8u);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem), 4);
}

// rocRoller's non-TN MX kernels read the K extent of A's scale tensor, which
// does not depend on the layout.
TEST(CustomKernelTest, SizeSumScaleAIsTheScaleBlockCountAlongK)
{
    for(bool transA : {true, false})
    {
        SCOPED_TRACE(::testing::Message() << "transA=" << transA);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::SizeSumScaleA,
                                     mxScaleProblem(64, 64, 128, transA, false)),
                  4);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::SizeSumScaleA,
                                     mxScaleProblem(64, 64, 96, transA, true)),
                  3);
    }
}

namespace
{
    // Grid x and ComputeUnits kernarg of a persistent probe kernel.
    std::pair<size_t, int32_t> computeUnitsLaunch(int smCountTarget, int persistentMaxCUs)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::int32, CustomArgSemantic::ComputeUnits});
        solution.customKernel.grid
            = {CustomGridSize::ComputeUnits, CustomGridSize::One, CustomGridSize::One};

        auto problem = dummyProblem();
        problem.setParams().setSmCountTarget(smCountTarget);
        auto device             = probeDevice();
        device.persistentMaxCUs = persistentMaxCUs;
        ContractionInputs inputs;
        StreamKSettings   sk;

        auto invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);

        EXPECT_EQ(invocation.numWorkGroups.y, 1u);
        EXPECT_EQ(invocation.numWorkGroups.z, 1u);
        EXPECT_EQ(invocation.args.size(), sizeof(int32_t));
        int32_t cuCount = 0;
        std::memcpy(&cuCount, invocation.args.data(), sizeof(cuCount));
        return {invocation.numWorkGroups.x, cuCount};
    }
}

// Persistent skinny-GEMM kernels stride by CU count; the launch grid and the
// ComputeUnits kernarg have to be the same value.
TEST(CustomKernelTest, ComputeUnitsGridAndArgFollowHardware)
{
    constexpr int cus = TensileLite::testing::_SPX_CU;
    EXPECT_EQ(computeUnitsLaunch(0, 0), std::make_pair(size_t{cus}, int32_t{cus}));
}

// The SM-count target and persistentMaxCUs cap that value as they cap Tensile's
// persistent grids, and the tighter cap wins.
TEST(CustomKernelTest, ComputeUnitsHonorTheCuBudget)
{
    constexpr int cus = TensileLite::testing::_SPX_CU;
    EXPECT_EQ(computeUnitsLaunch(64, 0), std::make_pair(size_t{64}, int32_t{64}));
    EXPECT_EQ(computeUnitsLaunch(64, 48), std::make_pair(size_t{48}, int32_t{48}));
    EXPECT_EQ(computeUnitsLaunch(0, 96), std::make_pair(size_t{96}, int32_t{96}));
    EXPECT_EQ(computeUnitsLaunch(4 * cus, 0), std::make_pair(size_t{cus}, int32_t{cus}));
}

namespace
{
    ContractionProblemGemm rocRollerStreamKProblem(size_t m, size_t n)
    {
        auto problem = ContractionProblemGemm::GEMM(false,
                                                    false,
                                                    m,
                                                    n,
                                                    1024,
                                                    std::max<size_t>(m, 1),
                                                    1024,
                                                    std::max<size_t>(m, 1),
                                                    1.0,
                                                    false,
                                                    1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        return problem;
    }

    // A rocRoller Stream-K probe kernel with square macrotiles. With no
    // occupancy recorded its grid is the CU count, capped by smCountTarget.
    // With readsKernargs it also reads its grid and data-parallel tile count;
    // without, it only launches on the grid and binds its partials.
    struct RocRollerStreamKProbe
    {
        ContractionSolution    solution;
        ContractionProblemGemm problem;

        RocRollerStreamKProbe(uint32_t tile,
                              int      smCountTarget,
                              bool     readsKernargs = true,
                              size_t   m             = 1024,
                              size_t   n             = 1024)
            : problem(rocRollerStreamKProblem(m, n))
        {
            if(readsKernargs)
            {
                configureProbeKernel(solution,
                                     {CustomArgType::uint32, CustomArgSemantic::RR_SKNumWGs});
                solution.customKernel.args.push_back(
                    {CustomArgType::uint32, CustomArgSemantic::RR_SKNumDPTiles});
            }
            else
                configureProbeKernel(
                    solution, {CustomArgType::address, CustomArgSemantic::RR_SKAddressPartials});
            solution.customKernel.macrotile = TensileLite::dim3(tile, tile, 128);
            solution.customKernel.grid
                = {CustomGridSize::RocRollerStreamKGrid, CustomGridSize::One, CustomGridSize::One};
            problem.setParams().setSmCountTarget(smCountTarget);
        }

        KernelInvocation launch() const
        {
            ContractionInputs inputs;
            StreamKSettings   sk;
            return solution.generateCustomCall<false>(problem, inputs, probeDevice(), sk);
        }

        bool refused() const
        {
            return solution.rocRollerStreamKHasDataParallelTiles(problem, probeDevice());
        }
    };
}

// The launch grid and the numWGs kernarg are one value, and a rocRoller Stream-K
// kernel launches exactly where its grid leaves no data-parallel tiles: up to
// but not including two tiles per workgroup. Selection refuses the same points.
TEST(CustomKernelTest, RocRollerStreamKLaunchRefusesDataParallelTiles)
{
    constexpr uint32_t cus = TensileLite::testing::_SPX_CU;

    // 1024x1024 in 128x128 tiles is 64 tiles.
    struct Case
    {
        int  grid;
        bool launches;
    };
    for(Case c : {Case{0, true}, // 64 tiles < 256 workgroups
                  Case{64, true}, // tiles == grid
                  Case{48, true}, // grid < tiles < 2 * grid
                  Case{33, true}, // just below 2 * grid
                  Case{32, false}, // tiles == 2 * grid
                  Case{8, false}})
    {
        uint32_t const        grid = c.grid == 0 ? cus : static_cast<uint32_t>(c.grid);
        RocRollerStreamKProbe probe(128, c.grid);
        EXPECT_EQ(probe.refused(), !c.launches) << "grid " << grid;
        if(!c.launches)
        {
            EXPECT_THROW(probe.launch(), std::runtime_error) << "grid " << grid;
            continue;
        }
        auto const fits = probe.launch();
        EXPECT_EQ(fits.numWorkGroups.x, grid);
        ASSERT_EQ(fits.args.size(), 2 * sizeof(uint32_t));
        uint32_t args[2] = {};
        std::memcpy(args, fits.args.data(), sizeof(args));
        EXPECT_EQ(args[0], grid);
        EXPECT_EQ(args[1], 0u) << "grid " << grid;
    }

    // 4096 tiles of 16x16 on the full grid: 16 per workgroup.
    EXPECT_THROW(RocRollerStreamKProbe(16, 0).launch(), std::runtime_error);
}

// The refusal hangs off the grid, not off the kernargs: a kernel that only
// launches on rocRoller's grid is refused at the same point.
TEST(CustomKernelTest, RocRollerStreamKGridAloneIsEnoughToRefuse)
{
    EXPECT_NO_THROW(RocRollerStreamKProbe(128, 33, false).launch());
    RocRollerStreamKProbe const tooSmall(128, 32, false);
    EXPECT_TRUE(tooSmall.refused());
    EXPECT_THROW(tooSmall.launch(), std::runtime_error);
}

// An empty problem has no tiles, on which origami divides by zero: the grid
// falls back to the CU budget without reaching it, and nothing is refused.
TEST(CustomKernelTest, RocRollerStreamKEmptyProblemLaunches)
{
    hip::HipAMDGPU device;
    device.processor          = AMDGPU::Processor::gfx950;
    device.computeUnitCount   = TensileLite::testing::_SPX_CU;
    device.analyticalHardware = std::make_shared<origami::hardware_t>(
        origami::hardware_t::get_hardware_for_arch(origami::hardware_t::architecture_t::gfx950,
                                                   TensileLite::testing::_SPX_CU,
                                                   163840,
                                                   262144,
                                                   4 * 1024 * 1024,
                                                   2100000));

    for(auto [m, n] : {std::pair<size_t, size_t>{0, 1024}, {1024, 0}, {0, 0}})
    {
        RocRollerStreamKProbe probe(128, 16, true, m, n);
        probe.solution.customKernel.occupancy = 2;
        EXPECT_FALSE(probe.solution.rocRollerStreamKHasDataParallelTiles(probe.problem, device))
            << m << "x" << n;
        ContractionInputs inputs;
        StreamKSettings   sk;
        auto const        launch
            = probe.solution.generateCustomCall<false>(probe.problem, inputs, device, sk);
        EXPECT_EQ(launch.numWorkGroups.x, 16u) << m << "x" << n;
    }
}

// Only rocRoller Stream-K kernels are refused, and only they resolve its grid.
TEST(CustomKernelTest, OnlyRocRollerStreamKKernelsUseItsGrid)
{
    ContractionSolution plain;
    configureProbeKernel(plain, {CustomArgType::uint32, CustomArgSemantic::SplitK});
    EXPECT_FALSE(plain.usesRocRollerStreamK());

    // 4096 tiles of 16x16 would be refused on any Stream-K grid.
    plain.customKernel.macrotile = TensileLite::dim3(16, 16, 128);
    EXPECT_FALSE(plain.rocRollerStreamKHasDataParallelTiles(dummyProblem(), probeDevice()));

    ContractionSolution kernargOnly;
    configureProbeKernel(kernargOnly, {CustomArgType::uint32, CustomArgSemantic::RR_SKNumWGs});
    EXPECT_TRUE(kernargOnly.usesRocRollerStreamK());

    ContractionSolution workspaceOnly;
    configureProbeKernel(workspaceOnly, {CustomArgType::uint32, CustomArgSemantic::SplitK});
    workspaceOnly.customKernel.workspaceType = CustomWorkspaceType::RocRollerStreamK;
    EXPECT_TRUE(workspaceOnly.usesRocRollerStreamK());

    // A Tensile-generated kernel is never launched on rocRoller's grid.
    workspaceOnly.customKernel.generated = true;
    EXPECT_FALSE(workspaceOnly.usesRocRollerStreamK());
}

namespace
{
    char const fusedPeerFlag = 0;
    char const fusedCounter  = 0;

    constexpr size_t peerGroupBytes     = FUSED_A2A_SLOT_COUNT * sizeof(void*);
    constexpr size_t peer0FlagOffset    = 0;
    constexpr size_t peer1FlagOffset    = peer0FlagOffset + peerGroupBytes;
    constexpr size_t counterOffset      = peer0FlagOffset + FUSED_A2A_MAX_RANKS * peerGroupBytes;
    constexpr size_t fusedMyRankOffset  = counterOffset + sizeof(void*);
    constexpr size_t fusedWOffset       = fusedMyRankOffset + sizeof(uint32_t);
    constexpr size_t fusedDrainOffset   = fusedWOffset + sizeof(uint32_t);
    constexpr size_t fusedAMOffset      = fusedDrainOffset + sizeof(uint32_t);

    template <typename T>
    T packedAt(KernelArguments const& args, size_t offset)
    {
        T result{};
        std::memcpy(&result, static_cast<uint8_t const*>(args.data()) + offset, sizeof(T));
        return result;
    }

    KernelInvocation fusedCustomCall()
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::FusedA2ASegment});

        auto problem = dummyProblem();
        problem.setFusedGemmA2A(true);
        problem.setWorld(2);
        problem.setFusedA2AExtent(256);

        FusedA2APeerFields peer0{};
        peer0[FUSED_A2A_SLOT_FLAG_PTR] = const_cast<char*>(&fusedPeerFlag);

        ContractionInputs inputs;
        inputs.fusedA2APeers   = {peer0, FusedA2APeerFields{}};
        inputs.fusedA2ACounter = const_cast<char*>(&fusedCounter);
        inputs.fusedA2AMyRank  = 1;
        inputs.fusedA2ADrain   = FUSED_A2A_DRAIN_RECV;

        return solution.generateCustomCall<false>(
            problem, inputs, probeDevice(), StreamKSettings{});
    }
}

TEST(CustomKernelTest, FusedSegmentPacksEveryField)
{
    auto fused = fusedCustomCall();
    ASSERT_EQ(fused.args.size(), FUSED_A2A_SEGMENT_BYTES);

    EXPECT_EQ(packedAt<void*>(fused.args, peer0FlagOffset), const_cast<char*>(&fusedPeerFlag));
    EXPECT_EQ(packedAt<void*>(fused.args, peer1FlagOffset), nullptr);
    EXPECT_EQ(packedAt<void*>(fused.args, counterOffset), const_cast<char*>(&fusedCounter));
    EXPECT_EQ(packedAt<uint32_t>(fused.args, fusedMyRankOffset), 1u);
    EXPECT_EQ(packedAt<uint32_t>(fused.args, fusedWOffset), 2u);
    EXPECT_EQ(packedAt<uint32_t>(fused.args, fusedDrainOffset), FUSED_A2A_DRAIN_RECV);
    EXPECT_EQ(packedAt<uint32_t>(fused.args, fusedAMOffset), 256u);
}
