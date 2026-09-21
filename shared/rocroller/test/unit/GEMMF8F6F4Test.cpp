// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/CodeGen/Utils.hpp>

#include "GEMMF8F6F4.hpp"
#include "GEMMTestBase.hpp"

namespace GEMMTests
{
    using namespace rocRoller;
    namespace SolutionParams = rocRoller::Parameters::Solution;

    void CheckGEMMF8F6F4(rocRoller::ContextPtr m_context,
                         std::string           mfma,
                         std::string           modifiers,
                         uint                  numMFMAs,
                         uint                  numBufferLoadsForC,
                         uint                  numBufferLoadsForAB,
                         uint                  numDSWrites,
                         uint                  numDSReads,
                         uint                  numTrLoads,
                         bool const            isF6Type            = false,
                         uint                  numScaleBufferLoads = 0,
                         uint                  numScaleDSWrites    = 0,
                         uint                  numScaleDSLoads     = 0)
    {
        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, mfma), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, modifiers), numMFMAs);

        EXPECT_EQ(countSubstring(generatedCode, "buffer_load"),
                  numBufferLoadsForC + numBufferLoadsForAB + numScaleBufferLoads);
        if(!isF6Type)
        {
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx4 "),
                      numBufferLoadsForC + numBufferLoadsForAB);
        }
        else
        {
            auto numDWordX3BufferLoads = countSubstring(generatedCode, "buffer_load_dwordx3 ");
            auto numDWordX4orX2DSBufferLoads = numBufferLoadsForAB - numDWordX3BufferLoads;
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx4 "),
                      numBufferLoadsForC + numDWordX4orX2DSBufferLoads / 2);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "),
                      numDWordX4orX2DSBufferLoads / 2);
        }
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_ubyte "), numScaleBufferLoads);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 0);

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b"), numDSWrites + numScaleDSWrites);
        if(!isF6Type)
        {
            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128 "), numDSWrites);
        }
        else
        {
            auto numB96DSWrites       = countSubstring(generatedCode, "ds_write_b96 ");
            auto numB128orB64DSWrites = numDSWrites - numB96DSWrites;
            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128 "), numB128orB64DSWrites / 2);
            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64 "), numB128orB64DSWrites / 2);
        }
        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b8"), numScaleDSWrites);

        if(!isF6Type)
        {
            EXPECT_EQ(countSubstring(generatedCode, "ds_read_b64_tr_b"), numTrLoads);
        }
        else
        {
            EXPECT_EQ(countSubstring(generatedCode, "ds_read_b96_tr_b6"), numTrLoads);
        }

        EXPECT_EQ(countSubstring(generatedCode, "ds_read"),
                  numDSReads + numScaleDSLoads + numTrLoads);
        EXPECT_EQ(countSubstring(generatedCode, "ds_read_u8 "), numScaleDSLoads);

        if(!isF6Type)
        {
            EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128 "), numDSReads);
        }
        else
        {
            EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128 "), numDSReads / 2);
            EXPECT_EQ(countSubstring(generatedCode, "ds_read_b64 "), numDSReads / 2);
        }
    }

    void CheckMFMAF8F6F4(rocRoller::ContextPtr m_context,
                         std::string           f8f6f4_inst,
                         std::string           modifier)
    {
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasMFMA_fp8))
        {
            auto generatedCode = m_context->instructions()->toString();

            auto mfma_count     = countSubstring(generatedCode, "v_mfma_");
            auto f8f6f4_count   = countSubstring(generatedCode, f8f6f4_inst);
            auto modifier_count = countSubstring(generatedCode, modifier);

            // All mfma instructions should be f8f6f4
            EXPECT_EQ(mfma_count, f8f6f4_count);
            // All f8f6f4 instructions should use 0b100 (FP4) as input matrix format
            EXPECT_EQ(f8f6f4_count, modifier_count);
        }
    }

    // ========================================================================
    // GEMMF8F6F4TestSuite
    // ========================================================================

    // Params are: A & B type, K tile size, (transA, transB), load A path, load B path
    class GEMMF8F6F4TestSuite
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>,
                                                   SolutionParams::LoadPath,
                                                   SolutionParams::LoadPath>>
    {
    };

    // ========================================================================
    // StreamKGEMMMXF8F6F4TestSuite
    // ========================================================================

    // Params are: A & B type, K tile size, (transA, transB), loadPathA, loadPathB, StreamKMode
    class StreamKGEMMMXF8F6F4TestSuite
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>,
                                                   SolutionParams::LoadPath,
                                                   SolutionParams::LoadPath,
                                                   rocRoller::StreamKMode>>
    {
    };

    TEST_P(StreamKGEMMMXF8F6F4TestSuite, GPU_StreamKUnrollPrefetchSwizzleScaledGEMMMX)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto [typeAB, MFMAK, transOp, loadPathA, loadPathB, mode] = std::get<1>(GetParam());

        // Note: The following cases are filtered out by FilterValidStreamKGEMMMXF8F6F4Params:
        // - Direct2LDS not yet supported for FP6/BF6

        AssertFatal(loadPathA == SolutionParams::LoadPath::BufferToLDSViaVGPR
                        || loadPathA == SolutionParams::LoadPath::BufferToLDS,
                    "Unexpected load path for A : ",
                    ShowValue(loadPathA));

        AssertFatal(loadPathB == SolutionParams::LoadPath::BufferToLDSViaVGPR
                        || loadPathB == SolutionParams::LoadPath::BufferToLDS,
                    "Unexpected load path for B : ",
                    ShowValue(loadPathB));

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto problem = GEMMProblemF8F6F4{waveM, waveN, waveK};

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        problem.numWGs = deviceProperties.multiProcessorCount;

        problem.macM = 128;
        problem.macN = 128;
        problem.macK = 128;

        problem.workgroupSizeX = 2 * problem.wavefrontSize;
        problem.workgroupSizeY = 2;

        problem.m = problem.macM * 4;
        problem.n = problem.macN * problem.numWGs / 2 + problem.macN * 2;

        ASSERT_GE(problem.m * problem.n / problem.macM / problem.macN, problem.numWGs);

        problem.streamK = mode;
        problem.k       = problem.macK * 8;

        std::tie(problem.transA, problem.transB) = transOp;

        problem.loadPathA = loadPathA;
        problem.loadPathB = loadPathB;

        problem.unrollK           = 2;
        problem.prefetch          = true;
        problem.prefetchInFlight  = 2;
        problem.prefetchLDSFactor = 2;

        // TODO: remove the if condition when SwizzleScale supports non-TN data layout
        if(problem.transA == "T" && problem.transB == "N")
        {
            problem.loadScalePathA = SolutionParams::LoadPath::BufferToVGPR;
            problem.loadScalePathB = SolutionParams::LoadPath::BufferToVGPR;

            problem.scaleAMode = Operations::ScaleMode::Separate;
            problem.scaleBMode = Operations::ScaleMode::Separate;

            problem.scaleTypeA = DataType::E8M0;
            problem.scaleTypeB = DataType::E8M0;

            problem.swizzleScale  = true;
            problem.swizzleM      = 64;
            problem.swizzleN      = 64;
            problem.swizzleK      = 8;
            problem.prefetchScale = true;

            problem.scaleBlockSize = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultScaleBlockSize);
        }

        uint const elementBits = DataTypeInfo::Get(typeAB).elementBits;

        switch(typeAB)
        {
        case DataType::FP8:
            basicGEMM<FP8, FP8, float>(problem);
            break;
        case DataType::BF8:
            basicGEMM<BF8, BF8, float>(problem);
            break;
        case DataType::FP6:
            basicGEMM<FP6, FP6, float>(problem);
            break;
        case DataType::BF6:
            basicGEMM<BF6, BF6, float>(problem);
            break;
        case DataType::FP4:
            basicGEMM<FP4, FP4, float>(problem);
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }
    }

    using StreamKGEMMMXF8F6F4TestParamGenerator
        = ::testing::internal::ParamGenerator<StreamKGEMMMXF8F6F4TestSuite::ParamType>;
    static auto FilterValidStreamKGEMMMXF8F6F4Params(
        StreamKGEMMMXF8F6F4TestParamGenerator&& inputParamGenerator)
    {
        using LP = SolutionParams::LoadPath;
        using DT = rocRoller::DataType;

        std::vector<StreamKGEMMMXF8F6F4TestSuite::ParamType> filtered;
        for(auto const& inputParam : inputParamGenerator)
        {
            auto const& params = std::get<1>(inputParam);

            auto const& typeAB    = std::get<0>(params);
            auto const& loadPathA = std::get<3>(params);
            auto const& loadPathB = std::get<4>(params);

            // Direct2LDS not yet supported for FP6/BF6
            if((typeAB == DT::FP6 || typeAB == DT::BF6)
               && (loadPathA == LP::BufferToLDS || loadPathB == LP::BufferToLDS))
            {
                continue;
            }

            filtered.push_back(inputParam);
        }

        return ::testing::ValuesIn(filtered);
    }

    INSTANTIATE_TEST_SUITE_P(
        StreamKGEMMMXF8F6F4Test,
        StreamKGEMMMXF8F6F4TestSuite,
        FilterValidStreamKGEMMMXF8F6F4Params(::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP8,
                                  rocRoller::DataType::BF8,
                                  rocRoller::DataType::FP6,
                                  rocRoller::DataType::BF6,
                                  rocRoller::DataType::FP4),
                ::testing::Values(64, 128),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::BufferToLDS),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::BufferToLDS),
                ::testing::Values(rocRoller::StreamKMode::Standard,
                                  rocRoller::StreamKMode::TwoTile,
                                  rocRoller::StreamKMode::TwoTileDPFirst))))); // StreamKMode

    // ========================================================================
    // GEMMF8F6F4TestSuite (tests)
    // ========================================================================

    TEST_P(GEMMF8F6F4TestSuite, GPU_GEMM_DataType_F8F6F4_Basic)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto [typeAB, MFMAK, transOp, loadPathA, loadPathB] = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto problem = GEMMProblemF8F6F4{waveM, waveN, waveK};

        std::tie(problem.transA, problem.transB) = transOp;

        problem.loadPathA = loadPathA;
        problem.loadPathB = loadPathB;

        uint const elementBits = DataTypeInfo::Get(typeAB).elementBits;

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        switch(typeAB)
        {
        case DataType::FP8:
            basicGEMM<FP8, FP8, float>(problem);
            break;
        case DataType::BF8:
            basicGEMM<BF8, BF8, float>(problem);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            basicGEMM<FP6, FP6, float>(problem);
            modifiers = "cbsz:0b010 blgp:0b010";
            break;
        case DataType::BF6:
            basicGEMM<BF6, BF6, float>(problem);
            modifiers = "cbsz:0b011 blgp:0b011";
            break;
        case DataType::FP4:
            basicGEMM<FP4, FP4, float>(problem);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        uint const wfs           = problem.wavefrontSize;
        uint const wgX           = problem.workgroupSizeX;
        uint const wgY           = problem.workgroupSizeY;
        uint const numDWavetiles = problem.macM * problem.macN / (waveM * waveN);
        uint const numWaves      = wgX * wgY / wfs;

        uint const numDWavetilesPerWave = numDWavetiles / numWaves;
        uint const numMFMAsPerWave      = problem.macK / waveK;
        uint const numMFMAs             = numDWavetilesPerWave * numMFMAsPerWave;

        auto const& arch                = m_context->targetArchitecture();
        uint const  elementsPerWavetile = waveM * waveK / wfs;
        uint const  elementsPerTrLoad   = bitsPerTransposeLoad(arch, elementBits) / elementBits;

        uint const bitsPerABMemOp = (elementBits == 6 ? 96 : 128);
        uint const trLoadsPerWave
            = elementsPerWavetile * elementBits / bitsPerTransposeLoad(arch, elementBits);
        uint const dsLoadsPerWave = elementsPerWavetile * elementBits / bitsPerABMemOp;

        uint const bitsLoadedForAB
            = (/*A*/ waveM * problem.macK + /*B*/ problem.macK * waveN) * elementBits;

        uint const elementBitsC   = DataTypeInfo::Get(DataType::Float).elementBits;
        uint const bitsLoadedForC = numDWavetilesPerWave * waveM * waveN * elementBitsC;

        uint const numBufferLoadsForC  = bitsLoadedForC / 128 / wfs;
        uint const numDSWrites         = bitsLoadedForAB / bitsPerABMemOp / wfs;
        uint const numBufferLoadsForAB = numDSWrites;

        uint numTrLoads = 0;
        uint numDSReads = 0;
        { // 2x2 jamming = 4 tiles. Each tile of A gets multiplied by 4 tiles of B.
            if(problem.transA == "T")
                numDSReads += /*number of A tiles*/ 1 * numMFMAsPerWave * dsLoadsPerWave;
            if(problem.transB == "N")
                numDSReads += /*number of B tiles*/ 4 * numMFMAsPerWave * dsLoadsPerWave;

            if(problem.transA == "N")
                numTrLoads += /*number of A tiles*/ 1 * numMFMAsPerWave * trLoadsPerWave;
            if(problem.transB == "T")
                numTrLoads += /*number of B tiles*/ 4 * numMFMAsPerWave * trLoadsPerWave;
        }

        bool const isF6 = typeAB == DataType::FP6 || typeAB == DataType::BF6;

        auto const mfma{fmt::format("v_mfma_f32_{}x{}x{}_f8f6f4", waveM, waveN, waveK)};

        CheckGEMMF8F6F4(m_context,
                        mfma,
                        modifiers,
                        numMFMAs,
                        numBufferLoadsForC,
                        numBufferLoadsForAB,
                        numDSWrites,
                        numDSReads,
                        numTrLoads,
                        isF6);
    }

    TEST_P(GEMMF8F6F4TestSuite, GPU_GEMM_Scaled_F8F6F4_Basic)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto [typeAB, MFMAK, transOp, loadPathA, loadPathB] = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto problem = GEMMProblemF8F6F4{waveM, waveN, waveK};

        std::tie(problem.transA, problem.transB) = transOp;

        uint const elementBits = DataTypeInfo::Get(typeAB).elementBits;

        problem.scaleAMode = Operations::ScaleMode::Separate;
        problem.scaleBMode = Operations::ScaleMode::Separate;

        problem.scaleTypeA = DataType::E8M0;
        problem.scaleTypeB = DataType::E8M0;

        problem.loadPathA      = loadPathA;
        problem.loadPathB      = loadPathB;
        problem.loadScalePathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        problem.loadScalePathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        problem.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        switch(typeAB)
        {
        case DataType::FP8:
            basicGEMM<FP8, FP8, float>(problem);
            break;
        case DataType::BF8:
            basicGEMM<BF8, BF8, float>(problem);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            basicGEMM<FP6, FP6, float>(problem);
            modifiers = "cbsz:0b010 blgp:0b010";
            break;
        case DataType::BF6:
            basicGEMM<BF6, BF6, float>(problem);
            modifiers = "cbsz:0b011 blgp:0b011";
            break;
        case DataType::FP4:
            basicGEMM<FP4, FP4, float>(problem);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        uint const wfs           = problem.wavefrontSize;
        uint const wgX           = problem.workgroupSizeX;
        uint const wgY           = problem.workgroupSizeY;
        uint const numDWavetiles = problem.macM * problem.macN / (waveM * waveN);
        uint const numWaves      = wgX * wgY / wfs;

        uint const numDWavetilesPerWave = numDWavetiles / numWaves;
        uint const numMFMAsPerWave      = problem.macK / waveK;
        uint const numMFMAs             = numDWavetilesPerWave * numMFMAsPerWave;

        auto const& arch                = m_context->targetArchitecture();
        uint const  elementsPerWavetile = waveM * waveK / wfs;
        uint const  elementsPerTrLoad   = bitsPerTransposeLoad(arch, elementBits) / elementBits;

        uint const bitsPerABMemOp = (elementBits == 6 ? 96 : 128);
        uint const trLoadsPerWave
            = elementsPerWavetile * elementBits / bitsPerTransposeLoad(arch, elementBits);
        uint const dsLoadsPerWave = elementsPerWavetile * elementBits / bitsPerABMemOp;

        uint const bitsLoadedForAB
            = (/*A*/ waveM * problem.macK + /*B*/ problem.macK * waveN) * elementBits;

        uint const elementBitsC   = DataTypeInfo::Get(DataType::Float).elementBits;
        uint const bitsLoadedForC = numDWavetilesPerWave * waveM * waveN * elementBitsC;

        uint const numBufferLoadsForC  = bitsLoadedForC / 128 / wfs;
        uint const numDSWrites         = bitsLoadedForAB / bitsPerABMemOp / wfs;
        uint const numBufferLoadsForAB = numDSWrites;

        uint numTrLoads = 0;
        uint numDSReads = 0;
        { // 2x2 jamming = 4 tiles. Each tile of A gets multiplied by 4 tiles of B.
            if(problem.transA == "T")
                numDSReads += /*number of A tiles*/ 1 * numMFMAsPerWave * dsLoadsPerWave;
            if(problem.transB == "N")
                numDSReads += /*number of B tiles*/ 4 * numMFMAsPerWave * dsLoadsPerWave;

            if(problem.transA == "N")
                numTrLoads += /*number of A tiles*/ 1 * numMFMAsPerWave * trLoadsPerWave;
            if(problem.transB == "T")
                numTrLoads += /*number of B tiles*/ 4 * numMFMAsPerWave * trLoadsPerWave;
        }

        uint const numScaleBufferLoads = (32 / 8);
        uint const numScaleDSWrites    = (32 / 8);
        uint const numScaleDSLoads     = (/*A*/ 1 + /*B*/ 4) * numMFMAsPerWave;

        bool const isF6 = typeAB == DataType::FP6 || typeAB == DataType::BF6;

        auto const mfma{fmt::format("v_mfma_scale_f32_{}x{}x{}_f8f6f4", waveM, waveN, waveK)};

        CheckGEMMF8F6F4(m_context,
                        mfma,
                        modifiers,
                        numMFMAs,
                        numBufferLoadsForC,
                        numBufferLoadsForAB,
                        numDSWrites,
                        numDSReads,
                        numTrLoads,
                        isF6,
                        numScaleBufferLoads,
                        numScaleDSWrites,
                        numScaleDSLoads);
    }

    TEST_P(GEMMF8F6F4TestSuite, GPU_GEMM_Scaled_F8F6F4_Dword)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto [typeAB, MFMAK, transOp, loadPathA, loadPathB] = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto gemm = GEMMProblemF8F6F4{waveM, waveN, waveK};

        std::tie(gemm.transA, gemm.transB) = transOp;

        uint const elementBits = DataTypeInfo::Get(typeAB).elementBits;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.macM = 128;
        gemm.macN = 128;
        gemm.macK = (typeAB != rocRoller::DataType::FP4) ? 256 : 512;
        gemm.m    = 2 * gemm.macM;
        gemm.n    = 3 * gemm.macN;
        gemm.k    = 4 * gemm.macK;

        gemm.loadPathA      = loadPathA;
        gemm.loadPathB      = loadPathB;
        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        switch(typeAB)
        {
        case DataType::FP8:
            basicGEMM<FP8, FP8, float>(gemm);
            break;
        case DataType::BF8:
            basicGEMM<BF8, BF8, float>(gemm);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            basicGEMM<FP6, FP6, float>(gemm);
            modifiers = "cbsz:0b010 blgp:0b010";
            break;
        case DataType::BF6:
            basicGEMM<BF6, BF6, float>(gemm);
            modifiers = "cbsz:0b011 blgp:0b011";
            break;
        case DataType::FP4:
            basicGEMM<FP4, FP4, float>(gemm);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        auto const mfma{fmt::format("v_mfma_scale_f32_{}x{}x{}_f8f6f4", waveM, waveN, waveK)};
        CheckMFMAF8F6F4(m_context, mfma, modifiers);

        uint const totalWorkitems = gemm.workgroupSizeX * gemm.workgroupSizeY;
        // Example A:256x128 => scaleA:256x4 => 1024 values/256 workitems => 4 values per workitem
        uint const numABScaleLoadStorePerWorkitem = (gemm.macM * (gemm.macK / 32)) / totalWorkitems;
        AssertFatal(
            numABScaleLoadStorePerWorkitem % 4 == 0,
            "long dword instructions require multiple of 4 scale(8-bit) values per workitem");

        std::string bufferLoad{"buffer_load_dword "};
        std::string dsWrite{"ds_write_b32"};
        uint const  factor = numABScaleLoadStorePerWorkitem / 4;
        AssertFatal(factor > 0 && factor <= 2,
                    "For the given macrotile, dword factor can't be greater than 2");
        if(factor == 2)
        {
            bufferLoad = "buffer_load_dwordx2 ";
            dsWrite    = "ds_write_b64";
        }

        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, bufferLoad), 2);
        EXPECT_EQ(countSubstring(generatedCode, dsWrite), 2);
    }

    TEST_P(GEMMF8F6F4TestSuite, GPU_GEMM_Scaled_F8F6F4_Swizzle_Prefetch2)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto [typeAB, MFMAK, transOp, loadPathA, loadPathB] = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto gemm = GEMMProblemF8F6F4{waveM, waveN, waveK};

        std::tie(gemm.transA, gemm.transB) = transOp;

        // TODO: enable the test when the code generation time is reduced
        if(waveK == 128)
            GTEST_SKIP() << "Skip 16x16x128 MFMA instruction due to long code generation time"
                         << std::endl;
        // TODO: enable the tests when SwizzleScale supports non-TN data layout
        if(gemm.transA != "T" || gemm.transB != "N")
            GTEST_SKIP() << "Non-TN test not yet supported for SwizzleScale" << std::endl;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 128;

        gemm.m = 2 * gemm.macM;
        gemm.n = 3 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA      = loadPathA;
        gemm.loadPathB      = loadPathB;
        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToVGPR;

        gemm.unrollK           = 2;
        gemm.prefetch          = true;
        gemm.prefetchInFlight  = 2;
        gemm.prefetchLDSFactor = 2;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.swizzleScale  = true;
        gemm.swizzleM      = 64;
        gemm.swizzleN      = 64;
        gemm.swizzleK      = 8;
        gemm.prefetchScale = true;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        switch(typeAB)
        {
        case DataType::FP8:
            basicGEMM<FP8, FP8, float>(gemm);
            break;
        case DataType::BF8:
            basicGEMM<BF8, BF8, float>(gemm);
            break;
        case DataType::FP6:
            basicGEMM<FP6, FP6, float>(gemm);
            break;
        case DataType::BF6:
            basicGEMM<BF6, BF6, float>(gemm);
            break;
        case DataType::FP4:
            basicGEMM<FP4, FP4, float>(gemm);
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_ubyte "), 0);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 0);
        // 1x4 wave config: NumAScaleLoadTiles = 128/64 = 2 and NumBScaleLoadTiles = 256/4/64 = 1
        // prefetched scale: 2 * 3 = 6
        EXPECT_GE(countSubstring(generatedCode, "buffer_load_dwordx2 "), 6);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMF8F6F4Test,
        GEMMF8F6F4TestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR))));

    // ========================================================================
    // MixedGEMMF8F6F4TestSuite
    // ========================================================================

    // Params are: A type, B type, K tile size, (transA, transB), load A path, load B path
    class MixedGEMMF8F6F4TestSuite
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>,
                                                   SolutionParams::LoadPath,
                                                   SolutionParams::LoadPath>>
    {
    };

    TEST_P(MixedGEMMF8F6F4TestSuite, GPU_GEMM_DataType_F8F6F4_Mixed)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto [typeA, typeB, MFMAK, transOp, loadPathA, loadPathB] = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto problem      = GEMMProblemF8F6F4{waveM, waveN, waveK};
        problem.loadPathA = loadPathA;
        problem.loadPathB = loadPathB;

        std::tie(problem.transA, problem.transB) = transOp;

        // TODO: enable non-TN F6 tests
        auto const elementBitsA = DataTypeInfo::Get(typeA).elementBits;
        auto const elementBitsB = DataTypeInfo::Get(typeB).elementBits;

        basicGEMMMixed(typeA, typeB, problem);

        auto const mfma{fmt::format("v_mfma_f32_{}x{}x{}_f8f6f4", waveM, waveN, waveK)};

        std::string modifierA = "defaultModiferA";
        std::string modifierB = "defaultModiferB";

        if(typeA == rocRoller::DataType::FP8)
            modifierA = "cbsz:0b000";
        else if(typeA == rocRoller::DataType::BF8)
            modifierA = "cbsz:0b001";
        else if(typeA == rocRoller::DataType::FP6)
            modifierA = "cbsz:0b010";
        else if(typeA == rocRoller::DataType::BF6)
            modifierA = "cbsz:0b011";
        else if(typeA == rocRoller::DataType::FP4)
            modifierA = "cbsz:0b100";
        else
            Throw<FatalError>("Unhandled data type for mixed GEMM.", ShowValue(typeA));

        if(typeB == rocRoller::DataType::FP8)
            modifierB = "blgp:0b000";
        else if(typeB == rocRoller::DataType::BF8)
            modifierB = "blgp:0b001";
        else if(typeB == rocRoller::DataType::FP6)
            modifierB = "blgp:0b010";
        else if(typeB == rocRoller::DataType::BF6)
            modifierB = "blgp:0b011";
        else if(typeB == rocRoller::DataType::FP4)
            modifierB = "blgp:0b100";
        else
            Throw<FatalError>("Unhandled data type for mixed GEMM.", ShowValue(typeB));

        CheckMFMAF8F6F4(m_context, mfma, modifierA + " " + modifierB);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMF8F6F4Test,
        MixedGEMMF8F6F4TestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(64, 128),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::GlobalToLDSViaVGPR))));

    // ========================================================================
    // ScaledMixedGEMMF8F6F4TestSuite
    // ========================================================================

    // Params are: A type, B type, K tile size, load A path, load B path,
    //   scale A mode, scale B mode, Load A scale path, Load B scale path, (transA, transB)
    class ScaledMixedGEMMF8F6F4TestSuite
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   rocRoller::DataType,
                                                   int,
                                                   SolutionParams::LoadPath,
                                                   SolutionParams::LoadPath,
                                                   rocRoller::Operations::ScaleMode,
                                                   rocRoller::Operations::ScaleMode,
                                                   SolutionParams::LoadPath,
                                                   SolutionParams::LoadPath,
                                                   std::pair<std::string, std::string>>>
    {
    };

    TEST_P(ScaledMixedGEMMF8F6F4TestSuite, GPU_GEMM_Scaled_F8F6F4_Mixed)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);

        auto [typeA,
              typeB,
              MFMAK,
              loadPathA,
              loadPathB,
              scaleAMode,
              scaleBMode,
              loadScalePathA,
              loadScalePathB,
              transOp]
            = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto problem      = GEMMProblemF8F6F4{waveM, waveN, waveK};
        problem.loadPathA = loadPathA;
        problem.loadPathB = loadPathB;

        std::tie(problem.transA, problem.transB) = transOp;

        // TODO: enable non-TN F6 tests
        auto const elementBitsA = DataTypeInfo::Get(typeA).elementBits;
        auto const elementBitsB = DataTypeInfo::Get(typeB).elementBits;

        problem.scaleAMode = scaleAMode;
        problem.scaleBMode = scaleBMode;

        problem.scaleTypeA = DataType::E8M0;
        problem.scaleTypeB = DataType::E8M0;

        problem.loadScalePathA = loadScalePathA;
        problem.loadScalePathB = loadScalePathB;

        if(scaleAMode == rocRoller::Operations::ScaleMode::Separate
           || scaleBMode == rocRoller::Operations::ScaleMode::Separate)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);
            problem.scaleBlockSize = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultScaleBlockSize);
        }

        basicGEMMMixed(typeA, typeB, problem);
    }

    using ScaledMixedGEMMF8F6F4TestParamGenerator
        = ::testing::internal::ParamGenerator<ScaledMixedGEMMF8F6F4TestSuite::ParamType>;
    static auto FilterValidScalePathAndScaleModeParams(
        ScaledMixedGEMMF8F6F4TestParamGenerator&& inputParamGenerator)
    {
        using LP = SolutionParams::LoadPath;
        using SM = rocRoller::Operations::ScaleMode;

        std::vector<ScaledMixedGEMMF8F6F4TestSuite::ParamType> filtered;
        for(auto const& inputParam : inputParamGenerator)
        {
            auto const& params = std::get<1>(inputParam);

            auto const& scaleAMode     = std::get<5>(params);
            auto const& scaleBMode     = std::get<6>(params);
            auto const& loadScalePathA = std::get<7>(params);
            auto const& loadScalePathB = std::get<8>(params);

            if((loadScalePathA != LP::BufferToVGPR or loadScalePathA != LP::GlobalToVGPR)
               && (scaleAMode == SM::None || scaleAMode == SM::SingleScale))
            {
                continue;
            }

            if((loadScalePathB != LP::BufferToVGPR or loadScalePathB != LP::GlobalToVGPR)
               && (scaleBMode == SM::None || scaleBMode == SM::SingleScale))
            {
                continue;
            }

            filtered.push_back(inputParam);
        }

        return ::testing::ValuesIn(filtered);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMF8F6F4Test,
        ScaledMixedGEMMF8F6F4TestSuite,
        FilterValidScalePathAndScaleModeParams(::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(64, 128),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                               ::testing::Values(rocRoller::Operations::ScaleMode::SingleScale,
                                                 rocRoller::Operations::ScaleMode::Separate),
                               ::testing::Values(rocRoller::Operations::ScaleMode::SingleScale,
                                                 rocRoller::Operations::ScaleMode::Separate),
                               ::testing::Values(SolutionParams::LoadPath::BufferToVGPR,
                                                 SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                               ::testing::Values(SolutionParams::LoadPath::BufferToVGPR,
                                                 SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T"))))));
} // namespace GEMMTests
