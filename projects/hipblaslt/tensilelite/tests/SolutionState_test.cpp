// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionTaskPredicates.hpp>
#include <Tensile/msgpack/SolutionState.hpp>

#include "TestData.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using namespace TensileLite;

namespace
{
    constexpr uint32_t NoteSection        = 7;
    constexpr uint32_t AMDGPUMetadataNote = 32;

    template <typename T>
    T read(std::string const& data, size_t offset)
    {
        if(offset + sizeof(T) > data.size())
            throw std::runtime_error("truncated code object");
        T rv;
        std::memcpy(&rv, data.data() + offset, sizeof(T));
        return rv;
    }

    // The AMDGPU metadata note of a 64-bit little-endian code object.
    std::string metadataNote(std::filesystem::path const& path)
    {
        std::ifstream file(path, std::ios::binary);
        std::string   data((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        if(data.compare(0, 4, std::string("\x7f") + "ELF") != 0 || read<uint8_t>(data, 4) != 2
           || read<uint8_t>(data, 5) != 1)
            throw std::runtime_error(path.string() + " is not a 64-bit little-endian ELF file");

        auto sectionOffset = read<uint64_t>(data, 0x28);
        auto sectionSize   = read<uint16_t>(data, 0x3A);
        auto sectionCount  = read<uint16_t>(data, 0x3C);
        for(uint16_t i = 0; i < sectionCount; i++)
        {
            auto header = sectionOffset + i * sectionSize;
            if(read<uint32_t>(data, header + 4) != NoteSection)
                continue;
            auto pos = read<uint64_t>(data, header + 0x18);
            auto end = pos + read<uint64_t>(data, header + 0x20);
            while(pos < end)
            {
                auto nameSize = read<uint32_t>(data, pos);
                auto descSize = read<uint32_t>(data, pos + 4);
                auto type     = read<uint32_t>(data, pos + 8);
                auto name     = data.substr(pos + 12, nameSize);
                pos += 12 + ((nameSize + 3) & ~3u);
                if(type == AMDGPUMetadataNote && name == std::string("AMDGPU", 7))
                    return data.substr(pos, descSize);
                pos += (descSize + 3) & ~3u;
            }
        }
        throw std::runtime_error(path.string() + " has no AMDGPU metadata note");
    }

    msgpack::object_handle metadata(std::string const& arch)
    {
        auto note = metadataNote(TestData::Instance().file("solution-state/" + arch, "o"));
        return msgpack::unpack(note.data(), note.size());
    }

    msgpack::object const& customConfig(msgpack::object const& metadata)
    {
        if(metadata.type == msgpack::type::MAP)
            for(uint32_t i = 0; i < metadata.via.map.size; i++)
            {
                auto const& entry = metadata.via.map.ptr[i];
                if(entry.key.type == msgpack::type::STR
                   && entry.key.as<std::string>() == "custom.config")
                    return entry.val;
            }
        throw std::runtime_error("metadata has no custom.config");
    }

    // A copy of map without the key at path; the rest is shared with map.
    msgpack::object without(msgpack::object const&          map,
                            std::vector<std::string> const& path,
                            msgpack::zone&                  zone,
                            size_t                          depth = 0)
    {
        auto const& source  = map.via.map;
        auto*       entries = static_cast<msgpack::object_kv*>(
            zone.allocate_align(sizeof(msgpack::object_kv) * source.size));
        uint32_t size = 0;
        for(uint32_t i = 0; i < source.size; i++)
        {
            auto entry = source.ptr[i];
            if(entry.key.as<std::string>() == path[depth])
            {
                if(depth + 1 == path.size())
                    continue;
                entry.val = without(entry.val, path, zone, depth + 1);
            }
            entries[size++] = entry;
        }
        msgpack::object rv;
        rv.type         = msgpack::type::MAP;
        rv.via.map.ptr  = entries;
        rv.via.map.size = size;
        return rv;
    }

    template <typename P, typename Base>
    P const& as(std::shared_ptr<Base> const& predicate)
    {
        auto const* rv = dynamic_cast<P const*>(predicate.get());
        if(rv == nullptr)
            throw std::runtime_error((predicate ? predicate->type() : std::string("null"))
                                     + " is not " + P::Type());
        return *rv;
    }

    // The committed kernels differ only in their architecture, which also
    // shows in their names and in MathClocksUnrolledLoop.
    struct Kernel
    {
        std::string       arch;
        AMDGPU::Processor processor;
        std::string       isa;
        std::string       mgripm;
        int               mathClocksUnrolledLoop;
    };

    std::ostream& operator<<(std::ostream& stream, Kernel const& kernel)
    {
        return stream << kernel.arch;
    }

    std::string solutionName(Kernel const& kernel)
    {
        return "Cijk_Ailk_Bljk_HHS_BH_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFE"
               "M1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMX"
               "SA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSU1_GSUA"
               "MBSK_GSUC0_GSUWGMRR0_GLS0_HPLR0_ISA"
               + kernel.isa
               + "_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA"
                 "16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_"
                 "MGRIPM"
               + kernel.mgripm
               + "_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC"
                 "0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL0_PAP0_PGL0_P"
                 "GR0_PLR0_PKA0_RAP0_SGROB0_SIA3_SS0_SU32_SUM0_SUS256_SPO0_SRVW0_SSO0_SVW4_SNLL0_S"
                 "IP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_"
                 "THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_U"
                 "IOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGM8_WGM"
                 "XCC1_WGMXCCGn1";
    }

    std::string kernelName(Kernel const& kernel)
    {
        return "Cijk_Ailk_Bljk_HHS_BH_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFE"
               "M1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMX"
               "SA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMBSK_"
               "GLS0_HPLR0_ISA"
               + kernel.isa
               + "_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA"
                 "16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_"
                 "MGRIPM"
               + kernel.mgripm
               + "_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC"
                 "0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL0_PAP0_PGL0_P"
                 "GR0_PLR0_PKA0_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_T"
                 "DMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMX"
                 "SB0_THM0_THWS0_TPSN_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_US"
                 "FGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGMXCC1";
    }

    // The values Solution.FromOriginalState in Tensile/Contractions.py gives
    // for the custom.config of each kernel under clients/tests/jit/data.
    class SolutionStateTest : public ::testing::TestWithParam<Kernel>
    {
    protected:
        void SetUp() override
        {
            int deviceCount = 0;
            if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
                GTEST_SKIP() << "WorkgroupMappingXCCCheck reads the current device";

            m_metadata = metadata(GetParam().arch);
            solution   = solutionFromState(customConfig(m_metadata.get()));
        }

        std::shared_ptr<ContractionSolution> solution;

    private:
        msgpack::object_handle m_metadata;
    };
}

TEST_P(SolutionStateTest, Names)
{
    EXPECT_EQ(solution->solutionName, solutionName(GetParam()));
    EXPECT_EQ(solution->kernelName, kernelName(GetParam()));
    EXPECT_FALSE(solution->debugKernel);
    EXPECT_EQ(solution->index, 0);
    EXPECT_EQ(solution->libraryLogicIndex, 0);
    EXPECT_TRUE(solution->ideals.empty());
    EXPECT_EQ(solution->linearModel.slope, 1.0);
    EXPECT_EQ(solution->linearModel.intercept, 0.0);
    EXPECT_EQ(solution->linearModel.max, 1000.0);
}

TEST_P(SolutionStateTest, ProblemType)
{
    using rocisa::DataType;
    auto const& type = solution->problemType;
    EXPECT_EQ(type.operationIdentifier, "Contraction_l_Ailk_Bljk_Cijk_Dijk");
    EXPECT_FALSE(type.transA);
    EXPECT_FALSE(type.transB);
    EXPECT_EQ(type.computeInputTypeA, DataType::Half);
    EXPECT_EQ(type.computeInputTypeB, DataType::Half);
    EXPECT_EQ(type.aType, DataType::Half);
    EXPECT_EQ(type.bType, DataType::Half);
    EXPECT_EQ(type.cType, DataType::Half);
    EXPECT_EQ(type.dType, DataType::Half);
    EXPECT_EQ(type.eType, DataType::Half);
    EXPECT_EQ(type.computeType, DataType::Float);
    EXPECT_TRUE(type.useBeta);
    EXPECT_EQ(type.useBias, 0);
    EXPECT_EQ(type.biasSrcWhiteList, std::vector<int>{3});
    EXPECT_FALSE(type.useE);
    EXPECT_FALSE(type.useGateResidual);
    EXPECT_TRUE(type.gateResidualDataTypeWhiteList.empty());
    EXPECT_EQ(type.useScaleAB, "");
    EXPECT_FALSE(type.useScaleCD);
    EXPECT_EQ(type.useScaleAlphaVec, 0);
    EXPECT_EQ(type.biasDataTypeWhiteList,
              (std::vector<DataType>{DataType::Float, DataType::Half}));
    EXPECT_TRUE(type.highPrecisionAccumulate);
    EXPECT_FALSE(type.useInitialStridesAB);
    EXPECT_FALSE(type.useInitialStridesCD);
    EXPECT_TRUE(type.stridedBatched);
    EXPECT_FALSE(type.groupedGemm);
    EXPECT_FALSE(type.useGradient);
    EXPECT_EQ(type.activationType, ActivationType::None);
    EXPECT_EQ(type.activationArgLength, 0);
    EXPECT_EQ(type.activationComputeDataType, DataType::Float);
    EXPECT_FALSE(type.activationNoGuard);
    EXPECT_EQ(type.sparse, 0);
    EXPECT_EQ(type.f32XdlMathOp, DataType::Float);
    EXPECT_FALSE(type.supportDeviceUserArguments);
    EXPECT_FALSE(type.outputAmaxD);
    EXPECT_FALSE(type.swizzleTensorA);
    EXPECT_FALSE(type.swizzleTensorB);
    EXPECT_EQ(type.metadataLayout, 0);
    EXPECT_EQ(type.mxBlockA, 0);
    EXPECT_EQ(type.mxBlockB, 0);
    EXPECT_EQ(type.mxTypeA, DataType::E8);
    EXPECT_EQ(type.mxTypeB, DataType::E8);
    EXPECT_EQ(type.mxScaleFormat, 0);
    EXPECT_FALSE(type.fusedGemmA2A);
}

TEST_P(SolutionStateTest, ProblemPredicate)
{
    using namespace Predicates::Contraction;
    using rocisa::DataType;
    auto const& all = as<Predicates::And<ContractionProblemGemm>>(solution->problemPredicate).value;

    std::vector<std::string> types;
    for(auto const& predicate : all)
        types.push_back(predicate->type());
    EXPECT_EQ(types,
              (std::vector<std::string>{"OperationIdentifierEqual",
                                        "BiasDataTypeWhiteList",
                                        "GateResidualDataTypeWhiteList",
                                        "BiasSrcWhiteList",
                                        "AmaxDCheck",
                                        "TypesEqual",
                                        "HighPrecisionAccumulate",
                                        "Activation",
                                        "ActivationComputeType",
                                        "ActivationNoGuard",
                                        "UseGradient",
                                        "UseBias",
                                        "UseE",
                                        "UseGateResidual",
                                        "DataTypeE",
                                        "StridedBatched",
                                        "GroupedGemm",
                                        "UseScaleAB",
                                        "UseScaleCD",
                                        "UseScaleAlphaVec",
                                        "Sparse",
                                        "F32XdlMathOp",
                                        "SupportDeviceUserArguments",
                                        "SwizzleTensorA",
                                        "SwizzleTensorB",
                                        "FusedGemmA2A",
                                        "MXBlockA",
                                        "MXBlockB",
                                        "SynchronizerSizeCheck",
                                        "WorkgroupNumberCheck",
                                        "LeadingFree0SizesGreaterOrEqual",
                                        "LeadingFree1SizesGreaterOrEqual",
                                        "KernelLanguageCompatible",
                                        "BufferLoadOffsetLimitCheck",
                                        "BufferLoadOffsetLimitCheck_Beta",
                                        "BufferStoreOffsetLimitCheck",
                                        "GlobalSplitUCheckMinK",
                                        "WorkgroupMappingXCCCheck"}));
    ASSERT_EQ(all.size(), 38);

    EXPECT_EQ(as<OperationIdentifierEqual>(all[0]).value, "Contraction_l_Ailk_Bljk_Cijk_Dijk");
    EXPECT_EQ(as<BiasDataTypeWhiteList>(all[1]).value,
              (std::vector<DataType>{DataType::Float, DataType::Half}));
    EXPECT_TRUE(as<GateResidualDataTypeWhiteList>(all[2]).value.empty());
    EXPECT_EQ(as<BiasSrcWhiteList>(all[3]).value, std::vector<int>{3});
    EXPECT_FALSE(as<AmaxDCheck>(all[4]).value);
    EXPECT_EQ(as<TypesEqual>(all[5]).value,
              (std::array<DataType, 6>{DataType::Half,
                                       DataType::Half,
                                       DataType::Half,
                                       DataType::Half,
                                       DataType::Half,
                                       DataType::Half}));
    EXPECT_TRUE(as<HighPrecisionAccumulateEqual>(all[6]).value);
    EXPECT_EQ(as<ActivationCheck>(all[7]).value, ActivationType::None);
    EXPECT_EQ(as<ActivationComputeTypeEqual>(all[8]).value, DataType::Float);
    EXPECT_FALSE(as<ActivationNoGuardEqual>(all[9]).value);
    EXPECT_FALSE(as<UseGradientEqual>(all[10]).value);
    EXPECT_EQ(as<UseBiasCheck>(all[11]).value, 0);
    EXPECT_FALSE(as<UseEEqual>(all[12]).value);
    EXPECT_FALSE(as<UseGateResidualEqual>(all[13]).value);
    EXPECT_EQ(as<DataTypeEEqual>(all[14]).value, DataType::Half);
    EXPECT_TRUE(as<StridedBatchedEqual>(all[15]).value);
    EXPECT_FALSE(as<GroupedGemmEqual>(all[16]).value);
    EXPECT_EQ(as<UseScaleABCheck>(all[17]).value, "");
    EXPECT_FALSE(as<UseScaleCDCheck>(all[18]).value);
    EXPECT_EQ(as<UseScaleAlphaVecCheck>(all[19]).value, 0);
    EXPECT_EQ(as<Sparse>(all[20]).value, 0);
    EXPECT_EQ(as<F32XdlMathOpEqual>(all[21]).value, DataType::Float);
    EXPECT_FALSE(as<SupportDeviceUserArguments>(all[22]).value);
    EXPECT_FALSE(as<SwizzleTensorA>(all[23]).value);
    EXPECT_FALSE(as<SwizzleTensorB>(all[24]).value);
    EXPECT_FALSE(as<FusedGemmA2A>(all[25]).value);
    EXPECT_EQ(as<MXBlockA>(all[26]).value, 0);
    EXPECT_EQ(as<MXBlockB>(all[27]).value, 0);

    auto const& synchronizerSize = as<SynchronizerSizeCheck>(all[28]);
    EXPECT_EQ(synchronizerSize.index, 0);
    EXPECT_EQ(synchronizerSize.value, (std::array<int, 6>{16, 16, 1, 4, 1, 1}));

    auto const& workgroupNumber = as<WorkgroupNumberCheck>(all[29]);
    EXPECT_EQ(workgroupNumber.index, 0);
    EXPECT_EQ(workgroupNumber.value, (std::array<int, 3>{16, 16, 1}));

    EXPECT_EQ(as<LeadingFree0SizesGreaterOrEqual>(all[30]).value, 4);
    EXPECT_EQ(as<LeadingFree1SizesGreaterOrEqual>(all[31]).value, 1);
    EXPECT_EQ(as<KernelLanguageCompatible>(all[32]).value, KernelLanguage::Assembly);

    auto const& bufferLoad = as<BufferLoadOffsetLimitCheck>(all[33]).value;
    EXPECT_EQ(bufferLoad.shiftPtrElemB, 0);
    EXPECT_EQ(bufferLoad.shiftPtrElemA, 4);
    EXPECT_EQ(bufferLoad.depthUorMT1, 16);
    EXPECT_EQ(bufferLoad.depthUorMT0, 16);

    EXPECT_EQ(as<BufferLoadOffsetLimitCheck_Beta>(all[34]).value, 16);
    EXPECT_EQ(as<BufferStoreOffsetLimitCheck>(all[35]).value, 16);
    EXPECT_EQ(as<GlobalSplitUCheckMinK>(all[36]).value, (std::array<int, 2>{32, 1}));
    EXPECT_EQ(as<WorkgroupMappingXCCCheck>(all[37]).value, (std::array<int, 2>{1, -1}));
}

TEST_P(SolutionStateTest, TaskPredicate)
{
    auto const& all = as<Predicates::And<Task>>(solution->taskPredicate).value;
    ASSERT_EQ(all.size(), 2);
    as<Predicates::Contraction::WorkspaceCheck>(all[0]);
    as<Predicates::Contraction::LaunchLimits>(all[1]);
}

TEST_P(SolutionStateTest, HardwarePredicate)
{
    auto const& amdgpu = as<Predicates::IsSubclass<Hardware, AMDGPU>>(solution->hardwarePredicate);
    EXPECT_EQ(as<Predicates::GPU::ProcessorEqual>(amdgpu.value).value, GetParam().processor);
}

TEST_P(SolutionStateTest, SizeMapping)
{
    auto const& s = solution->sizeMapping;
    EXPECT_EQ(s.waveNum, 1);
    EXPECT_EQ(s.workGroupSize, TensileLite::dim3(16, 4, 1));
    EXPECT_EQ(s.macroTile, TensileLite::dim3(16, 16, 1));
    EXPECT_EQ(s.threadTile.x, 1);
    EXPECT_EQ(s.threadTile.y, 1);
    EXPECT_EQ(s.matrixInstruction, (std::array<int, 4>{16, 16, 16, 1}));
    EXPECT_EQ(s.grvwA, 4);
    EXPECT_EQ(s.grvwB, 4);
    EXPECT_EQ(s.gwvwC, 4);
    EXPECT_EQ(s.gwvwD, 4);
    EXPECT_EQ(s.depthU, 16);
    EXPECT_EQ(s.staggerU, 32);
    EXPECT_EQ(s.staggerUMapping, 0);
    EXPECT_EQ(s.globalSplitUPGR, 16);
    EXPECT_EQ(s.globalSplitU, 1);
    EXPECT_EQ(s.staggerStrideShift, 3);
    EXPECT_EQ(s.workGroupMapping, 8);
    EXPECT_EQ(s.packBatchDims, 0);
    EXPECT_EQ(s.magicDivAlg, 2);
    EXPECT_EQ(s.tileProcessingStrategy, TileProcessingStrategy::None);
    EXPECT_EQ(s.workAssignment, WorkAssignment::StaticGrid);
    EXPECT_EQ(s.streamKAtomic, 0);
    EXPECT_EQ(s.prefetchAcrossPersistent, 0);
    EXPECT_FALSE(s.sourceKernel);
    EXPECT_EQ(s.globalAccumulation, 3);
    EXPECT_EQ(s.adaptiveGemmGSUA, 0);
    EXPECT_EQ(s.workspaceSizePerElemC, 4);
    EXPECT_EQ(s.workspaceSizePerElemBias, 0);
    EXPECT_TRUE(s.activationFused);
    EXPECT_EQ(s.workGroupMappingXCC, 1);
    EXPECT_EQ(s.workGroupMappingXCCGroup, -1);
    EXPECT_FALSE(s.globalSplitUCoalesced);
    EXPECT_FALSE(s.globalSplitUWorkGroupMappingRoundRobin);
    EXPECT_EQ(s.CUOccupancy, 5);
    EXPECT_EQ(s.PrefetchGlobalRead, 0);
    EXPECT_EQ(s.MathClocksUnrolledLoop, GetParam().mathClocksUnrolledLoop);
    EXPECT_EQ(s.synchronizerSizePerWG, 4);
    EXPECT_EQ(s.nonTemporalA, 0);
    EXPECT_EQ(s.nonTemporalB, 0);
    EXPECT_EQ(s.temporalHintA, 0);
    EXPECT_EQ(s.temporalHintB, 0);
    EXPECT_FALSE(s.hasTemporalHint);
    EXPECT_EQ(s.adaptiveGemmNTAB, 0);
    EXPECT_EQ(s.customMainLoopScheduling, 0);
    EXPECT_FALSE(s.useSubtileImpl);
    EXPECT_EQ(s.NonTemporalD, 0);
    EXPECT_EQ(s.WaveSeparateGlobalReadA, 0);
    EXPECT_EQ(s.WaveSeparateGlobalReadB, 0);
    EXPECT_EQ(s.UnrollLoopSwapGlobalReadOrder, 0);
    EXPECT_FALSE(s.DirectToVgprA);
    EXPECT_FALSE(s.DirectToVgprB);
    EXPECT_EQ(s.NumLoadsCoalescedA, 1);
    EXPECT_EQ(s.NumLoadsCoalescedB, 1);
    EXPECT_EQ(s.waveGroup, (std::array<int, 2>{1, 1}));
    EXPECT_EQ(s.VectorWidthA, 1);
    EXPECT_EQ(s.VectorWidthB, 1);
    EXPECT_EQ(s.LocalSplitU, 1);
    EXPECT_FALSE(s.DirectToLdsA);
    EXPECT_FALSE(s.DirectToLdsB);
    EXPECT_EQ(s.expertSchedulingMode, 0);
    EXPECT_EQ(s.clusterDim, TensileLite::dim3(1, 1, 1));
}

TEST_P(SolutionStateTest, CustomKernel)
{
    auto const& k = solution->customKernel;
    EXPECT_EQ(k.name, kernelName(GetParam()));

    std::vector<std::pair<CustomArgSemantic, CustomArgType>> args;
    for(auto const& arg : k.args)
    {
        args.emplace_back(arg.semantic, arg.type);
        EXPECT_EQ(arg.padding, 0);
        EXPECT_EQ(arg.index, 0);
    }
    using S = CustomArgSemantic;
    EXPECT_EQ(args,
              (std::vector<std::pair<CustomArgSemantic, CustomArgType>>{
                  {S::GemmInfo, CustomArgType::uint32},
                  {S::InternalArgs, CustomArgType::uint32},
                  {S::InternalArgs1, CustomArgType::uint32},
                  {S::NumWorkGroups, CustomArgType::uint32},
                  {S::SizeFree0, CustomArgType::uint32},
                  {S::SizeFree1, CustomArgType::uint32},
                  {S::SizeFree2, CustomArgType::uint32},
                  {S::SizeSum, CustomArgType::uint32},
                  {S::AddressD, CustomArgType::address},
                  {S::AddressC, CustomArgType::address},
                  {S::AddressA, CustomArgType::address},
                  {S::AddressB, CustomArgType::address},
                  {S::StrideD0, CustomArgType::uint32},
                  {S::StrideD1, CustomArgType::uint32},
                  {S::StrideC0, CustomArgType::uint32},
                  {S::StrideC1, CustomArgType::uint32},
                  {S::StrideA0, CustomArgType::uint32},
                  {S::StrideA1, CustomArgType::uint32},
                  {S::StrideB0, CustomArgType::uint32},
                  {S::StrideB1, CustomArgType::uint32},
                  {S::Alpha, CustomArgType::uint32},
                  {S::Beta, CustomArgType::uint32},
                  {S::AddressTD, CustomArgType::address},
                  {S::Synchronizer, CustomArgType::address},
                  {S::GSUSync, CustomArgType::uint32}}));

    EXPECT_EQ(k.macrotile, TensileLite::dim3(16, 16, 16));
    EXPECT_EQ(k.threads, TensileLite::dim3(64, 1, 1));
    EXPECT_EQ(k.grid.x, CustomGridSize::TilesXYBatchGSU);
    EXPECT_EQ(k.grid.y, CustomGridSize::One);
    EXPECT_EQ(k.grid.z, CustomGridSize::One);
    EXPECT_EQ(k.workspaceType, CustomWorkspaceType::None);
    EXPECT_EQ(k.workspaceSizePerElemC, 0);
    EXPECT_EQ(k.workspaceSizePerElemBias, 0);
    EXPECT_TRUE(k.generated);
}

TEST_P(SolutionStateTest, InternalArgsSupport)
{
    auto const& support = solution->internalArgsSupport;
    EXPECT_EQ(support.version, 3);
    EXPECT_EQ(support.persistentLoopArgsVersion, 0);
    EXPECT_TRUE(support.gsu);
    EXPECT_TRUE(support.wgm);
    EXPECT_TRUE(support.staggerU);
    EXPECT_FALSE(support.perTileExtraIters);
    EXPECT_TRUE(support.useUniversalArgs);
    EXPECT_FALSE(support.useSFC);
}

INSTANTIATE_TEST_SUITE_P(
    JitKernels,
    SolutionStateTest,
    ::testing::Values(Kernel{"gfx90a", AMDGPU::Processor::gfx90a, "90a", "1", 0},
                      Kernel{"gfx942", AMDGPU::Processor::gfx942, "942", "1", 256},
                      Kernel{"gfx950", AMDGPU::Processor::gfx950, "950", "3", 288}),
    [](::testing::TestParamInfo<Kernel> const& info) { return info.param.arch; });

TEST(SolutionState, RejectsAMissingKey)
{
    auto           handle = metadata("gfx950");
    msgpack::zone  zone;
    auto           state = without(customConfig(handle.get()), {"ProblemType", "Batched"}, zone);
    try
    {
        solutionFromState(state);
        FAIL() << "a state without ProblemType.Batched was accepted";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_STREQ(error.what(), "Solution state has no ProblemType.Batched");
    }
}
