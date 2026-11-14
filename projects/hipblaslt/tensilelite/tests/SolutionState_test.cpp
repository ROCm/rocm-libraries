// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// solutionFromState builds a TensileLite solution record from a derived
// solution state. These tests pack that state with msgpack::packer. They do
// not read assembly kernels.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>

// The headers declaring ContractionProblemGemm come before <msgpack.hpp>, which
// reaches the Windows SDK headers that define the enum name CONST as a macro.
#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>

#include <msgpack.hpp>

#include <Tensile/msgpack/SolutionState.hpp>

using namespace TensileLite;

namespace
{
    using ProblemPredicate = Predicates::Predicate<ContractionProblemGemm>;
    using Packer           = msgpack::packer<msgpack::sbuffer>;

    class StateMap
    {
    public:
        void boolean(std::string key, bool value)
        {
            add(std::move(key), [value](Packer& packer, msgpack::sbuffer&) { packer.pack(value); });
        }

        void integer(std::string key, int64_t value)
        {
            add(std::move(key), [value](Packer& packer, msgpack::sbuffer&) { packer.pack(value); });
        }

        // The assembler writes an empty string as float64 0. msgpack-cxx packs
        // an integral double as an integer, so write the note's encoding.
        void emptyString(std::string key)
        {
            add(std::move(key), [](Packer&, msgpack::sbuffer& buffer) {
                char const bytes[] = {char(0xcb), 0, 0, 0, 0, 0, 0, 0, 0};
                buffer.write(bytes, sizeof(bytes));
            });
        }

        void text(std::string key, std::string value)
        {
            add(std::move(key), [value](Packer& packer, msgpack::sbuffer&) { packer.pack(value); });
        }

        void integers(std::string key, std::vector<int64_t> values)
        {
            add(std::move(key), [values](Packer& packer, msgpack::sbuffer&) {
                packer.pack_array(values.size());
                for(auto value : values)
                    packer.pack(value);
            });
        }

        // Msgpack integer keys, the form the assembler writes for {0: 1}.
        void indexMap(std::string key, std::vector<std::pair<int64_t, int64_t>> entries)
        {
            add(std::move(key), [entries](Packer& packer, msgpack::sbuffer&) {
                packer.pack_map(entries.size());
                for(auto const& entry : entries)
                {
                    packer.pack(entry.first);
                    packer.pack(entry.second);
                }
            });
        }

        void nested(std::string key, StateMap const& child)
        {
            add(std::move(key),
                [child](Packer& packer, msgpack::sbuffer& buffer) { child.write(packer, buffer); });
        }

        void maps(std::string key, std::vector<StateMap> children)
        {
            add(std::move(key), [children](Packer& packer, msgpack::sbuffer& buffer) {
                packer.pack_array(children.size());
                for(auto const& child : children)
                    child.write(packer, buffer);
            });
        }

        void textKeyedNumbers(std::string key, std::vector<std::pair<std::string, double>> entries)
        {
            add(std::move(key), [entries](Packer& packer, msgpack::sbuffer&) {
                packer.pack_map(entries.size());
                for(auto const& entry : entries)
                {
                    packer.pack(entry.first);
                    packer.pack(entry.second);
                }
            });
        }

        msgpack::object_handle unpack() const
        {
            msgpack::sbuffer buffer;
            Packer           packer(buffer);
            write(packer, buffer);
            return msgpack::unpack(buffer.data(), buffer.size());
        }

    private:
        using Field = std::pair<std::string, std::function<void(Packer&, msgpack::sbuffer&)>>;

        void add(std::string key, std::function<void(Packer&, msgpack::sbuffer&)> writeValue)
        {
            m_fields.emplace_back(std::move(key), std::move(writeValue));
        }

        void write(Packer& packer, msgpack::sbuffer& buffer) const
        {
            packer.pack_map(m_fields.size());
            for(auto const& field : m_fields)
            {
                packer.pack(field.first);
                field.second(packer, buffer);
            }
        }

        std::vector<Field> m_fields;
    };

    struct StateOptions
    {
        bool                 batched            = true;
        bool                 mirrorSummationA   = false;
        bool                 swappedSummation   = false;
        bool                 conjugateA         = false;
        bool                 assertSizeAsScalar = false;
        bool                 emptySizeEqual     = false;
        bool                 skippedSizeEqual   = false;
        char const*          dataType           = "Half";
        char const*          destDataType       = nullptr;
        char const*          dataTypeA          = nullptr;
        char const*          dataTypeB          = nullptr;
        int                  useBias            = -1;
        char const*          biasSrc            = nullptr;
        std::vector<int64_t> scaleBlockSizesA;
        char const*          int4EncodingA     = nullptr;
        std::vector<int64_t> threadTile        = {4, 2, 1};
        std::vector<int64_t> matrixInstruction = {16, 16, 16, 1};
        char const*          tileStrategy      = nullptr;
        char const*          workAssignment    = nullptr;
        std::vector<int64_t> clusterDim;
        int                  kernArgsVersion           = 2;
        int                  persistentLoopArgsVersion = -1;
        bool                 sizeBounds                = false;
        bool                 badIdealKey               = false;
        bool                 oneLdsBuffer              = false;
        bool                 bufferLoadOff             = false;
        std::vector<int64_t> isa;
        char const*          deviceName = nullptr;
        // A hand-written CustomKernel block. uniformSummationOrder is left out
        // of it when negative, and written as false or true otherwise.
        bool customKernel                = false;
        int  customUniformSummationOrder = -1;
        // Top-level integer keys written after the others.
        std::vector<std::pair<std::string, int64_t>> extraIntegers;
    };

    StateMap problemType(StateOptions const& options)
    {
        StateMap type;
        if(options.swappedSummation)
        {
            // Summation 4 is listed before 3, so bound order is not problem-index
            // order. 4 is mirrored on A and 3 is mirrored on B.
            type.integer("TotalIndices", 5);
            type.integers("IndicesSummation", {4, 3});
            type.integer("NumIndicesC", 3);
            type.integers("IndicesBatch", {2});
            type.integers("IndicesFree", {0, 1});
            type.integers("IndexAssignmentsA", {0, 4, 3, 2});
            type.integers("IndexAssignmentsB", {4, 1, 3, 2});
            type.integers("MirrorDimsA", {4});
            type.integers("MirrorDimsB", {3});
        }
        else
        {
            type.integer("TotalIndices", 4);
            type.integers("IndicesSummation", {3});
            type.integer("NumIndicesC", 3);
            type.integers("IndicesBatch", {2});
            type.integers("IndicesFree", {0, 1});
            type.integers("IndexAssignmentsA", {0, 3, 2});
            type.integers("IndexAssignmentsB", {3, 1, 2});
            if(options.mirrorSummationA)
                type.integers("MirrorDimsA", {3});
        }
        type.text("DataType", options.dataType);
        type.text("DestDataType", options.destDataType ? options.destDataType : options.dataType);
        type.text("ComputeDataType", "Float");
        if(options.dataTypeA)
            type.text("DataTypeA", options.dataTypeA);
        if(options.dataTypeB)
            type.text("DataTypeB", options.dataTypeB);
        if(options.useBias >= 0)
            type.integer("UseBias", options.useBias);
        if(options.biasSrc)
            type.text("BiasSrc", options.biasSrc);
        if(!options.scaleBlockSizesA.empty())
            type.integers("ScaleBlockSizesA", options.scaleBlockSizesA);
        if(options.int4EncodingA)
            type.text("Int4EncodingA", options.int4EncodingA);
        if(options.batched)
            type.boolean("Batched", true);
        type.text("ActivationComputeDataType", "Float");
        type.boolean("TransposeA", false);
        type.boolean("TransposeB", true);
        type.boolean("ComplexConjugateA", options.conjugateA);
        type.boolean("ComplexConjugateB", false);
        type.boolean("HighPrecisionAccumulate", true);
        type.boolean("TLUA", false);
        type.boolean("TLUB", false);
        type.boolean("SwizzleTensorA", false);
        type.boolean("SwizzleTensorB", false);
        type.boolean("FusedGemmA2A", false);
        type.boolean("UseBeta", true);
        type.boolean("StridedBatched", true);
        return type;
    }

    StateMap internalSupport(StateOptions const& options)
    {
        StateMap support;
        support.integer("KernArgsVersion", options.kernArgsVersion);
        if(options.persistentLoopArgsVersion >= 0)
            support.integer("PersistentLoopArgsVersion", options.persistentLoopArgsVersion);
        support.boolean("SupportUserGSU", true);
        support.boolean("SupportCustomWGM", false);
        support.boolean("SupportCustomStaggerU", true);
        support.boolean("UseUniversalArgs", true);
        support.boolean("UseSFC", false);
        return support;
    }

    // A small derived solution state: the map Solution.FromOriginalState reads.
    // SolutionNameMin is float 0, which the metadata note uses for an empty
    // string. _GlobalAccumulation is the string "null", which the note uses
    // for nil.
    StateMap derivedState(StateOptions options = {})
    {
        StateMap state;
        state.emptyString("SolutionNameMin");
        state.text("KernelNameMin", "small_kernel");
        state.integer("SolutionIndex", 7);
        state.boolean("DebugKernel", false);
        state.nested("ProblemType", problemType(options));
        state.integers("ISA", options.isa.empty() ? std::vector<int64_t>{9, 4, 2} : options.isa);
        if(options.deviceName)
            state.text("DeviceNames", options.deviceName);
        state.text("_GlobalAccumulation", "null");
        state.integer("PrefetchGlobalRead", 2);
        state.boolean("EnableMatrixInstruction", false);
        state.integer("ThreadTile0", 4);
        state.integer("ThreadTile1", 2);
        state.integer("NumElementsPerBatchStore", 0);
        state.integer("NumElementsPerThread", 4);
        state.integer("NumThreads", 64);
        state.integer("WavefrontSize", 64);
        state.integers("SpaceFillingAlgo", {});
        state.integer("WorkGroupMapping", 5);
        state.boolean("DirectToVgprA", false);
        state.boolean("DirectToVgprB", false);
        state.boolean("DirectToLdsA", false);
        state.boolean("DirectToLdsB", false);
        state.integers("WorkGroup", {64, 1, 1});
        state.integer("MacroTile0", 32);
        state.integer("MacroTile1", 8);
        state.integers("ThreadTile", options.threadTile);
        state.integers("MatrixInstruction", options.matrixInstruction);
        state.integer("GlobalReadVectorWidthA", 4);
        state.integer("GlobalReadVectorWidthB", 2);
        state.integer("StoreVectorWidth", 4);
        state.integer("DepthU", 16);
        state.integer("GlobalSplitU", 1);
        state.text("KernelLanguage", "Assembly");
        state.integer("_WorkspaceSizePerElemC", 4);
        state.integer("_WorkspaceSizePerElemBias", 0);
        state.boolean("ActivationFused", true);
        state.integer("WorkGroupMappingXCC", 4);
        state.integer("WorkGroupMappingXCCGroup", 8);
        state.boolean("GlobalSplitUCoalesced", false);
        state.boolean("GlobalSplitUWorkGroupMappingRoundRobin", false);
        state.integer("CUOccupancy", 3);
        state.integer("MathClocksUnrolledLoop", 11);
        state.integer("NonTemporalA", 0);
        state.integer("NonTemporalB", 0);
        state.integer("UseCustomMainLoopSchedule", 0);
        state.integer("NonTemporalD", 0);
        state.integer("WaveSeparateGlobalReadA", 0);
        state.integer("WaveSeparateGlobalReadB", 0);
        state.integer("UnrollLoopSwapGlobalReadOrder", 0);
        state.integer("NumLoadsCoalescedA", 1);
        state.integer("NumLoadsCoalescedB", 1);
        state.integers("MIWaveGroup", {1, 1});
        state.integer("VectorWidthA", 1);
        state.integer("VectorWidthB", 1);
        state.integer("LocalSplitU", 1);
        state.integers("ClusterDim",
                       options.clusterDim.empty() ? std::vector<int64_t>{1, 1, 1}
                                                  : options.clusterDim);
        state.integers("PackedC0IndicesX", {});
        state.nested("InternalSupportParams", internalSupport(options));
        if(options.tileStrategy)
            state.text("TileProcessingStrategy", options.tileStrategy);
        if(options.workAssignment)
            state.text("WorkAssignment", options.workAssignment);
        if(options.oneLdsBuffer)
            state.integer("1LDSBuffer", 1);
        if(options.bufferLoadOff)
            state.integer("BufferLoad", 0);
        if(options.sizeBounds)
        {
            StateMap lower;
            lower.text("tag", "SizeGreaterThan");
            lower.integer("index", 0);
            lower.integer("value", 31);
            StateMap upper;
            upper.text("tag", "SizeLessThan");
            upper.integer("index", 0);
            upper.integer("value", 129);
            state.maps("CustomKernelSizePredicate", {lower, upper});
        }
        if(options.customKernel)
        {
            StateMap kernel;
            kernel.text("name", "hand_written");
            kernel.maps("args", {});
            kernel.integers("macrotile", {32, 8, 16});
            kernel.integers("threads", {64, 1, 1});
            kernel.maps("grid", {});
            if(options.customUniformSummationOrder >= 0)
                kernel.boolean("uniformSummationOrder", options.customUniformSummationOrder != 0);
            state.nested("CustomKernel", kernel);
        }
        if(options.badIdealKey)
            state.textKeyedNumbers("Ideals", {{"nope", 1.0}});

        if(options.assertSizeAsScalar)
            state.integer("AssertSizeEqual", 1);
        else if(options.emptySizeEqual)
            state.indexMap("AssertSizeEqual", {});
        else if(options.skippedSizeEqual)
            state.indexMap("AssertSizeEqual", {{0, -1}});
        for(auto const& [key, value] : options.extraIntegers)
            state.integer(key, value);
        return state;
    }

    StateOptions with(void (*edit)(StateOptions&))
    {
        StateOptions options;
        edit(options);
        return options;
    }

    std::shared_ptr<ContractionSolution> build(StateOptions options = {})
    {
        auto handle = derivedState(options).unpack();
        return solutionFromState(handle.get());
    }

    template <typename Object>
    Predicates::And<Object> const&
        conjunction(std::shared_ptr<Predicates::Predicate<Object>> const& predicate)
    {
        auto const* all = dynamic_cast<Predicates::And<Object> const*>(predicate.get());
        EXPECT_NE(all, nullptr);
        if(!all)
            throw std::runtime_error("predicate is not a conjunction");
        return *all;
    }

    template <typename Predicate>
    Predicate const* onlyDirect(std::shared_ptr<ProblemPredicate> const& root)
    {
        Predicate const* found = nullptr;
        for(auto const& child : conjunction<ContractionProblemGemm>(root).value)
        {
            if(auto cast = std::dynamic_pointer_cast<Predicate>(child))
            {
                EXPECT_EQ(found, nullptr) << Predicate::Type();
                found = cast.get();
            }
        }
        return found;
    }

    template <typename Predicate>
    void expectIndexValue(Predicate const* predicate, size_t index, size_t value)
    {
        ASSERT_NE(predicate, nullptr) << Predicate::Type();
        EXPECT_EQ(predicate->type(), Predicate::Type());
        EXPECT_EQ(predicate->index, index);
        EXPECT_EQ(predicate->value, value);
    }
}

TEST(SolutionState, RecordFromDerivedState)
{
    auto solution = build();
    ASSERT_NE(solution, nullptr);

    EXPECT_EQ(solution->solutionName, "");
    EXPECT_FALSE(solution->debugKernel);
    EXPECT_EQ(solution->index, 7);
    EXPECT_EQ(solution->libraryLogicIndex, 7);
    EXPECT_TRUE(solution->ideals.empty());
    EXPECT_EQ(solution->linearModel.slope, 1.0);
    EXPECT_EQ(solution->linearModel.intercept, 0.0);
    EXPECT_EQ(solution->linearModel.max, 1000.0);
    EXPECT_TRUE(solution->customKernel.name.empty());

    auto const& type = solution->problemType;
    EXPECT_EQ(type.operationIdentifier, "Contraction_l_Ailk_Bljk_Cijk_Dijk");
    EXPECT_FALSE(type.groupedGemm);
    EXPECT_EQ(type.activationType, ActivationType::None);
    EXPECT_EQ(type.f32XdlMathOp, rocisa::DataType::Float);
    EXPECT_EQ(type.mxBlockA, 0);
    EXPECT_EQ(type.mxTypeA, rocisa::DataType::Float);
    EXPECT_EQ(type.mxScaleFormat, 0);

    auto const& size = solution->sizeMapping;
    EXPECT_EQ(size.staggerU, 0);
    EXPECT_EQ(size.magicDivAlg, 1);
    EXPECT_EQ(size.tileProcessingStrategy, TileProcessingStrategy::None);
    EXPECT_EQ(size.workAssignment, WorkAssignment::StaticGrid);
    EXPECT_EQ(size.globalAccumulation, 0);
    EXPECT_EQ(size.expertSchedulingMode, 0);
    EXPECT_FALSE(size.OneLDSBuffer);
    EXPECT_FALSE(size.SourceSwap);
    EXPECT_EQ(size.UnrollMajorLDSA, -1);
    EXPECT_EQ(size.UnrollMajorLDSB, -1);
    EXPECT_FALSE(size.enableLDSTrA);
    EXPECT_FALSE(size.enableLDSTrB);
    EXPECT_TRUE(size.BufferLoad);
    EXPECT_TRUE(size.BufferStore);
    EXPECT_FALSE(type.stochasticRounding);

    EXPECT_EQ(solution->internalArgsSupport.persistentLoopArgsVersion, 0);
    EXPECT_EQ(onlyDirect<Predicates::Contraction::SizeEqual>(solution->problemPredicate), nullptr);
}

TEST(SolutionState, DictValuedAssertSizeEqual)
{
    using namespace Predicates::Contraction;
    EXPECT_EQ(onlyDirect<SizeEqual>(build(with([](StateOptions& options) {
                                        options.emptySizeEqual = true;
                                    }))->problemPredicate),
              nullptr);
    EXPECT_EQ(onlyDirect<SizeEqual>(build(with([](StateOptions& options) {
                                        options.skippedSizeEqual = true;
                                    }))->problemPredicate),
              nullptr);

    auto scalar = derivedState(with([](StateOptions& options) {
                      options.assertSizeAsScalar = true;
                  })).unpack();
    try
    {
        solutionFromState(scalar.get());
        FAIL() << "a scalar AssertSizeEqual was accepted";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(
            std::string(error.what()).find("AssertSizeEqual must be a dict of {index: value}"),
            std::string::npos);
    }
}

TEST(SolutionState, NamesAMirroredSummation)
{
    StateOptions options;
    options.mirrorSummationA = true;
    auto solution            = build(options);
    // Batch index 2 and summation 3 mirrored on A. The default fixture is the
    // unmirrored Contraction_l_Ailk_Bljk_Cijk_Dijk.
    EXPECT_EQ(solution->problemType.operationIdentifier, "Contraction_l_AiLk_Bljk_Cijk_Dijk");
}

TEST(SolutionState, NamesSummationOrder)
{
    StateOptions options;
    options.swappedSummation = true;
    auto solution            = build(options);
    EXPECT_EQ(solution->problemType.operationIdentifier, "Contraction_lm_AiLmk_BljMk_Cijk_Dijk");
}

TEST(SolutionState, NamesAConjugateOnARealType)
{
    StateOptions options;
    options.conjugateA = true;
    EXPECT_EQ(build(options)->problemType.operationIdentifier,
              "Contraction_l_AilkC_Bljk_Cijk_Dijk");
}

TEST(SolutionState, BuildsWithoutADevice)
{
    // Constructing the record must not call hipGetDevice. On a machine with a
    // GPU that call would store a non-zero count here. The test constructor
    // also leaves the count at 0, but it marks the query done, so evaluation
    // rejects the kernel instead of reading the device.
    auto        solution = build();
    auto const* xcc
        = onlyDirect<Predicates::Contraction::WorkgroupMappingXCCCheck>(solution->problemPredicate);
    ASSERT_NE(xcc, nullptr);
    EXPECT_EQ(xcc->value, (std::array<int, 2>{4, 8}));
    EXPECT_EQ(xcc->cuCount, size_t{0});

    auto problem = ContractionProblemGemm::GEMM(
        false, false, 1024, 1024, 1024, 1024, 1024, 1024, 1.0, false, 1);
    int        deviceCount = 0;
    hipError_t deviceErr   = hipGetDeviceCount(&deviceCount);
    if(deviceErr != hipSuccess || deviceCount == 0)
    {
        EXPECT_THROW((*xcc)(problem), std::runtime_error);
        return;
    }
    EXPECT_TRUE((*xcc)(problem));
    EXPECT_NE(xcc->cuCount, size_t{0});
}

TEST(SolutionState, ClusterStreamKAcceptsPersistentLoopArgsVersion2)
{
    StateOptions options;
    options.tileStrategy              = "StreamK";
    options.workAssignment            = "StaticGrid";
    options.clusterDim                = {2, 1};
    options.kernArgsVersion           = 3;
    options.persistentLoopArgsVersion = 2;
    auto solution                     = build(options);
    EXPECT_EQ(solution->sizeMapping.tileProcessingStrategy, TileProcessingStrategy::StreamK);
    EXPECT_EQ(solution->sizeMapping.workAssignment, WorkAssignment::StaticGrid);
    EXPECT_EQ(solution->sizeMapping.clusterDim, TensileLite::dim3(2, 1, 1));
    EXPECT_EQ(solution->internalArgsSupport.persistentLoopArgsVersion, 2);
    EXPECT_EQ(solution->internalArgsSupport.version, 3);
}

TEST(SolutionState, DataParallelAcceptsPersistentLoopArgsVersion1)
{
    StateOptions options;
    options.tileStrategy              = "DataParallel";
    options.kernArgsVersion           = 3;
    options.persistentLoopArgsVersion = 1;
    auto solution                     = build(options);
    EXPECT_EQ(solution->sizeMapping.tileProcessingStrategy, TileProcessingStrategy::DataParallel);
    EXPECT_EQ(solution->sizeMapping.workAssignment, WorkAssignment::StaticGrid);
    EXPECT_EQ(solution->internalArgsSupport.persistentLoopArgsVersion, 1);
    EXPECT_EQ(solution->internalArgsSupport.version, 3);
}

TEST(SolutionState, MixedMxOperandsUseTheirOwnComputeInputs)
{
    StateOptions options;
    options.dataType  = "F8";
    options.dataTypeA = "F8";
    options.dataTypeB = "B8";
    auto solution     = build(options);
    EXPECT_EQ(solution->problemType.aType, rocisa::DataType::Float8);
    EXPECT_EQ(solution->problemType.bType, rocisa::DataType::BFloat8);
    EXPECT_EQ(solution->problemType.computeInputTypeA, rocisa::DataType::Float8);
    EXPECT_EQ(solution->problemType.computeInputTypeB, rocisa::DataType::BFloat8);
}

// Without DataTypeA and DataTypeB, a packed FP8/BF8 DataType gives A the first
// half and B the second. The MAC inputs stay the packed type.
TEST(SolutionState, PackedFp8OperandsSplitIntoAAndB)
{
    using rocisa::DataType;
    struct Case
    {
        char const* name;
        DataType    packed;
        DataType    a;
        DataType    b;
    };
    for(auto const& c :
        {Case{"Float8BFloat8", DataType::Float8BFloat8, DataType::Float8, DataType::BFloat8},
         Case{"BFloat8Float8", DataType::BFloat8Float8, DataType::BFloat8, DataType::Float8},
         Case{"Float8BFloat8_fnuz",
              DataType::Float8BFloat8_fnuz,
              DataType::Float8_fnuz,
              DataType::BFloat8_fnuz},
         Case{"BFloat8Float8_fnuz",
              DataType::BFloat8Float8_fnuz,
              DataType::BFloat8_fnuz,
              DataType::Float8_fnuz}})
    {
        StateOptions options;
        options.dataType     = c.name;
        options.destDataType = "Half";
        auto const& type     = build(options)->problemType;
        EXPECT_EQ(type.aType, c.a) << c.name;
        EXPECT_EQ(type.bType, c.b) << c.name;
        EXPECT_EQ(type.computeInputTypeA, c.packed) << c.name;
        EXPECT_EQ(type.computeInputTypeB, c.packed) << c.name;
    }
}

TEST(SolutionState, BiasSrcNamesTheOperand)
{
    char const* const sources[] = {"A", "B", "C", "D"};
    for(int i = 0; i < 4; ++i)
    {
        StateOptions options;
        options.useBias = 1;
        options.biasSrc = sources[i];
        EXPECT_EQ(build(options)->problemType.biasSrcWhiteList, std::vector<int>{i}) << sources[i];
    }

    StateOptions options;
    options.useBias = 1;
    options.biasSrc = "E";
    try
    {
        build(options);
        FAIL() << "BiasSrc E was accepted";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("one of A, B, C and D"), std::string::npos)
            << error.what();
    }
}

TEST(SolutionState, Int4ARecordsGroupScalePredicates)
{
    using namespace Predicates::Contraction;
    StateOptions options;
    options.dataType         = "I4";
    options.dataTypeA        = "I4";
    options.scaleBlockSizesA = {128, 32};
    auto        solution     = build(options);
    auto const& root         = conjunction<ContractionProblemGemm>(solution->problemPredicate);
    Predicates::Or<ContractionProblemGemm> const* either = nullptr;
    for(auto const& child : root.value)
        if(auto cast = std::dynamic_pointer_cast<Predicates::Or<ContractionProblemGemm>>(child))
            either = cast.get();
    ASSERT_NE(either, nullptr);
    ASSERT_EQ(either->value.size(), 2u);
    auto first  = std::dynamic_pointer_cast<ScaleBlockSizeA>(either->value[0]);
    auto second = std::dynamic_pointer_cast<ScaleBlockSizeA>(either->value[1]);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->value, 32);
    EXPECT_EQ(second->value, 128);
    auto const* zero = onlyDirect<ScaleZeroPointA>(solution->problemPredicate);
    ASSERT_NE(zero, nullptr);
    EXPECT_FALSE(zero->value);
    auto const* encoding = onlyDirect<Int4EncodingA>(solution->problemPredicate);
    ASSERT_NE(encoding, nullptr);
    EXPECT_EQ(encoding->value, ContractionProblemGemm::Int4Encoding::Signed);
}

TEST(SolutionState, CustomKernelSizeBoundsArePredicates)
{
    using namespace Predicates::Contraction;
    StateOptions options;
    options.sizeBounds = true;
    auto solution      = build(options);
    expectIndexValue(onlyDirect<SizeGreaterThan>(solution->problemPredicate), 0, 31);
    expectIndexValue(onlyDirect<SizeLessThan>(solution->problemPredicate), 0, 129);
}

TEST(SolutionState, AssertAlphaValueOneIsTheOnlyAlphaValue)
{
    using namespace Predicates::Contraction;
    StateOptions options;
    options.extraIntegers = {{"AssertAlphaValue", 1}};
    auto alpha            = onlyDirect<AlphaValue>(build(options)->problemPredicate);
    ASSERT_NE(alpha, nullptr);
    EXPECT_EQ(alpha->value, ScalarValue::One);

    options.extraIntegers = {{"AssertAlphaValue", 2}};
    try
    {
        build(options);
        FAIL() << "AssertAlphaValue 2 was accepted";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("AssertAlphaValue"), std::string::npos)
            << error.what();
    }
}

TEST(SolutionState, CustomKernelUniformSummationOrderFollowsTheState)
{
    StateOptions options;
    options.customKernel = true;

    options.customUniformSummationOrder = 1;
    auto declared                       = build(options);
    EXPECT_EQ(declared->customKernel.name, "hand_written");
    EXPECT_TRUE(declared->customKernel.uniformSummationOrder);
    EXPECT_FALSE(declared->customKernel.generated);

    options.customUniformSummationOrder = 0;
    EXPECT_FALSE(build(options)->customKernel.uniformSummationOrder);

    // Contractions.CustomKernel.FromOriginalState defaults it to false.
    options.customUniformSummationOrder = -1;
    EXPECT_FALSE(build(options)->customKernel.uniformSummationOrder);
}

TEST(SolutionState, SizeMappingOptionalsFollowTheState)
{
    StateOptions options;
    options.oneLdsBuffer  = true;
    options.bufferLoadOff = true;
    auto        solution  = build(options);
    auto const& size      = solution->sizeMapping;
    EXPECT_TRUE(size.OneLDSBuffer);
    EXPECT_FALSE(size.BufferLoad);
    EXPECT_TRUE(size.BufferStore);
    EXPECT_EQ(size.WavefrontSize, 64);
}

TEST(SolutionState, IdealsKeyNamesTheFailure)
{
    StateOptions options;
    options.badIdealKey = true;
    try
    {
        build(options);
        FAIL() << "a non-integer Ideals key was accepted";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("Ideals key"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("nope"), std::string::npos);
    }
}

TEST(SolutionState, Gfx950DeviceNameSelectsTheChip)
{
    StateOptions options;
    options.isa          = {9, 5, 0};
    options.deviceName   = "Device 75a0";
    auto        solution = build(options);
    auto const& hardware = dynamic_cast<Predicates::IsSubclass<Hardware, AMDGPU> const&>(
        *solution->hardwarePredicate);
    auto const* all = dynamic_cast<Predicates::And<AMDGPU> const*>(hardware.value.get());
    ASSERT_NE(all, nullptr);
    bool sawProcessor = false;
    bool sawChip      = false;
    for(auto const& child : all->value)
    {
        if(auto processor = std::dynamic_pointer_cast<Predicates::GPU::ProcessorEqual>(child))
        {
            EXPECT_EQ(processor->value, AMDGPU::Processor::gfx950);
            sawProcessor = true;
        }
        if(auto chip = std::dynamic_pointer_cast<Predicates::GPU::PciChipIdEqual>(child))
        {
            EXPECT_EQ(chip->value, 0x75a0);
            sawChip = true;
        }
    }
    EXPECT_TRUE(sawProcessor);
    EXPECT_TRUE(sawChip);

    // Python matches the id's text, so a zero-padded id names no chip.
    options.deviceName = "Device 075a0";
    auto const& padded = dynamic_cast<Predicates::IsSubclass<Hardware, AMDGPU> const&>(
        *build(options)->hardwarePredicate);
    EXPECT_NE(dynamic_cast<Predicates::GPU::ProcessorEqual const*>(padded.value.get()), nullptr);
}

TEST(SolutionState, IsaSteppingIsHexadecimal)
{
    // ISA {9, 0, 10} is gfx90a, not gfx9010.
    StateOptions options;
    options.isa          = {9, 0, 10};
    auto        solution = build(options);
    auto const& hardware = dynamic_cast<Predicates::IsSubclass<Hardware, AMDGPU> const&>(
        *solution->hardwarePredicate);
    auto const& processor = dynamic_cast<Predicates::GPU::ProcessorEqual const&>(*hardware.value);
    EXPECT_EQ(processor.value, AMDGPU::Processor::gfx90a);
}

TEST(SolutionState, RejectsAMissingKey)
{
    auto state
        = derivedState(with([](StateOptions& options) { options.batched = false; })).unpack();
    try
    {
        solutionFromState(state.get());
        FAIL() << "a state without ProblemType.Batched was accepted";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_STREQ(error.what(), "Solution state has no ProblemType.Batched");
    }
}

// The Python pipeline translates StreamK and StreamKForceDPOnly and refuses
// PersistentLoop, so a derived state has none of them.
TEST(SolutionState, RejectsLegacySelectors)
{
    for(auto const& [key, value] : std::vector<std::pair<std::string, int64_t>>{
            {"StreamK", 3}, {"StreamKForceDPOnly", 0}, {"PersistentLoop", 1}})
    {
        StateOptions options;
        options.extraIntegers = {{key, value}};
        try
        {
            build(options);
            ADD_FAILURE() << "the legacy selector " << key << " was accepted";
        }
        catch(std::runtime_error const& error)
        {
            EXPECT_NE(std::string(error.what()).find("legacy selector " + key + ";"),
                      std::string::npos)
                << error.what();
        }
    }
}

TEST(SolutionState, RejectsADisagreeingPersistentLoop)
{
    StateOptions options;
    options.extraIntegers = {{"_PersistentLoop", 0}};
    EXPECT_NO_THROW(build(options));

    options.extraIntegers = {{"_PersistentLoop", 1}};
    try
    {
        build(options);
        FAIL() << "_PersistentLoop 1 was accepted without a persistent strategy";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("_PersistentLoop disagrees"), std::string::npos)
            << error.what();
    }
}

TEST(SolutionState, ShortTileListsLeaveTheRestZero)
{
    StateOptions options;
    options.threadTile        = {4, 2};
    options.matrixInstruction = {};
    auto const& size          = build(options)->sizeMapping;
    EXPECT_EQ(size.threadTile, TensileLite::dim3(4, 2, 0));
    EXPECT_EQ(size.matrixInstruction, (std::array<int, 4>{0, 0, 0, 0}));
}

TEST(SolutionState, SizeMappingTilesStartAtZero)
{
    // Storage that is not zero, as a stack or heap allocation can be.
    alignas(SizeMapping) unsigned char storage[sizeof(SizeMapping)];
    std::memset(storage, 0xff, sizeof(storage));
    auto* size = new(storage) SizeMapping;
    EXPECT_EQ(size->threadTile, TensileLite::dim3(0, 0, 0));
    EXPECT_EQ(size->matrixInstruction, (std::array<int, 4>{0, 0, 0, 0}));
    EXPECT_EQ(size->waveGroup, (std::array<int, 2>{0, 0}));
    size->~SizeMapping();
}

// Version 2 is only StreamK on a static grid with a spatial cluster. Each
// state builds at version 0.
TEST(SolutionState, RejectsPersistentLoopArgsVersion2WithoutASpatialCluster)
{
    StateOptions streamK;
    streamK.tileStrategy    = "StreamK";
    streamK.workAssignment  = "StaticGrid";
    streamK.clusterDim      = {2, 1};
    streamK.kernArgsVersion = 3;

    auto flat                 = streamK;
    flat.clusterDim           = {1, 1};
    auto dataParallel         = streamK;
    dataParallel.tileStrategy = "DataParallel";
    auto dynamic              = streamK;
    dynamic.workAssignment    = "DynamicWorkQueue";

    for(auto [name, options] : {std::pair{"1x1 cluster", flat},
                                std::pair{"DataParallel", dataParallel},
                                std::pair{"DynamicWorkQueue", dynamic}})
    {
        options.persistentLoopArgsVersion = 0;
        EXPECT_NO_THROW(build(options)) << name;
        options.persistentLoopArgsVersion = 2;
        EXPECT_THROW(build(options), std::runtime_error) << name;
    }
}

// Solution.FromOriginalState accepts the next two states; the library could
// not launch them.
TEST(SolutionState, RejectsAnUnsupportedStreamKClusterShape)
{
    StateOptions options;
    options.tileStrategy              = "StreamK";
    options.workAssignment            = "StaticGrid";
    options.clusterDim                = {1, 3};
    options.kernArgsVersion           = 3;
    options.persistentLoopArgsVersion = 2;
    try
    {
        build(options);
        FAIL() << "ClusterDim [1, 3] was accepted";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("a supported spatial cluster"), std::string::npos)
            << error.what();
    }
}

TEST(SolutionState, RejectsAnUnknownInt4EncodingA)
{
    StateOptions options;
    options.dataType         = "I4";
    options.dataTypeA        = "I4";
    options.scaleBlockSizesA = {32};
    options.int4EncodingA    = "Unsigned";
    try
    {
        build(options);
        FAIL() << "Int4EncodingA Unsigned was accepted";
    }
    catch(std::runtime_error const& error)
    {
        EXPECT_STREQ(error.what(), "Int4EncodingA must be Signed or UnsignedBias8");
    }
}
