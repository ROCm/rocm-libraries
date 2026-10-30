// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// solutionFromState has to emit the record Solution.FromOriginalState builds.
// The lines in *.record.txt are that record. test_solution_state_record.py
// fails when the Python builder changes them. This test fails after those
// lines are regenerated, until this builder emits the same ones.

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/ContractionTaskPredicates.hpp>

#include <Tensile/msgpack/SolutionState.hpp>

#include <rocisa/include/enum.hpp>

using namespace TensileLite;

namespace
{
    std::string jsonString(std::string const& text)
    {
        std::string out = "\"";
        for(unsigned char c : text)
        {
            if(c == '\\' || c == '"')
                out.push_back('\\');
            out.push_back(static_cast<char>(c));
        }
        out.push_back('"');
        return out;
    }

    std::string jsonBool(bool value)
    {
        return value ? "true" : "false";
    }

    template <typename T>
    std::string jsonNumber(T value)
    {
        return std::to_string(value);
    }

    std::string jsonInts(std::vector<int> const& values)
    {
        std::string out = "[";
        for(size_t i = 0; i < values.size(); ++i)
        {
            if(i != 0)
                out += ',';
            out += std::to_string(values[i]);
        }
        out += ']';
        return out;
    }

    template <size_t N>
    std::string jsonArray(std::array<int, N> const& values)
    {
        std::string out = "[";
        for(size_t i = 0; i < N; ++i)
        {
            if(i != 0)
                out += ',';
            out += std::to_string(values[i]);
        }
        out += ']';
        return out;
    }

    std::string jsonDim(TensileLite::dim3 const& value, int count)
    {
        int const parts[]
            = {static_cast<int>(value.x), static_cast<int>(value.y), static_cast<int>(value.z)};
        std::string out = "[";
        for(int i = 0; i < count; ++i)
        {
            if(i != 0)
                out += ',';
            out += std::to_string(parts[i]);
        }
        out += ']';
        return out;
    }

    std::string jsonDataTypes(std::vector<rocisa::DataType> const& types)
    {
        std::string out = "[";
        for(size_t i = 0; i < types.size(); ++i)
        {
            if(i != 0)
                out += ',';
            out += std::to_string(static_cast<int>(types[i]));
        }
        out += ']';
        return out;
    }

    std::string jsonType(rocisa::DataType type)
    {
        return jsonString(rocisa::toString(type));
    }

    template <typename Object>
    void collect(Predicates::Predicate<Object> const& predicate, std::vector<std::string>& out)
    {
        out.push_back(predicate.type());
        if(auto const* conj = dynamic_cast<Predicates::And<Object> const*>(&predicate))
        {
            for(auto const& term : conj->value)
                collect(*term, out);
        }
        else if(auto const* disj = dynamic_cast<Predicates::Or<Object> const*>(&predicate))
        {
            for(auto const& term : disj->value)
                collect(*term, out);
        }
        else if(auto const* neg = dynamic_cast<Predicates::Not<Object> const*>(&predicate))
        {
            collect(*neg->value, out);
        }
    }

    void collectHardware(Predicates::Predicate<Hardware> const& predicate,
                         std::vector<std::string>&              out)
    {
        out.push_back(predicate.type());
        if(auto const* gpu
           = dynamic_cast<Predicates::IsSubclass<Hardware, AMDGPU> const*>(&predicate))
            collect(*gpu->value, out);
    }

    std::string jsonStrings(std::vector<std::string> const& values)
    {
        std::string out = "[";
        for(size_t i = 0; i < values.size(); ++i)
        {
            if(i != 0)
                out += ',';
            out += jsonString(values[i]);
        }
        out += ']';
        return out;
    }

    std::string jsonArgs(std::vector<CustomArgDefinition> const& args)
    {
        std::string out = "[";
        for(size_t i = 0; i < args.size(); ++i)
        {
            if(i != 0)
                out += ',';
            out += "{\"semantic\":" + jsonString(toString(args[i].semantic))
                   + ",\"type\":" + jsonString(toString(args[i].type)) + "}";
        }
        out += ']';
        return out;
    }

    std::string jsonGrid(vector3<CustomGridSize> const& grid)
    {
        CustomGridSize const parts[] = {grid.x, grid.y, grid.z};
        std::string          out     = "[";
        for(int i = 0; i < 3; ++i)
        {
            if(i != 0)
                out += ',';
            out += jsonString(toString(parts[i]));
        }
        out += ']';
        return out;
    }

    std::map<std::string, std::string> recordLines(ContractionSolution const& solution)
    {
        std::map<std::string, std::string> lines;
        auto                               add = [&](std::string key, std::string value) {
            lines.emplace(std::move(key), std::move(value));
        };

        add("solutionName", jsonString(solution.solutionName));
        add("kernelName", jsonString(solution.kernelName));

        auto const& size = solution.sizeMapping;
        add("sizeMapping.waveNum", jsonNumber(size.waveNum));
        add("sizeMapping.WavefrontSize", jsonNumber(size.WavefrontSize));
        add("sizeMapping.workGroup", jsonDim(size.workGroupSize, 3));
        add("sizeMapping.macroTile", jsonDim(size.macroTile, 3));
        add("sizeMapping.threadTile", jsonDim(size.threadTile, 2));
        add("sizeMapping.matrixInstruction", jsonArray(size.matrixInstruction));
        add("sizeMapping.grvwA", jsonNumber(size.grvwA));
        add("sizeMapping.grvwB", jsonNumber(size.grvwB));
        add("sizeMapping.gwvwC", jsonNumber(size.gwvwC));
        add("sizeMapping.gwvwD", jsonNumber(size.gwvwD));
        add("sizeMapping.depthU", jsonNumber(size.depthU));
        add("sizeMapping.staggerU", jsonNumber(size.staggerU));
        add("sizeMapping.staggerUMapping", jsonNumber(size.staggerUMapping));
        add("sizeMapping.globalSplitUPGR", jsonNumber(size.globalSplitUPGR));
        add("sizeMapping.globalSplitU", jsonNumber(size.globalSplitU));
        add("sizeMapping.staggerStrideShift", jsonNumber(size.staggerStrideShift));
        add("sizeMapping.workGroupMapping", jsonNumber(size.workGroupMapping));
        add("sizeMapping.packBatchDims", jsonNumber(size.packBatchDims));
        add("sizeMapping.magicDivAlg", jsonNumber(size.magicDivAlg));
        add("sizeMapping.tileProcessingStrategy",
            jsonString(toString(size.tileProcessingStrategy)));
        add("sizeMapping.workAssignment", jsonString(toString(size.workAssignment)));
        add("sizeMapping.streamKAtomic", jsonNumber(size.streamKAtomic));
        add("sizeMapping.prefetchAcrossPersistent", jsonNumber(size.prefetchAcrossPersistent));
        add("sizeMapping.sourceKernel", jsonBool(size.sourceKernel));
        add("sizeMapping.globalAccumulation", jsonNumber(size.globalAccumulation));
        add("sizeMapping.adaptiveGemmGSUA", jsonNumber(size.adaptiveGemmGSUA));
        add("sizeMapping.workspaceSizePerElemC", jsonNumber(size.workspaceSizePerElemC));
        add("sizeMapping.workspaceSizePerElemBias", jsonNumber(size.workspaceSizePerElemBias));
        add("sizeMapping.activationFused", jsonBool(size.activationFused));
        add("sizeMapping.workGroupMappingXCC", jsonNumber(size.workGroupMappingXCC));
        add("sizeMapping.workGroupMappingXCCGroup", jsonNumber(size.workGroupMappingXCCGroup));
        add("sizeMapping.globalSplitUCoalesced", jsonBool(size.globalSplitUCoalesced));
        add("sizeMapping.globalSplitUWorkGroupMappingRoundRobin",
            jsonBool(size.globalSplitUWorkGroupMappingRoundRobin));
        add("sizeMapping.CUOccupancy", jsonNumber(size.CUOccupancy));
        add("sizeMapping.PrefetchGlobalRead", jsonNumber(size.PrefetchGlobalRead));
        add("sizeMapping.MathClocksUnrolledLoop", jsonNumber(size.MathClocksUnrolledLoop));
        add("sizeMapping.synchronizerSizePerWG", jsonNumber(size.synchronizerSizePerWG));
        add("sizeMapping.nonTemporalA", jsonNumber(size.nonTemporalA));
        add("sizeMapping.nonTemporalB", jsonNumber(size.nonTemporalB));
        add("sizeMapping.temporalHintA", jsonNumber(size.temporalHintA));
        add("sizeMapping.temporalHintB", jsonNumber(size.temporalHintB));
        add("sizeMapping.hasTemporalHint", jsonBool(size.hasTemporalHint));
        add("sizeMapping.adaptiveGemmNTAB", jsonNumber(size.adaptiveGemmNTAB));
        add("sizeMapping.customMainLoopScheduling", jsonNumber(size.customMainLoopScheduling));
        add("sizeMapping.useSubtileImpl", jsonBool(size.useSubtileImpl));
        add("sizeMapping.NonTemporalD", jsonNumber(size.NonTemporalD));
        add("sizeMapping.WaveSeparateGlobalReadA", jsonNumber(size.WaveSeparateGlobalReadA));
        add("sizeMapping.WaveSeparateGlobalReadB", jsonNumber(size.WaveSeparateGlobalReadB));
        add("sizeMapping.UnrollLoopSwapGlobalReadOrder",
            jsonNumber(size.UnrollLoopSwapGlobalReadOrder));
        add("sizeMapping.DirectToVgprA", jsonBool(size.DirectToVgprA));
        add("sizeMapping.DirectToVgprB", jsonBool(size.DirectToVgprB));
        add("sizeMapping.NumLoadsCoalescedA", jsonNumber(size.NumLoadsCoalescedA));
        add("sizeMapping.NumLoadsCoalescedB", jsonNumber(size.NumLoadsCoalescedB));
        add("sizeMapping.WaveGroup", jsonArray(size.waveGroup));
        add("sizeMapping.VectorWidthA", jsonNumber(size.VectorWidthA));
        add("sizeMapping.VectorWidthB", jsonNumber(size.VectorWidthB));
        add("sizeMapping.LocalSplitU", jsonNumber(size.LocalSplitU));
        add("sizeMapping.DirectToLdsA", jsonBool(size.DirectToLdsA));
        add("sizeMapping.DirectToLdsB", jsonBool(size.DirectToLdsB));
        add("sizeMapping.OneLDSBuffer", jsonBool(size.OneLDSBuffer));
        add("sizeMapping.SourceSwap", jsonBool(size.SourceSwap));
        add("sizeMapping.UnrollMajorLDSA", jsonNumber(size.UnrollMajorLDSA));
        add("sizeMapping.UnrollMajorLDSB", jsonNumber(size.UnrollMajorLDSB));
        add("sizeMapping.enableLDSTrA", jsonBool(size.enableLDSTrA));
        add("sizeMapping.enableLDSTrB", jsonBool(size.enableLDSTrB));
        add("sizeMapping.ExpertSchedulingMode", jsonNumber(size.expertSchedulingMode));
        add("sizeMapping.clusterDim", jsonDim(size.clusterDim, 2));
        add("sizeMapping.BufferLoad", jsonBool(size.BufferLoad));
        add("sizeMapping.BufferStore", jsonBool(size.BufferStore));

        auto const& args = solution.internalArgsSupport;
        add("internalArgsSupport.version", jsonNumber(args.version));
        add("internalArgsSupport.persistentLoopArgsVersion",
            jsonNumber(args.persistentLoopArgsVersion));
        add("internalArgsSupport.gsu", jsonBool(args.gsu));
        add("internalArgsSupport.wgm", jsonBool(args.wgm));
        add("internalArgsSupport.staggerU", jsonBool(args.staggerU));
        add("internalArgsSupport.perTileExtraIters", jsonBool(args.perTileExtraIters));
        add("internalArgsSupport.useUniversalArgs", jsonBool(args.useUniversalArgs));
        add("internalArgsSupport.useSFC", jsonBool(args.useSFC));

        auto const& problem = solution.problemType;
        add("problemType.operationIdentifier", jsonString(problem.operationIdentifier));
        add("problemType.transA", jsonBool(problem.transA));
        add("problemType.transB", jsonBool(problem.transB));
        add("problemType.computeInputTypeA", jsonType(problem.computeInputTypeA));
        add("problemType.computeInputTypeB", jsonType(problem.computeInputTypeB));
        add("problemType.aType", jsonType(problem.aType));
        add("problemType.bType", jsonType(problem.bType));
        add("problemType.cType", jsonType(problem.cType));
        add("problemType.dType", jsonType(problem.dType));
        add("problemType.eType", jsonType(problem.eType));
        add("problemType.computeType", jsonType(problem.computeType));
        add("problemType.useBeta", jsonBool(problem.useBeta));
        add("problemType.useBias", jsonNumber(problem.useBias));
        add("problemType.biasSrcWhiteList", jsonInts(problem.biasSrcWhiteList));
        add("problemType.useE", jsonBool(problem.useE));
        add("problemType.useGateResidual", jsonBool(problem.useGateResidual));
        add("problemType.gateResidualDataTypeWhiteList",
            jsonDataTypes(problem.gateResidualDataTypeWhiteList));
        add("problemType.useScaleAB", jsonString(problem.useScaleAB));
        add("problemType.useScaleCD", jsonBool(problem.useScaleCD));
        add("problemType.useScaleAlphaVec", jsonNumber(problem.useScaleAlphaVec));
        add("problemType.biasDataTypeWhiteList", jsonDataTypes(problem.biasDataTypeWhiteList));
        add("problemType.highPrecisionAccumulate", jsonBool(problem.highPrecisionAccumulate));
        add("problemType.useInitialStridesAB", jsonBool(problem.useInitialStridesAB));
        add("problemType.useInitialStridesCD", jsonBool(problem.useInitialStridesCD));
        add("problemType.stridedBatched", jsonBool(problem.stridedBatched));
        add("problemType.groupedGemm", jsonBool(problem.groupedGemm));
        add("problemType.useGradient", jsonBool(problem.useGradient));
        add("problemType.activationType", jsonString(ToString(problem.activationType)));
        add("problemType.activationArgLength", jsonNumber(problem.activationArgLength));
        add("problemType.activationComputeDataType", jsonType(problem.activationComputeDataType));
        add("problemType.activationNoGuard", jsonBool(problem.activationNoGuard));
        add("problemType.sparse", jsonNumber(problem.sparse));
        add("problemType.f32XdlMathOp", jsonType(problem.f32XdlMathOp));
        add("problemType.supportDeviceUserArguments", jsonBool(problem.supportDeviceUserArguments));
        add("problemType.outputAmaxD", jsonBool(problem.outputAmaxD));
        add("problemType.swizzleTensorA", jsonBool(problem.swizzleTensorA));
        add("problemType.swizzleTensorB", jsonBool(problem.swizzleTensorB));
        add("problemType.metadataLayout", jsonNumber(problem.metadataLayout));
        add("problemType.mxBlockA", jsonNumber(problem.mxBlockA));
        add("problemType.mxBlockB", jsonNumber(problem.mxBlockB));
        add("problemType.mxTypeA", jsonType(problem.mxTypeA));
        add("problemType.mxTypeB", jsonType(problem.mxTypeB));
        add("problemType.mxScaleFormat", jsonNumber(problem.mxScaleFormat));
        add("problemType.fusedGemmA2A", jsonBool(problem.fusedGemmA2A));
        add("problemType.stochasticRounding", jsonBool(problem.stochasticRounding));

        std::vector<std::string> tags;
        collect(*solution.problemPredicate, tags);
        add("problemPredicate", jsonStrings(tags));
        tags.clear();
        collect(*solution.taskPredicate, tags);
        add("taskPredicate", jsonStrings(tags));
        tags.clear();
        collectHardware(*solution.hardwarePredicate, tags);
        add("hardwarePredicate", jsonStrings(tags));

        auto const& kernel = solution.customKernel;
        add("customKernel.name", jsonString(kernel.name));
        add("customKernel.args", jsonArgs(kernel.args));
        add("customKernel.macrotile", jsonDim(kernel.macrotile, 3));
        add("customKernel.threads", jsonDim(kernel.threads, 3));
        add("customKernel.grid", jsonGrid(kernel.grid));
        add("customKernel.workspaceType", jsonString(toString(kernel.workspaceType)));
        add("customKernel.workspaceSizePerElemC", jsonNumber(kernel.workspaceSizePerElemC));
        add("customKernel.workspaceSizePerElemBias", jsonNumber(kernel.workspaceSizePerElemBias));
        add("customKernel.generated", jsonBool(kernel.generated));
        return lines;
    }

    std::string readText(std::string const& path)
    {
        std::ifstream      in(path);
        std::ostringstream buffer;
        buffer << in.rdbuf();
        return buffer.str();
    }

    std::vector<char> readBytes(std::string const& path)
    {
        std::ifstream in(path, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(in),
                                 std::istreambuf_iterator<char>());
    }

    std::map<std::string, std::string> parseLines(std::string const& text)
    {
        std::map<std::string, std::string> lines;
        std::istringstream                 stream(text);
        std::string                        line;
        while(std::getline(stream, line))
        {
            if(line.empty())
                continue;
            auto split = line.find('=');
            lines.emplace(line.substr(0, split), line.substr(split + 1));
        }
        return lines;
    }

    void expectRecord(std::string const& name)
    {
        std::string const directory = TENSILELITE_SOLUTION_RECORD_DIR;
        auto              bytes     = readBytes(directory + "/" + name + ".msgpack");
        ASSERT_FALSE(bytes.empty()) << name;
        auto handle   = msgpack::unpack(bytes.data(), bytes.size());
        auto solution = solutionFromState(handle.get());
        ASSERT_NE(solution, nullptr);

        auto expected = parseLines(readText(directory + "/" + name + ".record.txt"));
        auto got      = recordLines(*solution);
        for(auto const& [key, value] : expected)
        {
            auto found = got.find(key);
            EXPECT_TRUE(found != got.end()) << name << " has no C++ value for " << key;
            if(found != got.end())
                EXPECT_EQ(found->second, value) << name << " " << key;
        }
    }
}

TEST(SolutionRecord, MatchesThePythonBuilder)
{
    expectRecord("gfx90a-plain");
    expectRecord("gfx942-plain");
    expectRecord("gfx950-plain");
}
