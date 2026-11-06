// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// solutionFromState has to emit the record Solution.FromOriginalState builds.
// Each fixture in solution-state/record is a derived state (<name>.msgpack)
// with what the Python builder makes of it:
//
//   <name>.record.txt          one line per field of the library record, and the
//                              type sequence of each predicate tree
//   <name>.predicates.msgpack  the problem, task and hardware predicate trees
//                              with their values, in the {type, index, value}
//                              form Python's state() gives them
//
// test_solution_state_record.py fails when the Python builder changes either
// file. This test fails after they are regenerated, until this builder emits
// the same record and the same trees. The trees are compared node by node, so
// a threshold, an operand or a whole predicate the builder drops names its path.
// The fixtures are the directory listing, not a list kept in this file.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/ContractionTaskPredicates.hpp>

#include <Tensile/msgpack/SolutionState.hpp>
#include <Tensile/msgpack/Writing.hpp>

#include "TestData.hpp"

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
            // Contractions.CustomKernel writes `padding` only for an argument that has one.
            out += "{";
            if(args[i].padding != 0)
                out += "\"padding\":" + std::to_string(args[i].padding) + ",";
            out += "\"semantic\":" + jsonString(toString(args[i].semantic))
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
        // InternalArgsSupport.state() names the probe only when it is set.
        if(args.occupancyProbe)
            add("internalArgsSupport.occupancyProbe", jsonBool(args.occupancyProbe));
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
        add("problemType.mxBlockFreeA", jsonNumber(problem.mxBlockFreeA));
        add("problemType.mxBlockFreeB", jsonNumber(problem.mxBlockFreeB));
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
        add("customKernel.uniformSummationOrder", jsonBool(kernel.uniformSummationOrder));
        return lines;
    }

    // Fixtures sit in the test data directory next to the binary, in the build
    // tree and in an installed tree alike. A missing file names its path.
    std::string fixturePath(std::string const& relative)
    {
        return TestData::Instance().file("solution-state/" + relative).string();
    }

    std::ifstream openFixture(std::string const& path, std::ios::openmode mode)
    {
        std::ifstream in(path, mode);
        if(!in)
            throw std::runtime_error("cannot open test fixture " + path + " (test data directory "
                                     + TestData::Instance().dataDir().string() + ")");
        return in;
    }

    std::vector<char> readBytes(std::string const& path)
    {
        auto in = openFixture(path, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(in),
                                 std::istreambuf_iterator<char>());
    }

    std::string readText(std::string const& path)
    {
        auto               in = openFixture(path, std::ios::in);
        std::ostringstream buffer;
        buffer << in.rdbuf();
        return buffer.str();
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

    std::shared_ptr<ContractionSolution> solutionOf(std::string const&      name,
                                                    msgpack::object_handle& handle)
    {
        auto bytes = readBytes(fixturePath("record/" + name + ".msgpack"));
        if(bytes.empty())
            throw std::runtime_error(name + ".msgpack is empty");
        handle        = msgpack::unpack(bytes.data(), bytes.size());
        auto solution = solutionFromState(handle.get());
        if(!solution)
            throw std::runtime_error("solutionFromState returned no solution for " + name);
        return solution;
    }

    void expectRecord(std::string const& name)
    {
        msgpack::object_handle handle;
        auto                   solution = solutionOf(name, handle);

        auto expected = parseLines(readText(fixturePath("record/" + name + ".record.txt")));
        auto got      = recordLines(*solution);
        for(auto const& [key, value] : expected)
        {
            auto found = got.find(key);
            EXPECT_TRUE(found != got.end()) << name << " has no C++ value for " << key;
            if(found != got.end())
                EXPECT_EQ(found->second, value) << name << " " << key;
        }
        // Python records a custom kernel only for a state that has one.
        bool const hasCustomKernel = expected.count("customKernel.name") != 0;
        for(auto const& entry : got)
        {
            if(!hasCustomKernel && entry.first.rfind("customKernel.", 0) == 0)
                continue;
            EXPECT_TRUE(expected.count(entry.first) != 0)
                << name << " records " << entry.first << " that the Python record has not";
        }
    }

    std::string clip(std::string text)
    {
        constexpr size_t limit = 120;
        if(text.size() > limit)
            text = text.substr(0, limit) + "...";
        return text;
    }

    std::string show(msgpack::object const& value)
    {
        std::ostringstream out;
        out << value;
        return clip(out.str());
    }

    bool isInteger(msgpack::object const& value)
    {
        return value.type == msgpack::type::POSITIVE_INTEGER
               || value.type == msgpack::type::NEGATIVE_INTEGER;
    }

    bool isString(msgpack::object const& value)
    {
        return value.type == msgpack::type::STR;
    }

    std::string text(msgpack::object const& value)
    {
        return std::string(value.via.str.ptr, value.via.str.size);
    }

    // The map a predicate node is, by key; empty when the keys are not strings.
    std::map<std::string, msgpack::object const*> keyed(msgpack::object const& value,
                                                        std::string&           problem)
    {
        std::map<std::string, msgpack::object const*> out;
        for(uint32_t i = 0; i < value.via.map.size; ++i)
        {
            auto const& pair = value.via.map.ptr[i];
            if(!isString(pair.key))
            {
                problem = "a map key is not a string: " + show(pair.key);
                return {};
            }
            out.emplace(text(pair.key), &pair.val);
        }
        return out;
    }

    // " (<Type>)" for a node that is a predicate, so a path reads as a tree of
    // predicates and not as bare indices.
    std::string typeTag(msgpack::object const& value)
    {
        if(value.type != msgpack::type::MAP)
            return "";
        std::string problem;
        auto        fields = keyed(value, problem);
        auto        type   = fields.find("type");
        if(type == fields.end() || !isString(*type->second))
            return "";
        return ":" + text(*type->second);
    }

    // The enum value of a DataType name, or -1. Python's state() carries a
    // DataType as its enum value; the library writer encodes the name.
    int dataTypeValue(std::string const& name)
    {
        for(int i = 0; i < static_cast<int>(rocisa::DataType::Count); ++i)
            if(rocisa::toString(static_cast<rocisa::DataType>(i)) == name)
                return i;
        return -1;
    }

    // The two white lists hold DataTypes. Compares the C++ list, whichever
    // form it is in, with the enum values Python wrote.
    std::string dataTypeListMismatch(std::string const&     path,
                                     msgpack::object const& expected,
                                     msgpack::object const& actual)
    {
        if(expected.type != msgpack::type::ARRAY || actual.type != msgpack::type::ARRAY)
            return path + ": a DataType list is not an array";
        if(expected.via.array.size != actual.via.array.size)
            return path + ": Python has " + std::to_string(expected.via.array.size)
                   + " types, C++ has " + std::to_string(actual.via.array.size) + "; Python "
                   + show(expected) + ", C++ " + show(actual);
        for(uint32_t i = 0; i < expected.via.array.size; ++i)
        {
            auto const& want = expected.via.array.ptr[i];
            auto const& have = actual.via.array.ptr[i];
            auto const  value
                = isString(have)
                      ? dataTypeValue(text(have))
                      : (have.type == msgpack::type::POSITIVE_INTEGER ? int(have.via.u64) : -1);
            if(want.type != msgpack::type::POSITIVE_INTEGER || int(want.via.u64) != value)
                return path + "[" + std::to_string(i) + "]: Python " + show(want) + ", C++ "
                       + show(have);
        }
        return "";
    }

    // The path and values of the first node where two trees differ, in the
    // order of a depth-first walk with map keys sorted; empty when they are equal.
    // Integers compare by value whatever width they were packed in, and a bool
    // is not an integer.
    std::string firstMismatch(std::string const&     path,
                              msgpack::object const& expected,
                              msgpack::object const& actual)
    {
        auto differ = [&](std::string const& why) { return path + ": " + why; };
        auto values = [&] { return "Python " + show(expected) + ", C++ " + show(actual); };

        if(isInteger(expected) && isInteger(actual))
        {
            bool same = expected.type == actual.type
                        && (expected.type == msgpack::type::POSITIVE_INTEGER
                                ? expected.via.u64 == actual.via.u64
                                : expected.via.i64 == actual.via.i64);
            return same ? "" : differ(values());
        }
        if(expected.type != actual.type)
            return differ("different kinds: " + values());

        switch(expected.type)
        {
        case msgpack::type::NIL:
            return "";
        case msgpack::type::BOOLEAN:
            return expected.via.boolean == actual.via.boolean ? "" : differ(values());
        case msgpack::type::FLOAT32:
        case msgpack::type::FLOAT64:
            return expected.via.f64 == actual.via.f64 ? "" : differ(values());
        case msgpack::type::STR:
            return text(expected) == text(actual) ? "" : differ(values());
        case msgpack::type::ARRAY:
        {
            auto const common = std::min(expected.via.array.size, actual.via.array.size);
            for(uint32_t i = 0; i < common; ++i)
            {
                auto const& item = expected.via.array.ptr[i];
                auto        next = path + "[" + std::to_string(i) + typeTag(item) + "]";
                auto        diff = firstMismatch(next, item, actual.via.array.ptr[i]);
                if(!diff.empty())
                    return diff;
            }
            // The common elements agree, so the longer side has the extra ones.
            if(expected.via.array.size > actual.via.array.size)
            {
                auto const& extra = expected.via.array.ptr[common];
                return differ("C++ stops after " + std::to_string(common) + " of "
                              + std::to_string(expected.via.array.size) + " elements; Python has "
                              + "[" + std::to_string(common) + typeTag(extra) + "] " + show(extra));
            }
            if(expected.via.array.size < actual.via.array.size)
            {
                auto const& extra = actual.via.array.ptr[common];
                return differ("Python stops after " + std::to_string(common) + " of "
                              + std::to_string(actual.via.array.size) + " elements; C++ has ["
                              + std::to_string(common) + typeTag(extra) + "] " + show(extra));
            }
            return "";
        }
        case msgpack::type::MAP:
        {
            std::string problem;
            auto        want = keyed(expected, problem);
            auto        got  = keyed(actual, problem);
            if(!problem.empty())
                return differ(problem);
            auto const type      = want.find("type");
            bool const dataTypes = type != want.end() && isString(*type->second)
                                   && (text(*type->second) == "BiasDataTypeWhiteList"
                                       || text(*type->second) == "GateResidualDataTypeWhiteList");
            for(auto const& [key, value] : want)
            {
                auto found = got.find(key);
                if(found == got.end())
                    return differ("C++ has no '" + key + "' (Python " + show(*value) + ")");
                auto diff = dataTypes && key == "value"
                                ? dataTypeListMismatch(path + "." + key, *value, *found->second)
                                : firstMismatch(path + "." + key, *value, *found->second);
                if(!diff.empty())
                    return diff;
            }
            for(auto const& [key, value] : got)
                if(want.find(key) == want.end())
                    return differ("Python has no '" + key + "' (C++ " + show(*value) + ")");
            return "";
        }
        default:
            return differ("unsupported kind in the fixture: " + values());
        }
    }

    // The predicate trees of a built solution, encoded by the same traits the
    // library writer uses, and compared with what Python's state() gave.
    void expectPredicates(std::string const& name)
    {
        msgpack::object_handle handle;
        auto                   solution = solutionOf(name, handle);

        auto const bytes = MessagePackWriteSolutionData(*solution);
        auto const written
            = msgpack::unpack(reinterpret_cast<char const*>(bytes.data()), bytes.size());
        ASSERT_EQ(written.get().type, msgpack::type::MAP) << name;
        std::string problem;
        auto const  fields = keyed(written.get(), problem);
        ASSERT_TRUE(problem.empty()) << name << ": " << problem;

        auto const expectedBytes = readBytes(fixturePath("record/" + name + ".predicates.msgpack"));
        ASSERT_FALSE(expectedBytes.empty()) << name << ".predicates.msgpack is empty";
        auto const expected = msgpack::unpack(expectedBytes.data(), expectedBytes.size());
        ASSERT_EQ(expected.get().type, msgpack::type::MAP) << name;
        auto const wanted = keyed(expected.get(), problem);
        ASSERT_TRUE(problem.empty()) << name << ": " << problem;

        for(auto const& [tree, key] :
            std::map<std::string, std::string>{{"problem", "problemPredicate"},
                                               {"task", "taskPredicate"},
                                               {"hardware", "hardwarePredicate"}})
        {
            auto const have = fields.find(key);
            auto const want = wanted.find(tree);
            ASSERT_TRUE(have != fields.end()) << name << ": the written solution has no " << key;
            ASSERT_TRUE(want != wanted.end()) << name << ": the fixture has no " << tree;
            auto diff = firstMismatch(tree, *want->second, *have->second);
            EXPECT_TRUE(diff.empty()) << name << " " << diff;
        }
    }

    // Every fixture in the data directory, by the name of its record file.
    std::vector<std::string> fixtureNames()
    {
        std::string const        suffix = ".record.txt";
        std::vector<std::string> names;
        for(auto const& path : TestData::Instance().glob("solution-state/record/*" + suffix))
        {
            auto file = path.filename().string();
            names.push_back(file.substr(0, file.size() - suffix.size()));
        }
        std::sort(names.begin(), names.end());
        return names;
    }

    std::vector<std::string> fileLines(std::string const& path)
    {
        std::vector<std::string> lines;
        auto                     in = openFixture(path, std::ios::in);
        std::string              line;
        while(std::getline(in, line))
        {
            if(!line.empty())
                lines.push_back(line);
        }
        return lines;
    }

}

TEST(SolutionRecord, CatalogMatchesSolutionState)
{
    auto chips = supportedBuildChipIds();
    auto names = supportedDataTypeNames();
    EXPECT_EQ(chips, fileLines(fixturePath("catalog/catalog_chip_ids.txt")));
    EXPECT_EQ(names, fileLines(fixturePath("catalog/catalog_data_types.txt")));
}

TEST(SolutionRecord, FixturesAreComplete)
{
    auto const names = fixtureNames();
    ASSERT_FALSE(names.empty()) << "no fixture under " << fixturePath("record")
                                << " (test data directory "
                                << TestData::Instance().dataDir().string() << ")";
    for(auto const& name : names)
        for(auto const* extension :
            {".yaml", ".msgpack", ".record.txt", ".predicates.yaml", ".predicates.msgpack"})
        {
            auto const path = fixturePath("record/" + name + extension);
            EXPECT_TRUE(std::filesystem::is_regular_file(path)) << "missing fixture file " << path;
        }
}

TEST(SolutionRecord, MatchesThePythonBuilder)
{
    for(auto const& name : fixtureNames())
    {
        SCOPED_TRACE(name);
        expectRecord(name);
    }
}

TEST(SolutionRecord, PredicateValuesMatchThePythonBuilder)
{
    for(auto const& name : fixtureNames())
    {
        SCOPED_TRACE(name);
        expectPredicates(name);
    }
}
