// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// Builds TensileLite solution entries for the committed assembly under data/.
// The tests publish those entries into the JIT solution library. This is not a
// source-bundle writer.
namespace hipblaslt_jit_test
{
    namespace fs = std::filesystem;

    // What a main kernel's assembly records about itself: the kernel, the
    // target it was generated for, and the argument layout versions of its
    // custom.config block.
    struct Assembly
    {
        std::string kernel;
        std::string target; // the .amdgcn_target ID, e.g. "gfx950"
        int         kernArgsVersion           = -1;
        int         persistentLoopArgsVersion = 0; // written for DataParallel kernels only
    };

    inline Assembly readAssembly(const std::string& text)
    {
        // Whether line starts with prefix; rest is what follows it.
        const auto after = [](const std::string& line, const std::string& prefix, std::string& rest) {
            if(line.rfind(prefix, 0) != 0)
                return false;
            rest = line.substr(prefix.size());
            return true;
        };
        Assembly           result;
        std::istringstream lines(text);
        for(std::string line, rest; std::getline(lines, line);)
        {
            if(after(line, ".amdhsa_kernel ", rest) && result.kernel.empty())
                result.kernel = rest;
            else if(after(line, ".amdgcn_target \"amdgcn-amd-amdhsa--", rest))
                result.target = rest.substr(0, rest.find('"'));
            else if(after(line, "    KernArgsVersion: ", rest))
                result.kernArgsVersion = std::stoi(rest);
            else if(after(line, "    PersistentLoopArgsVersion: ", rest))
                result.persistentLoopArgsVersion = std::stoi(rest);
        }
        require(!result.kernel.empty() && !result.target.empty() && result.kernArgsVersion >= 0,
                "The assembly names no kernel, target or KernArgsVersion");
        return result;
    }

    struct Value
    {
        using Array = std::vector<Value>;
        using Map   = std::vector<std::pair<std::string, Value>>; // in order
        std::variant<bool, int64_t, std::string, Array, Map> data;
        Value(bool value) : data(value) {}
        Value(int value) : data(int64_t(value)) {}
        Value(const char* value) : data(std::string(value)) {}
        Value(std::string value) : data(std::move(value)) {}
        Value(Array value) : data(std::move(value)) {}
        Value(Map value) : data(std::move(value)) {}
    };
    using Array = Value::Array;
    using Map   = Value::Map;

    inline void encode(const Value& value, std::vector<uint8_t>& out)
    {
        const auto put = [&](uint8_t marker, uint64_t n, int bytes) {
            out.push_back(marker);
            for(int i = bytes - 1; i >= 0; --i)
                out.push_back(uint8_t(n >> (8 * i)));
        };
        // The fixed form below limit, otherwise the 32-bit form.
        const auto head = [&](uint8_t fixed, uint64_t limit, uint8_t wide, uint64_t n) {
            if(n < limit)
                out.push_back(uint8_t(fixed | n));
            else
                put(wide, n, 4);
        };
        if(const auto* b = std::get_if<bool>(&value.data))
            out.push_back(*b ? 0xc3 : 0xc2);
        else if(const auto* i = std::get_if<int64_t>(&value.data))
        {
            if(*i >= 0 && *i < 128)
                out.push_back(uint8_t(*i));
            else
                put(*i < 0 ? 0xd3 : 0xcf, *i, 8); // int64 or uint64
        }
        else if(const auto* s = std::get_if<std::string>(&value.data))
        {
            head(0xa0, 32, 0xdb, s->size());
            out.insert(out.end(), s->begin(), s->end());
        }
        else if(const auto* a = std::get_if<Array>(&value.data))
        {
            head(0x90, 16, 0xdd, a->size());
            for(const auto& item : *a)
                encode(item, out);
        }
        else
        {
            const auto& map = std::get<Map>(value.data);
            head(0x80, 16, 0xdf, map.size());
            for(const auto& [key, item] : map)
            {
                encode(key, out);
                encode(item, out);
            }
        }
    }

    // What TensileLite chose for a solution's kernel and the assembly does not
    // record, and where the assembly came from. The defaults are the gfx950
    // plain kernel's: FP16 A, B, C, D and E with FP32 compute, one wave64 per
    // 16x16 MFMA tile, GlobalSplitU=1 with the MultipleBufferSingleKernel
    // reduction, and no persistent tile processing.
    struct Description
    {
        bool        transA = false, transB = false, highPrecisionAccumulate = true;
        std::string type = "Half", computeType = "Float", tileProcessingStrategy = "None";

        std::array<int, 3> workGroup{16, 4, 1}, macroTile{16, 16, 1};
        std::array<int, 4> matrixInstruction{16, 16, 16, 1};
        int depthU = 16, globalReadVectorWidth = 4, storeVectorWidth = 4, magicDivAlg = 2;
        int globalSplitU = 1, globalSplitUPGR = 16, globalAccumulation = 3;
        int workspaceSizePerElemC = 4, synchronizerSizePerWG = 4;
        int staggerU = 32, staggerStrideShift = 3, workGroupMapping = 8;
        int occupancy = 5, mathClocksUnrolledLoop = 288, prefetchGlobalRead = 0;

        // The TensileLite revision that generated the assembly, and the SHA-256
        // of the bundle's recipe, DATA/<name>.yaml.
        std::string sourceRevision = "9118578c6b99df2f39ac474ef42807ff518685ef";
        std::string configSha256   = "4bba1b64f758a4d69c2dfe21113edca0e6f1008b573c0b3c3d5cadc48ade0c41";
    };

    inline Value predicate(const char* type, Value value)
    {
        return Map{{"type", type}, {"value", std::move(value)}};
    }

    // One solution of a written entry.
    struct Solution
    {
        Description values;
        std::string kernel; // its .amdhsa_kernel; may be empty when the sources hold one kernel
        std::string nameSuffix; // appended to the kernel name to name the solution
        Array       predicates; // problem predicates beyond the problem type's
        size_t      publishK = 512; // exact K published into the JIT solution library
    };

    // A written bundle: the committed sources it copies and its solutions, best first.
    struct Bundle
    {
        std::string           sources; // DATA/<target>/<sources>
        std::vector<Solution> solutions;
    };

    // The plain kernel's description on an architecture: gfx950's, except for
    // TensileLite's cycle estimate of the unrolled loop, which it does not
    // model for gfx90a.
    inline Description plainOn(int mathClocksUnrolledLoop)
    {
        Description values;
        values.mathClocksUnrolledLoop = mathClocksUnrolledLoop;
        return values;
    }

    // The plain kernel's description on each architecture that has its sources.
    inline const std::map<std::string, Description> plain{
        {"gfx90a", plainOn(0)}, {"gfx942", plainOn(256)}, {"gfx950", plainOn(288)}};

    // Every architecture's plain bundle, and plain-pair: the plain kernel as two
    // solutions, the first for K a multiple of 512 with WorkGroupMapping 8, the
    // second for any K with WorkGroupMapping 1.
    inline std::map<std::pair<std::string, std::string>, Bundle> describe()
    {
        std::map<std::pair<std::string, std::string>, Bundle> bundles;
        for(const auto& [target, values] : plain)
        {
            bundles[{target, "plain"}] = {"plain", {{values}}};
            auto multiple = values, any = values;
            multiple.workGroupMapping = 8;
            any.workGroupMapping      = 1;
            const Value kMultiple
                = Map{{"type", "BoundSizeMultiple"}, {"index", 0}, {"value", 512}};
            auto& pair = bundles[{target, "plain-pair"}]
                = {"plain", {{multiple, "", "_K512_WGM8", {kMultiple}}, {any, "", "_WGM1", {}}}};
            pair.solutions[0].publishK = 1024;
            pair.solutions[1].publishK = 256;
        }
        return bundles;
    }

    template <size_t N>
    inline Value array(const std::array<int, N>& values)
    {
        return Array(values.begin(), values.end());
    }

    // The entry's solution at index, and the problem predicate of its row.
    inline std::pair<Value, Value> solutionEntry(const Solution& s, const Assembly& kernel, int index)
    {
        const auto& d         = s.values;
        const auto  operation = std::string("Contraction_l_") + (d.transA ? "Alik" : "Ailk")
                               + (d.transB ? "_Bjlk" : "_Bljk") + "_Cijk_Dijk";
        const auto& t = d.type;
        const Map   problemType{
            {"operationIdentifier", operation}, {"transA", d.transA}, {"transB", d.transB},
            {"aType", t}, {"bType", t}, {"cType", t}, {"dType", t}, {"eType", t},
            {"computeInputTypeA", t}, {"computeInputTypeB", t}, {"computeType", d.computeType},
            {"useBeta", true}, {"highPrecisionAccumulate", d.highPrecisionAccumulate},
            {"stridedBatched", true}};
        // The problem type's predicates. The size guards TensileLite adds for
        // the tile (buffer offset limits, minimum sizes, GSU minimum K) are left
        // out: the tests offer each bundle only the problems it was made for.
        Array problem{predicate("OperationIdentifierEqual", operation),
                      predicate("TypesEqual", Array{t, t, t, t, t, t}),
                      predicate("HighPrecisionAccumulate", d.highPrecisionAccumulate),
                      predicate("StridedBatched", true)};
        for(const auto& [type, off] : Map{{"GroupedGemm", false}, {"SupportDeviceUserArguments", false},
                                          {"UseBias", 0}, {"Activation", "None"}, {"UseGradient", false},
                                          {"UseE", false}, {"UseScaleAB", ""}, {"UseScaleCD", false},
                                          {"UseScaleAlphaVec", 0}, {"AmaxDCheck", false}, {"Sparse", 0}})
            problem.push_back(predicate(type.c_str(), off));
        problem.insert(problem.end(), s.predicates.begin(), s.predicates.end());
        const auto& wg = d.workGroup;
        const Map   sizeMapping{
            {"waveNum", wg[0] * wg[1] * wg[2] / 64}, {"workGroup", array(wg)},
            {"macroTile", array(d.macroTile)}, {"matrixInstruction", array(d.matrixInstruction)},
            {"threadTile", Array{1, 1}}, {"depthU", d.depthU},
            {"grvwA", d.globalReadVectorWidth}, {"grvwB", d.globalReadVectorWidth},
            {"gwvwC", d.storeVectorWidth}, {"gwvwD", d.storeVectorWidth},
            {"globalSplitU", d.globalSplitU}, {"globalSplitUPGR", d.globalSplitUPGR},
            {"globalAccumulation", d.globalAccumulation}, {"tileProcessingStrategy", d.tileProcessingStrategy},
            {"workspaceSizePerElemC", d.workspaceSizePerElemC}, {"workspaceSizePerElemBias", 0},
            {"synchronizerSizePerWG", d.synchronizerSizePerWG}, {"magicDivAlg", d.magicDivAlg},
            {"staggerU", d.staggerU}, {"staggerUMapping", 0}, {"staggerStrideShift", d.staggerStrideShift},
            {"workGroupMapping", d.workGroupMapping}, {"workGroupMappingXCC", 1},
            {"workGroupMappingXCCGroup", -1}, {"CUOccupancy", d.occupancy},
            {"MathClocksUnrolledLoop", d.mathClocksUnrolledLoop}, {"PrefetchGlobalRead", d.prefetchGlobalRead},
            // Required by the reader; the same for every bundle.
            {"sourceKernel", false}, {"globalSplitUCoalesced", false},
            {"globalSplitUWorkGroupMappingRoundRobin", false}, {"customMainLoopScheduling", 0},
            {"nonTemporalA", 0}, {"nonTemporalB", 0}, {"NonTemporalD", 0},
            {"WaveSeparateGlobalReadA", 0}, {"WaveSeparateGlobalReadB", 0},
            {"UnrollLoopSwapGlobalReadOrder", 0}, {"DirectToVgprA", false}, {"DirectToVgprB", false},
            {"DirectToLdsA", false}, {"DirectToLdsB", false}, {"NumLoadsCoalescedA", 1},
            {"NumLoadsCoalescedB", 1}, {"VectorWidthA", 1}, {"VectorWidthB", 1},
            {"LocalSplitU", 1}, {"WaveGroup", Array{1, 1}}};
        const auto processor = kernel.target.substr(0, kernel.target.find(':'));
        const Map  solution{
            {"name", kernel.kernel + s.nameSuffix}, {"kernelName", kernel.kernel}, {"index", index},
            {"debugKernel", false},
            {"hardwarePredicate", predicate("AMDGPU", predicate("Processor", processor))},
            {"problemPredicate", predicate("And", problem)},
            {"taskPredicate", predicate("And", Array{Map{{"type", "WorkspaceCheck"}},
                                                     Map{{"type", "LaunchLimits"}}})},
            {"sizeMapping", sizeMapping},
            {"internalArgsSupport",
             Map{{"version", kernel.kernArgsVersion},
                 {"persistentLoopArgsVersion", kernel.persistentLoopArgsVersion},
                 {"gsu", true}, {"wgm", true}, {"staggerU", true}, {"useUniversalArgs", true},
                 {"useSFC", false}}},
            {"problemType", problemType}};
        return {solution, predicate("And", problem)};
    }

    // The entry of bundle, whose solution i runs kernels[i]: one Problem row per
    // solution, in order.
    inline std::vector<uint8_t> libraryEntry(const Bundle& bundle, const std::vector<Assembly>& kernels)
    {
        Array solutions, rows;
        for(size_t i = 0; i < kernels.size(); ++i)
        {
            auto [solution, problem] = solutionEntry(bundle.solutions[i], kernels[i], int(i));
            solutions.push_back(std::move(solution));
            rows.push_back(Map{{"predicate", std::move(problem)},
                               {"library", Map{{"type", "Single"}, {"index", int(i)}}}});
        }
        std::vector<uint8_t> bytes;
        encode(Map{{"solutions", std::move(solutions)},
                   {"library", Map{{"type", "Problem"}, {"rows", std::move(rows)}}}},
               bytes);
        return bytes;
    }


    // One solution published for an exact K, and the name recorded in the entry.
    struct Publication
    {
        int         local = 0;
        size_t      k     = 0;
        std::string kernel;
        std::string solutionName;
    };

    // A described solution set: its entry, the assembly units, and the K values
    // at which the JIT solution library should hold each solution.
    struct Prepared
    {
        std::string                         arch;
        std::string                         name;
        std::string                         compilerTarget;
        hipblaslt_jit::GeneratedSolution    solution;
        std::vector<Publication>            publications;
    };

    inline Prepared prepare(const fs::path&    sources,
                            const Bundle&      bundle,
                            const std::string& target,
                            const std::string& name)
    {
        std::map<std::string, Assembly> assembly;
        std::vector<hipblaslt_jit::BuildUnit> units;
        for(const auto& file : fs::directory_iterator(sources / "sources"))
        {
            if(file.path().extension() != ".s")
                continue;
            const auto text   = readFile(file.path());
            const auto kernel = readAssembly(text);
            require(kernel.target.substr(0, kernel.target.find(':')) == target,
                    file.path().filename().u8string() + " is not " + target + " assembly");
            assembly.emplace(kernel.kernel, kernel);
            units.push_back({file.path().filename().u8string(),
                             std::vector<uint8_t>(text.begin(), text.end()),
                             hipblaslt_jit::BuildUnit::Kind::Assembly,
                             {}});
        }
        std::sort(units.begin(), units.end(), [](const auto& a, const auto& b) {
            return a.name < b.name;
        });
        std::vector<Assembly> kernels;
        for(const auto& solution : bundle.solutions)
        {
            const auto found = solution.kernel.empty() && assembly.size() == 1
                                   ? assembly.begin()
                                   : assembly.find(solution.kernel);
            require(found != assembly.end(),
                    "No assembly of kernel '" + solution.kernel + "' in " + sources.u8string());
            kernels.push_back(found->second);
        }

        Prepared prepared;
        prepared.arch            = target;
        prepared.name            = name;
        prepared.compilerTarget  = kernels.front().target;
        prepared.solution.entry  = libraryEntry(bundle, kernels);
        for(const auto& kernel : kernels)
            if(std::find(prepared.solution.kernelNames.begin(),
                         prepared.solution.kernelNames.end(),
                         kernel.kernel)
               == prepared.solution.kernelNames.end())
                prepared.solution.kernelNames.push_back(kernel.kernel);
        prepared.solution.units = std::move(units);
        for(size_t i = 0; i < bundle.solutions.size(); ++i)
        {
            Publication publication;
            publication.local        = static_cast<int>(i);
            publication.k            = bundle.solutions[i].publishK;
            publication.kernel       = kernels[i].kernel;
            publication.solutionName = kernels[i].kernel + bundle.solutions[i].nameSuffix;
            prepared.publications.push_back(std::move(publication));
        }
        return prepared;
    }

    inline std::vector<Prepared> prepareSolutions(const fs::path& data)
    {
        const auto bundles = describe();
        for(const auto& target : fs::directory_iterator(data))
        {
            if(!target.is_directory())
                continue;
            const auto name = target.path().filename().u8string();
            for(const auto& sources : fs::directory_iterator(target))
            {
                const auto used = sources.path().filename().u8string();
                require(std::any_of(bundles.begin(),
                                    bundles.end(),
                                    [&](const auto& bundle) {
                                        return bundle.first.first == name
                                               && bundle.second.sources == used;
                                    }),
                        "No description of the committed sources " + name + '/' + used);
            }
        }
        std::vector<Prepared> prepared;
        for(const auto& [key, bundle] : bundles)
        {
            const auto& [target, name] = key;
            prepared.push_back(prepare(data / fs::u8path(target) / fs::u8path(bundle.sources),
                                       bundle,
                                       target,
                                       name));
        }
        return prepared;
    }
}
