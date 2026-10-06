// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// Writes the compiled-in resources of the JIT HipKittens backend.
//
// The two gfx950 TN variants below are the description of the one-solution
// library entries. Each entry is the predicates the backend matches and the
// custom-kernel launch record: K > 0, K a multiple of 128, M and N multiples of
// the 256 tile, and buffer-limit checks that span the whole matrix because the
// HIP kernel uses one buffer descriptor. The HIP template is embedded as
// written. The variant yaml files are the human-written description of the same
// kernels; this tool does not read them and does not call TensileLite.
//
// The MessagePack encoder is the one in clients/tests/jit/bundle_writer.cpp.
//
// usage: hipblaslt-hipkittens-write-entries --headers MANIFEST --source HIP --output CPP
namespace
{
    // Every column, so a buffer-limit check accepts an offset anywhere in the matrix.
    constexpr int64_t kAllColumns = 4294967295LL;

    struct Variant
    {
        const char*              name;
        const char*              type; // Tensile data type shared by A, B, C, D and E
        const char*              kernel;
        std::vector<const char*> flags;
        uint32_t                 vgprs;
    };

    // BF16, then FP16. Both compile gemm_tn_256x256x64_gfx950.hip; HK_FP16 selects
    // the FP16 kernel name in that template. The resource counts are what the built
    // kernel uses.
    std::vector<Variant> variants()
    {
        return {{"gemm_bf16_tn_256x256x64_gfx950",
                 "BFloat16",
                 "HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5",
                 {"-std=c++20", "-DKITTENS_CDNA4", "-w"},
                 237},
                {"gemm_f16_tn_256x256x64_gfx950",
                 "Half",
                 "HK_gemm_f16_TN_MT256x256x64_W2x4_gfx950_abi5",
                 {"-std=c++20", "-DKITTENS_CDNA4", "-DHK_FP16", "-w"},
                 238}};
    }

    struct Value
    {
        using Array = std::vector<Value>;
        using Map   = std::vector<std::pair<std::string, Value>>; // in order
        std::variant<bool, int64_t, std::string, Array, Map> data;
        Value(bool value)
            : data(value)
        {
        }
        Value(int value)
            : data(int64_t(value))
        {
        }
        Value(int64_t value)
            : data(value)
        {
        }
        Value(const char* value)
            : data(std::string(value))
        {
        }
        Value(std::string value)
            : data(std::move(value))
        {
        }
        Value(Array value)
            : data(std::move(value))
        {
        }
        Value(Map value)
            : data(std::move(value))
        {
        }
    };
    using Array = Value::Array;
    using Map   = Value::Map;

    void encode(const Value& value, std::vector<uint8_t>& out)
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

    Value predicate(const char* type, Value value)
    {
        return Map{{"type", type}, {"value", std::move(value)}};
    }

    Value predicateAt(const char* type, int index, int64_t value)
    {
        return Map{{"type", type}, {"index", index}, {"value", value}};
    }

    // The problem predicate of one variant, and of its only library row.
    Array problemPredicates(const char* type)
    {
        const Array same{type, type, type, type, type, type};
        return {predicateAt("BoundSizeMultiple", -1, 128),
                predicateAt("Free0SizeMultiple", 0, 256),
                predicateAt("Free1SizeMultiple", 0, 256),
                predicate("OperationIdentifierEqual", "Contraction_l_Alik_Bljk_Cijk_Dijk"),
                predicate("BiasDataTypeWhiteList", Array{"Float", type}),
                predicate("GateResidualDataTypeWhiteList", Array{}),
                predicate("BiasSrcWhiteList", Array{3}),
                predicate("AmaxDCheck", false),
                predicate("TypesEqual", same),
                predicate("HighPrecisionAccumulate", true),
                predicate("Activation", "None"),
                predicate("ActivationComputeType", "Float"),
                predicate("ActivationNoGuard", false),
                predicate("UseGradient", false),
                predicate("UseBias", 0),
                predicate("UseE", false),
                predicate("UseGateResidual", false),
                predicate("DataTypeE", type),
                predicate("StridedBatched", true),
                predicate("GroupedGemm", false),
                predicate("UseScaleAB", ""),
                predicate("UseScaleCD", false),
                predicate("UseScaleAlphaVec", 0),
                predicate("Sparse", 0),
                predicate("F32XdlMathOp", "Float"),
                predicate("SupportDeviceUserArguments", true),
                predicate("SwizzleTensorA", false),
                predicate("SwizzleTensorB", false),
                predicate("FusedGemmA2A", false),
                predicate("MXBlockA", 0),
                predicate("MXBlockB", 0),
                predicate("LeadingFree0SizesGreaterOrEqual", 1),
                predicate("LeadingFree1SizesGreaterOrEqual", 1),
                predicate("KernelLanguageCompatible", "Assembly"),
                predicate("BufferLoadOffsetLimitCheck",
                          Map{{"ShiftPtrElemA", 0},
                              {"ShiftPtrElemB", 0},
                              {"DUorMT0", kAllColumns},
                              {"DUorMT1", kAllColumns}}),
                predicate("BufferLoadOffsetLimitCheck_Beta", kAllColumns),
                predicate("BufferStoreOffsetLimitCheck", kAllColumns),
                predicate("GlobalSplitUCheckMinK", Array{32, 1}),
                predicate("WorkgroupMappingXCCCheck", Array{1, -1}),
                predicateAt("SizeGreaterThan", 3, 0)};
    }

    Value problemType(const char* type)
    {
        return Map{{"operationIdentifier", "Contraction_l_Alik_Bljk_Cijk_Dijk"},
                   {"transA", true},
                   {"transB", false},
                   {"computeInputTypeA", type},
                   {"computeInputTypeB", type},
                   {"aType", type},
                   {"bType", type},
                   {"cType", type},
                   {"dType", type},
                   {"eType", type},
                   {"computeType", "Float"},
                   {"useBeta", true},
                   {"useBias", 0},
                   {"biasSrcWhiteList", Array{3}},
                   {"useE", false},
                   {"useGateResidual", false},
                   {"gateResidualDataTypeWhiteList", Array{}},
                   {"useScaleAB", ""},
                   {"useScaleCD", false},
                   {"useScaleAlphaVec", 0},
                   {"biasDataTypeWhiteList", Array{"Float", type}},
                   {"highPrecisionAccumulate", true},
                   {"useInitialStridesAB", false},
                   {"useInitialStridesCD", false},
                   {"stridedBatched", true},
                   {"groupedGemm", false},
                   {"useGradient", false},
                   {"activationType", "None"},
                   {"activationArgLength", 0},
                   {"activationComputeDataType", "Float"},
                   {"activationNoGuard", false},
                   {"sparse", 0},
                   {"f32XdlMathOp", "Float"},
                   {"supportDeviceUserArguments", true},
                   {"outputAmaxD", false},
                   {"swizzleTensorA", false},
                   {"swizzleTensorB", false},
                   {"metadataLayout", 0},
                   {"mxBlockA", 0},
                   {"mxBlockB", 0},
                   {"mxTypeA", "E8"},
                   {"mxTypeB", "E8"},
                   {"mxScaleFormat", 0},
                   {"fusedGemmA2A", false}};
    }

    // What the custom kernel launches with. threads and macrotile are the HIP
    // kernel's; sizeMapping is the rest of the record the library reader requires.
    Value customKernel(const char* kernel)
    {
        const auto arg = [](const char* type, const char* semantic) {
            return Map{{"type", type}, {"semantic", semantic}};
        };
        return Map{{"name", kernel},
                   {"args",
                    Array{arg("address", "AddressA"),
                          arg("address", "AddressB"),
                          arg("address", "AddressC"),
                          arg("address", "AddressD"),
                          arg("uint32", "SizeFree0"),
                          arg("uint32", "SizeFree1"),
                          arg("uint32", "SizeSum"),
                          arg("float32", "Alpha"),
                          arg("float32", "Beta"),
                          arg("uint32", "StrideA0"),
                          arg("uint32", "StrideB0"),
                          arg("uint32", "StrideC0"),
                          arg("uint32", "StrideD0"),
                          arg("uint32", "StrideA1"),
                          arg("uint32", "StrideB1"),
                          arg("uint32", "StrideC1"),
                          arg("uint32", "StrideD1")}},
                   {"macrotile", Array{256, 256, 64}},
                   {"threads", Array{512, 1, 1}},
                   {"grid", Array{"TilesXY", "One", "Batch"}},
                   {"workspaceType", "None"},
                   {"workspaceSizePerElemC", 0},
                   {"workspaceSizePerElemBias", 0},
                   {"generated", false}};
    }

    Value sizeMapping()
    {
        return Map{{"waveNum", 8},
                   {"workGroup", Array{16, 16, 1}},
                   {"macroTile", Array{256, 256, 1}},
                   {"threadTile", Array{4, 4}},
                   {"matrixInstruction", Array{16, 16, 32, 1}},
                   {"grvwA", 1},
                   {"grvwB", 1},
                   {"gwvwC", 1},
                   {"gwvwD", 1},
                   {"depthU", 64},
                   {"staggerU", 0},
                   {"staggerUMapping", 0},
                   {"globalSplitUPGR", 16},
                   {"globalSplitU", 1},
                   {"staggerStrideShift", 0},
                   {"workGroupMapping", 8},
                   {"packBatchDims", 0},
                   {"magicDivAlg", 2},
                   {"tileProcessingStrategy", "None"},
                   {"workAssignment", "StaticGrid"},
                   {"streamKAtomic", 0},
                   {"prefetchAcrossPersistent", 0},
                   {"sourceKernel", false},
                   {"globalAccumulation", 2},
                   {"adaptiveGemmGSUA", 0},
                   {"workspaceSizePerElemC", 0},
                   {"workspaceSizePerElemBias", 0},
                   {"activationFused", true},
                   {"workGroupMappingXCC", 1},
                   {"workGroupMappingXCCGroup", -1},
                   {"globalSplitUCoalesced", false},
                   {"globalSplitUWorkGroupMappingRoundRobin", false},
                   {"CUOccupancy", -1},
                   {"PrefetchGlobalRead", 1},
                   {"MathClocksUnrolledLoop", 0},
                   {"synchronizerSizePerWG", 32768},
                   {"nonTemporalA", 0},
                   {"nonTemporalB", 0},
                   {"temporalHintA", 0},
                   {"temporalHintB", 0},
                   {"hasTemporalHint", false},
                   {"adaptiveGemmNTAB", 0},
                   {"customMainLoopScheduling", -1},
                   {"useSubtileImpl", false},
                   {"NonTemporalD", 0},
                   {"WaveSeparateGlobalReadA", 0},
                   {"WaveSeparateGlobalReadB", 0},
                   {"UnrollLoopSwapGlobalReadOrder", 0},
                   {"DirectToVgprA", false},
                   {"DirectToVgprB", false},
                   {"NumLoadsCoalescedA", 1},
                   {"NumLoadsCoalescedB", 1},
                   {"WaveGroup", Array{2, 4}},
                   {"VectorWidthA", -1},
                   {"VectorWidthB", -1},
                   {"LocalSplitU", 1},
                   {"DirectToLdsA", false},
                   {"DirectToLdsB", false},
                   {"ExpertSchedulingMode", 0},
                   {"clusterDim", Array{1, 1}}};
    }

    std::vector<uint8_t> libraryEntry(const Variant& variant)
    {
        const Array predicates = problemPredicates(variant.type);
        const Value problem    = predicate("And", predicates);
        const Map   solution{{"name", variant.kernel},
                           {"kernelName", variant.kernel},
                           {"problemType", problemType(variant.type)},
                           {"hardwarePredicate",
                            Map{{"type", "AMDGPU"},
                                {"value", Map{{"type", "Processor"}, {"value", "gfx950"}}}}},
                           {"problemPredicate", problem},
                           {"taskPredicate", Map{{"type", "LaunchLimits"}}},
                           {"sizeMapping", sizeMapping()},
                           {"customKernel", customKernel(variant.kernel)},
                           {"internalArgsSupport",
                            Map{{"version", 0},
                                {"persistentLoopArgsVersion", 0},
                                {"gsu", true},
                                {"wgm", true},
                                {"staggerU", true},
                                {"perTileExtraIters", false},
                                {"useUniversalArgs", true},
                                {"useSFC", false}}},
                           {"debugKernel", false},
                           {"libraryLogicIndex", -1},
                           {"index", 0},
                           {"ideals", Map{}},
                           {"linearModel", Map{}}};
        const Map entry{
            {"solutions", Array{solution}},
            {"library",
             Map{{"type", "Problem"},
                 {"rows",
                  Array{Map{{"predicate", problem},
                            {"library", Map{{"type", "Single"}, {"index", 0}}}}}}}}};
        std::vector<uint8_t> bytes;
        encode(entry, bytes);
        return bytes;
    }

    struct HeaderFile
    {
        std::string path;
        uint64_t    size = 0;
        std::string sha256;
    };

    void skipSpace(const std::string& text, size_t& pos)
    {
        while(pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
            ++pos;
    }

    // The next JSON string. The manifest's paths and hashes have no escapes.
    std::string jsonString(const std::string& text, size_t& pos)
    {
        skipSpace(text, pos);
        if(pos >= text.size() || text[pos] != '"')
            throw std::runtime_error("manifest.json expected a string");
        const auto end = text.find('"', pos + 1);
        if(end == std::string::npos)
            throw std::runtime_error("manifest.json has an unterminated string");
        auto value = text.substr(pos + 1, end - pos - 1);
        pos        = end + 1;
        return value;
    }

    void jsonKey(const std::string& text, size_t& pos, const char* key)
    {
        const auto found = jsonString(text, pos);
        if(found != key)
            throw std::runtime_error(std::string("manifest.json expected ") + key);
        skipSpace(text, pos);
        if(pos >= text.size() || text[pos] != ':')
            throw std::runtime_error(std::string("manifest.json expected ") + key);
        ++pos;
    }

    std::vector<HeaderFile> headerFiles(const std::string& manifest)
    {
        const auto files = manifest.find("\"files\"");
        if(files == std::string::npos)
            throw std::runtime_error("manifest.json has no files");
        auto pos = manifest.find('[', files);
        if(pos == std::string::npos)
            throw std::runtime_error("manifest.json has no files");
        ++pos;
        std::vector<HeaderFile> headers;
        while(true)
        {
            skipSpace(manifest, pos);
            if(pos < manifest.size() && manifest[pos] == ']')
                break;
            if(pos >= manifest.size() || manifest[pos] != '{')
                throw std::runtime_error("manifest.json has a malformed files entry");
            ++pos;
            HeaderFile file;
            jsonKey(manifest, pos, "path");
            file.path = jsonString(manifest, pos);
            skipSpace(manifest, pos);
            if(pos >= manifest.size() || manifest[pos] != ',')
                throw std::runtime_error("manifest.json has a malformed files entry");
            ++pos;
            jsonKey(manifest, pos, "size");
            skipSpace(manifest, pos);
            file.size = std::stoull(manifest.substr(pos));
            pos       = manifest.find_first_not_of("0123456789", pos);
            skipSpace(manifest, pos);
            if(pos >= manifest.size() || manifest[pos] != ',')
                throw std::runtime_error("manifest.json has a malformed files entry");
            ++pos;
            jsonKey(manifest, pos, "sha256");
            file.sha256 = jsonString(manifest, pos);
            if(file.sha256.size() != 64)
                throw std::runtime_error("manifest sha256 is not 64 hex digits: " + file.path);
            skipSpace(manifest, pos);
            if(pos >= manifest.size() || manifest[pos] != '}')
                throw std::runtime_error("manifest.json has a malformed files entry");
            ++pos;
            headers.push_back(std::move(file));
            skipSpace(manifest, pos);
            if(pos < manifest.size() && manifest[pos] == ',')
                ++pos;
        }
        if(headers.empty())
            throw std::runtime_error("manifest.json lists no header files");
        return headers;
    }

    std::string readFile(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        if(!in)
            throw std::runtime_error("Cannot read " + path);
        std::ostringstream text;
        text << in.rdbuf();
        if(!in)
            throw std::runtime_error("Cannot read " + path);
        return text.str();
    }

    void writeBytes(std::ostream& out, const std::string& name, const uint8_t* data, size_t size)
    {
        out << "        const unsigned char " << name << "[] = {";
        for(size_t i = 0; i < size; ++i)
        {
            if(i % 24 == 0)
                out << "\n            ";
            else
                out << ' ';
            out << int(data[i]) << ',';
        }
        out << "};\n";
    }

    std::string cxxString(const std::string& text)
    {
        std::string out = "\"";
        for(unsigned char c : text)
        {
            if(c == '\\' || c == '"')
                out += {'\\', char(c)};
            else if(c == '\n')
                out += "\\n";
            else
                out += char(c);
        }
        out += '"';
        return out;
    }

    std::string resourcesSource(const std::string&              manifest,
                                const std::string&              source,
                                const std::vector<HeaderFile>&  headers,
                                const std::vector<Variant>&     described,
                                const std::vector<std::vector<uint8_t>>& entries)
    {
        std::ostringstream out;
        out << "// Generated by hipblaslt-hipkittens-write-entries; do not edit.\n"
            << "#include \"hipblaslt-jit-hipkittens.hpp\"\n\n"
            << "namespace hipblaslt_ext::experimental::jit::hipkittens::detail\n{\n"
            << "    namespace\n    {\n";
        writeBytes(out,
                   "manifest",
                   reinterpret_cast<const uint8_t*>(manifest.data()),
                   manifest.size());
        writeBytes(out, "source", reinterpret_cast<const uint8_t*>(source.data()), source.size());
        for(size_t i = 0; i < entries.size(); ++i)
            writeBytes(out, "entry" + std::to_string(i), entries[i].data(), entries[i].size());
        out << "    }\n\n"
            << "    const Resources& resources()\n    {\n"
            << "        static const Resources value{\n"
            << "            {reinterpret_cast<const char*>(manifest), sizeof(manifest)},\n"
            << "            {\n";
        for(const auto& header : headers)
            out << "                {" << cxxString(header.path) << ", " << header.size << ", "
                << cxxString(header.sha256) << "},\n";
        out << "            },\n            {\n";
        for(size_t i = 0; i < described.size(); ++i)
        {
            const auto& variant = described[i];
            out << "                {" << cxxString(variant.name) << ",\n"
                << "                 \"gfx950\",\n"
                << "                 " << cxxString(variant.kernel) << ",\n"
                << "                 {reinterpret_cast<const char*>(source), sizeof(source)},\n"
                << "                 {reinterpret_cast<const char*>(entry" << i << "), sizeof(entry"
                << i << ")},\n"
                << "                 {";
            for(size_t flag = 0; flag < variant.flags.size(); ++flag)
                out << (flag ? ", " : "") << cxxString(variant.flags[flag]);
            out << "},\n"
                << "                 {84, 160000, " << variant.vgprs << ", 0}},\n";
        }
        out << "            }};\n        return value;\n    }\n}\n";
        return out.str();
    }
}

int main(int argc, char** argv)
try
{
    std::string headers, source, output;
    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto        take = [&](const char* flag, std::string& destination) {
            if(arg != flag)
                return false;
            if(++i >= argc)
                throw std::runtime_error(std::string(flag) + " needs a path");
            destination = argv[i];
            return true;
        };
        if(!take("--headers", headers) && !take("--source", source) && !take("--output", output))
            throw std::runtime_error("Unknown argument " + arg);
    }
    if(headers.empty() || source.empty() || output.empty())
        throw std::runtime_error("Usage: hipblaslt-hipkittens-write-entries "
                                 "--headers MANIFEST --source HIP --output CPP");

    const auto manifest = readFile(headers);
    const auto hip      = readFile(source);
    if(manifest.empty() || hip.empty())
        throw std::runtime_error("The manifest or the HIP template is empty");
    const auto described = variants();
    std::vector<std::vector<uint8_t>> entries;
    for(const auto& variant : described)
        entries.push_back(libraryEntry(variant));
    const auto text = resourcesSource(manifest, hip, headerFiles(manifest), described, entries);
    std::ofstream out(output, std::ios::binary);
    if(!out)
        throw std::runtime_error("Cannot write " + output);
    out << text;
    if(!out)
        throw std::runtime_error("Cannot write " + output);
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << "hipblaslt-hipkittens-write-entries: " << error.what() << '\n';
    return 1;
}
