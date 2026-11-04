// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Per-case runner for forward SDPA cases: execute, reference, compare and time.
//
// Each input row becomes one forward SDPA graph at the row's own tensor sizes. The runner
// asks one pinned engine whether it accepts the graph. For a selected row it then fills
// the inputs, runs the engine, runs the GPU reference and (below a size limit) the CPU
// reference, compares the outputs and writes one JSON line with the times and the result.
// An engine refusal or a reference decline is a recorded outcome.
//
// With --export-graphs DIR --arch X the runner loads no plugin, creates no handle and runs
// nothing. It writes each row's graph JSON and a bundle metadata sidecar under DIR, and one
// JSON line per row with the export outcome.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

#include <unistd.h>

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/PluginPaths.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <nlohmann/json.hpp>

#include "CaseGraph.hpp"
#include "RunnerSourceHash.hpp"
#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/PlanWorkspace.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/bundle/OutputComparison.hpp"
#include "harness/bundle/RngDumpHandoff.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/input-init/FillInputs.hpp"
#include "harness/input-init/InputFillRecipes.hpp"

namespace
{

using sdpa_case_flow::Row;
using Clock = std::chrono::steady_clock;
using Bundle = hipdnn_test_sdk::utilities::GraphTensorBundle;
using TensorPtr = std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>;
namespace fbd = hipdnn_flatbuffers_sdk::data_objects;
namespace it = hipdnn_integration_tests;

// Texts the backend uses when the pinned engine declines the graph.
const char* const ENGINE_REFUSED_MARKERS[]
    = {"not in a valid range of engine IDs", "Engine id is not applicable to this graph"};

bool engineRefused(const std::string& message)
{
    for(const char* marker : ENGINE_REFUSED_MARKERS)
    {
        if(message.find(marker) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

constexpr int EXIT_USAGE = 1;
constexpr int EXIT_ARCH_MISMATCH = 2;
constexpr int EXIT_ENGINE_NOT_LOADED = 3;

struct Options
{
    std::string input;
    std::string output;
    std::string classValues;
    std::string engine = "ASM_SDPA_ENGINE";
    std::vector<std::string> pluginPaths;
    std::string expectArch;
    std::string progressFile;
    int64_t start = 0;
    int64_t limit = -1;
    int64_t selectPerTier = -1;
    int64_t selectAcceptedPerTier = 0;
    std::set<std::string> selectLevels;
    double maxMflop = std::numeric_limits<double>::infinity();
    double maxTensorMb = 4096.0;
    double cpuRefMaxMflop = 300.0;
    int warmup = 3;
    int iterations = 10;
    unsigned int seed = 20261028U;
    bool noGpuRef = false;
    bool append = false;
    sdpa_case_flow::DescaleForm descaleForm = sdpa_case_flow::DescaleForm::BY_VALUE;
    std::vector<std::pair<std::string, std::string>> meta;
    std::string exportDir;
    std::string exportArch;
    std::string sourceSha256;
};

void printUsage(std::ostream& os)
{
    os << "Usage: hipdnn_sdpa_case_flow_runner --input LEVELS.csv --output RESULTS.jsonl "
          "--class-values class-to-value.json [options]\n"
          "  --engine NAME                 engine to pin by name (default ASM_SDPA_ENGINE)\n"
          "  --plugin-path PATH            load only these engine plugins; repeat for more\n"
          "  --expect-arch GFX             stop when HIP reports another arch\n"
          "  --start N                     skip the first N rows\n"
          "  --limit N                     use at most N rows after --start\n"
          "  --select-per-tier N           execute N rows per shape tier, spread over the file\n"
          "  --select-level NAME           execute every row of this level; repeat for more\n"
          "  --select-accepted-per-tier N  also execute up to N engine-accepted rows per tier\n"
          "  --max-mflop X                 do not execute rows above X MFLOP of work\n"
          "  --max-tensor-mb X             do not execute rows whose tensors need more than X MB "
          "(default 4096)\n"
          "  --cpu-ref-max-mflop X         run the CPU reference only up to X MFLOP (default "
          "300)\n"
          "  --warmup N / --iterations N   engine benchmark warm-up and timed runs (3 / 10)\n"
          "  --seed N                      input fill seed for rows without a case_key (rows\n"
          "                                with one use a seed taken from the case_key)\n"
          "  --no-gpu-ref                  do not run the GPU reference\n"
          "  --descale-form FORM           by-value (default) or device\n"
          "  --progress FILE               write the index of the row in progress to FILE\n"
          "  --append                      append to the output file\n"
          "  --meta KEY=VALUE              copy KEY=VALUE into the run record; repeat for more\n"
          "\n"
          "Export mode (no plugin, no handle, no execution):\n"
          "  --export-graphs DIR           write DIR/<case_key>/<case_key>.json and .meta.json\n"
          "  --arch GFX                    base arch the graphs are built for (required)\n"
          "  --source-sha256 HEX           sha256 of the case CSV the input rows come from "
          "(required)\n";
}

[[noreturn]] void usageError(const std::string& message)
{
    std::cerr << "error: " << message << "\n";
    printUsage(std::cerr);
    std::exit(EXIT_USAGE);
}

int64_t parseInteger(const std::string& flag, const std::string& text, int64_t minimum)
{
    try
    {
        size_t used = 0;
        const auto value = std::stoll(text, &used);
        if(used == text.size() && value >= minimum)
        {
            return value;
        }
    }
    catch(const std::exception&)
    {
    }
    usageError(flag + " needs an integer of at least " + std::to_string(minimum));
}

double parseDouble(const std::string& flag, const std::string& text)
{
    try
    {
        size_t used = 0;
        const auto value = std::stod(text, &used);
        if(used == text.size() && value >= 0.0)
        {
            return value;
        }
    }
    catch(const std::exception&)
    {
    }
    usageError(flag + " needs a non-negative number");
}

Options parseArgs(int argc, char** argv)
{
    Options opts;
    const auto next = [&](int& i, const std::string& flag) -> std::string {
        if(i + 1 >= argc)
        {
            usageError(flag + " needs a value");
        }
        return argv[++i];
    };

    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if(arg == "--input")
        {
            opts.input = next(i, arg);
        }
        else if(arg == "--output")
        {
            opts.output = next(i, arg);
        }
        else if(arg == "--class-values")
        {
            opts.classValues = next(i, arg);
        }
        else if(arg == "--engine")
        {
            opts.engine = next(i, arg);
        }
        else if(arg == "--plugin-path")
        {
            opts.pluginPaths.push_back(next(i, arg));
        }
        else if(arg == "--expect-arch")
        {
            opts.expectArch = next(i, arg);
        }
        else if(arg == "--progress")
        {
            opts.progressFile = next(i, arg);
        }
        else if(arg == "--start")
        {
            opts.start = parseInteger(arg, next(i, arg), 0);
        }
        else if(arg == "--limit")
        {
            opts.limit = parseInteger(arg, next(i, arg), 1);
        }
        else if(arg == "--select-per-tier")
        {
            opts.selectPerTier = parseInteger(arg, next(i, arg), 0);
        }
        else if(arg == "--select-accepted-per-tier")
        {
            opts.selectAcceptedPerTier = parseInteger(arg, next(i, arg), 0);
        }
        else if(arg == "--select-level")
        {
            opts.selectLevels.insert(next(i, arg));
        }
        else if(arg == "--max-mflop")
        {
            opts.maxMflop = parseDouble(arg, next(i, arg));
        }
        else if(arg == "--max-tensor-mb")
        {
            opts.maxTensorMb = parseDouble(arg, next(i, arg));
        }
        else if(arg == "--cpu-ref-max-mflop")
        {
            opts.cpuRefMaxMflop = parseDouble(arg, next(i, arg));
        }
        else if(arg == "--warmup")
        {
            opts.warmup = static_cast<int>(parseInteger(arg, next(i, arg), 0));
        }
        else if(arg == "--iterations")
        {
            opts.iterations = static_cast<int>(parseInteger(arg, next(i, arg), 1));
        }
        else if(arg == "--seed")
        {
            opts.seed = static_cast<unsigned int>(parseInteger(arg, next(i, arg), 0));
        }
        else if(arg == "--no-gpu-ref")
        {
            opts.noGpuRef = true;
        }
        else if(arg == "--append")
        {
            opts.append = true;
        }
        else if(arg == "--descale-form")
        {
            const auto form = next(i, arg);
            if(form == "by-value")
            {
                opts.descaleForm = sdpa_case_flow::DescaleForm::BY_VALUE;
            }
            else if(form == "device")
            {
                opts.descaleForm = sdpa_case_flow::DescaleForm::DEVICE;
            }
            else
            {
                usageError("--descale-form needs by-value or device");
            }
        }
        else if(arg == "--meta")
        {
            const auto pair = next(i, arg);
            const auto eq = pair.find('=');
            if(eq == std::string::npos || eq == 0)
            {
                usageError("--meta needs KEY=VALUE");
            }
            opts.meta.emplace_back(pair.substr(0, eq), pair.substr(eq + 1));
        }
        else if(arg == "--export-graphs")
        {
            opts.exportDir = next(i, arg);
        }
        else if(arg == "--arch")
        {
            opts.exportArch = next(i, arg);
        }
        else if(arg == "--source-sha256")
        {
            opts.sourceSha256 = next(i, arg);
        }
        else if(arg == "--help" || arg == "-h")
        {
            printUsage(std::cout);
            std::exit(0);
        }
        else
        {
            usageError("unknown argument '" + arg + "'");
        }
    }

    if(opts.input.empty() || opts.output.empty() || opts.classValues.empty())
    {
        usageError("--input, --output and --class-values are required");
    }
    if(std::filesystem::exists(opts.output) && !opts.append)
    {
        usageError("output file exists; pass --append to add to it");
    }
    if(!opts.exportDir.empty() || !opts.exportArch.empty() || !opts.sourceSha256.empty())
    {
        if(opts.exportDir.empty() || opts.exportArch.empty() || opts.sourceSha256.empty())
        {
            usageError("--export-graphs, --arch and --source-sha256 go together");
        }
        const bool hex
            = opts.sourceSha256.size() == 64
              && opts.sourceSha256.find_first_not_of("0123456789abcdef") == std::string::npos;
        if(!hex)
        {
            usageError("--source-sha256 needs 64 lower-case hex digits");
        }
        if(!opts.pluginPaths.empty() || !opts.expectArch.empty())
        {
            usageError("export mode loads no plugin and queries no device; drop --plugin-path "
                       "and --expect-arch");
        }
    }
    return opts;
}

// ------------------------------------------------------------------ JSON output

std::string jsonString(const std::string& text)
{
    std::string out = "\"";
    for(const char ch : text)
    {
        const auto uch = static_cast<unsigned char>(ch);
        switch(ch)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if(uch < 0x20)
            {
                std::ostringstream os;
                os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                   << static_cast<unsigned int>(uch);
                out += os.str();
            }
            else
            {
                out += ch;
            }
            break;
        }
    }
    return out + "\"";
}

std::string jsonNumber(double value)
{
    if(!std::isfinite(value))
    {
        return jsonString(std::isnan(value) ? "nan" : (value > 0 ? "inf" : "-inf"));
    }
    std::ostringstream os;
    os << std::setprecision(6) << value;
    return os.str();
}

std::string jsonStringArray(const std::vector<std::string>& items)
{
    std::string out = "[";
    for(size_t i = 0; i < items.size(); ++i)
    {
        out += (i == 0 ? "" : ",") + jsonString(items[i]);
    }
    return out + "]";
}

class JsonLine
{
public:
    JsonLine& str(const std::string& key, const std::string& value)
    {
        return raw(key, jsonString(value));
    }
    JsonLine& num(const std::string& key, double value)
    {
        return raw(key, jsonNumber(value));
    }
    JsonLine& integer(const std::string& key, int64_t value)
    {
        return raw(key, std::to_string(value));
    }
    JsonLine& boolean(const std::string& key, bool value)
    {
        return raw(key, value ? "true" : "false");
    }
    JsonLine& null(const std::string& key)
    {
        return raw(key, "null");
    }
    JsonLine& raw(const std::string& key, const std::string& value)
    {
        _text += (_text.size() == 1 ? "" : ",") + jsonString(key) + ":" + value;
        return *this;
    }
    std::string done() const
    {
        return _text + "}";
    }

private:
    std::string _text = "{";
};

// ------------------------------------------------------------------ input

struct CaseInput
{
    Row row;
    std::map<std::string, std::string> extra;
    std::vector<std::string> overrides;
};

std::vector<std::string> split(const std::string& text, char separator)
{
    std::vector<std::string> parts;
    std::string part;
    std::stringstream stream(text);
    while(std::getline(stream, part, separator))
    {
        parts.push_back(part);
    }
    if(!text.empty() && text.back() == separator)
    {
        parts.emplace_back();
    }
    return parts;
}

std::string trim(const std::string& text)
{
    const auto first = text.find_first_not_of(" \t\r");
    if(first == std::string::npos)
    {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r");
    return text.substr(first, last - first + 1);
}

[[noreturn]] void inputError(const std::string& message)
{
    std::cerr << "error: " << message << "\n";
    std::exit(EXIT_USAGE);
}

std::string readFile(const std::string& path, const std::string& what)
{
    std::ifstream in(path, std::ios::binary);
    if(!in)
    {
        inputError("cannot open " + what + " '" + path + "'");
    }
    return {(std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()};
}

std::vector<CaseInput> readInput(const std::string& path)
{
    const std::string content = readFile(path, "input");
    if(content.find('"') != std::string::npos)
    {
        inputError("quoted CSV fields are not supported");
    }
    std::vector<std::string> lines = split(content, '\n');
    while(!lines.empty() && trim(lines.back()).empty())
    {
        lines.pop_back();
    }
    if(lines.empty())
    {
        inputError("input has no header line");
    }
    std::vector<std::string> header;
    for(const auto& column : split(lines[0], ','))
    {
        header.push_back(trim(column));
    }
    const std::set<std::string> headerSet(header.begin(), header.end());
    for(const auto& required :
        {"case_id", "case_key", "level", "shape_tier", "sizes", "work_mflop"})
    {
        if(headerSet.count(required) == 0)
        {
            inputError(std::string("input header has no ") + required + " column");
        }
    }
    for(const auto& axis : sdpa_case_flow::modelAxisIds())
    {
        if(headerSet.count(axis) == 0)
        {
            inputError("input header has no column for model axis '" + axis + "'");
        }
    }
    const std::set<std::string> axisSet(sdpa_case_flow::modelAxisIds().begin(),
                                        sdpa_case_flow::modelAxisIds().end());

    std::vector<CaseInput> cases;
    for(size_t lineNo = 1; lineNo < lines.size(); ++lineNo)
    {
        if(trim(lines[lineNo]).empty())
        {
            continue;
        }
        const auto fields = split(lines[lineNo], ',');
        if(fields.size() != header.size())
        {
            inputError("line " + std::to_string(lineNo + 1) + " has "
                       + std::to_string(fields.size()) + " fields; the header has "
                       + std::to_string(header.size()));
        }
        CaseInput input;
        for(size_t i = 0; i < fields.size(); ++i)
        {
            const auto field = trim(fields[i]);
            if(axisSet.count(header[i]) != 0)
            {
                input.row[header[i]] = field;
            }
            else if(header[i] == "overrides")
            {
                for(const auto& token : split(field, ';'))
                {
                    if(!trim(token).empty())
                    {
                        input.overrides.push_back(trim(token));
                    }
                }
            }
            else
            {
                input.extra[header[i]] = field;
            }
        }
        cases.push_back(std::move(input));
    }
    return cases;
}

// ------------------------------------------------------------------ device and helpers

struct DeviceInfo
{
    int deviceCount = 0;
    std::string arch = "none";
    std::string name;
    std::string hipError;
};

DeviceInfo queryDevice()
{
    DeviceInfo info;
    int count = 0;
    if(const auto status = hipGetDeviceCount(&count); status != hipSuccess)
    {
        info.hipError = std::string("hipGetDeviceCount: ") + hipGetErrorString(status);
        return info;
    }
    info.deviceCount = count;
    if(count <= 0)
    {
        return info;
    }
    int device = 0;
    hipDeviceProp_t props{};
    if(hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&props, device) != hipSuccess)
    {
        info.hipError = "hipGetDeviceProperties failed";
        return info;
    }
    info.arch = props.gcnArchName;
    info.name = props.name;
    return info;
}

std::string baseArch(const std::string& arch)
{
    return arch.substr(0, arch.find(':'));
}

std::string hostName()
{
    std::array<char, 256> buffer{};
    if(gethostname(buffer.data(), buffer.size() - 1) != 0)
    {
        return "unknown";
    }
    return buffer.data();
}

double microsSince(Clock::time_point start)
{
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

void syncDevice()
{
    static_cast<void>(hipDeviceSynchronize());
}

// The integration harness default for an SDPA output: the per-format SDPA tolerance.
float sdpaTolerance(fbd::DataType type)
{
    namespace tol = hipdnn_test_sdk::utilities::sdpa;
    switch(type)
    {
    case fbd::DataType::FLOAT:
    case fbd::DataType::DOUBLE:
        return tol::getToleranceFwd<float>();
    case fbd::DataType::HALF:
        return tol::getToleranceFwd<hipdnn_data_sdk::types::half>();
    case fbd::DataType::BFLOAT16:
        return tol::getToleranceFwd<hipdnn_data_sdk::types::bfloat16>();
    default:
        return 1e-3f;
    }
}

// Non-finite elements of one compared pair. The element diff skips every pair with a NaN
// or an infinity in it, so these counts are reported beside it.
struct NonFiniteCounts
{
    int64_t expectedNan = 0;
    int64_t expectedPosInf = 0;
    int64_t expectedNegInf = 0;
    int64_t actualNan = 0;
    int64_t actualPosInf = 0;
    int64_t actualNegInf = 0;
    // Both sides infinite with the same sign.
    int64_t sameSignInf = 0;
    // At least one side non-finite, and not both infinite with the same sign.
    int64_t disagreements = 0;
};

struct DiffStats
{
    bool available = false;
    double maxAbs = 0.0;
    int64_t mismatches = 0;
    int64_t elements = 0;
    NonFiniteCounts nonFinite;
};

template <typename T>
NonFiniteCounts countNonFinite(hipdnn_data_sdk::utilities::ITensor& expected,
                               hipdnn_data_sdk::utilities::ITensor& actual)
{
    NonFiniteCounts counts;
    const auto raggedLayout = expected.raggedIterationInfo();
    if(expected.dims() != actual.dims() || raggedLayout != actual.raggedIterationInfo())
    {
        return counts;
    }
    hipdnn_data_sdk::utilities::TensorView<T> expectedView(expected);
    hipdnn_data_sdk::utilities::TensorView<T> actualView(actual);
    std::array<std::atomic<int64_t>, 8> tally{};
    const auto classify = [](float value) {
        if(std::isnan(value))
        {
            return 0;
        }
        if(std::isinf(value))
        {
            return value > 0.0f ? 1 : 2;
        }
        return 3;
    };
    auto countOne = [&](const std::vector<int64_t>& indices) {
        if(raggedLayout.has_value() && raggedLayout->isOutOfBlock(indices))
        {
            return;
        }
        const auto e = static_cast<float>(expectedView.getHostValue(indices));
        const auto a = static_cast<float>(actualView.getHostValue(indices));
        if(raggedLayout.has_value() && std::isnan(e))
        {
            return;
        }
        const int ce = classify(e);
        const int ca = classify(a);
        if(ce == 3 && ca == 3)
        {
            return;
        }
        if(ce != 3)
        {
            tally[static_cast<size_t>(ce)].fetch_add(1, std::memory_order_relaxed);
        }
        if(ca != 3)
        {
            tally[static_cast<size_t>(3 + ca)].fetch_add(1, std::memory_order_relaxed);
        }
        if((ce == 1 || ce == 2) && ce == ca)
        {
            tally[6].fetch_add(1, std::memory_order_relaxed);
        }
        else
        {
            tally[7].fetch_add(1, std::memory_order_relaxed);
        }
    };
    auto parallel = hipdnn_test_sdk::detail::makeParallelTensorFunctor(countOne, expected.dims());
    parallel(std::thread::hardware_concurrency());
    counts.expectedNan = tally[0];
    counts.expectedPosInf = tally[1];
    counts.expectedNegInf = tally[2];
    counts.actualNan = tally[3];
    counts.actualPosInf = tally[4];
    counts.actualNegInf = tally[5];
    counts.sameSignInf = tally[6];
    counts.disagreements = tally[7];
    return counts;
}

DiffStats diffTensors(fbd::DataType type,
                      hipdnn_data_sdk::utilities::ITensor& expected,
                      hipdnn_data_sdk::utilities::ITensor& actual,
                      float tolerance)
{
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::types::half;
    using hipdnn_test_sdk::utilities::computeTensorDiff;
    DiffStats stats;
    hipdnn_test_sdk::utilities::TensorDiffSummary summary{};
    switch(type)
    {
    case fbd::DataType::FLOAT:
        summary = computeTensorDiff<float>(expected, actual, tolerance, tolerance, 1);
        stats.nonFinite = countNonFinite<float>(expected, actual);
        break;
    case fbd::DataType::HALF:
        summary = computeTensorDiff<half>(expected, actual, tolerance, tolerance, 1);
        stats.nonFinite = countNonFinite<half>(expected, actual);
        break;
    case fbd::DataType::BFLOAT16:
        summary = computeTensorDiff<bfloat16>(expected, actual, tolerance, tolerance, 1);
        stats.nonFinite = countNonFinite<bfloat16>(expected, actual);
        break;
    default:
        return stats;
    }
    stats.available = true;
    stats.maxAbs = summary.maxAbsDiff;
    stats.mismatches = static_cast<int64_t>(summary.mismatchCount);
    stats.elements = static_cast<int64_t>(summary.totalElements);
    return stats;
}

bool isFp8(fbd::DataType type)
{
    return type == fbd::DataType::FP8_E4M3 || type == fbd::DataType::FP8_E5M2
           || type == fbd::DataType::FP8_E4M3_FNUZ || type == fbd::DataType::FP8_E5M2_FNUZ;
}

// Statistics over every compared element pair, passing or not. For an fp8 format the
// distance in representable values (ULP) comes from the sign-magnitude order of the codes.
struct ElementStats
{
    bool available = false;
    bool fp8 = false;
    int64_t compared = 0;
    double maxAbs = 0.0;
    double maxRel = 0.0;
    int64_t nonFinitePositionMismatches = 0;
    int64_t maxUlp = 0;
    int64_t oneUlp = 0;
    int64_t overOneUlp = 0;
};

template <typename T>
ElementStats elementStatsOf(hipdnn_data_sdk::utilities::ITensor& expected,
                            hipdnn_data_sdk::utilities::ITensor& actual,
                            bool fp8)
{
    ElementStats stats;
    const auto raggedLayout = expected.raggedIterationInfo();
    if(expected.dims() != actual.dims() || raggedLayout != actual.raggedIterationInfo())
    {
        return stats;
    }
    hipdnn_data_sdk::utilities::TensorView<T> expectedView(expected);
    hipdnn_data_sdk::utilities::TensorView<T> actualView(actual);
    struct alignas(64) Shard
    {
        std::atomic<int64_t> compared{0};
        std::atomic<int64_t> nonFinitePositionMismatches{0};
        std::atomic<int64_t> maxUlp{0};
        std::atomic<int64_t> oneUlp{0};
        std::atomic<int64_t> overOneUlp{0};
        std::atomic<double> maxAbs{0.0};
        std::atomic<double> maxRel{0.0};
    };
    constexpr size_t SHARDS = 64;
    std::vector<Shard> shards(SHARDS);
    const auto raiseTo = [](auto& target, auto value) {
        auto current = target.load(std::memory_order_relaxed);
        while(value > current
              && !target.compare_exchange_weak(current, value, std::memory_order_relaxed))
        {
        }
    };
    const auto classify = [](float value) {
        if(std::isnan(value))
        {
            return 0;
        }
        if(std::isinf(value))
        {
            return value > 0.0f ? 1 : 2;
        }
        return 3;
    };
    const auto ordinal = [](const T& value) {
        uint8_t code = 0;
        std::memcpy(&code, &value, 1);
        const int64_t magnitude = code & 0x7F;
        return (code & 0x80) != 0 ? -magnitude : magnitude;
    };
    auto visit = [&](const std::vector<int64_t>& indices) {
        if(raggedLayout.has_value() && raggedLayout->isOutOfBlock(indices))
        {
            return;
        }
        const T expectedValue = expectedView.getHostValue(indices);
        const T actualValue = actualView.getHostValue(indices);
        const auto e = static_cast<float>(expectedValue);
        const auto a = static_cast<float>(actualValue);
        if(raggedLayout.has_value() && std::isnan(e))
        {
            return;
        }
        const int ce = classify(e);
        const int ca = classify(a);
        double absDiff = 0.0;
        double relDiff = 0.0;
        int64_t ulp = 0;
        const bool finite = ce == 3 && ca == 3;
        if(finite)
        {
            absDiff = std::fabs(static_cast<double>(a) - static_cast<double>(e));
            const double scale
                = std::max(std::fabs(static_cast<double>(a)), std::fabs(static_cast<double>(e)));
            relDiff = scale > 0.0 ? absDiff / scale : 0.0;
            if constexpr(sizeof(T) == 1)
            {
                ulp = std::llabs(ordinal(expectedValue) - ordinal(actualValue));
            }
        }
        auto& shard = shards[std::hash<std::thread::id>{}(std::this_thread::get_id()) % SHARDS];
        shard.compared.fetch_add(1, std::memory_order_relaxed);
        if(ce != ca)
        {
            shard.nonFinitePositionMismatches.fetch_add(1, std::memory_order_relaxed);
        }
        if(!finite)
        {
            return;
        }
        raiseTo(shard.maxAbs, absDiff);
        raiseTo(shard.maxRel, relDiff);
        if(fp8 && ulp > 0)
        {
            raiseTo(shard.maxUlp, ulp);
            (ulp == 1 ? shard.oneUlp : shard.overOneUlp).fetch_add(1, std::memory_order_relaxed);
        }
    };
    auto parallel = hipdnn_test_sdk::detail::makeParallelTensorFunctor(visit, expected.dims());
    parallel(std::thread::hardware_concurrency());
    for(const auto& shard : shards)
    {
        stats.compared += shard.compared.load();
        stats.nonFinitePositionMismatches += shard.nonFinitePositionMismatches.load();
        stats.maxUlp = std::max(stats.maxUlp, shard.maxUlp.load());
        stats.oneUlp += shard.oneUlp.load();
        stats.overOneUlp += shard.overOneUlp.load();
        stats.maxAbs = std::max(stats.maxAbs, shard.maxAbs.load());
        stats.maxRel = std::max(stats.maxRel, shard.maxRel.load());
    }
    stats.available = true;
    stats.fp8 = fp8;
    return stats;
}

ElementStats elementStats(fbd::DataType type,
                          hipdnn_data_sdk::utilities::ITensor& expected,
                          hipdnn_data_sdk::utilities::ITensor& actual)
{
    namespace t = hipdnn_data_sdk::types;
    switch(type)
    {
    case fbd::DataType::FLOAT:
        return elementStatsOf<float>(expected, actual, false);
    case fbd::DataType::HALF:
        return elementStatsOf<t::half>(expected, actual, false);
    case fbd::DataType::BFLOAT16:
        return elementStatsOf<t::bfloat16>(expected, actual, false);
    case fbd::DataType::FP8_E4M3:
        return elementStatsOf<t::fp8_e4m3>(expected, actual, true);
    case fbd::DataType::FP8_E5M2:
        return elementStatsOf<t::fp8_e5m2>(expected, actual, true);
    case fbd::DataType::FP8_E4M3_FNUZ:
        return elementStatsOf<t::fp8_e4m3_fnuz>(expected, actual, true);
    case fbd::DataType::FP8_E5M2_FNUZ:
        return elementStatsOf<t::fp8_e5m2_fnuz>(expected, actual, true);
    default:
        return {};
    }
}

// Share of 1-ULP elements in an fp8 output above which a reference-to-reference result is flagged.
constexpr double FP8_ONE_ULP_FLAG_SHARE = 0.01;

// Bytes one tensor spans in memory, from its dims and strides.
double tensorBytes(const hipdnn_frontend::graph::TensorAttributes& attr)
{
    const auto& dims = attr.get_dim();
    const auto& strides = attr.get_stride();
    double span = 1.0;
    for(size_t i = 0; i < dims.size() && i < strides.size(); ++i)
    {
        span += static_cast<double>(std::max<int64_t>(dims[i] - 1, 0))
                * static_cast<double>(std::llabs(strides[i]));
    }
    double elementBytes = 4.0;
    switch(attr.get_data_type())
    {
    case hipdnn_frontend::DataType::DOUBLE:
    case hipdnn_frontend::DataType::INT64:
        elementBytes = 8.0;
        break;
    case hipdnn_frontend::DataType::HALF:
    case hipdnn_frontend::DataType::BFLOAT16:
        elementBytes = 2.0;
        break;
    case hipdnn_frontend::DataType::INT8:
    case hipdnn_frontend::DataType::UINT8:
    case hipdnn_frontend::DataType::BOOLEAN:
    case hipdnn_frontend::DataType::FP8_E4M3:
    case hipdnn_frontend::DataType::FP8_E5M2:
    case hipdnn_frontend::DataType::FP8_E4M3_FNUZ:
    case hipdnn_frontend::DataType::FP8_E5M2_FNUZ:
        elementBytes = 1.0;
        break;
    default:
        break;
    }
    return span * elementBytes;
}

// ------------------------------------------------------------------ input contents

// Every non-virtual tensor of a graph, ragged offset tensors included, and the uids that a
// node writes.
struct GraphTensors
{
    std::map<int64_t, TensorPtr> attrs;
    std::set<int64_t> outputs;
};

GraphTensors collectGraphTensors(hipdnn_frontend::graph::Graph& graph)
{
    GraphTensors found;
    const auto add = [&](const TensorPtr& attr) {
        if(attr != nullptr && !attr->get_is_virtual())
        {
            found.attrs.emplace(attr->get_uid(), attr);
        }
    };
    graph.visit([&](const hipdnn_frontend::graph::INode& node) {
        for(const auto& attr : node.getNodeOutputTensorAttributes())
        {
            add(attr);
            if(attr != nullptr && !attr->get_is_virtual())
            {
                found.outputs.insert(attr->get_uid());
            }
        }
        for(const auto& attr : node.getNodeInputTensorAttributes())
        {
            add(attr);
        }
    });
    for(const auto& [uid, attr] : std::map<int64_t, TensorPtr>(found.attrs))
    {
        add(attr->get_ragged_offset());
    }
    return found;
}

// The tensors no node writes, in ascending uid order.
std::vector<int64_t> leafInputUids(const GraphTensors& tensors)
{
    std::vector<int64_t> uids;
    for(const auto& [uid, attr] : tensors.attrs)
    {
        if(tensors.outputs.count(uid) == 0)
        {
            uids.push_back(uid);
        }
    }
    return uids;
}

bool isCaseKey(const std::string& key)
{
    return key.size() == 16 && key.find_first_not_of("0123456789abcdef") == std::string::npos;
}

// The per-case input seed: the first 8 hex digits of the case_key, masked to 31 bits.
int64_t caseSeed(const std::string& caseKey)
{
    return static_cast<int64_t>(std::stoull(caseKey.substr(0, 8), nullptr, 16) & 0x7fffffffULL);
}

// Gives every leaf input its own seed, drawn in ascending uid order from a generator seeded
// with `seed`: the same draws fillInputs makes from that global seed. A tensor keeps its seed
// when its uid changes.
void setInputSeeds(it::InputFillRecipes& recipes,
                   const std::vector<int64_t>& sortedLeafUids,
                   unsigned int seed)
{
    recipes.setGlobalSeed(seed);
    std::mt19937 generator(seed);
    for(const auto uid : sortedLeafUids)
    {
        recipes.setSeed(uid, static_cast<unsigned int>(generator()));
    }
}

// The content that becomes a FIXED fill. Every other content becomes a VALUES fill.
constexpr const char* FIXED_CONTENT_NAME = "attn_scale";

it::FillRecipe contentRecipe(const std::string& name, const sdpa_case_flow::Contents& contents)
{
    if(const auto* reals = std::get_if<std::vector<double>>(&contents.values))
    {
        if(name == FIXED_CONTENT_NAME && reals->size() == 1)
        {
            return it::FillRecipe::fixed(static_cast<float>(reals->front()));
        }
        return it::FillRecipe::values(*reals, contents.repeat);
    }
    return it::FillRecipe::values(std::get<std::vector<int64_t>>(contents.values), contents.repeat);
}

struct ContentInput
{
    std::string name;
    int64_t uid = 0;
};

// Sets one recipe per BuiltCase content on the leaf input of that name. Throws
// std::runtime_error when a content names no leaf input, or more than one.
std::vector<ContentInput> setContentRecipes(it::InputFillRecipes& recipes,
                                            const sdpa_case_flow::BuiltCase& built,
                                            const GraphTensors& tensors)
{
    std::vector<ContentInput> placed;
    for(const auto& [name, contents] : built.contents)
    {
        std::vector<int64_t> uids;
        for(const auto& [uid, attr] : tensors.attrs)
        {
            if(attr->get_name() == name && tensors.outputs.count(uid) == 0)
            {
                uids.push_back(uid);
            }
        }
        if(uids.size() != 1)
        {
            throw std::runtime_error("contents for '" + name + "' match "
                                     + std::to_string(uids.size())
                                     + " input tensors of the graph, not 1");
        }
        recipes.set(uids.front(), contentRecipe(name, contents));
        placed.push_back({name, uids.front()});
    }
    return placed;
}

// A short description of a recipe for the result records.
std::string recipeNote(const it::FillRecipe& recipe)
{
    constexpr size_t MAX_LISTED = 64;
    if(recipe.kind == it::FillRecipe::Kind::FIXED)
    {
        return "fixed " + jsonNumber(static_cast<double>(recipe.value));
    }
    if(recipe.kind != it::FillRecipe::Kind::VALUES || recipe.list == nullptr)
    {
        return "free";
    }
    std::string text = "values [";
    const size_t count
        = std::visit([](const auto& numbers) { return numbers.size(); }, *recipe.list);
    for(size_t i = 0; i < count && i < MAX_LISTED; ++i)
    {
        text += i == 0 ? "" : ",";
        if(const auto* integers = std::get_if<std::vector<int64_t>>(recipe.list.get()))
        {
            text += std::to_string((*integers)[i]);
        }
        else
        {
            text += jsonNumber(std::get<std::vector<double>>(*recipe.list)[i]);
        }
    }
    if(count > MAX_LISTED)
    {
        text += ",... " + std::to_string(count) + " values";
    }
    return text + (recipe.repeat ? "] repeat" : "] exact");
}

// ------------------------------------------------------------------ one case

struct Context
{
    const Options& opts;
    const sdpa_case_flow::ClassValues& values;
    std::string arch;
    hipdnnHandle_t handle = nullptr;
    int64_t engineId = 0;
    it::gpu_graph_executor::GpuReferenceGraphExecutor gpuRef;
    it::CpuReferenceGraphExecutorAdapter cpuRef;
};

struct Comparison
{
    std::string against;
    bool ran = false;
    bool pass = false;
    std::string error;
    std::vector<std::string> mismatched;
    std::vector<std::string> reports;
    double tFirstUs = 0.0;
    double tWarmUs = 0.0;
    double tDiffUs = 0.0;
    std::vector<std::pair<std::string, DiffStats>> diffs;
    // Reference-to-reference rule: fp8 outputs within 1 ULP with non-finite values in the same
    // places, every other output at the suite tolerance.
    bool refRule = false;
    bool refAgree = false;
    std::vector<std::string> refFlags;
    std::vector<std::pair<std::string, ElementStats>> elements;

    std::string json() const
    {
        JsonLine line;
        line.str("against", against).boolean("ran", ran);
        if(!ran)
        {
            return line.str("error", error).done();
        }
        if(refRule)
        {
            line.boolean("ref_agree", refAgree).raw("ref_flags", jsonStringArray(refFlags));
            JsonLine elementJson;
            for(const auto& [label, stats] : elements)
            {
                JsonLine one;
                if(stats.available)
                {
                    one.boolean("fp8", stats.fp8)
                        .integer("compared", stats.compared)
                        .num("max_abs", stats.maxAbs)
                        .num("max_rel", stats.maxRel)
                        .integer("nonfinite_position_mismatches",
                                 stats.nonFinitePositionMismatches);
                    if(stats.fp8)
                    {
                        one.integer("ulp_max", stats.maxUlp)
                            .integer("ulp_one", stats.oneUlp)
                            .integer("ulp_over_one", stats.overOneUlp)
                            .num("ulp_one_share",
                                 stats.compared > 0 ? static_cast<double>(stats.oneUlp)
                                                          / static_cast<double>(stats.compared)
                                                    : 0.0);
                    }
                }
                else
                {
                    one.str("note", "no element statistics for this format");
                }
                elementJson.raw(label, one.done());
            }
            line.raw("elements", elementJson.done());
        }
        line.boolean("pass", pass)
            .raw("mismatched", jsonStringArray(mismatched))
            .raw("reports", jsonStringArray(reports))
            .num("t_compare_first_us", tFirstUs)
            .num("t_compare_warm_us", tWarmUs)
            .num("t_diff_host_us", tDiffUs)
            .str("error", error);
        JsonLine tensorDiffs;
        for(const auto& [label, stats] : diffs)
        {
            JsonLine one;
            if(stats.available)
            {
                const auto& nf = stats.nonFinite;
                one.num("max_abs", stats.maxAbs)
                    .integer("mismatches", stats.mismatches)
                    .integer("elements", stats.elements)
                    .integer("nonfinite_disagreements", nf.disagreements)
                    .integer("same_sign_inf", nf.sameSignInf)
                    .integer("expected_nan", nf.expectedNan)
                    .integer("expected_pos_inf", nf.expectedPosInf)
                    .integer("expected_neg_inf", nf.expectedNegInf)
                    .integer("actual_nan", nf.actualNan)
                    .integer("actual_pos_inf", nf.actualPosInf)
                    .integer("actual_neg_inf", nf.actualNegInf);
            }
            else
            {
                one.str("note", "no element diff for this format");
            }
            tensorDiffs.raw(label, one.done());
        }
        return line.raw("tensors", tensorDiffs.done()).done();
    }
};

// Uids of the SDPA forward stats (log-sum-exp) outputs. A row with no visible key has
// log-sum-exp -inf in both references by contract, so these are graded with matching
// infinities.
std::set<int64_t>
    sdpaStatsUids(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper)
{
    std::set<int64_t> uids;
    for(uint32_t i = 0; i < wrapper.nodeCount(); ++i)
    {
        const auto* sdpa = wrapper.getNode(i).attributes_as_SdpaAttributes();
        if(sdpa != nullptr && sdpa->stats_tensor_uid().has_value())
        {
            uids.insert(sdpa->stats_tensor_uid().value());
        }
    }
    return uids;
}

// The matching-infinities validator has no device implementation, so stats outputs are
// compared on the host whatever `site` is.
Comparison compareBundles(const std::string& against,
                          const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                          const std::vector<int64_t>& outputUids,
                          const std::map<int64_t, TensorPtr>& attrs,
                          it::bundle::OutputTensors& actual,
                          Bundle& expected,
                          it::ValidationSite site,
                          bool refRule = false)
{
    Comparison result;
    result.against = against;
    result.refRule = refRule;
    const auto statsUids = sdpaStatsUids(wrapper);
    std::set<std::string> statsLabels;
    std::vector<int64_t> statsOutputs;
    std::vector<int64_t> otherOutputs;
    for(const auto uid : outputUids)
    {
        if(statsUids.count(uid) != 0)
        {
            statsLabels.insert(it::bundle::tensorLabel(uid, attrs.at(uid)->get_name()));
            statsOutputs.push_back(uid);
        }
        else
        {
            otherOutputs.push_back(uid);
        }
    }
    const auto toleranceFor = [&statsLabels](const std::string& label, fbd::DataType type) {
        const float t = sdpaTolerance(type);
        return statsLabels.count(label) != 0
                   ? it::bundle::ComparisonTolerance::allCloseMatchingInfinities(t, t)
                   : it::bundle::ComparisonTolerance::allClose(t, t);
    };
    const auto expectedFor = [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& {
        return *expected.tensors.at(uid);
    };
    const auto compareAll = [&]() {
        auto mismatches = it::bundle::compareOutputs(
            wrapper, otherOutputs, actual, expectedFor, toleranceFor, site, against);
        auto statsMismatches = it::bundle::compareOutputs(wrapper,
                                                          statsOutputs,
                                                          actual,
                                                          expectedFor,
                                                          toleranceFor,
                                                          it::ValidationSite::HOST,
                                                          against);
        mismatches.insert(mismatches.end(),
                          std::make_move_iterator(statsMismatches.begin()),
                          std::make_move_iterator(statsMismatches.end()));
        return mismatches;
    };
    try
    {
        auto start = Clock::now();
        auto mismatches = compareAll();
        syncDevice();
        result.tFirstUs = microsSince(start);
        start = Clock::now();
        mismatches = compareAll();
        syncDevice();
        result.tWarmUs = microsSince(start);
        result.ran = true;
        result.pass = mismatches.empty();
        for(const auto& m : mismatches)
        {
            result.mismatched.push_back(m.label);
            result.reports.push_back(m.report.substr(0, 600));
        }

        start = Clock::now();
        for(const auto uid : outputUids)
        {
            const auto& attr = attrs.at(uid);
            const auto type
                = hipdnn_test_sdk::utilities::frontendToSdkDataType(attr->get_data_type());
            result.diffs.emplace_back(
                it::bundle::tensorLabel(uid, attr->get_name()),
                diffTensors(type, *expected.tensors.at(uid), *actual.at(uid), sdpaTolerance(type)));
        }
        result.tDiffUs = microsSince(start);

        if(refRule)
        {
            const std::set<std::string> suiteMismatched(result.mismatched.begin(),
                                                        result.mismatched.end());
            result.refAgree = true;
            for(const auto uid : outputUids)
            {
                const auto& attr = attrs.at(uid);
                const auto type
                    = hipdnn_test_sdk::utilities::frontendToSdkDataType(attr->get_data_type());
                const auto label = it::bundle::tensorLabel(uid, attr->get_name());
                auto stats = elementStats(type, *expected.tensors.at(uid), *actual.at(uid));
                if(isFp8(type))
                {
                    if(!stats.available || stats.overOneUlp != 0
                       || stats.nonFinitePositionMismatches != 0)
                    {
                        result.refAgree = false;
                    }
                    if(stats.compared > 0
                       && static_cast<double>(stats.oneUlp) / static_cast<double>(stats.compared)
                              > FP8_ONE_ULP_FLAG_SHARE)
                    {
                        result.refFlags.push_back(label + ":fp8-one-ulp-share-above-1pct");
                    }
                }
                else if(suiteMismatched.count(label) != 0)
                {
                    result.refAgree = false;
                }
                result.elements.emplace_back(label, stats);
            }
        }
    }
    catch(const std::exception& e)
    {
        result.ran = false;
        result.error = std::string("exception: ") + e.what();
    }
    return result;
}

Comparison skippedComparison(const std::string& against, const std::string& reason)
{
    Comparison result;
    result.against = against;
    result.error = reason;
    return result;
}

// True when every byte of the output still holds the sentinel fill it had before the
// engine ran.
bool stillSentinel(hipdnn_data_sdk::utilities::ITensor& output,
                   const hipdnn_frontend::graph::TensorAttributes& attr)
{
    auto sentinel = hipdnn_test_sdk::utilities::createTensorFromAttribute(attr, false);
    sentinel->fillWithSentinelValue();
    const auto bytes = output.elementSpace() * output.elementSize();
    if(bytes != sentinel->elementSpace() * sentinel->elementSize())
    {
        return false;
    }
    return std::memcmp(output.rawHostData(), sentinel->rawHostData(), bytes) == 0;
}

// A 0/1 keep mask drawn with keep probability 1 - p, for references that have no engine
// mask to share.
void writeSyntheticKeepMask(hipdnn_data_sdk::utilities::ITensor& dump,
                            float dropoutProbability,
                            unsigned int seed)
{
    if(dump.elementSize() != sizeof(float))
    {
        throw std::runtime_error("rng_dump must be FLOAT");
    }
    std::mt19937 generator(seed);
    std::bernoulli_distribution keep(1.0 - static_cast<double>(dropoutProbability));
    std::vector<float> mask(dump.elementSpace());
    for(auto& value : mask)
    {
        value = keep(generator) ? 1.0f : 0.0f;
    }
    dump.fillWithData(mask.data(), mask.size() * sizeof(float));
}

void markOutputsDeviceModified(Bundle& bundle, const std::vector<int64_t>& outputUids)
{
    for(const auto uid : outputUids)
    {
        bundle.tensors.at(uid)->markDeviceModified();
    }
}

void markOutputsHostModified(Bundle& bundle, const std::vector<int64_t>& outputUids)
{
    for(const auto uid : outputUids)
    {
        bundle.tensors.at(uid)->markHostModified();
    }
}

// A copy of `source` that shares its input tensors and owns fresh, sentinel-filled outputs.
Bundle withFreshOutputs(Bundle& source,
                        const std::map<int64_t, TensorPtr>& attrs,
                        const std::vector<int64_t>& outputUids)
{
    Bundle copy;
    for(const auto& [uid, attr] : attrs)
    {
        if(source.isOutput(uid))
        {
            copy.addTensor(*attr,
                           hipdnn_test_sdk::utilities::createTensorFromAttribute(*attr, false));
        }
        else if(source.tensors.count(uid) != 0)
        {
            copy.addTensor(*attr, source.tensors.at(uid));
        }
    }
    for(const auto uid : outputUids)
    {
        copy.outputTensorIds.insert(uid);
    }
    copy.sentinelFillOutputTensors();
    return copy;
}

std::string runCase(size_t index,
                    const CaseInput& input,
                    bool selectByRule,
                    bool acceptedCapLeft,
                    Context& ctx,
                    bool& acceptedOut)
{
    acceptedOut = false;
    const auto caseStart = Clock::now();
    const auto extra = [&](const std::string& key) {
        const auto found = input.extra.find(key);
        return found == input.extra.end() ? std::string() : found->second;
    };
    double workMflop = 0.0;
    try
    {
        workMflop = std::stod(extra("work_mflop"));
    }
    catch(const std::exception&)
    {
        workMflop = std::numeric_limits<double>::infinity();
    }

    JsonLine line;
    line.str("record", "case")
        .integer("index", static_cast<int64_t>(index))
        .str("case_id", extra("case_id"))
        .str("case_key", extra("case_key"))
        .str("level", extra("level"))
        .str("tier", extra("shape_tier"))
        .str("origin", extra("origin"))
        .str("sizes", extra("sizes"))
        .num("work_mflop", workMflop)
        .str("est_seconds", extra("est_seconds"))
        .str("arch", ctx.arch);

    std::vector<std::string> unrealised;
    for(const auto& axis : sdpa_case_flow::modelAxisIds())
    {
        if(!sdpa_case_flow::isExecuteLevelAxis(axis))
        {
            continue;
        }
        const auto& value = input.row.at(axis);
        if(!value.empty() && value != "-" && value != sdpa_case_flow::sharedBaseValue(axis)
           && !sdpa_case_flow::isRealisedContentAxis(axis))
        {
            unrealised.push_back(axis + "=" + value);
        }
    }
    line.raw("unrealised_contents", jsonStringArray(unrealised));

    const auto finish
        = [&](const std::string& outcome, const std::string& stage, const std::string& message) {
              line.str("outcome", outcome)
                  .str("stage", stage)
                  .str("message", message)
                  .num("t_case_total_us", microsSince(caseStart));
              return line.done();
          };

    sdpa_case_flow::SizeOverride sizes;
    sdpa_case_flow::BuiltCase built;
    auto stageStart = Clock::now();
    try
    {
        sizes = sdpa_case_flow::parseSizeOverride(extra("sizes"));
        const sdpa_case_flow::CaseContext caseContext{
            ctx.values, ctx.opts.descaleForm, ctx.arch, sizes};
        built = sdpa_case_flow::buildCaseGraph(input.row, input.overrides, caseContext);
    }
    catch(const std::exception& e)
    {
        line.num("t_build_graph_us", microsSince(stageStart));
        return finish("error", "realise", e.what());
    }
    line.str("realised", built.realised);
    auto& graph = *built.graph;

    std::string graphOutcome;
    try
    {
        if(const auto validated = graph.validate(); !validated.is_good())
        {
            line.num("t_build_graph_us", microsSince(stageStart));
            return finish("frontend-rejected", "validate", validated.get_message());
        }
        if(const auto lowered = graph.build_operation_graph(ctx.handle); !lowered.is_good())
        {
            line.num("t_build_graph_us", microsSince(stageStart));
            return finish("frontend-rejected", "lower", lowered.get_message());
        }
        line.num("t_build_graph_us", microsSince(stageStart));

        stageStart = Clock::now();
        const auto queried = graph.create_execution_plan_ext(ctx.engineId, {});
        line.num("t_query_us", microsSince(stageStart));
        if(queried.is_good())
        {
            graphOutcome = "accepted";
        }
        else
        {
            graphOutcome = engineRefused(queried.get_message()) ? "not-applicable" : "error";
            line.str("engine_message", queried.get_message());
        }
    }
    catch(const std::exception& e)
    {
        return finish("error", "query", std::string("exception: ") + e.what());
    }
    line.str("graph_outcome", graphOutcome);
    acceptedOut = graphOutcome == "accepted";

    const bool selected = selectByRule || (acceptedOut && acceptedCapLeft);
    line.boolean("executed_selected", selected);
    if(!selected)
    {
        return finish(graphOutcome, "query", "not selected for execution");
    }
    if(workMflop > ctx.opts.maxMflop)
    {
        line.str("skip_reason", "size-gate");
        return finish(graphOutcome, "query", "work above --max-mflop");
    }

    try
    {
        bool planOk = false;
        if(acceptedOut)
        {
            stageStart = Clock::now();
            auto planResult = graph.check_support();
            if(planResult.is_good())
            {
                planResult = graph.build_plans();
            }
            line.num("t_plan_build_us", microsSince(stageStart));
            planOk = planResult.is_good();
            line.str("plan_build", planOk ? "ok" : "failed")
                .str("plan_build_message", planResult.get_message());
        }

        auto [blob, serErr] = graph.to_binary();
        if(!serErr.is_good() || blob.empty())
        {
            return finish(graphOutcome, "serialise", serErr.get_message());
        }
        const auto* fbGraph = fbd::GetGraph(blob.data());
        const auto wrapper
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
                blob.data(), blob.size());

        const auto graphTensors = collectGraphTensors(graph);
        const auto& attrs = graphTensors.attrs;
        const std::vector<int64_t> outputUids(graphTensors.outputs.begin(),
                                              graphTensors.outputs.end());
        Bundle engineBundle;
        for(const auto& [uid, attr] : attrs)
        {
            engineBundle.addTensor(
                *attr, hipdnn_test_sdk::utilities::createTensorFromAttribute(*attr, true));
        }
        engineBundle.outputTensorIds.insert(outputUids.begin(), outputUids.end());

        double tensorMb = 0.0;
        for(const auto& [uid, attr] : attrs)
        {
            tensorMb += tensorBytes(*attr) / (1024.0 * 1024.0);
        }
        line.num("tensor_mb", tensorMb);
        if(tensorMb > ctx.opts.maxTensorMb)
        {
            line.str("skip_reason", "memory-gate");
            return finish(graphOutcome, "memory-gate", "tensors above --max-tensor-mb");
        }

        stageStart = Clock::now();
        std::vector<std::string> contentNotes;
        const unsigned int seed = isCaseKey(extra("case_key"))
                                      ? static_cast<unsigned int>(caseSeed(extra("case_key")))
                                      : ctx.opts.seed;
        line.integer("seed", seed);
        try
        {
            const auto leafInputs = leafInputUids(graphTensors);
            it::InputFillRecipes recipes;
            setInputSeeds(recipes, leafInputs, seed);
            for(const auto& placed : setContentRecipes(recipes, built, graphTensors))
            {
                contentNotes.push_back(placed.name + ":" + recipeNote(recipes.fill(placed.uid)));
            }
            const auto fill
                = it::fillInputs(*fbGraph, engineBundle.tensors, leafInputs, recipes, nullptr);
            if(!fill.filled)
            {
                line.num("t_fill_us", microsSince(stageStart))
                    .raw("content_notes", jsonStringArray(contentNotes));
                const bool invalid = fill.status == it::FillResult::Status::INVALID;
                return finish(invalid ? "error" : graphOutcome, "fill", fill.reason);
            }
            engineBundle.sentinelFillOutputTensors();
        }
        catch(const std::exception& e)
        {
            line.num("t_fill_us", microsSince(stageStart))
                .raw("content_notes", jsonStringArray(contentNotes));
            return finish("error", "fill", std::string("exception: ") + e.what());
        }
        line.num("t_fill_us", microsSince(stageStart))
            .raw("content_notes", jsonStringArray(contentNotes));

        std::string engineStatus = acceptedOut ? (planOk ? "not-run" : "plan-failed") : "refused";
        if(planOk)
        {
            try
            {
                std::string error;
                auto workspace = it::allocatePlanWorkspace(graph, error);
                if(!workspace.has_value())
                {
                    engineStatus = "workspace-failed";
                    line.str("engine_error", error);
                }
                else
                {
                    auto variantPack = engineBundle.toDeviceVariantPack();
                    syncDevice();
                    stageStart = Clock::now();
                    const auto executed = graph.execute(ctx.handle, variantPack, workspace->get());
                    syncDevice();
                    line.num("t_engine_first_us", microsSince(stageStart));
                    if(!executed.is_good())
                    {
                        engineStatus = "execute-failed";
                        line.str("engine_error", executed.get_message());
                    }
                    else
                    {
                        engineStatus = "ok";
                        markOutputsDeviceModified(engineBundle, outputUids);

                        hipdnn_frontend::AutotuneConfig config;
                        config.strategy = hipdnn_frontend::AutotuneStrategy::FIXED_AVERAGE;
                        config.timedIterations = ctx.opts.iterations;
                        config.warmupIterations = ctx.opts.warmup;
                        config.engineIdFilter = {ctx.engineId};
                        std::vector<hipdnn_frontend::AutotuneResult> results;
                        stageStart = Clock::now();
                        const auto tuned = graph.autotune(
                            ctx.handle, variantPack, workspace->get(), config, {}, &results);
                        syncDevice();
                        line.num("t_engine_benchmark_wall_us", microsSince(stageStart));
                        bool found = false;
                        for(const auto& r : results)
                        {
                            if(r.engineId == ctx.engineId && r.iterationsRun > 0)
                            {
                                line.num("engine_avg_ms", r.avgTimeMs)
                                    .num("engine_min_ms", r.minTimeMs)
                                    .num("engine_stddev_ms", r.stddevMs)
                                    .integer("engine_iterations", r.iterationsRun);
                                found = true;
                            }
                        }
                        if(!found)
                        {
                            line.str("engine_benchmark_error",
                                     tuned.is_good() ? "no result for the engine"
                                                     : tuned.get_message());
                        }
                        markOutputsDeviceModified(engineBundle, outputUids);
                    }
                }
            }
            catch(const std::exception& e)
            {
                engineStatus = "exception";
                line.str("engine_error", e.what());
            }
        }
        line.str("engine_status", engineStatus);

        std::vector<std::string> unwrittenOutputs;
        if(engineStatus == "ok")
        {
            for(const auto uid : outputUids)
            {
                if(stillSentinel(*engineBundle.tensors.at(uid), *attrs.at(uid)))
                {
                    unwrittenOutputs.push_back("engine did not write output "
                                               + attrs.at(uid)->get_name());
                }
            }
            line.raw("engine_unwritten_outputs", jsonStringArray(unwrittenOutputs));
        }

        const auto rngDump = it::bundle::findSdpaRngDump(wrapper);
        std::string rngDumpProblem;
        std::string rngDumpSource;
        if(rngDump.has_value())
        {
            if(engineStatus == "ok")
            {
                rngDumpProblem = it::bundle::checkRngDump(*engineBundle.tensors.at(rngDump->uid),
                                                          rngDump->dropoutProbability);
                line.str("rng_dump_check", rngDumpProblem.empty() ? "ok" : rngDumpProblem);
            }
            rngDumpSource = engineStatus == "ok" && rngDumpProblem.empty() ? "engine" : "synthetic";
            line.str("rng_dump_source", rngDumpSource);
        }
        const auto handOffRngDump = [&](Bundle& reference) {
            if(!rngDump.has_value())
            {
                return;
            }
            auto& target = *reference.tensors.at(rngDump->uid);
            if(rngDumpSource == "engine")
            {
                it::bundle::copyRngDump(*engineBundle.tensors.at(rngDump->uid), target);
            }
            else
            {
                writeSyntheticKeepMask(target, rngDump->dropoutProbability, seed);
            }
        };
        const auto comparedUids = it::bundle::comparedOutputUids(outputUids, rngDump);

        Bundle gpuBundle = withFreshOutputs(engineBundle, attrs, outputUids);
        std::string gpuStatus = "not-run";
        if(!ctx.opts.noGpuRef)
        {
            try
            {
                if(!ctx.gpuRef.isApplicable(blob.data(), blob.size()))
                {
                    gpuStatus = "declined";
                    line.str("gpu_ref_message", "GPU reference has no plan for this graph");
                }
                else
                {
                    handOffRngDump(gpuBundle);
                    auto variantPack = gpuBundle.toDeviceVariantPack();
                    syncDevice();
                    stageStart = Clock::now();
                    ctx.gpuRef.execute(blob.data(), blob.size(), variantPack);
                    syncDevice();
                    line.num("t_gpu_ref_first_us", microsSince(stageStart));
                    stageStart = Clock::now();
                    ctx.gpuRef.execute(blob.data(), blob.size(), variantPack);
                    syncDevice();
                    line.num("t_gpu_ref_warm_us", microsSince(stageStart));
                    markOutputsDeviceModified(gpuBundle, outputUids);
                    gpuStatus = "ok";
                }
            }
            catch(const it::ReferenceCapabilityError& e)
            {
                gpuStatus = "declined";
                line.str("gpu_ref_message", e.what());
            }
            catch(const std::exception& e)
            {
                gpuStatus = "error";
                line.str("gpu_ref_message", e.what());
            }
        }
        line.str("gpu_ref_status", gpuStatus);

        Bundle cpuBundle = withFreshOutputs(engineBundle, attrs, outputUids);
        std::string cpuStatus = "size-gate";
        if(workMflop <= ctx.opts.cpuRefMaxMflop)
        {
            try
            {
                if(!ctx.cpuRef.isApplicable(blob.data(), blob.size()))
                {
                    cpuStatus = "declined";
                    line.str("cpu_ref_message", "CPU reference has no plan for this graph");
                }
                else
                {
                    handOffRngDump(cpuBundle);
                    auto variantPack = cpuBundle.toHostVariantPack();
                    stageStart = Clock::now();
                    ctx.cpuRef.execute(blob.data(), blob.size(), variantPack);
                    line.num("t_cpu_ref_us", microsSince(stageStart));
                    markOutputsHostModified(cpuBundle, outputUids);
                    cpuStatus = "ok";
                }
            }
            catch(const it::ReferenceCapabilityError& e)
            {
                cpuStatus = "declined";
                line.str("cpu_ref_message", e.what());
            }
            catch(const std::exception& e)
            {
                cpuStatus = "error";
                line.str("cpu_ref_message", e.what());
            }
        }
        line.str("cpu_ref_status", cpuStatus);

        std::vector<std::string> comparisons;
        std::string verdict = "no-comparison";
        const bool engineComparable = engineStatus == "ok" && rngDumpProblem.empty();
        const auto engineComparison =
            [&](const std::string& against, Bundle& reference, it::ValidationSite site) {
                const auto c
                    = engineComparable
                          ? compareBundles(against,
                                           wrapper,
                                           comparedUids,
                                           attrs,
                                           engineBundle.tensors,
                                           reference,
                                           site)
                          : skippedComparison(against, "rng_dump check failed: " + rngDumpProblem);
                comparisons.push_back(c.json());
                if(engineComparable && verdict == "no-comparison")
                {
                    verdict = c.ran ? (c.pass ? "pass" : "fail") : "compare-error";
                }
            };
        if(engineStatus == "ok" && gpuStatus == "ok")
        {
            engineComparison("engine-vs-gpu-ref", gpuBundle, it::ValidationSite::DEVICE);
        }
        if(engineStatus == "ok" && cpuStatus == "ok")
        {
            engineComparison("engine-vs-cpu-ref", cpuBundle, it::ValidationSite::HOST);
        }
        if(gpuStatus == "ok" && cpuStatus == "ok")
        {
            const auto c = compareBundles("gpu-ref-vs-cpu-ref",
                                          wrapper,
                                          comparedUids,
                                          attrs,
                                          gpuBundle.tensors,
                                          cpuBundle,
                                          it::ValidationSite::HOST,
                                          true);
            comparisons.push_back(c.json());
        }
        if(!rngDumpProblem.empty())
        {
            verdict = "fail";
        }
        if(!unwrittenOutputs.empty())
        {
            verdict = "unwritten-output";
        }
        std::string comparisonJson = "[";
        for(size_t i = 0; i < comparisons.size(); ++i)
        {
            comparisonJson += (i == 0 ? "" : ",") + comparisons[i];
        }
        line.raw("comparisons", comparisonJson + "]").str("verdict", verdict);
        return finish(graphOutcome, "done", "");
    }
    catch(const std::exception& e)
    {
        return finish(graphOutcome, "harness", std::string("exception: ") + e.what());
    }
}

// ------------------------------------------------------------------ graph export

constexpr const char* EXPORT_GRAPH_NAME = "sdpa_case_flow";
constexpr const char* EXPORT_GENERATOR = "hipdnn_sdpa_case_flow_runner";
constexpr const char* EXPORT_REFERENCE_SOURCE
    = "hipDNN GPU and CPU SDPA reference executors; the case is the row with this case_key in "
      "test-plans/sdpa-fwd/cases.csv";

// The contents a bundle carries for these graphs: a fixed attn_scale and the sequence lengths.
// Every other content keeps the row out of bundles.
bool isBundleContent(const std::string& name)
{
    return name == FIXED_CONTENT_NAME || name == "seq_len_q" || name == "seq_len_kv";
}

struct ExportResult
{
    std::string outcome;
    std::string stage;
    std::string message;
    std::string inputsNote;
};

bool writeText(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    out.close();
    return static_cast<bool>(out);
}

ExportResult exportCase(const CaseInput& input,
                        const Options& opts,
                        const sdpa_case_flow::ClassValues& values)
{
    ExportResult result;
    result.stage = "realise";
    const auto extra = [&](const std::string& key) {
        const auto found = input.extra.find(key);
        return found == input.extra.end() ? std::string() : found->second;
    };
    const auto fail = [&](const std::string& outcome, const std::string& message) {
        result.outcome = outcome;
        result.message = message;
        return result;
    };

    try
    {
        const std::string caseKey = extra("case_key");
        if(!isCaseKey(caseKey))
        {
            return fail("realise-error", "case_key '" + caseKey + "' is not 16 hex digits");
        }

        const auto sizes = sdpa_case_flow::parseSizeOverride(extra("sizes"));
        const sdpa_case_flow::CaseContext caseContext{
            values, opts.descaleForm, opts.exportArch, sizes};
        auto built = sdpa_case_flow::buildCaseGraph(input.row, input.overrides, caseContext);
        auto& graph = *built.graph;
        graph.set_name(EXPORT_GRAPH_NAME);

        result.stage = "contents";
        std::vector<std::string> notExpressible;
        for(const auto& [name, contents] : built.contents)
        {
            const auto* reals = std::get_if<std::vector<double>>(&contents.values);
            const size_t count = reals != nullptr
                                     ? reals->size()
                                     : std::get<std::vector<int64_t>>(contents.values).size();
            const bool fixedScale = name == FIXED_CONTENT_NAME && reals != nullptr && count == 1;
            const bool lengths = name != FIXED_CONTENT_NAME && reals == nullptr;
            if(!isBundleContent(name) || !(fixedScale || lengths))
            {
                notExpressible.push_back(
                    name + "(" + std::to_string(count)
                    + (reals != nullptr ? " float values)" : " index values)"));
            }
        }
        if(!notExpressible.empty())
        {
            std::string message = "contents a bundle cannot express:";
            for(const auto& item : notExpressible)
            {
                message += " " + item;
            }
            return fail("content-not-expressible", message);
        }

        result.stage = "validate";
        if(const auto validated = graph.validate(); !validated.is_good())
        {
            return fail("frontend-rejected", validated.get_message());
        }

        result.stage = "contents";
        const auto graphTensors = collectGraphTensors(graph);
        for(const auto& [uid, attr] : graphTensors.attrs)
        {
            if(attr->get_ragged_offset() != nullptr)
            {
                return fail("content-not-expressible",
                            "ragged tensor " + attr->get_name()
                                + ": the harness does not verify ragged graphs");
            }
        }
        const auto leafInputs = leafInputUids(graphTensors);
        it::InputFillRecipes recipes;
        setInputSeeds(recipes, leafInputs, static_cast<unsigned int>(caseSeed(caseKey)));
        const auto placed = setContentRecipes(recipes, built, graphTensors);

        result.stage = "serialise";
        auto [jsonText, serialised] = graph.to_json();
        if(!serialised.is_good() || jsonText.empty())
        {
            return fail("frontend-rejected", serialised.get_message());
        }
        const auto graphJson = nlohmann::json::parse(jsonText);

        result.stage = "contents";
        std::map<int64_t, std::string> jsonNames;
        if(graphJson.contains("tensors") && graphJson["tensors"].is_array())
        {
            for(const auto& tensor : graphJson["tensors"])
            {
                if(tensor.contains("uid") && tensor["uid"].is_number_integer())
                {
                    jsonNames[tensor["uid"].get<int64_t>()] = tensor.value("name", std::string());
                }
            }
        }
        for(const auto uid : leafInputs)
        {
            const auto found = jsonNames.find(uid);
            if(found == jsonNames.end() || found->second != graphTensors.attrs.at(uid)->get_name())
            {
                return fail("content-not-expressible",
                            "input uid " + std::to_string(uid) + " ("
                                + graphTensors.attrs.at(uid)->get_name()
                                + ") is not a tensor of that name in the graph JSON");
            }
        }

        nlohmann::json meta;
        meta["format_version"] = it::requiredFormatVersion(recipes);
        meta["generator"] = EXPORT_GENERATOR;
        meta["generator_version"] = SDPA_CASE_FLOW_RUNNER_SOURCE_HASH;
        meta["reference_source"] = EXPORT_REFERENCE_SOURCE;
        meta["seed"] = caseSeed(caseKey);
        meta["case_key"] = caseKey;
        meta["csv_case_id"] = extra("case_id");
        meta["generated_from_sha256"] = opts.sourceSha256;
        meta["inputs"] = recipes.toJson();

        for(const auto& content : placed)
        {
            result.inputsNote += (result.inputsNote.empty() ? "" : "; ") + content.name + " uid "
                                 + std::to_string(content.uid) + " "
                                 + recipeNote(recipes.fill(content.uid));
        }

        result.stage = "write";
        const auto caseDir = std::filesystem::path(opts.exportDir) / caseKey;
        std::filesystem::create_directories(caseDir);
        if(!writeText(caseDir / (caseKey + ".json"), graphJson.dump(2) + "\n")
           || !writeText(caseDir / (caseKey + ".meta.json"), meta.dump(2) + "\n"))
        {
            return fail("write-error", "cannot write the files under " + caseDir.string());
        }
        result.stage = "done";
        result.outcome = "exported";
        return result;
    }
    catch(const std::exception& e)
    {
        const bool frontendStage = result.stage == "validate" || result.stage == "serialise";
        const std::string outcome = result.stage == "realise" ? "realise-error"
                                    : frontendStage           ? "frontend-rejected"
                                    : result.stage == "write" ? "write-error"
                                                              : "content-not-expressible";
        return fail(outcome, std::string("exception: ") + e.what());
    }
    catch(...)
    {
        return fail("realise-error", "unknown exception");
    }
}

int runExport(const Options& opts,
              const std::vector<CaseInput>& cases,
              const sdpa_case_flow::ClassValues& values,
              int argc,
              char** argv)
{
    const auto runStart = Clock::now();
    std::ofstream out(opts.output, opts.append ? std::ios::app : std::ios::trunc);
    if(!out)
    {
        inputError("cannot open output '" + opts.output + "'");
    }
    std::error_code ec;
    std::filesystem::create_directories(opts.exportDir, ec);
    if(ec)
    {
        inputError("cannot create '" + opts.exportDir + "': " + ec.message());
    }

    JsonLine runRecord;
    runRecord.str("record", "export-run")
        .raw("argv", jsonStringArray(std::vector<std::string>(argv, argv + argc)))
        .str("hostname", hostName())
        .str("arch", opts.exportArch)
        .str("generator_version", SDPA_CASE_FLOW_RUNNER_SOURCE_HASH)
        .str("generated_from_sha256", opts.sourceSha256)
        .integer("input_rows", static_cast<int64_t>(cases.size()));
    for(const auto& [key, value] : opts.meta)
    {
        runRecord.str("meta_" + key, value);
    }
    out << runRecord.done() << "\n" << std::flush;

    std::map<std::string, int64_t> outcomes;
    const size_t first = static_cast<size_t>(opts.start);
    const size_t last = opts.limit > 0
                            ? std::min(cases.size(), first + static_cast<size_t>(opts.limit))
                            : cases.size();
    for(size_t index = first; index < last; ++index)
    {
        const auto& input = cases[index];
        const auto extra = [&](const std::string& key) {
            const auto found = input.extra.find(key);
            return found == input.extra.end() ? std::string() : found->second;
        };
        const auto result = exportCase(input, opts, values);
        ++outcomes[result.outcome];
        JsonLine line;
        line.str("record", "export")
            .integer("index", static_cast<int64_t>(index))
            .str("case_key", extra("case_key"))
            .str("case_id", extra("case_id"))
            .str("level", extra("level"))
            .str("arch", opts.exportArch)
            .str("outcome", result.outcome)
            .str("stage", result.stage)
            .str("message", result.message)
            .str("inputs", result.inputsNote);
        out << line.done() << "\n" << std::flush;
    }

    JsonLine summary;
    summary.str("record", "export-summary")
        .integer("rows_processed", static_cast<int64_t>(last > first ? last - first : 0));
    for(const auto& [outcome, count] : outcomes)
    {
        summary.integer(outcome, count);
    }
    summary.num("wall_s", microsSince(runStart) / 1.0e6);
    out << summary.done() << "\n" << std::flush;
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    const auto runStart = Clock::now();
    const Options opts = parseArgs(argc, argv);
    const std::vector<CaseInput> cases = readInput(opts.input);

    sdpa_case_flow::ClassValues values;
    try
    {
        values
            = sdpa_case_flow::parseClassValues(readFile(opts.classValues, "class-to-value file"));
    }
    catch(const std::exception& e)
    {
        inputError(e.what());
    }

    if(!opts.exportDir.empty())
    {
        return runExport(opts, cases, values, argc, argv);
    }

    std::ofstream out(opts.output, opts.append ? std::ios::app : std::ios::trunc);
    if(!out)
    {
        inputError("cannot open output '" + opts.output + "'");
    }

    const DeviceInfo device = queryDevice();
    JsonLine runRecord;
    runRecord.str("record", "run")
        .raw("argv", jsonStringArray(std::vector<std::string>(argv, argv + argc)))
        .str("hostname", hostName())
        .str("arch", device.arch)
        .str("device_name", device.name)
        .integer("device_count", device.deviceCount)
        .str("hip_error", device.hipError)
        .integer("input_rows", static_cast<int64_t>(cases.size()))
        .str("engine_requested", opts.engine);
    for(const auto& [key, value] : opts.meta)
    {
        runRecord.str("meta_" + key, value);
    }

    std::string fatal;
    int exitCode = 0;
    if(!opts.expectArch.empty() && baseArch(device.arch) != opts.expectArch)
    {
        fatal
            = "HIP reports arch '" + device.arch + "'; --expect-arch is '" + opts.expectArch + "'";
        exitCode = EXIT_ARCH_MISMATCH;
    }
    if(fatal.empty() && !opts.pluginPaths.empty())
    {
        const auto status = hipdnn_frontend::setEnginePluginPaths(
            opts.pluginPaths, hipdnn_frontend::PluginLoadingMode::MODE_ABSOLUTE);
        if(!status.is_good())
        {
            fatal = "plugin paths: " + status.get_message();
        }
    }

    Context ctx{opts, values, baseArch(device.arch)};
    if(fatal.empty())
    {
        if(hipdnnCreate(&ctx.handle) != HIPDNN_STATUS_SUCCESS)
        {
            ctx.handle = nullptr;
            fatal = "hipdnnCreate failed";
        }
        else if(hipdnnGetEngineIdByName_ext(ctx.handle, opts.engine.c_str(), &ctx.engineId)
                != HIPDNN_STATUS_SUCCESS)
        {
            fatal = "no loaded engine carries the name '" + opts.engine + "'";
            exitCode = EXIT_ENGINE_NOT_LOADED;
        }
        else
        {
            runRecord.str("engine_id", hipdnn_data_sdk::utilities::formatEngineIdHex(ctx.engineId));
        }
    }
    runRecord.str("fatal", fatal);
    out << runRecord.done() << "\n" << std::flush;
    if(!fatal.empty())
    {
        std::cerr << "fatal: " << fatal << "\n";
        return exitCode == 0 ? 1 : exitCode;
    }

    std::set<size_t> spread;
    if(opts.selectPerTier >= 0)
    {
        std::map<std::string, std::vector<size_t>> byTier;
        for(size_t i = 0; i < cases.size(); ++i)
        {
            byTier[cases[i].extra.count("shape_tier") ? cases[i].extra.at("shape_tier") : ""]
                .push_back(i);
        }
        for(const auto& [tier, indices] : byTier)
        {
            const auto n
                = std::min<size_t>(static_cast<size_t>(opts.selectPerTier), indices.size());
            for(size_t k = 0; k < n; ++k)
            {
                spread.insert(indices[k * indices.size() / n]);
            }
        }
    }
    const bool selectAll
        = opts.selectPerTier < 0 && opts.selectLevels.empty() && opts.selectAcceptedPerTier == 0;

    std::map<std::string, int64_t> acceptedExecuted;
    std::map<std::string, int64_t> outcomes;
    const size_t first = static_cast<size_t>(opts.start);
    const size_t last = opts.limit > 0
                            ? std::min(cases.size(), first + static_cast<size_t>(opts.limit))
                            : cases.size();
    for(size_t index = first; index < last; ++index)
    {
        if(!opts.progressFile.empty())
        {
            std::ofstream progress(opts.progressFile, std::ios::trunc);
            progress << index << "\n";
        }
        const auto& input = cases[index];
        const auto tierIt = input.extra.find("shape_tier");
        const std::string tier = tierIt == input.extra.end() ? "" : tierIt->second;
        const auto levelIt = input.extra.find("level");
        const bool byLevel
            = levelIt != input.extra.end() && opts.selectLevels.count(levelIt->second) != 0;
        const bool byRule = selectAll || byLevel || spread.count(index) != 0;
        const bool capLeft = acceptedExecuted[tier] < opts.selectAcceptedPerTier;
        bool accepted = false;
        const auto text = runCase(index, input, byRule, capLeft, ctx, accepted);
        if(accepted && !byRule && capLeft)
        {
            ++acceptedExecuted[tier];
        }
        ++outcomes[accepted ? "accepted" : "other"];
        out << text << "\n" << std::flush;
        std::cerr << "row " << index << " done\n";
    }

    JsonLine summary;
    summary.str("record", "summary")
        .integer("rows_processed", static_cast<int64_t>(last > first ? last - first : 0))
        .integer("accepted", outcomes["accepted"])
        .num("wall_s", microsSince(runStart) / 1.0e6);
    out << summary.done() << "\n" << std::flush;
    if(ctx.handle != nullptr)
    {
        static_cast<void>(hipdnnDestroy(ctx.handle));
    }
    return 0;
}
