// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>

#include <algorithm>
#include <charconv>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

// NOLINTBEGIN(google-global-names-in-headers)
using hipdnn_data_sdk::utilities::TensorLayout;

using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;
// NOLINTEND(google-global-names-in-headers)

// ERROR MACROS

#define HIP_CHECK(status)                                                                      \
    do                                                                                         \
    {                                                                                          \
        if((status) != hipSuccess)                                                             \
        {                                                                                      \
            std::cerr << "HIP Error: " << hipGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << '\n';                                      \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while(0)

#define HIPDNN_CHECK(status)                                                             \
    do                                                                                   \
    {                                                                                    \
        if((status) != HIPDNN_STATUS_SUCCESS)                                            \
        {                                                                                \
            std::cerr << "hipDNN Error: " << hipdnnGetErrorString(status) << " in file " \
                      << __FILE__ << " at line " << __LINE__ << '\n';                    \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while(0)

#define HIPDNN_FE_CHECK(statusObj)                                                        \
    do                                                                                    \
    {                                                                                     \
        auto const& status = statusObj;                                                   \
        if(!status.is_good())                                                             \
        {                                                                                 \
            std::cerr << "hipDNN Frontend Error: " << status.get_message() << " in file " \
                      << __FILE__ << " at line " << __LINE__ << '\n';                     \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while(0)

// Skip-aware variant of HIPDNN_FE_CHECK for use inside bool-returning sample
// callbacks (e.g. SampleRunner::operator()). On GRAPH_NOT_SUPPORTED the macro
// prints a clear skip message and `return true;` so the enclosing variant is
// counted as gracefully skipped (samples/README.md documents this contract).
// On any other non-good status, behavior matches HIPDNN_FE_CHECK (exit 1).
//
// The macro contains `return true;`, so it MUST only be used inside a
// bool-returning function context. For non-bool contexts (e.g. int main),
// use HIPDNN_FE_CHECK instead.
#define HIPDNN_FE_CHECK_SKIPPABLE(statusObj)                                                    \
    do                                                                                          \
    {                                                                                           \
        auto const& status = statusObj;                                                         \
        if(!status.is_good())                                                                   \
        {                                                                                       \
            if(status.get_code() == hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED)            \
            {                                                                                   \
                std::cout << "Skipping: no engine has an applicable solution for this "         \
                          << "graph on the current device. (" << status.get_message() << ")\n"; \
                return true;                                                                    \
            }                                                                                   \
            std::cerr << "hipDNN Frontend Error: " << status.get_message() << " in file "       \
                      << __FILE__ << " at line " << __LINE__ << '\n';                           \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while(0)

// Gracefully skip a sample when no GPU is present: prints a skip message and
// returns 0 from the enclosing function.
#define RETURN_SUCCESS_IF_NO_DEVICE()                                         \
    do                                                                        \
    {                                                                         \
        int deviceCount = 0;                                                  \
        if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) \
        {                                                                     \
            std::cout << "SKIPPED: No GPU devices available.\n";              \
            return 0;                                                         \
        }                                                                     \
    } while(0)

// SAMPLE TYPES

enum class SampleType
{
    GENERIC,
    BN_TRAINING,
    SDPA,
    BN_WITH_PASS_BY_VALUE, // batchnorm samples that support pass-by-value tensors (epsilon)
};

// Single source of truth for which sample types accept --runtime-pass-by-value, so the
// help text and the CLI parser can't drift out of sync as more sample types gain support.
inline bool supportsRuntimePassByValue(SampleType sampleType)
{
    return sampleType == SampleType::BN_TRAINING || sampleType == SampleType::BN_WITH_PASS_BY_VALUE
           || sampleType == SampleType::SDPA;
}

// HELP MESSAGE

inline void printSampleHelp(const std::string& sampleName,
                            SampleType sampleType = SampleType::GENERIC)
{
    std::cout << "Usage: " << sampleName << " [OPTIONS]\n"
              << "Options:\n"
              << "  --verify-cpu, -vc           Enable CPU reference validation\n"
              << "  --engine-id <int>           Preferred engine ID\n"
              << "  --engine-name <name>        Preferred engine name\n"
              << "  --dtype <fp32|fp16|bf16>    Data type\n";

    if(sampleType == SampleType::SDPA)
    {
        std::cout << "  --layout <bhsd|bshd>        Tensor layout\n";
    }
    else
    {
        std::cout << "  --layout <nchw|nhwc>        Tensor layout\n";
    }

    // SDPA's tensor shapes are hardcoded constants (batch/heads/seq_len/head_dim),
    // so the shape-related options below don't apply and would be misleading to list.
    if(sampleType != SampleType::SDPA)
    {
        std::cout << "  --dims N,C,H,W              Input dimensions\n"
                  << "  --filter K,R,S              Filter size (output channels, height, width)\n"
                  << "  --stride U,V                Stride\n"
                  << "  --padding PH,PW             Padding\n"
                  << "  --dilation DH,DW            Dilation\n";
    }

    if(sampleType == SampleType::BN_TRAINING)
    {
        std::cout << "  --batch-stats-only          Use batch statistics only\n"
                  << "  --full-training             Use running statistics\n";
    }

    if(supportsRuntimePassByValue(sampleType))
    {
        std::cout << "  --runtime-pass-by-value     Supply pass-by-value tensors as runtime host\n"
                  << "                              values instead of compile-time constants\n";
    }

    std::cout << "  --help, -h                  Show this help message\n\n";
}

// CONFIG

struct Config
{
    bool cpuValidation = false;
    bool useRunningStats = false;
    bool useRuntimePassByValue = false;

    int engineId = -1;
    std::string dtype;
    std::string layout;
    std::string engineName;

    std::vector<int64_t> dims;
    std::vector<int64_t> filter;
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
};

// PARSING UTILS

// Parses a single integer using std::from_chars, exiting with a clear error message
// instead of throwing on malformed input (unlike std::stoi/std::stoll).
template <typename T>
inline T parseInteger(const std::string& str, const std::string& context)
{
    T value{};
    const char* begin = str.data();
    const char* end = str.data() + str.size();

    auto [ptr, ec] = std::from_chars(begin, end, value);

    if(ec != std::errc() || ptr != end)
    {
        std::cerr << "Invalid integer value for " << context << ": \"" << str << "\"\n";
        exit(EXIT_FAILURE);
    }

    return value;
}

inline std::vector<int64_t> parseList(const std::string& str)
{
    std::vector<int64_t> result;
    std::stringstream ss(str);
    std::string item;

    while(std::getline(ss, item, ','))
    {
        result.push_back(parseInteger<int64_t>(item, "list argument"));
    }

    return result;
}

// Parses a comma-separated list and enforces it contains exactly `expectedSize`
// elements, exiting with a message naming the option and its expected format on
// mismatch. Used for every list-valued option so malformed/truncated input is
// caught explicitly instead of silently falling back to per-field defaults.
inline std::vector<int64_t> parseListWithLength(const std::string& str,
                                                size_t expectedSize,
                                                const std::string& optionName,
                                                const std::string& expectedFormat)
{
    auto result = parseList(str);

    if(result.size() != expectedSize)
    {
        std::cerr << optionName << " must contain " << expectedSize << " values (" << expectedFormat
                  << ")\n";
        exit(EXIT_FAILURE);
    }

    return result;
}

// Prints the resolved CLI configuration so users can visually confirm how their
// input was interpreted before the sample runs. Only non-default fields are shown
// to keep output concise for the common case of few/no options being passed.
inline void printConfig(const Config& config)
{
    std::cout << "Configuration:\n";
    std::cout << "  --verify-cpu: " << (config.cpuValidation ? "true" : "false") << '\n';

    if(config.engineId != -1)
    {
        std::cout << "  --engine-id: " << config.engineId << '\n';
    }
    if(!config.engineName.empty())
    {
        std::cout << "  --engine-name: " << config.engineName << '\n';
    }
    if(!config.dtype.empty())
    {
        std::cout << "  --dtype: " << config.dtype << '\n';
    }
    if(!config.layout.empty())
    {
        std::cout << "  --layout: " << config.layout << '\n';
    }

    auto printList = [](const char* name, const std::vector<int64_t>& values) {
        if(values.empty())
        {
            return;
        }

        std::cout << "  " << name << ": ";
        for(size_t i = 0; i < values.size(); ++i)
        {
            std::cout << values[i];
            if(i + 1 < values.size())
            {
                std::cout << ",";
            }
        }
        std::cout << '\n';
    };

    printList("--dims", config.dims);
    printList("--filter", config.filter);
    printList("--stride", config.stride);
    printList("--padding", config.padding);
    printList("--dilation", config.dilation);
    std::cout << "  --runtime-pass-by-value: " << (config.useRuntimePassByValue ? "true" : "false")
              << '\n';

    std::cout << '\n';
}

// CLI PARSER

inline Config
    parseCommandLineArgs(int argc, char** argv, SampleType sampleType = SampleType::GENERIC)
{
    Config config;

    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if(arg == "--verify-cpu" || arg == "-vc")
        {
            config.cpuValidation = true;
        }
        else if(arg == "--batch-stats-only" && sampleType == SampleType::BN_TRAINING)
        {
            config.useRunningStats = false;
        }
        else if(arg == "--full-training" && sampleType == SampleType::BN_TRAINING)
        {
            config.useRunningStats = true;
        }
        else if(arg == "--runtime-pass-by-value" && supportsRuntimePassByValue(sampleType))
        {
            config.useRuntimePassByValue = true;
        }
        else if(arg == "--engine-id")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--engine-id requires a value\n";
                exit(EXIT_FAILURE);
            }
            config.engineId = parseInteger<int>(argv[++i], "--engine-id");
        }
        else if(arg == "--engine-name")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--engine-name requires a value\n";
                exit(EXIT_FAILURE);
            }
            config.engineName = argv[++i];
        }
        else if(arg == "--dtype")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--dtype requires a value\n";
                exit(EXIT_FAILURE);
            }

            config.dtype = argv[++i];

            if(config.dtype != "fp32" && config.dtype != "fp16" && config.dtype != "bf16")
            {
                std::cerr << "Invalid value for --dtype: " << config.dtype
                          << " (expected: fp32, fp16, bf16)\n";
                exit(EXIT_FAILURE);
            }
        }
        else if(arg == "--layout")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--layout requires a value\n";
                exit(EXIT_FAILURE);
            }

            config.layout = argv[++i];

            if(sampleType == SampleType::SDPA)
            {
                if(config.layout != "bhsd" && config.layout != "bshd")
                {
                    std::cerr << "Invalid value for --layout: " << config.layout
                              << " (expected: bhsd, bshd)\n";
                    exit(EXIT_FAILURE);
                }
            }
            else if(config.layout != "nchw" && config.layout != "nhwc")
            {
                std::cerr << "Invalid value for --layout: " << config.layout
                          << " (expected: nchw, nhwc)\n";
                exit(EXIT_FAILURE);
            }
        }
        else if(arg == "--dims" && sampleType != SampleType::SDPA)
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--dims requires a value\n";
                exit(EXIT_FAILURE);
            }

            config.dims = parseListWithLength(argv[++i], 4, "--dims", "N,C,H,W");
        }
        else if(arg == "--filter" && sampleType != SampleType::SDPA)
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--filter requires a value\n";
                exit(EXIT_FAILURE);
            }
            config.filter = parseListWithLength(argv[++i], 3, "--filter", "K,R,S");
        }
        else if(arg == "--stride" && sampleType != SampleType::SDPA)
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--stride requires a value\n";
                exit(EXIT_FAILURE);
            }
            config.stride = parseListWithLength(argv[++i], 2, "--stride", "U,V");
        }
        else if(arg == "--padding" && sampleType != SampleType::SDPA)
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--padding requires a value\n";
                exit(EXIT_FAILURE);
            }
            config.padding = parseListWithLength(argv[++i], 2, "--padding", "PH,PW");
        }
        else if(arg == "--dilation" && sampleType != SampleType::SDPA)
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--dilation requires a value\n";
                exit(EXIT_FAILURE);
            }
            config.dilation = parseListWithLength(argv[++i], 2, "--dilation", "DH,DW");
        }
        else if(arg == "--help" || arg == "-h")
        {
            printSampleHelp(argv[0], sampleType);
            exit(EXIT_SUCCESS);
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << '\n';
            printSampleHelp(argv[0], sampleType);
            exit(EXIT_FAILURE);
        }
    }

    // Prevent conflicting options
    if(config.engineId != -1 && !config.engineName.empty())
    {
        std::cerr << "Specify either --engine-id or --engine-name, not both\n";
        exit(EXIT_FAILURE);
    }

    printConfig(config);

    return config;
}

template <typename F>
bool run(F&& f)
{
    bool allPassed = true;

    const std::vector<std::string> dtypes = {"fp32", "fp16", "bf16"};
    const std::vector<std::pair<std::string, TensorLayout>> layouts
        = {{"nchw", TensorLayout::NCHW}, {"nhwc", TensorLayout::NHWC}};

    for(const auto& dt : dtypes)
    {
        // Skip data types not requested via --dtype (empty config.dtype means "run all").
        if(!f.config.dtype.empty() && f.config.dtype != dt)
        {
            continue;
        }

        for(const auto& [layoutName, layout] : layouts)
        {
            // Skip layouts not requested via --layout (empty config.layout means "run all").
            if(!f.config.layout.empty() && f.config.layout != layoutName)
            {
                continue;
            }

            if(dt == "fp32")
            {
                allPassed &= f.template operator()<float, float>(layout);
            }
            else if(dt == "fp16")
            {
                allPassed &= f.template operator()<half, float>(layout);
            }
            else if(dt == "bf16")
            {
                allPassed &= f.template operator()<bfloat16, float>(layout);
            }
        }
    }

    return allPassed;
}

// ENGINE SELECTION

// Applies the engine preference from `config` (--engine-id or --engine-name) to `graph`.
// An unrecognized --engine-name almost always indicates a typo, so this exits with an
// error rather than silently continuing with a default/unintended engine. Centralized
// here so every sample gets consistent validation instead of duplicating this logic.
inline void setPreferredEngine(hipdnn_frontend::graph::Graph& graph, const Config& config)
{
    if(config.engineId != -1)
    {
        graph.set_preferred_engine_id_ext(config.engineId);
    }
    else if(!config.engineName.empty())
    {
        if(!hipdnn_data_sdk::utilities::isEngineNameRegistered(config.engineName))
        {
            std::cerr << "Error: Unknown engine name: " << config.engineName << '\n';
            exit(EXIT_FAILURE);
        }

        graph.set_preferred_engine_id_ext(
            hipdnn_data_sdk::utilities::engineNameToId(config.engineName));
    }
}

// Overload for the common case where the graph is held via shared_ptr.
inline void setPreferredEngine(const std::shared_ptr<hipdnn_frontend::graph::Graph>& graph,
                               const Config& config)
{
    setPreferredEngine(*graph, config);
}

// TENSOR HELPERS

inline std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    createTensor(const std::vector<int64_t>& dims,
                 hipdnn_frontend::DataType_t dataType,
                 const TensorLayout& layout = TensorLayout::NCHW)
{
    auto tensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    tensor->set_dim(dims).set_data_type(dataType).set_stride(
        hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder));
    return tensor;
}

inline int64_t
    getTensorElementCount(const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& tensor)
{
    int64_t count = 1;
    for(auto dim : tensor->get_dim())
    {
        count *= dim;
    }

    return count;
}

// SAMPLE RUNNER

struct SampleRunner
{
    hipdnnHandle_t handle;
    Config config;

    template <typename InputType, typename IntermediateType>
    bool operator()(const TensorLayout& layout);
};
