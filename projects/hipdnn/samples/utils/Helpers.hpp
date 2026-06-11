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
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream> // added
#include <vector>

// NOLINTBEGIN(google-global-names-in-headers)
using hipdnn_data_sdk::utilities::TensorLayout;

// Use portable custom types instead of HIP types (works with any C++ compiler)
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

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

// CLI HELP

enum class SampleType
{
    GENERIC,
    BN_TRAINING
};

inline void printSampleHelp(const std::string& sampleName,
                            SampleType sampleType = SampleType::GENERIC)
{
    std::cout << "Usage: " << sampleName << " [OPTIONS]\n"
              << "Options:\n"
              << "  --verify-cpu, -vc           Enable CPU reference validation\n"
              << "  --engine-id <int>           Preferred engine ID\n"
              << "  --dtype <fp32|fp16|bf16>    Data type\n"
              << "  --layout <nchw|nhwc>        Tensor layout\n"
              << "  --dims N,C,H,W              Input dimensions\n"
              << "  --filter R,S                Filter size\n"
              << "  --stride U,V                Stride\n"
              << "  --padding PH,PW             Padding\n"
              << "  --dilation DH,DW            Dilation\n";

    if(sampleType == SampleType::BN_TRAINING)
    {
        std::cout << "  --batch-stats-only          Use batch statistics only\n"
                  << "  --full-training             Use running statistics\n";
    }

    std::cout << "  --help, -h                  Show help\n";
}

// CONFIG

struct Config
{
    bool cpuValidation = false;
    bool useRunningStats = false;

    // NEW CLI fields
    int engine_id = -1;
    std::string dtype;
    std::string layout;

    std::vector<int64_t> dims;
    std::vector<int64_t> filter;
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
};

// PARSING UTILS

inline std::vector<int64_t> parseList(const std::string& str)
{
    std::vector<int64_t> result;
    std::stringstream ss(str);
    std::string item;

    while(std::getline(ss, item, ','))
    {
        result.push_back(std::stoll(item));
    }

    return result;
}

// CLI PARSER

inline Config
    parseCommandLineArgs(int argc, char** argv, SampleType sampleType = SampleType::GENERIC)
{
    auto config = Config{};

    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

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

        // NEW FLAGS

        else if(arg == "--engine-id")
        {
            config.engine_id = std::stoi(argv[++i]);
        }
        else if(arg == "--dtype")
        {
            config.dtype = argv[++i];
        }
        else if(arg == "--layout")
        {
            config.layout = argv[++i];
        }
        else if(arg == "--dims")
        {
            config.dims = parseList(argv[++i]);
        }
        else if(arg == "--filter")
        {
            config.filter = parseList(argv[++i]);
        }
        else if(arg == "--stride")
        {
            config.stride = parseList(argv[++i]);
        }
        else if(arg == "--padding")
        {
            config.padding = parseList(argv[++i]);
        }
        else if(arg == "--dilation")
        {
            config.dilation = parseList(argv[++i]);
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

    return config;
}

// RUN FUNCTION

template <typename F>
bool run(F&& f, const Config& config)
{
    bool allPassed = true;

    std::vector<std::string> dtypes;
    std::vector<TensorLayout> layouts;

    // dtype selection
    if(!config.dtype.empty())
        dtypes.push_back(config.dtype);
    else
        dtypes = {"fp32", "fp16", "bf16"};

    // layout selection
    if(!config.layout.empty())
        layouts.push_back(config.layout == "nhwc" ? TensorLayout::NHWC : TensorLayout::NCHW);
    else
        layouts = {TensorLayout::NCHW, TensorLayout::NHWC};

    for(const auto& dt : dtypes)
    {
        for(const auto& layout : layouts)
        {
            if(dt == "fp32")
                allPassed &= f.template operator()<float, float>(layout);
            else if(dt == "fp16")
                allPassed &= f.template operator()<half, float>(layout);
            else if(dt == "bf16")
                allPassed &= f.template operator()<bfloat16, float>(layout);
        }
    }

    return allPassed;
}

// TENSOR HELPERS

inline std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>
    createTensor(const std::vector<int64_t>& dims,
                 hipdnn_frontend::DataType_t dataType,
                 const TensorLayout& layout = TensorLayout::NCHW)
{
    auto tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    tensor->set_dim(dims).set_data_type(dataType).set_stride(
        hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder));
    return tensor;
}

inline int64_t
    getTensorElementCount(const std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>& tensor)
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
