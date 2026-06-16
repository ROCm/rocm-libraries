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

// SAMPLE TYPES

enum class SampleType
{
    GENERIC,
    BN_TRAINING
};

// HELP MESSAGE

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

    std::cout << "  --help, -h                  Show this help message\n\n";
}

// CONFIG

struct Config
{
    bool cpuValidation = false;
    bool useRunningStats = false;

    int engine_id = -1;
    std::string dtype;
    std::string layout;
    std::string engine_name;

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
        result.push_back(std::stoll(item));

    return result;
}

// CLI PARSER

inline Config
    parseCommandLineArgs(int argc, char** argv, SampleType sampleType = SampleType::GENERIC)
{
    Config config;

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
        else if(arg == "--engine-id")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--engine-id requires a value\n";
                exit(EXIT_FAILURE);
            }
            config.engine_id = std::stoi(argv[++i]);
        }
        else if(arg == "--engine-name")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--engine-name requires a value\n";
                exit(EXIT_FAILURE);
            }
            config.engine_name = argv[++i];
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

            if(config.layout != "nchw" && config.layout != "nhwc")
            {
                std::cerr << "Invalid value for --layout: " << config.layout
                          << " (expected: nchw, nhwc)\n";
                exit(EXIT_FAILURE);
            }
        }
        else if(arg == "--dims")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--dims requires a value\n";
                exit(EXIT_FAILURE);
            }

            config.dims = parseList(argv[++i]);

            if(config.dims.size() != 4)
            {
                std::cerr << "--dims must contain 4 values (N,C,H,W)\n";
                exit(EXIT_FAILURE);
            }
        }
        else if(arg == "--filter")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--filter requires a value\n";
                exit(EXIT_FAILURE);
            }
            config.filter = parseList(argv[++i]);
        }
        else if(arg == "--stride")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--stride requires a value\n";
                exit(EXIT_FAILURE);
            }
            config.stride = parseList(argv[++i]);
        }
        else if(arg == "--padding")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--padding requires a value\n";
                exit(EXIT_FAILURE);
            }
            config.padding = parseList(argv[++i]);
        }
        else if(arg == "--dilation")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "--dilation requires a value\n";
                exit(EXIT_FAILURE);
            }
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

    // Prevent conflicting options
    if(config.engine_id != -1 && !config.engine_name.empty())
    {
        std::cerr << "Specify either --engine-id or --engine-name, not both\n";
        exit(EXIT_FAILURE);
    }

    return config;
}

template <typename F>
bool run(F&& f)
{
    bool allPassed = true;

    std::vector<std::string> dtypes = {"fp32", "fp16", "bf16"};
    std::vector<TensorLayout> layouts = {TensorLayout::NCHW, TensorLayout::NHWC};

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
        count *= dim;

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
