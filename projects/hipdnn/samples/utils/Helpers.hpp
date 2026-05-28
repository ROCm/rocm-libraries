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
#include <vector>

using hipdnn_data_sdk::utilities::TensorLayout;

// Use portable custom types instead of HIP types (works with any C++ compiler)
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

#define HIP_CHECK(status)                                                                      \
    do                                                                                         \
    {                                                                                          \
        if(status != hipSuccess)                                                               \
        {                                                                                      \
            std::cerr << "HIP Error: " << hipGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                 \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while(0)

#define HIPDNN_CHECK(status)                                                             \
    do                                                                                   \
    {                                                                                    \
        if(status != HIPDNN_STATUS_SUCCESS)                                              \
        {                                                                                \
            std::cerr << "hipDNN Error: " << hipdnnGetErrorString(status) << " in file " \
                      << __FILE__ << " at line " << __LINE__ << std::endl;               \
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
                      << __FILE__ << " at line " << __LINE__ << std::endl;                \
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
                      << __FILE__ << " at line " << __LINE__ << std::endl;                      \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while(0)

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
              << "  --verify-cpu, -vc           Enable CPU reference validation\n";

    if(sampleType == SampleType::BN_TRAINING)
    {
        std::cout << "  --batch-stats-only          Use batch statistics only (no running stats)\n"
                  << "  --full-training             Use full training with running statistics\n";
    }

    std::cout << "  --help, -h                  Show this help message\n" << std::endl;
}

struct Config
{
    bool cpuValidation = false;
    bool useRunningStats = false;
};

inline Config
    parseCommandLineArgs(int argc, char* argv[], SampleType sampleType = SampleType::GENERIC)
{
    auto config = Config{};

    for(int i = 1; i < argc; ++i)
    {
        auto arg = std::string(argv[i]);

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
        else if(arg == "--help" || arg == "-h")
        {
            printSampleHelp(argv[0], sampleType);
            exit(EXIT_SUCCESS);
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printSampleHelp(argv[0], sampleType);
            exit(EXIT_FAILURE);
        }
    }

    return config;
}

template <typename F>
bool run(F&& f)
{
    bool allPassed = true;
    allPassed &= f.template operator()<float, float>(TensorLayout::NCHW);
    allPassed &= f.template operator()<half, float>(TensorLayout::NCHW);
    allPassed &= f.template operator()<bfloat16, float>(TensorLayout::NCHW);
    allPassed &= f.template operator()<float, float>(TensorLayout::NHWC);
    allPassed &= f.template operator()<half, float>(TensorLayout::NHWC);
    allPassed &= f.template operator()<bfloat16, float>(TensorLayout::NHWC);
    return allPassed;
}

inline std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>
    createTensor(const std::vector<int64_t>& dims,
                 hipdnn_frontend::DataType_t dataType,
                 const TensorLayout& layout = TensorLayout::NCHW)
{
    auto tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    tensor->set_dim(dims).set_data_type(dataType);
    tensor->set_stride(hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder));

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

struct SampleRunner
{
    hipdnnHandle_t handle;
    Config config;

    template <typename InputType, typename IntermediateType>
    bool operator()(const TensorLayout& layout);
};
