// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

using hipdnn_sdk::utilities::TensorLayout;

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

inline void printSampleHelp(const std::string& sampleName)
{
    std::cout << "Usage: " << sampleName << " [OPTIONS]\n"
              << "Options:\n"
              << "  --verify-cpu, -vc    Enable CPU reference validation\n"
              << "  --help, -h      Show this help message\n"
              << std::endl;
}

struct Config
{
    bool cpuValidation = false;
};

inline Config parseCommandLineArgs(int argc, char* argv[])
{
    auto config = Config{};

    for(int i = 1; i < argc; ++i)
    {
        auto arg = std::string(argv[i]);

        if(arg == "--verify-cpu" || arg == "-vc")
        {
            config.cpuValidation = true;
        }
        else if(arg == "--help" || arg == "-h")
        {
            printSampleHelp(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printSampleHelp(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return config;
}

template <typename F>
void run(F&& f)
{
    f.template operator()<float, float>(TensorLayout::NCHW);
    f.template operator()<half, float>(TensorLayout::NCHW);
    f.template operator()<hip_bfloat16, float>(TensorLayout::NCHW);
    f.template operator()<float, float>(TensorLayout::NHWC);
    f.template operator()<half, float>(TensorLayout::NHWC);
    f.template operator()<hip_bfloat16, float>(TensorLayout::NHWC);
}

inline std::shared_ptr<hipdnn_frontend::graph::Tensor_attributes>
    createTensor(const std::vector<int64_t>& dims,
                 hipdnn_frontend::DataType_t dataType,
                 const TensorLayout& layout = TensorLayout::NCHW)
{
    auto tensor = std::make_shared<hipdnn_frontend::graph::Tensor_attributes>();
    tensor->set_dim(dims).set_data_type(dataType);
    tensor->set_stride(hipdnn_sdk::utilities::generateStrides(dims, layout.strideOrder));

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
    void operator()(const TensorLayout& layout);
};
