// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/conv/RepackProgram.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include <hip/hiprtc.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "kernels/ConvRepackSources.hpp"

namespace hipblaslt_plugin
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr unsigned int BLOCK_SIZE = 256;
/// Grid-stride loops cover the rest; a cap keeps the block count a 32-bit launch dimension.
constexpr long long MAX_BLOCKS = 1LL << 20;

constexpr std::array<const char*, 7> ENTRY_POINTS = {"StridedCopy2",
                                                     "StridedCopy4",
                                                     "Im2Col2",
                                                     "Im2Col4",
                                                     "Col2ImFloat",
                                                     "Col2ImHalf",
                                                     "Col2ImBfloat16"};

void throwOnHipError(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string(what) + " failed: " + hipGetErrorString(status));
    }
}

void throwOnHiprtcError(hiprtcResult status, const char* what)
{
    if(status != HIPRTC_SUCCESS)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string(what) + " failed: " + hiprtcGetErrorString(status));
    }
}

/// The code object of the repack program for @p arch.
std::vector<char> compile(const std::string& arch)
{
    const std::array<const char*, 1> headers = {kernels::CONV_REPACK_ARGS_SOURCE};
    const std::array<const char*, 1> headerNames = {"ConvRepackArgs.h"};
    hiprtcProgram program = nullptr;
    throwOnHiprtcError(hiprtcCreateProgram(&program,
                                           kernels::CONV_REPACK_SOURCE,
                                           "ConvRepack.cpp",
                                           static_cast<int>(headers.size()),
                                           headers.data(),
                                           headerNames.data()),
                       "hiprtcCreateProgram");

    const std::string offloadArch = "--offload-arch=" + arch;
    const std::array<const char*, 3> options = {"-std=c++17", "-O3", offloadArch.c_str()};
    const auto result
        = hiprtcCompileProgram(program, static_cast<int>(options.size()), options.data());
    if(result != HIPRTC_SUCCESS)
    {
        size_t logSize = 0;
        std::string log;
        if(hiprtcGetProgramLogSize(program, &logSize) == HIPRTC_SUCCESS && logSize > 1)
        {
            log.resize(logSize);
            if(hiprtcGetProgramLog(program, log.data()) != HIPRTC_SUCCESS)
            {
                log.clear();
            }
        }
        hiprtcDestroyProgram(&program);
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "HIPRTC failed to compile the convolution repack kernels for " + arch + ": "
                + hiprtcGetErrorString(result) + "\n" + log);
    }

    size_t codeSize = 0;
    std::vector<char> code;
    auto status = hiprtcGetCodeSize(program, &codeSize);
    if(status == HIPRTC_SUCCESS)
    {
        code.resize(codeSize);
        status = hiprtcGetCode(program, code.data());
    }
    hiprtcDestroyProgram(&program);
    throwOnHiprtcError(status, "hiprtcGetCode");
    return code;
}

} // namespace

RepackProgram::RepackProgram()
{
    int device = 0;
    throwOnHipError(hipGetDevice(&device), "hipGetDevice");
    hipDeviceProp_t properties{};
    throwOnHipError(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
    const std::string arch = properties.gcnArchName;
    HIPDNN_PLUGIN_LOG_INFO("Compiling the convolution repack kernels for " << arch);

    const auto code = compile(arch);
    throwOnHipError(hipModuleLoadData(&_module, code.data()), "hipModuleLoadData");
    for(const char* entry : ENTRY_POINTS)
    {
        hipFunction_t function = nullptr;
        const auto status = hipModuleGetFunction(&function, _module, entry);
        if(status != hipSuccess)
        {
            static_cast<void>(hipModuleUnload(_module));
            throwOnHipError(status, "hipModuleGetFunction");
        }
        _functions.emplace(entry, function);
    }
}

RepackProgram::~RepackProgram()
{
    if(_module != nullptr)
    {
        const auto status = hipModuleUnload(_module);
        if(status != hipSuccess)
        {
            HIPDNN_PLUGIN_LOG_WARN("hipModuleUnload failed: " << hipGetErrorString(status));
        }
    }
}

std::string RepackProgram::entryName(const RepackStep& step)
{
    const std::string bytes = std::to_string(elementBytes(step.dataType));
    if(std::holds_alternative<StridedCopyArgs>(step.args))
    {
        return "StridedCopy" + bytes;
    }
    if(std::holds_alternative<Im2ColArgs>(step.args))
    {
        return "Im2Col" + bytes;
    }
    switch(step.dataType)
    {
    case data_objects::DataType::FLOAT:
        return "Col2ImFloat";
    case data_objects::DataType::HALF:
        return "Col2ImHalf";
    case data_objects::DataType::BFLOAT16:
        return "Col2ImBfloat16";
    default:
        break;
    }
    return "";
}

void RepackProgram::launch(const RepackStep& step,
                           const void* src,
                           void* dst,
                           hipStream_t stream) const
{
    const auto found = _functions.find(entryName(step));
    if(found == _functions.end())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "no repack kernel for this step");
    }

    // The kernels take their argument block by value; fill in its pointers.
    auto args = step.args;
    long long total = 0;
    if(auto* copy = std::get_if<StridedCopyArgs>(&args))
    {
        copy->src = src;
        copy->dst = dst;
        total = copy->total;
    }
    else if(auto* im2col = std::get_if<Im2ColArgs>(&args))
    {
        im2col->x = src;
        im2col->col = dst;
        total = im2col->total;
    }
    else if(auto* col2im = std::get_if<Col2ImArgs>(&args))
    {
        col2im->col = static_cast<const float*>(src);
        col2im->x = dst;
        total = col2im->total;
    }
    if(total <= 0)
    {
        return;
    }

    // Each kernel takes one parameter, the argument block.
    void* argBlock = std::visit([](auto& block) -> void* { return &block; }, args);
    std::array<void*, 1> params = {argBlock};
    const auto blocks = std::min((total + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
    throwOnHipError(hipModuleLaunchKernel(found->second,
                                          static_cast<unsigned int>(blocks),
                                          1,
                                          1,
                                          BLOCK_SIZE,
                                          1,
                                          1,
                                          0,
                                          stream,
                                          params.data(),
                                          nullptr),
                    "hipModuleLaunchKernel");
}

} // namespace hipblaslt_plugin
