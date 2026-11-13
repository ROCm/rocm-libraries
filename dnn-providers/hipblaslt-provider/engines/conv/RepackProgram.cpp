// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/conv/RepackProgram.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include <hip/hiprtc.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "kernels/ConvRepackLaunch.hpp"
#include "kernels/ConvRepackSources.hpp"

namespace hipblaslt_plugin
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr unsigned int BLOCK_SIZE = 256;
/// Grid-stride loops cover the rest; a cap keeps the block count a 32-bit launch dimension.
constexpr long long MAX_BLOCKS = 1LL << 20;

constexpr std::array<const char*, 65> ENTRY_POINTS = {"StridedCopy2",
                                                      "StridedCopy4",
                                                      "StridedTranspose2",
                                                      "StridedTranspose4",
                                                      "StridedTransposeNarrow2",
                                                      "StridedTransposeNarrow4",
                                                      "StridedCastFloatToFloat",
                                                      "StridedCastFloatToHalf",
                                                      "StridedCastFloatToBfloat16",
                                                      "StridedCastWideFloatToFloat",
                                                      "StridedCastWideFloatToHalf",
                                                      "StridedCastWideFloatToBfloat16",
                                                      "Im2Col2",
                                                      "Im2Col4",
                                                      "Im2ColDisjointOutputs2V1",
                                                      "Im2ColDisjointOutputs2V2",
                                                      "Im2ColDisjointOutputs2V4",
                                                      "Im2ColDisjointOutputs2V8",
                                                      "Im2ColDisjointOutputs4V1",
                                                      "Im2ColDisjointOutputs4V2",
                                                      "Im2ColDisjointOutputs4V4",
                                                      "Im2ColRows2V1",
                                                      "Im2ColRows2V8",
                                                      "Im2ColRows4V1",
                                                      "Im2ColRows4V4",
                                                      "Im2ColTaps2",
                                                      "Im2ColTaps4",
                                                      "Im2ColTapsStaged2",
                                                      "Im2ColTapsStaged4",
                                                      "Im2ColTile2",
                                                      "Im2ColTile4",
                                                      "Col2ImFloat",
                                                      "Col2ImDisjoint2",
                                                      "Col2ImDisjoint4",
                                                      "Col2ImDisjointWidth2V1",
                                                      "Col2ImDisjointWidth2V2",
                                                      "Col2ImDisjointWidth2V4",
                                                      "Col2ImDisjointWidth2V8",
                                                      "Col2ImDisjointChannels2V1",
                                                      "Col2ImDisjointChannels2V2",
                                                      "Col2ImDisjointChannels2V4",
                                                      "Col2ImDisjointChannels2V8",
                                                      "Col2ImDisjointWidth4V1",
                                                      "Col2ImDisjointWidth4V2",
                                                      "Col2ImDisjointWidth4V4",
                                                      "Col2ImDisjointChannels4V1",
                                                      "Col2ImDisjointChannels4V2",
                                                      "Col2ImDisjointChannels4V4",
                                                      "Col2ImHalf",
                                                      "Col2ImBfloat16",
                                                      "Col2ImRowsFloatV1",
                                                      "Col2ImRowsFloatV4",
                                                      "Col2ImRowsHalfV1",
                                                      "Col2ImRowsHalfV4",
                                                      "Col2ImRowsBfloat16V1",
                                                      "Col2ImRowsBfloat16V4",
                                                      "Col2ImGroupsFloatV1",
                                                      "Col2ImGroupsFloatV4",
                                                      "Col2ImGroupsHalfV1",
                                                      "Col2ImGroupsHalfV4",
                                                      "Col2ImGroupsBfloat16V1",
                                                      "Col2ImGroupsBfloat16V4",
                                                      "Col2ImChannelsFirstFloat",
                                                      "Col2ImChannelsFirstHalf",
                                                      "Col2ImChannelsFirstBfloat16"};

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
    const std::array<const char*, 2> headers
        = {kernels::CONV_REPACK_ARGS_SOURCE, kernels::CONV_REPACK_DISJOINT_SOURCE};
    const std::array<const char*, 2> headerNames = {"ConvRepackArgs.h", "ConvRepackDisjoint.h"};
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
    const std::string prefix
        = std::holds_alternative<StridedCastArgs>(step.args) ? "StridedCastFloatTo" : "Col2Im";
    switch(step.dataType)
    {
    case data_objects::DataType::FLOAT:
        // A cast to fp32 still sums batches (the batched backward weights' partials).
        return prefix + "Float";
    case data_objects::DataType::HALF:
        return prefix + "Half";
    case data_objects::DataType::BFLOAT16:
        return prefix + "Bfloat16";
    default:
        break;
    }
    return "";
}

void RepackProgram::launch(const RepackStep& step,
                           const void* src,
                           void* dst,
                           hipStream_t stream,
                           long long minRunThreads) const
{
    // The kernels take their argument block by value; fill in its pointers, then let the launch
    // rules (kernels/ConvRepackLaunch.hpp) pick im2col's and col2im's specialized kernels.
    auto args = step.args;
    long long total = 0;
    RepackLaunch launch;
    if(auto* copy = std::get_if<StridedCopyArgs>(&args))
    {
        copy->src = src;
        copy->dst = dst;
        total = copy->total;
        launch = stridedCopyLaunch(*copy, static_cast<int>(elementBytes(step.dataType)));
    }
    else if(auto* cast = std::get_if<StridedCastArgs>(&args))
    {
        cast->src = static_cast<const float*>(src);
        cast->dst = dst;
        total = cast->total;
        // Many partials into few outputs: one 64-thread block per output (castLaunch).
        launch = castLaunch(*cast, entryName(step));
    }
    else if(auto* im2col = std::get_if<Im2ColArgs>(&args))
    {
        im2col->x = src;
        im2col->col = dst;
        total = im2col->total;
        launch
            = im2colLaunch(*im2col, static_cast<int>(elementBytes(step.dataType)), minRunThreads);
    }
    else if(auto* col2im = std::get_if<Col2ImArgs>(&args))
    {
        col2im->col = static_cast<const float*>(src);
        col2im->x = dst;
        total = col2im->total;
        const auto generic = entryName(step); // Col2Im<Type>
        launch = col2imLaunch(*col2im,
                              generic.substr(std::string("Col2Im").size()),
                              static_cast<int>(elementBytes(step.dataType)),
                              minRunThreads);
    }
    if(total <= 0)
    {
        return;
    }
    if(launch.entry.empty())
    {
        launch.entry = entryName(step);
        launch.grid = static_cast<unsigned int>(
            std::min((total + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS));
        launch.block = BLOCK_SIZE;
    }
    const auto found = _functions.find(launch.entry);
    if(found == _functions.end())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "no repack kernel named " + launch.entry);
    }

    // Each kernel takes one parameter, the argument block.
    void* argBlock = std::visit([](auto& block) -> void* { return &block; }, args);
    std::array<void*, 1> params = {argBlock};
    throwOnHipError(hipModuleLaunchKernel(found->second,
                                          launch.grid,
                                          1,
                                          1,
                                          launch.block,
                                          1,
                                          1,
                                          static_cast<unsigned int>(launch.ldsBytes),
                                          stream,
                                          params.data(),
                                          nullptr),
                    "hipModuleLaunchKernel");
}

} // namespace hipblaslt_plugin
