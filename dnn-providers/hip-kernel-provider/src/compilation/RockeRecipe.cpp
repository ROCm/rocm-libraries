// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "compilation/RockeRecipe.hpp"
#include <amd_comgr.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <iostream>
#include <rocke/online.h>
#include <rocke/recipe_guard.h>
#include <stdexcept>

namespace hip_kernel_provider::compilation
{
namespace
{
using Bytes = RecipeBytes;
void require(bool condition, const std::string& message)
{
    if(!condition)
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, message);
}
void hip_check(hipError_t status)
{
    require(status == hipSuccess, hipGetErrorString(status));
}
void comgr_check(amd_comgr_status_t status)
{
    const char* message = nullptr;
    amd_comgr_status_string(status, &message);
    require(status == AMD_COMGR_STATUS_SUCCESS,
            std::string("COMGR: ") + (message ? message : "unknown"));
}
struct DataSet
{
    amd_comgr_data_set_t handle{};
    DataSet()
    {
        comgr_check(amd_comgr_create_data_set(&handle));
    }
    ~DataSet()
    {
        amd_comgr_destroy_data_set(handle);
    }
    DataSet(const DataSet&) = delete;
    DataSet& operator=(const DataSet&) = delete;
};
struct ActionInfo
{
    amd_comgr_action_info_t handle{};
    ActionInfo()
    {
        comgr_check(amd_comgr_create_action_info(&handle));
    }
    ~ActionInfo()
    {
        amd_comgr_destroy_action_info(handle);
    }
};
struct Data
{
    amd_comgr_data_t handle{};
    ~Data()
    {
        if(handle.handle)
            amd_comgr_release_data(handle);
    }
};
static void action(amd_comgr_action_kind_t kind, ActionInfo& info, DataSet& input, DataSet& output)
{
    auto status = amd_comgr_do_action(kind, info.handle, input.handle, output.handle);
    if(status != AMD_COMGR_STATUS_SUCCESS)
    {
        size_t count = 0;
        amd_comgr_action_data_count(output.handle, AMD_COMGR_DATA_KIND_LOG, &count);
        for(size_t i = 0; i < count; ++i)
        {
            Data log;
            size_t size = 0;
            if(amd_comgr_action_data_get_data(
                   output.handle, AMD_COMGR_DATA_KIND_LOG, i, &log.handle)
               != AMD_COMGR_STATUS_SUCCESS)
                continue;
            amd_comgr_get_data(log.handle, &size, nullptr);
            std::string text(size, '\0');
            amd_comgr_get_data(log.handle, &size, text.data());
            std::cerr << text << '\n';
        }
    }
    comgr_check(status);
}
static Bytes compile(const std::string& llvm, const std::string& target)
{
    DataSet input, bc, reloc, executable;
    ActionInfo info;
    Data source;
    comgr_check(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &source.handle));
    comgr_check(amd_comgr_set_data(source.handle, llvm.size(), llvm.data()));
    comgr_check(amd_comgr_set_data_name(source.handle, "kernel.ll"));
    comgr_check(amd_comgr_data_set_add(input.handle, source.handle));
    comgr_check(
        amd_comgr_action_info_set_isa_name(info.handle, ("amdgcn-amd-amdhsa--" + target).c_str()));
    comgr_check(amd_comgr_action_info_set_language(info.handle, AMD_COMGR_LANGUAGE_LLVM_IR));
    comgr_check(amd_comgr_action_info_set_logging(info.handle, true));
    const char* options[] = {"-O3"};
    comgr_check(amd_comgr_action_info_set_option_list(info.handle, options, 1));
    action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, input, bc);
    action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc, reloc);
    action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, reloc, executable);
    size_t count = 0;
    comgr_check(
        amd_comgr_action_data_count(executable.handle, AMD_COMGR_DATA_KIND_EXECUTABLE, &count));
    require(count == 1, "expected one executable");
    Data code;
    comgr_check(amd_comgr_action_data_get_data(
        executable.handle, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &code.handle));
    size_t size = 0;
    comgr_check(amd_comgr_get_data(code.handle, &size, nullptr));
    Bytes bytes(size);
    comgr_check(amd_comgr_get_data(code.handle, &size, reinterpret_cast<char*>(bytes.data())));
    return bytes;
}

} // namespace

bool recipeAdmits(const RecipeBytes& bytes,
                  const std::string& key,
                  const std::string& arch,
                  int64_t sequence)
{
    rocke_recipe_spec_int_t input{"S", sequence};
    rocke_guard_verdict_t verdict = ROCKE_GUARD_ABSENT;
    char error[1024]{};
    auto status = rocke_bundle_check_guard_cbor(bytes.data(),
                                                bytes.size(),
                                                key.c_str(),
                                                arch.substr(0, arch.find(':')).c_str(),
                                                &input,
                                                1,
                                                nullptr,
                                                0,
                                                0,
                                                &verdict,
                                                error,
                                                sizeof(error));
    if(status == ROCKE_ERR_KEY)
        return false;
    require(status == ROCKE_OK, std::string("recipe admission: ") + error);
    return verdict == ROCKE_GUARD_ADMITTED;
}

RockeRecipe::RockeRecipe(const RecipeBytes& bytes,
                         const std::string& key,
                         const std::string& arch,
                         int64_t sequence)
{
    require(recipeAdmits(bytes, key, arch, sequence), "recipe refused configuration");
    rocke_recipe_spec_int_t input{"S", sequence};
    char error[1024]{};
    rocke_launch_plan_t* plan = nullptr;
    auto status = rocke_bundle_plan_launch_cbor(bytes.data(),
                                                bytes.size(),
                                                key.c_str(),
                                                arch.substr(0, arch.find(':')).c_str(),
                                                &input,
                                                1,
                                                nullptr,
                                                0,
                                                &plan,
                                                error,
                                                sizeof(error));
    _plan.reset(plan);
    require(status == ROCKE_OK && plan, std::string("recipe launch plan: ") + error);
    require(rocke_launch_plan_geometry(plan, &_grid, &_block, &_lds),
            "recipe has no launch geometry");
    char* raw = nullptr;
    status = rocke_online_bundle_cbor_to_llvm(bytes.data(),
                                              bytes.size(),
                                              key.c_str(),
                                              arch.substr(0, arch.find(':')).c_str(),
                                              &input,
                                              1,
                                              nullptr,
                                              0,
                                              &raw,
                                              nullptr,
                                              nullptr,
                                              error,
                                              sizeof(error));
    std::unique_ptr<char, decltype(&rocke_online_free)> llvm(raw, rocke_online_free);
    require(status == ROCKE_OK && raw, std::string("recipe IR generation: ") + error);
    auto code = compile(raw, arch);
    hip_check(hipModuleLoadData(&_module, code.data()));
    auto hipStatus = hipModuleGetFunction(&_function, _module, rocke_launch_plan_kernel_name(plan));
    if(hipStatus != hipSuccess)
    {
        static_cast<void>(hipModuleUnload(_module));
        _module = nullptr;
        hip_check(hipStatus);
    }
    HIPDNN_PLUGIN_LOG_INFO("rocKE recipe compiled: key=" << key << " S=" << sequence);
}
RockeRecipe::~RockeRecipe()
{
    if(_module != nullptr)
        static_cast<void>(hipModuleUnload(_module));
}
void RockeRecipe::launch(void** extra, hipStream_t stream) const
{
    hip_check(hipModuleLaunchKernel(_function,
                                    _grid.x,
                                    _grid.y,
                                    _grid.z,
                                    _block.x,
                                    _block.y,
                                    _block.z,
                                    _lds,
                                    stream,
                                    nullptr,
                                    extra));
}
} // namespace hip_kernel_provider::compilation
