// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// LLVM IR text (.ll) -> HSACO, in-process via libamd_comgr.
// Adapted from ck_dsl_runtime/comgr.hpp for the rocKE conv sweep tool.
#pragma once

#include <amd_comgr/amd_comgr.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocke
{

class ComgrError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct Compiler
{
    static std::vector<std::byte> compile(const std::string& llvm_ir,
                                          const std::string& isa,
                                          const std::vector<std::string>& options = {"-O3"})
    {
        auto check = [](amd_comgr_status_t s, const char* where) {
            if(s != AMD_COMGR_STATUS_SUCCESS)
            {
                const char* msg = nullptr;
                amd_comgr_status_string(s, &msg);
                throw ComgrError(std::string(where) + ": status=" + std::to_string(s) + " ("
                                 + (msg ? msg : "") + ")");
            }
        };

        struct ComgrGuard
        {
            amd_comgr_data_set_t in_set{}, bc_set{}, rel_set{}, exe_set{};
            amd_comgr_action_info_t info{};
            amd_comgr_data_t src{}, exe{};
            bool has_in = false, has_bc = false, has_rel = false, has_exe = false;
            bool has_info = false, has_src = false, has_exe_data = false;
            ~ComgrGuard()
            {
                if(has_exe_data)
                    amd_comgr_release_data(exe);
                if(has_src)
                    amd_comgr_release_data(src);
                if(has_in)
                    amd_comgr_destroy_data_set(in_set);
                if(has_bc)
                    amd_comgr_destroy_data_set(bc_set);
                if(has_rel)
                    amd_comgr_destroy_data_set(rel_set);
                if(has_exe)
                    amd_comgr_destroy_data_set(exe_set);
                if(has_info)
                    amd_comgr_destroy_action_info(info);
            }
        } g;

        check(amd_comgr_create_data_set(&g.in_set), "create_data_set(in)");
        g.has_in = true;
        check(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &g.src), "create_data(src)");
        g.has_src = true;
        check(amd_comgr_set_data(g.src, llvm_ir.size(), llvm_ir.data()), "set_data(src)");
        check(amd_comgr_set_data_name(g.src, "kernel.ll"), "set_data_name(src)");
        check(amd_comgr_data_set_add(g.in_set, g.src), "data_set_add(src)");

        check(amd_comgr_create_action_info(&g.info), "create_action_info");
        g.has_info = true;
        check(amd_comgr_action_info_set_isa_name(g.info, isa.c_str()), "set_isa");
        check(amd_comgr_action_info_set_language(g.info, AMD_COMGR_LANGUAGE_LLVM_IR), "set_lang");
        std::vector<const char*> opt_ptrs;
        opt_ptrs.reserve(options.size());
        for(const auto& o : options)
            opt_ptrs.push_back(o.c_str());
        check(amd_comgr_action_info_set_option_list(g.info, opt_ptrs.data(), opt_ptrs.size()),
              "set_options");

        check(amd_comgr_create_data_set(&g.bc_set), "create_data_set(bc)");
        g.has_bc = true;
        check(
            amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, g.info, g.in_set, g.bc_set),
            "COMPILE_SOURCE_TO_BC");
        check(amd_comgr_create_data_set(&g.rel_set), "create_data_set(reloc)");
        g.has_rel = true;
        check(amd_comgr_do_action(
                  AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, g.info, g.bc_set, g.rel_set),
              "CODEGEN_BC_TO_RELOCATABLE");
        check(amd_comgr_create_data_set(&g.exe_set), "create_data_set(exe)");
        g.has_exe = true;
        check(amd_comgr_do_action(
                  AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, g.info, g.rel_set, g.exe_set),
              "LINK_RELOCATABLE_TO_EXECUTABLE");

        size_t count = 0;
        check(amd_comgr_action_data_count(g.exe_set, AMD_COMGR_DATA_KIND_EXECUTABLE, &count),
              "action_data_count");
        if(count == 0)
            throw ComgrError("comgr produced no EXECUTABLE");
        check(amd_comgr_action_data_get_data(g.exe_set, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &g.exe),
              "action_data_get_data");
        g.has_exe_data = true;
        size_t sz = 0;
        check(amd_comgr_get_data(g.exe, &sz, nullptr), "get_data(size)");
        std::vector<std::byte> hsaco(sz);
        check(amd_comgr_get_data(g.exe, &sz, reinterpret_cast<char*>(hsaco.data())),
              "get_data(read)");
        return hsaco;
    }

    static std::string isa_for(const std::string& gfx)
    {
        return "amdgcn-amd-amdhsa--" + gfx;
    }
};

} // namespace rocke
