// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Stage-4 compiler: AMDGPU LLVM IR text (.ll) -> HSACO, in-process via
// libamd_comgr. This is the C++ twin of ck_dsl/runtime/comgr.py and is
// byte-reproducible with it (verified: same .ll + isa + "-O3" -> identical
// HSACO bytes). No subprocess, no clang/hipcc, no Python.
#pragma once

#include <amd_comgr/amd_comgr.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ck_dsl {

class ComgrError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

struct Compiler {
    // Compile LLVM IR text to HSACO bytes for `isa`
    // (e.g. "amdgcn-amd-amdhsa--gfx950"). `options` defaults to {"-O3"} to
    // match the ck_dsl Python path.
    static std::vector<std::byte> compile(const std::string& llvm_ir, const std::string& isa,
                                          const std::vector<std::string>& options = {"-O3"}) {
        auto check = [](amd_comgr_status_t s, const char* where) {
            if (s != AMD_COMGR_STATUS_SUCCESS) {
                const char* msg = nullptr;
                amd_comgr_status_string(s, &msg);
                throw ComgrError(std::string(where) + ": status=" + std::to_string(s) + " (" +
                                 (msg ? msg : "") + ")");
            }
        };

        amd_comgr_data_set_t in_set{};
        check(amd_comgr_create_data_set(&in_set), "create_data_set(in)");
        amd_comgr_data_t src{};
        check(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &src), "create_data(src)");
        check(amd_comgr_set_data(src, llvm_ir.size(), llvm_ir.data()), "set_data(src)");
        check(amd_comgr_set_data_name(src, "kernel.ll"), "set_data_name(src)");
        check(amd_comgr_data_set_add(in_set, src), "data_set_add(src)");

        amd_comgr_action_info_t info{};
        check(amd_comgr_create_action_info(&info), "create_action_info");
        check(amd_comgr_action_info_set_isa_name(info, isa.c_str()), "set_isa");
        check(amd_comgr_action_info_set_language(info, AMD_COMGR_LANGUAGE_LLVM_IR), "set_lang");
        std::vector<const char*> opt_ptrs;
        opt_ptrs.reserve(options.size());
        for (const auto& o : options) opt_ptrs.push_back(o.c_str());
        check(amd_comgr_action_info_set_option_list(info, opt_ptrs.data(), opt_ptrs.size()),
              "set_options");

        amd_comgr_data_set_t bc_set{}, rel_set{}, exe_set{};
        check(amd_comgr_create_data_set(&bc_set), "create_data_set(bc)");
        check(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, in_set, bc_set),
              "COMPILE_SOURCE_TO_BC");
        check(amd_comgr_create_data_set(&rel_set), "create_data_set(reloc)");
        check(
            amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc_set, rel_set),
            "CODEGEN_BC_TO_RELOCATABLE");
        check(amd_comgr_create_data_set(&exe_set), "create_data_set(exe)");
        check(amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, rel_set,
                                  exe_set),
              "LINK_RELOCATABLE_TO_EXECUTABLE");

        size_t count = 0;
        check(amd_comgr_action_data_count(exe_set, AMD_COMGR_DATA_KIND_EXECUTABLE, &count),
              "action_data_count");
        if (count == 0) throw ComgrError("comgr produced no EXECUTABLE");
        amd_comgr_data_t exe{};
        check(amd_comgr_action_data_get_data(exe_set, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &exe),
              "action_data_get_data");
        size_t sz = 0;
        check(amd_comgr_get_data(exe, &sz, nullptr), "get_data(size)");
        std::vector<std::byte> hsaco(sz);
        check(amd_comgr_get_data(exe, &sz, reinterpret_cast<char*>(hsaco.data())),
              "get_data(read)");

        amd_comgr_release_data(exe);
        amd_comgr_release_data(src);
        amd_comgr_destroy_data_set(in_set);
        amd_comgr_destroy_data_set(bc_set);
        amd_comgr_destroy_data_set(rel_set);
        amd_comgr_destroy_data_set(exe_set);
        amd_comgr_destroy_action_info(info);
        return hsaco;
    }

    // Convenience: form the comgr ISA string from a gfx arch (e.g. "gfx950").
    static std::string isa_for(const std::string& gfx) {
        return "amdgcn-amd-amdhsa--" + gfx;
    }
};

}  // namespace ck_dsl
