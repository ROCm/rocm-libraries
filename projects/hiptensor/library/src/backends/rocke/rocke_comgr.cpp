/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <cstdio>
#include <cstring>
#include <mutex>

#include <amd_comgr/amd_comgr.h>

#include "rocke_comgr.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        namespace
        {
            std::mutex& comgrMutex()
            {
                static std::mutex mutex;
                return mutex;
            }

            // Releases COMGR handles at scope exit.
            struct DataSet
            {
                amd_comgr_data_set_t set{0};
                bool                 created = false;
                ~DataSet()
                {
                    if(created)
                    {
                        amd_comgr_destroy_data_set(set);
                    }
                }
                bool create()
                {
                    created = amd_comgr_create_data_set(&set) == AMD_COMGR_STATUS_SUCCESS;
                    return created;
                }
            };

            struct Data
            {
                amd_comgr_data_t data{0};
                bool             created = false;
                ~Data()
                {
                    if(created)
                    {
                        amd_comgr_release_data(data);
                    }
                }
            };

            struct ActionInfo
            {
                amd_comgr_action_info_t info{0};
                bool                    created = false;
                ~ActionInfo()
                {
                    if(created)
                    {
                        amd_comgr_destroy_action_info(info);
                    }
                }
                bool create()
                {
                    created = amd_comgr_create_action_info(&info) == AMD_COMGR_STATUS_SUCCESS;
                    return created;
                }
            };

            // The bytes of the first data object of `kind` in `set`.
            bool firstData(amd_comgr_data_set_t const& set,
                           amd_comgr_data_kind_t       kind,
                           std::vector<uint8_t>&       out)
            {
                size_t count = 0;
                if(amd_comgr_action_data_count(set, kind, &count) != AMD_COMGR_STATUS_SUCCESS
                   || count == 0)
                {
                    return false;
                }
                Data data;
                if(amd_comgr_action_data_get_data(set, kind, 0, &data.data)
                   != AMD_COMGR_STATUS_SUCCESS)
                {
                    return false;
                }
                data.created = true;
                size_t size  = 0;
                if(amd_comgr_get_data(data.data, &size, nullptr) != AMD_COMGR_STATUS_SUCCESS)
                {
                    return false;
                }
                out.resize(size);
                return amd_comgr_get_data(data.data, &size, reinterpret_cast<char*>(out.data()))
                       == AMD_COMGR_STATUS_SUCCESS;
            }

            void appendLog(amd_comgr_data_set_t const& set, std::string& log)
            {
                std::vector<uint8_t> text;
                if(firstData(set, AMD_COMGR_DATA_KIND_LOG, text) && !text.empty())
                {
                    log.append(reinterpret_cast<char const*>(text.data()), text.size());
                }
            }

            // The LLVM version of the linked compiler: its preprocessor's __clang_* macros.
            bool probeLlvmVersion(unsigned& major, unsigned& minor, unsigned& patch)
            {
                char const payload[] = "HIPTENSOR_LLVM_VERSION __clang_major__ __clang_minor__ "
                                       "__clang_patchlevel__\n";
                DataSet    inputs, outputs;
                Data       source;
                ActionInfo info;
                if(!inputs.create() || !outputs.create() || !info.create())
                {
                    return false;
                }
                if(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &source.data)
                   != AMD_COMGR_STATUS_SUCCESS)
                {
                    return false;
                }
                source.created  = true;
                char const* isa = nullptr;
                if(amd_comgr_set_data(source.data, sizeof(payload) - 1, payload)
                       != AMD_COMGR_STATUS_SUCCESS
                   || amd_comgr_set_data_name(source.data, "hiptensor_compiler_version.cl")
                          != AMD_COMGR_STATUS_SUCCESS
                   || amd_comgr_data_set_add(inputs.set, source.data) != AMD_COMGR_STATUS_SUCCESS
                   || amd_comgr_get_isa_name(0, &isa) != AMD_COMGR_STATUS_SUCCESS
                   || amd_comgr_action_info_set_isa_name(info.info, isa) != AMD_COMGR_STATUS_SUCCESS
                   || amd_comgr_action_info_set_language(info.info, AMD_COMGR_LANGUAGE_OPENCL_1_2)
                          != AMD_COMGR_STATUS_SUCCESS
                   || amd_comgr_do_action(AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR,
                                          info.info,
                                          inputs.set,
                                          outputs.set)
                          != AMD_COMGR_STATUS_SUCCESS)
                {
                    return false;
                }
                std::vector<uint8_t> text;
                if(!firstData(outputs.set, AMD_COMGR_DATA_KIND_SOURCE, text))
                {
                    return false;
                }
                text.push_back(0);
                char const* line = reinterpret_cast<char const*>(text.data());
                while(line != nullptr && *line != '\0')
                {
                    if(std::sscanf(line, "HIPTENSOR_LLVM_VERSION %u %u %u", &major, &minor, &patch)
                           == 3
                       && major != 0)
                    {
                        return true;
                    }
                    line = std::strchr(line, '\n');
                    if(line != nullptr)
                    {
                        line++;
                    }
                }
                return false;
            }
        } // namespace

        CompilerInfo const& compilerInfo()
        {
            static CompilerInfo const info = [] {
                CompilerInfo     out;
                std::scoped_lock lock(comgrMutex());
                amd_comgr_get_version(&out.comgrMajor, &out.comgrMinor);
                if(!probeLlvmVersion(out.llvmMajor, out.llvmMinor, out.llvmPatch))
                {
                    out.reason = "the linked COMGR library does not report its LLVM version";
                    return out;
                }
                // rocKE's flavor ladder (lower_llvm/core.cpp): LLVM 23 and later, 21 and 22,
                // older
                out.flavor = out.llvmMajor >= 23   ? "llvm23"
                             : out.llvmMajor >= 21 ? "llvm22"
                                                   : "llvm20";
                out.id     = "llvm-" + std::to_string(out.llvmMajor) + "."
                         + std::to_string(out.llvmMinor) + "." + std::to_string(out.llvmPatch)
                         + ";comgr-" + std::to_string(out.comgrMajor) + "."
                         + std::to_string(out.comgrMinor);
                out.available = true;
                return out;
            }();
            return info;
        }

        std::vector<std::string> const& jitCompileOptions()
        {
            // rocKE's options policy for kernels without MFMA accumulator placement attributes
            static std::vector<std::string> const options{"-O3"};
            return options;
        }

        std::string isaName(std::string const& baseArch)
        {
            return "amdgcn-amd-amdhsa--" + baseArch;
        }

        bool compileLlvmIr(std::string const&              llvmText,
                           std::string const&              isa,
                           std::vector<std::string> const& options,
                           std::vector<uint8_t>&           codeObject,
                           std::string&                    log)
        {
            std::scoped_lock lock(comgrMutex());
            DataSet          inputs, bc, relocatable, executable;
            Data             source;
            ActionInfo       info;
            if(!inputs.create() || !bc.create() || !relocatable.create() || !executable.create()
               || !info.create())
            {
                log = "COMGR: cannot create its objects";
                return false;
            }
            if(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &source.data)
               != AMD_COMGR_STATUS_SUCCESS)
            {
                log = "COMGR: cannot create the source";
                return false;
            }
            source.created = true;
            std::vector<char const*> opts;
            for(auto const& o : options)
            {
                opts.push_back(o.c_str());
            }
            if(amd_comgr_set_data(source.data, llvmText.size(), llvmText.data())
                   != AMD_COMGR_STATUS_SUCCESS
               || amd_comgr_set_data_name(source.data, "kernel.ll") != AMD_COMGR_STATUS_SUCCESS
               || amd_comgr_data_set_add(inputs.set, source.data) != AMD_COMGR_STATUS_SUCCESS
               || amd_comgr_action_info_set_isa_name(info.info, isa.c_str())
                      != AMD_COMGR_STATUS_SUCCESS
               || amd_comgr_action_info_set_language(info.info, AMD_COMGR_LANGUAGE_LLVM_IR)
                      != AMD_COMGR_STATUS_SUCCESS
               || amd_comgr_action_info_set_option_list(info.info, opts.data(), opts.size())
                      != AMD_COMGR_STATUS_SUCCESS
               || amd_comgr_action_info_set_logging(info.info, true) != AMD_COMGR_STATUS_SUCCESS)
            {
                log = "COMGR: cannot set up the compile of " + isa;
                return false;
            }
            struct Stage
            {
                amd_comgr_action_kind_t action;
                DataSet*                in;
                DataSet*                out;
                char const*             name;
            };
            Stage const stages[] = {
                {AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, &inputs, &bc, "COMPILE_SOURCE_TO_BC"},
                {AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                 &bc,
                 &relocatable,
                 "CODEGEN_BC_TO_RELOCATABLE"},
                {AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                 &relocatable,
                 &executable,
                 "LINK_RELOCATABLE_TO_EXECUTABLE"},
            };
            for(auto const& stage : stages)
            {
                if(amd_comgr_do_action(stage.action, info.info, stage.in->set, stage.out->set)
                   != AMD_COMGR_STATUS_SUCCESS)
                {
                    log = std::string("COMGR ") + stage.name + " failed: ";
                    appendLog(stage.out->set, log);
                    return false;
                }
            }
            if(!firstData(executable.set, AMD_COMGR_DATA_KIND_EXECUTABLE, codeObject))
            {
                log = "COMGR produced no code object";
                return false;
            }
            return true;
        }
    } // namespace rockebackend
} // namespace hiptensor
