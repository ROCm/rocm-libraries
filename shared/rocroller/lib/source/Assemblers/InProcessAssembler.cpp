/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <vector>

#include <amd_comgr/amd_comgr.h>
#include <fmt/core.h>

#include <rocRoller/Assemblers/InProcessAssembler.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Timer.hpp>

// Helper macro to check for amd_comgr errors
#define COMGR_CHECK(cmd)                                                                      \
    do                                                                                        \
    {                                                                                         \
        amd_comgr_status_t e = cmd;                                                           \
        if(e != amd_comgr_status_t::AMD_COMGR_STATUS_SUCCESS)                                 \
        {                                                                                     \
            std::ostringstream msg;                                                           \
            char const*        statusMsg;                                                     \
            amd_comgr_status_string(e, &statusMsg);                                           \
            msg << "amd comgr failure at line " << __LINE__ << ": " << std::string(statusMsg) \
                << std::endl;                                                                 \
            Log::error(msg.str());                                                            \
            AssertFatal(false, msg.str());                                                    \
        }                                                                                     \
    } while(0)

namespace rocRoller
{

    InProcessAssembler::InProcessAssembler() {}

    InProcessAssembler::~InProcessAssembler() {}

    RegisterComponent(InProcessAssembler);
    static_assert(Component::Component<InProcessAssembler>);

    bool InProcessAssembler::Match(Argument arg)
    {
        return arg == AssemblerType::InProcess;
    }

    AssemblerPtr InProcessAssembler::Build(Argument arg)
    {
        if(!Match(arg))
            return nullptr;

        return std::make_shared<InProcessAssembler>();
    }

    std::string InProcessAssembler::name() const
    {
        return Name;
    }

    std::vector<char> InProcessAssembler::assembleMachineCode(const std::string& machineCode,
                                                              const GPUArchitectureTarget& target)
    {
        return assembleMachineCode(machineCode, target, defaultKernelName);
    }

    std::vector<char> InProcessAssembler::assembleMachineCode(const std::string& machineCode,
                                                              const GPUArchitectureTarget& target,
                                                              const std::string& kernelName)
    {
        // Time assembleMachineCode function
        TIMER(t, "Assembler::assembleMachineCode");

        std::vector<char> result;
        size_t            dataOutSize   = 0;
        auto const        arch          = GPUArchitectureLibrary::getInstance()->GetArch(target);
        auto const        wavefrontSize = arch.GetCapability(GPUCapability::DefaultWavefrontSize);
        std::string const targetID      = fmt::format("amdgcn-amd-amdhsa--{}", target.toString());

        std::string dataName = kernelName.empty() ? defaultKernelName : kernelName;
        AssertFatal(!dataName.empty(), "InProcessAssembler needs a kernel name");

        amd_comgr_data_t        DataIn1, DataOut;
        amd_comgr_data_set_t    DataSetIn, DataSetOut, DataSetExec;
        amd_comgr_action_info_t DataAction;
        amd_comgr_status_t      Status;

        const char* CodeGenOptions[]
            = {"-mcode-object-version=5",
               (wavefrontSize == 64) ? "-mwavefrontsize64" : "-mno-wavefrontsize64"};
        size_t CodeGenOptionsCount = sizeof(CodeGenOptions) / sizeof(CodeGenOptions[0]);

        COMGR_CHECK(amd_comgr_create_data_set(&DataSetIn));
        COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataIn1));

        COMGR_CHECK(amd_comgr_set_data(DataIn1, machineCode.size(), machineCode.c_str()));
        COMGR_CHECK(amd_comgr_set_data_name(DataIn1, dataName.c_str()));
        COMGR_CHECK(amd_comgr_data_set_add(DataSetIn, DataIn1));

        COMGR_CHECK(amd_comgr_create_data_set(&DataSetOut));
        COMGR_CHECK(amd_comgr_create_data_set(&DataSetExec));

        COMGR_CHECK(amd_comgr_create_action_info(&DataAction));
        COMGR_CHECK(amd_comgr_action_info_set_isa_name(DataAction, targetID.c_str()));
        COMGR_CHECK(
            amd_comgr_action_info_set_option_list(DataAction, CodeGenOptions, CodeGenOptionsCount));

        COMGR_CHECK(amd_comgr_do_action(
            AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, DataAction, DataSetIn, DataSetOut));

        COMGR_CHECK(amd_comgr_do_action(
            AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, DataAction, DataSetOut, DataSetExec));

        COMGR_CHECK(amd_comgr_action_data_get_data(
            DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataOut));

        COMGR_CHECK(amd_comgr_get_data(DataOut, &dataOutSize, nullptr));
        AssertFatal(dataOutSize > 0, "No compiled kernel data");
        result.resize(dataOutSize);
        COMGR_CHECK(amd_comgr_get_data(DataOut, &dataOutSize, result.data()));

        COMGR_CHECK(amd_comgr_destroy_data_set(DataSetIn));
        COMGR_CHECK(amd_comgr_destroy_data_set(DataSetOut));
        COMGR_CHECK(amd_comgr_destroy_data_set(DataSetExec));
        COMGR_CHECK(amd_comgr_destroy_action_info(DataAction));
        COMGR_CHECK(amd_comgr_release_data(DataIn1));

        return result;
    }
}
