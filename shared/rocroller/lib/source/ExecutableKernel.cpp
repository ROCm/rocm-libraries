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

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/Utilities/Error.hpp>

#include <rocRoller/Assemblers/Assembler.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>

namespace rocRoller
{

    // Values using HIP datatypes are stored in hip_data
    // to avoid including HIP headers within ExecutableKernel.hpp
    struct ExecutableKernel::HIPData
    {
        hipModule_t   hipModule;
        hipFunction_t function;

        ~HIPData()
        {
            if(hipModule)
            {
                hipError_t error = hipModuleUnload(hipModule);
                if(error != hipSuccess)
                {
                    std::ostringstream msg;
                    msg << "HIP failure at hipModuleUnload (~HIPData()): "
                        << hipGetErrorString(error) << std::endl;
                    Log::error(msg.str());
                }
            }
        }
    };

    ExecutableKernel::ExecutableKernel()
        : m_kernelLoaded(false)
        , m_hipData(std::make_shared<HIPData>())
    {
    }

    hipFunction_t ExecutableKernel::getHipFunction() const
    {
        return m_hipData->function;
    }

    void ExecutableKernel::loadKernel(std::string const&           instructions,
                                      const GPUArchitectureTarget& target,
                                      std::string const&           kernelName)
    {
        auto assembler = Assembler::Get();

        std::vector<char> kernelObject
            = assembler->assembleMachineCode(instructions, target, kernelName);

        if(instructions.size())
        {
            HIP_CHECK(hipModuleLoadData(&(m_hipData->hipModule), kernelObject.data()));
            HIP_CHECK(hipModuleGetFunction(
                &(m_hipData->function), m_hipData->hipModule, kernelName.c_str()));
            m_kernelLoaded = true;
            m_kernelName   = kernelName;
        }
    }

    void ExecutableKernel::loadKernelFromFile(std::string const&           fileName,
                                              std::string const&           kernelName,
                                              const GPUArchitectureTarget& target)
    {
        std::string   machineCode;
        std::ifstream assemblyFile(fileName, std::ios::in | std::ios::binary);
        if(assemblyFile)
        {
            machineCode.assign(std::istreambuf_iterator<char>(assemblyFile),
                               std::istreambuf_iterator<char>());
            Log::debug("Alernate kernel read from: {}", fileName);
        }
        else
        {
            Log::debug("Error reading alternate kernel from: {}", fileName);
        }
        AssertFatal(assemblyFile, "Error reading alternate kernel.", ShowValue(fileName));

        loadKernel(machineCode, target, kernelName);
    }

    void ExecutableKernel::loadKernelFromCodeObjectFile(std::string const&           fileName,
                                                        std::string const&           kernelName,
                                                        const GPUArchitectureTarget& target)
    {
        Log::debug("ExecutableKernel::loadKernelFromCodeObjectFile: fileName {} kernelName {}",
                   fileName,
                   kernelName);

        auto kernelObject = readFile(fileName);

        HIP_CHECK(hipModuleLoadData(&(m_hipData->hipModule), kernelObject.data()));
        HIP_CHECK(
            hipModuleGetFunction(&(m_hipData->function), m_hipData->hipModule, kernelName.c_str()),
            ShowValue(kernelName));
        m_kernelLoaded = true;
        m_kernelName   = kernelName;
    }

    void ExecutableKernel::executeKernel(const KernelArguments&    args,
                                         const KernelInvocation&   invocation,
                                         std::shared_ptr<HIPTimer> timer,
                                         int                       iteration)
    {
        executeKernel(args, invocation, timer, iteration, timer->stream());
    }

    void ExecutableKernel::executeKernel(const KernelArguments&  args,
                                         const KernelInvocation& invocation,
                                         hipStream_t             stream)
    {
        executeKernel(args, invocation, nullptr, 0, stream);
    }

    void ExecutableKernel::executeKernel(const KernelArguments&    args,
                                         const KernelInvocation&   invocation,
                                         std::shared_ptr<HIPTimer> timer,
                                         int                       iteration,
                                         hipStream_t               stream)
    {
        // TODO: implement formatter for container and library types
        Log::debug(fmt::format("Launching kernel {}: Workgroup: [{}, {}, {}], Workitems: [{}, {}, "
                               "{}]\nKernel arguments: {}\n",
                               m_kernelName,
                               invocation.workgroupSize[0],
                               invocation.workgroupSize[1],
                               invocation.workgroupSize[2],
                               invocation.workitemCount[0],
                               invocation.workitemCount[1],
                               invocation.workitemCount[2],
                               args.toString()));

        if(!kernelLoaded())
            throw std::runtime_error("Attempting to execute a kernel before it is loaded");

        size_t argsSize = args.size();

        void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                   const_cast<void*>(args.data()),
                                   HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                   &argsSize,
                                   HIP_LAUNCH_PARAM_END};

        if(timer)
            HIP_TIC(timer, iteration);

        HIP_CHECK(hipExtModuleLaunchKernel(m_hipData->function,
                                           invocation.workitemCount[0],
                                           invocation.workitemCount[1],
                                           invocation.workitemCount[2],
                                           invocation.workgroupSize[0],
                                           invocation.workgroupSize[1],
                                           invocation.workgroupSize[2],
                                           invocation.sharedMemBytes,
                                           stream,
                                           nullptr,
                                           (void**)&hipLaunchParams,
                                           nullptr, // event
                                           nullptr // event
                                           ));
        if(timer)
            HIP_TOC(timer, iteration);
    }

}
