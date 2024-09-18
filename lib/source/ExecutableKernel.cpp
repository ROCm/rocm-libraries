#include <vector>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/Utilities/Error.hpp>

#include "Assemblers/Assembler.hpp"
#include "ExecutableKernel.hpp"
#include "Utilities/HipUtils.hpp"

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

    void ExecutableKernel::executeKernel(const KernelArguments&  args,
                                         const KernelInvocation& invocation)
    {
        executeKernel(args, invocation, nullptr, 0);
    }

    void ExecutableKernel::executeKernel(const KernelArguments&    args,
                                         const KernelInvocation&   invocation,
                                         std::shared_ptr<HIPTimer> timer,
                                         int                       iteration)
    {
        // TODO: Include this at a particular logging level.
        if(args.log())
        {
            std::cout << "Launching kernel " << m_kernelName << ": Workgroup: {"
                      << invocation.workgroupSize[0] << ", " << invocation.workgroupSize[1] << ", "
                      << invocation.workgroupSize[2] << "}, Workitems: {"
                      << invocation.workitemCount[0] << ", " << invocation.workitemCount[1] << ", "
                      << invocation.workitemCount[2] << "}" << std::endl;
            std::cout << args << std::endl;
        }

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
                                           timer ? timer->stream() : 0, // stream
                                           nullptr,
                                           (void**)&hipLaunchParams,
                                           nullptr, // event
                                           nullptr // event
                                           ));

        if(timer)
            HIP_TOC(timer, iteration);
    }
}
