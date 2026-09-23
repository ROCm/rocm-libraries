// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <rocRoller/Assemblers/Assembler.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/WorkgroupClusters_detail.hpp>

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

    ExecutableKernel::ExecutableKernel(GPUArchitectureTarget target)
        : m_kernelLoaded(false)
        , m_hipData(std::make_shared<HIPData>())
        , m_target(target)
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

        int idx = -1, currDeviceAsicRevisionId = -1;
        HIP_CHECK(hipGetDevice(&idx));
        HIP_CHECK(
            hipDeviceGetAttribute(&currDeviceAsicRevisionId, hipDeviceAttributeAsicRevision, idx));
        if(m_target.gfx == GPUArchitectureGFX::GFX1250)
        {
            if(currDeviceAsicRevisionId == 2)
            {
                int newRevId = Settings::Get(Settings::GFX1250AsicRevisionId);
                Log::warn("Overriding current device asic revision id from {} to {}",
                          currDeviceAsicRevisionId,
                          newRevId);
                currDeviceAsicRevisionId = newRevId;
            }
            AssertFatal(m_target.asicRevisionId == currDeviceAsicRevisionId,
                        "The kernel and current device must have the same target revision id.",
                        ShowValue(m_target.asicRevisionId),
                        ShowValue(currDeviceAsicRevisionId));
        }

        if(timer)
            HIP_TIC(timer, iteration);

        if(invocation.workgroupClusterSize)
        {
#ifdef ROCROLLER_HAS_HIP_WORKGROUP_CLUSTERS
            auto const workgroupClusterSize = *invocation.workgroupClusterSize;
            auto const numWorkgroupsX = invocation.workitemCount[0] / invocation.workgroupSize[0];
            auto const numWorkgroupsY = invocation.workitemCount[1] / invocation.workgroupSize[1];
            auto const numWorkgroupsZ = invocation.workitemCount[2] / invocation.workgroupSize[2];

            // TODO Can/should we validate earlier? (i.e. not immediately before launch)
            AssertFatal(WorkgroupClustersDetail::IsValidWorkgroupClusterSize(
                            workgroupClusterSize, {numWorkgroupsX, numWorkgroupsY, numWorkgroupsZ}),
                        fmt::format("Invalid cluster size: [{},{},{}].\n"
                                    "The number of workgroups [{},{},{}] in each dimension must "
                                    "be divisible by the cluster size in that dimension.\n"
                                    "Valid cluster sizes:{}",
                                    workgroupClusterSize[0],
                                    workgroupClusterSize[1],
                                    workgroupClusterSize[2],
                                    numWorkgroupsX,
                                    numWorkgroupsY,
                                    numWorkgroupsZ,
                                    [=]() {
                                        std::string s;
                                        auto const  valid
                                            = WorkgroupClustersDetail::ValidWorkgroupClusterSizes(
                                                {numWorkgroupsX, numWorkgroupsY, numWorkgroupsZ});
                                        for(const auto& v : valid)
                                            s += fmt::format(" [{},{},{}]", v[0], v[1], v[2]);
                                        return s;
                                    }()));

            hipLaunchAttribute attribute[1];
            HIP_LAUNCH_CONFIG  config = {0};

            config.gridDimX  = numWorkgroupsX;
            config.gridDimY  = numWorkgroupsY;
            config.gridDimZ  = numWorkgroupsZ;
            config.blockDimX = invocation.workgroupSize[0];
            config.blockDimY = invocation.workgroupSize[1];
            config.blockDimZ = invocation.workgroupSize[2];

            attribute[0].id               = hipLaunchAttributeClusterDimension;
            attribute[0].val.clusterDim.x = workgroupClusterSize[0];
            attribute[0].val.clusterDim.y = workgroupClusterSize[1];
            attribute[0].val.clusterDim.z = workgroupClusterSize[2];
            config.attrs                  = attribute;
            config.numAttrs               = 1;
            config.sharedMemBytes         = invocation.sharedMemBytes;

            Log::debug("Launching kernel {} with Clusters: [{}, {}, {}]",
                       m_kernelName,
                       workgroupClusterSize[0],
                       workgroupClusterSize[1],
                       workgroupClusterSize[2]);

            const HIP_LAUNCH_CONFIG* pConfig = &config;
            HIP_CHECK(hipDrvLaunchKernelEx(
                pConfig, m_hipData->function, nullptr, (void**)&hipLaunchParams));
#else
            Throw<FatalError>(
                "Workgroup cluster launch requested but the installed ROCm/HIP version does not "
                "support hipLaunchAttributeClusterDimension. "
                "Please upgrade to a ROCm version that includes workgroup cluster support.");
#endif /* ROCROLLER_HAS_HIP_WORKGROUP_CLUSTERS */
        }
        else
        {
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
        }

        if(timer)
            HIP_TOC(timer, iteration);
    }
}
