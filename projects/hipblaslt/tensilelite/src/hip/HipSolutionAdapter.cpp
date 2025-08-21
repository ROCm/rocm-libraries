/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <cstddef>

#include <Tensile/Debug.hpp>
#include <Tensile/EmbeddedData.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

//@TODO add alternative for windows
#ifndef _WIN32
#include <glob.h>
#endif
#include <regex>

namespace TensileLite
{
    namespace hip
    {
        SolutionAdapter::SolutionAdapter()
            : m_debug(Debug::Instance().printKernelArguments())
            , m_debugSkipLaunch(Debug::Instance().skipKernelLaunch())
        {
        }

        SolutionAdapter::SolutionAdapter(bool debug)
            : m_debug(debug)
        {
            m_debug = debug || Debug::Instance().printKernelArguments();
        }

        SolutionAdapter::SolutionAdapter(bool debug, std::string const& name)
            : m_debug(debug)
            , m_name(name)
        {
            m_debug = debug || Debug::Instance().printKernelArguments();
        }

        SolutionAdapter::~SolutionAdapter()
        {
            Debug::Instance().markerStart("UnloadCodeObjectFiles");
            for(auto module : m_modules)
                HIP_CHECK_PRINT(hipModuleUnload(module),
                    [&](hipError_t error) {
                        std::cerr << "hipModuleUnload failed: " << std::endl
                                << " error: " << hipGetErrorString(error) << std::endl;
                    }
                );
            Debug::Instance().markerStop();
        }

        std::string removeXnack(std::string coFilename)
        {
            std::string xnackVersion = "xnack"; //Extra character before and after xnack
            size_t      loc          = coFilename.find(xnackVersion);
            if(loc != std::string::npos)
                coFilename.replace(loc - 1, xnackVersion.length() + 2, "");

            return coFilename;
        }

        hipError_t SolutionAdapter::loadCodeObjectFile(std::string const& path)
        {
            Debug::Instance().markerStart("loadCodeObjectFile", path);
            hipModule_t module;

            hipError_t error = hipModuleLoad(&module, path.c_str());
            // Large problem sizes may cause global memory to run out of space 
            // when loading the module, which can lead to hipErrorLaunchFailure or hipErrorNoBinaryForGpu.
            if(error == hipErrorLaunchFailure || error == hipErrorNoBinaryForGpu)
            {        
                // Reset the error code from previous hipModuleLoad failure
                (void)hipGetLastError();
                std::cout << "Clearing modules and retrying hipModuleLoad" << std::endl;
                for(auto m_module : m_modules)
                {
                    HIP_CHECK_PRINT(hipModuleUnload(m_module),
                        [&](hipError_t error_t) {
                            std::cerr << "hipModuleUnload failed: " << std::endl
                                      << " error: " << hipGetErrorString(error_t) << std::endl;
                        }
                    );
                }
                // Need to clean up all these old modules' data structures, otherwise next problem will getKernel failed
                m_access.lock();
                m_modules.clear();
                m_loadedModuleNames.clear();
                m_loadedCOFiles.clear();
                m_kernels.clear();
                m_access.unlock();
                // Need to re-run lazy-loading for hsaco(helper kernels) module reload
                std::string lazyArch;
                std::string lazyDir;
                lazyArch = m_lazyLoadArchitecture;
                lazyDir  = m_codeObjectDirectory;
                HIP_CHECK_RETURN_WITH_LOG(initializeLazyLoading(lazyArch, lazyDir),
                    [&](hipError_t error_t) {
                        std::cerr << "initializeLazyLoading after module clear failed: " << std::endl
                                  << " error: " << hipGetErrorString(error_t) << std::endl;
                    }
                );
                HIP_CHECK_RETURN_WITH_LOG(hipModuleLoad(&module, path.c_str()),
                    [&](hipError_t error_t) {
                        std::cerr << "hipModuleLoad failed: " << path.c_str() << std::endl
                                  << " error: " << hipGetErrorString(error_t) << std::endl;
                    }
                );
            }
            else if(error)
            {
                std::cerr << "hipModuleLoad failed: " << path.c_str() << std::endl
                          << " error: " << hipGetErrorString(error) << std::endl;
                return error;
            }

            if(m_debug)
                std::cout << "loaded code object " << path << std::endl;

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.push_back(module);
                m_loadedModuleNames.push_back(concatenate("File ", path));

                //Isolate filename
                size_t start = path.rfind('/');
                start        = (start == std::string::npos) ? 0 : start + 1;
                m_loadedCOFiles.insert(removeXnack(std::string(path.begin() + start, path.end())));
            }
            Debug::Instance().markerStop();
            return hipSuccess;
        }

        hipError_t SolutionAdapter::loadCodeObjectBytes(std::vector<uint8_t> const& bytes)
        {
            return loadCodeObject(bytes.data());
        }

        hipError_t SolutionAdapter::loadCodeObject(const void* image)
        {
            hipModule_t module;

            HIP_CHECK_RETURN_WITH_LOG(hipModuleLoadData(&module, image),
                [&](hipError_t error) {
                    std::cerr << "hipModuleLoadData failed: " << std::endl
                            << " error: " << hipGetErrorString(error) << std::endl;
                }
            );

            if(m_debug)
                std::cout << "loaded code object data." << std::endl;

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.push_back(module);
                m_loadedModuleNames.push_back("Module from bytes");
            }
            return hipSuccess;
        }

        void SolutionAdapter::loadEmbeddedCodeObjects()
        {
            loadEmbeddedCodeObjects("");
        }

        void SolutionAdapter::loadEmbeddedCodeObjects(std::string const& key)
        {
            auto const& embeddedData = EmbeddedData<TensileLite::SolutionAdapter>::Get(key);

            if(embeddedData.size() == 0)
            {
                if(m_debug || Debug::Instance().printCodeObjectInfo())
                {
                    std::cerr << "Found no embedded code objects";
                    if(key != "")
                        std::cerr << " with the key " << key;

                    std::cerr << "." << std::endl;
                }
                return;
            }

            std::vector<hipModule_t> newModules;
            newModules.reserve(embeddedData.size());

            for(size_t i = 0; i < embeddedData.size(); i++)
            {
                hipModule_t nextModule;
                try
                {
                    auto error = hipModuleLoadData(&nextModule, embeddedData[i].data());

                    if(error == hipErrorUnknown || error == hipErrorSharedObjectInitFailed)
                        continue;
                    newModules.push_back(nextModule);
                    HIP_CHECK_EXC(error);

                    if(m_debug)
                        std::cout << "Loaded code object for key " << key << std::endl;
                }
                catch(std::runtime_error const& exc)
                {
                    std::cout << exc.what() << std::endl;
                }
            }

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.insert(m_modules.end(), newModules.begin(), newModules.end());
                m_loadedModuleNames.push_back(
                    concatenate("Embedded code object ", key, " (", newModules.size(), ")"));
            }
        }

        bool SolutionAdapter::FindCodeObject(std::string const& codeObjectFile)
        {
            //If required code object file hasn't yet been loaded, load it now
            m_access.lock();
            bool loaded = m_loadedCOFiles.find(removeXnack(codeObjectFile))
                          != m_loadedCOFiles.end();
            std::string codeObjectDir = m_codeObjectDirectory;
            m_access.unlock();

            if(!loaded)
            {
                //Try other xnack versions
                size_t     loc = codeObjectFile.rfind('.');
                hipError_t err;

                for(auto ver : {"", "-xnack-", "-xnack+"})
                {
                    std::string modifiedCOName = codeObjectFile;
                    modifiedCOName.insert(loc, ver);
                    err = loadCodeObjectFile(codeObjectDir + modifiedCOName);

                    if(err == hipSuccess)
                        break;
                }
                return false;
            }
            return true;
        }

        hipError_t SolutionAdapter::initKernel(std::string const& name)
        {
            hipFunction_t function;
            return getKernel(function, name);
        }

        hipError_t SolutionAdapter::initKernels(std::vector<std::string> const& kernelNames)
        {
            hipFunction_t rv;

            for(auto name : kernelNames)
            {
                hipFunction_t function;
                auto result = getKernel(function, name);
                if(result != hipSuccess)
                    return result;
            }

            return hipSuccess;
        }

        hipError_t SolutionAdapter::getKernel(hipFunction_t& rv, std::string const& name)
        {
            std::unique_lock<std::mutex> guard(m_access);
            hipError_t                   err = hipErrorNotFound;

            auto it = m_kernels.find(name);
            if(it != m_kernels.end())
            {
                rv = it->second;
                return hipSuccess;
            }

            for(auto module : m_modules)
            {
                err = hipModuleGetFunction(&rv, module, name.c_str());

                if(err == hipSuccess)
                {
                    m_kernels[name] = rv;
                    return err;
                }
                else if(err != hipErrorNotFound)
                {
                    return err;
                }
                else
                {
                    (void)hipGetLastError(); // clear hipErrorNotFound
                }
            }

            return err;
        }

        // We should update the constructor to set this to avoid
        // avoid separating construction and initialization
        void SolutionAdapter::codeObjectDir(std::string codeObjDir)
        {

            if(!codeObjDir.empty())
            {
                if(codeObjDir.back() != '/')
                {
                    codeObjDir += '/';
                }
            }

            m_access.lock();
            m_codeObjectDirectory = codeObjDir;
            m_access.unlock();
        }

        hipError_t SolutionAdapter::initializeLazyLoading(std::string arch,
                                                          std::string codeObjDir)
        {
            //Ensure there's a slash at the end of the path
            if(!codeObjDir.empty())
            {
                if(codeObjDir.back() != '/')
                {
                    codeObjDir += '/';
                }
            }

            //Remove xnack and sramecc qualifiers
            size_t loc = arch.find(":");
            if(loc != std::string::npos)
                arch.resize(loc);

            std::string helperKernelName = std::string("Kernels.so-000-") + arch;

            m_access.lock();

            // Record for module reload
            m_lazyLoadArchitecture = arch;
            m_codeObjectDirectory  = codeObjDir;

            //If required code object file hasn't yet been loaded, load it now
            bool loaded = m_loadedCOFiles.find(removeXnack(helperKernelName) + ".hsaco")
                          != m_loadedCOFiles.end();
            m_access.unlock();

            if(!loaded)
            {
                hipError_t err;
                //Try xnack variations
                for(auto ver : {"", "-xnack-", "-xnack+"})
                {
                    std::string modifiedCOName = helperKernelName + ver + ".hsaco";
                    err                        = loadCodeObjectFile(codeObjDir + modifiedCOName);

                    if(err == hipSuccess)
                    {
                        return err;
                    }
                    else if(err == hipErrorFileNotFound)
                    {
                        // We expect that we could fail for cases when we have xnack variations
                        // so clear hipErrorFileNotFound between iterations.
                        (void)hipGetLastError();
                    }
                }

                return err;
            }

            return hipSuccess;
        }

        hipError_t SolutionAdapter::launchKernel(KernelInvocation const& kernel)
        {
            return launchKernel(kernel, nullptr, nullptr, nullptr);
        }

        hipError_t SolutionAdapter::launchKernel(KernelInvocation const& kernel,
                                                 hipStream_t             stream,
                                                 hipEvent_t              startEvent,
                                                 hipEvent_t              stopEvent,
                                                 bool                    isKernelLoaded)
        {
            if(!isKernelLoaded && !kernel.codeObjectFile.empty())
            {
                FindCodeObject(kernel.codeObjectFile);
            }

            if(m_debug)
            {
                std::cout << "Kernel " << kernel.kernelName << std::endl;
                std::cout << " l" << kernel.workGroupSize << " x g" << kernel.numWorkGroups << " = "
                          << kernel.numWorkItems << std::endl;
                std::cout << kernel.args;
            }
            if(m_debugSkipLaunch)
            {
                std::cout << "DEBUG: Skip kernel execution" << std::endl;
                if(startEvent != nullptr)
                    HIP_CHECK_RETURN(hipEventRecord(startEvent, stream));
                if(stopEvent != nullptr)
                    HIP_CHECK_RETURN(hipEventRecord(stopEvent, stream));
                return hipSuccess;
            }

            hipFunction_t function;
            HIP_CHECK_RETURN_WITH_LOG(getKernel(function, kernel.kernelName),
                [&](hipError_t error) {
                    std::cerr << "getKernel failed: " << kernel.kernelName << std::endl
                            << " with workgroup size: " << kernel.workGroupSize << std::endl
                            << " with numWorkGroups : " << kernel.numWorkGroups << std::endl
                            << " with numWorkItems : " << kernel.numWorkItems << std::endl
                            << " error: " << hipGetErrorString(error) << std::endl;
                }
            );

            void*  kernelArgs = const_cast<void*>(kernel.args.data());
            size_t argsSize   = kernel.args.size();

            void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                       kernelArgs,
                                       HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                       &argsSize,
                                       HIP_LAUNCH_PARAM_END};

            if(startEvent != nullptr)
                HIP_CHECK_RETURN(hipEventRecord(startEvent, stream));

            if (kernel.enableCluster){
                HIP_LAUNCH_CONFIG config = {0};
                // The grid dimension is not affected by cluster launch, and is still enumerated
                // using number of blocks.
                // The grid dimension should be a multiple of cluster size.
                config.gridDimX = kernel.numWorkGroups.x;
                config.gridDimY = kernel.numWorkGroups.y;
                config.gridDimZ = kernel.numWorkGroups.z;
                config.blockDimX = kernel.workGroupSize.x;
                config.blockDimY = kernel.workGroupSize.y;
                config.blockDimZ = kernel.workGroupSize.z;
    
                hipLaunchAttribute attribute[1];
                attribute[0].id = hipLaunchAttributeClusterDimension;
                attribute[0].val.clusterDim.x = kernel.clusterDim.x; // Cluster size in X-dimension
                attribute[0].val.clusterDim.y = kernel.clusterDim.y;
                attribute[0].val.clusterDim.z = 1;
                config.attrs = attribute;
                config.numAttrs = 1;
                config.sharedMemBytes = kernel.sharedMemBytes;
    
                const HIP_LAUNCH_CONFIG *pConfig = &config;
                HIP_CHECK_RETURN_WITH_LOG(hipDrvLaunchKernelEx(pConfig,
                                                               function,
                                                               nullptr,
                                                               (void**)&hipLaunchParams),
                    [&](hipError_t error) {
                        std::cerr << "hipExtModuleLaunchKernel failed: " << kernel.kernelName << std::endl
                                  << " with workgroup size: " << kernel.workGroupSize << std::endl
                                  << " with numWorkGroups : " << kernel.numWorkGroups << std::endl
                                  << " with numWorkItems : " << kernel.numWorkItems << std::endl
                                  << " error: " << hipGetErrorString(error) << std::endl;
                    }
                );
            } else {
                HIP_CHECK_RETURN_WITH_LOG(hipExtModuleLaunchKernel(function,
                                                          kernel.numWorkItems.x,
                                                          kernel.numWorkItems.y,
                                                          kernel.numWorkItems.z,
                                                          kernel.workGroupSize.x,
                                                          kernel.workGroupSize.y,
                                                          kernel.workGroupSize.z,
                                                          kernel.sharedMemBytes, // sharedMem
                                                          stream, // stream
                                                          nullptr,
                                                          (void**)&hipLaunchParams,
                                                          nullptr, // event
                                                          nullptr // event
                                                          ),
                    [&](hipError_t error) {
                        std::cerr << "hipExtModuleLaunchKernel failed: " << kernel.kernelName << std::endl
                                  << " with workgroup size: " << kernel.workGroupSize << std::endl
                                  << " with numWorkGroups : " << kernel.numWorkGroups << std::endl
                                  << " with numWorkItems : " << kernel.numWorkItems << std::endl
                                  << " error: " << hipGetErrorString(error) << std::endl;
                    }
                );
            }

            if(stopEvent != nullptr)
                HIP_CHECK_RETURN(hipEventRecord(stopEvent, stream));
            return hipSuccess;
        }

        hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels)
        {
            for(auto const& k : kernels)
            {
                HIP_CHECK_RETURN_WITH_LOG(launchKernel(k),
                    [&](hipError_t error) {
                        std::cerr << "launchKernel failed: " << k.kernelName << std::endl
                                << " with workgroup size: " << k.workGroupSize << std::endl
                                << " with numWorkGroups : " << k.numWorkGroups << std::endl
                                << " with numWorkItems : " << k.numWorkItems << std::endl
                                << " error: " << hipGetErrorString(error) << std::endl;
                    }
                );
            }
            return hipSuccess;
        }

        hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels,
                                                  hipStream_t                          stream,
                                                  hipEvent_t                           startEvent,
                                                  hipEvent_t                           stopEvent,
                                                  bool                                 isKernelLoaded)
        {
            auto first = kernels.begin();
            auto last  = kernels.end() - 1;

            for(auto iter = kernels.begin(); iter != kernels.end(); iter++)
            {
                hipEvent_t kStart = nullptr;
                hipEvent_t kStop  = nullptr;

                if(iter == first)
                    kStart = startEvent;
                if(iter == last)
                    kStop = stopEvent;

                HIP_CHECK_RETURN_WITH_LOG(launchKernel(*iter, stream, kStart, kStop, isKernelLoaded),
                    [&](hipError_t error) {
                        std::cerr << "launchKernel failed: " << iter->kernelName << std::endl
                                << " with workgroup size: " << iter->workGroupSize << std::endl
                                << " with numWorkGroups : " << iter->numWorkGroups << std::endl
                                << " with numWorkItems : " << iter->numWorkItems << std::endl
                                << " error: " << hipGetErrorString(error) << std::endl;
                    }
                );
            }
            return hipSuccess;
        }

        hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels,
                                                  hipStream_t                          stream,
                                                  std::vector<hipEvent_t> const&       startEvents,
                                                  std::vector<hipEvent_t> const&       stopEvents)
        {
            if(kernels.size() != startEvents.size() || kernels.size() != stopEvents.size())
                throw std::runtime_error(concatenate("Must have an equal number of kernels (",
                                                     kernels.size(),
                                                     "), start events (",
                                                     startEvents.size(),
                                                     "), and stop events. (",
                                                     stopEvents.size(),
                                                     ")"));

            for(size_t i = 0; i < kernels.size(); i++)
            {
                HIP_CHECK_RETURN_WITH_LOG(launchKernel(kernels[i], stream, startEvents[i], stopEvents[i]),
                    [&](hipError_t error) {
                        std::cerr << "launchKernel failed: " << kernels[i].kernelName << std::endl
                                << " with workgroup size: " << kernels[i].workGroupSize << std::endl
                                << " with numWorkGroups : " << kernels[i].numWorkGroups << std::endl
                                << " with numWorkItems : " << kernels[i].numWorkItems << std::endl
                                << " error: " << hipGetErrorString(error) << std::endl;
                    }
                );
            }
            return hipSuccess;
        }

        std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter)
        {
            stream << "hip::SolutionAdapter";

            if(adapter.m_debug)
            {
                stream << "[" << std::endl;
                for(auto const& name : adapter.m_loadedModuleNames)
                    stream << name << std::endl;

                stream << "]";
            }

            stream << " (" << adapter.name() << ", " << adapter.m_modules.size()
                   << " total modules)" << std::endl;

            return stream;
        }

        std::ostream& operator<<(std::ostream& stream, std::shared_ptr<SolutionAdapter> const& ptr)
        {
            if(ptr)
            {
                return stream << "*" << *ptr;
            }
            else
            {
                return stream << "(nullptr)";
            }
        }
    } // namespace hip
} // namespace TensileLite
