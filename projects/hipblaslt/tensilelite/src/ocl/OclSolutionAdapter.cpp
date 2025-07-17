/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef CL_HPP_ENABLE_EXCEPTIONS
#error \
    "This implementation relies on CL exceptions to be enabled. Please define CL_HPP_ENABLE_EXCEPTIONS"
#endif

#include <CL/cl2.hpp>
#include <cstddef>

#include <Tensile/ocl/OclSolutionAdapter.hpp>
#include <Tensile/ocl/OclUtils.hpp>

#include <Tensile/Debug.hpp>
#include <Tensile/EmbeddedData.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    namespace ocl
    {
        SolutionAdapter::SolutionAdapter()
            : m_debug(Debug::Instance().printKernelArguments())
            , m_debugSkipLaunch(Debug::Instance().skipKernelLaunch())
            , m_context(cl::Context::getDefault())
            , m_device(cl::Device::getDefault())
        {
        }

        SolutionAdapter::SolutionAdapter(bool debug)
            : m_debug(debug || Debug::Instance().printKernelArguments())
            , m_debugSkipLaunch(Debug::Instance().skipKernelLaunch())
            , m_context(cl::Context::getDefault())
            , m_device(cl::Device::getDefault())
        {
        }

        SolutionAdapter::SolutionAdapter(bool debug, std::string const& name)
            : m_debug(debug || Debug::Instance().printKernelArguments())
            , m_debugSkipLaunch(Debug::Instance().skipKernelLaunch())
            , m_context(cl::Context::getDefault())
            , m_device(cl::Device::getDefault())
            , m_name(name)
        {
        }

        SolutionAdapter::SolutionAdapter(bool               debug,
                                         std::string const& name,
                                         cl::Context        context,
                                         cl::Device         device)
            : m_debug(debug || Debug::Instance().printKernelArguments())
            , m_debugSkipLaunch(Debug::Instance().skipKernelLaunch())
            , m_context(context)
            , m_device(device)
            , m_name(name)
        {
        }

        std::string SolutionAdapter::name() const
        {
            return m_name;
        }

        void SolutionAdapter::loadCodeObjectFile(std::string const& path)
        {
            try
            {
                auto progModule = clModuleLoad(m_context, {m_device}, path);

                if(m_debug)
                {
                    std::cout << "Loaded code object: " << path << std::endl;
                }

                addModule(concatenate("File ", path), progModule);
            }
            // Binary might not be for this specific device, which is OK.
            catch(cl::BuildError const& e)
            {
                if(m_debug)
                {
                    std::cerr << "Error code: " << e.err() << " " << e.what() << std::endl;
                    for(auto const& log : e.getBuildLog())
                    {
                        std::cerr << "Device: " << std::get<0>(log).getInfo<CL_DEVICE_NAME>() << " "
                                  << std::get<1>(log) << std::endl;
                    }
                }
            }
        }

        void SolutionAdapter::loadCodeObjectBytes(std::vector<uint8_t> const& bytes)
        {
            try
            {
                auto progModule = clModuleLoadData(m_context, {m_device}, bytes);

                if(m_debug)
                {
                    std::cout << "loaded code object data." << std::endl;
                }

                addModule("Module from bytes", progModule);
            }
            // Binary might not be for this specific device, which is OK.
            catch(cl::BuildError const& e)
            {
                if(m_debug)
                {
                    std::cerr << "Error code: " << e.err() << " " << e.what() << std::endl;
                    for(auto const& log : e.getBuildLog())
                    {
                        std::cerr << "Device: " << std::get<0>(log).getInfo<CL_DEVICE_NAME>() << " "
                                  << std::get<1>(log) << std::endl;
                    }
                }
            }
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
                    std::cout << "Found no embedded code objects";
                    if(key != "")
                    {
                        std::cout << " with the key " << key;
                    }

                    std::cout << "." << std::endl;
                }
                return;
            }

            std::vector<cl::Program> newModules;
            newModules.reserve(embeddedData.size());
            for(size_t i = 0; i < embeddedData.size(); i++)
            {
                try
                {
                    newModules.push_back(clModuleLoadData(m_context, {m_device}, embeddedData[i]));

                    if(m_debug)
                    {
                        std::cout << "Loaded code object for key " << key << std::endl;
                    }
                }
                // Binary might not be for this specific device, which is OK.
                catch(cl::BuildError const& e)
                {
                    if(m_debug)
                    {
                        std::cerr << "Error code: " << e.err() << " " << e.what() << std::endl;
                        for(auto const& log : e.getBuildLog())
                        {
                            std::cerr << "Device: " << std::get<0>(log).getInfo<CL_DEVICE_NAME>()
                                      << " " << std::get<1>(log) << std::endl;
                        }
                    }
                }
            }

            if(m_debug || Debug::Instance().printCodeObjectInfo())
            {
                std::cout << "Successfully loaded " << newModules.size()
                          << " embedded code object(s)\n";
            }
            addModules(concatenate("Embedded code object ", key, " (", newModules.size(), ")"),
                       newModules);
        }

        void SolutionAdapter::initKernel(std::string const& name)
        {
            getKernel(name);
        }

        cl::Kernel SolutionAdapter::getKernel(std::string const& name)
        {
            std::unique_lock<std::mutex> guard(m_access);

            auto it = m_kernels.find(name);
            if(it != m_kernels.end())
            {
                return it->second;
            }

            for(auto module : m_modules)
            {
                try
                {
                    return m_kernels[name] = cl::Kernel(module, name.c_str());
                }
                // Kernel may not exist in all modules
                catch(cl::Error& e)
                {
                    if(e.err() != CL_INVALID_KERNEL_NAME)
                    {
                        if(m_debug)
                        {
                            std::cerr << "KernelError: " << e.err() << std::endl;
                        }
                        throw;
                    }
                }
            }

            throw cl::Error(
                CL_INVALID_KERNEL_NAME,
                concatenate("Kernel ", name, " not found in any loaded module.").c_str());
        }

        void SolutionAdapter::launchKernel(KernelInvocation const& kernel)
        {
            launchKernel(kernel, cl::CommandQueue::getDefault());
        }

        void SolutionAdapter::launchKernel(KernelInvocation const& kernel,
                                           cl::CommandQueue        stream,
                                           cl::Event*              timingEvent /* = nullptr */)
        {
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
                if(timingEvent != nullptr)
                {
                    stream.enqueueMarkerWithWaitList(nullptr, timingEvent);
                }
            }

            auto kernelFunc = getKernel(kernel.kernelName);
            {
                unsigned int argIndex = 0;
                for(auto arg : kernel.args)
                {
                    kernelFunc.setArg(argIndex++, arg.second, arg.first);
                }
            }

            stream.enqueueNDRangeKernel(
                kernelFunc,
                cl::NDRange(),
                {kernel.numWorkItems.x, kernel.numWorkItems.y, kernel.numWorkItems.z},
                {kernel.workGroupSize.x, kernel.workGroupSize.y, kernel.workGroupSize.z},
                nullptr,
                timingEvent);
        }

        void SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels)
        {
            for(auto const& k : kernels)
            {
                launchKernel(k);
            }
        }

        void SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels,
                                            cl::CommandQueue                     stream,
                                            cl::Event* timingEvent /* = nullptr */)
        {

            static std::vector<cl::Event> events(kernels.size());
            cl_uint                       eventIndex = 0;
            for(auto const& k : kernels)
            {
                launchKernel(k, stream, &(events[eventIndex++]));
            }

            stream.enqueueMarkerWithWaitList(&events, timingEvent);
        }

        void SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels,
                                            cl::CommandQueue                     stream,
                                            std::vector<cl::Event>&              timingEvents)
        {
            if(kernels.size() != timingEvents.size())
            {
                throw std::runtime_error(concatenate("Must have an equal number of kernels (",
                                                     kernels.size(),
                                                     "), and timing events. (",
                                                     timingEvents.size(),
                                                     ")"));
            }

            for(size_t i = 0; i < kernels.size(); i++)
            {
                launchKernel(kernels[i], stream, &timingEvents[i]);
            }
        }

        void SolutionAdapter::addModule(std::string const& name, cl::Program const& module)
        {
            std::lock_guard<std::mutex> guard(m_access);
            m_modules.push_back(module);
            m_loadedModuleNames.push_back(name);
        }

        void SolutionAdapter::addModules(std::string const&              groupName,
                                         std::vector<cl::Program> const& modules)
        {
            std::lock_guard<std::mutex> guard(m_access);
            m_modules.insert(m_modules.end(), modules.begin(), modules.end());
            m_loadedModuleNames.push_back(groupName);
        }

        std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter)
        {
            stream << "ocl::SolutionAdapter";

            if(adapter.m_debug)
            {
                stream << "[" << std::endl;
                for(auto const& name : adapter.m_loadedModuleNames)
                {
                    stream << name << std::endl;
                }

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
    } // namespace ocl
} // namespace TensileLite
