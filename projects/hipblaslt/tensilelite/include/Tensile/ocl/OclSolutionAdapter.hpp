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

#ifndef OCL_SOLUTION_ADAPTER_HPP
#define OCL_SOLUTION_ADAPTER_HPP

#include <Tensile/Tensile.hpp>
#include <Tensile/ocl/OclFwd.hpp>

#include <mutex>

namespace TensileLite
{
    namespace ocl
    {
        class SolutionAdapter final : public TensileLite::SolutionAdapter
        {
        public:
            SolutionAdapter();
            SolutionAdapter(bool debug);
            SolutionAdapter(bool debug, std::string const& name);
            SolutionAdapter(bool               debug,
                            std::string const& name,
                            cl::Context        context,
                            cl::Device         device);
            ~SolutionAdapter() final = default;

            std::string name() const final;

            void loadCodeObjectFile(std::string const& path);
            void loadCodeObjectBytes(std::vector<uint8_t> const& bytes);

            void loadEmbeddedCodeObjects();
            void loadEmbeddedCodeObjects(std::string const& key);

            void launchKernel(KernelInvocation const& kernel);
            void launchKernel(KernelInvocation const& kernel,
                              cl::CommandQueue        stream,
                              cl::Event*              timingEvent = nullptr);

            void launchKernels(std::vector<KernelInvocation> const& kernels);
            void launchKernels(std::vector<KernelInvocation> const& kernels,
                               cl::CommandQueue                     stream,
                               cl::Event*                           timingEvent = nullptr);

            void launchKernels(std::vector<KernelInvocation> const& kernels,
                               cl::CommandQueue                     stream,
                               std::vector<cl::Event>&              timingEvents);

            void initKernel(std::string const& name);

        private:
            cl::Kernel getKernel(std::string const& name);
            void       addModule(std::string const& name, cl::Program const& module);
            void addModules(std::string const& groupName, std::vector<cl::Program> const& modules);

            std::mutex m_access;

            std::vector<cl::Program>                    m_modules;
            cl::Context                                 m_context;
            cl::Device                                  m_device;
            std::unordered_map<std::string, cl::Kernel> m_kernels;
            bool                                        m_debug           = false;
            bool                                        m_debugSkipLaunch = false;
            std::string                                 m_name            = "OclSolutionAdapter";

            std::vector<std::string> m_loadedModuleNames;

            friend std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter);
        };

        std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter);
        std::ostream& operator<<(std::ostream& stream, std::shared_ptr<SolutionAdapter> const& ptr);

    } // namespace ocl
} // namespace TensileLite

#endif //OCL_SOLUTION_ADAPTER_HPP
