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

#pragma once

#include <Tensile/AMDGPU.hpp>
#include <Tensile/Tensile.hpp>
#include <hip/hip_runtime.h>
#include <unordered_set>

#include <atomic>
#include <mutex>

#include <Tensile/Macros.hpp>

TENSILE_HIDDEN_BEGIN

namespace TensileLite
{
    namespace hip
    {
        class SolutionAdapter : public TensileLite::SolutionAdapter
        {
        public:
            SolutionAdapter();
            SolutionAdapter(bool debug);
            SolutionAdapter(bool debug, std::string const& name);
            ~SolutionAdapter();

            virtual std::string name() const
            {
                return m_name;
            }

            void codeObjectDir(std::string codeObjectDir);

            hipError_t loadCodeObjectFile(std::string const& path);

            hipError_t initializeLazyLoading(std::string architecture, std::string codeObjectDir);

            hipError_t loadCodeObject(const void* image);

            hipError_t loadCodeObjectBytes(std::vector<uint8_t> const& bytes);

            void loadEmbeddedCodeObjects();
            void loadEmbeddedCodeObjects(std::string const& key);

            hipError_t launchKernel(KernelInvocation const& kernel);
            hipError_t launchKernel(KernelInvocation const& kernel,
                                    hipStream_t             stream,
                                    hipEvent_t              startEvent,
                                    hipEvent_t              stopEvent,
                                    bool                    isKernelLoaded = false);

            hipError_t launchKernels(std::vector<KernelInvocation> const& kernels);

            hipError_t launchKernels(std::vector<KernelInvocation> const& kernels,
                                     hipStream_t                          stream,
                                     hipEvent_t                           startEvent,
                                     hipEvent_t                           stopEvent,
                                     bool                                 isKernelLoaded = false);

            hipError_t launchKernels(std::vector<KernelInvocation> const& kernels,
                                     hipStream_t                          stream,
                                     std::vector<hipEvent_t> const&       startEvents,
                                     std::vector<hipEvent_t> const&       stopEvents);

            bool FindCodeObject(std::string const& codeObjectFile);

            hipError_t initKernel(std::string const& name);

            hipError_t initKernels(std::vector<std::string> const& kernelNames);

            // I-cache rotation: load `extraCopies` additional independent
            // hipModule_t handles for `path`. Each call to hipModuleLoad
            // allocates fresh device memory for the code object, so the
            // resulting handles map to distinct device PCs - the prerequisite
            // for I-cache cold misses on rotation. The original handle (if
            // any) is left untouched; this function only appends extras.
            hipError_t loadCodeObjectFileExtraCopies(std::string const& path, int extraCopies);

            // Select which rotation copy the next launchKernel uses.
            // idx 0 = original m_modules (default). idx>0 selects the
            // matching slot in m_extraModuleCopies[idx - 1].
            void selectRotationCopy(int idx);

            // Total module sets available for rotation
            // (1 = original only, 2 = +1 extra copy, etc.).
            int numRotationModules();

        private:
            hipError_t getKernel(hipFunction_t& rv, std::string const& name);

            std::mutex m_access;

            std::vector<hipModule_t>                       m_modules;
            std::unordered_map<std::string, hipFunction_t> m_kernels;
            bool                                           m_debug           = false;
            bool                                           m_debugSkipLaunch = false;
            std::string                                    m_name            = "HipSolutionAdapter";
            std::string                                    m_codeObjectDirectory;

            std::vector<std::string>        m_loadedModuleNames;
            std::unordered_set<std::string> m_loadedCOFiles;

            // Extra rotation copies for I-cache cold-miss testing.
            //   m_extraModuleCopies[i] is parallel to m_modules and holds
            //     independent hipModule_t handles, produced by re-loading every
            //     already-loaded .co file once per extra copy.
            //   m_extraKernels[i] is the per-copy hipFunction_t cache, mirroring
            //     m_kernels (avoids calling hipModuleGetFunction on every launch).
            //   m_currentRotationCopy selects which copy the next getKernel uses:
            //     0      -> original m_modules / m_kernels;
            //     k > 0  -> m_extraModuleCopies[k-1] / m_extraKernels[k-1].
            //   std::atomic<int> lets selectRotationCopy() update without locking
            //     and getKernel() snapshot the index without locking; the actual
            //     module/cache search is still guarded by m_access.
            std::vector<std::vector<hipModule_t>>                       m_extraModuleCopies;
            std::vector<std::unordered_map<std::string, hipFunction_t>> m_extraKernels;
            std::atomic<int>                                            m_currentRotationCopy{0};

            // Record for module reload
            std::string m_lazyLoadArchitecture;

            friend std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter);
        };

        std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter);
        std::ostream& operator<<(std::ostream& stream, std::shared_ptr<SolutionAdapter> const& ptr);
    } // namespace hip
} // namespace TensileLite

TENSILE_HIDDEN_END
