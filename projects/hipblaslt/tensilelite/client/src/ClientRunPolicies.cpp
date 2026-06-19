// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ClientRunPolicies.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>

#if defined(__linux__)
#include <elf.h>
#endif

namespace TensileLite
{
    namespace Client
    {
        namespace
        {
#if defined(__linux__)
            std::uintmax_t getMinKernelSizeToGwEnd(std::string const& coPath)
            {
                std::ifstream f(coPath, std::ios::binary);
                if(!f)
                    return 0;

                Elf64_Ehdr eh{};
                f.read(reinterpret_cast<char*>(&eh), sizeof(eh));
                if(!f
                   || std::memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0
                   || eh.e_ident[EI_CLASS] != ELFCLASS64)
                    return 0;

                std::vector<Elf64_Shdr> shdrs(eh.e_shnum);
                f.seekg(eh.e_shoff);
                f.read(reinterpret_cast<char*>(shdrs.data()),
                       static_cast<std::streamsize>(eh.e_shnum * sizeof(Elf64_Shdr)));
                if(!f)
                    return 0;

                Elf64_Shdr const* symSh = nullptr;
                for(auto const& sh : shdrs)
                {
                    if(sh.sh_type == SHT_SYMTAB)
                    {
                        symSh = &sh;
                        break;
                    }
                }

                if(!symSh
                   || symSh->sh_link >= shdrs.size()
                   || symSh->sh_size == 0
                   || symSh->sh_size % sizeof(Elf64_Sym) != 0)
                    return 0;

                auto const& strSh = shdrs[symSh->sh_link];

                std::vector<char> strs(strSh.sh_size);
                f.seekg(strSh.sh_offset);
                f.read(strs.data(), static_cast<std::streamsize>(strSh.sh_size));
                if(!f)
                    return 0;

                auto const            symCount = symSh->sh_size / sizeof(Elf64_Sym);
                std::vector<Elf64_Sym> syms(symCount);
                f.seekg(symSh->sh_offset);
                f.read(reinterpret_cast<char*>(syms.data()),
                       static_cast<std::streamsize>(symSh->sh_size));
                if(!f)
                    return 0;

                std::vector<std::uint64_t> kernelStarts;
                std::vector<std::uint64_t> gwEndAddrs;
                for(auto const& s : syms)
                {
                    if(s.st_name >= strs.size())
                        continue;

                    char const*   name = &strs[s.st_name];
                    unsigned char type = ELF64_ST_TYPE(s.st_info);
                    unsigned char bind = ELF64_ST_BIND(s.st_info);

                    if(type == STT_FUNC && bind == STB_GLOBAL)
                        kernelStarts.push_back(s.st_value);
                    else if(std::strcmp(name, "label_GW_End") == 0)
                        gwEndAddrs.push_back(s.st_value);
                }

                if(kernelStarts.empty() || gwEndAddrs.empty())
                    return 0;

                std::sort(kernelStarts.begin(), kernelStarts.end());

                std::uintmax_t minSize = std::numeric_limits<std::uintmax_t>::max();
                for(auto end : gwEndAddrs)
                {
                    auto it = std::upper_bound(kernelStarts.begin(), kernelStarts.end(), end);
                    if(it == kernelStarts.begin())
                        continue;

                    std::uint64_t start = *(it - 1);
                    if(end <= start)
                        continue;

                    auto sz = static_cast<std::uintmax_t>(end - start);
                    if(sz < minSize)
                        minSize = sz;
                }

                return (minSize == std::numeric_limits<std::uintmax_t>::max()) ? 0 : minSize;
            }
#endif // defined(__linux__)
        } // namespace

        RotatingOutputPlan RotatingOutputPolicy::plan(size_t warmupRuns,
                                                      size_t syncs,
                                                      size_t enqueuesPerSync) const
        {
            RotatingOutputPlan rv;
            rv.warmupRuns      = warmupRuns;
            rv.syncs           = syncs;
            rv.enqueuesPerSync = enqueuesPerSync;

            auto benchmarkBufferNum = syncs * enqueuesPerSync;
            rv.maxRotatingBufferNum = static_cast<int32_t>(
                std::max(warmupRuns, benchmarkBufferNum));
            return rv;
        }

        int IcacheRotationCursor::nextIndex(int moduleCount)
        {
            assert(moduleCount > 0);
            return static_cast<int>(m_launchIndex++ % static_cast<size_t>(moduleCount));
        }

        void IcacheRotationCursor::reset()
        {
            m_launchIndex = 0;
        }

        KernelHotPathSizeFn IcacheRotationPolicy::defaultKernelHotPathSizeFn()
        {
#if defined(__linux__)
            return [](std::string const& coPath) { return getMinKernelSizeToGwEnd(coPath); };
#else
            return [](std::string const&) { return std::uintmax_t{0}; };
#endif
        }

        bool IcacheRotationPolicy::shouldLoadAutoCopies(int requestedCopies,
                                                        int currentModuleCount) const
        {
            return requestedCopies == -1 && currentModuleCount == 1;
        }

        IcacheAutoRotationPlan IcacheRotationPolicy::computeAutoPlan(
            size_t                          inputSlotCount,
            std::vector<std::string> const& codeObjectFilenames,
            int                             rotateSizeKB,
            KernelHotPathSizeFn const&      kernelHotPathSizeFn) const
        {
            IcacheAutoRotationPlan plan;
            // Preserve the current cast-before-subtract behavior.
            plan.extrasFromDataInit = static_cast<int>(inputSlotCount) - 1;

#if defined(__linux__)
            std::uintmax_t minKernelHotPathSize = 0;
            for(auto const& filename : codeObjectFilenames)
            {
                std::uintmax_t sz = kernelHotPathSizeFn ? kernelHotPathSizeFn(filename) : 0;
                if(sz > 0 && (minKernelHotPathSize == 0 || sz < minKernelHotPathSize))
                    minKernelHotPathSize = sz;
            }

            int clampedRotateSizeKB = std::max(rotateSizeKB, 0);
            plan.kernelHotPathSize  = minKernelHotPathSize;
            plan.cacheBudgetBytes   = std::uintmax_t(clampedRotateSizeKB) * 2 * 1024;

            if(minKernelHotPathSize == 0)
            {
                plan.extrasFromCache = 0;
            }
            else if(plan.cacheBudgetBytes > minKernelHotPathSize)
            {
                plan.extrasFromCache = static_cast<int>(plan.cacheBudgetBytes / minKernelHotPathSize
                                                        - 1);
            }
#else
            plan.extrasFromCache = rotateSizeKB;
#endif

            plan.extras = std::max(plan.extrasFromDataInit, plan.extrasFromCache);
            return plan;
        }
    } // namespace Client
} // namespace TensileLite
