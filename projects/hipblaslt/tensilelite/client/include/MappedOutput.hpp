// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/hip/HipUtils.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        // One output allocation whose pages are split into equal contiguous chunks, one per
        // device: chunk i is physical memory of deviceIds[i], and every device can read and write
        // the whole range. A kernel on one partition writes its result straight into the pages of
        // whichever partition owns them, so no copy is needed afterwards.
        class MappedOutput
        {
        public:
            MappedOutput(std::vector<int> const& deviceIds, size_t bytes)
                : m_deviceIds(deviceIds)
            {
                const size_t count = deviceIds.size();

                hipMemAllocationProp prop{};
                prop.type          = hipMemAllocationTypePinned;
                prop.location.type = hipMemLocationTypeDevice;

                size_t granularity = 1;
                for(int id : deviceIds)
                {
                    prop.location.id = id;
                    size_t g         = 0;
                    HIP_CHECK_EXC(hipMemGetAllocationGranularity(
                        &g, &prop, hipMemAllocationGranularityRecommended));
                    granularity = std::max(granularity, g);
                }

                const size_t perDevice = (bytes + count - 1) / count;
                m_chunkBytes = (perDevice + granularity - 1) / granularity * granularity;
                m_totalBytes = m_chunkBytes * count;

                HIP_CHECK_EXC(hipMemAddressReserve(&m_base, m_totalBytes, granularity, nullptr, 0));

                std::vector<hipMemAccessDesc> access(count);
                for(size_t i = 0; i < count; i++)
                {
                    prop.location.id = deviceIds[i];
                    hipMemGenericAllocationHandle_t handle{};
                    HIP_CHECK_EXC(hipMemCreate(&handle, m_chunkBytes, &prop, 0));
                    m_handles.push_back(handle);
                    HIP_CHECK_EXC(
                        hipMemMap(chunk(i), m_chunkBytes, 0, handle, 0));
                    access[i].location.type = hipMemLocationTypeDevice;
                    access[i].location.id   = deviceIds[i];
                    access[i].flags         = hipMemAccessFlagsProtReadWrite;
                }
                HIP_CHECK_EXC(hipMemSetAccess(m_base, m_totalBytes, access.data(), count));

                int previous = 0;
                static_cast<void>(hipGetDevice(&previous));
                HIP_CHECK_EXC(hipSetDevice(deviceIds[0]));
                HIP_CHECK_EXC(hipMemset(m_base, 0, m_totalBytes));
                static_cast<void>(hipSetDevice(previous));
            }

            ~MappedOutput()
            {
                for(size_t i = 0; i < m_handles.size(); i++)
                {
                    static_cast<void>(hipMemUnmap(chunk(i), m_chunkBytes));
                    static_cast<void>(hipMemRelease(m_handles[i]));
                }
                if(m_base != nullptr)
                    static_cast<void>(hipMemAddressFree(m_base, m_totalBytes));
            }

            MappedOutput(MappedOutput const&)            = delete;
            MappedOutput& operator=(MappedOutput const&) = delete;

            void* data() const
            {
                return m_base;
            }

        private:
            void* chunk(size_t i) const
            {
                return static_cast<char*>(m_base) + i * m_chunkBytes;
            }

            std::vector<int>                              m_deviceIds;
            std::vector<hipMemGenericAllocationHandle_t> m_handles;
            void*                                        m_base       = nullptr;
            size_t                                       m_chunkBytes = 0;
            size_t                                       m_totalBytes = 0;
        };
    } // namespace Client
} // namespace TensileLite
