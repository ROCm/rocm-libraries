// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/Tensile_fwd.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <hip/hip_runtime.h>

#include <memory>

namespace TensileLite
{
    namespace Client
    {
        class DataInitialization;

        class ScopedDevice
        {
        public:
            explicit ScopedDevice(int deviceId)
            {
                HIP_CHECK_EXC(hipGetDevice(&m_prevDeviceId));
                HIP_CHECK_EXC(hipSetDevice(deviceId));
            }

            ~ScopedDevice()
            {
                static_cast<void>(hipSetDevice(m_prevDeviceId));
            }

            ScopedDevice(ScopedDevice const&)            = delete;
            ScopedDevice& operator=(ScopedDevice const&) = delete;

        private:
            int m_prevDeviceId = 0;
        };

        struct DeviceContext
        {
            DeviceContext(int deviceId, bool useDefaultStream)
                : deviceId(deviceId)
                , adapter(std::make_shared<hip::SolutionAdapter>())
            {
                if(useDefaultStream)
                    return;
                ScopedDevice guard(deviceId);
                HIP_CHECK_EXC(hipStreamCreate(&stream));
            }

            ~DeviceContext()
            {
                if(stream != nullptr)
                    static_cast<void>(hipStreamDestroy(stream));
            }

            DeviceContext(DeviceContext const&)            = delete;
            DeviceContext& operator=(DeviceContext const&) = delete;

            int                                   deviceId;
            hipStream_t                           stream = nullptr;
            std::shared_ptr<hip::SolutionAdapter> adapter;
            std::shared_ptr<DataInitialization>   dataInit;
            std::shared_ptr<ProblemInputs>        inputs;
        };
    } // namespace Client
} // namespace TensileLite
