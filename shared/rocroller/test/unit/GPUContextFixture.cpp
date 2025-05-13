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

#include "GPUContextFixture.hpp"

#include "Utilities.hpp"

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>

rocRoller::ContextPtr BaseGPUContextFixture::createContextLocalDevice()
{
    return rocRoller::Context::ForDefaultHipDevice(testKernelName(), m_kernelOptions);
}

void BaseGPUContextFixture::SetUp()
{
    using namespace rocRoller;

    if(!isLocalDevice())
    {
        Settings::getInstance()->set(Settings::AllowUnknownInstructions, true);
    }
    ContextFixture::SetUp();

    ASSERT_EQ(true, m_context->targetArchitecture().HasCapability(GPUCapability::SupportedISA));

    if(isLocalDevice())
    {
        int deviceIdx = m_context->hipDeviceIndex();

        ASSERT_THAT(hipInit(0), HasHipSuccess(0));
        ASSERT_THAT(hipSetDevice(deviceIdx), HasHipSuccess(0));
    }
}

rocRoller::ContextPtr
    BaseGPUContextFixture::createContextForArch(rocRoller::GPUArchitectureTarget const& device)
{
    using namespace rocRoller;

    auto currentDevice = GPUArchitectureLibrary::getInstance()->GetDefaultHipDeviceArch();

    bool localDevice = currentDevice.target() == device;

    if(localDevice)
    {
        return Context::ForDefaultHipDevice(testKernelName(), m_kernelOptions);
    }
    else
    {
        return Context::ForTarget(device, testKernelName(), m_kernelOptions);
    }
}

rocRoller::ContextPtr CurrentGPUContextFixture::createContext()
{
    return createContextLocalDevice();
}
