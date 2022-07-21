
#include "GPUContextFixture.hpp"

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>

void getSupportedISAs()
{
    auto m_supported_isas = rocRoller::GPUArchitectureLibrary::getAllSupportedISAs();
}

void CurrentGPUContextFixture::SetUp()
{
    using namespace rocRoller;

    ASSERT_THAT(hipInit(0), HasHipSuccess(0));
    ASSERT_THAT(hipSetDevice(0), HasHipSuccess(0));

    ContextFixture::SetUp();

    ASSERT_EQ(true, m_context->targetArchitecture().HasCapability(GPUCapability::SupportedISA));
    ASSERT_EQ(true, isLocalDevice());
}

rocRoller::ContextPtr CurrentGPUContextFixture::createContext()
{
    return rocRoller::Context::ForDefaultHipDevice(testKernelName());
}

void GPUContextFixture::SetUp()
{
    using namespace rocRoller;
    ContextFixture::SetUp();

    ASSERT_EQ(true, m_context->targetArchitecture().HasCapability(GPUCapability::SupportedISA));

    if(isLocalDevice())
    {
        int deviceIdx = m_context->hipDeviceIndex();

        ASSERT_THAT(hipInit(0), HasHipSuccess(0));
        ASSERT_THAT(hipSetDevice(deviceIdx), HasHipSuccess(0));
    }
}

rocRoller::ContextPtr GPUContextFixture::createContext()
{
    using namespace rocRoller;

    auto device = GetParam();
    auto target = GPUArchitectureTarget(device);

    auto currentDevice = GPUArchitectureLibrary::GetDefaultHipDeviceArch();

    bool localDevice = currentDevice.target() == target;

    if(localDevice)
    {
        return Context::ForDefaultHipDevice(testKernelName());
    }
    else
    {
        return Context::ForTarget(target, testKernelName());
    }
}
