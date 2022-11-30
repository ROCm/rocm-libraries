
#include "GPUContextFixture.hpp"

#include "Utilities.hpp"

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>

rocRoller::ContextPtr BaseGPUContextFixture::createContextLocalDevice()
{
    return rocRoller::Context::ForDefaultHipDevice(testKernelName());
}

void BaseGPUContextFixture::SetUp()
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

rocRoller::ContextPtr BaseGPUContextFixture::createContextForArch(std::string const& device)
{
    using namespace rocRoller;

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

rocRoller::ContextPtr CurrentGPUContextFixture::createContext()
{
    return createContextLocalDevice();
}
