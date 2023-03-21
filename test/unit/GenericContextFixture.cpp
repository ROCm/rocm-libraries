
#include "GenericContextFixture.hpp"

#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>

void GenericContextFixture::SetUp()
{
    using namespace rocRoller;
    ContextFixture::SetUp();

    EXPECT_EQ(true, m_context->targetArchitecture().HasCapability(GPUCapability::SupportedISA));
}

rocRoller::ContextPtr GenericContextFixture::createContext()
{
    using namespace rocRoller;

    GPUArchitectureTarget target(targetArchitecture());
    return Context::ForTarget(target, testKernelName(), m_kernelOptions);
}

std::string GenericContextFixture::targetArchitecture()
{
    return "gfx1012:xnack+";
}
