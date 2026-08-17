// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GenericContextFixture.hpp"

#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>

void GenericContextFixture::SetUp()
{
    using namespace rocRoller;
    ContextFixture::SetUp();

    EXPECT_EQ(true, m_context->targetArchitecture().HasCapability(GPUCapability::SupportedISA));
}

rocRoller::GPUArchitectureTarget GenericContextFixture::targetArchitecture()
{
    return rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX908};
}

rocRoller::ContextPtr GenericContextFixture::createContext()
{
    using namespace rocRoller;

    return Context::ForTarget(targetArchitecture(), testKernelName(), m_kernelOptions);
}
