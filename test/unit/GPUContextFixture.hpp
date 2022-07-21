#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ContextFixture.hpp"
#include "Utilities.hpp"
#include <rocRoller/ExecutableKernel.hpp>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

class CurrentGPUContextFixture : public ContextFixture
{
protected:
    void SetUp() override;

    virtual rocRoller::ContextPtr createContext() override;
};

class GPUContextFixture : public ContextFixture, public ::testing::WithParamInterface<std::string>
{
protected:
    void SetUp() override;

    virtual rocRoller::ContextPtr createContext() override;
};

#define REQUIRE_ARCH_CAP(cap)                                                                 \
    do                                                                                        \
    {                                                                                         \
        if(!m_context->targetArchitecture().HasCapability(cap))                               \
        {                                                                                     \
            GTEST_SKIP() << m_context->targetArchitecture().target() << " has no capability " \
                         << cap << std::endl;                                                 \
        }                                                                                     \
    } while(0)
