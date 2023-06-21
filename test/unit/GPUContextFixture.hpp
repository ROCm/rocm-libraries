#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ContextFixture.hpp"
#include <rocRoller/Utilities/Utils.hpp>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

/**
 * Returns a (googletest) Generator that will yield every GPU ISA supported by rocRoller.
 *
 * Useful if you want to parameterize a test with combinations of each ISA with other
 * parameters. Example:
 * INSTANTIATE_TEST_SUITE_P(SuiteName,
 *                          FixtureClass,
 *                          ::testing::Combine(supportedISAValues(),
 *                                             ::testing::Values(1, 2, 4, 8, 12, 16, 20, 44)));
 */
inline auto supportedISAValues()
{
    return ::testing::ValuesIn(rocRoller::GPUArchitectureLibrary::getAllSupportedISAs());
}

/**
 * Returns a (googletest) Generator that will yield every GPU ISA supported by rocRoller, that
 * has MFMA instructions.
 *
 * Useful if you want to parameterize a test with combinations of each ISA with other
 * parameters. Example:
 * INSTANTIATE_TEST_SUITE_P(SuiteName,
 *                          FixtureClass,
 *                          ::testing::Combine(mfmaSupportedISAValues(),
 *                                             ::testing::Values(1, 2, 4, 8, 12, 16, 20, 44)));
 */
inline auto mfmaSupportedISAValues()
{
    return ::testing::ValuesIn(rocRoller::GPUArchitectureLibrary::getMFMASupportedISAs());
}

/**
 * Returns a (googletest) Generator that will yield just the local GPU ISA.
 *
 * Useful if you want to parameterize a test with combinations of each ISA with other parameters. Example:
 * INSTANTIATE_TEST_SUITE_P(SuiteName,
 *                          FixtureClass,
 *                          ::testing::Combine(currentGPUISA(),
 *                                             ::testing::Values(1, 2, 4, 8, 12, 16, 20, 44)));
 */
inline auto currentGPUISA()
{
    auto currentDevice = rocRoller::GPUArchitectureLibrary::GetDefaultHipDeviceArch();
    return ::testing::Values(currentDevice.target().toString());
}

/**
 * Returns a (googletest) Generator that will yield a single-item tuple for every GPU ISA supported by rocRoller.
 *
 * Useful if you want to parameterize a test with only each supported ISA. Example:
 * INSTANTIATE_TEST_SUITE_P(SuiteName,
 *                          FixtureClass,
 *                          supportedISATuples());
 */
inline auto supportedISATuples()
{
    return ::testing::Combine(supportedISAValues());
}

/**
 * Returns a (googletest) Generator that will yield a single-item tuple for every GPU ISA supported by rocRoller.
 *
 * Useful if you want to parameterize a test with only each supported ISA. Example:
 * INSTANTIATE_TEST_SUITE_P(SuiteName,
 *                          FixtureClass,
 *                          mfmaSupportedISATuples());
 */
inline auto mfmaSupportedISATuples()
{
    return ::testing::Combine(mfmaSupportedISAValues());
}

class BaseGPUContextFixture : public ContextFixture
{
protected:
    void SetUp() override;

    rocRoller::ContextPtr createContextLocalDevice();
    rocRoller::ContextPtr createContextForArch(std::string const& device);
};

class CurrentGPUContextFixture : public BaseGPUContextFixture
{
protected:
    virtual rocRoller::ContextPtr createContext() override;
};

template <typename... Ts>
class GPUContextFixtureParam : public BaseGPUContextFixture,
                               public ::testing::WithParamInterface<std::tuple<std::string, Ts...>>
{
protected:
    virtual rocRoller::ContextPtr createContext() override
    {
        std::string device = std::get<0>(this->GetParam());

        return this->createContextForArch(device);
    }
};

using GPUContextFixture = GPUContextFixtureParam<>;

#define REQUIRE_ARCH_CAP(cap)                                                                 \
    do                                                                                        \
    {                                                                                         \
        if(!m_context->targetArchitecture().HasCapability(cap))                               \
        {                                                                                     \
            GTEST_SKIP() << m_context->targetArchitecture().target() << " has no capability " \
                         << cap << std::endl;                                                 \
        }                                                                                     \
    } while(0)

#define REQUIRE_NOT_ARCH_CAP(cap)                                                                 \
    do                                                                                            \
    {                                                                                             \
        if(m_context->targetArchitecture().HasCapability(cap))                                    \
        {                                                                                         \
            GTEST_SKIP() << m_context->targetArchitecture().target() << " has capability " << cap \
                         << std::endl;                                                            \
        }                                                                                         \
    } while(0)
