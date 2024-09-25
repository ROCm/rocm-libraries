#pragma once

#include "ContextFixture.hpp"
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

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
    return ::testing::ValuesIn(
        rocRoller::GPUArchitectureLibrary::getInstance()->getAllSupportedISAs());
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
    return ::testing::ValuesIn(
        rocRoller::GPUArchitectureLibrary::getInstance()->getMFMASupportedISAs());
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
    if(rocRoller::GPUArchitectureLibrary::getInstance()->HasHipDevice())
    {
        auto currentDevice
            = rocRoller::GPUArchitectureLibrary::getInstance()->GetDefaultHipDeviceArch();
        return ::testing::Values(currentDevice.target());
    }
    else
    {
        // Give a dummy device
        return ::testing::Values(
            rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX90A});
    }
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
    rocRoller::ContextPtr createContextForArch(rocRoller::GPUArchitectureTarget const& device);
};

class CurrentGPUContextFixture : public BaseGPUContextFixture
{
protected:
    virtual rocRoller::ContextPtr createContext() override;
};

template <typename... Ts>
class GPUContextFixtureParam
    : public BaseGPUContextFixture,
      public ::testing::WithParamInterface<std::tuple<rocRoller::GPUArchitectureTarget, Ts...>>
{
protected:
    virtual rocRoller::ContextPtr createContext() override
    {
        rocRoller::GPUArchitectureTarget device = std::get<0>(this->GetParam());
        return this->createContextForArch(device);
    }
};

using GPUContextFixture = GPUContextFixtureParam<>;

#define REQUIRE_ARCH_CAP(cap)                                                   \
    do                                                                          \
    {                                                                           \
        if(!m_context->targetArchitecture().HasCapability(cap))                 \
        {                                                                       \
            GTEST_SKIP() << m_context->targetArchitecture().target().toString() \
                         << " has no capability " << cap << std::endl;          \
        }                                                                       \
    } while(0)

#define REQUIRE_NOT_ARCH_CAP(cap)                                               \
    do                                                                          \
    {                                                                           \
        if(m_context->targetArchitecture().HasCapability(cap))                  \
        {                                                                       \
            GTEST_SKIP() << m_context->targetArchitecture().target().toString() \
                         << " has capability " << cap << std::endl;             \
        }                                                                       \
    } while(0)
