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
 * Returns a (googletest) Generator that will yield every CDNA GPU ISA supported by rocRoller.
 *
 * Useful if you want to parameterize a test with combinations of each ISA with other
 * parameters. Example:
 * INSTANTIATE_TEST_SUITE_P(SuiteName,
 *                          FixtureClass,
 *                          ::testing::Combine(CDNAISAValues(),
 *                                             ::testing::Values(1, 2, 4, 8, 12, 16, 20, 44)));
 */
inline auto CDNAISAValues()
{
    return ::testing::ValuesIn(rocRoller::GPUArchitectureLibrary::getInstance()->getCDNAISAs());
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
 * Returns a (googletest) Generator that will yield every GPU ISA supported by rocRoller, that
 * has WMMA instructions.
 *
 * Useful if you want to parameterize a test with combinations of each ISA with other
 * parameters. Example:
 * INSTANTIATE_TEST_SUITE_P(SuiteName,
 *                          FixtureClass,
 *                          ::testing::Combine(wmmaSupportedISAValues(),
 *                                             ::testing::Values(1, 2, 4, 8, 12, 16, 20, 44)));
 */
inline auto wmmaSupportedISAValues()
{
    return ::testing::ValuesIn(
        rocRoller::GPUArchitectureLibrary::getInstance()->getWMMASupportedISAs());
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
 * Returns a (googletest) Generator that will yield a single-item tuple for CDNA GPU ISA supported by rocRoller.
 *
 * Useful if you want to parameterize a test with only each supported ISA. Example:
 * INSTANTIATE_TEST_SUITE_P(SuiteName,
 *                          FixtureClass,
 *                          CDNAISATuples());
 */
inline auto CDNAISATuples()
{
    return ::testing::Combine(CDNAISAValues());
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

inline auto wmmaSupportedISATuples()
{
    return ::testing::Combine(wmmaSupportedISAValues());
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

template <typename... Caps>
bool hasAnyOfTheseArchCaps(rocRoller::ContextPtr const context, Caps... caps)
{
    static_assert(sizeof...(caps) > 0);
    return (... || context->targetArchitecture().HasCapability(caps));
}

#define REQUIRE_ANY_OF_ARCH_CAP(...)                                                      \
    do                                                                                    \
    {                                                                                     \
        if(not hasAnyOfTheseArchCaps(m_context, __VA_ARGS__))                             \
        {                                                                                 \
            GTEST_SKIP() << m_context->targetArchitecture().target().toString()           \
                         << " has no capability " << concatenate_join(", ", __VA_ARGS__); \
        }                                                                                 \
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
