// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "TestContext.hpp"

#include <rocRoller/Utilities/Utils.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>

template <typename... Params>
TestContext TestContext::ForTestDevice(rocRoller::KernelOptions const& kernelOpts,
                                       Params const&... params)
{
    auto kernelName = KernelName(params...);
    auto ctx        = rocRoller::Context::ForDefaultHipDevice(kernelName, kernelOpts);
    return {ctx};
}

template <typename... Params>
TestContext TestContext::ForTarget(rocRoller::GPUArchitectureTarget const& arch,
                                   rocRoller::KernelOptions const&         kernelOpts,
                                   Params const&... params)
{
    auto kernelName = KernelName(arch, params...);
    auto ctx        = rocRoller::Context::ForTarget(arch, kernelName, kernelOpts);
    return {ctx};
}

template <typename... Params>
TestContext TestContext::ForTarget(rocRoller::GPUArchitecture const& arch,
                                   rocRoller::KernelOptions const&   kernelOpts,
                                   Params const&... params)
{
    auto kernelName = KernelName(arch.target(), params...);
    auto ctx        = rocRoller::Context::ForTarget(arch, kernelName, kernelOpts);
    return {ctx};
}

template <typename... Params>
TestContext TestContext::ForDefaultTarget(rocRoller::KernelOptions const& kernelOpts,
                                          Params const&... params)
{
    return ForTarget({rocRoller::GPUArchitectureGFX::GFX90A}, kernelOpts, params...);
}

template <typename... Params>
std::string TestContext::KernelName(Params const&... params)
{
    auto name = rocRoller::concatenate_join(
        " ", Catch::getResultCapture().getCurrentTestName(), params...);

    return EscapeKernelName(name);
}

/**
 * Converts 'name' into a valid C identifier.
 */
inline std::string TestContext::EscapeKernelName(std::string name)
{
    return rocRoller::escapeSymbolName(name);
}

inline rocRoller::ContextPtr TestContext::get()
{
    return m_context;
}

inline rocRoller::Context& TestContext::operator*()
{
    return *m_context;
}
inline rocRoller::Context* TestContext::operator->()
{
    return m_context.get();
}

inline std::string TestContext::output()
{
    return m_context->instructions()->toString();
}
