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

#include "TestContext.hpp"

#include <rocRoller/Utilities/Utils.hpp>

#include <catch2/catch_test_macros.hpp>

inline TestContext::TestContext(rocRoller::ContextPtr context)
    : m_context(context)
{
}

inline TestContext::~TestContext()
{
    m_context.reset();
    rocRoller::Settings::reset();
}

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

inline std::vector<rocRoller::Register::ValuePtr>
    TestContext::createRegisters(rocRoller::Register::Type const        regType,
                                 rocRoller::DataType const              dataType,
                                 size_t const                           amount,
                                 int const                              regCount,
                                 rocRoller::Register::AllocationOptions allocOptions)
{
    std::vector<rocRoller::Register::ValuePtr> regs;
    for(size_t i = 0; i < amount; i++)
    {
        auto reg = std::make_shared<rocRoller::Register::Value>(
            m_context, regType, dataType, regCount, allocOptions);
        try
        {
            reg->allocateNow();
        }
        catch(...)
        {
            std::cout << i << std::endl;
            throw;
        }
        regs.push_back(reg);
    }
    return regs;
}
