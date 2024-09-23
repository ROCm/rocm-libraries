
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
    auto kernelName = KernelName(arch, params...);
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
    char underscore = '_';
    char replaced   = '*';

    // Replace any character that isn't alphanumeric with underscore,
    // or one of a few special cases.

    std::map<char, char> specialCharacters = {{'+', 'p'}, {'-', 'm'}};
    for(auto& c : name)
    {
        if(!isalnum(c) && c != underscore)
        {
            auto iter = specialCharacters.find(c);
            if(iter != specialCharacters.end())
                c = iter->second;
            else
                c = replaced;
        }
    }

    // Delete any trailing '*'s.

    while(!name.empty() && name.back() == replaced)
        name.pop_back();

    std::string rv;
    rv.reserve(name.size());

    // Delete any leading '*'s, as well as any duplicate '*'s.

    for(auto const& c : name)
    {
        if(c != replaced || (!rv.empty() && rv.back() != replaced))
        {
            rv += c;
        }
    }

    // If the name is now completely empty, return '_'.
    if(rv.empty())
        rv += replaced;

    // Replace any '*' chars with '_'.
    for(auto& c : rv)
    {
        if(c == replaced)
            c = underscore;
    }

    return rv;
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
