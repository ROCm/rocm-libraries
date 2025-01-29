
#pragma once

#include "CustomAssertions.hpp"
#include "TestKernels.hpp"

template <int Idx, typename... Args>
void AssemblyTestKernel::appendArgs(rocRoller::KernelArguments& kargs,
                                    std::tuple<Args...> const&  args)
{
    std::string name = "";
    if(kargs.log())
        name = "arg" + std::to_string(Idx);

    if constexpr(Idx < sizeof...(Args))
    {
        kargs.append(name, std::get<Idx>(args));

        if constexpr((Idx + 1) < (sizeof...(Args)))
        {
            appendArgs<Idx + 1>(kargs, args);
        }
    }
}

template <typename... Args>
void AssemblyTestKernel::operator()(rocRoller::KernelInvocation const& invocation,
                                    Args const&... args)
{
    REQUIRE_TEST_TAG("gpu");
    auto kernel = getExecutableKernel();

    bool log = rocRoller::Log::getLogger()->should_log(rocRoller::LogLevel::Debug);

    rocRoller::KernelArguments kargs(log);
    appendArgs(kargs, std::forward_as_tuple(args...));

    kernel->executeKernel(kargs, invocation);
}
