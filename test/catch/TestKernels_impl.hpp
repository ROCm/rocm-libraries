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
