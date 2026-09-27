/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#include <miopen/config.h>
#include <miopen/kernel_warnings.hpp>
#include <miopen/stringutils.hpp>

#include <iterator>
#include <numeric>
#include <sstream>

namespace miopen {

static std::vector<std::string> HipKernelWarnings()
{
    std::vector<std::string> rv = {
        "-Weverything",
        "-Wno-c++98-compat",
        "-Wno-c++98-compat-pedantic",
        "-Wno-conversion",
        "-Wno-double-promotion",
        "-Wno-exit-time-destructors",
        "-Wno-extra-semi",
        "-Wno-extra-semi-stmt",
        "-Wno-float-conversion",
        "-Wno-gnu-anonymous-struct",
        "-Wno-gnu-zero-variadic-macro-arguments",
        "-Wno-missing-noreturn", // Workaround for HCC
        "-Wno-missing-prototypes",
        "-Wno-nested-anon-types",
        "-Wno-padded",
        "-Wno-return-std-move-in-c++11",
        "-Wno-shorten-64-to-32",
        "-Wno-sign-conversion",
        "-Wno-unknown-warning-option",
        "-Wno-unused-command-line-argument",
        "-Wno-weak-vtables",
        "-Wno-covered-switch-default",
        "-Wno-disabled-macro-expansion",
        "-Wno-undefined-reinterpret-cast",
#if HIP_PACKAGE_VERSION_FLAT >= 6001024000ULL
        "-Wno-unsafe-buffer-usage",
#endif
    };

    return rv;
}

static std::string MakeKernelWarningsString(const std::vector<std::string>& kernel_warnings,
                                            const std::string& prefix)
{
    return prefix + JoinStrings(kernel_warnings, prefix);
}

const std::string& HipKernelWarningsString()
{
    const std::string prefix = " ";

    static const std::string result = MakeKernelWarningsString(HipKernelWarnings(), prefix);
    return result;
}

} // namespace miopen
