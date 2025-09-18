// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "rtc_subprocess.h"
#include "../../shared/environment.h"
#include "../../shared/subprocess.h"
#include "library_path.h"

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

namespace fs = std::filesystem;

#ifdef WIN32
static const char* HELPER_EXE = "rocfft_rtc_helper.exe";
#else
static const char* HELPER_EXE = "rocfft_rtc_helper";
#endif

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#define ROCFFT_VERSION_STRING TO_STR(ROCFFT_VERSION)

static fs::path find_rtc_helper()
{
    // candidate directories for the helper
    std::vector<fs::path> helper_dirs;

    auto var = rocfft_getenv("ROCFFT_RTC_PROCESS_HELPER");
    if(!var.empty())
        return var;

    fs::path library_path = get_library_path();
    if(!library_path.empty())
    {
        // try same dir as library
        fs::path library_parent_path = library_path.parent_path();
        helper_dirs.push_back(library_parent_path);

        // try in a versioned library subdirectory
        fs::path subdir_path = library_parent_path / "rocfft" / ROCFFT_VERSION_STRING;
        helper_dirs.push_back(subdir_path);

        // look for helper in the candidate directories
        for(const auto& dir : helper_dirs)
        {
            auto helper_path = dir / HELPER_EXE;
            if(fs::exists(helper_path))
                return helper_path;
        }
    }
    throw std::runtime_error("unable to find rtc helper");
}

std::vector<char> compile_subprocess(const std::string& kernel_src, const std::string& gpu_arch)
{
    static std::string rtc_helper_exe = find_rtc_helper().string();

    // HACK: on Windows, rtc_helper_exe seems to have an embedded NUL
    // byte at the end.  Append c_str() to hide this.
    auto code = execute_subprocess(rtc_helper_exe.c_str(), {gpu_arch}, kernel_src);
    if(code.empty())
    {
        throw std::runtime_error("child process failed to produce code");
    }
    return code;
}
