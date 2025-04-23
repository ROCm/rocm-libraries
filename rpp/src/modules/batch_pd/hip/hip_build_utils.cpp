/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <sstream>
#include <optional>

#include "hip_build_utils.hpp"
#include "logger.hpp"

#define RPP_STRINGIZE_1(...) #__VA_ARGS__
#define RPP_STRINGIZE(...) RPP_STRINGIZE_1(__VA_ARGS__)

namespace rpp {

fs::path HipBuild(std::optional<TmpDir>& tmp_dir,
                                 const std::string& filename,
                                 std::string src,
                                 std::string params,
                                 const std::string& dev_name)
{
#ifdef __linux__
    if(dev_name.find("gfx80") != std::string::npos)
        RPP_THROW("HIP kernel are not supported on " + dev_name + " architecture");
    // write out the include files
    auto inc_list = GetKernelIncList();
    auto inc_path = tmp_dir->path;
    fs::create_directories(inc_path);
    for(auto inc_file : inc_list)
    {
        auto inc_src = GetKernelInc(inc_file);
        WriteFile(inc_src, inc_path / inc_file);
    }
    src += "\nint main() {}\n";
    WriteFile(src, tmp_dir->path / filename);
    params += " -amdgpu-target=" + dev_name;
    // params += " -Wno-unused-command-line-argument -c -fno-gpu-rdc -I. ";
    params += " -Wno-unused-command-line-argument -I. ";
   // params += RPP_STRINGIZE(HIP_HCC_FLAGS);
    params += " ";
    auto bin_file = tmp_dir->path / (filename + ".o");
    // compile with hcc
    auto env = std::string("KMOPTLLC=-mattr=+enable-ds128");
    tmp_dir->Execute(env + std::string(" ") + "/opt/rocm/bin/hipcc",
                     params + filename + " -o " + bin_file.string());
    if(!fs::exists(bin_file))
        RPP_THROW(filename + " failed to compile");
    auto hsaco = std::find_if(fs::directory_iterator{tmp_dir->path},
                              {},
                              [](auto entry) { return (entry.path().extension() == ".hsaco"); });

    if(hsaco == fs::directory_iterator{})
    {
        RPP_LOG_E("failed to find *.hsaco in " << hsaco->path().string());
    }

    return hsaco->path();
#else
    (void)filename;
    (void)params;
    RPP_THROW("HIP kernels are only supported in Linux");
#endif
}

void bin_file_to_str(const fs::path& file, std::string& buf)
{
    std::ifstream bin_file_ptr(file.string().c_str(), std::ios::binary);
    std::ostringstream bin_file_strm;
    bin_file_strm << bin_file_ptr.rdbuf();
    buf = bin_file_strm.str();
}


} // namespace rpp
