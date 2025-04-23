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

#include <chrono>
#include "tmp_dir.hpp"
#include "errors.hpp"
#include "logger.hpp"

namespace rpp {

void SystemCmd(std::string cmd)
{
#ifndef NDEBUG
    RPP_LOG_I(cmd);
#endif
// We shouldn't call system commands
#ifdef RPP_USE_CLANG_TIDY
    (void)cmd;
#else
    if(std::system(cmd.c_str()) != 0)
        RPP_THROW("Can't execute " + cmd);
#endif
}

TmpDir::TmpDir(std::string prefix)
    : path(fs::temp_directory_path() / ("rpp-" + prefix + "-" +
            std::to_string(std::chrono::system_clock::now().time_since_epoch().count())))
{
    fs::create_directories(this->path);
}

void TmpDir::Execute(std::string exe, std::string args)
{
    // std::cout<<"Invoking Execute routine\n";
    std::string cd  = "cd " + this->path.string() + "; ";
    std::string cmd = cd + exe + " " + args; // + " > /dev/null";
    SystemCmd(cmd);
    // std::cout<<"Done with Execute routine"<<cmd.c_str()<<std::endl;
}

TmpDir::~TmpDir() { fs::remove_all(this->path); }

} // namespace rpp
