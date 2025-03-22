/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#ifdef __GNUG__
#include <cxxabi.h>
#endif

using IsaVersion = std::array<int, 3>;

inline std::string getGfxNameTuple(const IsaVersion& isaVersion)
{
    /*Converts an ISA version to a gfx architecture name.

    Args:
        arch: An object representing the major, minor, and step version of the ISA.
    
    Returns:
        The name of the GPU architecture (e.g., 'gfx906').
    */
    const char int_to_hex[] = "0123456789abcdef";
    return std::string("gfx") + std::to_string(isaVersion[0]) + std::to_string(isaVersion[1])
           + int_to_hex[isaVersion[2]];
}

inline std::pair<int, std::string>
    run(const std::vector<char*>& cmd, const std::string& input, bool debug = false)
{
    int   p[2];
    pid_t pid;
    if(pipe(p) == -1)
    {
        throw std::runtime_error("cmd failed!");
    }

    pid = fork();
    if(pid == 0)
    {
        close(p[1]);
        dup2(p[0], STDIN_FILENO);
        if(!debug)
        {
            dup2(p[0], STDERR_FILENO);
            dup2(p[0], STDOUT_FILENO);
        }
        close(p[0]);
        execvp(cmd[0], cmd.data());
        perror("execvp");
        exit(1);
    }
    else
    {
        close(p[0]);
        write(p[1], input.c_str(), input.size());
        close(p[1]);
        char        buf[128] = {0};
        std::string result;
        read(p[0], buf, 128);
        result += buf;
        int rcode;
        waitpid(pid, &rcode, 0);
        if(WIFEXITED(rcode))
        {
            rcode = WEXITSTATUS(rcode);
        }
        return {rcode, result};
    }
    // Should not go here.
    return {0, "0"};
}

template <typename T>
bool checkInList(const T& a, const std::vector<T> b)
{
    return std::find(b.begin(), b.end(), a) != b.end();
}

template <typename T>
bool checkNotInList(const T& a, const std::vector<T> b)
{
    return std::find(b.begin(), b.end(), a) == b.end();
}

inline std::string demangle(const char* name)
{
    std::string result = name;
#ifdef __GNUG__
    int   status    = -1;
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    result          = (status == 0) ? demangled : name;
    free(demangled);
#else
#error "Windows not supported"
#endif
    return result;
}
