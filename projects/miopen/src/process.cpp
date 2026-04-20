/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/errors.hpp>
#include <miopen/process.hpp>
#include <string_view>

namespace miopen {

#ifdef _WIN32

#include <cstdio>

// Windows equivalents for POSIX popen/pclose
#define popen _popen
#define pclose _pclose

struct ProcessImpl
{
    ProcessImpl(std::string_view cmd) : path{cmd} {}

    void Create(std::string_view args,
                std::string_view cwd,
                std::ostream* out,
                const ProcessEnvironmentMap& additionalEnvironmentVariables)
    {
        outStream = out;

        // Build command left-to-right using stringstream
        std::stringstream cmdStream;

        // 1. Change directory (if specified)
        if(!cwd.empty())
            cmdStream << "cd /d " << cwd << " && ";

        // 2. Set environment variables (if any)
        for(const auto& envVariable : additionalEnvironmentVariables)
        {
            cmdStream << "set " << envVariable.first << "=" << envVariable.second << " && ";
        }

        // 3. Command path
        cmdStream << path.string();

        // 4. Arguments (if any)
        if(!args.empty())
            cmdStream << " " << args;

        // 5. Redirect stderr to stdout (if capturing output)
        if(out != nullptr)
            cmdStream << " 2>&1";

        std::string cmd = cmdStream.str();

        const auto fileMode = outStream != nullptr ? "r" : "w";
        pipe                = popen(cmd.c_str(), fileMode);
        if(pipe == nullptr)
            MIOPEN_THROW("Error: popen()");
    }

    int Wait()
    {
        if(outStream != nullptr)
        {
            // Buffer all output locally first to avoid unknown flush behavior of outStream
            std::ostringstream localBuffer;
            std::array<char, 1024> buffer{};

            while(feof(pipe) == 0)
            {
                if(fgets(buffer.data(), buffer.size(), pipe) != nullptr)
                    localBuffer << buffer.data();
            }

            // Single write to outStream for consistent behavior and performance
            *outStream << localBuffer.str();
        }

        auto status = pclose(pipe);
        return status;
    }

private:
    std::ostream* outStream = nullptr;
    fs::path path;
    FILE* pipe = nullptr;
};

#else

struct ProcessImpl
{
    ProcessImpl(std::string_view cmd) : path{cmd} {}

    void Create(std::string_view args,
                std::string_view cwd,
                std::ostream* out,
                const ProcessEnvironmentMap& additionalEnvironmentVariables)
    {
        outStream = out;
        std::string cmd{path.string()};
        if(!additionalEnvironmentVariables.empty())
        {
            std::stringstream environmentVariables;
            for(const auto& envVariable : additionalEnvironmentVariables)
            {
                environmentVariables << envVariable.first << "=" << envVariable.second << " ";
            }
            cmd.insert(0, environmentVariables.str());
        }
        if(!args.empty())
            cmd += " " + std::string{args};
        // When capturing output, redirect stderr to stdout so we capture both
        if(out != nullptr)
            cmd += " 2>&1";
        if(!cwd.empty())
            cmd.insert(0, "cd " + std::string{cwd} + "; ");

        const auto fileMode = outStream != nullptr ? "r" : "w";
        pipe                = popen(cmd.c_str(), fileMode);
        if(pipe == nullptr)
            MIOPEN_THROW("Error: popen()");
    }

    int Wait()
    {
        if(outStream != nullptr)
        {
            std::array<char, 1024> buffer{};

            while(feof(pipe) == 0)
            {
                if(fgets(buffer.data(), buffer.size(), pipe) != nullptr)
                    *outStream << buffer.data();
            }
        }

        auto status = pclose(pipe);
        return WEXITSTATUS(status);
    }

private:
    std::ostream* outStream;
    fs::path path;
    FILE* pipe = nullptr;
};

#endif

Process::Process(const fs::path& cmd) : impl{std::make_unique<ProcessImpl>(cmd.string())} {}

Process::~Process() noexcept = default;

int Process::operator()(std::string_view args,
                        const fs::path& cwd,
                        std::ostream* out,
                        const ProcessEnvironmentMap& additionalEnvironmentVariables)
{
    impl->Create(args, cwd.string(), out, additionalEnvironmentVariables);
    return impl->Wait();
}

ProcessAsync::ProcessAsync(const fs::path& cmd,
                           std::string_view args,
                           const fs::path& cwd,
                           std::ostream* out,
                           const ProcessEnvironmentMap& additionalEnvironmentVariables)
    : impl{std::make_unique<ProcessImpl>(cmd.string())}
{
    impl->Create(args, cwd.string(), out, additionalEnvironmentVariables);
}

ProcessAsync::~ProcessAsync() noexcept = default;

int ProcessAsync::Wait() { return impl->Wait(); }

ProcessAsync& ProcessAsync::operator=(ProcessAsync&& other) noexcept
{
    impl = std::move(other.impl);
    return *this;
}

ProcessAsync::ProcessAsync(ProcessAsync&& other) noexcept : impl{std::move(other.impl)} {}

} // namespace miopen
