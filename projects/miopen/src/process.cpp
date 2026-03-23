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

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

struct ProcessImpl
{
public:
    ProcessImpl(std::string_view cmd) : path{cmd} {}

    void Create(std::string_view args,
                std::string_view cwd,
                std::ostream* out,
                const ProcessEnvironmentMap& additionalEnvironmentVariables)
    {
        outStream = out;

        STARTUPINFOA info;
        ZeroMemory(&info, sizeof(STARTUPINFO));
        info.cb = sizeof(STARTUPINFO);

        // Set up pipe for stdout capture if output stream is provided
        if(outStream != nullptr)
        {
            SECURITY_ATTRIBUTES saAttr;
            saAttr.nLength              = sizeof(SECURITY_ATTRIBUTES);
            saAttr.bInheritHandle       = TRUE;
            saAttr.lpSecurityDescriptor = nullptr;

            if(CreatePipe(&hReadPipe, &hWritePipe, &saAttr, 0) == FALSE)
                MIOPEN_THROW("CreatePipe error: " + std::to_string(GetLastError()));

            // Ensure the read handle is not inherited
            SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

            info.hStdOutput = hWritePipe;
            info.hStdError  = hWritePipe;
            info.dwFlags |= STARTF_USESTDHANDLES;
        }

        // Build environment block for additional variables
        std::string envBlock;
        if(!additionalEnvironmentVariables.empty())
        {
            for(const auto& [key, value] : additionalEnvironmentVariables)
            {
                envBlock += key + "=" + value + '\0';
            }
            envBlock += '\0'; // Double null terminator
        }

        std::string cmd{path.string()};
        if(!args.empty())
            cmd += " " + std::string{args};

        // Refer to
        // CreateProcessA function (processthreadsapi.h)
        constexpr std::size_t BUFFER_CAPACITY = 32767;

        if(cmd.size() < BUFFER_CAPACITY)
            cmd.resize(BUFFER_CAPACITY, '\0');

        if(CreateProcess(path.string().c_str(),
                         cmd.data(),
                         nullptr,
                         nullptr,
                         outStream != nullptr ? TRUE : FALSE,
                         0,
                         envBlock.empty() ? nullptr : envBlock.data(),
                         cwd.empty() ? nullptr : cwd.data(),
                         &info,
                         &processInfo) == FALSE)
            MIOPEN_THROW("CreateProcess error: " + std::to_string(GetLastError()));

        // Close the write end of the pipe (child process has it now)
        if(hWritePipe != nullptr)
        {
            CloseHandle(hWritePipe);
            hWritePipe = nullptr;
        }
    }

    int Wait()
    {
        // Read output from pipe if capturing
        if(outStream != nullptr && hReadPipe != nullptr)
        {
            std::array<char, 1024> buffer{};
            DWORD bytesRead;

            while(ReadFile(hReadPipe, buffer.data(), buffer.size() - 1, &bytesRead, nullptr) &&
                  bytesRead > 0)
            {
                buffer[bytesRead] = '\0';
                *outStream << buffer.data();
            }
            CloseHandle(hReadPipe);
            hReadPipe = nullptr;
        }

        WaitForSingleObject(processInfo.hProcess, INFINITE);

        DWORD status;
        const auto getExitCodeStatus = GetExitCodeProcess(processInfo.hProcess, &status);

        CloseHandle(processInfo.hProcess);
        CloseHandle(processInfo.hThread);

        if(getExitCodeStatus == 0)
            MIOPEN_THROW("GetExitCodeProcess error: " + std::to_string(GetLastError()));

        return status;
    }

private:
    std::ostream* outStream = nullptr;
    fs::path path;
    PROCESS_INFORMATION processInfo{};
    HANDLE hReadPipe  = nullptr;
    HANDLE hWritePipe = nullptr;
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
