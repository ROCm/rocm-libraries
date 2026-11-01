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
#include <miopen/logger.hpp>
#include <miopen/process.hpp>
#include <miopen/stringutils.hpp>

#include <array>
#include <istream>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cwchar>
#else
#include <cerrno>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

// The child is started directly, never through a shell (/bin/sh -c or cmd.exe), so shell
// metacharacters in the command, arguments, working directory or environment values are
// passed through literally. See ROCM-26841 / ROCM-27008.

// Can be predefined to 0 to exercise the fork() fallback on a newer glibc.
#ifndef MIOPEN_PROCESS_HAS_SPAWN_CHDIR
#if !defined(_WIN32) && defined(__GLIBC__) && \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 29))
#define MIOPEN_PROCESS_HAS_SPAWN_CHDIR 1 // posix_spawn_file_actions_addchdir_np
#else
#define MIOPEN_PROCESS_HAS_SPAWN_CHDIR 0
#endif
#endif

namespace miopen {

#ifdef _WIN32

namespace {

std::wstring ToWide(std::string_view s)
{
    if(s.empty())
        return {};
    const auto len = static_cast<int>(s.size());
    const int size = MultiByteToWideChar(CP_ACP, 0, s.data(), len, nullptr, 0);
    std::wstring result(size, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.data(), len, result.data(), size);
    return result;
}

// Quotes one argument so that CommandLineToArgvW / the MSVC CRT parse it back unchanged.
std::wstring QuoteArgument(const std::wstring& arg)
{
    if(!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos)
        return arg;

    std::wstring quoted{L'"'};
    for(auto it = arg.begin();; ++it)
    {
        std::size_t backslashes = 0;
        while(it != arg.end() && *it == L'\\')
        {
            ++it;
            ++backslashes;
        }
        if(it == arg.end())
        {
            quoted.append(backslashes * 2, L'\\');
            break;
        }
        if(*it == L'"')
        {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted.push_back(*it);
        }
        else
        {
            quoted.append(backslashes, L'\\');
            quoted.push_back(*it);
        }
    }
    quoted.push_back(L'"');
    return quoted;
}

struct CaseInsensitiveLess
{
    bool operator()(const std::wstring& a, const std::wstring& b) const
    {
        return _wcsicmp(a.c_str(), b.c_str()) < 0;
    }
};

// Current environment with the overrides applied, as a sorted, double-NUL-terminated block.
std::wstring BuildEnvironmentBlock(const ProcessEnvironmentMap& overrides)
{
    std::map<std::wstring, std::wstring, CaseInsensitiveLess> vars;

    wchar_t* const block = GetEnvironmentStringsW();
    if(block != nullptr)
    {
        for(const wchar_t* p = block; *p != L'\0'; p += std::wcslen(p) + 1)
        {
            const std::wstring_view entry{p};
            // Search from 1: hidden per-drive entries look like "=C:=C:\dir".
            const auto eq = entry.find(L'=', 1);
            if(eq == std::wstring_view::npos)
                continue;
            vars.emplace(std::wstring{entry.substr(0, eq)}, std::wstring{entry.substr(eq + 1)});
        }
        FreeEnvironmentStringsW(block);
    }

    for(const auto& [name, value] : overrides)
        vars[ToWide(name)] = ToWide(value);

    std::wstring result;
    for(const auto& [name, value] : vars)
    {
        result += name;
        result += L'=';
        result += value;
        result += L'\0';
    }
    result += L'\0';
    return result;
}

struct UniqueHandle
{
    HANDLE handle = nullptr;

    UniqueHandle()                               = default;
    UniqueHandle(const UniqueHandle&)            = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;
    ~UniqueHandle() { Reset(); }

    void Reset()
    {
        if(handle != nullptr && handle != INVALID_HANDLE_VALUE)
            CloseHandle(handle);
        handle = nullptr;
    }
};

} // namespace

struct ProcessImpl
{
    ProcessImpl(std::string_view cmd) : path{cmd} {}

    void Create(std::string_view args,
                std::string_view cwd,
                std::ostream* out,
                const ProcessEnvironmentMap& additionalEnvironmentVariables,
                std::istream* in)
    {
        if(in != nullptr)
            MIOPEN_THROW("Process: redirecting stdin is not supported on Windows");

        outStream = out;

        // Arguments are handed to the child unchanged; the child parses its own command line.
        std::wstring cmdLine = QuoteArgument(path.wstring());
        if(!args.empty())
            cmdLine += L" " + ToWide(args);

        auto environment   = BuildEnvironmentBlock(additionalEnvironmentVariables);
        const auto wideCwd = ToWide(cwd);

        STARTUPINFOEXW startupInfo{};
        startupInfo.StartupInfo.cb = sizeof(startupInfo);
        DWORD flags                = CREATE_UNICODE_ENVIRONMENT;
        BOOL inheritHandles        = FALSE;

        UniqueHandle nulInput;
        UniqueHandle childOutput;
        std::vector<char> attributeListBuffer;
        LPPROC_THREAD_ATTRIBUTE_LIST attributeList = nullptr;
        std::array<HANDLE, 2> inheritedHandles{};

        if(out != nullptr)
        {
            SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
            if(CreatePipe(&outputRead.handle, &childOutput.handle, &inheritable, 0) == FALSE)
                MIOPEN_THROW("CreatePipe error: " + std::to_string(GetLastError()));
            if(SetHandleInformation(outputRead.handle, HANDLE_FLAG_INHERIT, 0) == FALSE)
                MIOPEN_THROW("SetHandleInformation error: " + std::to_string(GetLastError()));

            nulInput.handle = CreateFileW(L"NUL",
                                          GENERIC_READ,
                                          FILE_SHARE_READ | FILE_SHARE_WRITE,
                                          &inheritable,
                                          OPEN_EXISTING,
                                          FILE_ATTRIBUTE_NORMAL,
                                          nullptr);
            if(nulInput.handle == INVALID_HANDLE_VALUE)
                MIOPEN_THROW("CreateFile(NUL) error: " + std::to_string(GetLastError()));

            startupInfo.StartupInfo.dwFlags    = STARTF_USESTDHANDLES;
            startupInfo.StartupInfo.hStdInput  = nulInput.handle;
            startupInfo.StartupInfo.hStdOutput = childOutput.handle;
            startupInfo.StartupInfo.hStdError  = childOutput.handle;

            // Inherit only these two handles, so a process started concurrently from another
            // thread cannot pick up our pipe and keep it open past the child's exit.
            inheritedHandles = {nulInput.handle, childOutput.handle};
            SIZE_T size      = 0;
            InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
            attributeListBuffer.resize(size);
            attributeList =
                reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributeListBuffer.data());
            if(InitializeProcThreadAttributeList(attributeList, 1, 0, &size) == FALSE)
                MIOPEN_THROW("InitializeProcThreadAttributeList error: " +
                             std::to_string(GetLastError()));
            if(UpdateProcThreadAttribute(attributeList,
                                         0,
                                         PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                         inheritedHandles.data(),
                                         inheritedHandles.size() * sizeof(HANDLE),
                                         nullptr,
                                         nullptr) == FALSE)
            {
                DeleteProcThreadAttributeList(attributeList);
                MIOPEN_THROW("UpdateProcThreadAttribute error: " + std::to_string(GetLastError()));
            }
            startupInfo.lpAttributeList = attributeList;
            flags |= EXTENDED_STARTUPINFO_PRESENT;
            inheritHandles = TRUE;
        }

        const auto created     = CreateProcessW(nullptr,
                                            cmdLine.data(),
                                            nullptr,
                                            nullptr,
                                            inheritHandles,
                                            flags,
                                            environment.data(),
                                            wideCwd.empty() ? nullptr : wideCwd.c_str(),
                                            &startupInfo.StartupInfo,
                                            &processInfo);
        const auto createError = GetLastError();
        if(attributeList != nullptr)
            DeleteProcThreadAttributeList(attributeList);
        if(created == FALSE)
            MIOPEN_THROW("CreateProcess error: " + std::to_string(createError));
        // childOutput closes here, so reading outputRead sees EOF once the child exits.
    }

    int Wait()
    {
        if(outStream != nullptr)
        {
            // Buffer all output locally first to avoid unknown flush behavior of outStream
            std::ostringstream localBuffer;
            std::array<char, 1024> buffer{};
            DWORD bytesRead = 0;
            while(ReadFile(outputRead.handle,
                           buffer.data(),
                           static_cast<DWORD>(buffer.size()),
                           &bytesRead,
                           nullptr) != FALSE &&
                  bytesRead != 0)
            {
                localBuffer.write(buffer.data(), bytesRead);
            }
            outputRead.Reset();
            *outStream << localBuffer.str();
        }

        WaitForSingleObject(processInfo.hProcess, INFINITE);

        DWORD status                 = 0;
        const auto getExitCodeStatus = GetExitCodeProcess(processInfo.hProcess, &status);
        const auto exitCodeError     = GetLastError();
        CloseProcessHandles();

        if(getExitCodeStatus == FALSE)
            MIOPEN_THROW("GetExitCodeProcess error: " + std::to_string(exitCodeError));

        return static_cast<int>(status);
    }

    ~ProcessImpl() { CloseProcessHandles(); }

private:
    void CloseProcessHandles()
    {
        if(processInfo.hProcess != nullptr)
            CloseHandle(processInfo.hProcess);
        if(processInfo.hThread != nullptr)
            CloseHandle(processInfo.hThread);
        processInfo = {};
    }

    std::ostream* outStream = nullptr;
    fs::path path;
    UniqueHandle outputRead;
    PROCESS_INFORMATION processInfo{};
};

#else

namespace {

struct UniqueFd
{
    int fd = -1;

    UniqueFd()                           = default;
    UniqueFd(const UniqueFd&)            = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;
    ~UniqueFd() { Reset(); }

    void Reset()
    {
        if(fd >= 0)
            close(fd);
        fd = -1;
    }
};

void MakePipe(UniqueFd& readEnd, UniqueFd& writeEnd)
{
    std::array<int, 2> fds{};
    if(pipe2(fds.data(), O_CLOEXEC) != 0)
        MIOPEN_THROW("pipe2() failed, errno " + std::to_string(errno));
    readEnd.fd  = fds[0];
    writeEnd.fd = fds[1];
}

// Current environment with the overrides applied, as NAME=VALUE strings.
std::vector<std::string> BuildEnvironment(const ProcessEnvironmentMap& overrides)
{
    std::vector<std::string> result;
    for(char** entry = environ; entry != nullptr && *entry != nullptr; ++entry)
    {
        const std::string_view var{*entry};
        if(overrides.count(std::string{var.substr(0, var.find('='))}) == 0)
            result.emplace_back(var);
    }
    for(const auto& [name, value] : overrides)
        result.push_back(name + "=" + value);
    return result;
}

// Resolves a bare command name against PATH from the child's environment, like execvp.
std::string ResolveExecutable(const std::string& cmd, const std::vector<std::string>& environment)
{
    if(cmd.empty() || cmd.find('/') != std::string::npos)
        return cmd;

    for(const auto& var : environment)
    {
        if(!StartsWith(var, "PATH="))
            continue;
        for(const auto& dir : SplitDelim(var.substr(5), ':'))
        {
            const auto candidate = (dir.empty() ? fs::path{"."} : fs::path{dir}) / cmd;
            if(access(candidate.c_str(), X_OK) == 0)
                return candidate.string();
        }
        break;
    }
    return cmd;
}

std::vector<char*> NullTerminated(std::vector<std::string>& strings)
{
    std::vector<char*> result;
    result.reserve(strings.size() + 1);
    for(auto& s : strings)
        result.push_back(s.data());
    result.push_back(nullptr);
    return result;
}

} // namespace

struct ProcessImpl
{
    ProcessImpl(std::string_view cmd) : path{cmd} {}

    void Create(std::string_view args,
                std::string_view cwd,
                std::ostream* out,
                const ProcessEnvironmentMap& additionalEnvironmentVariables,
                std::istream* in)
    {
        if(in != nullptr && out != nullptr)
            MIOPEN_THROW("Process: redirecting both stdin and stdout is not supported");

        outStream = out;
        inStream  = in;

        // Arguments are split on whitespace, the same rule COMGR applies to build options.
        auto argStrings = SplitSpaceSeparated(std::string{args});
        argStrings.insert(argStrings.begin(), path.string());
        auto envStrings = BuildEnvironment(additionalEnvironmentVariables);
        const auto exe  = ResolveExecutable(path.string(), envStrings);
        const auto argv = NullTerminated(argStrings);
        const auto envp = NullTerminated(envStrings);
        const std::string dir{cwd};

        // The child's ends of the pipes; the parent keeps the other ends.
        UniqueFd childInput;
        UniqueFd childOutput;
        if(in != nullptr)
        {
            MakePipe(childInput, inputWrite);
        }
        else
        {
            childInput.fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
            if(childInput.fd < 0)
                MIOPEN_THROW("open(/dev/null) failed, errno " + std::to_string(errno));
        }
        if(out != nullptr)
            MakePipe(outputRead, childOutput);

        if(MIOPEN_PROCESS_HAS_SPAWN_CHDIR || dir.empty())
            Spawn(exe, argv, envp, dir, childInput.fd, childOutput.fd);
        else
            ForkExec(exe, argv, envp, dir, childInput.fd, childOutput.fd);
    }

    int Wait()
    {
        if(inputWrite.fd >= 0)
        {
            std::array<char, 1024> buffer{};
            while(*inStream)
            {
                inStream->read(buffer.data(), buffer.size());
                if(!WriteAll(buffer.data(), inStream->gcount()))
                    break;
            }
            inputWrite.Reset();
        }

        if(outputRead.fd >= 0)
        {
            // Buffer all output locally first to avoid unknown flush behavior of outStream
            std::ostringstream localBuffer;
            std::array<char, 1024> buffer{};
            for(;;)
            {
                const auto n = read(outputRead.fd, buffer.data(), buffer.size());
                if(n > 0)
                    localBuffer.write(buffer.data(), n);
                else if(n == 0 || errno != EINTR)
                    break;
            }
            outputRead.Reset();
            *outStream << localBuffer.str();
        }

        return Reap();
    }

    ~ProcessImpl()
    {
        inputWrite.Reset();
        outputRead.Reset();
        if(pid > 0)
        {
            try
            {
                std::ignore = Reap();
            }
            catch(...) // NOLINT(bugprone-empty-catch)
            {
            }
        }
    }

private:
    void Spawn(const std::string& exe,
               const std::vector<char*>& argv,
               const std::vector<char*>& envp,
               [[maybe_unused]] const std::string& dir,
               int childInput,
               int childOutput)
    {
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, childInput, STDIN_FILENO);
        if(childOutput >= 0)
        {
            posix_spawn_file_actions_adddup2(&actions, childOutput, STDOUT_FILENO);
            posix_spawn_file_actions_adddup2(&actions, childOutput, STDERR_FILENO);
        }
#if MIOPEN_PROCESS_HAS_SPAWN_CHDIR
        if(!dir.empty())
            posix_spawn_file_actions_addchdir_np(&actions, dir.c_str());
#endif
        const auto rc = posix_spawn(&pid, exe.c_str(), &actions, nullptr, argv.data(), envp.data());
        posix_spawn_file_actions_destroy(&actions);
        if(rc != 0)
        {
            // Report it the way a shell would: the command could not be executed.
            MIOPEN_LOG_W("posix_spawn(" << exe << ") failed, error " << rc);
            pid        = -1;
            exitStatus = 127;
        }
    }

    // Fallback for glibc < 2.29, which lacks posix_spawn_file_actions_addchdir_np.
    void ForkExec(const std::string& exe,
                  const std::vector<char*>& argv,
                  const std::vector<char*>& envp,
                  const std::string& dir,
                  int childInput,
                  int childOutput)
    {
        pid = fork();
        if(pid < 0)
            MIOPEN_THROW("fork() failed, errno " + std::to_string(errno));
        if(pid == 0)
        {
            // Only async-signal-safe calls between fork() and exec.
            if(dup2(childInput, STDIN_FILENO) < 0)
                _exit(127);
            if(childOutput >= 0 &&
               (dup2(childOutput, STDOUT_FILENO) < 0 || dup2(childOutput, STDERR_FILENO) < 0))
                _exit(127);
            if(chdir(dir.c_str()) != 0)
                _exit(127);
            execve(exe.c_str(), argv.data(), envp.data());
            _exit(127);
        }
    }

    bool WriteAll(const char* data, std::streamsize size) const
    {
        while(size > 0)
        {
            const auto n = write(inputWrite.fd, data, size);
            if(n < 0)
            {
                if(errno == EINTR)
                    continue;
                return false;
            }
            data += n;
            size -= n;
        }
        return true;
    }

    int Reap()
    {
        if(pid <= 0)
            return exitStatus;
        int status = 0;
        while(waitpid(pid, &status, 0) < 0)
        {
            if(errno != EINTR)
            {
                pid = -1;
                MIOPEN_THROW("waitpid() failed, errno " + std::to_string(errno));
            }
        }
        pid = -1;
        if(WIFEXITED(status))
            exitStatus = WEXITSTATUS(status);
        else if(WIFSIGNALED(status))
            exitStatus = 128 + WTERMSIG(status); // Same convention as the shell.
        else
            exitStatus = -1;
        return exitStatus;
    }

    std::ostream* outStream = nullptr;
    std::istream* inStream  = nullptr;
    fs::path path;
    UniqueFd inputWrite;
    UniqueFd outputRead;
    pid_t pid      = -1;
    int exitStatus = -1;
};

#endif

Process::Process(const fs::path& cmd) : impl{std::make_unique<ProcessImpl>(cmd.string())} {}

Process::~Process() noexcept = default;

int Process::operator()(std::string_view args,
                        const fs::path& cwd,
                        std::ostream* out,
                        const ProcessEnvironmentMap& additionalEnvironmentVariables,
                        std::istream* in)
{
    impl->Create(args, cwd.string(), out, additionalEnvironmentVariables, in);
    return impl->Wait();
}

ProcessAsync::ProcessAsync(const fs::path& cmd,
                           std::string_view args,
                           const fs::path& cwd,
                           std::ostream* out,
                           const ProcessEnvironmentMap& additionalEnvironmentVariables)
    : impl{std::make_unique<ProcessImpl>(cmd.string())}
{
    impl->Create(args, cwd.string(), out, additionalEnvironmentVariables, nullptr);
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
