// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "hipblaslt-jit-process.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <map>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace hipblaslt_jit::process
{
    namespace fs = std::filesystem;

    std::wstring detail::quoteWindowsArgument(const std::wstring& argument)
    {
        std::wstring result(1, L'"');
        std::size_t  slashes = 0;
        for(wchar_t ch : argument)
        {
            if(ch == L'\\')
            {
                ++slashes;
                continue;
            }
            result.append(ch == L'"' ? 2 * slashes + 1 : slashes, L'\\');
            slashes = 0;
            result += ch;
        }
        result.append(2 * slashes, L'\\');
        result += L'"';
        return result;
    }

    namespace
    {
        void require(bool condition, const std::string& message)
        {
            if(!condition)
                throw std::runtime_error(message);
        }

        void validate(const Request& request)
        {
            require(!request.argv.empty() && !request.argv.front().empty(),
                    "Provider executable is empty");
            for(const auto& argument : request.argv)
                require(argument.find('\0') == std::string::npos, "Provider argument contains NUL");
            for(const auto& entry : request.environment)
                require(!entry.first.empty() && entry.first.find('=') == std::string::npos
                            && entry.first.find('\0') == std::string::npos
                            && entry.second.find('\0') == std::string::npos,
                        "Invalid provider environment entry");
            require(!request.workingDirectory.empty() && !request.logPath.empty(),
                    "Provider cwd and log path must be specified");
            for(const auto& path : {request.workingDirectory, request.logPath})
                require(path.native().find(typename fs::path::value_type(0))
                            == fs::path::string_type::npos,
                        "Provider path contains NUL");
            require(fs::is_directory(request.workingDirectory),
                    "Provider working directory does not exist");
        }

#ifdef _WIN32
        std::wstring widen(const std::string& text)
        {
            if(text.empty())
                return {};
            require(text.size() <= INT_MAX, "Provider UTF-8 value is too long");
            int size = MultiByteToWideChar(CP_UTF8,
                                           MB_ERR_INVALID_CHARS,
                                           text.data(),
                                           static_cast<int>(text.size()),
                                           nullptr,
                                           0);
            require(size > 0, "Provider value is not valid UTF-8");
            std::wstring result(size, L'\0');
            require(MultiByteToWideChar(CP_UTF8,
                                        MB_ERR_INVALID_CHARS,
                                        text.data(),
                                        static_cast<int>(text.size()),
                                        result.data(),
                                        size)
                        == size,
                    "Cannot convert provider value to UTF-16");
            return result;
        }

        std::string narrow(const std::wstring& text)
        {
            if(text.empty())
                return {};
            int size = WideCharToMultiByte(CP_UTF8,
                                           0,
                                           text.data(),
                                           static_cast<int>(text.size()),
                                           nullptr,
                                           0,
                                           nullptr,
                                           nullptr);
            if(size <= 0)
                return "unavailable Windows error text";
            std::string result(size, '\0');
            WideCharToMultiByte(CP_UTF8,
                                0,
                                text.data(),
                                static_cast<int>(text.size()),
                                result.data(),
                                size,
                                nullptr,
                                nullptr);
            return result;
        }

        std::string windowsError(const char* operation, DWORD code)
        {
            wchar_t buffer[1024]{};
            DWORD   length
                = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 nullptr,
                                 code,
                                 0,
                                 buffer,
                                 1024,
                                 nullptr);
            return std::string(operation) + " (Windows error " + std::to_string(code)
                   + "): " + narrow(std::wstring(buffer, length));
        }

        void checkWindows(bool condition, const char* operation)
        {
            if(!condition)
                throw std::runtime_error(windowsError(operation, GetLastError()));
        }

        struct Handle
        {
            HANDLE value = INVALID_HANDLE_VALUE;
            explicit Handle(HANDLE handle = INVALID_HANDLE_VALUE)
                : value(handle)
            {
            }
            ~Handle()
            {
                if(value != INVALID_HANDLE_VALUE && value != nullptr)
                    CloseHandle(value);
            }
            Handle(const Handle&)            = delete;
            Handle& operator=(const Handle&) = delete;
        };

        struct EnvironmentLess
        {
            bool operator()(const std::wstring& a, const std::wstring& b) const
            {
                return CompareStringOrdinal(a.c_str(), -1, b.c_str(), -1, TRUE) == CSTR_LESS_THAN;
            }
        };

        std::vector<wchar_t> childEnvironment(const Request& request)
        {
            struct Block
            {
                LPWCH value = GetEnvironmentStringsW();
                ~Block()
                {
                    if(value)
                        FreeEnvironmentStringsW(value);
                }
            } inherited;
            checkWindows(inherited.value != nullptr, "GetEnvironmentStringsW");
            std::map<std::wstring, std::wstring, EnvironmentLess> values;
            for(const wchar_t* p = inherited.value; *p;)
            {
                std::wstring entry(p);
                p += entry.size() + 1;
                // Preserve Windows' hidden '=C:=...' per-drive directory entries.
                const auto equals = entry.find(L'=', entry.front() == L'=' ? 1 : 0);
                if(equals != std::wstring::npos)
                    values[entry.substr(0, equals)] = entry.substr(equals + 1);
            }
            for(const auto& entry : request.environment)
                values[widen(entry.first)] = widen(entry.second);
            std::vector<wchar_t> block;
            for(const auto& entry : values)
            {
                block.insert(block.end(), entry.first.begin(), entry.first.end());
                block.push_back(L'=');
                block.insert(block.end(), entry.second.begin(), entry.second.end());
                block.push_back(L'\0');
            }
            if(block.empty())
                block.push_back(L'\0');
            block.push_back(L'\0');
            return block;
        }

        std::wstring executablePath(const std::string& name)
        {
            auto path = fs::path(widen(name));
            if(path.has_parent_path() || path.has_root_name())
                return fs::absolute(path).native();
            const auto file = path.native();
            DWORD      size = SearchPathW(nullptr, file.c_str(), L".exe", 0, nullptr, nullptr);
            checkWindows(size != 0, "Find provider executable");
            std::vector<wchar_t> buffer(size + 1);
            DWORD                written = SearchPathW(nullptr,
                                        file.c_str(),
                                        L".exe",
                                        static_cast<DWORD>(buffer.size()),
                                        buffer.data(),
                                        nullptr);
            checkWindows(written != 0 && written < buffer.size(), "Find provider executable");
            return fs::absolute(fs::path(std::wstring(buffer.data(), written))).native();
        }

        void execute(const Request& request, Result& result)
        {
            auto         environment = childEnvironment(request);
            auto         cwd         = fs::absolute(request.workingDirectory).native();
            auto         logPath     = fs::absolute(request.logPath).native();
            std::wstring command;
            for(const auto& argument : request.argv)
            {
                if(!command.empty())
                    command += L' ';
                command += detail::quoteWindowsArgument(widen(argument));
            }
            require(command.size() < 32767,
                    "Provider Windows command line exceeds 32767 UTF-16 units");
            std::vector<wchar_t> mutableCommand(command.begin(), command.end());
            mutableCommand.push_back(L'\0');
            SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
            Handle              log(CreateFileW(logPath.c_str(),
                                   GENERIC_WRITE,
                                   FILE_SHARE_READ,
                                   &security,
                                   CREATE_NEW,
                                   FILE_ATTRIBUTE_NORMAL,
                                   nullptr));
            checkWindows(log.value != INVALID_HANDLE_VALUE, "Create provider log");
            Handle input(CreateFileW(L"NUL",
                                     GENERIC_READ,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE,
                                     &security,
                                     OPEN_EXISTING,
                                     FILE_ATTRIBUTE_NORMAL,
                                     nullptr));
            checkWindows(input.value != INVALID_HANDLE_VALUE, "Open provider stdin");
            const auto executable = executablePath(request.argv.front());

            struct Attributes
            {
                std::vector<unsigned char>   storage;
                LPPROC_THREAD_ATTRIBUTE_LIST list = nullptr;
                ~Attributes()
                {
                    if(list)
                        DeleteProcThreadAttributeList(list);
                }
            } attributes;
            SIZE_T bytes = 0;
            InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
            checkWindows(bytes > 0, "Size process attributes");
            attributes.storage.resize(bytes);
            auto* list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.storage.data());
            checkWindows(InitializeProcThreadAttributeList(list, 1, 0, &bytes) != FALSE,
                         "Initialize process attributes");
            attributes.list  = list;
            HANDLE handles[] = {input.value, log.value};
            checkWindows(UpdateProcThreadAttribute(list,
                                                   0,
                                                   PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                                   handles,
                                                   sizeof(handles),
                                                   nullptr,
                                                   nullptr)
                             != FALSE,
                         "Set inherited provider handles");
            STARTUPINFOEXW startup{};
            startup.StartupInfo.cb         = sizeof(startup);
            startup.StartupInfo.dwFlags    = STARTF_USESTDHANDLES;
            startup.StartupInfo.hStdInput  = input.value;
            startup.StartupInfo.hStdOutput = log.value;
            startup.StartupInfo.hStdError  = log.value;
            startup.lpAttributeList        = list;
            PROCESS_INFORMATION process{};
            checkWindows(CreateProcessW(executable.c_str(),
                                        mutableCommand.data(),
                                        nullptr,
                                        nullptr,
                                        TRUE,
                                        EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
                                        environment.data(),
                                        cwd.c_str(),
                                        &startup.StartupInfo,
                                        &process)
                             != FALSE,
                         "Create provider process");
            Handle processHandle(process.hProcess);
            Handle threadHandle(process.hThread);
            result.started = true;
            checkWindows(WaitForSingleObject(processHandle.value, INFINITE) == WAIT_OBJECT_0,
                         "Wait for provider");
            DWORD exitCode = 0;
            checkWindows(GetExitCodeProcess(processHandle.value, &exitCode) != FALSE,
                         "Read provider exit status");
            result.exited   = true;
            result.exitCode = exitCode;
        }
#else
        std::string posixError(const char* operation, int code)
        {
            return std::string(operation) + ": "
                   + std::error_code(code, std::generic_category()).message();
        }

        struct Descriptor
        {
            int value = -1;
            explicit Descriptor(int descriptor = -1)
                : value(descriptor)
            {
            }
            ~Descriptor()
            {
                if(value >= 0)
                    close(value);
            }
            Descriptor(const Descriptor&)            = delete;
            Descriptor& operator=(const Descriptor&) = delete;
        };

        // The parent may have closed a standard descriptor. Keep our descriptors
        // above 2 so adddup2/addclose never collide with one another.
        int ownedDescriptor(int descriptor)
        {
            if(descriptor < 0 || descriptor > STDERR_FILENO)
                return descriptor;
            int moved = fcntl(descriptor, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
            int code  = errno;
            close(descriptor);
            errno = code;
            return moved;
        }

        std::vector<std::string> childEnvironment(const Request& request)
        {
            std::map<std::string, std::string> values;
            if(environ)
                for(char** p = environ; *p; ++p)
                {
                    std::string entry(*p);
                    const auto  equals = entry.find('=');
                    if(equals != std::string::npos)
                        values[entry.substr(0, equals)] = entry.substr(equals + 1);
                }
            for(const auto& entry : request.environment)
                values[entry.first] = entry.second;
            std::vector<std::string> environment;
            for(const auto& entry : values)
                environment.push_back(entry.first + "=" + entry.second);
            return environment;
        }

        void execute(const Request& request, Result& result)
        {
            auto               environment = childEnvironment(request);
            std::vector<char*> envp;
            for(auto& entry : environment)
                envp.push_back(entry.data());
            envp.push_back(nullptr);
            auto arguments = request.argv;
            if(fs::path(arguments.front()).has_parent_path())
                arguments.front() = fs::absolute(arguments.front()).string();
            std::vector<char*> argv;
            for(auto& argument : arguments)
                argv.push_back(argument.data());
            argv.push_back(nullptr);
            const auto cwd     = fs::absolute(request.workingDirectory).string();
            const auto logPath = fs::absolute(request.logPath).string();
            Descriptor log(ownedDescriptor(
                open(logPath.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600)));
            require(log.value >= 0, posixError("Create provider log", errno));
            Descriptor input(ownedDescriptor(open("/dev/null", O_RDONLY | O_CLOEXEC)));
            require(input.value >= 0, posixError("Open provider stdin", errno));
            struct Actions
            {
                posix_spawn_file_actions_t value;
                Actions()
                {
                    int code = posix_spawn_file_actions_init(&value);
                    require(code == 0, posixError("Initialize spawn actions", code));
                }
                ~Actions()
                {
                    posix_spawn_file_actions_destroy(&value);
                }
            } actions;
            int code = posix_spawn_file_actions_addchdir_np(&actions.value, cwd.c_str());
            if(code == 0)
                code = posix_spawn_file_actions_adddup2(&actions.value, input.value, STDIN_FILENO);
            if(code == 0)
                code = posix_spawn_file_actions_adddup2(&actions.value, log.value, STDOUT_FILENO);
            if(code == 0)
                code = posix_spawn_file_actions_adddup2(&actions.value, log.value, STDERR_FILENO);
            if(code == 0)
                code = posix_spawn_file_actions_addclose(&actions.value, input.value);
            if(code == 0)
                code = posix_spawn_file_actions_addclose(&actions.value, log.value);
            require(code == 0, posixError("Configure provider spawn", code));
            pid_t child = -1;
            code        = posix_spawnp(
                &child, argv.front(), &actions.value, nullptr, argv.data(), envp.data());
            require(code == 0, posixError("Start provider", code));
            result.started = true;
            int   status   = 0;
            pid_t waited;
            do
            {
                waited = waitpid(child, &status, 0);
            } while(waited < 0 && errno == EINTR);
            require(waited == child, posixError("Wait for provider", errno));
            if(WIFEXITED(status))
            {
                result.exited   = true;
                result.exitCode = static_cast<std::uint32_t>(WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status))
                result.terminationSignal = WTERMSIG(status);
            else
                throw std::runtime_error("Provider returned an unknown wait status");
        }
#endif
    }

    Result run(const Request& request)
    {
        Result result;
        try
        {
            validate(request);
            execute(request, result);
            if(result.terminationSignal)
                result.error
                    = "Provider terminated by signal " + std::to_string(result.terminationSignal);
            else if(result.exitCode)
                result.error = "Provider exited with code " + std::to_string(result.exitCode);
        }
        catch(const std::exception& error)
        {
            result.error = error.what();
        }
        return result;
    }
}
