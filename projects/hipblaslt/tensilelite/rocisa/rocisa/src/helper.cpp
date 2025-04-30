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

#include "helper.hpp"

// This file includes platform specific helpers that differ between POSIX and
// Windows. The alternatives are switched at a whole-file level. Please do
// not use inline/fine-grained ifdefs.

#ifdef _WIN32

#include <windows.h>
// windows.h must be loaded before other windows headers.
#include <dbghelp.h>

// Linker hint to ensure dbghelp.lib is linked.
#pragma comment(lib, "dbghelp.lib")

std::pair<int, std::string> run(const std::vector<char*>& cmd, const std::string& input, bool debug)
{
    struct State
    {
        State()
        {
            ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));
        }
        ~State()
        {
            if(hChildStd_OUT_Rd)
                CloseHandle(hChildStd_OUT_Rd);
            if(hChildStd_OUT_Wr)
                CloseHandle(hChildStd_OUT_Wr);
            if(hChildStd_IN_Rd)
                CloseHandle(hChildStd_IN_Rd);
            if(hChildStd_IN_Wr)
                CloseHandle(hChildStd_IN_Wr);
            if(piProcInfo.hProcess)
                CloseHandle(piProcInfo.hProcess);
            if(piProcInfo.hThread)
                CloseHandle(piProcInfo.hThread);
        }
        HANDLE hChildStd_OUT_Rd = NULL;
        HANDLE hChildStd_OUT_Wr = NULL;
        HANDLE hChildStd_IN_Rd  = NULL;
        HANDLE hChildStd_IN_Wr  = NULL;

        PROCESS_INFORMATION piProcInfo;
    };
    State state;

    // Windows implementation using CreateProcess and pipes.
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength              = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle       = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    // Create a pipe for the child process's STDOUT.
    if(!CreatePipe(&state.hChildStd_OUT_Rd, &state.hChildStd_OUT_Wr, &saAttr, 0))
        throw std::runtime_error("Stdout pipe creation failed");
    // Ensure the read handle is NOT inherited.
    if(!SetHandleInformation(state.hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0))
        throw std::runtime_error("Stdout SetHandleInformation failed");

    // Create a pipe for the child process's STDIN.
    if(!CreatePipe(&state.hChildStd_IN_Rd, &state.hChildStd_IN_Wr, &saAttr, 0))
        throw std::runtime_error("Stdin pipe creation failed");
    if(!SetHandleInformation(state.hChildStd_IN_Wr, HANDLE_FLAG_INHERIT, 0))
        throw std::runtime_error("Stdin SetHandleInformation failed");

    // Build a command-line string from the vector of arguments.
    if(cmd.empty() || cmd.back() != nullptr)
    {
        throw std::runtime_error("Expected non-empty null terminated list of args");
    }
    std::string commandLine;
    for(size_t i = 0; i < cmd.size() - 1; ++i)
    {
        if(i > 0)
            commandLine += " ";
        std::string arg(cmd[i]);
        // Quote the argument if it contains spaces.
        // Note that this can still fail to quote properly on more esoteric
        // things like embedded quotation marks, etc. Since it is being used
        // to invoke controlled tools, we do not expect this and do not guard
        // against it.
        if(arg.find(' ') != std::string::npos)
            commandLine += "\"" + arg + "\"";
        else
            commandLine += arg;
    }
    // Create a mutable command-line buffer as required by CreateProcess.
    std::vector<char> cmdLineMutable(commandLine.begin(), commandLine.end());
    cmdLineMutable.push_back('\0');

    STARTUPINFOA siStartInfo;
    ZeroMemory(&siStartInfo, sizeof(STARTUPINFOA));
    siStartInfo.cb        = sizeof(STARTUPINFOA);
    siStartInfo.hStdInput = state.hChildStd_IN_Rd;
    if(!debug)
    {
        siStartInfo.hStdOutput = state.hChildStd_OUT_Wr;
        siStartInfo.hStdError  = state.hChildStd_OUT_Wr;
    }
    else
    {
        siStartInfo.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        siStartInfo.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    }
    siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

    if(!CreateProcessA(NULL,
                       cmdLineMutable.data(),
                       NULL,
                       NULL,
                       TRUE,
                       0,
                       NULL,
                       NULL,
                       &siStartInfo,
                       &state.piProcInfo))
    {
        std::string message("CreateProcess failed: ");
        message.append(commandLine);
        throw std::runtime_error(std::move(message));
    }

    // Write to child's stdin and close to signal EOF
    if(!input.empty())
    {
        DWORD written = 0;
        if(!WriteFile(state.hChildStd_IN_Wr,
                      input.data(),
                      static_cast<DWORD>(input.size()),
                      &written,
                      nullptr))
        {
            std::string message("Failed to write to child's stdin: ");
            message.append(commandLine);
            throw std::runtime_error(std::move(message));
        }
    }

    CloseHandle(state.hChildStd_IN_Wr);
    state.hChildStd_IN_Wr = NULL;
    CloseHandle(state.hChildStd_IN_Rd);
    state.hChildStd_IN_Rd = NULL;

    std::string result;
    char        buffer[128];
    DWORD       readBytes = 0;
    while(true)
    {
        // See how many bytes are waiting (non‐blocking)
        DWORD available = 0;
        if(!PeekNamedPipe(state.hChildStd_OUT_Rd, nullptr, 0, nullptr, &available, nullptr))
        {
            DWORD err = GetLastError();
            throw std::runtime_error("PeekNamedPipe failed with error " + std::to_string(err));
        }

        // If no data is pending, check if the child has exited
        if(available == 0)
        {
            if(WaitForSingleObject(state.piProcInfo.hProcess, 0) == WAIT_OBJECT_0)
                break; // child is done and no more output
            Sleep(1); // back off briefly to avoid spinning
            continue; // retry
        }

        // Read up to what's available (capped to buffer size)
        DWORD toRead    = (available < sizeof(buffer)) ? available : sizeof(buffer);
        DWORD bytesRead = 0;
        BOOL  success   = ReadFile(state.hChildStd_OUT_Rd, buffer, toRead, &bytesRead, nullptr);
        if(!success)
        {
            DWORD err = GetLastError();
            if(err == ERROR_BROKEN_PIPE)
                break; // pipe closed by child
            throw std::runtime_error("ReadFile failed with error " + std::to_string(err));
        }

        result.append(buffer, bytesRead);
    }
    CloseHandle(state.hChildStd_OUT_Wr);
    state.hChildStd_OUT_Wr = NULL;
    CloseHandle(state.hChildStd_OUT_Rd);
    state.hChildStd_OUT_Rd = NULL;

    // Wait for the child process to finish.
    WaitForSingleObject(state.piProcInfo.hProcess, INFINITE);
    DWORD exitCode = 0;
    GetExitCodeProcess(state.piProcInfo.hProcess, &exitCode);
    return {static_cast<int>(exitCode), result};
}

std::string demangle(const char* name)
{
    std::string result              = name;
    char        demangledName[1024] = {0};
    // UNDNAME_COMPLETE flag produces the full undecorated name.
    if(UnDecorateSymbolName(name, demangledName, sizeof(demangledName), UNDNAME_COMPLETE))
    {
        result = demangledName;
    }
    return result;
}

#else
// POSIX implementation.
#include <sys/wait.h>
#include <unistd.h>
#ifdef __GNUG__
#include <cxxabi.h>
#endif

std::pair<int, std::string> run(const std::vector<char*>& cmd, const std::string& input, bool debug)
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

std::string demangle(const char* name)
{
    std::string result    = name;
    int         status    = -1;
    char*       demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    result                = (status == 0) ? demangled : name;
    free(demangled);
    return result;
}

#endif // POSIX
