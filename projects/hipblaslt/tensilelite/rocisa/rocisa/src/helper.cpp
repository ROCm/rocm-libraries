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

#include <cstdlib>

// This file includes platform specific helpers that differ between POSIX and
// Windows. The alternatives are switched at a whole-file level. Please do
// not use inline/fine-grained ifdefs.

#ifdef _WIN32

#include <windows.h>
// windows.h must be loaded before other windows headers.
#include <dbghelp.h>

// Linker hint to ensure dbghelp.lib is linked.
#pragma comment(lib, "dbghelp.lib")

// The child's output is collected whenever it isn't shown (debug), so
// captureOutput changes nothing here.
std::pair<int, std::string>
    run(const std::vector<char*>& cmd, const std::string& input, bool debug, bool)
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
    // run() is called from several threads at once (rocIsa::initMany), and
    // CreateProcess hands each child every inheritable handle in the process.
    // So the pipes start non-inheritable and only the child's ends are made
    // inheritable: if a parent end were ever inheritable, another thread's
    // child could hold this child's stdin open, and it would never see EOF.
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength              = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle       = FALSE;
    saAttr.lpSecurityDescriptor = NULL;

    // Create a pipe for the child process's STDOUT.
    if(!CreatePipe(&state.hChildStd_OUT_Rd, &state.hChildStd_OUT_Wr, &saAttr, 0))
        throw std::runtime_error("Stdout pipe creation failed");
    if(!SetHandleInformation(state.hChildStd_OUT_Wr, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT))
        throw std::runtime_error("Stdout SetHandleInformation failed");

    // Create a pipe for the child process's STDIN.
    if(!CreatePipe(&state.hChildStd_IN_Rd, &state.hChildStd_IN_Wr, &saAttr, 0))
        throw std::runtime_error("Stdin pipe creation failed");
    if(!SetHandleInformation(state.hChildStd_IN_Rd, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT))
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
    bool        childExited = false;
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
            if(childExited)
                break;
            DWORD waitResult = WaitForSingleObject(state.piProcInfo.hProcess, 0);
            if(waitResult == WAIT_FAILED)
                throw std::runtime_error("Waiting for the child failed with error "
                                         + std::to_string(GetLastError()));
            if(waitResult == WAIT_OBJECT_0)
            {
                // The child may have written after PeekNamedPipe saw an empty
                // pipe. Check again after observing its exit to drain that output.
                childExited = true;
                continue;
            }
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

    // Wait for the child process to finish. Without its exit code there is no
    // answer; 0 would read as an accepted source.
    DWORD exitCode = 0;
    if(WaitForSingleObject(state.piProcInfo.hProcess, INFINITE) != WAIT_OBJECT_0
       || !GetExitCodeProcess(state.piProcInfo.hProcess, &exitCode))
    {
        DWORD err = GetLastError();
        throw std::runtime_error("Waiting for the child failed with error " + std::to_string(err));
    }
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

std::string nullDevicePath()
{
    // Not plain "NUL": LLVM turns a relative path under a long working directory
    // into a \\?\ path, where NUL is an ordinary file name.
    return "\\\\.\\NUL";
}

#else
// POSIX implementation.
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstring>
#include <cxxabi.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace
{
    // Writes input to the child's stdin and, given its output pipe, reads what
    // it writes until it closes the pipe. Both go through one poll loop, so a
    // child that writes before it has read all of its input can't fill the
    // output pipe while we wait to write. Closes both descriptors. Returns 0, or
    // the errno of a failed poll, when the output may be incomplete.
    int exchange(int toChild, const std::string& input, int fromChild, std::string& output)
    {
        size_t written = 0;
        int    error   = 0;
        if(input.empty())
        {
            close(toChild);
            toChild = -1;
        }
        while(toChild != -1 || fromChild != -1)
        {
            // poll skips the negative descriptors.
            pollfd fds[2] = {{toChild, POLLOUT, 0}, {fromChild, POLLIN, 0}};
            // Python's signal handlers don't restart interrupted system calls, so
            // poll, write, read and waitpid below retry on EINTR. An interrupted
            // write would leave the child an empty or partial source, which the
            // assembler may well accept.
            if(poll(fds, 2, -1) == -1)
            {
                if(errno == EINTR)
                    continue;
                error = errno;
                break;
            }
            if(toChild != -1 && fds[0].revents)
            {
                // Once poll reports the pipe writable, PIPE_BUF bytes go in without blocking.
                ssize_t n = write(toChild,
                                  input.data() + written,
                                  std::min<size_t>(input.size() - written, PIPE_BUF));
                if(n > 0)
                    written += static_cast<size_t>(n);
                // Done, or the child stopped reading; its exit status tells the rest.
                if(written == input.size() || (n == -1 && errno != EINTR))
                {
                    close(toChild);
                    toChild = -1;
                }
            }
            if(fromChild != -1 && fds[1].revents)
            {
                char    buffer[4096];
                ssize_t n = read(fromChild, buffer, sizeof(buffer));
                if(n > 0)
                    output.append(buffer, static_cast<size_t>(n));
                else if(n == 0 || errno != EINTR)
                {
                    close(fromChild);
                    fromChild = -1;
                }
            }
        }
        if(toChild != -1)
            close(toChild);
        if(fromChild != -1)
            close(fromChild);
        return error;
    }
}

std::pair<int, std::string>
    run(const std::vector<char*>& cmd, const std::string& input, bool debug, bool captureOutput)
{
    // The pipes feed input to the child's stdin and, with captureOutput, bring
    // back its stdout and stderr. O_CLOEXEC keeps them out of children forked
    // concurrently by other threads: an inherited write end would keep this
    // child's stdin, or our end of its output, from ever seeing EOF.
    int   p[2];
    int   o[2] = {-1, -1};
    pid_t pid;
    if(pipe2(p, O_CLOEXEC) == -1)
    {
        throw std::runtime_error("cmd failed!");
    }
    if(captureOutput && pipe2(o, O_CLOEXEC) == -1)
    {
        close(p[0]);
        close(p[1]);
        throw std::runtime_error("cmd failed!");
    }

    pid = fork();
    if(pid == -1)
    {
        close(p[0]);
        close(p[1]);
        if(captureOutput)
        {
            close(o[0]);
            close(o[1]);
        }
        throw std::runtime_error("fork failed!");
    }
    if(pid == 0)
    {
        // dup2 clears FD_CLOEXEC on the new descriptor, but a descriptor that
        // already has the right number keeps it, so clear it there instead. The
        // pipe fds themselves are closed by execvp.
        auto redirect = [](int fd, int target) {
            if(fd == target)
                fcntl(fd, F_SETFD, 0);
            else
                dup2(fd, target);
        };
        redirect(p[0], STDIN_FILENO);
        if(captureOutput)
        {
            redirect(o[1], STDOUT_FILENO);
            redirect(o[1], STDERR_FILENO);
        }
        else if(!debug)
        {
            // Discard the child's output; only its exit status is reported.
            int devNull = open("/dev/null", O_WRONLY);
            if(devNull != -1)
            {
                dup2(devNull, STDOUT_FILENO);
                dup2(devNull, STDERR_FILENO);
                if(devNull > STDERR_FILENO)
                    close(devNull);
            }
        }
        execvp(cmd[0], cmd.data());
        perror("execvp");
        // _exit (not exit) is required in a forked child to avoid re-running the
        // parent's atexit handlers.
        _exit(execFailedExitCode);
    }
    else
    {
        close(p[0]);
        if(captureOutput)
            close(o[1]);
        std::string result;
        int         lost   = exchange(p[1], input, o[0], result);
        int         status = 0;
        pid_t       waited;
        do
            waited = waitpid(pid, &status, 0);
        while(waited == -1 && errno == EINTR);
        // Without the child's status there is no answer; status 0 would read as accepted.
        if(waited == -1)
            throw std::runtime_error(std::string("waitpid failed: ") + std::strerror(errno));
        if(lost)
            throw std::runtime_error(std::string("poll failed: ") + std::strerror(lost));
        // Like a shell, a child killed by a signal is 128 + the signal, so it can't
        // read as an exit code (SIGHUP would otherwise look like a rejection, 1).
        int rcode = status;
        if(WIFEXITED(status))
            rcode = WEXITSTATUS(status);
        else if(WIFSIGNALED(status))
            rcode = 128 + WTERMSIG(status);
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

std::string nullDevicePath()
{
    return "/dev/null";
}

#endif // POSIX
