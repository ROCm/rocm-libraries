/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#include <cerrno>
#include <csetjmp>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <regex>
#ifdef WIN32
#include <windows.h>
#define strcasecmp(A, B) _stricmp(A, B)
#else
#include <pthread.h>
#include <unistd.h>
#endif

#include "client_utility.hpp"
#include "gtest_helpers.hpp"
#include "rocblas.h"

testing::AssertionResult status_match(rocblas_status expected, rocblas_status status)
{
    if(expected == status)
        return testing::AssertionSuccess();
    else
        return testing::AssertionFailure() << "got " << rocblas_status_to_string(status)
                                           << " instead of " << rocblas_status_to_string(expected);
}

void rocblas_client_set_gtest_filter(const char* filter_string)
{
#ifdef GOOGLE_TEST
    testing::GTEST_FLAG(filter) = filter_string;
    // newer gtest form caused compile errors
    // GTEST_FLAG_SET(filter, filter_string);
#endif
}

/*********************************************
 * Signal-handling for detecting test faults *
 *********************************************/

// State of the signal handler
static thread_local struct
{
    // Whether sigjmp_buf is set and catching signals is enabled
    volatile sig_atomic_t enabled = false;

    // sigjmp_buf describing stack frame to go back to
#ifndef WIN32
    sigjmp_buf sigjmp_buf;
#else
    jmp_buf sigjmp_buf;
#endif

    // The signal which was received
    volatile sig_atomic_t signal;
} t_handler;

// Signal handler (must have external "C" linkage)
extern "C" void rocblas_test_signal_handler(int sig)
{
    int saved_errno = errno; // Save errno

    // If the signal handler is disabled, because we're not in the middle of
    // running a rocBLAS test, restore this signal's disposition to default,
    // and reraise the signal
    if(!t_handler.enabled)
    {
        signal(sig, SIG_DFL);
        errno = saved_errno;
        raise(sig);
        return;
    }

    rocblas_cerr << "SIGNAL raised in: "
                 << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;

#ifndef WIN32
    // If this is an alarm timeout, we abort
    if(sig == SIGALRM)
    {
        static constexpr char msg[]
            = "\nAborting tests due to an alarm timeout.\n\n"
              "This could be due to a deadlock caused by mutexes being left locked\n"
              "after a previous test's signal was caught and partially recovered from.\n";
        // We must use write() because it's async-signal-safe and other IO might be blocked
        write(STDERR_FILENO, msg, sizeof(msg) - 1);
        rocblas_abort();
    }
#endif

    // Jump back to the handler code after setting handler.signal
    // Note: This bypasses stack unwinding, and may lead to memory leaks, but
    // it is better than crashing.
    t_handler.signal = sig;
    errno            = saved_errno;
#ifndef WIN32
    siglongjmp(t_handler.sigjmp_buf, true);
#else
    longjmp(t_handler.sigjmp_buf, true);
#endif
}

// Set up signal handlers
void rocblas_test_sigaction()
{
#ifndef WIN32
    struct sigaction act;
    act.sa_flags = 0;
    sigfillset(&act.sa_mask);
    act.sa_handler = rocblas_test_signal_handler;

    // Catch SIGALRM and synchronous signals
    for(int sig : {SIGALRM, SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV})
        sigaction(sig, &act, nullptr);
#else
    for(int sig : {SIGABRT, SIGFPE, SIGILL, SIGINT, SIGSEGV, SIGTERM})
        signal(sig, rocblas_test_signal_handler);
#endif
}

static const unsigned test_timeout = [] {
    // Number of seconds each test is allowed to take before all testing is killed.
    constexpr unsigned TEST_TIMEOUT = 600;
    unsigned           timeout;
    const char*        env = getenv("ROCBLAS_TEST_TIMEOUT");
    return env && sscanf(env, "%u", &timeout) == 1 ? timeout : TEST_TIMEOUT;
}();

// Lambda wrapper which detects signals and exceptions in an invokable function
void catch_signals_and_exceptions_as_failures(std::function<void()> test, bool set_alarm)
{
#ifdef GOOGLE_TEST

    // Save the current handler (to allow nested calls to this function)
    auto old_handler = t_handler;

#ifndef WIN32
    // Set up the return point, and handle siglongjmp returning back to here
    if(sigsetjmp(t_handler.sigjmp_buf, true))
    {
#if(__GLIBC__ < 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 32)
        FAIL() << "Received " << sys_siglist[t_handler.signal] << " signal";
#else
        FAIL() << "Received " << sigdescr_np(t_handler.signal) << " signal";
#endif
    }
#else
    if(setjmp(t_handler.sigjmp_buf))
    {
        FAIL() << "Received signal";
    }
#endif
    else
    {
#ifndef WIN32
        // Alarm to detect deadlocks or hangs
        if(set_alarm)
            alarm(test_timeout);
#endif
        // Enable the signal handler
        t_handler.enabled = true;

        // Run the test function, catching signals and exceptions
        try
        {
            test();
        }
        catch(const std::bad_alloc& e)
        {
            GTEST_SKIP() << LIMITED_RAM_STRING;
        }
        catch(const std::exception& e)
        {
            FAIL() << "Received uncaught exception: " << e.what();
        }
        catch(...)
        {
            FAIL() << "Received uncaught exception";
        }
    }

#ifndef WIN32
    // Cancel the alarm if it was set
    if(set_alarm)
        alarm(0);
#endif
    // Restore the previous handler
    t_handler = old_handler;

    const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();

    if(hipPeekAtLastError() != hipSuccess)
    {
        rocblas_cerr << "hipGetLastError at end of test: "
                     << (test_info ? test_info->name() : "Unknown") << std::endl;
        (void)rocblas_internal_convert_hip_to_rocblas_status_and_log(
            hipGetLastError()); // clear last error
    }

#endif
}
