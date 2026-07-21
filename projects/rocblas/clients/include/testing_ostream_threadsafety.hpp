/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "cblas_interface.hpp"
#include "client_utility.hpp"
#include "rocblas.hpp"
#include "rocblas_math.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include <cstdio>
#include <sstream>
#include <thread>
#ifdef WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>
#define FDOPEN(A, B) _fdopen(A, B)
#define OPEN(A) _open(A, _O_WRONLY | _O_CREAT | _O_TRUNC | _O_APPEND, _S_IREAD | _S_IWRITE);
#define CLOSE(A) _close(A)
#else
#define FDOPEN(A, B) fdopen(A, B)
#define OPEN(A) open(A, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644);
#define CLOSE(A) close(A)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error no filesystem found
#endif

inline void testing_ostream_threadsafety(const Arguments& arg)
{
    constexpr size_t NTIMES  = 10; // Number of times the tests are repeated across files
    constexpr size_t NLINES  = 5000; // Number of lines each thread outputs
    constexpr size_t MAXLEN  = 2000; // Maximum length of random strings
    constexpr size_t NTHREAD = 16; // Number of threads to run simultaneously
    constexpr size_t SIGLEN  = 16; // Number of characters in signature

    // Signature for detecting garbled output
    auto sig = [](const std::string& s) {
        char h[SIGLEN + 1];
        snprintf(h, sizeof(h), "%0*zX", int(SIGLEN), tuple_helper::hash(s));
        return std::string(h);
    };

    // Verify that the signature matches the string
    auto check_sig
        = [&](const std::string& s) { return s.substr(0, SIGLEN) == sig(s.substr(SIGLEN)); };

    // Each thread writes random strings with signature checksums
    auto thread_func = [&](int fd) {
        rocblas_internal_ostream os(fd);
        for(size_t i = 0; i < NLINES; ++i)
        {
            // Random ASCII string
            auto s = random_string(MAXLEN);

            // Write the signature followed by the random string, flushing at the end
            os << sig(s) << s << std::endl;
        }
    };

    rocblas_seedrand();

    for(size_t n = 0; n < NTIMES; ++n)
    {
        // Open a file in /tmp
        //char path[] = "/tmp/rocblas-XXXXXX";
        fs::path          path;
        std::string       uniquestr;
        const std::string alphanum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv";
        int               stringlength = alphanum.length() - 1;
        uniquestr                      = "rocblas-";
        for(auto n : {0, 1, 2, 3, 4, 5})
            uniquestr += alphanum.at(rand() % stringlength);
        path   = fs::temp_directory_path() / uniquestr;
        int fd = OPEN(path.generic_string().c_str());
        if(fd == -1)
        {
            FAIL() << "Cannot open temporary file " << path;
            return;
        }

        // Launch NTHREAD threads, creating a rocblas_internal_ostream for each thread by duplicating fd
        std::thread threads[NTHREAD];
        for(auto& t : threads)
            t = std::thread(thread_func, fd);

        // Wait for the threads to exit
        for(auto& t : threads)
            t.join();

        // Close the original file descriptor
        if(CLOSE(fd))
            FAIL() << "Could not close filehandle for " << path;

        // Reopen the file to check its integrity
        std::ifstream is(path);
        if(!is.is_open())
        {
            FAIL() << "Could not open " << path;
            return;
        }

        // For each line in the file, make sure its signature matches.
        // This detects interleaved IO which causes garbled output.
        for(std::string line; std::getline(is, line);)
        {
            if(!check_sig(line))
            {
                FAIL() << " detected garbled output in " << path << ":\n\n" << line << "\n";
                return;
            }
        }

        is.close();

#ifdef WIN32
        // need all file descriptors closed to allow file removal on windows before process exits
        rocblas_internal_ostream::clear_workers();
#endif
        // If there were no failures, erase the temporary file
        fs::remove(path);
    }
}
