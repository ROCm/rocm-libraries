// Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCFFT_CLIENT_EXCEPT_H
#define ROCFFT_CLIENT_EXCEPT_H

#include <stdexcept>

// exception type to throw when we want to skip a problem
struct ROCFFT_SKIP : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

// exception type to throw when we want to consider a problem failed
struct ROCFFT_FAIL : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

// catch exceptions that may occur in test cases
#define ROCFFT_CATCH_TEST_EXCEPTIONS                                                \
    catch(const std::bad_alloc&)                                                    \
    {                                                                               \
        /* explicitly clear cache */                                                \
        reference_fft_data_t::clear_cache();                                        \
        GTEST_SKIP() << "host memory allocation failure";                           \
    }                                                                               \
    catch(const hip_runtime_error& e)                                               \
    {                                                                               \
        if(skip_runtime_fails)                                                      \
            GTEST_SKIP() << e.what() << "\nHIP error code: " << e.hip_error << "."; \
        else                                                                        \
            GTEST_FAIL() << e.what() << "\nHIP error code: " << e.hip_error << "."; \
    }                                                                               \
    catch(const HOSTBUF_MEM_USAGE& e)                                               \
    {                                                                               \
        /* explicitly clear cache */                                                \
        reference_fft_data_t::clear_cache();                                        \
        GTEST_SKIP() << e.what();                                                   \
    }                                                                               \
    catch(const DEVICEBUF_MEM_USAGE& e)                                             \
    {                                                                               \
        GTEST_SKIP() << e.what();                                                   \
    }                                                                               \
    catch(const ROCFFT_SKIP& e)                                                     \
    {                                                                               \
        GTEST_SKIP() << e.what();                                                   \
    }                                                                               \
    catch(const ROCFFT_FAIL& e)                                                     \
    {                                                                               \
        GTEST_FAIL() << e.what();                                                   \
    }                                                                               \
    catch(const fft_params::unimplemented_exception& e)                             \
    {                                                                               \
        GTEST_SKIP() << "Unimplemented exception: " << e.what();                    \
    }                                                                               \
    catch(...)                                                                      \
    {                                                                               \
        GTEST_FAIL() << "unidentified exception caught during test.";               \
    }

#endif
