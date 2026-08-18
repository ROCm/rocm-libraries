/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "allocate_free.hpp"
#include "def.hpp"
#include "log.hpp"

#ifdef SUPPORT_MULTINODE
#include "communicator.hpp"
#endif

#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace rocalution
{
    //#define MEM_ALIGNMENT 64
    //#define LONG_PTR size_t
    //#define LONG_PTR long

    template <typename DataType>
    void allocate_host(int64_t n, DataType** ptr)
    {
        log_debug(0, "allocate_host()", "* begin", n, *ptr);

        if(n > 0)
        {
            assert(*ptr == NULL);

            // *********************************************************
            // C++ style
            //    *ptr = new DataType[n];
            // *********************************************************

            // *********************************************************
            // C style
            // *ptr =  (DataType *) malloc(n*sizeof(DataType));
            // *********************************************************

            // *********************************************************
            // C style (zero-set)
            // *ptr = (DataType *) calloc(n, sizeof(DatatType));
            // *********************************************************

            // *********************************************************
            // Aligned allocation
            // total n = (n*datatype) + (alignment-1) + (void ptr)
            // void *non_aligned =  malloc(n*sizeof(DataType)+(MEM_ALIGNMENT-1)+sizeof(void*));
            // assert(non_aligned != NULL);

            // void *aligned = (void*)( ((LONG_PTR)(non_aligned)+MEM_ALIGNMENT+sizeof(void*) ) &
            // ~(MEM_ALIGNMENT-1) );
            // *((void**)aligned-1) = non_aligned;

            // *ptr = (DataType*) aligned;

            // LOG_INFO("A " << *ptr << " " <<  aligned << " " << non_aligned << " "<<  sizeof(DataType)
            // << " " << n);
            // *********************************************************

            // *********************************************************
            // C++ style and error handling

            *ptr = new(std::nothrow) DataType[n];

            if(!(*ptr))
            { // nullptr
                LOG_INFO("Cannot allocate memory");
                LOG_VERBOSE_INFO(2, "Size of the requested buffer = " << n * sizeof(DataType));
                FATAL_ERROR(__FILE__, __LINE__);
            }
            // *********************************************************

            assert(*ptr != NULL);
        }

        log_debug(0, "allocate_host()", "* end", *ptr);
    }

#ifndef SUPPORT_HIP
    template <typename DataType>
    void allocate_pinned(int64_t n, DataType** ptr)
    {
        log_debug(0, "allocate_pinned()", n, *ptr);

        allocate_host(n, ptr);
    }
#endif

    template <typename DataType>
    void free_host(DataType** ptr)
    {
        log_debug(0, "free_host()", *ptr);

        if(*ptr == NULL)
        {
            // Do nothing
            return;
        }

        // *********************************************************
        // C++ style
        delete[] * ptr;
        // *********************************************************

        // *********************************************************
        // C style
        // free(*ptr);
        // *********************************************************

        // *********************************************************
        // Aligned allocation
        //  free(*((void **)*ptr-1));
        // *********************************************************

        *ptr = NULL;
    }

#ifndef SUPPORT_HIP
    template <typename DataType>
    void free_pinned(DataType** ptr)
    {
        log_debug(0, "free_host()", *ptr);

        free_host(ptr);
    }
#endif

    template <typename DataType>
    void set_to_zero_host(int64_t n, DataType* ptr)
    {
        log_debug(0, "set_to_zero_host()", n, ptr);

        if(n > 0)
        {
            assert(ptr != NULL);

            memset(ptr, 0, n * sizeof(DataType));

            // for (int64_t i=0; i<n; ++i)
            //   ptr[i] = DataType(0);
        }
    }

    template <typename DataType>
    void copy_h2h(int64_t n, const DataType* src, DataType* dst)
    {
        log_debug(0, "copy_h2h()", n, src, dst);

        if(n > 0)
        {
            assert(src != NULL);
            assert(dst != NULL);

#if 0
            _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int64_t i = 0; i < n; ++i)
            {
                dst[i] = src[i];
            }
#else
            memcpy(dst, src, sizeof(DataType) * n);
#endif
        }
    }

    template void allocate_host<float>(int64_t, float**);
    template void allocate_host<double>(int64_t, double**);
#ifdef SUPPORT_COMPLEX
    template void allocate_host<std::complex<float>>(int64_t, std::complex<float>**);
    template void allocate_host<std::complex<double>>(int64_t, std::complex<double>**);
#endif
    template void allocate_host<bool>(int64_t, bool**);
    template void allocate_host<int>(int64_t, int**);
    template void allocate_host<unsigned int>(int64_t, unsigned int**);
    template void allocate_host<int64_t>(int64_t, int64_t**);
    template void allocate_host<char>(int64_t, char**);
#ifdef SUPPORT_MULTINODE
    template void allocate_host<MRequest>(int64_t, MRequest**);
#endif

#ifndef SUPPORT_HIP
    template void allocate_pinned<float>(int64_t, float**);
    template void allocate_pinned<double>(int64_t, double**);
#ifdef SUPPORT_COMPLEX
    template void allocate_pinned<std::complex<float>>(int64_t, std::complex<float>**);
    template void allocate_pinned<std::complex<double>>(int64_t, std::complex<double>**);
#endif
#endif

    template void free_host<float>(float**);
    template void free_host<double>(double**);
#ifdef SUPPORT_COMPLEX
    template void free_host<std::complex<float>>(std::complex<float>**);
    template void free_host<std::complex<double>>(std::complex<double>**);
#endif
    template void free_host<bool>(bool**);
    template void free_host<int>(int**);
    template void free_host<unsigned int>(unsigned int**);
    template void free_host<int64_t>(int64_t**);
    template void free_host<char>(char**);
#ifdef SUPPORT_MULTINODE
    template void free_host<MRequest>(MRequest**);
#endif

#ifndef SUPPORT_HIP
    template void free_pinned<float>(float**);
    template void free_pinned<double>(double**);
#ifdef SUPPORT_COMPLEX
    template void free_pinned<std::complex<float>>(std::complex<float>**);
    template void free_pinned<std::complex<double>>(std::complex<double>**);
#endif
#endif

    template void set_to_zero_host<float>(int64_t, float*);
    template void set_to_zero_host<double>(int64_t, double*);
#ifdef SUPPORT_COMPLEX
    template void set_to_zero_host<std::complex<float>>(int64_t, std::complex<float>*);
    template void set_to_zero_host<std::complex<double>>(int64_t, std::complex<double>*);
#endif
    template void set_to_zero_host<bool>(int64_t, bool*);
    template void set_to_zero_host<int>(int64_t, int*);
    template void set_to_zero_host<unsigned int>(int64_t, unsigned int*);
    template void set_to_zero_host<int64_t>(int64_t, int64_t*);
    template void set_to_zero_host<char>(int64_t, char*);

    template void copy_h2h<float>(int64_t, const float*, float*);
    template void copy_h2h<double>(int64_t, const double*, double*);
#ifdef SUPPORT_COMPLEX
    template void
        copy_h2h<std::complex<float>>(int64_t, const std::complex<float>*, std::complex<float>*);
    template void
        copy_h2h<std::complex<double>>(int64_t, const std::complex<double>*, std::complex<double>*);
#endif
    template void copy_h2h<bool>(int64_t, const bool*, bool*);
    template void copy_h2h<int>(int64_t, const int*, int*);
    template void copy_h2h<unsigned int>(int64_t, const unsigned int*, unsigned int*);
    template void copy_h2h<int64_t>(int64_t, const int64_t*, int64_t*);
    template void copy_h2h<char>(int64_t, const char*, char*);
} // namespace rocalution
