/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "client_utility.hpp"
#include "rocblas_device_malloc.hpp"

int main()
{
    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    size_t sizes[] = {512, 1025, 10 * 1024 * 1024};

    if(rocblas_is_device_memory_size_query(handle))
    {
        rocblas_cerr << "rocblas_is_device_memory_size_query() incorrectly returned true"
                     << std::endl;
        return 1;
    }

    CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));

    if(!rocblas_is_device_memory_size_query(handle))
    {
        rocblas_cerr << "rocblas_is_device_memory_size_query() incorrectly returned false"
                     << std::endl;
        return 1;
    }

    CHECK_ALLOC_QUERY(rocblas_set_optimal_device_memory_size(handle, sizes[0], sizes[1], sizes[2]));

    for(size_t size : sizes)
    {
        try
        {
            rocblas_device_malloc mem(handle, size);

            if(!mem)
            {
                rocblas_cerr << "rocblas_device_malloc failed" << std::endl;
                return 1;
            }

            void* ptr = static_cast<void*>(mem);

            hipMemset(ptr, 0, size);

            rocblas_device_malloc_base& ref = mem;
        }
        catch(...)
        {
            rocblas_cerr << "exception thrown for malloc size " << size << std::endl;
            return 1;
        }

        rocblas_device_malloc       mem2 = rocblas_device_malloc(handle, size);
        rocblas_device_malloc_base& ref2 = mem2;
        rocblas_device_malloc       mem3 = std::move(mem2);
    }

    try
    {
        rocblas_device_malloc mem(handle, sizes[0], sizes[1], sizes[2]);

        if(!mem)
        {
            rocblas_cerr << "rocblas_device_malloc failed" << std::endl;
            return 1;
        }

        for(size_t i = 0; i < 3; ++i)
        {
            void* ptr = mem[i];
            if(!ptr)
            {
                rocblas_cerr << "nullptr returned from rocblas_device_malloc[" << i << "]"
                             << std::endl;
                return 1;
            }
            hipMemset(ptr, 0, sizes[i]);
        }

        rocblas_device_malloc_base& ref = mem;
    }
    catch(...)
    {
        rocblas_cerr << "exception thrown for malloc size " << sizes[0] << "," << sizes[1] << ","
                     << sizes[2] << std::endl;
        return 1;
    }

    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));

    rocblas_cout << "All tests passed" << std::endl;
}
