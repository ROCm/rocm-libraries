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

#include "hip_allocate_free.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "hip_kernels_general.hpp"
#include "hip_utils.hpp"
#include "rocalution/utils/types.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <complex>

namespace rocalution
{
#ifdef ROCALUTION_HIP_PINNED_MEMORY
    template <typename DataType>
    void allocate_host(int64_t n, DataType** ptr)
    {
        log_debug(0, "allocate_host()", n, ptr);

        if(n > 0)
        {
            assert(*ptr == NULL);

            //    *ptr = new DataType[n];

            hipMallocHost((void**)ptr, n * sizeof(DataType));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            assert(*ptr != NULL);
        }
    }

    template <typename DataType>
    void free_host(DataType** ptr)
    {
        log_debug(0, "free_host()", *ptr);

        assert(*ptr != NULL);

        //  delete[] *ptr;
        hipFreeHost(*ptr);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        *ptr = NULL;
    }
#endif

    template <typename DataType>
    void allocate_hip(int64_t n, DataType** ptr)
    {
        log_debug(0, "allocate_hip()", n, ptr);

        if(n > 0)
        {
            assert(*ptr == NULL);

            hipMalloc((void**)ptr, n * sizeof(DataType));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            assert(*ptr != NULL);
        }
    }

    template <typename DataType>
    void allocate_pinned(int64_t n, DataType** ptr)
    {
        log_debug(0, "allocate_pinned()", n, ptr);

        if(n > 0)
        {
            assert(*ptr == NULL);

            if(_rocalution_available_accelerator() == true)
            {
                hipHostMalloc((void**)ptr, n * sizeof(DataType));
                CHECK_HIP_ERROR(__FILE__, __LINE__);
            }
            else
            {
                allocate_host(n, ptr);
            }

            assert(*ptr != NULL);
        }
    }

    template <typename DataType>
    void free_hip(DataType** ptr)
    {
        log_debug(0, "free_hip()", *ptr);

        if(*ptr != NULL)
        {
            hipFree(*ptr);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            *ptr = NULL;
        }
    }

    template <typename DataType>
    void free_pinned(DataType** ptr)
    {
        log_debug(0, "free_pinned()", *ptr);

        if(*ptr != NULL)
        {
            if(_rocalution_available_accelerator() == true)
            {
                hipHostFree(*ptr);
                CHECK_HIP_ERROR(__FILE__, __LINE__);
            }
            else
            {
                free_host(ptr);
            }

            *ptr = NULL;
        }
    }

    template <typename DataType>
    void set_to_zero_hip(int blocksize, int64_t n, DataType* ptr, bool async, hipStream_t stream)
    {
        log_debug(0, "set_to_zero_hip()", blocksize, n, ptr, async, stream);

        if(n > 0)
        {
            assert(ptr != NULL);

            if(async == false)
            {
                hipMemset(ptr, 0, n * sizeof(DataType));
            }
            else
            {
                hipMemsetAsync(ptr, 0, n * sizeof(DataType), stream);
            }

            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename DataType>
    void set_to_one_hip(int blocksize, int64_t n, DataType* ptr, bool async, hipStream_t stream)
    {
        log_debug(0, "set_to_one_hip()", blocksize, n, ptr, async, stream);
        if(n > 0)
        {
            assert(ptr != NULL);
            // 1D accessing, no stride
            dim3 BlockSize(blocksize);
            dim3 GridSize(n / blocksize + 1);

            if(async == false)
            {
                kernel_set_to_value<<<GridSize, BlockSize>>>(n, ptr, static_cast<DataType>(1));
            }
            else
            {
                kernel_set_to_value<<<GridSize, BlockSize, 0, stream>>>(
                    n, ptr, static_cast<DataType>(1));
            }

            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename DataType>
    void set_to_value_hip(
        int blocksize, int64_t n, DataType* ptr, DataType value, bool async, hipStream_t stream)
    {
        log_debug(0, "set_to_value_hip()", blocksize, n, ptr, value, async, stream);

        if(n > 0)
        {
            assert(ptr != NULL);

            // 1D accessing, no stride
            dim3 BlockSize(blocksize);
            dim3 GridSize(n / blocksize + 1);

            if(async == false)
            {
                kernel_set_to_value<<<GridSize, BlockSize>>>(n, ptr, value);
            }
            else
            {
                kernel_set_to_value<<<GridSize, BlockSize, 0, stream>>>(n, ptr, value);
            }

            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename DataType>
    void copy_d2h(int64_t n, const DataType* src, DataType* dst, bool async, hipStream_t stream)
    {
        log_debug(0, "copy_d2h()", n, src, dst);

        if(n > 0)
        {
            assert(src != NULL);
            assert(dst != NULL);

            if(async == false)
            {
                hipMemcpy(dst, src, sizeof(DataType) * n, hipMemcpyDeviceToHost);
            }
            else
            {
                hipMemcpyAsync(dst, src, sizeof(DataType) * n, hipMemcpyDeviceToHost, stream);
            }

            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename DataType>
    void copy_h2d(int64_t n, const DataType* src, DataType* dst, bool async, hipStream_t stream)
    {
        log_debug(0, "copy_h2d()", n, src, dst);

        if(n > 0)
        {
            assert(src != NULL);
            assert(dst != NULL);

            if(async == false)
            {
                hipMemcpy(dst, src, sizeof(DataType) * n, hipMemcpyHostToDevice);
            }
            else
            {
                hipMemcpyAsync(dst, src, sizeof(DataType) * n, hipMemcpyHostToDevice, stream);
            }

            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename DataType>
    void copy_d2d(int64_t n, const DataType* src, DataType* dst, bool async, hipStream_t stream)
    {
        log_debug(0, "copy_d2d()", n, src, dst);

        if(n > 0)
        {
            assert(src != NULL);
            assert(dst != NULL);

            if(async == false)
            {
                hipMemcpy(dst, src, sizeof(DataType) * n, hipMemcpyDeviceToDevice);
            }
            else
            {
                hipMemcpyAsync(dst, src, sizeof(DataType) * n, hipMemcpyDeviceToDevice, stream);
            }

            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

#ifdef ROCALUTION_HIP_PINNED_MEMORY
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
#endif

    template void allocate_hip<float>(int64_t, float**);
    template void allocate_hip<double>(int64_t, double**);
#ifdef SUPPORT_COMPLEX
    template void allocate_hip<std::complex<float>>(int64_t, std::complex<float>**);
    template void allocate_hip<std::complex<double>>(int64_t, std::complex<double>**);
#endif
    template void allocate_hip<bool>(int64_t, bool**);
    template void allocate_hip<int>(int64_t, int**);
    template void allocate_hip<unsigned int>(int64_t, unsigned int**);
    template void allocate_hip<int64_t>(int64_t, int64_t**);
    template void allocate_hip<char>(int64_t, char**);
    template void allocate_hip<mis_tuple>(int64_t, mis_tuple**);

    template void allocate_pinned<float>(int64_t, float**);
    template void allocate_pinned<double>(int64_t, double**);
#ifdef SUPPORT_COMPLEX
    template void allocate_pinned<std::complex<float>>(int64_t, std::complex<float>**);
    template void allocate_pinned<std::complex<double>>(int64_t, std::complex<double>**);
#endif

    template void free_hip<float>(float**);
    template void free_hip<double>(double**);
#ifdef SUPPORT_COMPLEX
    template void free_hip<std::complex<float>>(std::complex<float>**);
    template void free_hip<std::complex<double>>(std::complex<double>**);
#endif
    template void free_hip<bool>(bool**);
    template void free_hip<int>(int**);
    template void free_hip<unsigned int>(unsigned int**);
    template void free_hip<int64_t>(int64_t**);
    template void free_hip<char>(char**);
    template void free_hip<mis_tuple>(mis_tuple**);

    template void free_pinned<float>(float**);
    template void free_pinned<double>(double**);
#ifdef SUPPORT_COMPLEX
    template void free_pinned<std::complex<float>>(std::complex<float>**);
    template void free_pinned<std::complex<double>>(std::complex<double>**);
#endif

    template void set_to_zero_hip<float>(int, int64_t, float*, bool, hipStream_t);
    template void set_to_zero_hip<double>(int, int64_t, double*, bool, hipStream_t);
#ifdef SUPPORT_COMPLEX
    template void
        set_to_zero_hip<std::complex<float>>(int, int64_t, std::complex<float>*, bool, hipStream_t);
    template void set_to_zero_hip<std::complex<double>>(
        int, int64_t, std::complex<double>*, bool, hipStream_t);
#endif
    template void set_to_zero_hip<bool>(int, int64_t, bool*, bool, hipStream_t);
    template void set_to_zero_hip<int>(int, int64_t, int*, bool, hipStream_t);
    template void set_to_zero_hip<int64_t>(int, int64_t, int64_t*, bool, hipStream_t);

    template void set_to_one_hip<float>(int, int64_t, float*, bool, hipStream_t);
    template void set_to_one_hip<double>(int, int64_t, double*, bool, hipStream_t);
#ifdef SUPPORT_COMPLEX
    template void
        set_to_one_hip<std::complex<float>>(int, int64_t, std::complex<float>*, bool, hipStream_t);
    template void set_to_one_hip<std::complex<double>>(
        int, int64_t, std::complex<double>*, bool, hipStream_t);
#endif
    template void set_to_one_hip<bool>(int, int64_t, bool*, bool, hipStream_t);
    template void set_to_one_hip<int>(int, int64_t, int*, bool, hipStream_t);
    template void set_to_one_hip<int64_t>(int, int64_t, int64_t*, bool, hipStream_t);

    template void set_to_value_hip<float>(int, int64_t, float*, float, bool, hipStream_t);
    template void set_to_value_hip<double>(int, int64_t, double*, double, bool, hipStream_t);
#ifdef SUPPORT_COMPLEX
    template void set_to_value_hip<std::complex<float>>(
        int, int64_t, std::complex<float>*, std::complex<float>, bool, hipStream_t);
    template void set_to_value_hip<std::complex<double>>(
        int, int64_t, std::complex<double>*, std::complex<double>, bool, hipStream_t);
#endif
    template void set_to_value_hip<bool>(int, int64_t, bool*, bool, bool, hipStream_t);
    template void set_to_value_hip<int>(int, int64_t, int*, int, bool, hipStream_t);
    template void set_to_value_hip<int64_t>(int, int64_t, int64_t*, int64_t, bool, hipStream_t);

    template void copy_d2h<float>(int64_t, const float*, float*, bool, hipStream_t);
    template void copy_d2h<double>(int64_t, const double*, double*, bool, hipStream_t);
#ifdef SUPPORT_COMPLEX
    template void copy_d2h<std::complex<float>>(
        int64_t, const std::complex<float>*, std::complex<float>*, bool, hipStream_t);
    template void copy_d2h<std::complex<double>>(
        int64_t, const std::complex<double>*, std::complex<double>*, bool, hipStream_t);
#endif
    template void copy_d2h<int>(int64_t, const int*, int*, bool, hipStream_t);
    template void copy_d2h<int64_t>(int64_t, const int64_t*, int64_t*, bool, hipStream_t);
    template void copy_d2h<bool>(int64_t, const bool*, bool*, bool, hipStream_t);

    template void copy_h2d<float>(int64_t, const float*, float*, bool, hipStream_t);
    template void copy_h2d<double>(int64_t, const double*, double*, bool, hipStream_t);
#ifdef SUPPORT_COMPLEX
    template void copy_h2d<std::complex<float>>(
        int64_t, const std::complex<float>*, std::complex<float>*, bool, hipStream_t);
    template void copy_h2d<std::complex<double>>(
        int64_t, const std::complex<double>*, std::complex<double>*, bool, hipStream_t);
#endif
    template void copy_h2d<int>(int64_t, const int*, int*, bool, hipStream_t);
    template void copy_h2d<int64_t>(int64_t, const int64_t*, int64_t*, bool, hipStream_t);
    template void copy_h2d<bool>(int64_t, const bool*, bool*, bool, hipStream_t);

    template void copy_d2d<float>(int64_t, const float*, float*, bool, hipStream_t);
    template void copy_d2d<double>(int64_t, const double*, double*, bool, hipStream_t);
#ifdef SUPPORT_COMPLEX
    template void copy_d2d<std::complex<float>>(
        int64_t, const std::complex<float>*, std::complex<float>*, bool, hipStream_t);
    template void copy_d2d<std::complex<double>>(
        int64_t, const std::complex<double>*, std::complex<double>*, bool, hipStream_t);
#endif
    template void copy_d2d<int>(int64_t, const int*, int*, bool, hipStream_t);
    template void copy_d2d<int64_t>(int64_t, const int64_t*, int64_t*, bool, hipStream_t);
    template void copy_d2d<bool>(int64_t, const bool*, bool*, bool, hipStream_t);
} // namespace rocalution
