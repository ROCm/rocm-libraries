/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#ifndef GUARD_HIPSPARSE_MANAGE_PTR
#define GUARD_HIPSPARSE_MANAGE_PTR

#include "arg_check.hpp"
#include "utility.hpp"

#include <cstdio>
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <memory>
#include <stdexcept>

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    }

namespace hipsparse_test
{

    // device_malloc wraps hipMalloc. A failed allocation throws so the caller does
    // not continue with an invalid pointer. A successful hipMalloc(0) can still
    // return nullptr.
    static void* device_malloc(size_t byte_size)
    {
        void*      pointer = nullptr;
        hipError_t status  = hipMalloc(&pointer, byte_size);
        if(status != hipSuccess)
        {
            // Surface enough context to tell an out-of-memory condition apart
            // from a lost/faulted device on CI runners.
            size_t     free_mem   = 0;
            size_t     total_mem  = 0;
            hipError_t mem_status = hipMemGetInfo(&free_mem, &total_mem);

            char message[512];
            if(mem_status == hipSuccess)
            {
                snprintf(message,
                         sizeof(message),
                         "device_malloc: hipMalloc of %zu bytes failed: hip error %d (%s). "
                         "Device memory free/total = %zu/%zu MB.",
                         byte_size,
                         status,
                         hipGetErrorString(status),
                         free_mem >> 20,
                         total_mem >> 20);
            }
            else
            {
                snprintf(message,
                         sizeof(message),
                         "device_malloc: hipMalloc of %zu bytes failed: hip error %d (%s). "
                         "hipMemGetInfo also failed: hip error %d (%s) -- device may be lost.",
                         byte_size,
                         status,
                         hipGetErrorString(status),
                         mem_status,
                         hipGetErrorString(mem_status));
            }

            throw std::runtime_error(message);
        }
        return pointer;
    }

    // device_free wraps hipFree and provides same API as free
    static void device_free(void* ptr)
    {
        PRINT_IF_HIP_ERROR(hipFree(ptr));
    }

    // Constructors throw so a failed create cannot be used by the rest of the test.
    // Destructors record a failed destroy without throwing, so teardown stays safe
    // while another exception is unwinding.
    static void throw_if_hipsparse_error(hipsparseStatus_t status, const char* call)
    {
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw_hipsparse_error(call, status);
        }
    }

    static void log_if_hipsparse_error(hipsparseStatus_t status, const char* call)
    {
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            log_hipsparse_error(call, status);
        }
    }

    struct descr_struct
    {
        hipsparseMatDescr_t descr = nullptr;

        descr_struct(const descr_struct&) = delete;
        descr_struct& operator=(const descr_struct&) = delete;

        descr_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateMatDescr(&descr), "hipsparseCreateMatDescr");
        }

        ~descr_struct()
        {
            if(descr != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyMatDescr(descr), "hipsparseDestroyMatDescr");
            }
        }
    };

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    struct hyb_struct
    {
        hipsparseHybMat_t hyb = nullptr;

        hyb_struct(const hyb_struct&) = delete;
        hyb_struct& operator=(const hyb_struct&) = delete;

        hyb_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateHybMat(&hyb), "hipsparseCreateHybMat");
        }

        ~hyb_struct()
        {
            if(hyb != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyHybMat(hyb), "hipsparseDestroyHybMat");
            }
        }
    };
#endif

    struct bsrsv2_struct
    {
        bsrsv2Info_t info = nullptr;

        bsrsv2_struct(const bsrsv2_struct&) = delete;
        bsrsv2_struct& operator=(const bsrsv2_struct&) = delete;

        bsrsv2_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateBsrsv2Info(&info), "hipsparseCreateBsrsv2Info");
        }

        ~bsrsv2_struct()
        {
            if(info != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyBsrsv2Info(info),
                                       "hipsparseDestroyBsrsv2Info");
            }
        }
    };

    struct bsrsm2_struct
    {
        bsrsm2Info_t info = nullptr;

        bsrsm2_struct(const bsrsm2_struct&) = delete;
        bsrsm2_struct& operator=(const bsrsm2_struct&) = delete;

        bsrsm2_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateBsrsm2Info(&info), "hipsparseCreateBsrsm2Info");
        }

        ~bsrsm2_struct()
        {
            if(info != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyBsrsm2Info(info),
                                       "hipsparseDestroyBsrsm2Info");
            }
        }
    };

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    struct csrsv2_struct
    {
        csrsv2Info_t info = nullptr;

        csrsv2_struct(const csrsv2_struct&) = delete;
        csrsv2_struct& operator=(const csrsv2_struct&) = delete;

        csrsv2_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateCsrsv2Info(&info), "hipsparseCreateCsrsv2Info");
        }

        ~csrsv2_struct()
        {
            if(info != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyCsrsv2Info(info),
                                       "hipsparseDestroyCsrsv2Info");
            }
        }
    };

    struct csrsm2_struct
    {
        csrsm2Info_t info = nullptr;

        csrsm2_struct(const csrsm2_struct&) = delete;
        csrsm2_struct& operator=(const csrsm2_struct&) = delete;

        csrsm2_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateCsrsm2Info(&info), "hipsparseCreateCsrsm2Info");
        }

        ~csrsm2_struct()
        {
            if(info != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyCsrsm2Info(info),
                                       "hipsparseDestroyCsrsm2Info");
            }
        }
    };
#endif

    struct bsrilu02_struct
    {
        bsrilu02Info_t info = nullptr;

        bsrilu02_struct(const bsrilu02_struct&) = delete;
        bsrilu02_struct& operator=(const bsrilu02_struct&) = delete;

        bsrilu02_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateBsrilu02Info(&info),
                                     "hipsparseCreateBsrilu02Info");
        }

        ~bsrilu02_struct()
        {
            if(info != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyBsrilu02Info(info),
                                       "hipsparseDestroyBsrilu02Info");
            }
        }
    };

    struct csrilu02_struct
    {
        csrilu02Info_t info = nullptr;

        csrilu02_struct(const csrilu02_struct&) = delete;
        csrilu02_struct& operator=(const csrilu02_struct&) = delete;

        csrilu02_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateCsrilu02Info(&info),
                                     "hipsparseCreateCsrilu02Info");
        }

        ~csrilu02_struct()
        {
            if(info != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyCsrilu02Info(info),
                                       "hipsparseDestroyCsrilu02Info");
            }
        }
    };

    struct bsric02_struct
    {
        bsric02Info_t info = nullptr;

        bsric02_struct(const bsric02_struct&) = delete;
        bsric02_struct& operator=(const bsric02_struct&) = delete;

        bsric02_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateBsric02Info(&info),
                                     "hipsparseCreateBsric02Info");
        }

        ~bsric02_struct()
        {
            if(info != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyBsric02Info(info),
                                       "hipsparseDestroyBsric02Info");
            }
        }
    };

    struct csric02_struct
    {
        csric02Info_t info = nullptr;

        csric02_struct(const csric02_struct&) = delete;
        csric02_struct& operator=(const csric02_struct&) = delete;

        csric02_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateCsric02Info(&info),
                                     "hipsparseCreateCsric02Info");
        }

        ~csric02_struct()
        {
            if(info != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyCsric02Info(info),
                                       "hipsparseDestroyCsric02Info");
            }
        }
    };

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    struct csrgemm2_struct
    {
        csrgemm2Info_t info = nullptr;

        csrgemm2_struct(const csrgemm2_struct&) = delete;
        csrgemm2_struct& operator=(const csrgemm2_struct&) = delete;

        csrgemm2_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateCsrgemm2Info(&info),
                                     "hipsparseCreateCsrgemm2Info");
        }

        ~csrgemm2_struct()
        {
            if(info != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyCsrgemm2Info(info),
                                       "hipsparseDestroyCsrgemm2Info");
            }
        }
    };
#endif

    struct prune_struct
    {
        pruneInfo_t info = nullptr;

        prune_struct(const prune_struct&) = delete;
        prune_struct& operator=(const prune_struct&) = delete;

        prune_struct()
        {
            throw_if_hipsparse_error(hipsparseCreatePruneInfo(&info), "hipsparseCreatePruneInfo");
        }

        ~prune_struct()
        {
            if(info != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyPruneInfo(info),
                                       "hipsparseDestroyPruneInfo");
            }
        }
    };

    struct csru2csr_struct
    {
        csru2csrInfo_t info = nullptr;

        csru2csr_struct(const csru2csr_struct&) = delete;
        csru2csr_struct& operator=(const csru2csr_struct&) = delete;

        csru2csr_struct()
        {
            throw_if_hipsparse_error(hipsparseCreateCsru2csrInfo(&info),
                                     "hipsparseCreateCsru2csrInfo");
        }

        ~csru2csr_struct()
        {
            if(info != nullptr)
            {
                log_if_hipsparse_error(hipsparseDestroyCsru2csrInfo(info),
                                       "hipsparseDestroyCsru2csrInfo");
            }
        }
    };

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
    struct spgemm_struct
    {
        hipsparseSpGEMMDescr_t descr = nullptr;

        spgemm_struct(const spgemm_struct&) = delete;
        spgemm_struct& operator=(const spgemm_struct&) = delete;

        spgemm_struct()
        {
            throw_if_hipsparse_error(hipsparseSpGEMM_createDescr(&descr),
                                     "hipsparseSpGEMM_createDescr");
        }

        ~spgemm_struct()
        {
            if(descr != nullptr)
            {
                log_if_hipsparse_error(hipsparseSpGEMM_destroyDescr(descr),
                                       "hipsparseSpGEMM_destroyDescr");
            }
        }
    };
#endif

    // clang-format 19 (math-ci) rewrites "#if(" to "#if (", which clang-format
    // 18 (the repo pre-commit hook) reverts; disable so both formatters agree.
    // clang-format off
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 13030)
    // clang-format on
    struct spgeam_struct
    {
        hipsparseSpGEAMDescr_t descr = nullptr;

        spgeam_struct(const spgeam_struct&) = delete;
        spgeam_struct& operator=(const spgeam_struct&) = delete;

        spgeam_struct()
        {
            throw_if_hipsparse_error(hipsparseSpGEAM_createDescr(&descr),
                                     "hipsparseSpGEAM_createDescr");
        }

        ~spgeam_struct()
        {
            if(descr != nullptr)
            {
                log_if_hipsparse_error(hipsparseSpGEAM_destroyDescr(descr),
                                       "hipsparseSpGEAM_destroyDescr");
            }
        }
    };
#endif

} // namespace hipsparse_test

using hipsparse_unique_ptr = std::unique_ptr<void, void (*)(void*)>;

#endif // GUARD_HIPSPARSE_MANAGE_PTR
