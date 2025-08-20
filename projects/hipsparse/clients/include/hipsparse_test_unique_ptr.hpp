/* ************************************************************************
 * Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <cstdio>
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <memory>

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

    // device_malloc wraps hipMalloc and provides same API as malloc
    static void* device_malloc(size_t byte_size)
    {
        void* pointer;
        PRINT_IF_HIP_ERROR(hipMalloc(&pointer, byte_size));
        return pointer;
    }

    // device_free wraps hipFree and provides same API as free
    static void device_free(void* ptr)
    {
        PRINT_IF_HIP_ERROR(hipFree(ptr));
    }

    struct handle_struct
    {
        hipsparseHandle_t handle;
        handle_struct()
        {
            hipsparseStatus_t status = hipsparseCreate(&handle);
            verify_hipsparse_status_success(status, "ERROR: handle_struct constructor");
        }

        ~handle_struct()
        {
            hipsparseStatus_t status = hipsparseDestroy(handle);
            verify_hipsparse_status_success(status, "ERROR: handle_struct destructor");
        }
    };

    struct descr_struct
    {
        hipsparseMatDescr_t descr;
        descr_struct()
        {
            hipsparseStatus_t status = hipsparseCreateMatDescr(&descr);
            verify_hipsparse_status_success(status, "ERROR: descr_struct constructor");
        }

        ~descr_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyMatDescr(descr);
            verify_hipsparse_status_success(status, "ERROR: descr_struct destructor");
        }
    };

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    struct hyb_struct
    {
        hipsparseHybMat_t hyb;
        hyb_struct()
        {
            hipsparseStatus_t status = hipsparseCreateHybMat(&hyb);
            verify_hipsparse_status_success(status, "ERROR: hyb_struct constructor");
        }

        ~hyb_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyHybMat(hyb);
            verify_hipsparse_status_success(status, "ERROR: hyb_struct destructor");
        }
    };
#endif

    struct bsrsv2_struct
    {
        bsrsv2Info_t info;
        bsrsv2_struct()
        {
            hipsparseStatus_t status = hipsparseCreateBsrsv2Info(&info);
            verify_hipsparse_status_success(status, "ERROR: bsrsv2_struct constructor");
        }

        ~bsrsv2_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyBsrsv2Info(info);
            verify_hipsparse_status_success(status, "ERROR: bsrsv2_struct destructor");
        }
    };

    struct bsrsm2_struct
    {
        bsrsm2Info_t info;
        bsrsm2_struct()
        {
            hipsparseStatus_t status = hipsparseCreateBsrsm2Info(&info);
            verify_hipsparse_status_success(status, "ERROR: bsrsm2_struct constructor");
        }

        ~bsrsm2_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyBsrsm2Info(info);
            verify_hipsparse_status_success(status, "ERROR: bsrsm2_struct destructor");
        }
    };

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    struct csrsv2_struct
    {
        csrsv2Info_t info;
        csrsv2_struct()
        {
            hipsparseStatus_t status = hipsparseCreateCsrsv2Info(&info);
            verify_hipsparse_status_success(status, "ERROR: csrsv2_struct constructor");
        }

        ~csrsv2_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyCsrsv2Info(info);
            verify_hipsparse_status_success(status, "ERROR: csrsv2_struct destructor");
        }
    };

    struct csrsm2_struct
    {
        csrsm2Info_t info;
        csrsm2_struct()
        {
            hipsparseStatus_t status = hipsparseCreateCsrsm2Info(&info);
            verify_hipsparse_status_success(status, "ERROR: csrsm2_struct constructor");
        }

        ~csrsm2_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyCsrsm2Info(info);
            verify_hipsparse_status_success(status, "ERROR: csrsm2_struct destructor");
        }
    };
#endif

    struct bsrilu02_struct
    {
        bsrilu02Info_t info;
        bsrilu02_struct()
        {
            hipsparseStatus_t status = hipsparseCreateBsrilu02Info(&info);
            verify_hipsparse_status_success(status, "ERROR: bsrilu02_struct constructor");
        }

        ~bsrilu02_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyBsrilu02Info(info);
            verify_hipsparse_status_success(status, "ERROR: bsrilu02_struct destructor");
        }
    };

    struct csrilu02_struct
    {
        csrilu02Info_t info;
        csrilu02_struct()
        {
            hipsparseStatus_t status = hipsparseCreateCsrilu02Info(&info);
            verify_hipsparse_status_success(status, "ERROR: csrilu02_struct constructor");
        }

        ~csrilu02_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyCsrilu02Info(info);
            verify_hipsparse_status_success(status, "ERROR: csrilu02_struct destructor");
        }
    };

    struct bsric02_struct
    {
        bsric02Info_t info;
        bsric02_struct()
        {
            hipsparseStatus_t status = hipsparseCreateBsric02Info(&info);
            verify_hipsparse_status_success(status, "ERROR: bsric02_struct constructor");
        }

        ~bsric02_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyBsric02Info(info);
            verify_hipsparse_status_success(status, "ERROR: bsric02_struct destructor");
        }
    };

    struct csric02_struct
    {
        csric02Info_t info;
        csric02_struct()
        {
            hipsparseStatus_t status = hipsparseCreateCsric02Info(&info);
            verify_hipsparse_status_success(status, "ERROR: csric02_struct constructor");
        }

        ~csric02_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyCsric02Info(info);
            verify_hipsparse_status_success(status, "ERROR: csric02_struct destructor");
        }
    };

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    struct csrgemm2_struct
    {
        csrgemm2Info_t info;
        csrgemm2_struct()
        {
            hipsparseStatus_t status = hipsparseCreateCsrgemm2Info(&info);
            verify_hipsparse_status_success(status, "ERROR: csrgemm2_struct constructor");
        }

        ~csrgemm2_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyCsrgemm2Info(info);
            verify_hipsparse_status_success(status, "ERROR: csrgemm2_struct destructor");
        }
    };
#endif

    struct prune_struct
    {
        pruneInfo_t info;
        prune_struct()
        {
            hipsparseStatus_t status = hipsparseCreatePruneInfo(&info);
            verify_hipsparse_status_success(status, "ERROR: prune_struct constructor");
        }

        ~prune_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyPruneInfo(info);
            verify_hipsparse_status_success(status, "ERROR: prune_struct destructor");
        }
    };

    struct csru2csr_struct
    {
        csru2csrInfo_t info;
        csru2csr_struct()
        {
            hipsparseStatus_t status = hipsparseCreateCsru2csrInfo(&info);
            verify_hipsparse_status_success(status, "ERROR: csru2csr_struct constructor");
        }

        ~csru2csr_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyCsru2csrInfo(info);
            verify_hipsparse_status_success(status, "ERROR: csru2csr_struct destructor");
        }
    };

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
    struct spgemm_struct
    {
        hipsparseSpGEMMDescr_t descr;
        spgemm_struct()
        {
            hipsparseStatus_t status = hipsparseSpGEMM_createDescr(&descr);
            verify_hipsparse_status_success(status, "ERROR: spgemm_struct constructor");
        }

        ~spgemm_struct()
        {
            hipsparseStatus_t status = hipsparseSpGEMM_destroyDescr(descr);
            verify_hipsparse_status_success(status, "ERROR: spgemm_struct destructor");
        }
    };
#endif

} // namespace hipsparse_test

using hipsparse_unique_ptr = std::unique_ptr<void, void (*)(void*)>;

#endif // GUARD_HIPSPARSE_MANAGE_PTR
