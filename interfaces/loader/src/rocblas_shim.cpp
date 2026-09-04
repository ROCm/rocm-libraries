// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <rocblas/rocblas.h>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>

#include "rocm/interfaces/loader.h"

struct _rocblas_handle {
    std::shared_ptr<rocm::interfaces::BlasContext> context;
};

namespace {
std::shared_ptr<rocm::interfaces::ProviderRegistry> registry() {
    static std::once_flag once;
    static std::shared_ptr<rocm::interfaces::ProviderRegistry> value;
    std::call_once(once, [] {
        const char* path = std::getenv("ROCM_INTERFACES_BLAS_PROVIDER");
        if (!path || !*path) {
            throw std::runtime_error("ROCM_INTERFACES_BLAS_PROVIDER is not set");
        }
        value = std::make_shared<rocm::interfaces::ProviderRegistry>();
        value->add_module(ROCM_INTERFACES_DOMAIN_BLAS, 0, 0, path);
    });
    return value;
}

rocm_interfaces_abi_header header(size_t size) {
    return {static_cast<uint32_t>(size), ROCM_INTERFACES_ABI_MAJOR, ROCM_INTERFACES_ABI_MINOR};
}

rocm_blas_scalar scalar(const float* value, rocblas_pointer_mode mode) {
    return {header(sizeof(rocm_blas_scalar)), rocblas_datatype_f32_r,
            mode == rocblas_pointer_mode_device ? ROCM_BLAS_SCALAR_DEVICE : ROCM_BLAS_SCALAR_HOST,
            value};
}

rocm_blas_matrix matrix(void* data, int64_t rows, int64_t columns, int64_t leading_dimension,
                        int64_t stride) {
    return {header(sizeof(rocm_blas_matrix)),
            rocblas_datatype_f32_r,
            data,
            rows,
            columns,
            leading_dimension,
            stride};
}

rocblas_status matmul(rocblas_handle handle, rocblas_operation trans_a, rocblas_operation trans_b,
                      int64_t m, int64_t n, int64_t k, const float* alpha, const float* a,
                      int64_t lda, int64_t stride_a, const float* b, int64_t ldb, int64_t stride_b,
                      const float* beta, float* c, int64_t ldc, int64_t stride_c,
                      int64_t batch_count, uint32_t index_width, rocm_blas_batch_kind batch_kind) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!alpha || !beta) return rocblas_status_invalid_pointer;
    if (m < 0 || n < 0 || k < 0 || batch_count < 0) return rocblas_status_invalid_size;
    rocm_blas_matmul_request request{};
    request.header = header(sizeof(request));
    request.index_width = index_width;
    request.batch_kind = batch_kind;
    request.batch_count = batch_count;
    request.operation_a = trans_a;
    request.operation_b = trans_b;
    request.compute_type = rocblas_datatype_f32_r;
    rocblas_pointer_mode pointer_mode;
    rocblas_status status = rocblas_get_pointer_mode(handle, &pointer_mode);
    if (status != rocblas_status_success) return status;
    request.alpha = scalar(alpha, pointer_mode);
    request.beta = scalar(beta, pointer_mode);
    request.a = matrix(const_cast<float*>(a), trans_a == rocblas_operation_none ? m : k,
                       trans_a == rocblas_operation_none ? k : m, lda, stride_a);
    request.b = matrix(const_cast<float*>(b), trans_b == rocblas_operation_none ? k : n,
                       trans_b == rocblas_operation_none ? n : k, ldb, stride_b);
    request.c = matrix(c, m, n, ldc, stride_c);
    request.d = request.c;
    return handle->context->matmul_execute(request);
}
}  // namespace

extern "C" {

rocblas_status rocblas_create_handle(rocblas_handle* result) try {
    if (!result) return rocblas_status_invalid_pointer;
    rocm_interfaces_device_key device{};
    device.header = header(sizeof(device));
    if (const char* gfx = std::getenv("ROCM_INTERFACES_TEST_GFX")) {
        device.gfx_arch = static_cast<uint32_t>(std::strtoul(gfx, nullptr, 10));
    }
    auto* handle = new (std::nothrow) _rocblas_handle;
    if (!handle) return rocblas_status_memory_error;
    try {
        handle->context = rocm::interfaces::BlasContext::create(registry(), device);
    } catch (...) {
        delete handle;
        throw;
    }
    *result = handle;
    return rocblas_status_success;
} catch (...) {
    return rocblas_status_internal_error;
}

rocblas_status rocblas_destroy_handle(rocblas_handle handle) {
    if (!handle) return rocblas_status_invalid_handle;
    delete handle;
    return rocblas_status_success;
}

rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream) {
    if (!handle) return rocblas_status_invalid_handle;
    handle->context->set_stream(stream);
    return rocblas_status_success;
}

rocblas_status rocblas_get_stream(rocblas_handle handle, hipStream_t* stream) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!stream) return rocblas_status_invalid_pointer;
    *stream = static_cast<hipStream_t>(handle->context->stream());
    return rocblas_status_success;
}

rocblas_status rocblas_set_pointer_mode(rocblas_handle handle, rocblas_pointer_mode pointer_mode) {
    if (!handle) return rocblas_status_invalid_handle;
    if (pointer_mode != rocblas_pointer_mode_host && pointer_mode != rocblas_pointer_mode_device)
        return rocblas_status_invalid_value;
    handle->context->set_pointer_mode(static_cast<uint32_t>(pointer_mode));
    return rocblas_status_success;
}

rocblas_status rocblas_get_pointer_mode(rocblas_handle handle, rocblas_pointer_mode* pointer_mode) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!pointer_mode) return rocblas_status_invalid_pointer;
    *pointer_mode = static_cast<rocblas_pointer_mode>(handle->context->pointer_mode());
    return rocblas_status_success;
}

rocblas_status rocblas_saxpy(rocblas_handle handle, rocblas_int n, const float* alpha,
                             const float* x, rocblas_int incx, float* y, rocblas_int incy) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!alpha || (!x && n) || (!y && n)) return rocblas_status_invalid_pointer;
    if (n < 0 || !incx || !incy) return rocblas_status_invalid_size;
    rocblas_pointer_mode mode;
    rocblas_status status = rocblas_get_pointer_mode(handle, &mode);
    if (status != rocblas_status_success) return status;
    rocm_blas_vector_request request{};
    request.header = header(sizeof(request));
    request.opcode = ROCM_BLAS_VECTOR_AXPY;
    request.index_width = 32;
    request.batch_kind = ROCM_BLAS_BATCH_SINGLE;
    request.batch_count = 1;
    request.alpha = scalar(alpha, mode);
    request.x = {
        header(sizeof(request.x)), rocblas_datatype_f32_r, const_cast<float*>(x), n, incx, 0};
    request.y = {header(sizeof(request.y)), rocblas_datatype_f32_r, y, n, incy, 0};
    return handle->context->vector_execute(request);
}

rocblas_status rocblas_sdot(rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx,
                            const float* y, rocblas_int incy, float* result) {
    if (!handle) return rocblas_status_invalid_handle;
    if ((!x && n) || (!y && n) || !result) return rocblas_status_invalid_pointer;
    if (n < 0 || !incx || !incy) return rocblas_status_invalid_size;
    rocm_blas_vector_request request{};
    request.header = header(sizeof(request));
    request.opcode = ROCM_BLAS_VECTOR_DOT;
    request.index_width = 32;
    request.batch_kind = ROCM_BLAS_BATCH_SINGLE;
    request.batch_count = 1;
    request.x = {
        header(sizeof(request.x)), rocblas_datatype_f32_r, const_cast<float*>(x), n, incx, 0};
    request.y = {
        header(sizeof(request.y)), rocblas_datatype_f32_r, const_cast<float*>(y), n, incy, 0};
    request.result = result;
    return handle->context->vector_execute(request);
}

rocblas_status rocblas_sgemm(rocblas_handle handle, rocblas_operation trans_a,
                             rocblas_operation trans_b, rocblas_int m, rocblas_int n, rocblas_int k,
                             const float* alpha, const float* a, rocblas_int lda, const float* b,
                             rocblas_int ldb, const float* beta, float* c, rocblas_int ldc) {
    return matmul(handle, trans_a, trans_b, m, n, k, alpha, a, lda, 0, b, ldb, 0, beta, c, ldc, 0,
                  1, 32, ROCM_BLAS_BATCH_SINGLE);
}

rocblas_status rocblas_sgemm_64(rocblas_handle handle, rocblas_operation trans_a,
                                rocblas_operation trans_b, int64_t m, int64_t n, int64_t k,
                                const float* alpha, const float* a, int64_t lda, const float* b,
                                int64_t ldb, const float* beta, float* c, int64_t ldc) {
    return matmul(handle, trans_a, trans_b, m, n, k, alpha, a, lda, 0, b, ldb, 0, beta, c, ldc, 0,
                  1, 64, ROCM_BLAS_BATCH_SINGLE);
}

rocblas_status rocblas_sgemm_strided_batched(rocblas_handle handle, rocblas_operation trans_a,
                                             rocblas_operation trans_b, rocblas_int m,
                                             rocblas_int n, rocblas_int k, const float* alpha,
                                             const float* a, rocblas_int lda,
                                             rocblas_stride stride_a, const float* b,
                                             rocblas_int ldb, rocblas_stride stride_b,
                                             const float* beta, float* c, rocblas_int ldc,
                                             rocblas_stride stride_c, rocblas_int batch_count) {
    return matmul(handle, trans_a, trans_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                  beta, c, ldc, stride_c, batch_count, 32, ROCM_BLAS_BATCH_STRIDED);
}

}  // extern "C"
