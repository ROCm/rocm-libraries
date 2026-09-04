// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

#include "rocm/interfaces/experimental/blas_narrow_v2.h"

#ifndef ROCM_INTERFACES_ROCBLAS_BACKEND_SONAME
#define ROCM_INTERFACES_ROCBLAS_BACKEND_SONAME "librocblas.so.5"
#endif
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define ROCM_INTERFACES_SANITIZED_BUILD 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define ROCM_INTERFACES_SANITIZED_BUILD 1
#endif

namespace {

struct Backend {
    void* module = nullptr;
    decltype(&rocblas_create_handle) create_handle = nullptr;
    decltype(&rocblas_destroy_handle) destroy_handle = nullptr;
    decltype(&rocblas_set_stream) set_stream = nullptr;
    decltype(&rocblas_get_stream) get_stream = nullptr;
    decltype(&rocblas_set_pointer_mode) set_pointer_mode = nullptr;
    decltype(&rocblas_get_pointer_mode) get_pointer_mode = nullptr;
    decltype(&rocblas_saxpy) saxpy = nullptr;
    decltype(&rocblas_saxpy_64) saxpy_64 = nullptr;
    decltype(&rocblas_scopy) scopy = nullptr;
    decltype(&rocblas_scopy_64) scopy_64 = nullptr;
    decltype(&rocblas_sscal) sscal = nullptr;
    decltype(&rocblas_sscal_64) sscal_64 = nullptr;
    decltype(&rocblas_sswap) sswap = nullptr;
    decltype(&rocblas_sswap_64) sswap_64 = nullptr;
};

Backend backend;
std::once_flag backend_once;

struct DlCloser {
    void operator()(void* handle) const noexcept {
        if (handle) dlclose(handle);
    }
};

const char* backend_path() {
    const char* override_path = std::getenv("ROCM_INTERFACES_REAL_ROCBLAS_LIBRARY");
    return override_path && *override_path ? override_path : ROCM_INTERFACES_ROCBLAS_BACKEND_SONAME;
}

template <typename Pointer>
void resolve(void* module, Pointer& result, const char* name) {
    dlerror();
    void* symbol = dlsym(module, name);
    const char* error = dlerror();
    if (!symbol || error)
        throw std::runtime_error(std::string("canonical rocBLAS backend is missing ") + name);
    result = reinterpret_cast<Pointer>(symbol);
}

void initialize_backend() {
    std::call_once(backend_once, [] {
        int flags = RTLD_NOW | RTLD_LOCAL;
#if defined(RTLD_DEEPBIND) && !defined(ROCM_INTERFACES_SANITIZED_BUILD)
        flags |= RTLD_DEEPBIND;
#endif
        void* opened = dlopen(backend_path(), flags);
        if (!opened) {
            const char* error = dlerror();
            throw std::runtime_error(std::string("cannot load canonical rocBLAS backend: ") +
                                     (error ? error : "unknown loader error"));
        }
        std::unique_ptr<void, DlCloser> guard(opened);
        Backend candidate;
        candidate.module = opened;
        resolve(opened, candidate.create_handle, "rocblas_create_handle");
        resolve(opened, candidate.destroy_handle, "rocblas_destroy_handle");
        resolve(opened, candidate.set_stream, "rocblas_set_stream");
        resolve(opened, candidate.get_stream, "rocblas_get_stream");
        resolve(opened, candidate.set_pointer_mode, "rocblas_set_pointer_mode");
        resolve(opened, candidate.get_pointer_mode, "rocblas_get_pointer_mode");
        resolve(opened, candidate.saxpy, "rocblas_saxpy");
        resolve(opened, candidate.saxpy_64, "rocblas_saxpy_64");
        resolve(opened, candidate.scopy, "rocblas_scopy");
        resolve(opened, candidate.scopy_64, "rocblas_scopy_64");
        resolve(opened, candidate.sscal, "rocblas_sscal");
        resolve(opened, candidate.sscal_64, "rocblas_sscal_64");
        resolve(opened, candidate.sswap, "rocblas_sswap");
        resolve(opened, candidate.sswap_64, "rocblas_sswap_64");
        backend = candidate;
        guard.release();
    });
}

struct Context {
    rocblas_handle handle = nullptr;
};

rocblas_status create_context(const rocm_blas_v2_context_options* options, void** result) {
    if (!options || !result) return rocblas_status_invalid_pointer;
    *result = nullptr;
    auto context = std::unique_ptr<Context>(new (std::nothrow) Context);
    if (!context) return rocblas_status_memory_error;
    rocblas_status status = backend.create_handle(&context->handle);
    if (status != rocblas_status_success) return status;
    *result = context.release();
    return rocblas_status_success;
}

void destroy_context(void* opaque) {
    auto context = std::unique_ptr<Context>(static_cast<Context*>(opaque));
    if (context && context->handle) backend.destroy_handle(context->handle);
}

bool fits_i32(int64_t value) {
    return value >= std::numeric_limits<rocblas_int>::min() &&
           value <= std::numeric_limits<rocblas_int>::max();
}

rocblas_status prepare(Context* context, const rocm_blas_v2_execution& execution,
                       const rocm_blas_v2_scalar* scalar) {
    const auto requested_stream = static_cast<hipStream_t>(execution.stream);
    rocblas_status status = backend.set_stream(context->handle, requested_stream);
    if (status != rocblas_status_success) return status;
    hipStream_t observed_stream = nullptr;
    status = backend.get_stream(context->handle, &observed_stream);
    if (status != rocblas_status_success || observed_stream != requested_stream)
        return rocblas_status_internal_error;
    if (scalar) {
        status = backend.set_pointer_mode(context->handle, scalar->location);
        if (status != rocblas_status_success) return status;
        rocblas_pointer_mode observed_mode = rocblas_pointer_mode_host;
        status = backend.get_pointer_mode(context->handle, &observed_mode);
        if (status != rocblas_status_success || observed_mode != scalar->location)
            return rocblas_status_internal_error;
    }
    return rocblas_status_success;
}

rocblas_status vector_transform(void* opaque,
                                const rocm_blas_v2_vector_transform_request* request) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request) return rocblas_status_invalid_pointer;
    if (request->execution.batch_kind != ROCM_BLAS_V2_BATCH_SINGLE ||
        request->execution.batch_count != 1)
        return rocblas_status_not_implemented;
    if (request->x.length < 0) return rocblas_status_invalid_size;
    if (request->x.length == 0) return rocblas_status_success;
    if (request->x.data_type != rocblas_datatype_f32_r) return rocblas_status_not_implemented;
    if (request->operation != ROCM_BLAS_V2_VECTOR_SCALE &&
        request->y.data_type != rocblas_datatype_f32_r)
        return rocblas_status_not_implemented;
    if (!request->x.memory.base) return rocblas_status_invalid_pointer;
    if (request->operation != ROCM_BLAS_V2_VECTOR_SCALE && !request->y.memory.base)
        return rocblas_status_invalid_pointer;
    auto* context = static_cast<Context*>(opaque);
    const bool needs_scalar = request->operation == ROCM_BLAS_V2_VECTOR_SCALE ||
                              request->operation == ROCM_BLAS_V2_VECTOR_AXPY;
    if (needs_scalar &&
        (request->alpha.data_type != rocblas_datatype_f32_r || !request->alpha.value))
        return rocblas_status_invalid_pointer;
    rocblas_status status =
        prepare(context, request->execution, needs_scalar ? &request->alpha : nullptr);
    if (status != rocblas_status_success) return status;

    const int64_t n = request->x.length;
    const int64_t incx = request->x.increment;
    const int64_t incy = request->y.increment;
    const auto* x = static_cast<const float*>(request->x.memory.base);
    auto* mutable_x = static_cast<float*>(request->x.memory.base);
    auto* y = static_cast<float*>(request->y.memory.base);
    if (request->execution.index_width == ROCM_BLAS_V2_INDEX_32 &&
        (!fits_i32(n) || !fits_i32(incx) || !fits_i32(incy)))
        return rocblas_status_invalid_size;

    switch (request->operation) {
        case ROCM_BLAS_V2_VECTOR_SCALE:
            return request->execution.index_width == ROCM_BLAS_V2_INDEX_64
                       ? backend.sscal_64(context->handle, n,
                                          static_cast<const float*>(request->alpha.value),
                                          mutable_x, incx)
                       : backend.sscal(context->handle, static_cast<rocblas_int>(n),
                                       static_cast<const float*>(request->alpha.value), mutable_x,
                                       static_cast<rocblas_int>(incx));
        case ROCM_BLAS_V2_VECTOR_COPY:
            return request->execution.index_width == ROCM_BLAS_V2_INDEX_64
                       ? backend.scopy_64(context->handle, n, x, incx, y, incy)
                       : backend.scopy(context->handle, static_cast<rocblas_int>(n), x,
                                       static_cast<rocblas_int>(incx), y,
                                       static_cast<rocblas_int>(incy));
        case ROCM_BLAS_V2_VECTOR_SWAP:
            return request->execution.index_width == ROCM_BLAS_V2_INDEX_64
                       ? backend.sswap_64(context->handle, n, mutable_x, incx, y, incy)
                       : backend.sswap(context->handle, static_cast<rocblas_int>(n), mutable_x,
                                       static_cast<rocblas_int>(incx), y,
                                       static_cast<rocblas_int>(incy));
        case ROCM_BLAS_V2_VECTOR_AXPY:
            return request->execution.index_width == ROCM_BLAS_V2_INDEX_64
                       ? backend.saxpy_64(context->handle, n,
                                          static_cast<const float*>(request->alpha.value), x, incx,
                                          y, incy)
                       : backend.saxpy(context->handle, static_cast<rocblas_int>(n),
                                       static_cast<const float*>(request->alpha.value), x,
                                       static_cast<rocblas_int>(incx), y,
                                       static_cast<rocblas_int>(incy));
        default:
            return rocblas_status_not_implemented;
    }
}

template <typename Request>
rocblas_status unsupported(void* opaque, const Request* request) {
    if (!opaque) return rocblas_status_invalid_handle;
    return request ? rocblas_status_not_implemented : rocblas_status_invalid_pointer;
}

rocblas_status vector_reduce(void* context, const rocm_blas_v2_vector_reduce_request* request) {
    return unsupported(context, request);
}
rocblas_status vector_rotate(void* context, const rocm_blas_v2_rotation_request* request) {
    return unsupported(context, request);
}
rocblas_status matrix_vector(void* context, const rocm_blas_v2_matrix_vector_request* request) {
    return unsupported(context, request);
}
rocblas_status rank_update(void* context, const rocm_blas_v2_rank_update_request* request) {
    return unsupported(context, request);
}
rocblas_status matmul_query(void* context, const rocm_blas_v2_matmul_request* request,
                            rocm_blas_v2_solution*, size_t, size_t* count) {
    if (count) *count = 0;
    return unsupported(context, request);
}
rocblas_status matmul(void* context, const rocm_blas_v2_matmul_request* request,
                      rocm_blas_v2_matmul_result*) {
    return unsupported(context, request);
}
rocblas_status structured_matrix(void* context,
                                 const rocm_blas_v2_structured_matrix_request* request) {
    return unsupported(context, request);
}
rocblas_status triangular_matrix(void* context,
                                 const rocm_blas_v2_triangular_matrix_request* request) {
    return unsupported(context, request);
}
rocblas_status matrix_transform(void* context,
                                const rocm_blas_v2_matrix_transform_request* request) {
    return unsupported(context, request);
}

const rocm_blas_v2_provider table = {
    {sizeof(table), ROCM_INTERFACES_ABI_MAJOR, ROCM_INTERFACES_ABI_MINOR},
    create_context,
    destroy_context,
    vector_transform,
    vector_reduce,
    vector_rotate,
    matrix_vector,
    rank_update,
    matmul_query,
    matmul,
    structured_matrix,
    triangular_matrix,
    matrix_transform,
};

void trace_failure(const rocm_interfaces_provider_request* request, const char* message) {
    if (request && request->host && request->host->trace)
        request->host->trace(request->host->user_data, "blas_v2", "backend_load_failure", message,
                             std::char_traits<char>::length(message));
}

}  // namespace

extern "C" ROCM_INTERFACES_EXPORT rocm_interfaces_status ROCM_INTERFACES_CALL
rocm_interfaces_provider_query_v1(const rocm_interfaces_provider_request* request,
                                  rocm_interfaces_provider_response* response) {
    if (!request || !response || request->domain != ROCM_INTERFACES_DOMAIN_BLAS_V2 ||
        request->header.abi_major != ROCM_INTERFACES_ABI_MAJOR ||
        response->header.struct_size < sizeof(*response) ||
        request->required_table_size > sizeof(table))
        return ROCM_INTERFACES_STATUS_INCOMPATIBLE_ABI;
    try {
        initialize_backend();
    } catch (const std::exception& error) {
        trace_failure(request, error.what());
        return ROCM_INTERFACES_STATUS_PROVIDER_FAILURE;
    } catch (...) {
        trace_failure(request, "unknown canonical rocBLAS backend failure");
        return ROCM_INTERFACES_STATUS_PROVIDER_FAILURE;
    }
    response->header = {sizeof(*response), ROCM_INTERFACES_ABI_MAJOR, ROCM_INTERFACES_ABI_MINOR};
    response->provider_id = "system-rocblas-narrow-v2";
    response->build_id = "interfaces-real-v1";
    response->dispatch_table = &table;
    response->dispatch_table_size = sizeof(table);
    response->capability_mask = 0;
    return ROCM_INTERFACES_STATUS_SUCCESS;
}
