// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocblas_narrow_v2_runtime.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>

#include "rocm/interfaces/runtime/provider_registry.h"

#ifndef ROCM_INTERFACES_DEFAULT_ROCBLAS_NARROW_V2_MANIFEST
#define ROCM_INTERFACES_DEFAULT_ROCBLAS_NARROW_V2_MANIFEST ""
#endif

struct _rocblas_handle {
    const rocm_blas_v2_provider* table = nullptr;
    void* context = nullptr;
    hipStream_t stream = nullptr;
    rocblas_pointer_mode pointer_mode = rocblas_pointer_mode_host;
    rocblas_atomics_mode atomics_mode = rocblas_atomics_allowed;
    rocblas_math_mode math_mode = rocblas_default_math;
};

namespace rocm::interfaces {
namespace {
struct Selected {
    std::shared_ptr<ProviderRegistry> registry;
    std::shared_ptr<const ProviderLease> lease;
    const rocm_blas_v2_provider* table;
};
const Selected* selected() noexcept {
    static std::once_flag once;
    static std::unique_ptr<Selected> value;
    std::call_once(once, [] {
        try {
            auto registry = std::make_shared<ProviderRegistry>();
            const char* direct = std::getenv("ROCM_INTERFACES_BLAS_V2_PROVIDER");
            const char* configured_manifest =
                std::getenv("ROCM_INTERFACES_BLAS_V2_PROVIDER_MANIFEST");
            if (direct && *direct) {
                registry->add_module(ROCM_INTERFACES_DOMAIN_BLAS_V2, 0, 0, direct);
            } else {
                std::filesystem::path manifest =
                    configured_manifest && *configured_manifest
                        ? std::filesystem::path(configured_manifest)
                        : std::filesystem::path(ROCM_INTERFACES_DEFAULT_ROCBLAS_NARROW_V2_MANIFEST);
                if (manifest.empty() || !std::filesystem::is_regular_file(manifest)) return;
                registry->load_manifest(manifest);
            }
            auto lease =
                registry->select(ROCM_INTERFACES_DOMAIN_BLAS_V2, 0, sizeof(rocm_blas_v2_provider));
            auto* table = static_cast<const rocm_blas_v2_provider*>(lease->table());
            if (table && table->create_context && table->destroy_context)
                value = std::make_unique<Selected>(Selected{registry, lease, table});
        } catch (...) {
            value.reset();
        }
    });
    return value.get();
}
template <class Request, class Member>
rocblas_status invoke(rocblas_handle handle, const Request* request, Member member) noexcept {
    if (!handle) return rocblas_status_invalid_handle;
    if (!handle->table || !member) return rocblas_status_not_implemented;
    try {
        return member(handle->context, request);
    } catch (...) {
        return rocblas_status_internal_error;
    }
}
}  // namespace

rocm_interfaces_abi_header narrow_v2_header(size_t size) noexcept {
    return {static_cast<uint32_t>(size), ROCM_INTERFACES_ABI_MAJOR, ROCM_INTERFACES_ABI_MINOR};
}
rocm_blas_v2_execution narrow_v2_execution(rocblas_handle handle, rocm_blas_v2_index_width width,
                                           rocm_blas_v2_batch_kind batch, int64_t count) noexcept {
    rocm_blas_v2_execution result{};
    result.header = narrow_v2_header(sizeof(result));
    result.stream = handle ? handle->stream : nullptr;
    result.index_width = width;
    result.batch_kind = batch;
    result.batch_count = count;
    return result;
}
rocblas_pointer_mode narrow_v2_pointer_mode(rocblas_handle h) noexcept {
    return h ? h->pointer_mode : rocblas_pointer_mode_host;
}

#define ROCM_NARROW_DISPATCH(Type, field)                                         \
    rocblas_status narrow_v2_dispatch(rocblas_handle h, const Type* r) noexcept { \
        return invoke(h, r, h && h->table ? h->table->field : nullptr);           \
    }
ROCM_NARROW_DISPATCH(rocm_blas_v2_vector_transform_request, vector_transform)
ROCM_NARROW_DISPATCH(rocm_blas_v2_vector_reduce_request, vector_reduce)
ROCM_NARROW_DISPATCH(rocm_blas_v2_rotation_request, vector_rotate)
ROCM_NARROW_DISPATCH(rocm_blas_v2_matrix_vector_request, matrix_vector)
ROCM_NARROW_DISPATCH(rocm_blas_v2_rank_update_request, rank_update)
ROCM_NARROW_DISPATCH(rocm_blas_v2_structured_matrix_request, structured_matrix)
ROCM_NARROW_DISPATCH(rocm_blas_v2_triangular_matrix_request, triangular_matrix)
ROCM_NARROW_DISPATCH(rocm_blas_v2_matrix_transform_request, matrix_transform)
rocblas_status narrow_v2_dispatch(rocblas_handle h, const rocm_blas_v2_matmul_request* r) noexcept {
    if (!h) return rocblas_status_invalid_handle;
    if (!h->table || !h->table->matmul) return rocblas_status_not_implemented;
    rocm_blas_v2_matmul_result result{narrow_v2_header(sizeof(result)),
                                      ROCM_BLAS_V2_SOLUTION_EXECUTED, 0};
    try {
        return h->table->matmul(h->context, r, &result);
    } catch (...) {
        return rocblas_status_internal_error;
    }
}
#undef ROCM_NARROW_DISPATCH
}  // namespace rocm::interfaces

extern "C" {
rocblas_status rocblas_create_handle(rocblas_handle* out) {
    if (!out) return rocblas_status_invalid_pointer;
    *out = nullptr;
    const auto* s = rocm::interfaces::selected();
    if (!s) return rocblas_status_internal_error;
    auto* h = new (std::nothrow) _rocblas_handle{};
    if (!h) return rocblas_status_memory_error;
    rocm_blas_v2_context_options options{};
    options.header = rocm::interfaces::narrow_v2_header(sizeof(options));
    options.host = &s->registry->host_services();
    h->table = s->table;
    rocblas_status status = h->table->create_context(&options, &h->context);
    if (status != rocblas_status_success) {
        delete h;
        return status;
    }
    *out = h;
    return rocblas_status_success;
}
rocblas_status rocblas_destroy_handle(rocblas_handle h) {
    if (!h) return rocblas_status_invalid_handle;
    try {
        h->table->destroy_context(h->context);
    } catch (...) {
        return rocblas_status_internal_error;
    }
    delete h;
    return rocblas_status_success;
}
rocblas_status rocblas_set_stream(rocblas_handle h, hipStream_t stream) {
    if (!h) return rocblas_status_invalid_handle;
    h->stream = stream;
    return rocblas_status_success;
}
rocblas_status rocblas_get_stream(rocblas_handle h, hipStream_t* stream) {
    if (!h) return rocblas_status_invalid_handle;
    if (!stream) return rocblas_status_invalid_pointer;
    *stream = h->stream;
    return rocblas_status_success;
}
rocblas_status rocblas_set_pointer_mode(rocblas_handle h, rocblas_pointer_mode mode) {
    if (!h) return rocblas_status_invalid_handle;
    if (mode != rocblas_pointer_mode_host && mode != rocblas_pointer_mode_device)
        return rocblas_status_invalid_value;
    h->pointer_mode = mode;
    return rocblas_status_success;
}
rocblas_status rocblas_get_pointer_mode(rocblas_handle h, rocblas_pointer_mode* mode) {
    if (!h) return rocblas_status_invalid_handle;
    if (!mode) return rocblas_status_invalid_pointer;
    *mode = h->pointer_mode;
    return rocblas_status_success;
}
}
