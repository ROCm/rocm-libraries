// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocblas_bridge_runtime.h"

#include <cstdarg>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#include "rocm/interfaces/runtime/provider_registry.h"

#ifndef ROCM_INTERFACES_DEFAULT_ROCBLAS_MANIFEST
#define ROCM_INTERFACES_DEFAULT_ROCBLAS_MANIFEST ""
#endif

struct _rocblas_handle {
    const rocm_rocblas_bridge_v1* table = nullptr;
    rocblas_handle native_handle = nullptr;
};

namespace rocm::interfaces {
namespace {
struct Bridge {
    std::shared_ptr<ProviderRegistry> registry;
    std::shared_ptr<const ProviderLease> lease;
    const rocm_rocblas_bridge_v1* table;
};

const Bridge* bridge() noexcept {
    static std::once_flag once;
    static std::unique_ptr<Bridge> value;
    std::call_once(once, [] {
        try {
            auto registry = std::make_shared<ProviderRegistry>();
            const char* direct = std::getenv("ROCM_INTERFACES_ROCBLAS_BRIDGE_PROVIDER");
            const char* configured_manifest =
                std::getenv("ROCM_INTERFACES_ROCBLAS_PROVIDER_MANIFEST");
            if (direct && *direct) {
                registry->add_module(ROCM_INTERFACES_DOMAIN_ROCBLAS_BRIDGE, 0, 0, direct);
            } else {
                std::filesystem::path manifest =
                    configured_manifest && *configured_manifest
                        ? std::filesystem::path(configured_manifest)
                        : std::filesystem::path(ROCM_INTERFACES_DEFAULT_ROCBLAS_MANIFEST);
                if (manifest.empty() || !std::filesystem::is_regular_file(manifest)) return;
                registry->load_manifest(manifest);
            }
            auto lease = registry->select(ROCM_INTERFACES_DOMAIN_ROCBLAS_BRIDGE, 0,
                                          sizeof(rocm_rocblas_bridge_v1));
            auto* table = static_cast<const rocm_rocblas_bridge_v1*>(lease->table());
            if (table && table->rocblas_create_handle && table->rocblas_destroy_handle) {
                value =
                    std::make_unique<Bridge>(Bridge{std::move(registry), std::move(lease), table});
            }
        } catch (...) {
            value.reset();
        }
    });
    return value.get();
}
}  // namespace

const rocm_rocblas_bridge_v1* rocblas_bridge_table() noexcept {
    const Bridge* selected = bridge();
    return selected ? selected->table : nullptr;
}

const rocm_rocblas_bridge_v1* rocblas_bridge_table(rocblas_handle handle) noexcept {
    return handle ? handle->table : nullptr;
}

rocblas_handle rocblas_bridge_native_handle(rocblas_handle handle) noexcept {
    return handle ? handle->native_handle : nullptr;
}
}  // namespace rocm::interfaces

extern "C" {

rocblas_status rocblas_create_handle(rocblas_handle* result) {
    if (!result) return rocblas_status_invalid_pointer;
    *result = nullptr;
    const auto* selected = rocm::interfaces::rocblas_bridge_table();
    if (!selected || !selected->rocblas_create_handle) return rocblas_status_internal_error;
    rocblas_handle native = nullptr;
    rocblas_status status;
    try {
        status = selected->rocblas_create_handle(&native);
    } catch (...) {
        return rocblas_status_internal_error;
    }
    if (status != rocblas_status_success) return status;
    if (!native) return rocblas_status_internal_error;
    auto* wrapper = new (std::nothrow) _rocblas_handle;
    if (!wrapper) {
        try {
            selected->rocblas_destroy_handle(native);
        } catch (...) {
        }
        return rocblas_status_memory_error;
    }
    wrapper->table = selected;
    wrapper->native_handle = native;
    *result = wrapper;
    return rocblas_status_success;
}

rocblas_status rocblas_destroy_handle(rocblas_handle handle) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!handle->table || !handle->table->rocblas_destroy_handle)
        return rocblas_status_internal_error;
    rocblas_status status;
    try {
        status = handle->table->rocblas_destroy_handle(handle->native_handle);
    } catch (...) {
        return rocblas_status_internal_error;
    }
    if (status == rocblas_status_success) delete handle;
    return status;
}

rocblas_status rocblas_set_optimal_device_memory_size_impl(rocblas_handle handle, size_t count,
                                                           ...) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!handle->table || !handle->table->rocblas_set_optimal_device_memory_sizes)
        return rocblas_status_internal_error;
    try {
        std::vector<size_t> sizes(count);
        va_list arguments;
        va_start(arguments, count);
        for (size_t i = 0; i < count; ++i) sizes[i] = va_arg(arguments, size_t);
        va_end(arguments);
        return handle->table->rocblas_set_optimal_device_memory_sizes(handle->native_handle, count,
                                                                      sizes.data());
    } catch (...) {
        return rocblas_status_memory_error;
    }
}

rocblas_status rocblas_device_malloc_alloc(rocblas_handle handle,
                                           struct rocblas_device_malloc_base** result, size_t count,
                                           ...) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!handle->table || !handle->table->rocblas_device_malloc_alloc_sizes)
        return rocblas_status_internal_error;
    try {
        std::vector<size_t> sizes(count);
        va_list arguments;
        va_start(arguments, count);
        for (size_t i = 0; i < count; ++i) sizes[i] = va_arg(arguments, size_t);
        va_end(arguments);
        return handle->table->rocblas_device_malloc_alloc_sizes(handle->native_handle, result,
                                                                count, sizes.data());
    } catch (...) {
        return rocblas_status_memory_error;
    }
}

}  // extern "C"
