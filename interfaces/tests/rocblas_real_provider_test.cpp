// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <dlfcn.h>
#include <rocblas/rocblas.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "rocblas_bridge_generated.h"
#include "rocm/interfaces/experimental/blas_narrow_v2.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define ROCM_INTERFACES_SANITIZED_BUILD 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define ROCM_INTERFACES_SANITIZED_BUILD 1
#endif

namespace {

template <typename Function>
Function symbol(void* module, const char* name) {
    dlerror();
    void* address = dlsym(module, name);
    const char* error = dlerror();
    if (!address || error) throw std::runtime_error(std::string("missing backend symbol: ") + name);
    return reinterpret_cast<Function>(address);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        setenv("ROCM_INTERFACES_ROCBLAS_BRIDGE_PROVIDER", REAL_PROVIDER_PATH, 1);
        setenv("ROCM_INTERFACES_REAL_ROCBLAS_LIBRARY", REAL_ROCBLAS_PATH, 1);

        int flags = RTLD_NOW | RTLD_LOCAL;
#if defined(RTLD_DEEPBIND) && !defined(ROCM_INTERFACES_SANITIZED_BUILD)
        flags |= RTLD_DEEPBIND;
#endif
        void* backend = dlopen(REAL_ROCBLAS_PATH, flags);
        if (!backend) throw std::runtime_error(dlerror());

        void* exhaustive_provider = dlopen(REAL_PROVIDER_PATH, RTLD_NOW | RTLD_LOCAL);
        if (!exhaustive_provider) throw std::runtime_error(dlerror());
        const auto exhaustive_query = reinterpret_cast<rocm_interfaces_provider_query_fn>(
            dlsym(exhaustive_provider, ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL));
        require(exhaustive_query != nullptr, "real exhaustive provider query export is missing");
        rocm_interfaces_provider_request exhaustive_request{};
        exhaustive_request.header = {sizeof(exhaustive_request), ROCM_INTERFACES_ABI_MAJOR,
                                     ROCM_INTERFACES_ABI_MINOR};
        exhaustive_request.domain = ROCM_INTERFACES_DOMAIN_ROCBLAS_BRIDGE;
        exhaustive_request.required_table_size = sizeof(rocm_rocblas_bridge_v1);
        rocm_interfaces_provider_response exhaustive_response{};
        exhaustive_response.header = {sizeof(exhaustive_response), ROCM_INTERFACES_ABI_MAJOR,
                                      ROCM_INTERFACES_ABI_MINOR};
        require(exhaustive_query(&exhaustive_request, &exhaustive_response) ==
                    ROCM_INTERFACES_STATUS_SUCCESS,
                "real exhaustive provider did not bind canonical rocBLAS");
        const auto* exhaustive_table =
            static_cast<const rocm_rocblas_bridge_v1*>(exhaustive_response.dispatch_table);
        const auto direct_grouped = reinterpret_cast<decltype(&rocblas_sgemm_grouped_batched)>(
            dlsym(backend, "rocblas_sgemm_grouped_batched"));
        if (!direct_grouped)
            require(exhaustive_table->rocblas_sgemm_grouped_batched(
                        reinterpret_cast<rocblas_handle>(1), nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr, 0, nullptr) == rocblas_status_success,
                    "grouped-GEMM compatibility adapter rejected an empty group list");

        const auto direct_status_to_string =
            symbol<decltype(&rocblas_status_to_string)>(backend, "rocblas_status_to_string");
        const auto direct_version_size = symbol<decltype(&rocblas_get_version_string_size)>(
            backend, "rocblas_get_version_string_size");
        const auto direct_version =
            symbol<decltype(&rocblas_get_version_string)>(backend, "rocblas_get_version_string");
        const auto direct_create =
            symbol<decltype(&rocblas_create_handle)>(backend, "rocblas_create_handle");
        const auto direct_destroy =
            symbol<decltype(&rocblas_destroy_handle)>(backend, "rocblas_destroy_handle");
        const auto direct_set_pointer =
            symbol<decltype(&rocblas_set_pointer_mode)>(backend, "rocblas_set_pointer_mode");
        const auto direct_get_pointer =
            symbol<decltype(&rocblas_get_pointer_mode)>(backend, "rocblas_get_pointer_mode");
        const auto direct_saxpy = symbol<decltype(&rocblas_saxpy)>(backend, "rocblas_saxpy");

        require(std::strcmp(rocblas_status_to_string(rocblas_status_invalid_handle),
                            direct_status_to_string(rocblas_status_invalid_handle)) == 0,
                "status strings differ between direct and provider paths");

        size_t direct_size = 0;
        size_t provider_size = 0;
        require(direct_version_size(&direct_size) == rocblas_status_success,
                "direct version-size query failed");
        require(rocblas_get_version_string_size(&provider_size) == rocblas_status_success,
                "provider version-size query failed");
        require(direct_size == provider_size && direct_size > 0,
                "version-size result differs between direct and provider paths");
        std::string direct_text(direct_size, '\0');
        std::string provider_text(provider_size, '\0');
        require(direct_version(direct_text.data(), direct_text.size()) == rocblas_status_success,
                "direct version query failed");
        require(rocblas_get_version_string(provider_text.data(), provider_text.size()) ==
                    rocblas_status_success,
                "provider version query failed");
        require(direct_text == provider_text,
                "version result differs between direct and provider paths");

        float alpha = 1.0f;
        float x = 0.0f;
        float y = 0.0f;
        require(direct_saxpy(nullptr, 1, &alpha, &x, 1, &y, 1) ==
                    rocblas_saxpy(nullptr, 1, &alpha, &x, 1, &y, 1),
                "invalid-handle SAXPY behavior differs");

        rocblas_handle direct_handle = nullptr;
        rocblas_handle provider_handle = nullptr;
        const rocblas_status direct_create_status = direct_create(&direct_handle);
        const rocblas_status provider_create_status = rocblas_create_handle(&provider_handle);
        require(direct_create_status == provider_create_status,
                "handle creation status differs between direct and provider paths");
        if (direct_create_status == rocblas_status_success) {
            require(direct_handle && provider_handle, "successful handle creation returned null");
            require(direct_set_pointer(direct_handle, rocblas_pointer_mode_device) ==
                        rocblas_set_pointer_mode(provider_handle, rocblas_pointer_mode_device),
                    "set-pointer-mode status differs");
            rocblas_pointer_mode direct_mode = rocblas_pointer_mode_host;
            rocblas_pointer_mode provider_mode = rocblas_pointer_mode_host;
            require(direct_get_pointer(direct_handle, &direct_mode) ==
                        rocblas_get_pointer_mode(provider_handle, &provider_mode),
                    "get-pointer-mode status differs");
            require(direct_mode == provider_mode, "pointer modes differ");

            if (direct_grouped)
                require(direct_grouped(direct_handle, nullptr, nullptr, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, 0, nullptr) ==
                            rocblas_sgemm_grouped_batched(provider_handle, nullptr, nullptr,
                                                          nullptr, nullptr, nullptr, nullptr,
                                                          nullptr, nullptr, nullptr, nullptr,
                                                          nullptr, nullptr, nullptr, 0, nullptr),
                        "native grouped-GEMM behavior differs");

            require(direct_destroy(direct_handle) == rocblas_status_success,
                    "direct handle destruction failed");
            require(rocblas_destroy_handle(provider_handle) == rocblas_status_success,
                    "provider handle destruction failed");
        }

        void* narrow_provider = dlopen(REAL_NARROW_PROVIDER_PATH, RTLD_NOW | RTLD_LOCAL);
        if (!narrow_provider) throw std::runtime_error(dlerror());
        const auto narrow_query = reinterpret_cast<rocm_interfaces_provider_query_fn>(
            dlsym(narrow_provider, ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL));
        require(narrow_query != nullptr, "real narrow provider query export is missing");
        rocm_interfaces_provider_request narrow_request{};
        narrow_request.header = {sizeof(narrow_request), ROCM_INTERFACES_ABI_MAJOR,
                                 ROCM_INTERFACES_ABI_MINOR};
        narrow_request.domain = ROCM_INTERFACES_DOMAIN_BLAS_V2;
        narrow_request.required_table_size = sizeof(rocm_blas_v2_provider);
        rocm_interfaces_provider_response narrow_response{};
        narrow_response.header = {sizeof(narrow_response), ROCM_INTERFACES_ABI_MAJOR,
                                  ROCM_INTERFACES_ABI_MINOR};
        require(narrow_query(&narrow_request, &narrow_response) == ROCM_INTERFACES_STATUS_SUCCESS,
                "real narrow provider did not bind canonical rocBLAS");
        const auto* narrow_table =
            static_cast<const rocm_blas_v2_provider*>(narrow_response.dispatch_table);
        rocm_blas_v2_context_options narrow_options{};
        narrow_options.header = {sizeof(narrow_options), ROCM_INTERFACES_ABI_MAJOR,
                                 ROCM_INTERFACES_ABI_MINOR};
        void* narrow_context = nullptr;
        rocblas_handle narrow_direct_handle = nullptr;
        const rocblas_status narrow_direct_status = direct_create(&narrow_direct_handle);
        const rocblas_status narrow_provider_status =
            narrow_table->create_context(&narrow_options, &narrow_context);
        require(narrow_direct_status == narrow_provider_status,
                "narrow provider handle status differs from canonical rocBLAS");
        if (narrow_direct_status == rocblas_status_success) {
            direct_destroy(narrow_direct_handle);
            narrow_table->destroy_context(narrow_context);
        }
        dlclose(narrow_provider);
        dlclose(exhaustive_provider);
        dlclose(backend);
        std::cout << "real rocBLAS provider differential checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
