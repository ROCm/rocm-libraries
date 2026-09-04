// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCM_INTERFACES_COMMON_H_
#define ROCM_INTERFACES_COMMON_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define ROCM_INTERFACES_EXPORT __declspec(dllexport)
#define ROCM_INTERFACES_CALL __cdecl
#else
#define ROCM_INTERFACES_EXPORT __attribute__((visibility("default")))
#define ROCM_INTERFACES_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ROCM_INTERFACES_ABI_MAJOR 1u
#define ROCM_INTERFACES_ABI_MINOR 1u
#define ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL "rocm_interfaces_provider_query_v1"

typedef enum rocm_interfaces_status {
    ROCM_INTERFACES_STATUS_SUCCESS = 0,
    ROCM_INTERFACES_STATUS_INVALID_ARGUMENT = 1,
    ROCM_INTERFACES_STATUS_OUT_OF_MEMORY = 2,
    ROCM_INTERFACES_STATUS_NOT_SUPPORTED = 3,
    ROCM_INTERFACES_STATUS_NO_SOLUTION = 4,
    ROCM_INTERFACES_STATUS_INCOMPATIBLE_ABI = 5,
    ROCM_INTERFACES_STATUS_PROVIDER_FAILURE = 6,
    ROCM_INTERFACES_STATUS_INVALID_OBJECT = 7,
    ROCM_INTERFACES_STATUS_INTERNAL_ERROR = 8
} rocm_interfaces_status;

typedef enum rocm_interfaces_domain {
    ROCM_INTERFACES_DOMAIN_BLAS = 1,
    ROCM_INTERFACES_DOMAIN_SOLVER = 2,
    ROCM_INTERFACES_DOMAIN_RAND = 3,
    ROCM_INTERFACES_DOMAIN_BLASLT = 4,
    ROCM_INTERFACES_DOMAIN_ROCBLAS_BRIDGE = 5,
    // Experimental semantic rocBLAS protocol. Kept separate from BLAS v1 so
    // the spike can compare both contracts without pretending ABI continuity.
    ROCM_INTERFACES_DOMAIN_BLAS_V2 = 6
} rocm_interfaces_domain;

typedef struct rocm_interfaces_abi_header {
    uint32_t struct_size;
    uint16_t abi_major;
    uint16_t abi_minor;
} rocm_interfaces_abi_header;

typedef struct rocm_interfaces_device_key {
    rocm_interfaces_abi_header header;
    int32_t device_ordinal;
    uint32_t gfx_arch;
    uint64_t feature_mask;
} rocm_interfaces_device_key;

typedef void*(ROCM_INTERFACES_CALL* rocm_interfaces_allocate_fn)(void* user_data, size_t size,
                                                                 size_t alignment);
typedef void(ROCM_INTERFACES_CALL* rocm_interfaces_deallocate_fn)(void* user_data, void* allocation,
                                                                  size_t alignment);
typedef void(ROCM_INTERFACES_CALL* rocm_interfaces_trace_fn)(void* user_data, const char* domain,
                                                             const char* operation,
                                                             const void* payload,
                                                             size_t payload_size);

typedef struct rocm_interfaces_host_services {
    rocm_interfaces_abi_header header;
    void* user_data;
    rocm_interfaces_allocate_fn allocate;
    rocm_interfaces_deallocate_fn deallocate;
    rocm_interfaces_trace_fn trace;
} rocm_interfaces_host_services;

typedef struct rocm_interfaces_provider_request {
    rocm_interfaces_abi_header header;
    rocm_interfaces_domain domain;
    uint32_t required_table_size;
    const rocm_interfaces_host_services* host;
} rocm_interfaces_provider_request;

typedef struct rocm_interfaces_provider_response {
    rocm_interfaces_abi_header header;
    const char* provider_id;
    const char* build_id;
    const void* dispatch_table;
    uint32_t dispatch_table_size;
    uint64_t capability_mask;
} rocm_interfaces_provider_response;

typedef rocm_interfaces_status(ROCM_INTERFACES_CALL* rocm_interfaces_provider_query_fn)(
    const rocm_interfaces_provider_request* request, rocm_interfaces_provider_response* response);

#ifdef __cplusplus
}
#endif
#endif
