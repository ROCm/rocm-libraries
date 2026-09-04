// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCM_INTERFACES_RAND_H_
#define ROCM_INTERFACES_RAND_H_

#include <rocrand/rocrand.h>

#include "rocm/interfaces/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rocm_rand_generator_kind {
    ROCM_RAND_GENERATOR_DEVICE = 1,
    ROCM_RAND_GENERATOR_HOST = 2,
    ROCM_RAND_GENERATOR_HOST_BLOCKING = 3
} rocm_rand_generator_kind;

typedef enum rocm_rand_distribution {
    ROCM_RAND_RAW = 1,
    ROCM_RAND_UNIFORM = 2,
    ROCM_RAND_NORMAL = 3,
    ROCM_RAND_LOG_NORMAL = 4,
    ROCM_RAND_POISSON = 5,
    ROCM_RAND_DISCRETE = 6
} rocm_rand_distribution;

typedef enum rocm_rand_output_type {
    ROCM_RAND_U8 = 1,
    ROCM_RAND_U16 = 2,
    ROCM_RAND_U32 = 3,
    ROCM_RAND_U64 = 4,
    ROCM_RAND_F16 = 5,
    ROCM_RAND_F32 = 6,
    ROCM_RAND_F64 = 7
} rocm_rand_output_type;

typedef struct rocm_rand_generator_options {
    rocm_interfaces_abi_header header;
    const rocm_interfaces_host_services* host;
    rocm_rand_generator_kind kind;
    rocrand_rng_type algorithm;
    uint64_t seed[4];
    uint64_t offset;
    rocrand_ordering ordering;
    uint32_t dimensions;
} rocm_rand_generator_options;

typedef struct rocm_rand_generate_request {
    rocm_interfaces_abi_header header;
    rocm_interfaces_device_key device;
    void* stream;
    rocm_rand_distribution distribution;
    rocm_rand_output_type output_type;
    void* output;
    size_t count;
    double parameter_a;
    double parameter_b;
    void* distribution_token;
} rocm_rand_generate_request;

typedef struct rocm_rand_provider_v1 {
    rocm_interfaces_abi_header header;
    rocrand_status(ROCM_INTERFACES_CALL* create_generator)(const rocm_rand_generator_options*,
                                                           void**);
    void(ROCM_INTERFACES_CALL* destroy_generator)(void*);
    rocrand_status(ROCM_INTERFACES_CALL* configure_generator)(void*,
                                                              const rocm_rand_generator_options*);
    rocrand_status(ROCM_INTERFACES_CALL* generate)(void*, const rocm_rand_generate_request*);
} rocm_rand_provider_v1;

#ifdef __cplusplus
}
#endif
#endif
