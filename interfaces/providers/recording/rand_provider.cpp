// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <new>

#include "recording.h"
#include "rocm/interfaces/rand.h"

namespace {
struct Generator {
    rocm_rand_generator_options options;
};

rocrand_status create_generator(const rocm_rand_generator_options* options, void** result) {
    if (!options || !result) return ROCRAND_STATUS_OUT_OF_RANGE;
    auto* generator = new (std::nothrow) Generator{*options};
    if (!generator) return ROCRAND_STATUS_ALLOCATION_FAILED;
    *result = generator;
    rocm::interfaces::recording::trace(options->host, "rand", "create_generator", options,
                                       sizeof(*options));
    return ROCRAND_STATUS_SUCCESS;
}

void destroy_generator(void* opaque) {
    auto* generator = static_cast<Generator*>(opaque);
    rocm::interfaces::recording::trace(generator->options.host, "rand", "destroy_generator",
                                       nullptr, 0);
    delete generator;
}

rocrand_status configure_generator(void* opaque, const rocm_rand_generator_options* options) {
    if (!opaque) return ROCRAND_STATUS_NOT_CREATED;
    if (!options) return ROCRAND_STATUS_OUT_OF_RANGE;
    static_cast<Generator*>(opaque)->options = *options;
    rocm::interfaces::recording::trace(options->host, "rand", "configure_generator", options,
                                       sizeof(*options));
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status generate(void* opaque, const rocm_rand_generate_request* request) {
    if (!opaque) return ROCRAND_STATUS_NOT_CREATED;
    if (!request || !request->output) return ROCRAND_STATUS_OUT_OF_RANGE;
    auto* generator = static_cast<Generator*>(opaque);
    rocm::interfaces::recording::trace(generator->options.host, "rand", "generate", request,
                                       sizeof(*request));
    return ROCRAND_STATUS_SUCCESS;
}

const rocm_rand_provider_v1 table = {
    rocm::interfaces::recording::header(sizeof(rocm_rand_provider_v1)),
    create_generator,
    destroy_generator,
    configure_generator,
    generate,
};
}  // namespace

extern "C" ROCM_INTERFACES_EXPORT rocm_interfaces_status ROCM_INTERFACES_CALL
rocm_interfaces_provider_query_v1(const rocm_interfaces_provider_request* request,
                                  rocm_interfaces_provider_response* response) {
    return rocm::interfaces::recording::query(request, response, ROCM_INTERFACES_DOMAIN_RAND,
                                              "recording-rand", &table);
}
