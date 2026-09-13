// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "rocblas_bridge_generated.h"

namespace {

std::string trace_message;

void trace(void*, const char*, const char* operation, const void* payload, size_t size) {
    if (operation && std::string(operation) == "backend_load_failure" && payload)
        trace_message.assign(static_cast<const char*>(payload), size);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("expected provider and backend paths");
        setenv("ROCM_INTERFACES_REAL_ROCBLAS_LIBRARY", argv[2], 1);
        void* provider = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!provider) throw std::runtime_error(dlerror());
        auto query = reinterpret_cast<rocm_interfaces_provider_query_fn>(
            dlsym(provider, ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL));
        if (!query) throw std::runtime_error("provider query export is missing");

        rocm_interfaces_host_services host{};
        host.header = {sizeof(host), ROCM_INTERFACES_ABI_MAJOR, ROCM_INTERFACES_ABI_MINOR};
        host.trace = trace;
        rocm_interfaces_provider_request request{};
        request.header = {sizeof(request), ROCM_INTERFACES_ABI_MAJOR, ROCM_INTERFACES_ABI_MINOR};
        request.domain = ROCM_INTERFACES_DOMAIN_ROCBLAS_BRIDGE;
        request.required_table_size = sizeof(rocm_rocblas_bridge_v1);
        request.host = &host;
        rocm_interfaces_provider_response response{};
        response.header = {sizeof(response), ROCM_INTERFACES_ABI_MAJOR, ROCM_INTERFACES_ABI_MINOR};
        if (query(&request, &response) != ROCM_INTERFACES_STATUS_PROVIDER_FAILURE)
            throw std::runtime_error("bad backend did not fail provider negotiation");
        if (trace_message.empty())
            throw std::runtime_error("backend failure did not produce a host trace");
        dlclose(provider);
        std::cout << "real provider failed closed: " << trace_message << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
