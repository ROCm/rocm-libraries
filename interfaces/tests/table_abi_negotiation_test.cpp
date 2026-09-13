// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "rocm/interfaces/common.h"
#include "rocm/interfaces/runtime/provider_registry.h"

namespace {

using rocm::interfaces::ProviderRegistry;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct ProviderShape {
    uint16_t abi_minor;
    uint32_t table_size;
};

ProviderShape g_shape{};
const unsigned char g_dispatch_bytes[256] = {0};

rocm_interfaces_status shaped_query(const rocm_interfaces_provider_request* request,
                                    rocm_interfaces_provider_response* response) {
    if (!request || !response || request->domain != ROCM_INTERFACES_DOMAIN_BLAS)
        return ROCM_INTERFACES_STATUS_INCOMPATIBLE_ABI;
    response->header = {static_cast<uint32_t>(sizeof(*response)), ROCM_INTERFACES_ABI_MAJOR,
                        g_shape.abi_minor};
    response->provider_id = "table-abi-probe";
    response->build_id = "table-abi";
    response->dispatch_table = g_dispatch_bytes;
    response->dispatch_table_size = g_shape.table_size;
    response->capability_mask = 0;
    return ROCM_INTERFACES_STATUS_SUCCESS;
}

bool provider_accepted(uint16_t abi_minor, uint32_t table_size, uint32_t required_table_size) {
    g_shape = {abi_minor, table_size};
    ProviderRegistry registry;
    registry.add_builtin(ROCM_INTERFACES_DOMAIN_BLAS, 0, 0, "table-abi-probe", shaped_query);
    try {
        return registry.select(ROCM_INTERFACES_DOMAIN_BLAS, 942, required_table_size) != nullptr;
    } catch (const std::exception&) {
        return false;
    }
}

void test_table_abi_negotiation() {
    const uint16_t runtime_minor = ROCM_INTERFACES_ABI_MINOR;
    require(runtime_minor >= 1,
            "runtime ABI minor must be >= 1 for the old-provider case to be reachable");
    const uint32_t required = 64;

    require(provider_accepted(runtime_minor, required, required),
            "exact minor and exact table size must be accepted");
    require(provider_accepted(static_cast<uint16_t>(runtime_minor + 1), required * 2, required),
            "newer minor with a larger table (optional tail) must be accepted");
    require(!provider_accepted(runtime_minor, required / 2, required),
            "a table short of the required prefix must be rejected");
    require(!provider_accepted(static_cast<uint16_t>(runtime_minor - 1), required * 2, required),
            "a provider older than the runtime minor must be rejected");

    const bool ample_new_minor =
        provider_accepted(static_cast<uint16_t>(runtime_minor + 1), required * 2, required);
    const bool ample_old_minor =
        provider_accepted(static_cast<uint16_t>(runtime_minor - 1), required * 2, required);
    require(ample_new_minor && !ample_old_minor, "accept/reject must toggle on abi_minor alone");

    const bool ample_table = provider_accepted(runtime_minor, required, required);
    const bool short_table = provider_accepted(runtime_minor, required - 1, required);
    require(ample_table && !short_table, "accept/reject must toggle on dispatch_table_size alone");
}

}  // namespace

int main() {
    try {
        test_table_abi_negotiation();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "table-ABI negotiation: prefix + abi_minor floor enforced, discriminating\n";
    return EXIT_SUCCESS;
}
