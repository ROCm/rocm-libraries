// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCM_INTERFACES_RECORDING_PROVIDER_H_
#define ROCM_INTERFACES_RECORDING_PROVIDER_H_

#include <cstddef>
#include <cstdint>

#include "rocm/interfaces/common.h"

namespace rocm::interfaces::recording {

inline rocm_interfaces_abi_header header(size_t size) {
    return {static_cast<uint32_t>(size), ROCM_INTERFACES_ABI_MAJOR, ROCM_INTERFACES_ABI_MINOR};
}

inline void trace(const rocm_interfaces_host_services* host, const char* domain,
                  const char* operation, const void* payload, size_t payload_size) {
    if (host && host->trace) {
        host->trace(host->user_data, domain, operation, payload, payload_size);
    }
}

template <typename Table>
rocm_interfaces_status query(const rocm_interfaces_provider_request* request,
                             rocm_interfaces_provider_response* response,
                             rocm_interfaces_domain domain, const char* provider_id,
                             const Table* table) {
    if (!request || !response || request->domain != domain ||
        request->header.abi_major != ROCM_INTERFACES_ABI_MAJOR ||
        response->header.struct_size < sizeof(*response) ||
        request->required_table_size > sizeof(Table)) {
        return ROCM_INTERFACES_STATUS_INCOMPATIBLE_ABI;
    }
    response->header = header(sizeof(*response));
    response->provider_id = provider_id;
    response->build_id = "interfaces-spike-v1";
    response->dispatch_table = table;
    response->dispatch_table_size = sizeof(Table);
    response->capability_mask = 0;
    return ROCM_INTERFACES_STATUS_SUCCESS;
}

}  // namespace rocm::interfaces::recording
#endif
