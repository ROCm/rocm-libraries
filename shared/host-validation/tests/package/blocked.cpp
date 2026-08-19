// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/backends/blocked.hpp>

int main() {
    roc::host_validation::BlockedGemmBackend backend;
    return backend.backend() == roc::host_validation::GemmBackend::Blocked ? 0 : 1;
}
