// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/backends/tiled.hpp>

int main() {
    roc::host_validation::TiledGemmBackend backend;
    return backend.backend() == roc::host_validation::GemmBackend::Tiled ? 0 : 1;
}
