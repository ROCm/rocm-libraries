// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/backends/blas.hpp>

int main() {
    roc::host_validation::BlasGemmBackend backend;
    roc::host_validation::TransformingBlasGemmBackend transforming;
    return backend.backend() == roc::host_validation::GemmBackend::Blas &&
                   transforming.backend() == roc::host_validation::GemmBackend::Blas
               ? 0
               : 1;
}
