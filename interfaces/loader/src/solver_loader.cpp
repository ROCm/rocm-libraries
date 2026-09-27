// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <stdexcept>
#include <utility>

#include "rocm/interfaces/loader.h"

namespace rocm::interfaces {
namespace {
rocm_interfaces_abi_header header(size_t size) {
    return {static_cast<uint32_t>(size), ROCM_INTERFACES_ABI_MAJOR, ROCM_INTERFACES_ABI_MINOR};
}

rocblas_status ROCM_INTERFACES_CALL vector_execute(void* opaque,
                                                   const rocm_blas_vector_request* request) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request) return rocblas_status_invalid_pointer;
    try {
        return static_cast<BlasContext*>(opaque)->vector_execute(*request);
    } catch (...) {
        return rocblas_status_internal_error;
    }
}

rocblas_status ROCM_INTERFACES_CALL matmul_execute(void* opaque,
                                                   const rocm_blas_matmul_request* request) {
    if (!opaque) return rocblas_status_invalid_handle;
    if (!request) return rocblas_status_invalid_pointer;
    try {
        return static_cast<BlasContext*>(opaque)->matmul_execute(*request);
    } catch (...) {
        return rocblas_status_internal_error;
    }
}
}  // namespace

std::unique_ptr<SolverContext> SolverContext::create(std::shared_ptr<BlasContext> blas) {
    if (!blas) throw std::invalid_argument("borrowed BLAS context is null");
    auto lease =
        blas->registry_->select(ROCM_INTERFACES_DOMAIN_SOLVER, blas->options_.device.gfx_arch,
                                sizeof(rocm_solver_provider_v1), blas->lease_->cohort_id());
    auto* table = static_cast<const rocm_solver_provider_v1*>(lease->table());
    if (!table->create_context || !table->destroy_context || !table->query_workspace ||
        !table->execute) {
        throw std::runtime_error("solver provider has a null required entry");
    }
    rocm_solver_context_options options{};
    options.header = header(sizeof(options));
    options.host = &blas->registry_->host_services();
    options.device = blas->options_.device;
    options.stream = blas->stream();
    options.blas.header = header(sizeof(options.blas));
    options.blas.context = blas.get();
    options.blas.vector_execute = vector_execute;
    options.blas.matmul_execute = matmul_execute;
    void* provider_context = nullptr;
    rocblas_status status = table->create_context(&options, &provider_context);
    if (status != rocblas_status_success || !provider_context) {
        throw std::runtime_error("solver provider context creation failed");
    }
    return std::unique_ptr<SolverContext>(
        new SolverContext(std::move(blas), std::move(lease), table, provider_context));
}

SolverContext::SolverContext(std::shared_ptr<BlasContext> blas,
                             std::shared_ptr<const ProviderLease> lease,
                             const rocm_solver_provider_v1* table, void* provider_context)
    : blas_(std::move(blas)),
      lease_(std::move(lease)),
      table_(table),
      provider_context_(provider_context) {}

SolverContext::~SolverContext() {
    table_->destroy_context(provider_context_);
}

rocblas_status SolverContext::query_workspace(const rocm_solver_request& request,
                                              size_t* workspace_size) {
    if (!workspace_size) return rocblas_status_invalid_pointer;
    return table_->query_workspace(provider_context_, &request, workspace_size);
}

rocblas_status SolverContext::execute(const rocm_solver_request& request) {
    return table_->execute(provider_context_, &request);
}

}  // namespace rocm::interfaces
