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
}  // namespace

std::shared_ptr<BlasContext> BlasContext::create(std::shared_ptr<ProviderRegistry> registry,
                                                 rocm_interfaces_device_key device) {
    if (!registry) throw std::invalid_argument("BLAS provider registry is null");
    auto lease = registry->select(ROCM_INTERFACES_DOMAIN_BLAS, device.gfx_arch,
                                  sizeof(rocm_blas_provider_v1));
    auto* table = static_cast<const rocm_blas_provider_v1*>(lease->table());
    if (!table->create_context || !table->destroy_context || !table->vector_execute ||
        !table->matmul_execute) {
        throw std::runtime_error("BLAS provider has a null required entry");
    }
    rocm_blas_context_options options{};
    options.header = header(sizeof(options));
    options.host = &registry->host_services();
    options.device = device;
    if (!options.device.header.struct_size) options.device.header = header(sizeof(device));
    void* provider_context = nullptr;
    rocblas_status status = table->create_context(&options, &provider_context);
    if (status != rocblas_status_success || !provider_context) {
        throw std::runtime_error("BLAS provider context creation failed");
    }
    return std::shared_ptr<BlasContext>(
        new BlasContext(std::move(registry), std::move(lease), table, provider_context, options));
}

BlasContext::BlasContext(std::shared_ptr<ProviderRegistry> registry,
                         std::shared_ptr<const ProviderLease> lease,
                         const rocm_blas_provider_v1* table, void* provider_context,
                         rocm_blas_context_options options)
    : registry_(std::move(registry)),
      lease_(std::move(lease)),
      table_(table),
      provider_context_(provider_context),
      options_(options) {}

BlasContext::~BlasContext() {
    table_->destroy_context(provider_context_);
}

std::unique_ptr<BlasLtContext> BlasLtContext::create(std::shared_ptr<ProviderRegistry> registry,
                                                     rocm_interfaces_device_key device) {
    if (!registry) throw std::invalid_argument("BLASLt provider registry is null");
    auto lease = registry->select(ROCM_INTERFACES_DOMAIN_BLASLT, device.gfx_arch,
                                  sizeof(rocm_blaslt_provider_v1));
    auto* lt = static_cast<const rocm_blaslt_provider_v1*>(lease->table());
    if (!lt || !lt->create_context || !lt->destroy_context || !lt->heuristic || !lt->matmul) {
        throw std::runtime_error("BLASLt provider has a null required entry");
    }
    rocm_blas_context_options options{};
    options.header = header(sizeof(options));
    options.host = &registry->host_services();
    options.device = device;
    if (!options.device.header.struct_size) options.device.header = header(sizeof(device));
    void* provider_context = nullptr;
    rocblas_status status = lt->create_context(&options, &provider_context);
    if (status != rocblas_status_success || !provider_context) {
        throw std::runtime_error("BLASLt provider context creation failed");
    }
    return std::unique_ptr<BlasLtContext>(
        new BlasLtContext(std::move(registry), std::move(lease), lt, provider_context));
}

BlasLtContext::BlasLtContext(std::shared_ptr<ProviderRegistry> registry,
                             std::shared_ptr<const ProviderLease> lease,
                             const rocm_blaslt_provider_v1* lt, void* provider_context)
    : registry_(std::move(registry)),
      lease_(std::move(lease)),
      lt_(lt),
      provider_context_(provider_context) {}

BlasLtContext::~BlasLtContext() {
    lt_->destroy_context(provider_context_);
}

rocblas_status BlasLtContext::heuristic(const rocm_blas_matmul_request& request,
                                        rocm_blaslt_heuristic_result* results, size_t capacity,
                                        size_t* count) {
    if (!count || (capacity && !results)) return rocblas_status_invalid_pointer;
    return lt_->heuristic(provider_context_, &request, results, capacity, count);
}

rocblas_status BlasLtContext::matmul(const rocm_blas_matmul_request& request) {
    return lt_->matmul(provider_context_, &request);
}

void BlasContext::set_stream(void* stream) {
    std::lock_guard lock(mutex_);
    options_.stream = stream;
}

void* BlasContext::stream() const {
    std::lock_guard lock(mutex_);
    return options_.stream;
}

void BlasContext::set_pointer_mode(uint32_t mode) {
    std::lock_guard lock(mutex_);
    options_.pointer_mode = mode;
}

uint32_t BlasContext::pointer_mode() const {
    std::lock_guard lock(mutex_);
    return options_.pointer_mode;
}

rocblas_status BlasContext::vector_execute(rocm_blas_vector_request request) {
    request.header = header(sizeof(request));
    return table_->vector_execute(provider_context_, &request);
}

rocblas_status BlasContext::matmul_execute(rocm_blas_matmul_request request) {
    request.header = header(sizeof(request));
    return table_->matmul_execute(provider_context_, &request);
}

}  // namespace rocm::interfaces
