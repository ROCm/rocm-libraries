// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <stdexcept>
#include <utility>

#include "rocm/interfaces/loader.h"

namespace rocm::interfaces {

std::unique_ptr<RandGenerator> RandGenerator::create(std::shared_ptr<ProviderRegistry> registry,
                                                     rocm_rand_generator_options options) {
    if (!registry) throw std::invalid_argument("RAND provider registry is null");
    return std::unique_ptr<RandGenerator>(new RandGenerator(std::move(registry), options));
}

RandGenerator::RandGenerator(std::shared_ptr<ProviderRegistry> registry,
                             rocm_rand_generator_options options)
    : registry_(std::move(registry)), options_(options) {
    options_.host = &registry_->host_services();
}

RandGenerator::~RandGenerator() {
    if (table_ && provider_generator_) table_->destroy_generator(provider_generator_);
}

rocrand_status RandGenerator::bind(uint32_t gfx_arch) {
    if (lease_) return ROCRAND_STATUS_SUCCESS;
    lease_ =
        registry_->select(ROCM_INTERFACES_DOMAIN_RAND, gfx_arch, sizeof(rocm_rand_provider_v1));
    table_ = static_cast<const rocm_rand_provider_v1*>(lease_->table());
    if (!table_->create_generator || !table_->destroy_generator || !table_->configure_generator ||
        !table_->generate) {
        return ROCRAND_STATUS_VERSION_MISMATCH;
    }
    return table_->create_generator(&options_, &provider_generator_);
}

rocrand_status RandGenerator::configure(const rocm_rand_generator_options& options) {
    std::lock_guard lock(mutex_);
    options_ = options;
    options_.host = &registry_->host_services();
    if (!table_) return ROCRAND_STATUS_SUCCESS;
    return table_->configure_generator(provider_generator_, &options_);
}

rocrand_status RandGenerator::generate(const rocm_rand_generate_request& request) {
    std::lock_guard lock(mutex_);
    rocrand_status status = bind(request.device.gfx_arch);
    if (status != ROCRAND_STATUS_SUCCESS) return status;
    return table_->generate(provider_generator_, &request);
}

const std::string& RandGenerator::provider_id() const {
    if (!lease_) throw std::logic_error("RAND generator has not selected a provider");
    return lease_->provider_id();
}

}  // namespace rocm::interfaces
