// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCM_INTERFACES_LOADER_H_
#define ROCM_INTERFACES_LOADER_H_

#include <memory>
#include <mutex>

#include "rocm/interfaces/blas.h"
#include "rocm/interfaces/rand.h"
#include "rocm/interfaces/runtime/provider_registry.h"
#include "rocm/interfaces/solver.h"

namespace rocm::interfaces {

class SolverContext;

class BlasLtContext {
   public:
    static std::unique_ptr<BlasLtContext> create(std::shared_ptr<ProviderRegistry> registry,
                                                 rocm_interfaces_device_key device);
    ~BlasLtContext();

    BlasLtContext(const BlasLtContext&) = delete;
    BlasLtContext& operator=(const BlasLtContext&) = delete;

    rocblas_status heuristic(const rocm_blas_matmul_request& request,
                             rocm_blaslt_heuristic_result* results, size_t capacity, size_t* count);
    rocblas_status matmul(const rocm_blas_matmul_request& request);
    const std::string& provider_id() const noexcept {
        return lease_->provider_id();
    }
    const std::string& cohort_id() const noexcept {
        return lease_->cohort_id();
    }

   private:
    BlasLtContext(std::shared_ptr<ProviderRegistry> registry,
                  std::shared_ptr<const ProviderLease> lease, const rocm_blaslt_provider_v1* lt,
                  void* provider_context);
    std::shared_ptr<ProviderRegistry> registry_;
    std::shared_ptr<const ProviderLease> lease_;
    const rocm_blaslt_provider_v1* lt_;
    void* provider_context_;
};

class BlasContext : public std::enable_shared_from_this<BlasContext> {
   public:
    static std::shared_ptr<BlasContext> create(std::shared_ptr<ProviderRegistry> registry,
                                               rocm_interfaces_device_key device);
    ~BlasContext();

    BlasContext(const BlasContext&) = delete;
    BlasContext& operator=(const BlasContext&) = delete;

    void set_stream(void* stream);
    void* stream() const;
    void set_pointer_mode(uint32_t mode);
    uint32_t pointer_mode() const;
    rocblas_status vector_execute(rocm_blas_vector_request request);
    rocblas_status matmul_execute(rocm_blas_matmul_request request);
    const std::string& provider_id() const noexcept {
        return lease_->provider_id();
    }
    const std::string& cohort_id() const noexcept {
        return lease_->cohort_id();
    }

   private:
    friend class SolverContext;
    BlasContext(std::shared_ptr<ProviderRegistry> registry,
                std::shared_ptr<const ProviderLease> lease, const rocm_blas_provider_v1* table,
                void* provider_context, rocm_blas_context_options options);

    std::shared_ptr<ProviderRegistry> registry_;
    std::shared_ptr<const ProviderLease> lease_;
    const rocm_blas_provider_v1* table_;
    void* provider_context_;
    mutable std::mutex mutex_;
    rocm_blas_context_options options_{};
};

class SolverContext {
   public:
    static std::unique_ptr<SolverContext> create(std::shared_ptr<BlasContext> blas);
    ~SolverContext();

    SolverContext(const SolverContext&) = delete;
    SolverContext& operator=(const SolverContext&) = delete;

    rocblas_status query_workspace(const rocm_solver_request& request, size_t* workspace_size);
    rocblas_status execute(const rocm_solver_request& request);
    const std::string& provider_id() const noexcept {
        return lease_->provider_id();
    }

   private:
    SolverContext(std::shared_ptr<BlasContext> blas, std::shared_ptr<const ProviderLease> lease,
                  const rocm_solver_provider_v1* table, void* provider_context);

    std::shared_ptr<BlasContext> blas_;
    std::shared_ptr<const ProviderLease> lease_;
    const rocm_solver_provider_v1* table_;
    void* provider_context_;
};

class RandGenerator {
   public:
    static std::unique_ptr<RandGenerator> create(std::shared_ptr<ProviderRegistry> registry,
                                                 rocm_rand_generator_options options);
    ~RandGenerator();

    RandGenerator(const RandGenerator&) = delete;
    RandGenerator& operator=(const RandGenerator&) = delete;

    rocrand_status configure(const rocm_rand_generator_options& options);
    rocrand_status generate(const rocm_rand_generate_request& request);
    const std::string& provider_id() const;

   private:
    RandGenerator(std::shared_ptr<ProviderRegistry> registry, rocm_rand_generator_options options);
    rocrand_status bind(uint32_t gfx_arch);

    std::shared_ptr<ProviderRegistry> registry_;
    std::shared_ptr<const ProviderLease> lease_;
    const rocm_rand_provider_v1* table_ = nullptr;
    void* provider_generator_ = nullptr;
    rocm_rand_generator_options options_{};
    std::mutex mutex_;
};

}  // namespace rocm::interfaces
#endif
