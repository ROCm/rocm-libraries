// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCM_INTERFACES_RUNTIME_PROVIDER_REGISTRY_H_
#define ROCM_INTERFACES_RUNTIME_PROVIDER_REGISTRY_H_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rocm/interfaces/common.h"
#include "rocm/interfaces/runtime/module.h"

namespace rocm::interfaces {

class ProviderLease {
   public:
    const std::string& provider_id() const noexcept {
        return provider_id_;
    }
    const std::string& cohort_id() const noexcept {
        return cohort_id_;
    }
    const void* table() const noexcept {
        return response_.dispatch_table;
    }
    uint32_t table_size() const noexcept {
        return response_.dispatch_table_size;
    }

   private:
    friend class ProviderRegistry;
    std::string provider_id_;
    std::string cohort_id_;
    std::shared_ptr<Module> module_;
    rocm_interfaces_provider_response response_{};
};

class ProviderRegistry {
   public:
    explicit ProviderRegistry(rocm_interfaces_host_services host_services = {});

    void load_manifest(const std::filesystem::path& path);

    void add_module(rocm_interfaces_domain domain, uint32_t gfx_arch, int priority,
                    const std::filesystem::path& path,
                    std::string query_symbol = ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL,
                    std::string cohort_id = {});
    void add_builtin(rocm_interfaces_domain domain, uint32_t gfx_arch, int priority,
                     std::string provider_id, rocm_interfaces_provider_query_fn query,
                     std::string cohort_id = {});

    std::shared_ptr<const ProviderLease> select(rocm_interfaces_domain domain, uint32_t gfx_arch,
                                                uint32_t required_table_size,
                                                const std::string& required_cohort = {});
    const rocm_interfaces_host_services& host_services() const noexcept {
        return host_;
    }

   private:
    struct Entry {
        rocm_interfaces_domain domain;
        uint32_t gfx_arch;
        int priority;
        std::string provider_id;
        std::string cohort_id;
        std::filesystem::path path;
        std::string query_symbol;
        rocm_interfaces_provider_query_fn query = nullptr;
        std::shared_ptr<Module> module;
    };

    std::shared_ptr<const ProviderLease> query_entry(Entry& entry, uint32_t required_table_size);

    rocm_interfaces_host_services host_{};
    std::mutex mutex_;
    std::vector<Entry> entries_;
    std::vector<std::shared_ptr<const ProviderLease>> leases_;
};

}  // namespace rocm::interfaces
#endif
