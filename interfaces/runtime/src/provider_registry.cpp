// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm/interfaces/runtime/provider_registry.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace rocm::interfaces {
namespace {

rocm_interfaces_abi_header abi_header(size_t size) {
    return {static_cast<uint32_t>(size), ROCM_INTERFACES_ABI_MAJOR, ROCM_INTERFACES_ABI_MINOR};
}

bool supports_gfx(uint32_t configured, uint32_t requested) {
    return configured == 0 || configured == requested;
}

rocm_interfaces_domain parse_domain(const std::string& value) {
    if (value == "blas") return ROCM_INTERFACES_DOMAIN_BLAS;
    if (value == "blaslt") return ROCM_INTERFACES_DOMAIN_BLASLT;
    if (value == "solver") return ROCM_INTERFACES_DOMAIN_SOLVER;
    if (value == "rand") return ROCM_INTERFACES_DOMAIN_RAND;
    if (value == "rocblas_bridge") return ROCM_INTERFACES_DOMAIN_ROCBLAS_BRIDGE;
    if (value == "blas_v2") return ROCM_INTERFACES_DOMAIN_BLAS_V2;
    throw std::invalid_argument("unknown provider domain: " + value);
}

}  // namespace

ProviderRegistry::ProviderRegistry(rocm_interfaces_host_services host_services)
    : host_(host_services) {
    if (host_.header.struct_size == 0) host_.header = abi_header(sizeof(host_));
}

void ProviderRegistry::load_manifest(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open provider manifest: " + path.string());
    nlohmann::json document;
    stream >> document;
    if (!document.is_object() || document.value("schema_version", 0) != 1 ||
        !document.contains("providers") || !document["providers"].is_array()) {
        throw std::invalid_argument("invalid provider manifest schema: " + path.string());
    }
    const std::filesystem::path base = std::filesystem::weakly_canonical(path).parent_path();
    for (const nlohmann::json& item : document["providers"]) {
        if (!item.is_object()) throw std::invalid_argument("provider entry must be an object");
        const std::string id = item.at("id").get<std::string>();
        const rocm_interfaces_domain domain = parse_domain(item.at("domain").get<std::string>());
        const std::filesystem::path relative = item.at("module").get<std::string>();
        if (id.empty() || relative.empty() || relative.is_absolute()) {
            throw std::invalid_argument("provider id and relative module path are required");
        }
        const std::filesystem::path module = std::filesystem::weakly_canonical(base / relative);
        const auto mismatch = std::mismatch(base.begin(), base.end(), module.begin(), module.end());
        if (mismatch.first != base.end()) {
            throw std::invalid_argument("provider module escapes manifest directory");
        }
        const int priority = item.value("priority", 0);
        const std::string cohort = item.value("cohort", std::string{});
        const std::string query_symbol =
            item.value("query_symbol", std::string(ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL));
        const nlohmann::json gfx = item.value("gfx", nlohmann::json::array({0}));
        if (!gfx.is_array() || gfx.empty())
            throw std::invalid_argument("gfx must be a nonempty array");
        for (const nlohmann::json& architecture : gfx) {
            if (!architecture.is_number_unsigned() && !architecture.is_number_integer()) {
                throw std::invalid_argument("gfx entries must be numeric; zero means wildcard");
            }
            std::lock_guard lock(mutex_);
            entries_.push_back({domain, architecture.get<uint32_t>(), priority, id, cohort, module,
                                query_symbol, nullptr, nullptr});
        }
    }
}

void ProviderRegistry::add_module(rocm_interfaces_domain domain, uint32_t gfx_arch, int priority,
                                  const std::filesystem::path& path, std::string query_symbol,
                                  std::string cohort_id) {
    if (path.empty()) throw std::invalid_argument("provider module path is empty");
    std::lock_guard lock(mutex_);
    entries_.push_back({domain,
                        gfx_arch,
                        priority,
                        {},
                        std::move(cohort_id),
                        path,
                        std::move(query_symbol),
                        nullptr,
                        nullptr});
}

void ProviderRegistry::add_builtin(rocm_interfaces_domain domain, uint32_t gfx_arch, int priority,
                                   std::string provider_id, rocm_interfaces_provider_query_fn query,
                                   std::string cohort_id) {
    if (!query) throw std::invalid_argument("builtin provider query is null");
    std::lock_guard lock(mutex_);
    entries_.push_back({domain,
                        gfx_arch,
                        priority,
                        std::move(provider_id),
                        std::move(cohort_id),
                        {},
                        {},
                        query,
                        nullptr});
}

std::shared_ptr<const ProviderLease> ProviderRegistry::query_entry(Entry& entry,
                                                                   uint32_t required_table_size) {
    if (!entry.query) {
        entry.module = Module::open(entry.path);
        entry.query = reinterpret_cast<rocm_interfaces_provider_query_fn>(
            entry.module->symbol(entry.query_symbol.c_str()));
    }

    rocm_interfaces_provider_request request{};
    request.header = abi_header(sizeof(request));
    request.domain = entry.domain;
    request.required_table_size = required_table_size;
    request.host = &host_;

    auto lease = std::make_shared<ProviderLease>();
    lease->response_.header = abi_header(sizeof(lease->response_));
    rocm_interfaces_status status = entry.query(&request, &lease->response_);
    if (status != ROCM_INTERFACES_STATUS_SUCCESS) return nullptr;
    if (lease->response_.header.abi_major != ROCM_INTERFACES_ABI_MAJOR ||
        lease->response_.header.abi_minor < ROCM_INTERFACES_ABI_MINOR ||
        lease->response_.header.struct_size < sizeof(rocm_interfaces_provider_response) ||
        !lease->response_.dispatch_table ||
        lease->response_.dispatch_table_size < required_table_size ||
        !lease->response_.provider_id || !*lease->response_.provider_id) {
        return nullptr;
    }
    if (!entry.provider_id.empty() && entry.provider_id != lease->response_.provider_id) {
        return nullptr;
    }
    lease->provider_id_ = lease->response_.provider_id;
    lease->cohort_id_ = entry.cohort_id;
    lease->module_ = entry.module;
    leases_.push_back(lease);
    return lease;
}

std::shared_ptr<const ProviderLease> ProviderRegistry::select(rocm_interfaces_domain domain,
                                                              uint32_t gfx_arch,
                                                              uint32_t required_table_size,
                                                              const std::string& required_cohort) {
    std::lock_guard lock(mutex_);
    std::vector<Entry*> candidates;
    for (Entry& entry : entries_) {
        if (entry.domain == domain && supports_gfx(entry.gfx_arch, gfx_arch) &&
            (required_cohort.empty() || entry.cohort_id == required_cohort)) {
            candidates.push_back(&entry);
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Entry* a, const Entry* b) {
        const bool a_exact = a->gfx_arch != 0;
        const bool b_exact = b->gfx_arch != 0;
        if (a_exact != b_exact) return a_exact > b_exact;
        return a->priority > b->priority;
    });
    for (Entry* entry : candidates) {
        if (auto lease = query_entry(*entry, required_table_size)) return lease;
    }
    throw std::runtime_error("no compatible provider for requested domain and gfx");
}

}  // namespace rocm::interfaces
