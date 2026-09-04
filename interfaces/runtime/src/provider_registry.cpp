// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm/interfaces/runtime/provider_registry.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <tuple>
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

void require_object_shape(const nlohmann::json& value, std::initializer_list<const char*> required,
                          std::initializer_list<const char*> allowed, const std::string& context) {
    if (!value.is_object()) throw std::invalid_argument(context + " must be an object");
    std::set<std::string> allowed_keys;
    for (const char* key : allowed) allowed_keys.emplace(key);
    for (auto iterator = value.begin(); iterator != value.end(); ++iterator) {
        if (!allowed_keys.contains(iterator.key()))
            throw std::invalid_argument(context + " has unknown key: " + iterator.key());
    }
    for (const char* key : required) {
        if (!value.contains(key)) throw std::invalid_argument(context + " is missing key: " + key);
    }
}

std::string required_string(const nlohmann::json& value, const char* key,
                            const std::string& context) {
    const auto& field = value.at(key);
    if (!field.is_string() || field.get_ref<const std::string&>().empty())
        throw std::invalid_argument(context + "." + key + " must be a nonempty string");
    return field.get<std::string>();
}

bool is_within(const std::filesystem::path& base, const std::filesystem::path& candidate) {
    const std::filesystem::path relative = candidate.lexically_relative(base);
    return !relative.empty() && !relative.is_absolute() && *relative.begin() != "..";
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
    try {
        stream >> document;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("invalid provider manifest JSON " + path.string() + ": " +
                                    error.what());
    }
    require_object_shape(document, {"schema_version", "providers"}, {"schema_version", "providers"},
                         "provider manifest");
    if (!document["schema_version"].is_number_integer() ||
        document["schema_version"].get<int>() != 1 || !document["providers"].is_array() ||
        document["providers"].empty())
        throw std::invalid_argument("invalid provider manifest schema: " + path.string());

    const std::filesystem::path base = std::filesystem::weakly_canonical(path).parent_path();
    std::vector<Entry> parsed;
    std::set<std::tuple<rocm_interfaces_domain, uint32_t, std::string>> identities;
    for (const nlohmann::json& item : document["providers"]) {
        require_object_shape(
            item, {"id", "domain", "module"},
            {"id", "domain", "module", "cohort", "query_symbol", "priority", "gfx"},
            "provider entry");
        const std::string id = required_string(item, "id", "provider entry");
        const rocm_interfaces_domain domain =
            parse_domain(required_string(item, "domain", "provider entry"));
        const std::filesystem::path relative = required_string(item, "module", "provider entry");
        if (relative.is_absolute())
            throw std::invalid_argument("provider module path must be relative");
        const std::filesystem::path module = std::filesystem::weakly_canonical(base / relative);
        if (!is_within(base, module)) {
            throw std::invalid_argument("provider module escapes manifest directory");
        }
        if (!std::filesystem::is_regular_file(module))
            throw std::invalid_argument("provider module is not a regular file: " +
                                        module.string());

        int priority = 0;
        if (item.contains("priority")) {
            if (!item["priority"].is_number_integer())
                throw std::invalid_argument("provider priority must be an integer");
            const int64_t raw = item["priority"].get<int64_t>();
            if (raw < std::numeric_limits<int>::min() || raw > std::numeric_limits<int>::max())
                throw std::invalid_argument("provider priority is out of range");
            priority = static_cast<int>(raw);
        }
        std::string cohort;
        if (item.contains("cohort")) {
            if (!item["cohort"].is_string())
                throw std::invalid_argument("provider cohort must be a string");
            cohort = item["cohort"].get<std::string>();
        }
        std::string query_symbol = ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL;
        if (item.contains("query_symbol"))
            query_symbol = required_string(item, "query_symbol", "provider entry");
        const nlohmann::json gfx = item.value("gfx", nlohmann::json::array({0}));
        if (!gfx.is_array() || gfx.empty())
            throw std::invalid_argument("gfx must be a nonempty array");
        for (const nlohmann::json& architecture : gfx) {
            if (!architecture.is_number_integer()) {
                throw std::invalid_argument("gfx entries must be numeric; zero means wildcard");
            }
            const int64_t raw = architecture.get<int64_t>();
            if (raw < 0 || static_cast<uint64_t>(raw) > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument("gfx entry is out of range");
            const uint32_t arch = static_cast<uint32_t>(raw);
            if (!identities.emplace(domain, arch, id).second)
                throw std::invalid_argument("duplicate provider id/domain/gfx entry: " + id);
            parsed.push_back(
                {domain, arch, priority, id, cohort, module, query_symbol, nullptr, nullptr});
        }
    }
    std::lock_guard lock(mutex_);
    entries_.insert(entries_.end(), std::make_move_iterator(parsed.begin()),
                    std::make_move_iterator(parsed.end()));
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
