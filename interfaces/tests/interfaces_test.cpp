// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rocm/interfaces/experimental/blas_narrow_v2.h"
#include "rocm/interfaces/loader.h"

namespace {
struct TraceLog {
    std::vector<std::string> operations;
};

void trace(void* opaque, const char* domain, const char* operation, const void*, size_t) {
    static_cast<TraceLog*>(opaque)->operations.emplace_back(std::string(domain) + "." + operation);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

rocm_interfaces_abi_header header(size_t size) {
    return {static_cast<uint32_t>(size), ROCM_INTERFACES_ABI_MAJOR, ROCM_INTERFACES_ABI_MINOR};
}

void test_public_enum_invariants() {
    static_assert(rocblas_status_success == 0);
    static_assert(rocblas_status_invalid_handle == 1);
    static_assert(rocblas_operation_none == 111);
    static_assert(rocblas_operation_transpose == 112);
    static_assert(rocblas_operation_conjugate_transpose == 113);
    static_assert(ROCRAND_STATUS_SUCCESS == 0);
    static_assert(ROCRAND_STATUS_VERSION_MISMATCH == 100);
    static_assert(ROCM_BLAS_V2_VECTOR_AXPY == 4);
    static_assert(sizeof(rocm_blas_v2_provider) >= 12 * sizeof(void*));
}

void test_end_to_end() {
    TraceLog log;
    rocm_interfaces_host_services services{};
    services.header = header(sizeof(services));
    services.user_data = &log;
    services.trace = trace;
    auto registry = std::make_shared<rocm::interfaces::ProviderRegistry>(services);
    registry->add_module(ROCM_INTERFACES_DOMAIN_BLAS, 0, 0, BLAS_PROVIDER_PATH,
                         ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL, "recording-blas-cohort");
    registry->add_module(ROCM_INTERFACES_DOMAIN_BLASLT, 0, 0, BLASLT_PROVIDER_PATH,
                         ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL, "recording-blas-cohort");
    registry->add_module(ROCM_INTERFACES_DOMAIN_SOLVER, 0, 0, SOLVER_PROVIDER_PATH,
                         ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL, "recording-blas-cohort");
    registry->add_module(ROCM_INTERFACES_DOMAIN_RAND, 0, 0, RAND_PROVIDER_PATH);

    rocm_interfaces_device_key device{};
    device.header = header(sizeof(device));
    device.device_ordinal = 0;
    device.gfx_arch = 942;
    auto blas = rocm::interfaces::BlasContext::create(registry, device);
    require(blas->provider_id() == "recording-blas-legacy", "wrong BLAS provider");
    blas->set_stream(reinterpret_cast<void*>(0x1234));
    require(blas->stream() == reinterpret_cast<void*>(0x1234), "stream state not edge-owned");

    float alpha = 1.0f;
    rocm_blas_vector_request vector{};
    vector.header = header(sizeof(vector));
    vector.opcode = ROCM_BLAS_VECTOR_AXPY;
    vector.index_width = 32;
    vector.batch_kind = ROCM_BLAS_BATCH_SINGLE;
    vector.batch_count = 1;
    vector.alpha = {header(sizeof(vector.alpha)), rocblas_datatype_f32_r, ROCM_BLAS_SCALAR_HOST,
                    &alpha};
    vector.x = {header(sizeof(vector.x)),
                rocblas_datatype_f32_r,
                reinterpret_cast<void*>(0x1000),
                16,
                1,
                0};
    vector.y = {header(sizeof(vector.y)),
                rocblas_datatype_f32_r,
                reinterpret_cast<void*>(0x2000),
                16,
                1,
                0};
    require(blas->vector_execute(vector) == rocblas_status_success, "BLAS vector request failed");

    auto blaslt = rocm::interfaces::BlasLtContext::create(registry, device);
    require(blaslt->provider_id() != blas->provider_id(),
            "legacy rocBLAS and hipBLASLt providers unexpectedly share a DSO");
    require(blaslt->cohort_id() == blas->cohort_id(),
            "separate legacy BLAS providers did not select the same cohort");
    rocm_blas_matmul_request lt_request{};
    lt_request.header = header(sizeof(lt_request));
    rocm_blaslt_heuristic_result heuristic{};
    size_t heuristic_count = 0;
    require(
        blaslt->heuristic(lt_request, &heuristic, 1, &heuristic_count) == rocblas_status_success &&
            heuristic_count == 1,
        "BLASLt table from combined provider failed");

    auto solver = rocm::interfaces::SolverContext::create(blas);
    rocm_solver_request solve{};
    solve.header = header(sizeof(solve));
    solve.operation = ROCM_SOLVER_GETRF;
    solve.data_type = rocblas_datatype_f32_r;
    solve.index_width = 32;
    solve.m = 8;
    solve.n = 8;
    size_t workspace_size = 0;
    require(solver->query_workspace(solve, &workspace_size) == rocblas_status_success,
            "solver workspace query failed");
    require(workspace_size == 256, "unexpected solver workspace size");
    require(solver->execute(solve) == rocblas_status_success, "solver execute failed");

    rocm_rand_generator_options random_options{};
    random_options.header = header(sizeof(random_options));
    random_options.kind = ROCM_RAND_GENERATOR_DEVICE;
    random_options.algorithm = ROCRAND_RNG_PSEUDO_DEFAULT;
    auto random = rocm::interfaces::RandGenerator::create(registry, random_options);
    require(log.operations.end() ==
                std::find(log.operations.begin(), log.operations.end(), "rand.create_generator"),
            "RAND provider selected before first device operation");
    uint32_t fake_output = 0;
    rocm_rand_generate_request generation{};
    generation.header = header(sizeof(generation));
    generation.device = device;
    generation.distribution = ROCM_RAND_RAW;
    generation.output_type = ROCM_RAND_U32;
    generation.output = &fake_output;
    generation.count = 1;
    require(random->generate(generation) == ROCRAND_STATUS_SUCCESS, "RAND generation failed");
    require(random->provider_id() == "recording-rand", "wrong RAND provider");

    require(std::find(log.operations.begin(), log.operations.end(), "blas.vector_execute") !=
                log.operations.end(),
            "BLAS request was not recorded");
    require(std::find(log.operations.begin(), log.operations.end(), "solver.execute") !=
                log.operations.end(),
            "solver request was not recorded");
    require(std::find(log.operations.begin(), log.operations.end(), "rand.generate") !=
                log.operations.end(),
            "RAND request was not recorded");
}

void test_manifest_loading() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "rocm-interfaces-manifest-test";
    std::filesystem::create_directories(directory);
    const std::filesystem::path provider = directory / "blas-provider.so";
    std::filesystem::copy_file(BLAS_PROVIDER_PATH, provider,
                               std::filesystem::copy_options::overwrite_existing);
    const std::filesystem::path manifest = directory / "providers.json";
    std::ofstream stream(manifest);
    stream
        << R"({"schema_version":1,"providers":[{"id":"recording-blas-legacy","domain":"blas","module":"blas-provider.so","gfx":[942],"priority":10}]})";
    stream.close();
    auto registry = std::make_shared<rocm::interfaces::ProviderRegistry>();
    registry->load_manifest(manifest);
    auto lease = registry->select(ROCM_INTERFACES_DOMAIN_BLAS, 942, sizeof(rocm_blas_provider_v1));
    require(lease->provider_id() == "recording-blas-legacy", "manifest selected wrong provider");
}

void test_manifest_validation() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "rocm-interfaces-strict-manifest-test";
    std::filesystem::create_directories(directory);
    const std::filesystem::path provider = directory / "blas-provider.so";
    std::filesystem::copy_file(BLAS_PROVIDER_PATH, provider,
                               std::filesystem::copy_options::overwrite_existing);
    const std::filesystem::path manifest = directory / "providers.json";

    const auto rejected = [&](const std::string& contents, const char* label) {
        std::ofstream stream(manifest);
        stream << contents;
        stream.close();
        rocm::interfaces::ProviderRegistry registry;
        bool failed = false;
        try {
            registry.load_manifest(manifest);
        } catch (const std::invalid_argument&) {
            failed = true;
        }
        require(failed, label);
        bool mutated = false;
        try {
            (void)registry.select(ROCM_INTERFACES_DOMAIN_BLAS, 942, sizeof(rocm_blas_provider_v1));
            mutated = true;
        } catch (const std::runtime_error&) {
        }
        require(!mutated, "a rejected manifest partially mutated the registry");
    };

    rejected(
        R"({"schema_version":1,"providers":[{"id":"recording-blas-legacy","domain":"blas","module":"blas-provider.so"}],"unknown":true})",
        "manifest accepted an unknown root key");
    rejected(
        R"({"schema_version":1,"providers":[{"id":"recording-blas-legacy","domain":"blas","module":"blas-provider.so","unknown":true}]})",
        "manifest accepted an unknown provider key");
    rejected(
        R"({"schema_version":1,"providers":[{"id":"recording-blas-legacy","domain":"blas","module":"/tmp/provider.so"}]})",
        "manifest accepted an absolute module path");
    rejected(
        R"({"schema_version":1,"providers":[{"id":"recording-blas-legacy","domain":"blas","module":"../provider.so"}]})",
        "manifest accepted a module path escaping its directory");
    rejected(
        R"({"schema_version":1,"providers":[{"id":"recording-blas-legacy","domain":"blas","module":"blas-provider.so","gfx":[-1]}]})",
        "manifest accepted a negative gfx value");
    rejected(
        R"({"schema_version":1,"providers":[{"id":"recording-blas-legacy","domain":"blas","module":"blas-provider.so","gfx":[942,942]}]})",
        "manifest accepted a duplicate id/domain/gfx entry");
    rejected(R"({"schema_version":1,"providers":[]})", "manifest accepted an empty provider list");
    rejected(
        R"({"schema_version":1,"providers":[{"id":"recording-blas-legacy","domain":"blas","module":"blas-provider.so"},{"id":"broken","domain":"blas","module":"blas-provider.so","unknown":true}]})",
        "manifest accepted a partially valid document");
}

void test_host_service_isolation() {
    TraceLog first_log;
    TraceLog second_log;
    rocm_interfaces_host_services first_services{};
    first_services.header = header(sizeof(first_services));
    first_services.user_data = &first_log;
    first_services.trace = trace;
    rocm_interfaces_host_services second_services = first_services;
    second_services.user_data = &second_log;
    auto first_registry = std::make_shared<rocm::interfaces::ProviderRegistry>(first_services);
    auto second_registry = std::make_shared<rocm::interfaces::ProviderRegistry>(second_services);
    first_registry->add_module(ROCM_INTERFACES_DOMAIN_BLAS, 0, 0, BLAS_PROVIDER_PATH);
    second_registry->add_module(ROCM_INTERFACES_DOMAIN_BLAS, 0, 0, BLAS_PROVIDER_PATH);
    rocm_interfaces_device_key device{};
    device.header = header(sizeof(device));
    device.gfx_arch = 942;
    auto first = rocm::interfaces::BlasContext::create(first_registry, device);
    auto second = rocm::interfaces::BlasContext::create(second_registry, device);
    const size_t first_before = first_log.operations.size();
    const size_t second_before = second_log.operations.size();
    rocm_blas_vector_request request{};
    request.header = header(sizeof(request));
    request.batch_count = 1;
    request.x.length = 1;
    request.y.length = 1;
    require(first->vector_execute(request) == rocblas_status_success,
            "first registry provider call failed");
    require(first_log.operations.size() == first_before + 1,
            "first registry lost its host service binding");
    require(second_log.operations.size() == second_before,
            "provider leaked a trace into the second registry");
}

void test_combined_blas_provider() {
    auto registry = std::make_shared<rocm::interfaces::ProviderRegistry>();
    registry->add_module(ROCM_INTERFACES_DOMAIN_BLAS, 0, 0, COMBINED_BLAS_PROVIDER_PATH,
                         ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL, "integrated-recording");
    registry->add_module(ROCM_INTERFACES_DOMAIN_BLASLT, 0, 0, COMBINED_BLAS_PROVIDER_PATH,
                         ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL, "integrated-recording");
    rocm_interfaces_device_key device{};
    device.header = header(sizeof(device));
    device.gfx_arch = 942;
    auto blas = rocm::interfaces::BlasContext::create(registry, device);
    auto blaslt = rocm::interfaces::BlasLtContext::create(registry, device);
    require(blas->provider_id() == blaslt->provider_id(),
            "combined replacement did not answer both BLAS domains");
    require(blas->cohort_id() == blaslt->cohort_id(),
            "combined replacement lost its cohort identity");
}
}  // namespace

int main() {
    try {
        test_public_enum_invariants();
        test_end_to_end();
        test_manifest_loading();
        test_manifest_validation();
        test_host_service_isolation();
        test_combined_blas_provider();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "all interfaces tests passed\n";
    return EXIT_SUCCESS;
}
