#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rocm/interfaces/loader.h"

namespace {

using rocm::interfaces::BlasContext;
using rocm::interfaces::ProviderRegistry;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

rocm_interfaces_abi_header header(size_t size) {
    return {static_cast<uint32_t>(size), static_cast<uint16_t>(ROCM_INTERFACES_ABI_MAJOR),
            static_cast<uint16_t>(ROCM_INTERFACES_ABI_MINOR)};
}

rocm_interfaces_device_key device_key(int ordinal, uint32_t gfx_arch) {
    rocm_interfaces_device_key device{};
    device.header = header(sizeof(device));
    device.device_ordinal = ordinal;
    device.gfx_arch = gfx_arch;
    return device;
}

class Gate {
   public:
    explicit Gate(unsigned expected) : expected_(expected) {}
    void arrive_and_wait() {
        arrived_.fetch_add(1, std::memory_order_acq_rel);
        while (arrived_.load(std::memory_order_acquire) < expected_) {
            std::this_thread::yield();
        }
    }

   private:
    const unsigned expected_;
    std::atomic<unsigned> arrived_{0};
};

constexpr unsigned kThreads = 16;
constexpr unsigned kIterations = 32;

void stress_concurrent_select_shared_registry() {
    auto registry = std::make_shared<ProviderRegistry>();
    registry->add_module(ROCM_INTERFACES_DOMAIN_BLAS, 0, 0, BLAS_CLASSIC_PROVIDER_PATH,
                         ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL, "recording-blas-cohort");

    Gate gate(kThreads);
    std::atomic<unsigned> ok{0};
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t] {
            gate.arrive_and_wait();
            try {
                for (unsigned i = 0; i < kIterations; ++i) {
                    if (t < 2) {
                        registry->add_module(
                            ROCM_INTERFACES_DOMAIN_BLAS, 0, 0, BLAS_CLASSIC_PROVIDER_PATH,
                            ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL, "recording-blas-cohort");
                    }
                    auto lease = registry->select(ROCM_INTERFACES_DOMAIN_BLAS, 942,
                                                  sizeof(rocm_blas_provider_v1));
                    if (lease && lease->provider_id() == "recording-blas-legacy") {
                        ok.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        failed.store(true, std::memory_order_relaxed);
                    }
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    for (auto& w : workers) w.join();
    require(!failed.load(), "concurrent select returned wrong/failed provider");
    require(ok.load() == kThreads * kIterations, "not all concurrent selects succeeded");
}

void stress_multistream_multidevice_dispatch() {
    auto registry = std::make_shared<ProviderRegistry>();
    registry->add_module(ROCM_INTERFACES_DOMAIN_BLAS, 0, 0, BLAS_CLASSIC_PROVIDER_PATH,
                         ROCM_INTERFACES_PROVIDER_QUERY_SYMBOL, "recording-blas-cohort");

    auto shared = BlasContext::create(registry, device_key(0, 942));

    Gate gate(kThreads);
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t] {
            gate.arrive_and_wait();
            try {
                auto own = BlasContext::create(registry, device_key(static_cast<int>(t), 942));
                rocm_blas_vector_request request{};
                request.header = header(sizeof(request));
                request.opcode = ROCM_BLAS_VECTOR_AXPY;
                request.batch_count = 1;
                request.x.length = 1;
                request.y.length = 1;
                for (unsigned i = 0; i < kIterations; ++i) {
                    if (own->vector_execute(request) != rocblas_status_success)
                        failed.store(true, std::memory_order_relaxed);
                    shared->set_stream(reinterpret_cast<void*>(static_cast<uintptr_t>(t + 1)));
                    (void)shared->stream();
                    shared->set_pointer_mode(t & 1u);
                    (void)shared->pointer_mode();
                    if (shared->vector_execute(request) != rocblas_status_success)
                        failed.store(true, std::memory_order_relaxed);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    for (auto& w : workers) w.join();
    require(!failed.load(), "multi-stream/device dispatch failed under contention");
}

void stress_bridge_call_once() {
    setenv("ROCM_INTERFACES_ROCBLAS_BRIDGE_PROVIDER", ROCBLAS_BRIDGE_PROVIDER_PATH, 1);

    Gate gate(kThreads);
    std::atomic<unsigned> created{0};
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < kThreads; ++t) {
        workers.emplace_back([&] {
            gate.arrive_and_wait();
            for (unsigned i = 0; i < kIterations; ++i) {
                rocblas_handle handle = nullptr;
                rocblas_status status = rocblas_create_handle(&handle);
                if (status != rocblas_status_success || handle == nullptr) {
                    failed.store(true, std::memory_order_relaxed);
                    continue;
                }
                created.fetch_add(1, std::memory_order_relaxed);
                if (rocblas_destroy_handle(handle) != rocblas_status_success)
                    failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    for (auto& w : workers) w.join();
    require(!failed.load(), "bridge create/destroy failed under call_once contention");
    require(created.load() == kThreads * kIterations, "not all bridge handles were created");
}

std::atomic<int> g_inflight{0};
std::atomic<int> g_max_inflight{0};

struct ProbeContext {
    rocm_blas_context_options options;
};

rocblas_status probe_create_context(const rocm_blas_context_options* options, void** result) {
    if (!options || !result) return rocblas_status_invalid_pointer;
    auto* ctx = new (std::nothrow) ProbeContext{*options};
    if (!ctx) return rocblas_status_memory_error;
    *result = ctx;
    return rocblas_status_success;
}

void probe_destroy_context(void* opaque) {
    delete static_cast<ProbeContext*>(opaque);
}

rocblas_status probe_vector_execute(void* opaque, const rocm_blas_vector_request*) {
    if (!opaque) return rocblas_status_invalid_handle;
    int cur = g_inflight.fetch_add(1, std::memory_order_acq_rel) + 1;
    int prev = g_max_inflight.load(std::memory_order_relaxed);
    while (cur > prev &&
           !g_max_inflight.compare_exchange_weak(prev, cur, std::memory_order_relaxed));
    volatile int sink = 0;
    for (int spin = 0; spin < 2000; ++spin) sink = sink + 1;
    (void)sink;
    g_inflight.fetch_sub(1, std::memory_order_acq_rel);
    return rocblas_status_success;
}

rocblas_status probe_matmul_execute(void*, const rocm_blas_matmul_request*) {
    return rocblas_status_success;
}

const rocm_blas_provider_v1 g_probe_table = {header(sizeof(rocm_blas_provider_v1)),
                                             probe_create_context, probe_destroy_context,
                                             probe_vector_execute, probe_matmul_execute};

rocm_interfaces_status probe_query(const rocm_interfaces_provider_request* request,
                                   rocm_interfaces_provider_response* response) {
    if (!request || !response || request->domain != ROCM_INTERFACES_DOMAIN_BLAS)
        return ROCM_INTERFACES_STATUS_INCOMPATIBLE_ABI;
    response->header = header(sizeof(*response));
    response->provider_id = "ops04-hotpath-probe";
    response->build_id = "ops04";
    response->dispatch_table = &g_probe_table;
    response->dispatch_table_size = sizeof(g_probe_table);
    response->capability_mask = 0;
    return ROCM_INTERFACES_STATUS_SUCCESS;
}

int probe_select_hot_path() {
    g_inflight.store(0);
    g_max_inflight.store(0);
    auto registry = std::make_shared<ProviderRegistry>();
    registry->add_builtin(ROCM_INTERFACES_DOMAIN_BLAS, 0, 0, "ops04-hotpath-probe", probe_query);
    auto context = BlasContext::create(registry, device_key(0, 942));

    Gate gate(kThreads);
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < kThreads; ++t) {
        workers.emplace_back([&] {
            rocm_blas_vector_request request{};
            request.header = header(sizeof(request));
            request.batch_count = 1;
            gate.arrive_and_wait();
            for (unsigned i = 0; i < kIterations * 8; ++i) {
                if (context->vector_execute(request) != rocblas_status_success)
                    failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    for (auto& w : workers) w.join();
    require(!failed.load(), "hot-path probe execute failed");
    return g_max_inflight.load();
}

std::mutex g_cache_mutex;
std::map<int, int> g_kernel_cache;
int g_compile_count = 0;

struct JitContext {
    rocm_blas_context_options options;
};

rocblas_status jit_create_context(const rocm_blas_context_options* options, void** result) {
    if (!options || !result) return rocblas_status_invalid_pointer;
    auto* ctx = new (std::nothrow) JitContext{*options};
    if (!ctx) return rocblas_status_memory_error;
    *result = ctx;
    return rocblas_status_success;
}

void jit_destroy_context(void* opaque) {
    delete static_cast<JitContext*>(opaque);
}

rocblas_status jit_vector_execute(void* opaque, const rocm_blas_vector_request* request) {
    if (!opaque || !request) return rocblas_status_invalid_handle;
    const int key = static_cast<int>(request->opcode);
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    if (g_kernel_cache.find(key) == g_kernel_cache.end()) {
        ++g_compile_count;
        g_kernel_cache.emplace(key, 1);
    }
    return rocblas_status_success;
}

rocblas_status jit_matmul_execute(void*, const rocm_blas_matmul_request*) {
    return rocblas_status_success;
}

const rocm_blas_provider_v1 g_jit_table = {header(sizeof(rocm_blas_provider_v1)),
                                           jit_create_context, jit_destroy_context,
                                           jit_vector_execute, jit_matmul_execute};

rocm_interfaces_status jit_query(const rocm_interfaces_provider_request* request,
                                 rocm_interfaces_provider_response* response) {
    if (!request || !response || request->domain != ROCM_INTERFACES_DOMAIN_BLAS)
        return ROCM_INTERFACES_STATUS_INCOMPATIBLE_ABI;
    response->header = header(sizeof(*response));
    response->provider_id = "ops04-jit-cache-model";
    response->build_id = "ops04";
    response->dispatch_table = &g_jit_table;
    response->dispatch_table_size = sizeof(g_jit_table);
    response->capability_mask = 0;
    return ROCM_INTERFACES_STATUS_SUCCESS;
}

void stress_jit_cache_obligation() {
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_kernel_cache.clear();
        g_compile_count = 0;
    }
    constexpr int kDistinctKeys = 2;
    auto registry = std::make_shared<ProviderRegistry>();
    registry->add_builtin(ROCM_INTERFACES_DOMAIN_BLAS, 0, 0, "ops04-jit-cache-model", jit_query);
    auto context = BlasContext::create(registry, device_key(0, 942));

    Gate gate(kThreads);
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t] {
            rocm_blas_vector_request request{};
            request.header = header(sizeof(request));
            request.batch_count = 1;
            request.opcode = (t & 1u) ? ROCM_BLAS_VECTOR_DOT : ROCM_BLAS_VECTOR_AXPY;
            gate.arrive_and_wait();
            for (unsigned i = 0; i < kIterations; ++i) {
                if (context->vector_execute(request) != rocblas_status_success)
                    failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    for (auto& w : workers) w.join();
    require(!failed.load(), "jit-cache-model execute failed");
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    require(g_compile_count == kDistinctKeys,
            "lazy-compile cache compiled a key more than once (provider guard failed)");
}

}  // namespace

int main() {
    try {
        stress_concurrent_select_shared_registry();
        std::cout << "[ok] concurrent select + add_module on shared registry\n";

        stress_multistream_multidevice_dispatch();
        std::cout << "[ok] multi-stream / multi-device dispatch\n";

        stress_bridge_call_once();
        std::cout << "[ok] rocBLAS bridge std::call_once one-time init\n";

        const int max_inflight = probe_select_hot_path();
        std::cout << "[data] max concurrent threads inside vector_execute = " << max_inflight
                  << " (of " << kThreads << " workers)\n";
        require(max_inflight >= 2,
                "per-op dispatch serialized to 1 in-flight: a global lock is on the hot path");
        std::cout << "[finding] select() global mutex is NOT on the per-op dispatch hot path\n";

        stress_jit_cache_obligation();
        std::cout << "[ok] lazy-compile-cache obligation model (provider-owned guard)\n";

        std::cout << "all OPS-04 concurrency stress tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
