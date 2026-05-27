// okl_run.cpp - generic runner for one packaged hipBLASLt kernel.
//
// Reads a key=value config file produced by `okl.py --package OUT_DIR`
// (alongside a copy of the .co), loads the kernel, builds the legacy
// 104-byte kernarg buffer, and launches with hipblaslt-bench-style timing
// (2 cold + 10 hot iters, single sync, CPU wall clock). Verifies the
// output is all zero (input was zero-filled).
//
// No dependency on libTensile or libhipblaslt - links only against the HIP
// runtime.
//
// Compile: /opt/rocm/bin/hipcc -O3 -std=c++17 okl_run.cpp -o okl_run
// Run:     ./okl_run path/to/kernel.conf
//
// Config-file keys (see okl.py --package for what populates them):
//   co_file               - .co filename (resolved relative to conf file dir)
//   kernel_symbol         - exact symbol name from the .co
//   internal_args         - u32, kernarg bytes [4..7]   (from TENSILE_DB dump)
//   internal_args1        - u32, kernarg bytes [8..11]  (from TENSILE_DB dump)
//   macro_tile_0          - u32, MT0 (from _MT<a>x<b>x<c>_ in kernel name)
//   macro_tile_1          - u32, MT1
//   workgroup_size_threads- u32, total threads/workgroup (e.g. 256)
//   kernarg_size          - u32, total kernarg buffer bytes (default 104)
//   m, n, k, batch        - u32, problem dims
//   size_{a,b,c,d}_bytes  - u64, raw allocation sizes
//   stride_{d,c,a,b}_{0,1}- u32, fields at kernarg offsets [64..95]
//   alpha, beta           - float, written at kernarg [96..99] and [100..103]

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#define HIP_CHECK(c)                                                          \
    do {                                                                      \
        hipError_t e = (c);                                                   \
        if (e) {                                                              \
            fprintf(stderr, "HIP error %d at %s:%d: %s\n", e, __FILE__,       \
                    __LINE__, hipGetErrorString(e));                          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

struct Config {
    std::string co_file;
    std::string kernel_symbol;

    uint32_t internal_args  = 0;
    uint32_t internal_args1 = 0;
    uint32_t macro_tile_0   = 0;
    uint32_t macro_tile_1   = 0;
    uint32_t workgroup_size = 0;
    uint32_t kernarg_size   = 104;

    uint32_t m = 0, n = 0, k = 0, batch = 1;

    uint64_t size_a_bytes = 0, size_b_bytes = 0;
    uint64_t size_c_bytes = 0, size_d_bytes = 0;

    uint32_t stride_d_0 = 0, stride_d_1 = 0;
    uint32_t stride_c_0 = 0, stride_c_1 = 0;
    uint32_t stride_a_0 = 0, stride_a_1 = 0;
    uint32_t stride_b_0 = 0, stride_b_1 = 0;

    float alpha = 1.0f, beta = 0.0f;
};

static std::string trim(std::string s) {
    auto issp = [](unsigned char c) { return std::isspace(c); };
    while (!s.empty() && issp(s.front())) s.erase(s.begin());
    while (!s.empty() && issp(s.back())) s.pop_back();
    return s;
}

static uint64_t parse_u64(const std::string& v) {
    return std::strtoull(v.c_str(), nullptr, 0);  // 0 = auto base (handles 0x)
}

static Config load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "okl_run: cannot open config %s\n", path.c_str());
        std::exit(1);
    }
    std::map<std::string, std::string> kv;
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        auto eq = line.find('=');
        if (eq == std::string::npos) {
            if (!trim(line).empty()) {
                fprintf(stderr, "okl_run: %s:%d: ignoring malformed line\n",
                        path.c_str(), lineno);
            }
            continue;
        }
        kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }

    auto req_str = [&](const char* k) -> std::string {
        auto it = kv.find(k);
        if (it == kv.end()) {
            fprintf(stderr, "okl_run: config missing required key '%s'\n", k);
            std::exit(1);
        }
        return it->second;
    };
    auto req_u32 = [&](const char* k) {
        return uint32_t(parse_u64(req_str(k)));
    };
    auto req_u64 = [&](const char* k) { return parse_u64(req_str(k)); };
    auto opt_u32 = [&](const char* k, uint32_t d) {
        auto it = kv.find(k);
        return it == kv.end() ? d : uint32_t(parse_u64(it->second));
    };
    auto opt_f32 = [&](const char* k, float d) {
        auto it = kv.find(k);
        return it == kv.end()
                   ? d
                   : float(std::strtod(it->second.c_str(), nullptr));
    };

    Config c;
    c.co_file        = req_str("co_file");
    c.kernel_symbol  = req_str("kernel_symbol");
    c.internal_args  = req_u32("internal_args");
    c.internal_args1 = req_u32("internal_args1");
    c.macro_tile_0   = req_u32("macro_tile_0");
    c.macro_tile_1   = req_u32("macro_tile_1");
    c.workgroup_size = req_u32("workgroup_size_threads");
    c.kernarg_size   = opt_u32("kernarg_size", 104);

    c.m            = req_u32("m");
    c.n            = req_u32("n");
    c.k            = req_u32("k");
    c.batch        = opt_u32("batch", 1);
    c.size_a_bytes = req_u64("size_a_bytes");
    c.size_b_bytes = req_u64("size_b_bytes");
    c.size_c_bytes = req_u64("size_c_bytes");
    c.size_d_bytes = req_u64("size_d_bytes");

    c.stride_d_0 = req_u32("stride_d_0");
    c.stride_d_1 = opt_u32("stride_d_1", 0);
    c.stride_c_0 = req_u32("stride_c_0");
    c.stride_c_1 = opt_u32("stride_c_1", 0);
    c.stride_a_0 = req_u32("stride_a_0");
    c.stride_a_1 = opt_u32("stride_a_1", 0);
    c.stride_b_0 = req_u32("stride_b_0");
    c.stride_b_1 = opt_u32("stride_b_1", 0);

    c.alpha = opt_f32("alpha", 1.0f);
    c.beta  = opt_f32("beta", 0.0f);
    return c;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s path/to/kernel.conf\n", argv[0]);
        return 1;
    }
    std::filesystem::path conf_path(argv[1]);
    Config c = load_config(conf_path.string());

    // Resolve .co path relative to conf dir if not absolute.
    std::filesystem::path co_path = c.co_file;
    if (co_path.is_relative()) co_path = conf_path.parent_path() / co_path;

    // 1. Allocate buffers.
    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
    HIP_CHECK(hipMalloc(&dA, c.size_a_bytes));
    HIP_CHECK(hipMalloc(&dB, c.size_b_bytes));
    HIP_CHECK(hipMalloc(&dC, c.size_c_bytes));
    HIP_CHECK(hipMalloc(&dD, c.size_d_bytes));

    // 2. Zero-fill inputs (D should also end up zero if launch is correct).
    HIP_CHECK(hipMemset(dA, 0, c.size_a_bytes));
    HIP_CHECK(hipMemset(dB, 0, c.size_b_bytes));
    HIP_CHECK(hipMemset(dC, 0, c.size_c_bytes));
    HIP_CHECK(hipMemset(dD, 0xee, c.size_d_bytes));  // poison so we can detect write

    // 3. Load module and resolve kernel.
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, co_path.c_str()));
    hipFunction_t kernel;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, c.kernel_symbol.c_str()));

    // 4. Build the kernarg buffer (legacy ABI, see research §6).
    std::vector<uint8_t> kernarg(c.kernarg_size, 0);
    auto put_u32 = [&](size_t off, uint32_t v) {
        std::memcpy(kernarg.data() + off, &v, 4);
    };
    auto put_ptr = [&](size_t off, void* p) {
        std::memcpy(kernarg.data() + off, &p, 8);
    };
    auto put_f32 = [&](size_t off, float v) {
        std::memcpy(kernarg.data() + off, &v, 4);
    };

    uint32_t numWG = ((c.m + c.macro_tile_0 - 1) / c.macro_tile_0) *
                     ((c.n + c.macro_tile_1 - 1) / c.macro_tile_1) * c.batch;

    put_u32(0,   1);                 // gemm_count (argType=0, in-kernarg)
    put_u32(4,   c.internal_args);
    put_u32(8,   c.internal_args1);
    put_u32(12,  numWG);
    put_u32(16,  c.m);
    put_u32(20,  c.n);
    put_u32(24,  c.batch);
    put_u32(28,  c.k);
    put_ptr(32,  dD);
    put_ptr(40,  dC);
    put_ptr(48,  dA);
    put_ptr(56,  dB);
    put_u32(64,  c.stride_d_0); put_u32(68, c.stride_d_1);
    put_u32(72,  c.stride_c_0); put_u32(76, c.stride_c_1);
    put_u32(80,  c.stride_a_0); put_u32(84, c.stride_a_1);
    put_u32(88,  c.stride_b_0); put_u32(92, c.stride_b_1);
    put_f32(96,  c.alpha);
    put_f32(100, c.beta);

    // 5. Launch (driver-style param buffer).
    size_t ksize = kernarg.size();
    void* launch_params[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg.data(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &ksize,
        HIP_LAUNCH_PARAM_END};

    uint32_t globalX = numWG * c.workgroup_size;

    auto launch = [&]() {
        HIP_CHECK(hipExtModuleLaunchKernel(
            kernel, globalX, 1, 1, c.workgroup_size, 1, 1,
            /*sharedMemBytes=*/0, /*stream=*/nullptr, nullptr, launch_params,
            nullptr, nullptr));
    };

    // 6. hipblaslt-bench-style timing: 2 cold + 10 hot, CPU clock, single sync.
    constexpr int COLD_ITERS = 2;
    constexpr int HOT_ITERS  = 10;
    for (int i = 0; i < COLD_ITERS; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < HOT_ITERS; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double us_per_iter = total_us / HOT_ITERS;
    double flops  = 2.0 * double(c.m) * c.n * c.k * c.batch;
    double gflops = flops / us_per_iter * 1e-3;

    printf("conf:      %s\n", conf_path.string().c_str());
    printf("co:        %s\n", co_path.string().c_str());
    printf("kernel:    %.80s%s\n", c.kernel_symbol.c_str(),
           c.kernel_symbol.size() > 80 ? "..." : "");
    printf("problem:   M=%u N=%u K=%u batch=%u  alpha=%g beta=%g\n",
           c.m, c.n, c.k, c.batch, c.alpha, c.beta);
    printf("grid:      %u workgroups x %u threads = %u global threads\n",
           numWG, c.workgroup_size, globalX);
    printf("iters:     %d hot (after %d cold), single sync, CPU wall clock\n",
           HOT_ITERS, COLD_ITERS);
    printf("time:      %.3f us / iter   (hot window: %.3f us / %d calls)\n",
           us_per_iter, total_us, HOT_ITERS);
    printf("perf:      %.1f gflops\n", gflops);

    // 7. Sanity: D must have been written (poison overwritten) and equal zero
    //    (since A=B=C=0). Cheap pass: total bytes summed via host read.
    std::vector<uint8_t> hostD(c.size_d_bytes);
    HIP_CHECK(hipMemcpy(hostD.data(), dD, c.size_d_bytes, hipMemcpyDeviceToHost));
    bool poisoned = std::all_of(hostD.begin(), hostD.end(),
                                [](uint8_t b) { return b == 0xee; });
    bool all_zero = std::all_of(hostD.begin(), hostD.end(),
                                [](uint8_t b) { return b == 0; });
    if (poisoned) {
        fprintf(stderr, "FAIL: D still poisoned (0xee everywhere) - kernel "
                        "did not write\n");
        std::exit(2);
    }
    if (!all_zero) {
        size_t nonzero = std::count_if(hostD.begin(), hostD.end(),
                                       [](uint8_t b) { return b != 0; });
        fprintf(stderr, "WARN: D has %zu/%zu non-zero bytes (expected 0 from "
                        "zero inputs)\n",
                nonzero, hostD.size());
    } else {
        printf("verify:    OK (D fully overwritten and zero, as expected for "
               "A=B=C=0)\n");
    }

    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC)); HIP_CHECK(hipFree(dD));
    HIP_CHECK(hipModuleUnload(module));
    return 0;
}
