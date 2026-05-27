// okl_run.cpp - generic runner for one packaged hipBLASLt kernel.
//
// Reads a key=value config file produced by `okl.py --package OUT_DIR`
// (alongside a copy of the .co), loads the kernel, builds the kernarg buffer
// from a metadata-derived slot list, and launches with hipblaslt-bench-style
// timing (2 cold + 10 hot iters, single sync, CPU wall clock). Verifies the
// D buffer was overwritten and matches the expected output for zero inputs.
//
// No dependency on libTensile or libhipblaslt - links only against the HIP
// runtime.
//
// Compile: /opt/rocm/bin/hipcc -O3 -std=c++17 okl_run.cpp -o okl_run
// Run:     ./okl_run path/to/kernel.conf
//
// Config-file shape (see okl.py --package and research §9 for what populates it):
//
// --- Loading (validated by load_kernel) ---
//   co_file               - .co filename (resolved relative to conf file dir)
//   kernel_symbol         - exact symbol name from the .co (must exist; verified
//                           via hipModuleGetFunction + hipFuncGetAttribute)
//
// --- ABI / launch ---
//   kernarg_size            - total kernarg buffer bytes (from ELF
//                             .kernarg_segment_size)
//   workgroup_size_threads  - threads per workgroup (from the dump's
//                             `l(N, 1, 1) x g(...)` line)
//   m, n, k, batch          - problem dims (echoed for diagnostics; the slot
//                             list is what actually drives the launch)
//   buffer = name=<role> bytes=<N> init=<zero|poison>
//                           - one per logical buffer the kernel reads/writes
//   slot = offset=<O> size=<S> kind=<value|buffer> ...
//                           - one per kernarg field. For kind=value carries
//                             ctype=<u32|f32|f64|i32|...> and value=<literal>.
//                             For kind=buffer carries buffer=<role> to bind
//                             to one of the declared buffers.
//
// The runner walks the buffers list (hipMalloc + hipMemset per the init mode),
// then walks the slot list in order, encoding each value at its declared
// offset. There are no hardcoded ABI offsets or field names anymore.

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
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
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

// ---------------------------------------------------------------------------
// Slot record. One per kernarg field in the kernel's amdhsa.kernels[*].args
// metadata. Two flavors:
//   - kind=value : we encode `value` (typed by ctype) into `size` bytes at
//                  `offset`.
//   - kind=buffer: we look up the device pointer for `buffer` (a role name
//                  declared in the buffers list) and encode it as 8 bytes
//                  at `offset`.
// ---------------------------------------------------------------------------
struct Slot {
    size_t      offset = 0;
    size_t      size   = 0;
    std::string kind;        // "value" | "buffer"
    std::string name;        // diagnostic only
    // For kind=value:
    std::string ctype;       // "u32", "i32", "f32", "f64", ...
    // Value is stored as raw bytes already encoded per ctype, sized to `size`.
    std::vector<uint8_t> value_bytes;
    // For kind=buffer:
    std::string buffer;      // role name
};

struct BufferDecl {
    std::string role;
    uint64_t    bytes = 0;
    std::string init = "zero";  // "zero" | "poison"
};

struct Config {
    std::string co_file;
    std::string kernel_symbol;

    uint32_t kernarg_size   = 0;
    uint32_t workgroup_size = 0;

    // Echoed for diagnostics; not used to compute anything.
    uint32_t m = 0, n = 0, k = 0, batch = 1;
    uint32_t macro_tile_0 = 0, macro_tile_1 = 0;

    std::vector<BufferDecl> buffers;
    std::vector<Slot>       slots;
};

static std::string trim(std::string s) {
    auto issp = [](unsigned char c) { return std::isspace(c); };
    while (!s.empty() && issp(s.front())) s.erase(s.begin());
    while (!s.empty() && issp(s.back())) s.pop_back();
    return s;
}

static uint64_t parse_u64(const std::string& v) {
    return std::strtoull(v.c_str(), nullptr, 0);
}

// Tokenize one ` slot = key1=v1 key2=v2 ...` body into a map.
static std::unordered_map<std::string, std::string>
parse_kv_list(const std::string& body) {
    std::unordered_map<std::string, std::string> out;
    std::istringstream iss(body);
    std::string tok;
    while (iss >> tok) {
        auto eq = tok.find('=');
        if (eq == std::string::npos) continue;
        out[tok.substr(0, eq)] = tok.substr(eq + 1);
    }
    return out;
}

// Pack a numeric value into `size` little-endian bytes for the given ctype.
// Returns false if ctype is unknown or value parses unrepresentably.
static bool encode_value(const std::string& ctype, const std::string& vstr,
                         size_t size, std::vector<uint8_t>& out) {
    out.assign(size, 0);
    if (ctype == "u32" || ctype == "i32") {
        if (size != 4) return false;
        uint32_t v = uint32_t(parse_u64(vstr));
        std::memcpy(out.data(), &v, 4);
        return true;
    }
    if (ctype == "u64" || ctype == "i64") {
        if (size != 8) return false;
        uint64_t v = parse_u64(vstr);
        std::memcpy(out.data(), &v, 8);
        return true;
    }
    if (ctype == "u16" || ctype == "i16") {
        if (size != 2) return false;
        uint16_t v = uint16_t(parse_u64(vstr));
        std::memcpy(out.data(), &v, 2);
        return true;
    }
    if (ctype == "u8" || ctype == "i8") {
        if (size != 1) return false;
        uint8_t v = uint8_t(parse_u64(vstr));
        out[0] = v;
        return true;
    }
    if (ctype == "f32") {
        if (size != 4) return false;
        float v = std::strtof(vstr.c_str(), nullptr);
        std::memcpy(out.data(), &v, 4);
        return true;
    }
    if (ctype == "f64") {
        if (size != 8) return false;
        double v = std::strtod(vstr.c_str(), nullptr);
        std::memcpy(out.data(), &v, 8);
        return true;
    }
    if (ctype == "pkf16") {
        // Packed half: lower 16 bits = beta, upper 16 = beta. We don't try to
        // FP-convert here; the dump's raw u32 value already carries it.
        if (size != 4) return false;
        uint32_t v = uint32_t(parse_u64(vstr));
        std::memcpy(out.data(), &v, 4);
        return true;
    }
    // Conservative fallback: treat as integer of the slot's width.
    if (size == 4) {
        uint32_t v = uint32_t(parse_u64(vstr));
        std::memcpy(out.data(), &v, 4);
        return true;
    }
    if (size == 8) {
        uint64_t v = parse_u64(vstr);
        std::memcpy(out.data(), &v, 8);
        return true;
    }
    return false;
}

static Config load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "okl_run: cannot open config %s\n", path.c_str());
        std::exit(1);
    }
    Config c;
    std::map<std::string, std::string> scalars;
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) {
            fprintf(stderr, "okl_run: %s:%d: ignoring malformed line\n",
                    path.c_str(), lineno);
            continue;
        }
        std::string key  = trim(line.substr(0, eq));
        std::string body = trim(line.substr(eq + 1));

        if (key == "slot") {
            auto kv = parse_kv_list(body);
            Slot s;
            s.offset = size_t(parse_u64(kv["offset"]));
            s.size   = size_t(parse_u64(kv["size"]));
            s.kind   = kv.count("kind") ? kv["kind"] : "value";
            s.name   = kv.count("name") ? kv["name"] : "";
            if (s.kind == "buffer") {
                s.buffer = kv["buffer"];
            } else {
                s.ctype = kv.count("ctype") ? kv["ctype"] : "u32";
                if (!encode_value(s.ctype, kv["value"], s.size, s.value_bytes)) {
                    fprintf(stderr,
                            "okl_run: %s:%d: cannot encode slot '%s' "
                            "(ctype=%s size=%zu value=%s)\n",
                            path.c_str(), lineno, s.name.c_str(),
                            s.ctype.c_str(), s.size, kv["value"].c_str());
                    std::exit(1);
                }
            }
            c.slots.push_back(std::move(s));
            continue;
        }
        if (key == "buffer") {
            auto kv = parse_kv_list(body);
            BufferDecl b;
            b.role  = kv["name"];
            b.bytes = parse_u64(kv["bytes"]);
            b.init  = kv.count("init") ? kv["init"] : "zero";
            c.buffers.push_back(std::move(b));
            continue;
        }
        scalars[key] = body;
    }

    auto req_str = [&](const char* k) -> std::string {
        auto it = scalars.find(k);
        if (it == scalars.end()) {
            fprintf(stderr, "okl_run: config missing required key '%s'\n", k);
            std::exit(1);
        }
        return it->second;
    };
    auto opt_u32 = [&](const char* k, uint32_t d) {
        auto it = scalars.find(k);
        return it == scalars.end() ? d : uint32_t(parse_u64(it->second));
    };
    c.co_file        = req_str("co_file");
    c.kernel_symbol  = req_str("kernel_symbol");
    c.kernarg_size   = uint32_t(parse_u64(req_str("kernarg_size")));
    c.workgroup_size = uint32_t(parse_u64(req_str("workgroup_size_threads")));
    c.m              = opt_u32("m", 0);
    c.n              = opt_u32("n", 0);
    c.k              = opt_u32("k", 0);
    c.batch          = opt_u32("batch", 1);
    c.macro_tile_0   = opt_u32("macro_tile_0", 0);
    c.macro_tile_1   = opt_u32("macro_tile_1", 0);
    return c;
}

// ----------------------------------------------------------------------------
// Loading interface (see kernel-packaging-research.md §8).
// Unchanged from part 1.
// ----------------------------------------------------------------------------

struct LoadedKernel {
    hipModule_t   module;
    hipFunction_t function;
    int           num_regs;
    int           lds_bytes;
};

static LoadedKernel load_kernel(const std::filesystem::path& co_path,
                                const std::string& kernel_symbol) {
    if (!std::filesystem::exists(co_path)) {
        fprintf(stderr, "okl_run: .co file not found: %s\n",
                co_path.string().c_str());
        std::exit(1);
    }

    LoadedKernel lk{};
    hipError_t err = hipModuleLoad(&lk.module, co_path.c_str());
    if (err == hipErrorNoBinaryForGpu) {
        fprintf(stderr,
                "okl_run: hipModuleLoad rejected %s (hipErrorNoBinaryForGpu).\n"
                "  This usually means the bundle has no slice for the running "
                "GPU arch. Use:\n"
                "    clang-offload-bundler --list --type=o --input=%s\n"
                "  to see which targets ARE in the bundle.\n",
                co_path.string().c_str(), co_path.string().c_str());
        std::exit(1);
    } else if (err != hipSuccess) {
        fprintf(stderr, "okl_run: hipModuleLoad(%s) failed: %s\n",
                co_path.string().c_str(), hipGetErrorString(err));
        std::exit(1);
    }

    err = hipModuleGetFunction(&lk.function, lk.module, kernel_symbol.c_str());
    if (err == hipErrorNotFound) {
        fprintf(stderr,
                "okl_run: kernel symbol not found in module:\n"
                "  symbol: %s\n"
                "  co    : %s\n"
                "HIP does not enumerate module symbols. To list what IS there:\n"
                "    clang-offload-bundler --unbundle --type=o --input=%s \\\n"
                "        --output=/tmp/host.o --output=/tmp/dev.o \\\n"
                "        --targets=host-x86_64-unknown-linux-gnu-,"
                "hipv4-amdgcn-amd-amdhsa--<arch>\n"
                "    llvm-readobj --notes /tmp/dev.o | grep '\\.symbol:'\n",
                kernel_symbol.c_str(), co_path.string().c_str(),
                co_path.string().c_str());
        (void)hipModuleUnload(lk.module);
        std::exit(1);
    } else if (err != hipSuccess) {
        fprintf(stderr, "okl_run: hipModuleGetFunction failed: %s\n",
                hipGetErrorString(err));
        (void)hipModuleUnload(lk.module);
        std::exit(1);
    }

    lk.num_regs = -1;
    lk.lds_bytes = -1;
    int v = 0;
    if (hipFuncGetAttribute(&v, HIP_FUNC_ATTRIBUTE_NUM_REGS, lk.function) ==
        hipSuccess)
        lk.num_regs = v;
    if (hipFuncGetAttribute(&v, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                            lk.function) == hipSuccess)
        lk.lds_bytes = v;
    return lk;
}

// ----------------------------------------------------------------------------
// numWG: total workgroups for the launch. We get it directly from one of the
// slots (the by_value slot named "numWG" - this is what Tensile computed for
// the captured launch). The runner doesn't reproduce the StreamK / GSU /
// transposeC math; it replays exactly what was captured.
// ----------------------------------------------------------------------------
static uint32_t numwg_from_slots(const std::vector<Slot>& slots) {
    for (const auto& s : slots) {
        if (s.kind == "value" && s.name == "numWG" && s.value_bytes.size() == 4) {
            uint32_t v;
            std::memcpy(&v, s.value_bytes.data(), 4);
            return v;
        }
    }
    fprintf(stderr, "okl_run: no slot named 'numWG' in conf; cannot compute "
                    "grid size.\n");
    std::exit(1);
}

// ----------------------------------------------------------------------------
// Pieces of main(), broken out for readability.
// ----------------------------------------------------------------------------

/**
 * Set of allocated device buffers, indexed by their config-declared `role`
 * name (typically "A", "B", "C", "D", plus feature-gated ones like "bias").
 */
struct BufferSet {
    std::unordered_map<std::string, void*>    ptrs;   /**< role -> device pointer */
    std::unordered_map<std::string, uint64_t> sizes;  /**< role -> bytes */
};

/**
 * Result of `time_kernel`: matches hipblaslt-bench's reported timing model
 * (steady-state mean over a tight loop following untimed warmup).
 */
struct TimingResult {
    int    cold_iters;   /**< Number of untimed warmup launches. */
    int    hot_iters;    /**< Number of timed launches in the hot loop. */
    double total_us;     /**< Wall-clock microseconds for the hot window. */
    double us_per_iter;  /**< total_us / hot_iters. */
};

/**
 * Allocate one device buffer per declaration in `c.buffers` and initialize
 * it per its `init` mode: "poison" fills with 0xee (so we can later detect
 * regions the kernel didn't write), anything else zero-fills.
 *
 * Returns a BufferSet mapping each buffer's role -> device pointer and
 * role -> allocation size, used downstream by `build_kernarg` (to plug
 * pointers into pointer slots) and `verify_d_buffer` (to read D back).
 *
 * Aborts via HIP_CHECK if any `hipMalloc` / `hipMemset` fails.
 */
static BufferSet allocate_buffers(const Config& c) {
    BufferSet bs;
    for (const auto& b : c.buffers) {
        void* p = nullptr;
        HIP_CHECK(hipMalloc(&p, b.bytes));
        HIP_CHECK(hipMemset(p, b.init == "poison" ? 0xee : 0, b.bytes));
        bs.ptrs[b.role]  = p;
        bs.sizes[b.role] = b.bytes;
    }
    return bs;
}

/**
 * Build the flat kernarg byte buffer by walking the config's slot list in
 * order. For each slot:
 *   - `kind=buffer` slots: look up the device pointer for the named buffer
 *     role and memcpy 8 bytes into the slot's offset.
 *   - `kind=value`  slots: memcpy the pre-encoded `value_bytes` (whose
 *     width was set by okl.py from the kernel's HSA `.args` metadata) into
 *     the slot's offset.
 *
 * Bounds-checks every slot against `c.kernarg_size`; exits with a clean
 * named-slot error message on overrun, on a buffer reference to a role we
 * didn't allocate, or on a buffer slot whose size isn't 8 bytes.
 */
static std::vector<uint8_t>
build_kernarg(const Config& c, const BufferSet& bs) {
    std::vector<uint8_t> kernarg(c.kernarg_size, 0);
    for (const auto& s : c.slots) {
        if (s.offset + s.size > kernarg.size()) {
            fprintf(stderr, "okl_run: slot '%s' overruns kernarg buffer "
                            "(offset=%zu size=%zu kernarg_size=%u)\n",
                    s.name.c_str(), s.offset, s.size, c.kernarg_size);
            std::exit(1);
        }
        if (s.kind == "buffer") {
            auto it = bs.ptrs.find(s.buffer);
            if (it == bs.ptrs.end()) {
                fprintf(stderr, "okl_run: slot '%s' references unknown buffer "
                                "role '%s'\n",
                        s.name.c_str(), s.buffer.c_str());
                std::exit(1);
            }
            if (s.size != 8) {
                fprintf(stderr, "okl_run: buffer slot '%s' has size=%zu "
                                "(expected 8 for a device pointer)\n",
                        s.name.c_str(), s.size);
                std::exit(1);
            }
            void* p = it->second;
            std::memcpy(kernarg.data() + s.offset, &p, 8);
        } else {
            std::memcpy(kernarg.data() + s.offset, s.value_bytes.data(),
                        s.size);
        }
    }
    return kernarg;
}

/**
 * Launch `fn` COLD_ITERS times to warm caches (with a single sync after the
 * warmup), then HOT_ITERS times in a tight loop with one sync at the end,
 * timing the hot window with `std::chrono::steady_clock`. Matches
 * hipblaslt-bench's default-mode timing methodology (CPU wall clock with one
 * sync per timing window; see clients/common/include/argument_model.hpp:80-94
 * and testing_matmul.hpp:5293-5396).
 *
 * `kernarg` is passed in by reference but never mutated; we hand the bytes
 * off to the HIP driver via HIP_LAUNCH_PARAM_BUFFER_POINTER.
 */
static TimingResult time_kernel(hipFunction_t fn,
                                std::vector<uint8_t>& kernarg,
                                uint32_t workgroup_size,
                                uint32_t global_threads) {
    constexpr int COLD_ITERS = 2;
    constexpr int HOT_ITERS  = 10;

    size_t ksize = kernarg.size();
    void*  launch_params[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg.data(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &ksize,
        HIP_LAUNCH_PARAM_END};

    auto launch = [&]() {
        HIP_CHECK(hipExtModuleLaunchKernel(
            fn, global_threads, 1, 1, workgroup_size, 1, 1,
            /*sharedMemBytes=*/0, /*stream=*/nullptr, nullptr, launch_params,
            nullptr, nullptr));
    };

    for (int i = 0; i < COLD_ITERS; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < HOT_ITERS; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    TimingResult r;
    r.cold_iters  = COLD_ITERS;
    r.hot_iters   = HOT_ITERS;
    r.total_us    = std::chrono::duration<double, std::micro>(t1 - t0).count();
    r.us_per_iter = r.total_us / HOT_ITERS;
    return r;
}

/**
 * Emit the standard runner preamble + timing + performance lines to stdout.
 * Lines: conf path, .co path, kernel symbol (truncated to 80 chars),
 * resource estimate from hipFuncGetAttribute (regs/LDS), problem dims,
 * kernarg layout summary, grid dims, iter counts, hot-window time, gflops.
 */
static void print_report(const std::filesystem::path& conf_path,
                         const std::filesystem::path& co_path,
                         const Config&        c,
                         const LoadedKernel&  lk,
                         uint32_t             num_workgroups,
                         uint32_t             global_threads,
                         const TimingResult&  timing) {
    double flops  = 2.0 * double(c.m) * c.n * c.k * c.batch;
    double gflops = flops / timing.us_per_iter * 1e-3;

    printf("conf:      %s\n", conf_path.string().c_str());
    printf("co:        %s\n", co_path.string().c_str());
    printf("kernel:    %.80s%s\n", c.kernel_symbol.c_str(),
           c.kernel_symbol.size() > 80 ? "..." : "");
    if (lk.num_regs >= 0 || lk.lds_bytes >= 0) {
        printf("resources: regs=%d lds=%d bytes\n", lk.num_regs, lk.lds_bytes);
    }
    printf("problem:   M=%u N=%u K=%u batch=%u\n", c.m, c.n, c.k, c.batch);
    printf("kernarg:   %u bytes, %zu slots, %zu buffers\n",
           c.kernarg_size, c.slots.size(), c.buffers.size());
    printf("grid:      %u workgroups x %u threads = %u global threads\n",
           num_workgroups, c.workgroup_size, global_threads);
    printf("iters:     %d hot (after %d cold), single sync, CPU wall clock\n",
           timing.hot_iters, timing.cold_iters);
    printf("time:      %.3f us / iter   (hot window: %.3f us / %d calls)\n",
           timing.us_per_iter, timing.total_us, timing.hot_iters);
    printf("perf:      %.1f gflops\n", gflops);
}

/**
 * Read the D buffer back to host and check (a) the kernel actually wrote it
 * (no 0xee poison bytes left from the init) and (b) the result is all zero
 * (since A, B, C were all zero-initialized and beta=0, alpha*0+beta*0=0
 * regardless of dtype). Prints "verify: OK" on full pass.
 *
 * Exits 2 on detection of a never-written D (full poison preserved). Warns
 * to stderr on partial non-zero results (kernel wrote, but result wasn't
 * the algebraically expected zero - usually means alpha/beta wasn't honored
 * or one of the inputs wasn't really zero on the device). Returns silently
 * if D isn't in the buffer set (some kernels with feature-gated outputs).
 */
static void verify_d_buffer(const BufferSet& bs) {
    auto dit = bs.ptrs.find("D");
    if (dit == bs.ptrs.end()) return;

    uint64_t bytes = bs.sizes.at("D");
    std::vector<uint8_t> hostD(bytes);
    HIP_CHECK(hipMemcpy(hostD.data(), dit->second, bytes,
                        hipMemcpyDeviceToHost));

    bool poisoned = std::all_of(hostD.begin(), hostD.end(),
                                [](uint8_t b) { return b == 0xee; });
    bool all_zero = std::all_of(hostD.begin(), hostD.end(),
                                [](uint8_t b) { return b == 0; });
    if (poisoned) {
        fprintf(stderr, "FAIL: D still poisoned (0xee everywhere) - "
                        "kernel did not write\n");
        std::exit(2);
    }
    if (!all_zero) {
        size_t nonzero = std::count_if(hostD.begin(), hostD.end(),
                                       [](uint8_t b) { return b != 0; });
        fprintf(stderr, "WARN: D has %zu/%zu non-zero bytes (expected 0 "
                        "from zero inputs)\n",
                nonzero, hostD.size());
    } else {
        printf("verify:    OK (D fully overwritten and zero, as expected "
               "for A=B=C=0)\n");
    }
}

/**
 * Free every device buffer in `bs` and unload the HIP module.
 */
static void cleanup(const BufferSet& bs, hipModule_t module) {
    for (const auto& kv : bs.ptrs) HIP_CHECK(hipFree(kv.second));
    HIP_CHECK(hipModuleUnload(module));
}

// ----------------------------------------------------------------------------
// Entry point.
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    // 1. Parse argv and load the conf file.
    if (argc < 2) {
        fprintf(stderr, "usage: %s path/to/kernel.conf\n", argv[0]);
        return 1;
    }
    std::filesystem::path conf_path(argv[1]);
    Config c = load_config(conf_path.string());

    // 2. Resolve the .co path relative to the conf file's directory.
    std::filesystem::path co_path = c.co_file;
    if (co_path.is_relative()) co_path = conf_path.parent_path() / co_path;

    // 3. Allocate and init every declared device buffer.
    BufferSet bs = allocate_buffers(c);

    // 4. Load the module and resolve the kernel symbol.
    LoadedKernel lk = load_kernel(co_path, c.kernel_symbol);

    // 5. Build the kernarg buffer from the slot list.
    std::vector<uint8_t> kernarg = build_kernarg(c, bs);

    // 6. Compute the launch grid and time the kernel (bench-style 2 cold + 10 hot).
    uint32_t num_workgroups = numwg_from_slots(c.slots);
    uint32_t global_threads = num_workgroups * c.workgroup_size;
    TimingResult timing     = time_kernel(lk.function, kernarg,
                                          c.workgroup_size, global_threads);

    // 7. Print the runner preamble + timing + perf to stdout.
    print_report(conf_path, co_path, c, lk, num_workgroups, global_threads,
                 timing);

    // 8. Sanity-check: D should have been overwritten and end up zero.
    verify_d_buffer(bs);

    // 9. Free every device buffer and unload the module.
    cleanup(bs, lk.module);
    return 0;
}
