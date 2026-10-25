// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * End-to-end host proof:
 *
 *   C++ WmmaGemmSpec -> C++ KernelDef -> ck.dsl.ir/v1
 *     -> Rust gfx1151 instruction encoder -> Rust ELF/AMDHSA writer -> HSACO
 *
 * No LLVM IR is produced and no compiler tool, COMGR API, compiler shared
 * library, image template, or code-object cache is used.
 */

#include "rocke/code_object_ffi.h"
#include "rocke/instance_gfx1151_wmma_gemm.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

constexpr const char* kArch = "gfx1151";
constexpr const char* kBaseName = "rocke_direct_wmma_gemm";

struct DirectRun
{
    std::string symbol;
    std::vector<uint8_t> hsaco;
    std::string diagnostics;
    double cpp_build_ms = 0.0;
    double rust_compile_ms = 0.0;
    double total_ms = 0.0;
};

std::string bytes_to_string(const uint8_t* data, size_t size)
{
    if(!data || size == 0)
        return {};
    return std::string(reinterpret_cast<const char*>(data), size);
}

double milliseconds(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void clear_compiler_environment()
{
#if defined(_WIN32)
    for(const char* name : {"ROCKE_COMGR_LIB", "HIP_CLANG_PATH"})
        _putenv_s(name, "");
#else
    for(const char* name : {"ROCKE_COMGR_LIB", "HIP_CLANG_PATH"})
        unsetenv(name);
#endif
}

DirectRun compile_once(bool block_x_is_m)
{
    const auto build_started = Clock::now();
    rocke_wmma_gemm_gfx1151_spec_t spec = rocke_wmma_gemm_gfx1151_spec_default();
    spec.name = kBaseName;
    spec.block_x_is_m = block_x_is_m;

    char reason[512] = {};
    if(!rocke_wmma_gemm_gfx1151_is_valid_spec(&spec, kArch, reason, sizeof(reason)))
        throw std::runtime_error(std::string("invalid C++ WMMA GEMM spec: ") + reason);

    char symbol[256] = {};
    if(rocke_wmma_gemm_gfx1151_kernel_name(&spec, symbol, sizeof(symbol)) != ROCKE_OK)
        throw std::runtime_error("C++ GEMM spec could not form its kernel symbol");

    rocke_ir_builder_t builder;
    if(rocke_ir_builder_init(&builder, symbol) != ROCKE_OK)
        throw std::runtime_error("C++ IR builder initialization failed");

    rocke_kernel_def_t* kernel = rocke_build_wmma_gemm_gfx1151(&builder, &spec, kArch);
    if(!kernel)
    {
        const char* message = rocke_ir_builder_error(&builder);
        const std::string detail = message ? message : "unknown builder failure";
        rocke_ir_builder_free(&builder);
        throw std::runtime_error("C++ GEMM builder failed: " + detail);
    }

    char* serialized_ir = nullptr;
    if(rocke_ir_serialize(kernel, &serialized_ir) != ROCKE_OK || !serialized_ir)
    {
        rocke_ir_builder_free(&builder);
        throw std::runtime_error("C++ GEMM IR serialization failed");
    }
    const auto build_finished = Clock::now();

    rocke_direct_gemm_compile_request_v1 request = {};
    request.struct_size = sizeof(request);
    request.serialized_ir = reinterpret_cast<const uint8_t*>(serialized_ir);
    request.serialized_ir_size = std::strlen(serialized_ir);

    rocke_codeobj_compile_result_v1 result = {};
    result.struct_size = sizeof(result);
    const auto compile_started = Clock::now();
    const int32_t status = rocke_codeobj_compile_gfx1151_wmma_gemm_v1(&request, &result);
    const auto compile_finished = Clock::now();
    std::free(serialized_ir);
    rocke_ir_builder_free(&builder);

    if(status != 0 || result.status != 0)
    {
        const std::string diagnostic = bytes_to_string(result.diagnostics, result.diagnostics_size);
        rocke_codeobj_compile_result_dispose_v1(&result);
        throw std::runtime_error("Rust direct GEMM compiler failed: " + diagnostic);
    }

    DirectRun run;
    run.symbol = symbol;
    run.hsaco.assign(result.hsaco, result.hsaco + result.hsaco_size);
    run.diagnostics = bytes_to_string(result.diagnostics, result.diagnostics_size);
    run.cpp_build_ms = milliseconds(build_started, build_finished);
    run.rust_compile_ms = milliseconds(compile_started, compile_finished);
    run.total_ms = milliseconds(build_started, compile_finished);
    rocke_codeobj_compile_result_dispose_v1(&result);

    if(run.hsaco.size() < 4
       || std::memcmp(run.hsaco.data(),
                      "\x7f"
                      "ELF",
                      4)
              != 0)
        throw std::runtime_error("Rust direct compiler returned a non-ELF artifact");
    const auto symbol_begin = reinterpret_cast<const uint8_t*>(run.symbol.data());
    if(std::search(
           run.hsaco.begin(), run.hsaco.end(), symbol_begin, symbol_begin + run.symbol.size())
       == run.hsaco.end())
        throw std::runtime_error("generated HSACO does not contain its C++ kernel symbol");
    constexpr const char* llvm_linker = "Linker: AMD LLVM";
    const auto llvm_linker_begin = reinterpret_cast<const uint8_t*>(llvm_linker);
    if(std::search(run.hsaco.begin(),
                   run.hsaco.end(),
                   llvm_linker_begin,
                   llvm_linker_begin + std::strlen(llvm_linker))
       != run.hsaco.end())
        throw std::runtime_error("generated HSACO incorrectly claims LLVM linker provenance");
    return run;
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if(values.size() % 2 == 0)
        return (values[middle - 1] + values[middle]) / 2.0;
    return values[middle];
}

int parse_iterations(const char* text)
{
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if(!end || *end != '\0' || value < 1 || value > 10000)
        throw std::runtime_error("iterations must be an integer in 1..=10000");
    return static_cast<int>(value);
}
} // namespace

int main(int argc, char** argv)
try
{
    if(argc > 3)
    {
        std::cerr << "usage: " << argv[0] << " [HSACO_OUT] [ITERATIONS]\n";
        return 2;
    }
    if(rocke_codeobj_ffi_abi_version() != ROCKE_CODEOBJ_FFI_ABI_VERSION)
        throw std::runtime_error("Rust code-object library has an incompatible ABI");

    clear_compiler_environment();
    const std::string output_path = argc >= 2 ? argv[1] : "";
    const int iterations = argc >= 3 ? parse_iterations(argv[2]) : 3;

    std::vector<DirectRun> runs;
    runs.reserve(static_cast<size_t>(iterations));
    for(int iteration = 0; iteration < iterations; ++iteration)
        runs.push_back(compile_once(true));

    const DirectRun xn = compile_once(false);
    if(runs.back().symbol == xn.symbol || runs.back().hsaco == xn.hsaco)
        throw std::runtime_error("xm/xn C++ graphs did not produce distinct generated artifacts");

    if(!output_path.empty())
    {
        std::ofstream output(output_path, std::ios::binary);
        if(!output)
            throw std::runtime_error("cannot create HSACO output file");
        output.write(reinterpret_cast<const char*>(runs.back().hsaco.data()),
                     static_cast<std::streamsize>(runs.back().hsaco.size()));
        if(!output)
            throw std::runtime_error("cannot write HSACO output file");
    }

    std::vector<double> cpp_build;
    std::vector<double> rust_compile;
    std::vector<double> total;
    for(const DirectRun& run : runs)
    {
        cpp_build.push_back(run.cpp_build_ms);
        rust_compile.push_back(run.rust_compile_ms);
        total.push_back(run.total_ms);
    }
    std::cout << "{\"iterations\":" << iterations
              << ",\"cpp_ir_build_ms_median\":" << median(cpp_build)
              << ",\"rust_direct_compile_ms_median\":" << median(rust_compile)
              << ",\"native_total_ms_median\":" << median(total)
              << ",\"hsaco_bytes\":" << runs.back().hsaco.size() << ",\"xm_symbol\":\""
              << runs.back().symbol << "\"" << ",\"xn_symbol\":\"" << xn.symbol << "\""
              << ",\"external_compiler\":false,\"cache\":false}\n";
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
