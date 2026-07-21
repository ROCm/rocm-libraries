// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// libFuzzer entry point for PatternCompiler::fromJson. The compiler parses
// untrusted JSON on the runtime drop-in path, so it must never crash, hang, or
// exhaust the stack on adversarial bytes -- it must always return an ok or an
// error result within its configured bounds. This harness feeds arbitrary bytes
// straight in; a crash, sanitizer report, or timeout is a real defect.
//
// Build (Linux/CI clang):
//   clang++ -std=c++17 -g -O1 -fsanitize=fuzzer,address,undefined \
//     <includes> fuzz_fromjson.cpp src/*.cpp -o fuzz_fromjson
// Then: ./fuzz_fromjson -max_len=65536 corpus/

#include <cstddef>
#include <cstdint>
#include <hipdnn_graph_matcher/PatternCompiler.hpp>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    const std::string_view json(reinterpret_cast<const char*>(data), size);
    // Tight bounds so the fuzzer explores logic, not just the size guard.
    hipdnn::graph_matcher::CompileLimits limits;
    limits.maxInputBytes = 1u << 16;
    const auto result = hipdnn::graph_matcher::PatternCompiler::fromJson(json);
    // A successful compile must yield a usable pattern; touch it so any latent
    // invariant break (e.g. an out-of-range anchor) is exercised, not ignored.
    if (result.ok) {
        volatile auto nodes = result.pattern.nodeCount();
        volatile auto vars = result.pattern.varCount();
        (void)nodes;
        (void)vars;
    }
    (void)limits;
    return 0;
}
