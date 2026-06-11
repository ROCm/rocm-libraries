// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// C++ side of the dispatcher selection-parity harness.
//
// Loads the REAL shipped per-arch GEMM bundle into the REAL ck_dsl::ArtifactStore
// and runs the REAL ck_dsl::Dispatcher (the runtime twin of ck_dsl.dispatch).
// For each problem on stdin (or from --shapes file) it prints the selected
// kernel identity decomposed into the fields that are common to both the C++
// manifest and the Python UniversalGemmSpec:
//
//   kernel_name, block_m, block_n, block_k, pipeline, epilogue
//
// The C++ manifest cache_key == kernel_name, and pipeline/epilogue live in the
// retained raw JSON. Selection here is CPU-only (no GPU, no comgr); we only read
// manifests and run Dispatcher::select.
//
// Output: one JSON object per line (JSONL) on stdout. Diagnostics on stderr.
//
// Build (header-only selection path; HIP headers are pulled transitively but no
// device is touched):
//   hipcc -std=c++17 -I <provider>/runtime/include cpp_select.cpp -o cpp_select
//
// Run:
//   ./cpp_select --bundle <provider>/kernels/gfx950 --shapes shapes.txt

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ck_dsl_runtime/artifact_store.hpp"
#include "ck_dsl_runtime/dispatcher.hpp"

using namespace ck_dsl;

namespace {

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
            o += c;
        } else {
            o += c;
        }
    }
    return o;
}

// Look up the manifest the Dispatcher picked and extract the two trait strings
// (pipeline, epilogue) from its retained raw JSON, plus the tile block sizes.
struct CppPick {
    bool found = false;
    std::string kernel_name;
    int block_m = 0, block_n = 0, block_k = 0;
    std::string pipeline;
    std::string epilogue;
};

CppPick resolve(const ArtifactStore& store, const Dispatcher::Choice& c) {
    CppPick p;
    if (!c.valid()) return p;
    // The Choice carries the cache_key (== store id). Find the entry.
    for (const auto& kv : store.entries()) {
        if (kv.first != c.cache_key) continue;
        const Manifest& m = kv.second.manifest;
        p.found = true;
        p.kernel_name = m.kernel_name;
        p.block_m = m.block_m;
        p.block_n = m.block_n;
        p.block_k = m.block_k;
        p.pipeline = m.raw.get_str("pipeline");
        p.epilogue = m.raw.get_str("epilogue");
        return p;
    }
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    std::string bundle;
    std::string shapes_file;
    std::string arch = "gfx950";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc)
            bundle = argv[++i];
        else if (a == "--shapes" && i + 1 < argc)
            shapes_file = argv[++i];
        else if (a == "--arch" && i + 1 < argc)
            arch = argv[++i];
    }
    if (bundle.empty()) {
        std::fprintf(stderr, "usage: %s --bundle <dir> [--shapes file] [--arch gfx950]\n", argv[0]);
        return 2;
    }

    ArtifactStore store;
    size_t n = store.add_bundle(bundle);
    std::fprintf(stderr, "[cpp] loaded %zu manifest(s) from %s\n", n, bundle.c_str());
    int gemm_entries = 0;
    for (const auto& kv : store.entries())
        if (kv.second.manifest.kind.rfind("gemm", 0) == 0) ++gemm_entries;
    std::fprintf(stderr, "[cpp] gemm candidate manifests: %d\n", gemm_entries);

    Dispatcher disp(store);  // FirstFit (default), the twin of CandidateRegistry priority select

    std::istream* in = &std::cin;
    std::ifstream fin;
    if (!shapes_file.empty()) {
        fin.open(shapes_file);
        if (!fin) {
            std::fprintf(stderr, "[cpp] cannot open shapes file %s\n", shapes_file.c_str());
            return 2;
        }
        in = &fin;
    }

    std::string line;
    while (std::getline(*in, line)) {
        // strip comments / blanks
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::istringstream ls(line);
        long M = 0, N = 0, K = 0;
        if (!(ls >> M >> N >> K)) continue;

        Problem prob;
        prob.op = "gemm";
        prob.dtype = "fp16";
        prob.layout = "RCR";
        prob.arch = arch;
        prob.M = M;
        prob.N = N;
        prob.K = K;

        Dispatcher::Choice choice = disp.select(prob);
        CppPick pick = resolve(store, choice);

        // Emit one JSONL record.
        std::printf(
            "{\"M\":%ld,\"N\":%ld,\"K\":%ld,\"selected\":%s,"
            "\"kernel_name\":\"%s\",\"block_m\":%d,\"block_n\":%d,\"block_k\":%d,"
            "\"pipeline\":\"%s\",\"epilogue\":\"%s\"}\n",
            M, N, K, pick.found ? "true" : "false", json_escape(pick.kernel_name).c_str(),
            pick.block_m, pick.block_n, pick.block_k, json_escape(pick.pipeline).c_str(),
            json_escape(pick.epilogue).c_str());
    }
    return 0;
}
