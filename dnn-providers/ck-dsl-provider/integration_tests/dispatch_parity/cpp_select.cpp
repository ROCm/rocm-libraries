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
    std::string dtype = "fp16";
    std::string op = "gemm";  // "gemm" | "norm" | "conv" | "attention" | "moe"
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc)
            bundle = argv[++i];
        else if (a == "--shapes" && i + 1 < argc)
            shapes_file = argv[++i];
        else if (a == "--arch" && i + 1 < argc)
            arch = argv[++i];
        else if (a == "--dtype" && i + 1 < argc)
            dtype = argv[++i];
        else if (a == "--op" && i + 1 < argc)
            op = argv[++i];
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

        Problem prob;
        prob.op = op;
        prob.dtype = dtype;
        prob.arch = arch;

        if (op == "norm") {
            // Norm shape line: rows cols [kind]   (kind: rmsnorm|layernorm)
            long rows = 0, cols = 0;
            std::string kind = "rmsnorm";
            if (!(ls >> rows >> cols)) continue;
            ls >> kind;  // optional
            prob.rows = rows;
            prob.cols = cols;
            prob.kind = kind;

            Dispatcher::Choice choice = disp.select(prob);
            CppPick pick = resolve(store, choice);
            std::printf(
                "{\"rows\":%ld,\"cols\":%ld,\"kind\":\"%s\",\"selected\":%s,"
                "\"kernel_name\":\"%s\",\"block_m\":%d,\"block_n\":%d,\"block_k\":%d,"
                "\"pipeline\":\"%s\",\"epilogue\":\"%s\"}\n",
                rows, cols, json_escape(kind).c_str(), pick.found ? "true" : "false",
                json_escape(pick.kernel_name).c_str(), pick.block_m, pick.block_n, pick.block_k,
                json_escape(pick.pipeline).c_str(), json_escape(pick.epilogue).c_str());
            continue;
        }

        if (op == "moe") {
            // MoE shape line: num_tokens hidden intermediate num_experts top_k dtype
            long nt = 0, hid = 0, inter = 0, ne = 0, tk = 0;
            std::string mdtype;
            if (!(ls >> nt >> hid >> inter >> ne >> tk >> mdtype)) continue;
            prob.M = nt;          // informational; MoE dims are runtime args
            prob.dtype = mdtype;  // per-line dtype drives the element-path gate

            Dispatcher::Choice choice = disp.select(prob);
            CppPick pick = resolve(store, choice);
            std::string mpath;
            int tile_m = 0, tile_n = 0, tile_k = 0, atom_k = 0;
            if (pick.found) {
                for (const auto& kv : store.entries()) {
                    if (kv.first != choice.cache_key) continue;
                    const auto& mm = kv.second.manifest;
                    if (mm.raw.has("moe_path")) mpath = mm.raw.get_str("moe_path");
                    if (mm.raw.has("tile_m")) tile_m = (int)mm.raw.get_int("tile_m");
                    if (mm.raw.has("tile_n_inter")) tile_n = (int)mm.raw.get_int("tile_n_inter");
                    if (mm.raw.has("tile_k_gu")) tile_k = (int)mm.raw.get_int("tile_k_gu");
                    if (mm.raw.has("atom_k")) atom_k = (int)mm.raw.get_int("atom_k");
                    break;
                }
            }
            std::printf(
                "{\"num_tokens\":%ld,\"hidden\":%ld,\"intermediate\":%ld,"
                "\"num_experts\":%ld,\"top_k\":%ld,\"dtype\":\"%s\",\"selected\":%s,"
                "\"kernel_name\":\"%s\",\"path\":\"%s\",\"tile_m\":%d,"
                "\"tile_n_inter\":%d,\"tile_k_gu\":%d,\"atom_k\":%d}\n",
                nt, hid, inter, ne, tk, json_escape(mdtype).c_str(), pick.found ? "true" : "false",
                json_escape(pick.kernel_name).c_str(), json_escape(mpath).c_str(), tile_m, tile_n,
                tile_k, atom_k);
            continue;
        }

        if (op == "attention") {
            // Attention shape line:
            //   batch nhq nhk seqlen_q seqlen_k hdim [sliding_window block_kv num_sms]
            long b = 0, nhq = 0, nhk = 0, sq = 0, sk = 0, hd = 0;
            if (!(ls >> b >> nhq >> nhk >> sq >> sk >> hd)) continue;
            long sw = 0, bkv = 16, nsms = 120;
            ls >> sw >> bkv >> nsms;  // optional
            if (bkv <= 0) bkv = 16;
            if (nsms <= 0) nsms = 120;
            prob.batch = b;
            prob.nhead_q = nhq;
            prob.nhead_k = nhk;
            prob.seqlen_q = sq;
            prob.seqlen_k = sk;
            prob.hdim_q = hd;
            prob.hdim_v = hd;
            prob.total_q = b * sq;
            prob.num_seqs = b;
            prob.sliding_window = sw;
            prob.block_kv = bkv;
            prob.num_sms = nsms;

            Dispatcher::Choice choice = disp.select(prob);
            CppPick pick = resolve(store, choice);
            // For attention the structural identity is (path, head_size,
            // block_size); path lives in the manifest raw JSON. Re-read it.
            std::string path;
            if (pick.found) {
                for (const auto& kv : store.entries()) {
                    if (kv.first == choice.cache_key) {
                        if (kv.second.manifest.raw.has("path"))
                            path = kv.second.manifest.raw.get_str("path");
                        break;
                    }
                }
            }
            std::printf(
                "{\"batch\":%ld,\"nhead_q\":%ld,\"nhead_k\":%ld,\"seqlen_q\":%ld,"
                "\"seqlen_k\":%ld,\"hdim\":%ld,\"sliding_window\":%ld,"
                "\"block_kv\":%ld,\"num_sms\":%ld,\"selected\":%s,"
                "\"kernel_name\":\"%s\",\"path\":\"%s\",\"head_size\":%d,"
                "\"block_size\":%d}\n",
                b, nhq, nhk, sq, sk, hd, sw, bkv, nsms, pick.found ? "true" : "false",
                json_escape(pick.kernel_name).c_str(), json_escape(path).c_str(), pick.block_m,
                pick.block_n);
            continue;
        }

        if (op == "conv") {
            // Conv shape line: N C K Hi Wi Y X [pad_h pad_w stride_h stride_w]
            long cN = 0, cC = 0, cK = 0, Hi = 0, Wi = 0, Y = 0, X = 0;
            if (!(ls >> cN >> cC >> cK >> Hi >> Wi >> Y >> X)) continue;
            long ph = 0, pw = 0, sh = 1, sw = 1;
            ls >> ph >> pw >> sh >> sw;  // optional; default pad=0 stride=1
            prob.conv_N = cN;
            prob.conv_C = cC;
            prob.conv_K = cK;
            prob.Hi = Hi;
            prob.Wi = Wi;
            prob.Y = Y;
            prob.X = X;
            prob.pad_h = (int)ph;
            prob.pad_w = (int)pw;
            prob.stride_h = sh ? (int)sh : 1;
            prob.stride_w = sw ? (int)sw : 1;
            prob.conv_G = 1;

            Dispatcher::Choice choice = disp.select(prob);
            CppPick pick = resolve(store, choice);
            std::printf(
                "{\"N\":%ld,\"C\":%ld,\"K\":%ld,\"Hi\":%ld,\"Wi\":%ld,\"Y\":%ld,"
                "\"X\":%ld,\"pad_h\":%ld,\"pad_w\":%ld,\"stride_h\":%ld,"
                "\"stride_w\":%ld,\"selected\":%s,\"kernel_name\":\"%s\","
                "\"block_m\":%d,\"block_n\":%d,\"block_k\":%d,"
                "\"pipeline\":\"%s\",\"epilogue\":\"%s\"}\n",
                cN, cC, cK, Hi, Wi, Y, X, ph, pw, (long)prob.stride_h, (long)prob.stride_w,
                pick.found ? "true" : "false", json_escape(pick.kernel_name).c_str(), pick.block_m,
                pick.block_n, pick.block_k, json_escape(pick.pipeline).c_str(),
                json_escape(pick.epilogue).c_str());
            continue;
        }

        // Default: GEMM shape line: M N K
        long M = 0, N = 0, K = 0;
        if (!(ls >> M >> N >> K)) continue;
        prob.layout = "RCR";
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
