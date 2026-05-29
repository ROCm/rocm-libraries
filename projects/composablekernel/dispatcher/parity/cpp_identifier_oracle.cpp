// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// C++ runtime oracle for parity deliverable (b).
//
// Reads flat `key=value` descriptions of dispatcher KernelKeys on stdin,
// rebuilds each real ck_tile::dispatcher::KernelKey using the string_to_*
// helpers from kernel_key.hpp, and prints KernelKey::encode_identifier().
//
// Multiple configs may be batched in one invocation: separate them with a line
// containing only "---". One identifier is printed per config, in order. This
// keeps the parity checker to a single process spawn even for the full
// cartesian product (hundreds of thousands of configs).
//
// kernel_key.hpp is pure host C++ (<sstream>/<string>/<tuple>), so this builds
// and runs with a plain host compiler -- NO GPU and NO hipcc required:
//
//     g++ -std=c++17 -I <dispatcher/include> cpp_identifier_oracle.cpp -o oracle
//
// The Python side (check_identifier_parity.py) feeds the configs' fields, reads
// the printed identifiers back, and diffs them against
// identifier.encode_identifier (the Python oracle). Agreement proves codegen
// (Python) and runtime (C++) produce the same registry key byte-for-byte.

#include "ck_tile/dispatcher/kernel_key.hpp"

#include <iostream>
#include <map>
#include <string>

namespace dispatcher = ck_tile::dispatcher;

namespace {

bool to_bool(const std::string& v)
{
    return v == "1" || v == "true" || v == "True" || v == "TRUE";
}

// Look up a required key; abort with a clear message if the Python side forgot
// to emit it (keeps the two serializers honest).
const std::string& require(const std::map<std::string, std::string>& kv, const std::string& key)
{
    auto it = kv.find(key);
    if(it == kv.end())
    {
        std::cerr << "cpp_identifier_oracle: missing required field '" << key << "'\n";
        std::exit(2);
    }
    return it->second;
}

std::string identifier_from(const std::map<std::string, std::string>& kv)
{
    dispatcher::KernelKey key{};

    auto& sig = key.signature;
    sig.dtype_a             = dispatcher::string_to_dtype(require(kv, "dtype_a"));
    sig.dtype_b             = dispatcher::string_to_dtype(require(kv, "dtype_b"));
    sig.dtype_c             = dispatcher::string_to_dtype(require(kv, "dtype_c"));
    sig.dtype_acc           = dispatcher::string_to_dtype(require(kv, "dtype_acc"));
    sig.layout_a            = dispatcher::string_to_layout(require(kv, "layout_a"));
    sig.layout_b            = dispatcher::string_to_layout(require(kv, "layout_b"));
    sig.layout_c            = dispatcher::string_to_layout(require(kv, "layout_c"));
    sig.transpose_a         = to_bool(require(kv, "transpose_a"));
    sig.transpose_b         = to_bool(require(kv, "transpose_b"));
    sig.grouped             = to_bool(require(kv, "grouped"));
    sig.split_k             = static_cast<std::uint8_t>(std::stoi(require(kv, "split_k")));
    sig.elementwise_op      = require(kv, "elementwise_op");
    sig.num_d_tensors       = static_cast<std::uint8_t>(std::stoi(require(kv, "num_d_tensors")));
    sig.structured_sparsity = to_bool(require(kv, "structured_sparsity"));

    auto& alg              = key.algorithm;
    alg.tile_shape.m       = static_cast<std::uint16_t>(std::stoi(require(kv, "tile_m")));
    alg.tile_shape.n       = static_cast<std::uint16_t>(std::stoi(require(kv, "tile_n")));
    alg.tile_shape.k       = static_cast<std::uint16_t>(std::stoi(require(kv, "tile_k")));
    alg.wave_shape.m       = static_cast<std::uint8_t>(std::stoi(require(kv, "warp_m")));
    alg.wave_shape.n       = static_cast<std::uint8_t>(std::stoi(require(kv, "warp_n")));
    alg.wave_shape.k       = static_cast<std::uint8_t>(std::stoi(require(kv, "warp_k")));
    alg.warp_tile_shape.m  = static_cast<std::uint8_t>(std::stoi(require(kv, "warp_tile_m")));
    alg.warp_tile_shape.n  = static_cast<std::uint8_t>(std::stoi(require(kv, "warp_tile_n")));
    alg.warp_tile_shape.k  = static_cast<std::uint8_t>(std::stoi(require(kv, "warp_tile_k")));
    alg.pipeline           = dispatcher::string_to_pipeline(require(kv, "pipeline"));
    alg.scheduler          = dispatcher::string_to_scheduler(require(kv, "scheduler"));
    alg.epilogue           = dispatcher::string_to_epilogue(require(kv, "epilogue"));
    alg.block_size         = static_cast<std::uint16_t>(std::stoi(require(kv, "block_size")));
    alg.double_buffer      = to_bool(require(kv, "double_buffer"));
    alg.persistent         = to_bool(require(kv, "persistent"));
    alg.preshuffle         = to_bool(require(kv, "preshuffle"));
    alg.transpose_c        = to_bool(require(kv, "transpose_c"));
    alg.num_wave_groups    = static_cast<std::uint8_t>(std::stoi(require(kv, "num_wave_groups")));
    alg.pad_m              = to_bool(require(kv, "pad_m"));
    alg.pad_n              = to_bool(require(kv, "pad_n"));
    alg.pad_k              = to_bool(require(kv, "pad_k"));

    key.gfx_arch = require(kv, "gfx_arch");

    return key.encode_identifier();
}

} // namespace

int main()
{
    std::map<std::string, std::string> kv;
    std::string line;
    std::string out;

    auto flush = [&]() {
        if(!kv.empty())
        {
            out += identifier_from(kv);
            out += "\n";
            kv.clear();
        }
    };

    while(std::getline(std::cin, line))
    {
        if(line == "---")
        {
            flush();
            continue;
        }
        if(line.empty())
            continue;
        auto eq = line.find('=');
        if(eq == std::string::npos)
            continue;
        kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    flush();

    std::cout << out;
    return 0;
}
