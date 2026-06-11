// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Parsed ck_dsl manifest (`ck.dsl.example.manifest/v1`). This is the portable
// ABI boundary the C++ runtime consumes: kernel entry name, argument signature
// (for kernarg packing), and grid/block geometry.
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "ck_dsl_runtime/json.hpp"

namespace ck_dsl {

// One kernel argument as declared by the kernel's ABI. `type` is one of
//   "ptr<f16, global>", "ptr<bf16, global>", "ptr<f32, global>", "ptr<i32, global>",
//   "i32", "i64", "f32"
// `size_bytes` is the packed width (pointers 8, scalars match their width).
struct ArgSpec {
    std::string name;
    std::string type;
    int size_bytes = 0;
    bool is_pointer() const {
        return type.rfind("ptr<", 0) == 0;
    }
    // Packed/aligned width. Some signatures (e.g. attention) omit size_bytes;
    // derive it from the type: pointers and i64 are 8 bytes, i32/f32 are 4.
    int width() const {
        if (size_bytes > 0) return size_bytes;
        if (is_pointer() || type == "i64") return 8;
        return 4;  // i32, f32
    }
};

struct Manifest {
    std::string schema;       // "ck.dsl.example.manifest/v1"
    std::string kind;         // "gemm_fp16", "conv_fp16", "attention_unified", ...
    std::string kernel_name;  // == the HSACO entry symbol == hipModuleGetFunction name
    std::string hsaco;        // basename of the .hsaco (if shipped alongside)
    std::string cache_key;    // stable identity (optional; falls back to kernel_name)
    int threads_per_block = 256;

    // Grid geometry. Either grid_explicit is set, or (block_m/n + grid_order)
    // drive ceil-div tiling from the problem dims.
    int block_m = 0, block_n = 0, block_k = 0;
    std::string grid_order = "MN";  // "MN" or "NM"
    std::optional<std::array<int, 3>> grid_explicit;

    std::vector<ArgSpec> args_signature;
    bool sig_has_bytes = false;

    // The full parsed JSON, retained so engine-specific code can read per-kind
    // extras (conv[13], default_shape, attention_config, ...) without re-parsing.
    json::Value raw;

    std::string id() const {
        return cache_key.empty() ? kernel_name : cache_key;
    }

    static Manifest from_json(const json::Value& v) {
        Manifest m;
        m.raw = v;
        m.schema = v.get_str("schema");
        m.kind = v.get_str("kind");
        m.kernel_name = v.get_str("kernel_name");
        m.hsaco = v.get_str("hsaco");
        m.cache_key = v.get_str("cache_key");
        m.threads_per_block = static_cast<int>(v.get_int("threads_per_block", 256));
        m.block_m = static_cast<int>(v.get_int("block_m", 0));
        m.block_n = static_cast<int>(v.get_int("block_n", 0));
        m.block_k = static_cast<int>(v.get_int("block_k", 0));
        m.grid_order = v.get_str("grid_order", "MN");
        m.sig_has_bytes = v.get_int("sig_has_bytes", 0) != 0;
        if (v.has("grid_explicit") && v.at("grid_explicit").is_array()) {
            const auto& a = v.at("grid_explicit").as_array();
            if (a.size() == 3)
                m.grid_explicit = std::array<int, 3>{static_cast<int>(a[0].as_int()),
                                                     static_cast<int>(a[1].as_int()),
                                                     static_cast<int>(a[2].as_int())};
        }
        if (v.has("args_signature")) {
            for (const auto& e : v.at("args_signature").as_array()) {
                ArgSpec a;
                a.name = e.get_str("name");
                a.type = e.get_str("type");
                a.size_bytes = static_cast<int>(e.get_int("size_bytes", 0));
                m.args_signature.push_back(std::move(a));
            }
        }
        return m;
    }

    static Manifest parse(const std::string& json_text) {
        return from_json(json::parse(json_text));
    }
};

}  // namespace ck_dsl
