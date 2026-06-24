// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/headers.hpp"
#include "ck/host/ck_tile_headers_preprocessor.hpp"
#include "ck/host/rtc_std_shims.hpp"
#include "ck_headers.hpp"
#include "ck_tile_headers.hpp"
#include "ck_codegen_headers.hpp"
#include "ck_rocm_cxx_headers.hpp"

namespace ck {
namespace host {

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif
const std::string config_header = "";
#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif

std::unordered_map<std::string_view, std::string_view> GetHeaders()
{
    auto headers = ck_headers();
    headers.insert(std::make_pair("ck/config.h", config_header));
    return headers;
}

std::unordered_map<std::string, std::string> GetTileHeaders()
{
    auto tile_hdrs    = ck_tile_headers();
    auto codegen_hdrs = ck_codegen_headers();

    std::unordered_map<std::string, std::string> result;
    result.reserve(tile_hdrs.size() + codegen_hdrs.size());

    for(auto& [name, content] : tile_hdrs)
    {
        if(name == "ck_tile/core/utility/env.hpp" ||
           name == "ck_tile/core/utility/gemm_validation.hpp")
        {
            // env.hpp: host-only env parsing. gemm_validation.hpp: host-only
            // std::string/runtime_error validation helpers, used nowhere on the
            // device path. Both are unusable (and unneeded) under hipRTC.
            result.emplace(std::string(name), "");
            continue;
        }
        result.emplace(std::string(name), strip_host_bodies(content));
    }

    for(auto& [name, content] : codegen_hdrs)
        result.emplace(std::string(name), std::string(content));

    // rocm-cxx standard library: expose under include names "rocm/..." by
    // remapping the embedded "rocm-cxx/" prefix.
    constexpr std::string_view rocm_cxx_prefix = "rocm-cxx/";
    for(auto& [name, content] : ck_rocm_cxx_headers())
    {
        std::string key(name);
        if(key.rfind(rocm_cxx_prefix, 0) == 0)
            key = "rocm/" + key.substr(rocm_cxx_prefix.size());
        result.emplace(std::move(key), std::string(content));
    }

    // Standard-header-named shims that bridge ck_tile's std:: usage onto
    // rocm-cxx (or provide minimal/stub definitions) so no system C++ headers
    // are consulted under hipRTC.
    for(const auto& [name, content] : GetRtcStdShims())
        result.emplace(name, content);

    // HIP headers unavailable under hipRTC. The device runtime is provided
    // implicitly, and ck_tile only needs the include to resolve, so empty
    // stubs are sufficient.
    result.emplace("hip/hip_bfloat16.h", "");

    return result;
}

} // namespace host
} // namespace ck
