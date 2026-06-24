// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <unordered_map>

namespace ck {
namespace host {

// Returns standard-header-named shims used at hipRTC compile time. Each entry
// maps a standard include name (e.g. "type_traits", "cstdint") to header
// content that either bridges the name into namespace std from the in-repo
// rocm-cxx library, provides a minimal self-contained definition, or is an
// empty/forward-declaration stub for host-only headers whose bodies are
// stripped before embedding.
const std::unordered_map<std::string, std::string>& GetRtcStdShims();

} // namespace host
} // namespace ck
