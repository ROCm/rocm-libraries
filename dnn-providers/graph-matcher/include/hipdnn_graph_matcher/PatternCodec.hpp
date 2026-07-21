// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// PatternCodec: the flat wire form of a CompiledPattern. It converges the two
// ingestion paths of plan §8 -- the build-time (AOT) path serializes patterns
// compiled once at build and embeds the bytes; the runtime path compiles JSON at
// load. Both produce a CompiledPattern, and `serialize` renders it to a compact,
// versioned, little-endian byte blob that `deserialize` restores with no JSON
// parse and no re-run of the semantic compiler.
//
// The format is self-describing (magic + format version) and deterministic (an
// interned string pool built in a fixed order), so identical patterns yield
// identical bytes -- the property golden-bytes tests rely on. `deserialize` is
// the third untrusted-input parser (installed AOT bundles, drop-in caches), so
// it is fully bounded and fails closed on any truncation, bad offset, or index.

#include <cstdint>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace hipdnn::graph_matcher {

// Current wire-format version. Bumped on any incompatible layout change;
// deserialize refuses a blob whose version it does not understand.
inline constexpr uint16_t kPatternWireVersion = 1;

struct DeserializeResult {
    bool ok = false;
    CompiledPattern pattern;
    std::string error;

    explicit operator bool() const noexcept {
        return ok;
    }
};

struct PatternCodec {
    // Renders `pattern` to a self-describing, deterministic byte blob. Pure;
    // never fails for a well-formed pattern.
    static std::vector<uint8_t> serialize(const CompiledPattern& pattern);

    // Restores a CompiledPattern from bytes produced by serialize(). Treats the
    // input as untrusted: validates magic/version/endianness and every count,
    // offset, string length, and id against the buffer, failing closed with a
    // message. Does not throw.
    static DeserializeResult deserialize(const uint8_t* data, size_t size);

    static DeserializeResult deserialize(const std::vector<uint8_t>& bytes) {
        return deserialize(bytes.data(), bytes.size());
    }

    // Emits `bytes` as a C++ source snippet defining
    // `static constexpr unsigned char <symbol>[] = { ... };` plus a
    // `<symbol>_size`, for build-time (AOT) embedding of compiled criteria.
    static std::string emitEmbeddedArray(std::string_view symbol,
                                         const std::vector<uint8_t>& bytes);
};

}  // namespace hipdnn::graph_matcher
