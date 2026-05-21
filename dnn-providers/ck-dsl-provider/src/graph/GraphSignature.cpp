// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GraphSignature.hpp"

#include <cstdint>
#include <cstring>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <sstream>
#include <string>
#include <string_view>

#include "version.h"

namespace ck_dsl_provider {

namespace {

constexpr std::uint64_t kFnv1aOffset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kFnv1aPrime = 0x100000001b3ULL;

inline std::uint64_t fnv1aFold(std::uint64_t h, std::uint8_t byte) {
    return (h ^ static_cast<std::uint64_t>(byte)) * kFnv1aPrime;
}

inline std::uint64_t fnv1aBytes(std::uint64_t h, const void* data, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h = fnv1aFold(h, p[i]);
    }
    return h;
}

inline std::uint64_t fnv1aString(std::uint64_t h, std::string_view s) {
    return fnv1aBytes(h, s.data(), s.size());
}

inline std::uint64_t fnv1aI64(std::uint64_t h, std::int64_t v) {
    return fnv1aBytes(h, &v, sizeof(v));
}

inline std::uint64_t fnv1aI32(std::uint64_t h, std::int32_t v) {
    return fnv1aBytes(h, &v, sizeof(v));
}

[[noreturn]] void badParam(const std::string& msg) {
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "GraphSignature: " + msg);
}

const GraphSignature::TensorAttributes& lookupTensor(const GraphSignature::TensorMap& tensorMap,
                                                     std::int64_t uid, const char* role) {
    auto it = tensorMap.find(uid);
    if (it == tensorMap.end() || it->second == nullptr) {
        std::ostringstream oss;
        oss << "tensor map missing entry for " << role << " uid=" << uid;
        badParam(oss.str());
    }
    return *it->second;
}

void check4dDims(const GraphSignature::TensorAttributes& t, const char* role) {
    if (t.dims() == nullptr || t.dims()->size() != 4) {
        std::ostringstream oss;
        oss << role << " dims must be 4-D; got size "
            << (t.dims() == nullptr ? 0u : t.dims()->size());
        badParam(oss.str());
    }
}

void checkSpatialAttr(const flatbuffers::Vector<std::int64_t>* attr, const char* name) {
    if (attr == nullptr || attr->size() != 2) {
        std::ostringstream oss;
        oss << "conv attribute '" << name << "' must be a 2-element vector (2-D conv); got size "
            << (attr == nullptr ? 0u : attr->size());
        badParam(oss.str());
    }
}

}  // namespace

SignatureHash GraphSignature::computeForConvFwd(std::string_view opKind,
                                                const ConvolutionFwdAttributes& convAttr,
                                                const TensorMap& tensorMap) {
    const auto& X = lookupTensor(tensorMap, convAttr.x_tensor_uid(), "X");
    const auto& W = lookupTensor(tensorMap, convAttr.w_tensor_uid(), "W");
    const auto& Y = lookupTensor(tensorMap, convAttr.y_tensor_uid(), "Y");

    check4dDims(X, "X");
    check4dDims(W, "W");
    check4dDims(Y, "Y");
    checkSpatialAttr(convAttr.pre_padding(), "pre_padding");
    checkSpatialAttr(convAttr.stride(), "stride");
    checkSpatialAttr(convAttr.dilation(), "dilation");

    std::uint64_t h = kFnv1aOffset;

    // Provider/DSL version string. Fold the entire macro contents
    // (including the git SHA suffix) so any DSL or provider change
    // bumps the namespace. Using the C string literal keeps the
    // dependency at compile time -- no need to thread the version
    // through the signature inputs at runtime.
    h = fnv1aString(h, CK_DSL_PROVIDER_VERSION_STRING);

    // Separator byte. Defensive against accidental aliasing if a
    // future input happens to abut a numerically-identical version
    // suffix.
    h = fnv1aFold(h, 0x00);

    h = fnv1aString(h, opKind);
    h = fnv1aFold(h, 0x00);

    // Dtype trio. Encoded as the raw enum value (single i32) per
    // tensor so a dtype change (HALF -> FLOAT, etc.) gives a
    // different hash even if the shape is unchanged.
    h = fnv1aI32(h, static_cast<std::int32_t>(X.data_type()));
    h = fnv1aI32(h, static_cast<std::int32_t>(W.data_type()));
    h = fnv1aI32(h, static_cast<std::int32_t>(Y.data_type()));

    // Shape trio. Fold all four logical dims per tensor (NCHW order
    // for X/Y, KCRS for W). We include Y's dims even though they're
    // derivable from X/W + conv attrs -- a malformed graph where Y is
    // a different shape than the conv arithmetic predicts should miss
    // the cache and not collide with a well-formed graph.
    for (std::uint32_t i = 0; i < 4; ++i) {
        h = fnv1aI64(h, X.dims()->Get(i));
    }
    for (std::uint32_t i = 0; i < 4; ++i) {
        h = fnv1aI64(h, W.dims()->Get(i));
    }
    for (std::uint32_t i = 0; i < 4; ++i) {
        h = fnv1aI64(h, Y.dims()->Get(i));
    }

    // Conv knobs. Padding/stride/dilation are 2-element vectors per
    // ``checkSpatialAttr``; post_padding is folded as a defense in
    // depth so an asymmetric-padding regression hashes differently
    // (the adapter would reject it, but the cache shouldn't return a
    // symmetric-padding kernel from a similar-looking key).
    if (convAttr.post_padding() != nullptr) {
        for (std::uint32_t i = 0; i < convAttr.post_padding()->size(); ++i) {
            h = fnv1aI64(h, convAttr.post_padding()->Get(i));
        }
    }
    h = fnv1aFold(h, 0x00);
    for (std::uint32_t i = 0; i < 2; ++i) {
        h = fnv1aI64(h, convAttr.pre_padding()->Get(i));
    }
    h = fnv1aFold(h, 0x00);
    for (std::uint32_t i = 0; i < 2; ++i) {
        h = fnv1aI64(h, convAttr.stride()->Get(i));
    }
    h = fnv1aFold(h, 0x00);
    for (std::uint32_t i = 0; i < 2; ++i) {
        h = fnv1aI64(h, convAttr.dilation()->Get(i));
    }
    h = fnv1aFold(h, 0x00);

    h = fnv1aI32(h, static_cast<std::int32_t>(convAttr.conv_mode()));

    return static_cast<SignatureHash>(h);
}

}  // namespace ck_dsl_provider
