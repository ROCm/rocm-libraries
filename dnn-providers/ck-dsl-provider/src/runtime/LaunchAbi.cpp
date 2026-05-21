// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "LaunchAbi.hpp"

#include <cstring>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <sstream>
#include <string>

namespace ck_dsl_provider {

namespace {

std::uint16_t naturalSize(ArgSchema::Kind kind) {
    switch (kind) {
        case ArgSchema::Kind::Pointer:
            return 8;
        case ArgSchema::Kind::I32:
            return 4;
        case ArgSchema::Kind::I64:
            return 8;
        case ArgSchema::Kind::F32:
            return 4;
        case ArgSchema::Kind::F16:
            return 2;
    }
    // Unreachable; the enum is exhaustive above. Throwing here keeps
    // the static checker quiet without inserting an unreachable
    // intrinsic the test build wouldn't exercise.
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "LaunchAbi::naturalSize: unhandled ArgSchema::Kind enumerator");
}

bool tagMatchesKind(ArgValue::Tag tag, ArgSchema::Kind kind) {
    switch (kind) {
        case ArgSchema::Kind::Pointer:
            return tag == ArgValue::Tag::Pointer;
        case ArgSchema::Kind::I32:
            return tag == ArgValue::Tag::I32;
        case ArgSchema::Kind::I64:
            return tag == ArgValue::Tag::I64;
        case ArgSchema::Kind::F32:
            return tag == ArgValue::Tag::F32;
        case ArgSchema::Kind::F16:
            return tag == ArgValue::Tag::F16;
    }
    return false;
}

void appendBytes(std::vector<std::byte>& buf, const void* src, std::size_t n) {
    const auto* p = static_cast<const std::byte*>(src);
    buf.insert(buf.end(), p, p + n);
}

void alignTo(std::vector<std::byte>& buf, std::size_t alignment) {
    if (alignment <= 1) {
        return;
    }
    std::size_t cur = buf.size();
    std::size_t rem = cur % alignment;
    if (rem != 0) {
        buf.insert(buf.end(), alignment - rem, std::byte{0});
    }
}

}  // namespace

std::vector<std::byte> LaunchAbi::pack(const std::vector<ArgSchema>& schema,
                                       const std::vector<ArgValue>& values) {
    if (schema.size() != values.size()) {
        std::ostringstream oss;
        oss << "LaunchAbi::pack: arg count mismatch: schema has " << schema.size()
            << " slots but caller supplied " << values.size() << " values";
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       oss.str());
    }

    std::vector<std::byte> buf;
    // Pre-reserve a small starting capacity to avoid the first few
    // reallocations; the actual size depends on the alignment padding
    // each slot demands, which is computed below.
    buf.reserve(64);

    for (std::size_t i = 0; i < schema.size(); ++i) {
        const ArgSchema& slot = schema[i];
        const ArgValue& val = values[i];

        std::uint16_t natural = naturalSize(slot.kind);
        if (slot.size != natural) {
            std::ostringstream oss;
            oss << "LaunchAbi::pack: schema slot " << i << " (name='" << slot.name
                << "') declares size " << slot.size << " but natural size for its kind is "
                << natural;
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           oss.str());
        }

        if (!tagMatchesKind(val.tag, slot.kind)) {
            std::ostringstream oss;
            oss << "LaunchAbi::pack: schema slot " << i << " (name='" << slot.name
                << "') expects kind tag " << static_cast<int>(slot.kind)
                << " but caller supplied value tag " << static_cast<int>(val.tag);
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           oss.str());
        }

        alignTo(buf, slot.align == 0 ? natural : slot.align);

        switch (slot.kind) {
            case ArgSchema::Kind::Pointer: {
                void* p = std::get<void*>(val.value);
                appendBytes(buf, &p, sizeof(p));
                break;
            }
            case ArgSchema::Kind::I32: {
                auto v = std::get<std::int32_t>(val.value);
                appendBytes(buf, &v, sizeof(v));
                break;
            }
            case ArgSchema::Kind::I64: {
                auto v = std::get<std::int64_t>(val.value);
                appendBytes(buf, &v, sizeof(v));
                break;
            }
            case ArgSchema::Kind::F32: {
                float v = std::get<float>(val.value);
                appendBytes(buf, &v, sizeof(v));
                break;
            }
            case ArgSchema::Kind::F16: {
                auto v = std::get<std::uint16_t>(val.value);
                appendBytes(buf, &v, sizeof(v));
                break;
            }
        }
    }

    return buf;
}

}  // namespace ck_dsl_provider
