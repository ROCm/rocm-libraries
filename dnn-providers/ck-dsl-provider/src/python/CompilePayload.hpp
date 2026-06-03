// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <utility>
#include <vector>

namespace ck_dsl_provider {

/// Interpreter-neutral compile payload.
///
/// Mirrors the Python dict that ``ck_dsl_provider.compile_service.compile`` /
/// ``is_applicable`` consume, but carries no interpreter type. Adapters (e.g.
/// ``convImplicitGemmSpecToPayload``) build a ``PayloadDict`` with plain C++;
/// ``CompileServiceBridge`` marshals it to an ``mp_obj_t`` dict in one place,
/// under the interpreter lock. This decouples the per-op adapters from the
/// embedded interpreter (previously they built ``py::dict`` directly).
struct PayloadValue;

/// Ordered list of key->value pairs (insertion order preserved; the Python side
/// splats by keyword, so order is not semantically required but kept stable).
using PayloadDict = std::vector<std::pair<std::string, PayloadValue>>;

struct PayloadValue {
    enum class Kind { Int, Bool, Str, None, Dict };

    Kind kind = Kind::None;
    long long intVal = 0;
    bool boolVal = false;
    std::string strVal;
    PayloadDict dictVal;

    static PayloadValue ofInt(long long v) {
        PayloadValue x;
        x.kind = Kind::Int;
        x.intVal = v;
        return x;
    }
    static PayloadValue ofBool(bool v) {
        PayloadValue x;
        x.kind = Kind::Bool;
        x.boolVal = v;
        return x;
    }
    static PayloadValue ofStr(std::string v) {
        PayloadValue x;
        x.kind = Kind::Str;
        x.strVal = std::move(v);
        return x;
    }
    static PayloadValue ofNone() {
        return PayloadValue{};
    }
    static PayloadValue ofDict(PayloadDict d) {
        PayloadValue x;
        x.kind = Kind::Dict;
        x.dictVal = std::move(d);
        return x;
    }
};

}  // namespace ck_dsl_provider
