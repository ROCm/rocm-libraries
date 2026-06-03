// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CompileServiceBridge.hpp"

#include <cstring>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <mutex>
#include <string>
#include <vector>

#include "EmbeddedInterpreter.hpp"

extern "C" {
#include "py/builtin.h"
#include "py/objstr.h"
#include "py/runtime.h"
}

// ---------------------------------------------------------------------------
// nlr / C++ RAII discipline
//
// MicroPython signals errors with setjmp/longjmp (nlr), which does NOT unwind
// C++ stack frames -- any object with a non-trivial destructor created between
// nlr_push() and the failure point would leak / be skipped. Therefore every
// nlr-protected region below uses ONLY POD/raw locals (mp_obj_t handles, const
// char*, fixed arrays). All in-region errors (missing field, bad type) are
// raised as MicroPython exceptions (mp_raise_*) and caught by the nlr handler,
// which converts them to HipdnnPluginException AFTER nlr_pop (a normal C++
// throw, no nlr frame active). C++ objects (std::string / KernelArtifact) are
// built only after nlr_pop, from the raw locals (valid while the lock is held).
// ---------------------------------------------------------------------------

namespace ck_dsl_provider {

namespace {

mp_obj_t marshalValue(const PayloadValue& v);  // fwd

mp_obj_t marshalDict(const PayloadDict& d) {
    mp_obj_t obj = mp_obj_new_dict(d.size());
    for (const auto& kv : d) {
        mp_obj_dict_store(obj, mp_obj_new_str(kv.first.data(), kv.first.size()),
                          marshalValue(kv.second));
    }
    return obj;
}

mp_obj_t marshalValue(const PayloadValue& v) {
    switch (v.kind) {
        case PayloadValue::Kind::Int:
            return mp_obj_new_int_from_ll(v.intVal);
        case PayloadValue::Kind::Bool:
            return mp_obj_new_bool(v.boolVal);
        case PayloadValue::Kind::Str:
            return mp_obj_new_str(v.strVal.data(), v.strVal.size());
        case PayloadValue::Kind::None:
            return mp_const_none;
        case PayloadValue::Kind::Dict:
            return marshalDict(v.dictVal);
    }
    return mp_const_none;
}

// Non-raising dict lookup by string key. Returns MP_OBJ_NULL if absent.
mp_obj_t dictGet(mp_obj_t dict, const char* key) {
    mp_map_t* map = mp_obj_dict_get_map(dict);
    mp_map_elem_t* e = mp_map_lookup(map, mp_obj_new_str(key, strlen(key)), MP_MAP_LOOKUP);
    return (e == nullptr) ? MP_OBJ_NULL : e->value;
}

// Required dict lookup; raises a MicroPython KeyError (caught by the nlr
// handler) if the key is absent.
mp_obj_t dictGetReq(mp_obj_t dict, const char* key) {
    mp_obj_t v = dictGet(dict, key);
    if (v == MP_OBJ_NULL) {
        mp_raise_msg_varg(&mp_type_KeyError, MP_ERROR_TEXT("compile-service result missing '%s'"),
                          key);
    }
    return v;
}

void getUint3(mp_obj_t tup, std::uint32_t out[3]) {
    size_t n = 0;
    mp_obj_t* items = nullptr;
    mp_obj_get_array(tup, &n, &items);
    if (n != 3) {
        mp_raise_ValueError(MP_ERROR_TEXT("expected a 3-tuple (grid/block)"));
    }
    for (size_t i = 0; i < 3; ++i) {
        out[i] = static_cast<std::uint32_t>(mp_obj_get_int(items[i]));
    }
}

// POD capture of one arg_schema slot (raw pointers into MicroPython str storage,
// valid until the next GC / interpreter call; we copy out before then).
struct RawArg {
    const char* name;
    size_t nameLen;
    const char* kind;
    size_t kindLen;
    long size;
    long align;
    bool hasSize;
    bool hasAlign;
};

constexpr size_t kMaxArgs = 64;

// Format a MicroPython exception into a fixed buffer (no C++ objects in the
// nlr-protected region) and throw HipdnnPluginException. Never returns.
[[noreturn]] void raiseFromMpException(mp_obj_t exc, const char* context) {
    char buf[512];
    buf[0] = '\0';
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        vstr_t vstr;
        mp_print_t print;
        vstr_init_print(&vstr, 200, &print);
        mp_obj_print_helper(&print, exc, PRINT_EXC);
        size_t len = vstr_len(&vstr);
        if (len >= sizeof(buf)) {
            len = sizeof(buf) - 1;
        }
        std::memcpy(buf, vstr_str(&vstr), len);
        buf[len] = '\0';
        vstr_clear(&vstr);
        nlr_pop();
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        std::string(context) + ": " + (buf[0] != '\0' ? buf : "<python exception>"));
}

constexpr std::size_t kHsacoMaxBytes = 256ULL * 1024 * 1024;

}  // namespace

CompileServiceBridge::CompileServiceBridge() {
    EmbeddedInterpreter::ensureInitialized();
    std::lock_guard<std::mutex> lock(EmbeddedInterpreter::interpreterMutex());

    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        // Import the package (its __init__ imports .compile_service) then fetch
        // the submodule. ck_dsl is frozen/bundled, so no sys.path manipulation.
        mp_obj_t pkg = mp_import_name(qstr_from_str("ck_dsl_provider"), mp_const_none,
                                      MP_OBJ_NEW_SMALL_INT(0));
        mp_obj_t mod = mp_load_attr(pkg, qstr_from_str("compile_service"));
        _module = reinterpret_cast<void*>(mod);  // kept alive by loaded-modules dict
        nlr_pop();
    } else {
        raiseFromMpException(MP_OBJ_FROM_PTR(nlr.ret_val), "CompileServiceBridge::ctor");
    }
    HIPDNN_PLUGIN_LOG_INFO("CompileServiceBridge: imported frozen ck_dsl_provider.compile_service");
}

CompileServiceBridge::~CompileServiceBridge() noexcept = default;

namespace {

// Run `fn(args...)` returning a dict, and marshal it into a KernelArtifact.
// `fn`/`args` are produced under the same nlr region. Caller holds the lock.
KernelArtifact callCompileLike(mp_obj_t module, const char* attr, const mp_obj_t* args,
                               size_t nArgs, const char* context) {
    // --- nlr-protected region: POD locals only ---
    const char* kernelName = nullptr;
    size_t kernelNameLen = 0;
    const char* kind = nullptr;
    size_t kindLen = 0;
    const char* isa = nullptr;
    size_t isaLen = 0;
    bool hasIsa = false;
    const char* hsaco = nullptr;
    size_t hsacoLen = 0;
    std::uint32_t grid[3] = {0, 0, 0};
    std::uint32_t block[3] = {0, 0, 0};
    std::uint32_t ldsBytes = 0;
    RawArg rawArgs[kMaxArgs];
    size_t nRawArgs = 0;

    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_obj_t fn = mp_load_attr(module, qstr_from_str(attr));
        mp_obj_t result = mp_call_function_n_kw(fn, nArgs, 0, args);

        hsaco = mp_obj_str_get_data(dictGetReq(result, "hsaco"), &hsacoLen);
        kernelName = mp_obj_str_get_data(dictGetReq(result, "kernel_name"), &kernelNameLen);
        kind = mp_obj_str_get_data(dictGetReq(result, "kind"), &kindLen);
        getUint3(dictGetReq(result, "grid"), grid);
        getUint3(dictGetReq(result, "block"), block);
        ldsBytes = static_cast<std::uint32_t>(mp_obj_get_int(dictGetReq(result, "lds_bytes")));

        mp_obj_t isaObj = dictGet(result, "isa");
        if (isaObj != MP_OBJ_NULL) {
            isa = mp_obj_str_get_data(isaObj, &isaLen);
            hasIsa = true;
        }

        size_t n = 0;
        mp_obj_t* items = nullptr;
        mp_obj_get_array(dictGetReq(result, "arg_schema"), &n, &items);
        if (n > kMaxArgs) {
            mp_raise_ValueError(MP_ERROR_TEXT("arg_schema too large"));
        }
        for (size_t i = 0; i < n; ++i) {
            mp_obj_t e = items[i];
            RawArg& a = rawArgs[i];
            mp_obj_t nameObj = dictGet(e, "name");
            a.name = (nameObj != MP_OBJ_NULL) ? mp_obj_str_get_data(nameObj, &a.nameLen) : "";
            if (nameObj == MP_OBJ_NULL) {
                a.nameLen = 0;
            }
            a.kind = mp_obj_str_get_data(dictGetReq(e, "kind"), &a.kindLen);
            mp_obj_t sizeObj = dictGet(e, "size");
            a.hasSize = (sizeObj != MP_OBJ_NULL);
            a.size = a.hasSize ? mp_obj_get_int(sizeObj) : 0;
            mp_obj_t alignObj = dictGet(e, "align");
            a.hasAlign = (alignObj != MP_OBJ_NULL);
            a.align = a.hasAlign ? mp_obj_get_int(alignObj) : 0;
        }
        nRawArgs = n;
        nlr_pop();
    } else {
        raiseFromMpException(MP_OBJ_FROM_PTR(nlr.ret_val), context);
    }
    // --- nlr region done; safe to build C++ objects from the raw captures ---

    if (hsacoLen > kHsacoMaxBytes) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string(context) + ": HSACO blob exceeds size cap");
    }

    KernelArtifact artifact;
    artifact.kernelName.assign(kernelName, kernelNameLen);
    artifact.kind.assign(kind, kindLen);
    if (hasIsa) {
        artifact.isa.assign(isa, isaLen);
    }
    artifact.grid = KernelArtifact::GridSpec{grid[0], grid[1], grid[2]};
    artifact.block = KernelArtifact::BlockSpec{block[0], block[1], block[2]};
    artifact.ldsBytes = ldsBytes;
    artifact.hsaco.resize(hsacoLen);
    if (hsacoLen > 0) {
        std::memcpy(artifact.hsaco.data(), hsaco, hsacoLen);
    }
    artifact.argSchema.reserve(nRawArgs);
    for (size_t i = 0; i < nRawArgs; ++i) {
        ArgSchema slot;
        slot.name.assign(rawArgs[i].name, rawArgs[i].nameLen);
        slot.kind = parseArgKind(std::string(rawArgs[i].kind, rawArgs[i].kindLen));
        if (rawArgs[i].hasSize) {
            slot.size = static_cast<std::uint16_t>(rawArgs[i].size);
        }
        if (rawArgs[i].hasAlign) {
            slot.align = static_cast<std::uint16_t>(rawArgs[i].align);
        }
        artifact.argSchema.push_back(std::move(slot));
    }
    return artifact;
}

}  // namespace

KernelArtifact CompileServiceBridge::compileSmoke(std::string_view arch) {
    std::lock_guard<std::mutex> lock(EmbeddedInterpreter::interpreterMutex());
    mp_obj_t module = reinterpret_cast<mp_obj_t>(_module);
    mp_obj_t args[1];
    nlr_buf_t nlr;
    KernelArtifact artifact;
    if (nlr_push(&nlr) == 0) {
        args[0] = mp_obj_new_str(arch.data(), arch.size());
        nlr_pop();
    } else {
        raiseFromMpException(MP_OBJ_FROM_PTR(nlr.ret_val), "CompileServiceBridge::compileSmoke");
    }
    artifact =
        callCompileLike(module, "compile_smoke", args, 1, "CompileServiceBridge::compileSmoke");
    HIPDNN_PLUGIN_LOG_INFO("CompileServiceBridge::compileSmoke arch='"
                           << std::string(arch) << "' kernel='" << artifact.kernelName
                           << "' hsaco_bytes=" << artifact.hsaco.size());
    return artifact;
}

KernelArtifact CompileServiceBridge::compile(std::string_view opKind, const PayloadDict& payload,
                                             std::string_view arch) {
    std::lock_guard<std::mutex> lock(EmbeddedInterpreter::interpreterMutex());
    mp_obj_t module = reinterpret_cast<mp_obj_t>(_module);
    mp_obj_t args[3];
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        args[0] = mp_obj_new_str(opKind.data(), opKind.size());
        args[1] = marshalDict(payload);
        args[2] = mp_obj_new_str(arch.data(), arch.size());
        nlr_pop();
    } else {
        raiseFromMpException(MP_OBJ_FROM_PTR(nlr.ret_val), "CompileServiceBridge::compile");
    }
    KernelArtifact artifact =
        callCompileLike(module, "compile", args, 3, "CompileServiceBridge::compile");
    HIPDNN_PLUGIN_LOG_INFO("CompileServiceBridge::compile op_kind='"
                           << std::string(opKind) << "' kernel='" << artifact.kernelName
                           << "' hsaco_bytes=" << artifact.hsaco.size());
    return artifact;
}

std::pair<bool, std::string> CompileServiceBridge::isApplicable(std::string_view opKind,
                                                                const PayloadDict& payload,
                                                                std::string_view arch) {
    std::lock_guard<std::mutex> lock(EmbeddedInterpreter::interpreterMutex());
    mp_obj_t module = reinterpret_cast<mp_obj_t>(_module);

    bool ok = false;
    const char* reasonPtr = nullptr;
    size_t reasonLen = 0;

    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_obj_t args[3] = {mp_obj_new_str(opKind.data(), opKind.size()), marshalDict(payload),
                            mp_obj_new_str(arch.data(), arch.size())};
        mp_obj_t fn = mp_load_attr(module, qstr_from_str("is_applicable"));
        mp_obj_t result = mp_call_function_n_kw(fn, 3, 0, args);

        size_t n = 0;
        mp_obj_t* items = nullptr;
        mp_obj_get_array(result, &n, &items);
        if (n != 2) {
            mp_raise_ValueError(MP_ERROR_TEXT("is_applicable must return (ok, reason)"));
        }
        ok = mp_obj_is_true(items[0]);
        reasonPtr = mp_obj_str_get_data(items[1], &reasonLen);
        nlr_pop();
    } else {
        raiseFromMpException(MP_OBJ_FROM_PTR(nlr.ret_val), "CompileServiceBridge::isApplicable");
    }
    return {ok, std::string(reasonPtr, reasonLen)};
}

}  // namespace ck_dsl_provider
