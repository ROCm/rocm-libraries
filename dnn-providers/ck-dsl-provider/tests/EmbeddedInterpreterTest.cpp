// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstring>
#include <mutex>

#include "python/EmbeddedInterpreter.hpp"

extern "C" {
#include "py/runtime.h"
}

using ck_dsl_provider::EmbeddedInterpreter;

namespace {

// Import a module by name under the interpreter lock (no GIL in MicroPython),
// resetting the GC stack top to this frame the same way the bridge does.
// Returns true if the import succeeds, false if MicroPython raised.
bool importSucceeds(const char* name) {
    std::lock_guard<std::mutex> lock(EmbeddedInterpreter::interpreterMutex());
    EmbeddedInterpreter::setCallStackTop(__builtin_frame_address(0));
    nlr_buf_t nlr;
    bool ok = false;
    if (nlr_push(&nlr) == 0) {
        mp_obj_t module =
            mp_import_name(qstr_from_str(name), mp_const_none, MP_OBJ_NEW_SMALL_INT(0));
        ok = (module != MP_OBJ_NULL);
        nlr_pop();
    } else {
        ok = false;
    }
    return ok;
}

}  // namespace

// The std::call_once init runs once across the suite; the first test to run
// performs initialization, subsequent tests observe initializationCount() == 1
// and isInitialized() == true.
TEST(TestEmbeddedInterpreter, InitializesOnce) {
    const unsigned before = EmbeddedInterpreter::initializationCount();
    EmbeddedInterpreter::ensureInitialized();
    ASSERT_TRUE(EmbeddedInterpreter::isInitialized());

    const unsigned afterFirst = EmbeddedInterpreter::initializationCount();
    EXPECT_LE(afterFirst, 1u) << "init counter should never exceed 1";
    if (before == 0u) {
        EXPECT_EQ(afterFirst, 1u) << "first call must have actually initialised the interpreter";
    }

    // Second call must be a no-op: counter stays put.
    EmbeddedInterpreter::ensureInitialized();
    EXPECT_EQ(EmbeddedInterpreter::initializationCount(), afterFirst);
    EXPECT_TRUE(EmbeddedInterpreter::isInitialized());
}

// The provider's Python ships frozen into the plugin: importing it proves the
// frozen module table is linked in and the import machinery resolves it without
// any filesystem / sys.path.
TEST(TestEmbeddedInterpreter, ImportsFrozenProviderPackage) {
    EmbeddedInterpreter::ensureInitialized();
    EXPECT_TRUE(importSucceeds("ck_dsl_provider"))
        << "frozen ck_dsl_provider package must be importable";
}

// A builtin module imports through the same locked path -- and a second import
// after the first released the lock confirms the mutex is reusable in sequence.
TEST(TestEmbeddedInterpreter, ImportsBuiltinModule) {
    EmbeddedInterpreter::ensureInitialized();
    EXPECT_TRUE(importSucceeds("sys"));
    EXPECT_TRUE(importSucceeds("sys")) << "interpreter lock must be reusable across calls";
}

// Driving the native comgr module with invalid LLVM IR must raise cleanly
// (no crash) and free its resources -- exercises comgr_build_hsaco's error path
// + modcomgr's error reporting. Under ASAN this also guards against the comgr /
// modcomgr error-path leaks. comgr cross-compiles, so no GPU is needed.
TEST(TestEmbeddedInterpreter, ComgrRejectsInvalidIr) {
    EmbeddedInterpreter::ensureInitialized();
    std::lock_guard<std::mutex> lock(EmbeddedInterpreter::interpreterMutex());
    EmbeddedInterpreter::setCallStackTop(__builtin_frame_address(0));

    const char* badIr = "this is not valid llvm ir";
    const char* isa = "amdgcn-amd-amdhsa--gfx950";

    bool raised = false;
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_obj_t comgr =
            mp_import_name(qstr_from_str("comgr"), mp_const_none, MP_OBJ_NEW_SMALL_INT(0));
        mp_obj_t fn = mp_load_attr(comgr, qstr_from_str("build_hsaco"));
        mp_obj_t args[3];
        args[0] = mp_obj_new_str(badIr, std::strlen(badIr));
        args[1] = mp_obj_new_str(isa, std::strlen(isa));
        args[2] = mp_obj_new_list(0, nullptr);
        (void)mp_call_function_n_kw(fn, 3, 0, args);
        nlr_pop();  // reached only if comgr unexpectedly succeeded
    } else {
        raised = true;
    }
    EXPECT_TRUE(raised) << "comgr.build_hsaco must raise on invalid IR";
}
