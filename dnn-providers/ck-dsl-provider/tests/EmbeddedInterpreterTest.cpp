// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <string>

#include "ckdsl_provider_paths.h"
#include "python/EmbeddedInterpreter.hpp"

namespace py = pybind11;
using ck_dsl_provider::EmbeddedInterpreter;

namespace {
bool startsWith(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}
}  // namespace

// All three tests live in one TestSuite so the call_once init runs once
// across the suite while still letting each TEST verify a separate
// behaviour. The std::call_once contract means that the very first test
// to run is the one that performs initialization; subsequent tests
// observe initializationCount() == 1 and Py_IsInitialized() == true.
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

TEST(TestEmbeddedInterpreter, CanImportStdlib) {
    EmbeddedInterpreter::ensureInitialized();

    // Acquire the GIL once and drive the import + a trivial json
    // roundtrip; mirrors the spike at WIP/pybind11_rtld_local_spike/
    // and proves that the embedded interpreter's sys.path resolves
    // CPython's stdlib correctly.
    py::gil_scoped_acquire gil;
    py::module_ json = py::module_::import("json");
    ASSERT_FALSE(json.is_none());

    py::dict payload;
    payload["ok"] = 1;
    auto encoded = json.attr("dumps")(payload).cast<std::string>();
    EXPECT_NE(encoded.find("\"ok\""), std::string::npos);
    EXPECT_NE(encoded.find('1'), std::string::npos);
}

TEST(TestEmbeddedInterpreter, SurvivesGilReentry) {
    EmbeddedInterpreter::ensureInitialized();

    // Nested py::gil_scoped_acquire scopes are the canonical pattern
    // for C++ code that might be called from a context that already
    // holds the GIL (e.g. a Python-driven callback). They must not
    // deadlock, must not throw, and must leave the GIL state intact
    // for the outer scope to keep using the interpreter.
    {
        py::gil_scoped_acquire outer;
        py::module_ sys = py::module_::import("sys");
        ASSERT_FALSE(sys.is_none());
        {
            py::gil_scoped_acquire inner;
            py::object versionInfo = sys.attr("version_info");
            auto major = versionInfo.attr("major").cast<int>();
            EXPECT_EQ(major, 3) << "embedded interpreter must be Python 3.x";
        }
        // Outer scope still functional after the inner scope releases.
        auto executable = sys.attr("executable").cast<std::string>();
        EXPECT_FALSE(executable.empty());
    }
}

// The interpreter must run from the bundled python-build-standalone
// prefix, not from any host Python. sys.prefix and the stdlib's os.py
// both resolving under the baked prefix proves PyConfig.home took effect
// and the host environment did not win (isolated config).
TEST(TestEmbeddedInterpreter, StdlibResolvesFromBundledPrefix) {
    EmbeddedInterpreter::ensureInitialized();

    py::gil_scoped_acquire gil;
    const std::string home{ck_dsl_provider::kCkDslPythonHome};
    ASSERT_FALSE(home.empty()) << "kCkDslPythonHome must be baked in";

    py::module_ sys = py::module_::import("sys");
    auto prefix = sys.attr("prefix").cast<std::string>();
    EXPECT_TRUE(startsWith(prefix, home))
        << "sys.prefix='" << prefix << "' is not under bundled home '" << home << "'";

    py::module_ os = py::module_::import("os");
    auto osFile = os.attr("__file__").cast<std::string>();
    EXPECT_TRUE(startsWith(osFile, home))
        << "os.__file__='" << osFile << "' is not under bundled home '" << home << "'";
}

// ctypes -> comgr is the mandatory FFI path the CK DSL compile pipeline
// drives. _ctypes is statically linked into the bundled libpython, so
// this confirms it loads and can open libamd_comgr through the embedded
// interpreter. Host-only: opening the library needs no GPU.
TEST(TestEmbeddedInterpreter, CtypesLoadsComgr) {
    EmbeddedInterpreter::ensureInitialized();

    py::gil_scoped_acquire gil;
    py::module_ ctypes = py::module_::import("ctypes");
    py::object handle = ctypes.attr("CDLL")("libamd_comgr.so");
    EXPECT_FALSE(handle.is_none());
}
