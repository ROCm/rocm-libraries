// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <string>

#include "python/EmbeddedInterpreter.hpp"

namespace py = pybind11;
using ck_dsl_provider::EmbeddedInterpreter;

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
