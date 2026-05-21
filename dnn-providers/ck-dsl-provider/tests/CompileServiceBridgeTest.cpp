// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <string>
#include <string_view>

#include "CkDslContainer.hpp"
#include "ckdsl_provider_paths.h"
#include "python/CompileServiceBridge.hpp"
#include "python/EmbeddedInterpreter.hpp"
#include "python/PythonError.hpp"

namespace py = pybind11;
using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::CompileServiceBridge;
using ck_dsl_provider::EmbeddedInterpreter;
using ck_dsl_provider::PythonError;

namespace {

bool endsWith(std::string_view text, std::string_view suffix) {
    if (suffix.size() > text.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), text.rbegin());
}

}  // namespace

TEST(TestCompileServiceBridge, NoopSmoke) {
    CkDslContainer container;
    auto& bridge = container.compileServiceBridge();

    py::dict result = bridge.noopSmoke();

    py::gil_scoped_acquire gil;
    ASSERT_TRUE(result.contains("smoke"));
    auto smoke = result["smoke"].cast<std::string>();
    EXPECT_EQ(smoke, "ok");

    ASSERT_TRUE(result.contains("ck_dsl_module_path"));
    auto modulePath = result["ck_dsl_module_path"].cast<std::string>();
    EXPECT_TRUE(endsWith(modulePath, "composablekernel/python/ck_dsl/__init__.py"))
        << "unexpected ck_dsl.__file__: " << modulePath;
}

TEST(TestCompileServiceBridge, RaisesOnUnknownAttr) {
    CkDslContainer container;
    auto& bridge = container.compileServiceBridge();

    // Drive a Python AttributeError through the bridge's own module
    // handle so we exercise the PythonError translation path. This is
    // the same boundary the I-7 compile() will sit on, so verifying it
    // now is the cheapest place to catch a regression in the
    // py::error_already_set → HipdnnPluginException conversion.
    try {
        py::gil_scoped_acquire gil;
        py::object result = bridge.moduleForTesting().attr("nonexistent_function")();
        (void)result;
        FAIL() << "expected attr lookup to throw py::error_already_set";
    } catch (const py::error_already_set& error) {
        try {
            PythonError::raise(error, "TestCompileServiceBridge::RaisesOnUnknownAttr");
            FAIL() << "PythonError::raise must not return";
        } catch (const hipdnn_plugin_sdk::HipdnnPluginException& translated) {
            EXPECT_EQ(translated.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
            std::string message = translated.getMessage();
            EXPECT_NE(message.find("AttributeError"), std::string::npos)
                << "translated message missing Python type name: " << message;
            EXPECT_NE(message.find("nonexistent_function"), std::string::npos)
                << "translated message missing Python detail: " << message;
        }
    }
}

TEST(TestCompileServiceBridge, IdempotentSysPathInjection) {
    // Constructing the container twice should NOT result in duplicate
    // entries on sys.path for either of the baked-in package paths.
    // (Container is normally a per-process singleton via
    // SharedContainerManager; tests construct it directly to keep the
    // unit boundary small.)
    CkDslContainer firstContainer;
    CkDslContainer secondContainer;
    (void)firstContainer;
    (void)secondContainer;

    py::gil_scoped_acquire gil;
    py::module_ sys = py::module_::import("sys");
    py::list sysPath = sys.attr("path").cast<py::list>();

    const std::string ckDslPath{ck_dsl_provider::kCkDslPythonPackagePath};
    const std::string providerPath{ck_dsl_provider::kCkDslProviderPythonPackagePath};

    int ckDslHits = 0;
    int providerHits = 0;
    for (py::handle entry : sysPath) {
        try {
            if (!py::isinstance<py::str>(entry)) {
                continue;
            }
            auto value = entry.cast<std::string>();
            if (value == ckDslPath) {
                ++ckDslHits;
            } else if (value == providerPath) {
                ++providerHits;
            }
        } catch (const py::error_already_set&) {
            continue;
        }
    }

    EXPECT_EQ(ckDslHits, 1) << "ck_dsl path appears " << ckDslHits << " times on sys.path";
    EXPECT_EQ(providerHits, 1) << "ck_dsl_provider path appears " << providerHits
                               << " times on sys.path";
}
