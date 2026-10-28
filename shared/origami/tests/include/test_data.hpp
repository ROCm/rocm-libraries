// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>

/// Test data directory @p name: the copy installed with the tests
/// (<test binary dir>/origami/test-data/<name>) if there is one, else @p source_dir.
std::filesystem::path origami_test_data(const std::filesystem::path& name, const char* source_dir);

/// The OLC fixtures (tests/data/olc).
std::filesystem::path olc_test_data();

/// This test binary, or "" where the platform does not say.
std::filesystem::path test_executable();
