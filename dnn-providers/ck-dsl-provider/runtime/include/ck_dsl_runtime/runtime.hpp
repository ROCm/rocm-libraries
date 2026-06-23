// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Umbrella include for the ck_dsl C++ runtime: the transparent component model
// (Manifest, Compiler, Kernel, ArtifactStore, Dispatcher) that loads, compiles,
// and launches ck_dsl-generated kernels with no Python dependency.
#pragma once

#include "ck_dsl_runtime/artifact_store.hpp"
#include "ck_dsl_runtime/comgr.hpp"
#include "ck_dsl_runtime/dispatcher.hpp"
#include "ck_dsl_runtime/json.hpp"
#include "ck_dsl_runtime/kernel.hpp"
#include "ck_dsl_runtime/manifest.hpp"
