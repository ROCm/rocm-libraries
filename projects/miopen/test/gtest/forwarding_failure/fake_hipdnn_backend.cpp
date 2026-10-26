// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// A library with the hipDNN backend's file name and none of its functions.
// hipDNN's frontend loads it, finds no version to check and refuses it, so
// script/test_forwarding_failure_modes.py can make hipDNN unavailable without
// removing the real backend.

#include <cstdio>

namespace {
// Tells the test whether a case loaded the fake.
struct AnnounceLoad
{
    AnnounceLoad() { std::fputs("FAKE_HIPDNN_BACKEND_LOADED\n", stderr); }
};

const AnnounceLoad announceLoad;
} // namespace
