// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// core/rocke_abi.cpp -- reports the compatibility levels this binary was built
// with. See rocke/abi.h for what the two numbers mean and when to bump them.
//
// The point of compiling these rather than exposing only the macros is that a
// caller's macros come from ITS copy of the header, while these come from the
// library that is actually loaded. Comparing the two is the only way a ctypes
// or dlopen caller can detect that it is talking to a different build than it
// was compiled against -- which, for the binary ABI, is a memory-safety
// question rather than a correctness one.
//
// Like rocke_build_id.cpp, this TU returns constants and must stay off every
// lowering and emission path, so the .ll byte-identity contract is unaffected
// by what these report.

extern "C" {
#include "rocke/abi.h"
}

extern "C" int rocke_abi_version(void)
{
    return ROCKE_ABI_VERSION;
}

extern "C" int rocke_recipe_abi_level(void)
{
    return ROCKE_RECIPE_ABI;
}
