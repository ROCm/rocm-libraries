// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_qk_scale_spec_validate.cpp -- rocke_qk_scale_spec_validate
 * rejects a bad layout or block size with the Python QkScaleSpec message, bad
 * value included (tests/core/test_qk_scale_spec.py pins the Python side). The
 * layout is an enum in C, so an unknown layout is reported by its value.
 */
#include <cstdio>
#include <cstring>
#include <string>

extern "C" {
#include "rocke/helper_rocke.helpers.qk_scale.h"
}

namespace
{

int failures = 0;

void expect(bool ok, const std::string& what)
{
    if(!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

void expect_rejected(const rocke_qk_scale_spec_t& spec, const char* want)
{
    char reason[128] = "unset";
    expect(rocke_qk_scale_spec_validate(&spec, reason, sizeof(reason)) == 0,
           std::string("accepted a spec that should give: ") + want);
    expect(std::strcmp(reason, want) == 0,
           std::string("reason '") + reason + "' != '" + want + "'");
}

} // namespace

int main()
{
    rocke_qk_scale_spec_t spec;

    rocke_qk_scale_spec_init(&spec, ROCKE_QK_SCALE_PER_BLOCK);
    spec.scale_block = 0;
    expect_rejected(spec, "per_block scale_block must be > 0 (got 0)");
    spec.scale_block = -4;
    expect_rejected(spec, "per_block scale_block must be > 0 (got -4)");

    rocke_qk_scale_spec_init(&spec, ROCKE_QK_SCALE_PER_HEAD);
    spec.layout = (rocke_qk_scale_layout_t)7;
    expect_rejected(spec, "QkScaleSpec.layout must be 'per_head' or 'per_block', got 7");

    // Valid specs clear the reason; a NULL buffer is allowed.
    char reason[16] = "unset";
    rocke_qk_scale_spec_init(&spec, ROCKE_QK_SCALE_PER_BLOCK);
    spec.scale_block = 64;
    expect(rocke_qk_scale_spec_validate(&spec, reason, sizeof(reason)) == 1 && reason[0] == '\0',
           "a per_block spec with scale_block 64 is valid and clears the reason");
    rocke_qk_scale_spec_init(&spec, ROCKE_QK_SCALE_PER_HEAD);
    expect(rocke_qk_scale_spec_validate(&spec, nullptr, 0) == 1, "a per_head spec is valid");
    expect(rocke_qk_scale_spec_validate(nullptr, nullptr, 0) == 0, "a NULL spec is rejected");

    if(failures)
        std::fprintf(stderr, "%d qk_scale spec check(s) failed\n", failures);
    return failures ? 1 : 0;
}
