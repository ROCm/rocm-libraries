// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Host coverage for the target identity contract in test_arch_target_identity.py.
#include "rocke/arch_target.h"

#include <cstdio>
#include <cstring>

using query_fn = const char* (*)(const char*, char*, size_t);
static int failures = 0;

static void check(query_fn query, const char* input, const char* expected)
{
    // Check every capacity through an exact fit and one byte larger, with
    // sentinels proving that truncation and invalid arguments stay in bounds.
    for(size_t cap = 0; cap <= strlen(expected) + 2; ++cap)
    {
        char out[256];
        memset(out, '#', sizeof(out));
        const char* result = query(input, out, cap);
        size_t len = cap == 0 ? 0 : (strlen(expected) < cap ? strlen(expected) : cap - 1);
        if((cap == 0 && result != NULL)
           || (cap != 0 && (result != out || memcmp(out, expected, len) != 0 || out[len] != '\0'))
           || out[cap] != '#')
        {
            fprintf(
                stderr, "FAIL: input '%s', expected '%s', capacity %zu\n", input, expected, cap);
            ++failures;
        }
    }
    char out = '#';
    if(query(NULL, &out, 1) != NULL || out != '#' || query(input, NULL, 1) != NULL)
        ++failures;
}

static void check_target(const char* target, const char* base, const char* compiler)
{
    char isa[256];
    snprintf(isa, sizeof(isa), "amdgcn-amd-amdhsa--%s", target);
    check(rocke_target_id_from_isa, isa, target);
    check(rocke_target_id_from_isa, target, target);
    check(rocke_base_arch_from_target_id, target, base);
    check(rocke_compiler_target_from_target_id, target, compiler);
    check(rocke_arch_from_isa, isa, base);
    check(rocke_arch_from_isa, target, base);
}

int main(int argc, char** argv)
{
    // Query mode lets external Python/C++ comparisons use the real engine.
    if(argc > 1)
    {
        for(int i = 1; i < argc; ++i)
        {
            char target[1024], base[1024], compiler[1024], arch[1024];
            rocke_target_id_from_isa(argv[i], target, sizeof(target));
            rocke_base_arch_from_target_id(target, base, sizeof(base));
            rocke_compiler_target_from_target_id(target, compiler, sizeof(compiler));
            rocke_arch_from_isa(argv[i], arch, sizeof(arch));
            printf("%s\t%s\t%s\t%s\n", target, base, compiler, arch);
        }
        return 0;
    }

    int count = 0;
    const char* const* arches = rocke_known_arches(&count);
    if(count == 0)
        ++failures;
    for(int i = 0; i < count; ++i)
        check_target(arches[i], arches[i], arches[i]);

    check_target("gfx1250-strict", "gfx1250", "gfx1250");
    check_target("gfx942:sramecc+:xnack-", "gfx942", "gfx942:sramecc+:xnack-");
    check_target("gfx1250-strict:xnack-", "gfx1250", "gfx1250:xnack-");
    check_target("gfx11-generic-strict:xnack-", "gfx11-generic", "gfx11-generic:xnack-");
    check_target("gfx00a-profile:unknown+", "gfx00a", "gfx00a:unknown+");
    check_target("gfx942:", "gfx942", "gfx942:");
    check(rocke_arch_from_isa, "unexpected-target", "unexpected-target");
    check(rocke_target_id_from_isa, "unexpected-target", "unexpected-target");
    check(rocke_base_arch_from_target_id, "unexpected-target", "unexpected-target");
    check(rocke_compiler_target_from_target_id, "unexpected-target", "unexpected-target");
    check(rocke_arch_from_isa, "amdgcn-amd-amdhsa-opencl-gfx1250-strict", "gfx1250");
    check(rocke_target_id_from_isa, "gfx-named-prefix-gfx942:xnack-", "gfx942:xnack-");
    check(rocke_arch_from_isa, "gfx-named-prefix-gfx942:xnack-", "gfx942");
    check(rocke_target_id_from_isa, "", "");
    check(rocke_base_arch_from_target_id, "", "");
    check(rocke_compiler_target_from_target_id, "", "");
    check(rocke_arch_from_isa, "", "");
    return failures == 0 ? 0 : 1;
}
