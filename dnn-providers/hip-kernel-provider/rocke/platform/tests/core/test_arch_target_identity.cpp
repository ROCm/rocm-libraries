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

static void check_parse(const char* value, const char* expected)
{
    // expected NULL: the value is refused with the shared refusal text.
    char out[256], err[512], want[512];
    memset(out, '#', sizeof(out));
    memset(err, '#', sizeof(err));
    const char* result = rocke_parse_target_id(value, out, sizeof(out), err, sizeof(err));
    bool ok;
    if(expected)
        ok = result == out && strcmp(out, expected) == 0 && err[0] == '#';
    else
    {
        snprintf(want,
                 sizeof(want),
                 "'%s' is not a target ID: expected an optional amdgcn-amd-amdhsa-- or "
                 "amdgcn-amd-amdhsa-unknown- prefix, a processor such as gfx942 or "
                 "gfx11-generic, then any :feature+ or :feature- suffixes",
                 value);
        ok = result == NULL && out[0] == '#' && strcmp(err, want) == 0;
    }
    if(!ok)
    {
        fprintf(stderr, "FAIL: parse_target_id('%s')\n", value);
        ++failures;
    }
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
            char target[1024], base[1024], compiler[1024], arch[1024], generic[1024];
            // A refused value prints as "!" and the refusal text.
            char processor[1024], parsed[1025];
            rocke_target_id_from_isa(argv[i], target, sizeof(target));
            rocke_processor_from_target_id(target, processor, sizeof(processor));
            if(rocke_parse_target_id(
                   argv[i], parsed, sizeof(parsed), parsed + 1, sizeof(parsed) - 1)
               == NULL)
                parsed[0] = '!';
            rocke_base_arch_from_target_id(target, base, sizeof(base));
            rocke_compiler_target_from_target_id(target, compiler, sizeof(compiler));
            rocke_arch_from_isa(argv[i], arch, sizeof(arch));
            if(rocke_generic_arch_from_target_id(target, generic, sizeof(generic)) == NULL)
                strcpy(generic, "-");
            printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                   target,
                   base,
                   compiler,
                   arch,
                   generic,
                   processor,
                   parsed);
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
    check_target("gfx1100", "gfx11-generic", "gfx1100");
    check_target("gfx1101", "gfx11-generic", "gfx1101");
    check_target("gfx1101:unknown+", "gfx11-generic", "gfx1101:unknown+");
    check_target("gfx1101-strict", "gfx11-generic", "gfx1101");
    check_target("gfx1101-strict:unknown+", "gfx11-generic", "gfx1101:unknown+");
    check_target("gfx1102", "gfx11-generic", "gfx1102");
    check_target("gfx1103", "gfx11-generic", "gfx1103");
    check_target("gfx1150", "gfx11-generic", "gfx1150");
    check_target("gfx1151", "gfx1151", "gfx1151");
    check_target("gfx1151-strict:unknown+", "gfx1151", "gfx1151:unknown+");
    check_target("gfx1152", "gfx11-generic", "gfx1152");
    check_target("gfx1153", "gfx11-generic", "gfx1153");
    check_target("gfx1154", "gfx11-generic", "gfx1154");
    check_target("gfx1170", "gfx1170", "gfx1170");
    check_target("gfx1200", "gfx12-generic", "gfx1200");
    check_target("gfx1200-strict:xnack-", "gfx12-generic", "gfx1200:xnack-");
    check_target("gfx1201", "gfx1201", "gfx1201");
    check_target("gfx1101_bad", "gfx1101", "gfx1101_bad");
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
    check(rocke_generic_arch_from_target_id, "gfx1100", "gfx11-generic");
    check(rocke_generic_arch_from_target_id, "gfx1151-strict:xnack-", "gfx11-generic");
    check(rocke_generic_arch_from_target_id, "gfx11-generic", "gfx11-generic");
    check(rocke_processor_from_target_id, "gfx942:sramecc+:xnack-", "gfx942");
    check(rocke_processor_from_target_id, "gfx1250-strict", "gfx1250");
    check(rocke_processor_from_target_id, "gfx1100", "gfx1100");
    check(rocke_processor_from_target_id, "gfx11-generic:xnack-", "gfx11-generic");
    check_parse("gfx950", "gfx950");
    check_parse("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-", "gfx942:sramecc+:xnack-");
    check_parse("amdgcn-amd-amdhsa-unknown-gfx9-4-generic:xnack+", "gfx9-4-generic:xnack+");
    check_parse("", NULL);
    check_parse("notgfx950", NULL);
    check_parse("amdgcn-amd-amdhsa-unknown-:gfx950", NULL);
    check_parse("amdgcn-amd-amdhsa-linux-gfx950", NULL);
    check_parse("gfx950:xnack", NULL);
    check_parse("gfx950-", NULL);
    char parsed[8] = "#";
    if(rocke_parse_target_id(NULL, parsed, sizeof(parsed), NULL, 0) != NULL
       || rocke_parse_target_id("gfx950", NULL, 1, NULL, 0) != NULL
       || rocke_parse_target_id("gfx950", parsed, 0, NULL, 0) != NULL
       || rocke_parse_target_id("bad", parsed, sizeof(parsed), NULL, 0) != NULL || parsed[0] != '#')
        ++failures;
    char generic[32] = "#";
    if(rocke_generic_arch_from_target_id("gfx942", generic, sizeof(generic)) != NULL
       || rocke_generic_arch_from_target_id("gfx11000", generic, sizeof(generic)) != NULL
       || generic[0] != '#')
        ++failures;
    return failures == 0 ? 0 : 1;
}
