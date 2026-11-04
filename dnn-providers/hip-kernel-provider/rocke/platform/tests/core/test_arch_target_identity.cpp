// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Host coverage for the target identity contract in test_arch_target_identity.py.
#include "rocke/arch_target.h"
#include "rocke/arch_target_internal.h"
#include "rocke/lower_llvm.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

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

// The compile-target rule at llvm23, in the shape check() takes. llvm23's
// compiler accepts every catalogued processor name but gfx1154.
static const char* compiler_at_llvm23(const char* target, char* out, size_t cap)
{
    return rocke_compiler_target_for_flavor(target, "llvm23", out, cap, NULL, 0);
}

// The compiler target for one (target, flavor), and the message that comes with
// it: expected NULL means the rule must refuse, with message_start opening the
// error; otherwise message_start opens the fallback note ("" for none).
static void check_compile(const char* target,
                          const char* flavor,
                          const char* expected,
                          const char* message_start)
{
    char out[256];
    char message[512];
    const char* result = rocke_compiler_target_for_flavor(
        target, flavor, out, sizeof(out), message, sizeof(message));
    const bool ok = expected == NULL ? result == NULL : result == out && strcmp(out, expected) == 0;
    if(!ok || strncmp(message, message_start, strlen(message_start)) != 0
       || (message_start[0] == '\0' && message[0] != '\0'))
    {
        fprintf(stderr,
                "FAIL: compile target of '%s' at %s: got '%s' (%s), expected '%s'\n",
                target,
                flavor,
                result ? out : "(refused)",
                message,
                expected ? expected : "(refused)");
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

// A processor row that takes its facts from its generic target (arch_specs.json
// "facts_of") must be that target in every field but `gfx`, sharing the same
// MMA array rather than a copy of it.
static void check_facts_of(const char* gfx, const char* generic)
{
    const rocke_arch_target_t* t = rocke_arch_target_from_gfx(gfx);
    const rocke_arch_target_t* g = rocke_arch_target_from_gfx(generic);
    if(t == NULL || g == NULL)
    {
        fprintf(stderr, "FAIL: facts_of '%s' -> '%s': unknown target\n", gfx, generic);
        ++failures;
        return;
    }
    const rocke_memory_caps_t& m = t->memory;
    const rocke_memory_caps_t& gm = g->memory;
    const bool same = strcmp(t->gfx, gfx) == 0 && strcmp(t->family, g->family) == 0
                      && strcmp(t->target_family, g->target_family) == 0
                      && t->wave_size == g->wave_size
                      && t->lds_capacity_bytes == g->lds_capacity_bytes
                      && t->vmcnt_bits == g->vmcnt_bits && t->mma.ops == g->mma.ops
                      && t->mma.num_ops == g->mma.num_ops && m.has_async_lds == gm.has_async_lds
                      && m.has_async_global_lds == gm.has_async_global_lds
                      && m.has_ds_read_tr == gm.has_ds_read_tr && m.has_tdm == gm.has_tdm
                      && m.buffer_load_max_dwords == gm.buffer_load_max_dwords
                      && m.async_lds_max_dwords == gm.async_lds_max_dwords
                      && m.has_scalar_data_prefetch == gm.has_scalar_data_prefetch
                      && m.has_global_prefetch == gm.has_global_prefetch
                      && m.has_cluster_launch == gm.has_cluster_launch
                      && m.has_multicast_load == gm.has_multicast_load
                      && t->limits.max_threads_per_block == g->limits.max_threads_per_block
                      && t->limits.vgprs == g->limits.vgprs && t->limits.agprs == g->limits.agprs
                      && t->limits.sgprs == g->limits.sgprs;
    if(!same)
    {
        fprintf(stderr, "FAIL: '%s' does not hold exactly the facts of '%s'\n", gfx, generic);
        ++failures;
    }
}

// The generic target the generic_targets mirror lists gfx under, or NULL.
static const char* generic_listing(const char* gfx)
{
    for(const rocke_ati_generic_row_t* g = rocke_ati_generic_targets; g->generic != NULL; ++g)
        for(const char* const* member = g->members; *member != NULL; ++member)
            if(strcmp(*member, gfx) == 0)
                return g->generic;
    return NULL;
}

// One query: the identity of `isa` as tab-separated fields on one line.
static void print_identity(const char* isa)
{
    char target[1024], base[1024], compiler[1024], arch[1024], generic[1024];
    rocke_target_id_from_isa(isa, target, sizeof(target));
    rocke_base_arch_from_target_id(target, base, sizeof(base));
    rocke_compiler_target_from_target_id(target, compiler, sizeof(compiler));
    rocke_arch_from_isa(isa, arch, sizeof(arch));
    if(rocke_generic_arch_from_target_id(target, generic, sizeof(generic)) == NULL)
        strcpy(generic, "-");
    printf("%s\t%s\t%s\t%s\t%s", target, base, compiler, arch, generic);
    // The compile target and its message at every LLVM flavor; "!" for
    // a refused target.
    for(int i = 0; i < rocke_llvm_flavor_count(); ++i)
    {
        const char* flavor = rocke_llvm_flavor_at(i);
        char message[1024];
        if(rocke_compiler_target_for_flavor(
               target, flavor, compiler, sizeof(compiler), message, sizeof(message))
           == NULL)
            strcpy(compiler, "!");
        printf("\t%s\t%s", compiler, message);
    }
    printf("\n");
}

// Query mode: one input per line on standard input (the list outgrows the
// command-line length Windows allows), each answered by print_identity.
static int run_queries()
{
    char line[1024];
    while(fgets(line, sizeof(line), stdin) != NULL)
    {
        size_t len = strlen(line);
        if(len > 0 && line[len - 1] != '\n' && !feof(stdin))
        {
            fprintf(stderr, "FAIL: a query line is longer than %zu bytes\n", sizeof(line) - 2);
            return 2;
        }
        while(len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        print_identity(line);
    }
    return 0;
}

int main(int argc, char** argv)
{
    // Query mode lets external Python/C++ comparisons use the real engine.
    if(argc == 2 && strcmp(argv[1], "--query") == 0)
        return run_queries();
    if(argc > 1)
    {
        fprintf(stderr, "usage: %s [--query < inputs]\n", argv[0]);
        return 2;
    }

    int count = 0;
    const char* const* arches = rocke_known_arches(&count);
    if(count == 0)
        ++failures;
    for(int i = 0; i < count; ++i)
    {
        check_target(arches[i], arches[i], arches[i]);
        check(compiler_at_llvm23,
              arches[i],
              strcmp(arches[i], "gfx1154") == 0 ? "gfx11-generic" : arches[i]);
    }

    check_target("gfx1250-strict", "gfx1250", "gfx1250");
    check_target("gfx942:sramecc+:xnack-", "gfx942", "gfx942:sramecc+:xnack-");
    check_target("gfx1250-strict:xnack-", "gfx1250", "gfx1250:xnack-");
    check_target("gfx11-generic-strict:xnack-", "gfx11-generic", "gfx11-generic:xnack-");
    check_target("gfx1100", "gfx1100", "gfx1100");
    check_target("gfx1101", "gfx1101", "gfx1101");
    check_target("gfx1101:unknown+", "gfx1101", "gfx1101:unknown+");
    check_target("gfx1101-strict", "gfx1101", "gfx1101");
    check_target("gfx1101-strict:unknown+", "gfx1101", "gfx1101:unknown+");
    check_target("gfx1102", "gfx1102", "gfx1102");
    check_target("gfx1103", "gfx1103", "gfx1103");
    check_target("gfx1150", "gfx1150", "gfx1150");
    check_target("gfx1151", "gfx1151", "gfx1151");
    check_target("gfx1151-strict:unknown+", "gfx1151", "gfx1151:unknown+");
    check_target("gfx1152", "gfx1152", "gfx1152");
    check_target("gfx1153", "gfx1153", "gfx1153");
    check_target("gfx1154", "gfx1154", "gfx1154");
    check(compiler_at_llvm23, "gfx1154-strict:xnack-", "gfx11-generic:xnack-");
    check_target("gfx1170", "gfx1170", "gfx1170");
    check_target("gfx1200", "gfx1200", "gfx1200");
    check_target("gfx1200-strict:xnack-", "gfx1200", "gfx1200:xnack-");
    check_target("gfx1201", "gfx1201", "gfx1201");
    check_target("gfx1101_bad", "gfx1101", "gfx1101_bad");
    check_target("gfx00a-profile:unknown+", "gfx00a", "gfx00a:unknown+");
    check_target("gfx942:", "gfx942", "gfx942:");
    check(rocke_arch_from_isa, "unexpected-target", "unexpected-target");
    check(rocke_target_id_from_isa, "unexpected-target", "unexpected-target");
    check(rocke_base_arch_from_target_id, "unexpected-target", "unexpected-target");
    check(rocke_compiler_target_from_target_id, "unexpected-target", "unexpected-target");
    check(compiler_at_llvm23, "unexpected-target", "unexpected-target");
    check(rocke_arch_from_isa, "amdgcn-amd-amdhsa-opencl-gfx1250-strict", "gfx1250");
    check(rocke_target_id_from_isa, "gfx-named-prefix-gfx942:xnack-", "gfx942:xnack-");
    check(rocke_arch_from_isa, "gfx-named-prefix-gfx942:xnack-", "gfx942");
    check(rocke_target_id_from_isa, "", "");
    check(rocke_base_arch_from_target_id, "", "");
    check(rocke_compiler_target_from_target_id, "", "");
    check(compiler_at_llvm23, "", "");
    check(rocke_arch_from_isa, "", "");
    // Every member of a supported generic target (read from the generic_targets
    // mirror) has its own row holding exactly its generic's facts.
    for(const rocke_ati_generic_row_t* g = rocke_ati_generic_targets; g->generic != NULL; ++g)
        for(const char* const* member = g->members; *member != NULL; ++member)
            check_facts_of(*member, g->generic);
    // A registry row names the generic it copies (its facts_of string) exactly
    // when generic_targets lists it under that generic, and then holds exactly
    // that generic's facts: the string agrees with the struct it copies.
    for(int i = 0; i < rocke_ati_arch_registry_len; ++i)
    {
        const rocke_ati_arch_row_t& row = rocke_ati_arch_registry[i];
        const char* listed_under = generic_listing(row.gfx);
        if((row.facts_of == NULL) != (listed_under == NULL)
           || (row.facts_of != NULL && strcmp(row.facts_of, listed_under) != 0))
        {
            fprintf(stderr,
                    "FAIL: registry row '%s' takes its facts from '%s', but generic_targets "
                    "lists it under '%s'\n",
                    row.gfx,
                    row.facts_of ? row.facts_of : "(none)",
                    listed_under ? listed_under : "(none)");
            ++failures;
        }
        if(row.facts_of != NULL)
            check_facts_of(row.gfx, row.facts_of);
    }
    // The compile-target rule per flavor (measured: llvm20 accepts neither
    // gfx1154 nor gfx1250, llvm22 and llvm23 do not accept gfx1154, llvm24
    // accepts both).
    const char* fallback = "the llvm20 compiler does not accept the processor 'gfx1154'; "
                           "compiling for 'gfx11-generic' instead";
    check_compile("gfx1154", "llvm20", "gfx11-generic", fallback);
    check_compile("gfx1154-strict:xnack-", "llvm20", "gfx11-generic:xnack-", fallback);
    check_compile("gfx1154", "llvm23", "gfx11-generic", "the llvm23 compiler does not accept");
    check_compile("gfx1154", "llvm24", "gfx1154", "");
    check_compile("gfx1250", "llvm24", "gfx1250", "");
    check_compile("gfx1153", "llvm20", "gfx1153", "");
    check_compile("gfx1250", "llvm23", "gfx1250", "");
    check_compile("gfx1250",
                  "llvm20",
                  NULL,
                  "the llvm20 compiler does not accept the processor 'gfx1250', and its "
                  "arch_specs.json row points at no generic target that compiler accepts");
    check_compile("gfx00a", "llvm20", "gfx00a", "");
    check_compile("gfx1154", "llvm-without-a-column", "gfx1154", "");
    check(rocke_generic_arch_from_target_id, "gfx1100", "gfx11-generic");
    check(rocke_generic_arch_from_target_id, "gfx1151-strict:xnack-", "gfx11-generic");
    check(rocke_generic_arch_from_target_id, "gfx11-generic", "gfx11-generic");
    char generic[32] = "#";
    if(rocke_generic_arch_from_target_id("gfx942", generic, sizeof(generic)) != NULL
       || rocke_generic_arch_from_target_id("gfx11000", generic, sizeof(generic)) != NULL
       || generic[0] != '#')
        ++failures;
    return failures == 0 ? 0 : 1;
}
