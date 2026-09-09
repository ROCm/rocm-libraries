// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/flavor_datalayout_ssot.cpp -- host unit test holding the two
 * C-side flavor->datalayout tables together.
 *
 * The engine answers "what datalayout does this LLVM flavor use?" in two
 * places: rocke_isa_datalayout_for_flavor (isa/backend.cpp, the ISA-layer
 * accessor the standalone C emitters call) and rocke_ll_datalayout_for_flavor
 * (lower_llvm/data.cpp, what the lowerer stamps into the module header).
 * Python has one function, _datalayout_for_flavor, that both of its layers
 * call. Two copies of a string table is a drift hazard with no natural
 * failure: an llvm23 branch missing from one table can emit IR that differs
 * only in a field the runtime tolerates today and is silently wrong the day a
 * stricter path (bitcode, a tighter verifier) rejects it. This test makes the
 * two agree by construction over the whole flavor ladder, so adding a rung or
 * changing a layout has to be done in both places.
 *
 * Plain executable: returns non-zero on the first failed check (a clean run is
 * the pass criterion). Registered via tests/CMakeLists.txt so it is installed
 * into the provider test artifact and run under ctest by TheRock CI.
 */
#include <cstdio>
#include <cstring>

#include "rocke/isa_backend.h"
#include "rocke/lower_llvm.h"
#include "rocke/lower_llvm_internal.h"

static int g_failures = 0;

#define CHECK(cond, msg)                                                      \
    do                                                                        \
    {                                                                         \
        if(!(cond))                                                           \
        {                                                                     \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_failures;                                                     \
        }                                                                     \
    } while(0)

int main(void)
{
    const char* err = NULL;
    rocke_isa_backend_t be = rocke_backend_for("gfx950", &err);
    int n = rocke_llvm_flavor_count();
    int i;

    CHECK(err == NULL, "gfx950 must resolve to an ISA backend");
    CHECK(n > 0, "the flavor ladder must be non-empty");

    for(i = 0; i < n; ++i)
    {
        const char* name = rocke_llvm_flavor_at(i);
        rocke_llvm_flavor_t f = rocke_llvm_flavor_from_name(name);
        const char* isa_dl = rocke_isa_datalayout_for_flavor(&be, f);
        const char* ll_dl = ckc::rocke_ll_datalayout_for_flavor(f);

        CHECK(f != ROCKE_LLVM_FLAVOR_AUTO, name);
        CHECK(isa_dl != NULL && ll_dl != NULL, name);
        if(!isa_dl || !ll_dl)
        {
            continue;
        }
        if(strcmp(isa_dl, ll_dl) != 0)
        {
            fprintf(stderr,
                    "FAIL: datalayout tables disagree for flavor '%s'\n"
                    "  rocke_isa_datalayout_for_flavor: %s\n"
                    "  rocke_ll_datalayout_for_flavor:  %s\n",
                    name,
                    isa_dl,
                    ll_dl);
            ++g_failures;
        }
    }

    /* The p8 field is the one that moves between generations, and it is what
     * the flavor ladder's `modern` column encodes. Pin the mapping so a rung
     * added with the wrong generation is caught here rather than as a golden
     * hash churn with no explanation. */
    for(i = 0; i < n; ++i)
    {
        const char* name = rocke_llvm_flavor_at(i);
        rocke_llvm_flavor_t f = rocke_llvm_flavor_from_name(name);
        const char* dl = ckc::rocke_ll_datalayout_for_flavor(f);
        bool modern = ckc::rocke_ll_flavor_is_modern(f);
        const char* want = modern ? "-p8:128:128:128:48-" : "-p8:128:128-";

        CHECK(dl != NULL && strstr(dl, want) != NULL, name);
    }

    /* The ELF symbol-mangling spec m:e is the second drift axis, orthogonal to
     * p8: absent in LLVM 20 / 22, present in LLVM 23. Pin it per flavor so a
     * wrong or missing m:e is caught here with an explanation -- the same guard
     * the p8 field gets above -- instead of surfacing as golden hash churn. */
    for(i = 0; i < n; ++i)
    {
        const char* name = rocke_llvm_flavor_at(i);
        rocke_llvm_flavor_t f = rocke_llvm_flavor_from_name(name);
        const char* dl = ckc::rocke_ll_datalayout_for_flavor(f);
        bool want_me = (f == ROCKE_LLVM_FLAVOR_LLVM23);
        bool has_me = dl != NULL && strstr(dl, "-m:e-") != NULL;

        CHECK(dl != NULL && has_me == want_me, name);
    }

    if(g_failures == 0)
    {
        printf("flavor_datalayout_ssot: OK (%d flavors)\n", n);
    }
    return g_failures == 0 ? 0 : 1;
}
