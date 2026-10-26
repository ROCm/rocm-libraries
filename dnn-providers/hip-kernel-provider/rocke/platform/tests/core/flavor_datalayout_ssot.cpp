// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/flavor_datalayout_ssot.cpp -- host unit test holding the C-side
 * compiler-lane and structural emission-profile APIs together.
 *
 * The ISA accessor delegates to the lowerer's profile resolver so there is one
 * C-side datalayout table. This test walks every public lane and proves that the
 * API layers resolve the same layout and that each profile's structural fields
 * agree with the rendered string.
 *
 * Plain executable: returns non-zero on the first failed check (a clean run is
 * the pass criterion). Registered via tests/CMakeLists.txt so it is installed
 * into the provider test artifact and run under ctest by TheRock CI.
 */
#include <cstdio>
#include <cstring>
#include <exception>

#include "rocke/isa_backend.h"
#include "rocke/lower_llvm.h"
#include "rocke/lower_llvm_internal.h"

static_assert(ROCKE_LLVM_FLAVOR_AUTO == 0);
static_assert(ROCKE_LLVM_FLAVOR_LLVM20 == 1);
static_assert(ROCKE_LLVM_FLAVOR_LLVM22 == 2);
static_assert(ROCKE_LLVM_FLAVOR_LLVM23 == 3);
static_assert(ROCKE_LLVM_FLAVOR_LLVM24 == 4);

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

    rocke_strbuf_t preamble = {};
    CHECK(rocke_strbuf_init(&preamble, 32) == 0, "preamble buffer initializes");
    CHECK(rocke_strbuf_append(&preamble, "stale") == 0, "preamble buffer accepts a sentinel");
    CHECK(rocke_isa_module_preamble_for_flavor(&be, ROCKE_LLVM_FLAVOR_AUTO, &preamble) == -1,
          "AUTO is not a concrete preamble flavor");
    CHECK(strcmp(rocke_strbuf_cstr(&preamble), "") == 0,
          "a rejected preamble leaves no malformed IR");
    CHECK(rocke_isa_module_preamble_for_flavor(&be, (rocke_llvm_flavor_t)99, &preamble) == -1,
          "an out-of-range preamble flavor is rejected");
    rocke_strbuf_free(&preamble);

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
     * p8. Pin it to the structural profile rather than compiler-lane names. */
    for(i = 0; i < n; ++i)
    {
        const char* name = rocke_llvm_flavor_at(i);
        rocke_llvm_flavor_t f = rocke_llvm_flavor_from_name(name);
        const char* dl = ckc::rocke_ll_datalayout_for_flavor(f);
        const ckc::rocke_ll_emission_profile_t* profile = ckc::rocke_ll_emission_profile(f);
        bool want_me = profile && profile->datalayout == ckc::ROCKE_LL_DATALAYOUT_PROFILE_LLVM23;
        bool has_me = dl != NULL && strstr(dl, "-m:e-") != NULL;

        CHECK(profile != NULL && dl != NULL && has_me == want_me, name);
    }

    const ckc::rocke_ll_emission_profile_t* llvm23
        = ckc::rocke_ll_emission_profile(ROCKE_LLVM_FLAVOR_LLVM23);
    const ckc::rocke_ll_emission_profile_t* llvm24
        = ckc::rocke_ll_emission_profile(ROCKE_LLVM_FLAVOR_LLVM24);
    CHECK(llvm23 != NULL && llvm23 == llvm24, "llvm23 and llvm24 share one emission profile");
    CHECK(ckc::rocke_ll_profile_has(ROCKE_LLVM_FLAVOR_LLVM24, ckc::ROCKE_LL_CAP_LLVM23_TARGET_OPS),
          "llvm24 admits LLVM23-generation target operations");
    CHECK(ckc::rocke_ll_profile_has(ROCKE_LLVM_FLAVOR_LLVM24, ckc::ROCKE_LL_CAP_SCALED_MFMA_9_ARG),
          "llvm24 uses the nine-argument scaled-MFMA ABI");
    CHECK(ckc::rocke_ll_flavor_for_llvm_major(23) == ROCKE_LLVM_FLAVOR_LLVM23,
          "LLVM 23 resolves to llvm23");
    CHECK(ckc::rocke_ll_flavor_for_llvm_major(24) == ROCKE_LLVM_FLAVOR_LLVM24,
          "LLVM 24 resolves to llvm24");

    try
    {
        (void)ckc::rocke_ll_flavor_for_llvm_major(25);
        CHECK(false, "an unqualified future compiler major must fail");
    }
    catch(const std::exception& error)
    {
        CHECK(strstr(error.what(), "latest qualified lane llvm24") != NULL,
              "future-major failure names the qualification frontier");
    }
    const char* invalid_flavors[] = {"llvm25", "LLVM24", " llvm24 "};
    for(const char* invalid : invalid_flavors)
    {
        try
        {
            (void)ckc::rocke_ll_explicit_flavor(invalid);
            CHECK(false, "an unknown explicit flavor must fail");
        }
        catch(const std::exception& error)
        {
            CHECK(strstr(error.what(), invalid) != NULL,
                  "unknown explicit-flavor failure names the bad value");
            CHECK(strstr(error.what(), "expected one of") != NULL,
                  "unknown explicit-flavor failure lists qualified lanes");
        }
    }

    if(g_failures == 0)
    {
        printf("flavor_datalayout_ssot: OK (%d flavors)\n", n);
    }
    return g_failures == 0 ? 0 : 1;
}
