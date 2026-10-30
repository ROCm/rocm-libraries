// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/flavor_decl_chain.cpp -- host unit test for the C++ half of the
 * intrinsic declaration rule.
 *
 * The rule (Python _decls_for_flavor): a compiler lane's declaration table is
 * the base table plus the own overrides of every declaration ABI generation up
 * to and including the one its emission profile selects, in
 * rocke_ll_declaration_abi_t order; each generation's override array holds only
 * the entries whose text changed at that generation.
 * tests/core/test_flavor_decl_tables.py states it for the Python tables; this
 * file states it for the C++ tables, generation by generation and lane by
 * lane, so an override array that copies an older one, or a resolver that
 * merges differently, fails here with the key named instead of as golden hash
 * churn. The byte-identity gate holds the two engines' texts together.
 *
 * Plain executable: reports every failed check and returns non-zero when any
 * failed (a clean run is the pass criterion). Registered via
 * tests/CMakeLists.txt.
 */
#include <cstdio>
#include <cstring>

#include "rocke/llvm_profile_internal.h"
#include "rocke/lower_llvm.h"
#include "rocke/lower_llvm_internal.h"

static int g_failures = 0;

#define CHECK(cond, what, key)                                                             \
    do                                                                                     \
    {                                                                                      \
        if(!(cond))                                                                        \
        {                                                                                  \
            fprintf(stderr, "FAIL: %s [%s] (%s:%d)\n", (what), (key), __FILE__, __LINE__); \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while(0)

static const int OLDEST = (int)ckc::ROCKE_LL_DECLARATION_ABI_LLVM20;
static const int NEWEST = (int)ckc::ROCKE_LL_DECLARATION_ABI_LLVM23;

/* The text generation `abi`'s own override array gives `key`, or NULL. */
static const char* own_text(int abi, const char* key)
{
    int n = 0;
    const ckc::rocke_ll_decl_t* own
        = ckc::rocke_ll_declaration_abi_own_overrides((ckc::rocke_ll_declaration_abi_t)abi, &n);
    for(int j = 0; j < n; j++)
    {
        if(strcmp(own[j].key, key) == 0)
            return own[j].decl;
    }
    return NULL;
}

/* The text generation `abi` should give `key`, built from the base table and
 * the own override arrays alone (not from the resolver under test): the newest
 * own override at or below `abi`, else the base text. */
static const char* expected_text(int abi, int index)
{
    const char* text = ckc::ROCKE_LL_INTRINSIC_DECLS[index].decl;
    for(int g = OLDEST; g <= abi; g++)
    {
        const char* own = own_text(g, ckc::ROCKE_LL_INTRINSIC_DECLS[index].key);
        if(own)
            text = own;
    }
    return text;
}

static bool in_base(const char* key)
{
    for(int i = 0; i < ckc::ROCKE_LL_INTRINSIC_DECLS_COUNT; i++)
    {
        if(strcmp(ckc::ROCKE_LL_INTRINSIC_DECLS[i].key, key) == 0)
            return true;
    }
    return false;
}

int main(void)
{
    const int lanes = rocke_llvm_flavor_count();
    const int nbase = ckc::ROCKE_LL_INTRINSIC_DECLS_COUNT;

    /* The oldest generation is the base table itself. */
    {
        int n = -1;
        ckc::rocke_ll_declaration_abi_own_overrides((ckc::rocke_ll_declaration_abi_t)OLDEST, &n);
        CHECK(n == 0, "the oldest generation overrides nothing", "LLVM20");
    }

    /* Overrides change the text of existing keys only, and only the entries
     * that changed at this generation (a copy of an inherited entry is a
     * second home that drifts when the older one is edited). */
    for(int g = OLDEST + 1; g <= NEWEST; g++)
    {
        int n = 0;
        const ckc::rocke_ll_decl_t* own
            = ckc::rocke_ll_declaration_abi_own_overrides((ckc::rocke_ll_declaration_abi_t)g, &n);
        for(int j = 0; j < n; j++)
        {
            CHECK(in_base(own[j].key), "an override names a base key", own[j].key);
            const char* inherited = NULL;
            for(int i = 0; i < nbase; i++)
            {
                if(strcmp(ckc::ROCKE_LL_INTRINSIC_DECLS[i].key, own[j].key) == 0)
                    inherited = expected_text(g - 1, i);
            }
            CHECK(inherited == NULL || strcmp(inherited, own[j].decl) != 0,
                  "an override array re-lists an entry it inherits",
                  own[j].key);
        }
    }

    /* Every lane resolves its profile's generation: the base table with each
     * generation's own entries on top, keys in base order. */
    for(int r = 0; r < lanes; r++)
    {
        rocke_llvm_flavor_t flavor = rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(r));
        const ckc::rocke_ll_emission_profile_t* profile = ckc::rocke_ll_emission_profile(flavor);
        CHECK(profile != NULL, "every lane has an emission profile", rocke_llvm_flavor_at(r));
        if(!profile)
            continue;
        for(int i = 0; i < nbase; i++)
        {
            ckc::rocke_ll_decl_t d = ckc::rocke_ll_decl_at(flavor, i);
            const char* want = expected_text((int)profile->declarations, i);
            CHECK(strcmp(d.key, ckc::ROCKE_LL_INTRINSIC_DECLS[i].key) == 0,
                  "a resolved table keeps the base order",
                  d.key);
            CHECK(d.decl != NULL && want != NULL && strcmp(d.decl, want) == 0,
                  "a lane resolves the newest override at or below its generation",
                  d.key);
            CHECK(ckc::rocke_ll_decl_text(flavor, d.key) == d.decl,
                  "key lookup and table order agree",
                  d.key);
        }
    }

    /* No silent fallback for a flavor without a profile. */
    CHECK(ckc::rocke_ll_decl_text(ROCKE_LLVM_FLAVOR_AUTO, ckc::ROCKE_LL_INTRINSIC_DECLS[0].key)
              == NULL,
          "AUTO has no declaration table",
          ckc::ROCKE_LL_INTRINSIC_DECLS[0].key);

    if(g_failures == 0)
        printf("flavor_decl_chain: OK (%d lanes, %d keys)\n", lanes, nbase);
    return g_failures == 0 ? 0 : 1;
}
