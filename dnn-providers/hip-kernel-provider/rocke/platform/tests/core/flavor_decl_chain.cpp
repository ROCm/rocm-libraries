// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/flavor_decl_chain.cpp -- host unit test for the C++ half of the
 * per-flavor intrinsic declaration rule.
 *
 * The rule (Python _decls_for_flavor): a flavor's declaration table is the base
 * table plus the own overrides of every flavor up to and including it, in
 * flavor-ladder order; each flavor's override array holds only the entries
 * whose text changed at that flavor. tests/core/test_flavor_decl_tables.py
 * states it for the Python tables; this file states it for the C++ tables,
 * rung by rung, so an override array that copies an older one, or a resolver
 * that merges differently, fails here with the key named instead of as golden
 * hash churn. The byte-identity gate holds the two engines' texts together.
 *
 * Plain executable: reports every failed check and returns non-zero when any
 * failed (a clean run is the pass criterion). Registered via
 * tests/CMakeLists.txt.
 */
#include <cstdio>
#include <cstring>

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

static rocke_llvm_flavor_t ladder_flavor(int rung)
{
    return rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(rung));
}

/* The text `flavor`'s own override array gives `key`, or NULL. */
static const char* own_text(rocke_llvm_flavor_t flavor, const char* key)
{
    int n = 0;
    const ckc::rocke_ll_decl_t* own = ckc::rocke_ll_flavor_own_overrides(flavor, &n);
    for(int j = 0; j < n; j++)
    {
        if(strcmp(own[j].key, key) == 0)
            return own[j].decl;
    }
    return NULL;
}

/* The text rung `rung` should give base entry `index`, built from the base
 * table and the own override arrays alone (not from the resolver under test):
 * the newest own override at or below the rung, else the base text. */
static const char* expected_text(int rung, int index)
{
    const char* text = ckc::ROCKE_LL_INTRINSIC_DECLS[index].decl;
    for(int k = 1; k <= rung; k++)
    {
        const char* own = own_text(ladder_flavor(k), ckc::ROCKE_LL_INTRINSIC_DECLS[index].key);
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
    const int rungs = rocke_llvm_flavor_count();
    const int nbase = ckc::ROCKE_LL_INTRINSIC_DECLS_COUNT;

    /* The oldest flavor is the base table itself. */
    {
        rocke_llvm_flavor_t oldest = ladder_flavor(0);
        int n = -1;
        ckc::rocke_ll_flavor_own_overrides(oldest, &n);
        CHECK(n == 0, "the oldest flavor overrides nothing", rocke_llvm_flavor_at(0));
        for(int i = 0; i < nbase; i++)
        {
            ckc::rocke_ll_decl_t d = ckc::rocke_ll_decl_at(oldest, i);
            CHECK(d.decl == ckc::ROCKE_LL_INTRINSIC_DECLS[i].decl,
                  "the oldest flavor resolves the base text",
                  d.key);
        }
    }

    for(int r = 1; r < rungs; r++)
    {
        rocke_llvm_flavor_t flavor = ladder_flavor(r);
        rocke_llvm_flavor_t parent = ladder_flavor(r - 1);
        int n = 0;
        const ckc::rocke_ll_decl_t* own = ckc::rocke_ll_flavor_own_overrides(flavor, &n);

        /* Overrides change the text of existing keys only, and only the
         * entries that changed at this flavor (a copy of an inherited entry is
         * a second home that drifts when the older one is edited). */
        for(int j = 0; j < n; j++)
        {
            CHECK(in_base(own[j].key), "an override names a base key", own[j].key);
            const char* inherited = ckc::rocke_ll_decl_text(parent, own[j].key);
            CHECK(inherited == NULL || strcmp(inherited, own[j].decl) != 0,
                  "an override array re-lists an entry it inherits",
                  own[j].key);
        }

        /* Each rung = its parent's table with this rung's own entries on top,
         * keys in base order. */
        for(int i = 0; i < nbase; i++)
        {
            ckc::rocke_ll_decl_t d = ckc::rocke_ll_decl_at(flavor, i);
            const char* want = expected_text(r, i);
            CHECK(strcmp(d.key, ckc::ROCKE_LL_INTRINSIC_DECLS[i].key) == 0,
                  "a resolved table keeps the base order",
                  d.key);
            CHECK(d.decl != NULL && want != NULL && strcmp(d.decl, want) == 0,
                  "a flavor resolves its own override, else its parent's text",
                  d.key);
            CHECK(ckc::rocke_ll_decl_text(flavor, d.key) == d.decl,
                  "key lookup and table order agree",
                  d.key);
        }
    }

    /* No silent fallback for a flavor that is not on the ladder. */
    CHECK(ckc::rocke_ll_decl_text(ROCKE_LLVM_FLAVOR_AUTO, ckc::ROCKE_LL_INTRINSIC_DECLS[0].key)
              == NULL,
          "AUTO has no declaration table",
          ckc::ROCKE_LL_INTRINSIC_DECLS[0].key);

    if(g_failures == 0)
        printf("flavor_decl_chain: OK (%d flavors, %d keys)\n", rungs, nbase);
    return g_failures == 0 ? 0 : 1;
}
