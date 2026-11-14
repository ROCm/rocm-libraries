// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The attention backward validators re-run __post_init__.
//
// A Python spec is frozen and checked at construction, so its validator never
// sees a bad field. The C spec is a plain struct: a caller can edit a field
// after *_spec_init, or skip init. Without the re-check, a misspelled
// page_mode built a "direct" kernel and a misspelled persist_decode built a
// kv_tile_major one, both with status 0. Each validator (which every builder
// calls first) must reject such a spec with init's own message.

#include <cstdio>
#include <cstring>
#include <string>

#include "rocke/instance_attention_bwd_unified.h"
#include "rocke/instance_attention_dense_bwd_dq.h"
#include "rocke/instance_attention_dense_bwd_spec.h"

static int g_failures = 0;

static void expect(bool ok, const std::string& what)
{
    if(!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// status OK, or report err.msg
static bool ok(rocke_status_t st, const rocke_attn_bwd_err_t& e, const std::string& what)
{
    if(st == ROCKE_OK)
        return true;
    std::fprintf(stderr, "FAIL: %s: %s\n", what.c_str(), e.msg);
    ++g_failures;
    return false;
}

// a refusal whose message names the field
static void refused(rocke_status_t st,
                    const rocke_attn_bwd_err_t& e,
                    const char* field,
                    const std::string& what)
{
    expect(st != ROCKE_OK, what + ": accepted (status 0)");
    if(st != ROCKE_OK)
        expect(std::strstr(e.msg, field) != nullptr,
               what + ": message does not name " + field + ": " + e.msg);
}

// The problem fields Python requires (the defaults leave them unset).
template <class S>
static void problem(S& s)
{
    s.seqlen_q = 256;
    s.seqlen_kv = 256;
    s.num_query_heads = 8;
    s.num_kv_heads = 8;
    s.head_size = 64;
}

static void unified()
{
    rocke_attn_bwd_err_t e{};
    rocke_attn_bwd_unified_spec_t u = rocke_attn_bwd_unified_spec_default();
    u.head_size = 64;
    if(!ok(rocke_attn_bwd_unified_spec_init(&u, &e), e, "unified init"))
        return;
    rocke_attn_bwd_unified_spec_t out;
    ok(rocke_attn_bwd_unified_validate_spec(&u, "gfx950", &out, &e), e, "unified validate");

    rocke_attn_bwd_unified_spec_t bad = u;
    bad.page_mode = "atomc"; // edited after init
    refused(rocke_attn_bwd_unified_validate_spec(&bad, "gfx950", &out, &e),
            e,
            "page_mode",
            "unified validate, page_mode edited after init");

    // never initialised: init would refuse it; validate must refuse it the
    // same way (init edits its argument, so it runs on a copy)
    rocke_attn_bwd_unified_spec_t raw = rocke_attn_bwd_unified_spec_default();
    raw.head_size = 64;
    raw.page_mode = "atomc";
    rocke_attn_bwd_err_t init_e{};
    rocke_attn_bwd_unified_spec_t probe = raw;
    rocke_attn_bwd_unified_spec_init(&probe, &init_e);
    refused(rocke_attn_bwd_unified_validate_spec(&raw, "gfx950", &out, &e),
            e,
            "page_mode",
            "unified validate, page_mode without init");
    expect(std::strcmp(e.msg, init_e.msg) == 0,
           std::string("unified: validate's message differs from init's: ") + e.msg + " vs "
               + init_e.msg);
}

static void dense()
{
    rocke_attn_bwd_err_t e{};
    rocke_attn_bwd_dense_spec_t d = rocke_attn_bwd_dense_spec_default("gfx950");
    problem(d);
    if(!ok(rocke_attn_bwd_dense_spec_init(&d, &e), e, "dense init"))
        return;
    rocke_attn_bwd_spec_t out;
    ok(rocke_attn_bwd_dense_validate(&d, "gfx950", &out, &e), e, "dense validate");

    rocke_attn_bwd_dense_spec_t bad = d;
    bad.persistent = true;
    bad.persist_decode = "hkv_majr"; // edited after init
    refused(rocke_attn_bwd_dense_validate(&bad, "gfx950", &out, &e),
            e,
            "persist_decode",
            "dense validate, persist_decode edited after init");

    rocke_attn_bwd_err_t init_e{};
    rocke_attn_bwd_dense_spec_init(&bad, &init_e);
    expect(std::strcmp(e.msg, init_e.msg) == 0,
           std::string("dense: validate's message differs from init's: ") + e.msg + " vs "
               + init_e.msg);
}

static void dense_dq()
{
    rocke_attn_bwd_err_t e{};
    rocke_attn_bwd_dense_dq_spec_t d = rocke_attn_bwd_dense_dq_spec_default();
    problem(d);
    if(!ok(rocke_attn_bwd_dense_dq_spec_init(&d, &e), e, "dense dQ init"))
        return;
    rocke_attn_bwd_dq_spec_t out;
    ok(rocke_attn_bwd_dense_dq_validate(&d, "gfx950", &out, &e), e, "dense dQ validate");

    rocke_attn_bwd_dense_dq_spec_t bad = d;
    bad.dtype = "fp61"; // edited after init
    refused(rocke_attn_bwd_dense_dq_validate(&bad, "gfx950", &out, &e),
            e,
            "dtype",
            "dense dQ validate, dtype edited after init");
}

int main()
{
    unified();
    dense();
    dense_dq();
    if(g_failures)
    {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("OK: unified, dense and dense dQ validators re-run __post_init__\n");
    return 0;
}
