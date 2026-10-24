// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/instances/test_attention_dense_spec.cpp -- C parity for the shared
 * dense attention spec contract (rocke/instance_attention_dense_spec.h).
 *
 * The validator and preflight reasons never reach the emitted IR, so the
 * byte-identity gate cannot see a C/Python divergence in them. Both engines are
 * pinned to one file instead: library/tests/golden/attention_dense_spec_vectors.json,
 * embedded into this binary at build time (attention_dense_spec_vectors.inc, generated
 * by tests/CMakeLists.txt). The Python side,
 * library/tests/test_attention_dense_spec_vectors.py, recomputes every case with
 * the current Python and owns regenerating the file; this test requires the same
 * answers from the C++ port. A Python change therefore fails there first, and the
 * regenerated file then fails here until the C++ port is updated to match.
 */
#include <cstdio>
#include <cstring>
#include <string>

#include "rocke/arena.h"
#include "rocke/instance_attention_dense_spec.h"
#include "rocke/json_dom.h"

namespace
{

const char k_vectors_json[] = {
#include "attention_dense_spec_vectors.inc"
    0};

int g_failures = 0;

#define CHECK(cond)                                                                          \
    do                                                                                       \
    {                                                                                        \
        if(!(cond))                                                                          \
        {                                                                                    \
            std::fprintf(stderr, "FAIL %s: %s (%s:%d)\n", where, #cond, __FILE__, __LINE__); \
            ++g_failures;                                                                    \
        }                                                                                    \
    } while(0)

#define CHECK_STR(got, want)                                          \
    do                                                                \
    {                                                                 \
        const char* got_ = (got);                                     \
        const char* want_ = (want);                                   \
        if(want_ == nullptr || std::strcmp(got_, want_) != 0)         \
        {                                                             \
            std::fprintf(stderr,                                      \
                         "FAIL %s (%s:%d)\n  got:  %s\n  want: %s\n", \
                         where,                                       \
                         __FILE__,                                    \
                         __LINE__,                                    \
                         got_,                                        \
                         want_ ? want_ : "(missing)");                \
            ++g_failures;                                             \
        }                                                             \
    } while(0)

const rocke_attention_dense_traits_t* traits()
{
    static const rocke_attention_dense_traits_t t = rocke_attention_dense_traits_base();
    return &t;
}

bool validate_base(const void* ctx, char* reason, size_t cap)
{
    return rocke_attention_dense_spec_validate(
        static_cast<const rocke_attention_dense_spec_t*>(ctx), traits(), reason, cap);
}

bool json_bool(const jd_val_t* v, bool* out)
{
    if(v == nullptr || v->kind != JD_BOOL)
        return false;
    *out = v->b;
    return true;
}

bool json_int(const jd_val_t* v, int* out)
{
    double d;
    if(!rocke_jnum(v, &d))
        return false;
    *out = static_cast<int>(d);
    return static_cast<double>(*out) == d;
}

/* Optional[str]: a string, or JSON null for Python None. */
bool json_opt_str(const jd_val_t* v, const char** out)
{
    if(v == nullptr)
        return false;
    if(v->kind == JD_NULL)
    {
        *out = nullptr;
        return true;
    }
    *out = rocke_jstr(v);
    return *out != nullptr;
}

/* Fill one spec field from its JSON value. Returns false for a field this C++
 * port does not know: the Python dataclass gained or renamed a field, and the C
 * struct has to follow. */
bool set_field(rocke_attention_dense_spec_t* s, const char* name, const jd_val_t* v)
{
    struct IntField
    {
        const char* name;
        int rocke_attention_dense_spec_t::*member;
    };
    struct BoolField
    {
        const char* name;
        bool rocke_attention_dense_spec_t::*member;
    };
    struct StrField
    {
        const char* name;
        const char* rocke_attention_dense_spec_t::*member;
    };
    static const IntField ints[] = {
        {"batch", &rocke_attention_dense_spec_t::batch},
        {"seqlen_q", &rocke_attention_dense_spec_t::seqlen_q},
        {"seqlen_kv", &rocke_attention_dense_spec_t::seqlen_kv},
        {"num_query_heads", &rocke_attention_dense_spec_t::num_query_heads},
        {"num_kv_heads", &rocke_attention_dense_spec_t::num_kv_heads},
        {"head_size", &rocke_attention_dense_spec_t::head_size},
        {"sliding_window", &rocke_attention_dense_spec_t::sliding_window},
        {"block_m", &rocke_attention_dense_spec_t::block_m},
        {"block_n", &rocke_attention_dense_spec_t::block_n},
        {"waves_per_eu", &rocke_attention_dense_spec_t::waves_per_eu},
        {"lds_k_group_pad", &rocke_attention_dense_spec_t::lds_k_group_pad},
        {"num_persistent", &rocke_attention_dense_spec_t::num_persistent},
        {"block_size", &rocke_attention_dense_spec_t::block_size},
        {"num_kv_blocks", &rocke_attention_dense_spec_t::num_kv_blocks},
    };
    static const BoolField bools[] = {
        {"causal", &rocke_attention_dense_spec_t::causal},
        {"ragged", &rocke_attention_dense_spec_t::ragged},
        {"varlen", &rocke_attention_dense_spec_t::varlen},
        {"persistent", &rocke_attention_dense_spec_t::persistent},
        {"interleave", &rocke_attention_dense_spec_t::interleave},
        {"lazy_rescale", &rocke_attention_dense_spec_t::lazy_rescale},
        {"paged", &rocke_attention_dense_spec_t::paged},
        {"use_sinks", &rocke_attention_dense_spec_t::use_sinks},
        {"causal_bottom_right", &rocke_attention_dense_spec_t::causal_bottom_right},
    };
    static const StrField strs[] = {
        {"dtype", &rocke_attention_dense_spec_t::dtype},
        {"persist_decode", &rocke_attention_dense_spec_t::persist_decode},
    };
    for(const auto& f : ints)
        if(std::strcmp(f.name, name) == 0)
            return json_int(v, &(s->*f.member));
    for(const auto& f : bools)
        if(std::strcmp(f.name, name) == 0)
            return json_bool(v, &(s->*f.member));
    for(const auto& f : strs)
        if(std::strcmp(f.name, name) == 0)
            return json_opt_str(v, &(s->*f.member));
    return false;
}

void check_case(int index, const jd_val_t* c)
{
    char where[32];
    std::snprintf(where, sizeof(where), "case %d", index);

    const jd_val_t* spec_json = rocke_jget(c, "spec");
    const jd_val_t* expect = rocke_jget(c, "expect");
    CHECK(spec_json != nullptr && spec_json->kind == JD_OBJ && expect != nullptr);
    if(spec_json == nullptr || spec_json->kind != JD_OBJ || expect == nullptr)
        return;

    rocke_attention_dense_spec_t s = rocke_attention_dense_spec_default();
    for(int i = 0; i < spec_json->obj_len; ++i)
    {
        const char* name = spec_json->obj[i].key;
        if(!set_field(&s, name, spec_json->obj[i].val))
        {
            std::fprintf(stderr,
                         "FAIL %s: spec field '%s' is unknown to the C++ port or has the "
                         "wrong type; mirror the Python dataclass change in "
                         "rocke_attention_dense_spec_t\n",
                         where,
                         name);
            ++g_failures;
            return;
        }
    }

    bool want_valid = false;
    bool want_preflight = false;
    CHECK(json_bool(rocke_jget(expect, "valid"), &want_valid));
    CHECK(json_bool(rocke_jget(expect, "preflight_ok"), &want_preflight));

    char reason[ROCKE_ATTENTION_DENSE_REASON_CAP] = "";
    const bool valid = rocke_attention_dense_spec_validate(&s, traits(), reason, sizeof(reason));
    CHECK(valid == want_valid);
    CHECK_STR(valid ? "" : reason, rocke_jstr(rocke_jget(expect, "reason")));

    char pre[ROCKE_ATTENTION_DENSE_REASON_CAP] = "x";
    const bool pok = rocke_attention_dense_spec_preflight(
        &s, traits()->type_name, validate_base, &s, pre, sizeof(pre));
    CHECK(pok == want_preflight);
    CHECK_STR(pre, rocke_jstr(rocke_jget(expect, "preflight_reason")));

    if(!want_valid)
        return;
    rocke_attention_dense_name_parts_t parts;
    rocke_attention_dense_name_parts_base(&s, &parts);
    char name[ROCKE_ATTENTION_DENSE_NAME_CAP];
    CHECK(rocke_attention_dense_kernel_name(&s, &parts, name, sizeof(name)) == ROCKE_OK);
    CHECK_STR(name, rocke_jstr(rocke_jget(expect, "kernel_name")));
    CHECK_STR(rocke_attention_dense_spec_resolved_persist_decode(&s),
              rocke_jstr(rocke_jget(expect, "resolved_persist_decode")));
    int want_waves = -1;
    int want_qpk = -1;
    CHECK(json_int(rocke_jget(expect, "num_waves"), &want_waves));
    CHECK(json_int(rocke_jget(expect, "num_queries_per_kv"), &want_qpk));
    CHECK(rocke_attention_dense_spec_num_waves(&s) == want_waves);
    CHECK(rocke_attention_dense_spec_num_queries_per_kv(&s) == want_qpk);
}

void check_vectors()
{
    const char* where = "vectors file";
    rocke_arena_t arena;
    CHECK(rocke_arena_init(&arena, 0) == 0);
    char err[256] = "";
    const jd_val_t* root = rocke_json_parse(k_vectors_json, &arena, err, sizeof(err));
    if(root == nullptr)
    {
        std::fprintf(stderr, "FAIL: attention_dense_spec_vectors.json does not parse: %s\n", err);
        ++g_failures;
        rocke_arena_destroy(&arena);
        return;
    }
    const jd_val_t* cases = rocke_jget(root, "cases");
    CHECK(cases != nullptr && cases->kind == JD_ARR && cases->arr_len > 0);
    if(cases != nullptr && cases->kind == JD_ARR)
    {
        for(int i = 0; i < cases->arr_len; ++i)
            check_case(i, cases->arr[i]);
        std::printf("checked %d cases from attention_dense_spec_vectors.json\n", cases->arr_len);
    }
    rocke_arena_destroy(&arena);
}

void check_defaults_and_tables()
{
    const char* where = "defaults";
    const rocke_attention_dense_spec_t s = rocke_attention_dense_spec_default();
    CHECK(s.causal && !s.ragged && !s.varlen && s.lazy_rescale && !s.persistent);
    CHECK(s.block_m == 256 && s.block_n == 64 && s.waves_per_eu == 2);
    CHECK(s.lds_k_group_pad == 8 && s.num_persistent == 256);
    CHECK_STR(s.dtype, "bf16");
    CHECK_STR(s.persist_decode, "auto");
    CHECK_STR(ROCKE_DENSE_TILE_GEOMETRIES[0].name, "default");
    CHECK(ROCKE_DENSE_TILE_GEOMETRIES[0].block_m == 256
          && ROCKE_DENSE_TILE_GEOMETRIES[0].block_n == 64);
    CHECK_STR(ROCKE_DENSE_TILE_GEOMETRIES[1].name, "bm128");
    CHECK(ROCKE_DENSE_TILE_GEOMETRIES[1].block_m == 128
          && ROCKE_DENSE_TILE_GEOMETRIES[1].block_n == 64);
}

void check_name_buffer_too_small()
{
    const char* where = "name buffer";
    rocke_attention_dense_spec_t s = rocke_attention_dense_spec_default();
    s.batch = 1;
    s.seqlen_q = s.seqlen_kv = 512;
    s.num_query_heads = 8;
    s.num_kv_heads = 2;
    s.head_size = 128;
    rocke_attention_dense_name_parts_t parts;
    rocke_attention_dense_name_parts_base(&s, &parts);
    char name[8];
    CHECK(rocke_attention_dense_kernel_name(&s, &parts, name, sizeof(name)) == ROCKE_ERR_VALUE);
    CHECK(rocke_attention_dense_kernel_name(NULL, &parts, name, sizeof(name)) == ROCKE_ERR_VALUE);
}

} // namespace

int main()
{
    check_vectors();
    check_defaults_and_tables();
    check_name_buffer_too_small();
    if(g_failures != 0)
    {
        std::fprintf(stderr, "rocke_attention_dense_spec: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("rocke_attention_dense_spec: OK\n");
    return 0;
}
