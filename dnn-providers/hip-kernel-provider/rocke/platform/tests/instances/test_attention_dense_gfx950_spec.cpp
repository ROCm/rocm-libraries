// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Concrete gfx950 parity: only raw inputs cross into the implementation. Shared
// hook fixtures are not an oracle for gfx950. Optional --vectors FILE accepts
// the same oracle schema for live Python/native smoke cases, without pybind.
#include "rocke/instance_gfx950_attention_dense.h"
#include "rocke/json_dom.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

namespace
{
const char vectors[] = {
#include "attention_dense_gfx950_spec_vectors.inc"
    0};
using Spec = rocke_gfx950_attention_dense_spec_t;
using Base = rocke_attention_dense_spec_t;
int failures = 0;
std::string where;
#define CHECK(x)                                                                          \
    do                                                                                    \
    {                                                                                     \
        if(!(x))                                                                          \
        {                                                                                 \
            std::fprintf(stderr, "FAIL %s: %s (line %d)\n", where.c_str(), #x, __LINE__); \
            ++failures;                                                                   \
        }                                                                                 \
    } while(0)
const jd_val_t* get(const jd_val_t* v, const char* k)
{
    return rocke_jget(v, k);
}
bool eqstr(const char* a, const jd_val_t* b)
{
    const char* s = rocke_jstr(b);
    return a && s && std::strcmp(a, s) == 0;
}
bool eqnum(int64_t a, const jd_val_t* b)
{
    double d;
    return rocke_jnum(b, &d) && static_cast<double>(a) == d;
}
bool eqbool(bool a, const jd_val_t* b)
{
    return b && b->kind == JD_BOOL && a == b->b;
}
bool arr(const jd_val_t* v, size_t n)
{
    return v && v->kind == JD_ARR && static_cast<size_t>(v->arr_len) == n;
}
enum Kind
{
    Integer,
    Boolean,
    String,
    Float,
    OptInt,
    OptBool
};
struct Field
{
    const char* name;
    Kind kind;
    size_t offset;
    size_t presence;
};
#define B(n, k)                                            \
    {                                                      \
        #n, k, offsetof(Spec, base) + offsetof(Base, n), 0 \
    }
#define F(n, k)                     \
    {                               \
        #n, k, offsetof(Spec, n), 0 \
    }
#define O(n, k)                                           \
    {                                                     \
        #n, k, offsetof(Spec, n), offsetof(Spec, has_##n) \
    }
const Field fields[] = {B(batch, Integer),
                        B(seqlen_q, Integer),
                        B(seqlen_kv, Integer),
                        B(num_query_heads, Integer),
                        B(num_kv_heads, Integer),
                        B(head_size, Integer),
                        B(sliding_window, Integer),
                        B(block_m, Integer),
                        B(block_n, Integer),
                        B(waves_per_eu, Integer),
                        B(lds_k_group_pad, Integer),
                        B(num_persistent, Integer),
                        B(block_size, Integer),
                        B(num_kv_blocks, Integer),
                        B(chiplet_num_xcds, Integer),
                        B(causal, Boolean),
                        B(ragged, Boolean),
                        B(varlen, Boolean),
                        B(persistent, Boolean),
                        B(interleave, Boolean),
                        B(lazy_rescale, Boolean),
                        B(paged, Boolean),
                        B(use_sinks, Boolean),
                        B(causal_bottom_right, Boolean),
                        B(dtype, String),
                        B(persist_decode, String),
                        B(nonpersist_decode, String),
                        F(lds_v_row_pad, Integer),
                        F(wide_lds_dma, Boolean),
                        F(emit_lse, Boolean),
                        F(right_bound, Integer),
                        F(kv_segments, Integer),
                        F(kv_storage_dtype, String),
                        F(lds_num_buffers, Integer),
                        F(lds_k_row_pad, Integer),
                        F(lazy_rescale_threshold, Float),
                        F(use_exp2_fast, Boolean),
                        O(exp_per_pv_step, OptInt),
                        F(partial_vmcnt_prefetch, Boolean),
                        F(pv_priority, Integer),
                        O(pv_sched_fence, OptBool),
                        F(pv_sched_fence_mask, Integer),
                        O(pv_sched_group_template, OptBool),
                        F(pv_sched_group_ds_read, Integer),
                        O(iglp_mode, OptInt),
                        F(pv_loop_order, String),
                        F(causal_diag_split, Boolean),
                        F(o_store_width, Integer)};
#undef B
#undef F
#undef O
bool field(Spec& s, const Field& f, const jd_val_t* v, bool assign)
{
    if(!v)
        return false;
    char* p = reinterpret_cast<char*>(&s) + f.offset;
    if(f.kind == OptInt || f.kind == OptBool)
    {
        bool* present = reinterpret_cast<bool*>(reinterpret_cast<char*>(&s) + f.presence);
        if(assign)
            *present = v->kind != JD_NULL;
        if(*present != (v->kind != JD_NULL))
            return false;
        if(!*present)
            return true;
    }
    if(f.kind == Integer || f.kind == OptInt)
    {
        double d;
        if(!rocke_jnum(v, &d) || d < std::numeric_limits<int>::min()
           || d > std::numeric_limits<int>::max())
            return false;
        int n = static_cast<int>(d);
        if(n != d)
            return false;
        if(assign)
            *reinterpret_cast<int*>(p) = n;
        return *reinterpret_cast<int*>(p) == n;
    }
    if(f.kind == Boolean || f.kind == OptBool)
    {
        if(v->kind != JD_BOOL)
            return false;
        if(assign)
            *reinterpret_cast<bool*>(p) = v->b;
        return *reinterpret_cast<bool*>(p) == v->b;
    }
    if(f.kind == Float)
    {
        double d;
        if(!rocke_jnum(v, &d))
            return false;
        if(assign)
            *reinterpret_cast<double*>(p) = d;
        return *reinterpret_cast<double*>(p) == d;
    }
    const char* text = v->kind == JD_NULL ? nullptr : rocke_jstr(v);
    if(v->kind != JD_NULL && !text)
        return false;
    auto out = reinterpret_cast<const char**>(p);
    if(assign)
        *out = text;
    return (!text && !*out) || (text && *out && std::strcmp(text, *out) == 0);
}
bool load(Spec& s, const jd_val_t* v)
{
    if(!v || v->kind != JD_OBJ)
        return false;
    for(int i = 0; i < v->obj_len; ++i)
    {
        bool known = false;
        for(const auto& f : fields)
            if(std::strcmp(f.name, v->obj[i].key) == 0)
            {
                known = true;
                if(!field(s, f, v->obj[i].val, true))
                    return false;
                break;
            }
        if(!known)
            return false;
    }
    return true;
}
void strings(const char* const* values, size_t n, const jd_val_t* want)
{
    CHECK(arr(want, n));
    if(!arr(want, n))
        return;
    for(size_t i = 0; i < n; ++i)
        CHECK(eqstr(values[i], want->arr[i]));
}
void parts(const char values[][ROCKE_ATTENTION_DENSE_NAME_PART_CAP], int n, const jd_val_t* want)
{
    CHECK(arr(want, n));
    if(!arr(want, n))
        return;
    for(int i = 0; i < n; ++i)
        CHECK(eqstr(values[i], want->arr[i]));
}
void check_case(const jd_val_t* c, const jd_val_t* root, rocke_arena_t* arena)
{
    Spec s = rocke_gfx950_attention_dense_spec_default();
    if(const auto* base = get(root, "base"))
        CHECK(load(s, base));
    CHECK(load(s, get(c, "spec")));
    const auto* e = get(c, "expect");
    CHECK(eqnum(rocke_gfx950_attention_dense_kv_elem_bytes(&s), get(e, "kv_elem_bytes")));
    char reason[4096] = {};
    bool valid = rocke_gfx950_attention_dense_spec_validate(&s, reason, sizeof(reason));
    CHECK(eqbool(valid, get(e, "valid")));
    const auto* validation_reason = get(e, "reason");
    CHECK(validation_reason);
    if(validation_reason && validation_reason->kind != JD_NULL)
        CHECK(eqstr(reason, validation_reason));
    bool ok = rocke_gfx950_supports_attention_dense(
        &s, rocke_jstr(get(c, "arch")), reason, sizeof(reason));
    CHECK(eqbool(ok, get(e, "supports")));
    const auto* support_reason = get(e, "supports_reason");
    CHECK(support_reason);
    if(support_reason && support_reason->kind != JD_NULL)
        CHECK(eqstr(reason, support_reason));
    if(!ok)
        return;
#define NUM(n) CHECK(eqnum(rocke_gfx950_attention_dense_##n(&s), get(e, #n)))
#define BOOL(n) CHECK(eqbool(rocke_gfx950_attention_dense_##n(&s), get(e, #n)))
    NUM(resolved_exp_per_pv_step);
    NUM(resolved_iglp_mode);
    BOOL(resolved_pv_sched_fence);
    BOOL(resolved_pv_sched_group_template);
    BOOL(runtime_shape);
#undef NUM
#undef BOOL
    CHECK(eqstr(rocke_gfx950_attention_dense_resolved_pv_loop_order(&s),
                get(e, "resolved_pv_loop_order")));
    CHECK(eqstr(rocke_attention_dense_spec_resolved_persist_decode(&s.base, s.right_bound != 0),
                get(e, "resolved_persist_decode")));
    CHECK(eqstr(rocke_attention_dense_spec_resolved_nonpersist_decode(&s.base, s.right_bound != 0),
                get(e, "resolved_nonpersist_decode")));
    rocke_gfx950_dense_fields_t names{};
    CHECK(rocke_gfx950_attention_dense_runtime_param_fields(&s, &names) == ROCKE_OK);
    strings(names.names, names.count, get(e, "runtime_param_fields"));
    CHECK(rocke_gfx950_attention_dense_shape_kernarg_fields(&s, &names) == ROCKE_OK);
    strings(names.names, names.count, get(e, "shape_kernarg_fields"));
    char name[4096];
    CHECK(rocke_gfx950_attention_dense_kernel_name(&s, name, sizeof(name)) == ROCKE_OK);
    CHECK(eqstr(name, get(e, "kernel_name")));
    rocke_attention_dense_name_parts_t p{};
    CHECK(rocke_gfx950_attention_dense_name_parts(&s, &p) == ROCKE_OK);
    const auto* np = get(e, "name_parts");
    CHECK(eqbool(p.head_parts, get(np, "head_parts")));
    CHECK(eqbool(p.shape_parts, get(np, "shape_parts")));
    parts(p.layout, p.num_layout, get(np, "layout"));
    parts(p.band, p.num_band, get(np, "band"));
    parts(p.algorithm, p.num_algorithm, get(np, "algorithm"));
    // Only the active grid's tag contributes to the name. The shared C metadata
    // also retains the inactive tag, which Python's fixture deliberately omits.
    CHECK(eqstr(s.base.persistent ? p.persist_decode : p.nonpersist_decode,
                get(np, s.base.persistent ? "persist_decode" : "nonpersist_decode")));
    int64_t lds = 0;
    CHECK(rocke_gfx950_dense_static_lds_bytes(s.base.head_size,
                                              s.base.block_n,
                                              s.lds_k_row_pad,
                                              s.lds_v_row_pad,
                                              s.base.lds_k_group_pad,
                                              s.wide_lds_dma,
                                              &lds)
          == ROCKE_OK);
    CHECK(eqnum(lds, get(e, "lds_bytes")));
    int dims[3];
    CHECK(rocke_gfx950_attention_dense_grid(&s, dims) == ROCKE_OK);
    const auto* grid = get(e, "grid");
    CHECK(arr(grid, 3));
    if(arr(grid, 3))
        for(int i = 0; i < 3; ++i)
            CHECK(eqnum(dims[i], grid->arr[i]));
    CHECK(rocke_gfx950_attention_dense_block(&s, dims) == ROCKE_OK);
    const auto* block = get(e, "block");
    CHECK(arr(block, 3));
    if(arr(block, 3))
        for(int i = 0; i < 3; ++i)
            CHECK(eqnum(dims[i], block->arr[i]));
    const rocke_sig_entry_t* sig = nullptr;
    size_t n = 0;
    CHECK(rocke_gfx950_attention_dense_signature(arena, &s, &sig, &n) == ROCKE_OK);
    const auto* signature = get(e, "signature");
    // An external live case may instead carry the full Python signature inline.
    if(!signature)
    {
        const auto* table = get(root, "signatures");
        double id;
        if(table && table->kind == JD_ARR && rocke_jnum(get(e, "signature_id"), &id) && id >= 0
           && id < table->arr_len && static_cast<int>(id) == id)
            signature = table->arr[static_cast<int>(id)];
    }
    CHECK(arr(signature, n));
    if(arr(signature, n))
        for(size_t i = 0; i < n; ++i)
        {
            CHECK(eqstr(sig[i].name, get(signature->arr[i], "name")));
            CHECK(eqstr(sig[i].type, get(signature->arr[i], "type")));
        }
    rocke_gfx950_dense_runtime_args_t args{};
    bool runtime_ok = rocke_gfx950_attention_dense_runtime_args(&s, &args, reason, sizeof(reason));
    const auto* expected_args = get(e, "runtime_args");
    CHECK(expected_args && runtime_ok == (expected_args->kind != JD_NULL));
    CHECK(eqstr(reason, get(e, "runtime_args_reason")));
    if(runtime_ok)
    {
        CHECK(arr(expected_args, args.count));
        if(arr(expected_args, args.count))
            for(size_t i = 0; i < args.count; ++i)
            {
                const auto* a = expected_args->arr[i];
                CHECK(arr(a, 2));
                if(arr(a, 2))
                {
                    CHECK(eqstr(args.items[i].name, a->arr[0]));
                    CHECK(eqnum(args.items[i].value, a->arr[1]));
                }
            }
    }
}
void check_vectors(const char* text)
{
    rocke_arena_t arena;
    where = "JSON";
    CHECK(rocke_arena_init(&arena, 0) == 0);
    char error[256] = {};
    const auto* root = rocke_json_parse(text, &arena, error, sizeof(error));
    if(!root)
    {
        std::fprintf(stderr, "JSON: %s\n", error);
        ++failures;
        rocke_arena_destroy(&arena);
        return;
    }
    const auto* defaults = get(root, "defaults");
    if(defaults)
    {
        where = "defaults";
        Spec s = rocke_gfx950_attention_dense_spec_default();
        for(const auto& f : fields)
            CHECK(field(s, f, get(defaults, f.name), false) || !get(defaults, f.name));
        const auto* fs = get(root, "fields");
        CHECK(arr(fs, sizeof(fields) / sizeof(fields[0])));
        if(fs && fs->kind == JD_ARR)
            for(int i = 0; i < fs->arr_len; ++i)
            {
                bool found = false;
                for(const auto& f : fields)
                    found = found || eqstr(f.name, fs->arr[i]);
                CHECK(found);
            }
    }
    const auto* legality = get(root, "tile_legality");
    if(legality)
        for(int d : {0, 64, 128})
        {
            where = "tile legality D" + std::to_string(d);
            int tiles[ROCKE_GFX950_DENSE_MAX_TILES][2];
            size_t n = rocke_gfx950_dense_tile_legality(d, tiles);
            const auto* expected = get(legality, std::to_string(d).c_str());
            CHECK(arr(expected, n));
            if(arr(expected, n))
                for(size_t i = 0; i < n; ++i)
                {
                    CHECK(arr(expected->arr[i], 2));
                    if(arr(expected->arr[i], 2))
                        for(int j = 0; j < 2; ++j)
                            CHECK(eqnum(tiles[i][j], expected->arr[i]->arr[j]));
                }
        }
    const auto* cases = get(root, "cases");
    CHECK(cases && cases->kind == JD_ARR);
    if(cases && cases->kind == JD_ARR)
        for(int i = 0; i < cases->arr_len; ++i)
        {
            where = "case " + std::to_string(i);
            check_case(cases->arr[i], root, &arena);
        }
    rocke_arena_destroy(&arena);
}
} // namespace
int main(int argc, char** argv)
{
    std::string input;
    if(argc == 3 && std::strcmp(argv[1], "--vectors") == 0)
    {
        std::ifstream f(argv[2]);
        if(!f)
        {
            std::fprintf(stderr, "cannot read %s\n", argv[2]);
            return 2;
        }
        input.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    else if(argc != 1)
    {
        std::fprintf(stderr, "usage: %s [--vectors FILE]\n", argv[0]);
        return 2;
    }
    check_vectors(input.empty() ? vectors : input.c_str());
    if(failures)
    {
        std::fprintf(stderr, "gfx950 spec: %d failures\n", failures);
        return 1;
    }
    std::puts("rocke_attention_dense_gfx950_spec: OK");
    return 0;
}
