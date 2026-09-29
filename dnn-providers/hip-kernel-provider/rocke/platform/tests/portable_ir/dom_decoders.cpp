// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/portable_ir/dom_decoders.cpp -- unit tests for the portable-IR DOM
 * decoders (cpp/portable_ir/json_dom.cpp and cbor_dom.cpp).
 *
 * Pure data level, no kernels: builds CBOR blobs by hand, decodes them, and
 * checks the resulting jd_val_t tree. The load-bearing case is
 * test_json_cbor_equivalence -- everything downstream (recipe VM, bundle
 * reader, IR importer) walks jd_val_t and never learns which wire format it
 * came from, so "CBOR decodes to the same DOM as the equivalent JSON" is the
 * single property that lets the compact shipping format reuse the JSON
 * consumers verbatim.
 *
 * Also pins the failure behaviour: malformed input must return NULL with a
 * diagnostic, never read past the buffer. A runtime that loads artifacts off
 * disk has to survive a truncated file.
 */
#include <cmath>
#include <cstdio>
#include <cstring>

#include "rocke/arena.h"
#include "rocke/cbor_dom.h"
#include "rocke/json_dom.h"

namespace
{

int g_fail = 0;

void check(bool cond, const char* msg)
{
    if(!cond)
    {
        std::printf("  FAIL: %s\n", msg);
        ++g_fail;
    }
}

bool num_eq(const jd_val_t* v, double want)
{
    double d = 0.0;
    return v && rocke_jnum(v, &d) && d == want;
}

// Structural equality of two DOM trees. Order-sensitive for arrays AND maps:
// both decoders preserve insertion order, and the recipe VM relies on it (a
// recipe program is an ordered list).
bool jd_equal(const jd_val_t* a, const jd_val_t* b)
{
    if(!a || !b || a->kind != b->kind)
    {
        return false;
    }
    switch(a->kind)
    {
    case JD_NULL:
        return true;
    case JD_BOOL:
        return a->b == b->b;
    case JD_NUM:
        return a->num == b->num;
    case JD_STR:
        return std::strcmp(a->str, b->str) == 0;
    case JD_ARR:
        if(a->arr_len != b->arr_len)
        {
            return false;
        }
        for(int i = 0; i < a->arr_len; ++i)
        {
            if(!jd_equal(a->arr[i], b->arr[i]))
            {
                return false;
            }
        }
        return true;
    case JD_OBJ:
        if(a->obj_len != b->obj_len)
        {
            return false;
        }
        for(int i = 0; i < a->obj_len; ++i)
        {
            if(std::strcmp(a->obj[i].key, b->obj[i].key) != 0
               || !jd_equal(a->obj[i].val, b->obj[i].val))
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

void test_scalars(rocke_arena_t* a)
{
    char err[128];

    // uint 42: major 0, 1-byte argument.
    const unsigned char u42[] = {0x18, 0x2A};
    const jd_val_t* v = rocke_cbor_parse(u42, sizeof u42, a, err, sizeof err);
    check(v && v->kind == JD_NUM && num_eq(v, 42), "uint 42");

    // uint 300: major 0, 2-byte argument (crosses the 1-byte boundary).
    const unsigned char u300[] = {0x19, 0x01, 0x2C};
    check(num_eq(rocke_cbor_parse(u300, sizeof u300, a, err, sizeof err), 300),
          "uint 300 (2-byte arg)");

    // negint -10: major 1, argument 9 -> -1 - 9.
    const unsigned char n10[] = {0x29};
    check(num_eq(rocke_cbor_parse(n10, sizeof n10, a, err, sizeof err), -10), "negint -10");

    // float64 1.5.
    const unsigned char f15[] = {0xFB, 0x3F, 0xF8, 0, 0, 0, 0, 0, 0};
    v = rocke_cbor_parse(f15, sizeof f15, a, err, sizeof err);
    check(v && v->kind == JD_NUM && v->num == 1.5, "float64 1.5");

    // text "f32" -- the shape every dtype attr in a recipe takes.
    const unsigned char s[] = {0x63, 'f', '3', '2'};
    v = rocke_cbor_parse(s, sizeof s, a, err, sizeof err);
    check(v && v->kind == JD_STR && std::strcmp(v->str, "f32") == 0, "text 'f32'");

    const unsigned char t = 0xF5, f = 0xF4, nul = 0xF6;
    v = rocke_cbor_parse(&t, 1, a, err, sizeof err);
    check(v && v->kind == JD_BOOL && v->b, "true");
    v = rocke_cbor_parse(&f, 1, a, err, sizeof err);
    check(v && v->kind == JD_BOOL && !v->b, "false");
    v = rocke_cbor_parse(&nul, 1, a, err, sizeof err);
    check(v && v->kind == JD_NULL, "null");
}

void test_array(rocke_arena_t* a)
{
    char err[128];
    // [1, 2, 3]
    const unsigned char arr[] = {0x83, 0x01, 0x02, 0x03};
    const jd_val_t* v = rocke_cbor_parse(arr, sizeof arr, a, err, sizeof err);
    check(v && v->kind == JD_ARR && v->arr_len == 3, "array length 3");
    check(v && v->arr_len == 3 && num_eq(v->arr[0], 1) && num_eq(v->arr[2], 3),
          "array elements in order");
}

void test_json_cbor_equivalence(rocke_arena_t* a)
{
    char err[128];
    // {"a": 1, "b": [true, null]}
    const unsigned char m[] = {0xA2, 0x61, 'a', 0x01, 0x61, 'b', 0x82, 0xF5, 0xF6};
    const jd_val_t* cv = rocke_cbor_parse(m, sizeof m, a, err, sizeof err);
    check(cv && cv->kind == JD_OBJ && cv->obj_len == 2, "map with two members");
    check(cv && num_eq(rocke_jget(cv, "a"), 1), "map['a'] == 1");
    const jd_val_t* b = cv ? rocke_jget(cv, "b") : nullptr;
    check(b && b->kind == JD_ARR && b->arr_len == 2 && b->arr[0]->kind == JD_BOOL && b->arr[0]->b
              && b->arr[1]->kind == JD_NULL,
          "map['b'] == [true, null]");

    const jd_val_t* jv = rocke_json_parse("{\"a\": 1, \"b\": [true, null]}", a, err, sizeof err);
    check(jv != nullptr, "json parse ok");
    check(jd_equal(cv, jv), "CBOR DOM == JSON DOM");
}

void test_malformed_input_is_rejected(rocke_arena_t* a)
{
    char err[128];

    // Array header claims 3 elements, only 1 follows.
    const unsigned char trunc[] = {0x83, 0x01};
    err[0] = '\0';
    check(rocke_cbor_parse(trunc, sizeof trunc, a, err, sizeof err) == nullptr,
          "truncated array rejected");
    check(err[0] != '\0', "truncated array sets a diagnostic");

    // Text string claims 5 bytes, only 2 follow.
    const unsigned char badstr[] = {0x65, 'h', 'i'};
    check(rocke_cbor_parse(badstr, sizeof badstr, a, err, sizeof err) == nullptr,
          "truncated string rejected");

    check(rocke_cbor_parse(reinterpret_cast<const unsigned char*>(""), 0, a, err, sizeof err)
              == nullptr,
          "empty input rejected");
}

void test_recipe_shaped_map(rocke_arena_t* a)
{
    char err[128];
    // {"op":"emit","in":["x","y"]} -- one instruction of a recipe program.
    const unsigned char r[] = {
        0xA2, 0x62, 'o', 'p', 0x64, 'e', 'm', 'i', 't', 0x62, 'i', 'n', 0x82, 0x61, 'x', 0x61, 'y'};
    const jd_val_t* v = rocke_cbor_parse(r, sizeof r, a, err, sizeof err);
    const char* op = v ? rocke_jstr(rocke_jget(v, "op")) : nullptr;
    const jd_val_t* in = v ? rocke_jget(v, "in") : nullptr;
    check(op && std::strcmp(op, "emit") == 0, "recipe op == emit");
    check(in && in->kind == JD_ARR && in->arr_len == 2
              && std::strcmp(rocke_jstr(in->arr[0]), "x") == 0
              && std::strcmp(rocke_jstr(in->arr[1]), "y") == 0,
          "recipe operands == [x, y]");
}

} // namespace

int main()
{
    rocke_arena_t a;
    if(rocke_arena_init(&a, 0) != 0)
    {
        std::printf("arena init failed\n");
        return 2;
    }
    std::printf("portable_ir dom_decoders:\n");
    test_scalars(&a);
    test_array(&a);
    test_json_cbor_equivalence(&a);
    test_malformed_input_is_rejected(&a);
    test_recipe_shaped_map(&a);
    rocke_arena_destroy(&a);

    if(g_fail == 0)
    {
        std::printf("PASS: all DOM decoder unit tests passed.\n");
        return 0;
    }
    std::printf("FAIL: %d check(s) failed.\n", g_fail);
    return 1;
}
