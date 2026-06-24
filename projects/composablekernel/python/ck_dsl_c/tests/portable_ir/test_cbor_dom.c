/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * test_cbor_dom.c -- unit tests for the C recipe-replay DOM decoders
 * (src/portable_ir/cbor_dom.c and json_dom.c). Pure data-level: builds CBOR
 * blobs by hand, decodes them, and checks the resulting jd_val_t DOM; also
 * proves CBOR and JSON decode to an equal DOM, and that malformed CBOR fails
 * cleanly (no crash). Built + run by run_unit_tests.sh.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/cbor_dom.h"
#include "ckc/json_dom.h"

static int g_fail = 0;
#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            printf("  FAIL: %s\n", (msg));                                        \
            g_fail++;                                                             \
        }                                                                         \
    } while (0)

static int num_eq(const jd_val_t* v, double want)
{
    double d;
    return v && ckc_jnum(v, &d) && d == want;
}

/* Recursive structural equality of two DOM trees (order-sensitive for arrays and
 * maps -- both decoders preserve insertion order). */
static int jd_equal(const jd_val_t* a, const jd_val_t* b)
{
    if (!a || !b || a->kind != b->kind)
        return 0;
    switch (a->kind) {
    case JD_NULL:
        return 1;
    case JD_BOOL:
        return a->b == b->b;
    case JD_NUM:
        return a->num == b->num;
    case JD_STR:
        return strcmp(a->str, b->str) == 0;
    case JD_ARR:
        if (a->arr_len != b->arr_len)
            return 0;
        for (int i = 0; i < a->arr_len; i++)
            if (!jd_equal(a->arr[i], b->arr[i]))
                return 0;
        return 1;
    case JD_OBJ:
        if (a->obj_len != b->obj_len)
            return 0;
        for (int i = 0; i < a->obj_len; i++)
            if (strcmp(a->obj[i].key, b->obj[i].key) != 0
                    || !jd_equal(a->obj[i].val, b->obj[i].val))
                return 0;
        return 1;
    }
    return 0;
}

static void test_scalars(ckc_arena_t* a)
{
    char err[128];
    /* uint 42 = 0x18 0x2A (major0, 1-byte arg) */
    unsigned char u42[] = {0x18, 0x2A};
    jd_val_t* v = ckc_cbor_parse(u42, sizeof u42, a, err, sizeof err);
    CHECK(v && v->kind == JD_NUM && num_eq(v, 42), "uint 42");

    /* uint 300 = 0x19 0x01 0x2C (major0, 2-byte arg) */
    unsigned char u300[] = {0x19, 0x01, 0x2C};
    v = ckc_cbor_parse(u300, sizeof u300, a, err, sizeof err);
    CHECK(num_eq(v, 300), "uint 300 (2-byte)");

    /* negint -10 = 0x29 (major1, arg 9 -> -1-9) */
    unsigned char n10[] = {0x29};
    v = ckc_cbor_parse(n10, sizeof n10, a, err, sizeof err);
    CHECK(num_eq(v, -10), "negint -10");

    /* float64 1.5 = 0xFB 3F F8 00 00 00 00 00 00 */
    unsigned char f15[] = {0xFB, 0x3F, 0xF8, 0, 0, 0, 0, 0, 0};
    v = ckc_cbor_parse(f15, sizeof f15, a, err, sizeof err);
    CHECK(v && v->kind == JD_NUM && v->num == 1.5, "float64 1.5");

    /* text "f32" = 0x63 'f' '3' '2' */
    unsigned char s[] = {0x63, 'f', '3', '2'};
    v = ckc_cbor_parse(s, sizeof s, a, err, sizeof err);
    CHECK(v && v->kind == JD_STR && strcmp(v->str, "f32") == 0, "text 'f32'");

    /* true / false / null */
    unsigned char t = 0xF5, f = 0xF4, nul = 0xF6;
    CHECK((v = ckc_cbor_parse(&t, 1, a, err, sizeof err)) && v->kind == JD_BOOL && v->b,
          "true");
    CHECK((v = ckc_cbor_parse(&f, 1, a, err, sizeof err)) && v->kind == JD_BOOL && !v->b,
          "false");
    CHECK((v = ckc_cbor_parse(&nul, 1, a, err, sizeof err)) && v->kind == JD_NULL, "null");
}

static void test_array(ckc_arena_t* a)
{
    char err[128];
    /* [1, 2, 3] = 0x83 0x01 0x02 0x03 */
    unsigned char arr[] = {0x83, 0x01, 0x02, 0x03};
    jd_val_t* v = ckc_cbor_parse(arr, sizeof arr, a, err, sizeof err);
    CHECK(v && v->kind == JD_ARR && v->arr_len == 3, "array length 3");
    CHECK(v && v->arr_len == 3 && num_eq(v->arr[0], 1) && num_eq(v->arr[2], 3),
          "array elements");
}

static void test_map_and_json_equiv(ckc_arena_t* a)
{
    char err[128];
    /* {"a": 1, "b": [true, null]}
     * = 0xA2 0x61 'a' 0x01 0x61 'b' 0x82 0xF5 0xF6 */
    unsigned char m[] = {0xA2, 0x61, 'a', 0x01, 0x61, 'b', 0x82, 0xF5, 0xF6};
    jd_val_t* cv = ckc_cbor_parse(m, sizeof m, a, err, sizeof err);
    CHECK(cv && cv->kind == JD_OBJ && cv->obj_len == 2, "map two members");
    CHECK(cv && num_eq(ckc_jget(cv, "a"), 1), "map['a']==1");
    const jd_val_t* b = cv ? ckc_jget(cv, "b") : NULL;
    CHECK(b && b->kind == JD_ARR && b->arr_len == 2 && b->arr[0]->kind == JD_BOOL
              && b->arr[0]->b && b->arr[1]->kind == JD_NULL,
          "map['b']==[true,null]");

    /* The same value as JSON must decode to an equal DOM. */
    jd_val_t* jv = ckc_json_parse("{\"a\": 1, \"b\": [true, null]}", a, err, sizeof err);
    CHECK(jv != NULL, "json parse ok");
    CHECK(jd_equal(cv, jv), "CBOR DOM == JSON DOM");
}

static void test_errors(ckc_arena_t* a)
{
    char err[128];
    /* array header says 3 elems but only 1 present -> must fail, not crash */
    unsigned char trunc[] = {0x83, 0x01};
    err[0] = '\0';
    jd_val_t* v = ckc_cbor_parse(trunc, sizeof trunc, a, err, sizeof err);
    CHECK(v == NULL, "truncated array rejected");
    CHECK(err[0] != '\0', "truncated array sets diagnostic");

    /* text string claiming 5 bytes with only 2 present -> fail */
    unsigned char badstr[] = {0x65, 'h', 'i'};
    v = ckc_cbor_parse(badstr, sizeof badstr, a, err, sizeof err);
    CHECK(v == NULL, "truncated string rejected");

    /* empty input -> fail cleanly */
    v = ckc_cbor_parse((const unsigned char*)"", 0, a, err, sizeof err);
    CHECK(v == NULL, "empty input rejected");
}

static void test_nested_recipe_shape(ckc_arena_t* a)
{
    char err[128];
    /* A recipe-ish map: {"op":"emit","in":["x","y"]}
     * 0xA2
     *   0x62 'o' 'p'            "op"
     *   0x64 'e' 'm' 'i' 't'    "emit"
     *   0x62 'i' 'n'            "in"
     *   0x82 0x61 'x' 0x61 'y'  ["x","y"] */
    unsigned char r[] = {0xA2, 0x62, 'o', 'p', 0x64, 'e', 'm', 'i', 't',
                         0x62, 'i', 'n', 0x82, 0x61, 'x', 0x61, 'y'};
    jd_val_t* v = ckc_cbor_parse(r, sizeof r, a, err, sizeof err);
    const char* op = v ? ckc_jstr(ckc_jget(v, "op")) : NULL;
    const jd_val_t* in = v ? ckc_jget(v, "in") : NULL;
    CHECK(op && strcmp(op, "emit") == 0, "recipe op == emit");
    CHECK(in && in->kind == JD_ARR && in->arr_len == 2
              && strcmp(ckc_jstr(in->arr[0]), "x") == 0
              && strcmp(ckc_jstr(in->arr[1]), "y") == 0,
          "recipe in == [x,y]");
}

int main(void)
{
    ckc_arena_t a;
    if (ckc_arena_init(&a, 0) != 0) {
        printf("arena init failed\n");
        return 2;
    }
    printf("test_cbor_dom:\n");
    test_scalars(&a);
    test_array(&a);
    test_map_and_json_equiv(&a);
    test_errors(&a);
    test_nested_recipe_shape(&a);
    ckc_arena_destroy(&a);

    if (g_fail == 0) {
        printf("PASS: all DOM decoder unit tests passed.\n");
        return 0;
    }
    printf("FAIL: %d checks failed.\n", g_fail);
    return 1;
}
