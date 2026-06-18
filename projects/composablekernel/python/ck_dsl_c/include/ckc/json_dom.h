/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/json_dom.h -- a small dependency-free JSON parser into an arena-owned
 * tagged DOM. Shared by the recipe VM (recipe_vm.c) and intended to be the
 * canonical JSON reader for the ckc tooling (ir_import_json.c carries an older
 * embedded copy that can be migrated onto this module).
 *
 * The whole DOM is bump-allocated from the caller-provided arena; there is no
 * per-node free -- destroy the arena once parsing/consumption is done.
 */
#ifndef CKC_JSON_DOM_H
#define CKC_JSON_DOM_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    JD_NULL,
    JD_BOOL,
    JD_NUM,
    JD_STR,
    JD_ARR,
    JD_OBJ
} jd_kind_t;

typedef struct jd_val jd_val_t;
typedef struct
{
    char* key;
    jd_val_t* val;
} jd_member_t;

struct jd_val
{
    jd_kind_t kind;
    bool b;
    double num;
    char* str;
    jd_val_t** arr;
    int arr_len;
    jd_member_t* obj;
    int obj_len;
};

/* Parse `text` (NUL-terminated) into a DOM allocated from `arena`. Returns the
 * root, or NULL on failure (a diagnostic is written into err/err_cap). */
jd_val_t* ckc_json_parse(const char* text, ckc_arena_t* arena, char* err, size_t err_cap);

/* Object member lookup (NULL if not an object or key absent). */
jd_val_t* ckc_jget(const jd_val_t* o, const char* key);
/* String value, or NULL if not a string. */
const char* ckc_jstr(const jd_val_t* v);
/* Number value into *out; returns false if not a number. */
bool ckc_jnum(const jd_val_t* v, double* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_JSON_DOM_H */
