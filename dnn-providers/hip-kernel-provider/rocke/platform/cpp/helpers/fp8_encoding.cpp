// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * fp8_encoding.cpp -- the C++ engine port of rocke.helpers.fp8_encoding: the
 * four fp8 / bf8 encodings and the per-target table of fp8_element_format.
 */

#include "rocke/helper_rocke.helpers.fp8_encoding.h"

#include <stdio.h>
#include <string.h>

const rocke_mx_element_format_t ROCKE_FP8_E4M3_OCP = {"OCP E4M3", 448.0};
const rocke_mx_element_format_t ROCKE_BF8_E5M2_OCP = {"OCP E5M2", 57344.0};
const rocke_mx_element_format_t ROCKE_FP8_E4M3_FNUZ = {"E4M3 FNUZ", 240.0};
const rocke_mx_element_format_t ROCKE_BF8_E5M2_FNUZ = {"E5M2 FNUZ", 57344.0};

/* Python _FP8_ELEMENT_FORMATS, in the Python table's order (the refusal
 * message lists the targets in it). */
typedef struct rocke_fp8_encoding_row
{
    const char* arch;
    const rocke_mx_element_format_t* fp8e4m3;
    const rocke_mx_element_format_t* bf8e5m2;
} rocke_fp8_encoding_row_t;

static const rocke_fp8_encoding_row_t ROCKE_FP8_ELEMENT_FORMATS[] = {
    {"gfx942", &ROCKE_FP8_E4M3_FNUZ, &ROCKE_BF8_E5M2_FNUZ},
    {"gfx950", &ROCKE_FP8_E4M3_OCP, &ROCKE_BF8_E5M2_OCP},
    {"gfx1250", &ROCKE_FP8_E4M3_OCP, &ROCKE_BF8_E5M2_OCP},
};

#define ROCKE_FP8_ELEMENT_FORMATS_COUNT \
    (sizeof(ROCKE_FP8_ELEMENT_FORMATS) / sizeof(ROCKE_FP8_ELEMENT_FORMATS[0]))

/* Python repr() of a string argument: quoted, or None for NULL. */
static int format_repr(char* out, size_t cap, const char* value)
{
    return value != NULL ? snprintf(out, cap, "'%s'", value) : snprintf(out, cap, "None");
}

const rocke_mx_element_format_t*
    rocke_fp8_element_format(const char* dtype, const char* arch, char* err, size_t err_cap)
{
    char repr[64];
    size_t i;
    size_t used;

    /* Python _FP8_DTYPES = ("bf8e5m2", "fp8e4m3") */
    if(dtype == NULL || (strcmp(dtype, "fp8e4m3") != 0 && strcmp(dtype, "bf8e5m2") != 0))
    {
        if(err != NULL && err_cap > 0)
        {
            format_repr(repr, sizeof(repr), dtype);
            snprintf(err,
                     err_cap,
                     "fp8_element_format failed - dtype=%s, expected one of "
                     "('bf8e5m2', 'fp8e4m3')",
                     repr);
        }
        return NULL;
    }
    for(i = 0; arch != NULL && i < ROCKE_FP8_ELEMENT_FORMATS_COUNT; ++i)
    {
        if(strcmp(arch, ROCKE_FP8_ELEMENT_FORMATS[i].arch) == 0)
        {
            return strcmp(dtype, "fp8e4m3") == 0 ? ROCKE_FP8_ELEMENT_FORMATS[i].fp8e4m3
                                                 : ROCKE_FP8_ELEMENT_FORMATS[i].bf8e5m2;
        }
    }
    if(err == NULL || err_cap == 0)
    {
        return NULL;
    }
    /* Python: f"... arch={arch!r}, expected one of {tuple(_FP8_ELEMENT_FORMATS)}" */
    format_repr(repr, sizeof(repr), arch);
    used = (size_t)snprintf(
        err, err_cap, "fp8_element_format failed - arch=%s, expected one of (", repr);
    for(i = 0; i < ROCKE_FP8_ELEMENT_FORMATS_COUNT && used < err_cap; ++i)
    {
        used += (size_t)snprintf(err + used,
                                 err_cap - used,
                                 "%s'%s'",
                                 i == 0 ? "" : ", ",
                                 ROCKE_FP8_ELEMENT_FORMATS[i].arch);
    }
    if(used < err_cap)
    {
        snprintf(err + used, err_cap - used, ")");
    }
    return NULL;
}
