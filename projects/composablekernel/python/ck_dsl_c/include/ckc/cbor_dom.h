/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/cbor_dom.h -- decode a CBOR blob into the SAME arena-owned tagged DOM as
 * ckc/json_dom.h (jd_val_t). Lets the recipe VM and bundle reader consume CBOR
 * recipes/bundles with zero changes to the consumers (they already walk jd_val_t
 * via ckc_jget / ckc_jstr / ckc_jnum).
 *
 * Supports the subset recipes use: unsigned int, negative int, float64, text
 * string, array, map (text keys), bool, null. Big-endian per RFC 8949.
 */
#ifndef CKC_CBOR_DOM_H
#define CKC_CBOR_DOM_H

#include <stddef.h>

#include "ckc/arena.h"
#include "ckc/json_dom.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Parse `len` bytes of CBOR at `data` into a DOM allocated from `arena`. Returns
 * the root, or NULL on failure (diagnostic written into err/err_cap). */
jd_val_t* ckc_cbor_parse(const unsigned char* data, size_t len, ckc_arena_t* arena,
                         char* err, size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_CBOR_DOM_H */
