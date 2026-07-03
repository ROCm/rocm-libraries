/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_helper_rocke.helpers.spec.h -- C99 port of a single symbol from
 * rocke/helpers/spec.py:
 *
 *   Python                              C99 (this header)
 *   ---------------------------------   ---------------------------------------
 *   kernel_name_join(...)              rocke_kernel_name_join(...)
 *
 * This is a pure string helper: it does NOT call the IR builder (rocke_b_*). It
 * is a byte-for-bit value producer whose result is later baked into the IR /
 * manifest (the kernel name string), so a byte-identical IR sequence follows
 * from a byte-identical return value.
 *
 * Error model mirrors the rest of the C port: an out-param + rocke_status_t
 * return code stands in for the Python `raise` / overflow cases.
 */
#ifndef ROCKE_HELPER_HELPER_ROCKE_HELPERS_SPEC_H
#define ROCKE_HELPER_HELPER_ROCKE_HELPERS_SPEC_H

#include <stddef.h>

#include "rocke/ir.h" /* rocke_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * kernel_name_join
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   def kernel_name_join(prefix, *parts, flags=None) -> str:
 *       body = "_".join(p for p in (prefix, *parts) if p)
 *       if flags:
 *           for name, on in flags.items():
 *               if on:
 *                   body += f"_{name}"
 *       return body.replace("/", "_")
 *
 * `parts` is an ordered NULL-or-empty-skipping list. `flags` is an ordered
 * (name, on) list; entries with on != 0 contribute an "_<name>" suffix in
 * iteration order -- matching CPython 3.7+ dict insertion order, which the
 * callers rely on. Every "/" in the final string is rewritten to "_".
 *
 * The result is written NUL-terminated into `out` (capacity `out_cap`). On
 * success returns ROCKE_OK and, if out_len != NULL, sets *out_len to the byte
 * length (excluding the NUL). If the buffer is too small returns
 * ROCKE_ERR_VALUE and writes nothing usable. */
rocke_status_t rocke_kernel_name_join(const char* prefix,
                                      const char* const* parts,
                                      size_t num_parts,
                                      const char* const* flag_names,
                                      const int* flag_on,
                                      size_t num_flags,
                                      char* out,
                                      size_t out_cap,
                                      size_t* out_len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_HELPER_ROCKE_HELPERS_SPEC_H */
