// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of ck_dsl/helpers/spec.py: kernel_name_join.
 *
 * Faithful, builder-free value producer. See the header for the original
 * Python and the contract. The goal is a byte-identical return value so the
 * downstream IR / manifest (the kernel name string baked into the manifest) is
 * byte-identical to the Python.
 */
#include "ckc/helper_helper_ck_dsl.helpers.spec.h"

/* INTEGRATION NOTE (no symbols of its own).
 *   ckc_kernel_name_join is the canonical C99 port of
 *   ck_dsl/helpers/spec.py:kernel_name_join and is DEFINED once in the full
 *   spec-helper translation unit
 *     helper_ck_dsl.helpers.spec.c
 *   (declared in ckc/helper_ck_dsl.helpers.spec.h, which
 *    ckc/helper_helper_ck_dsl.helpers.spec.h re-exposes). Re-defining the same
 *   symbol here produced a duplicate-definition link error against that TU, so
 *   this part-file no longer carries its own copy -- callers that include the
 *   helper_helper header resolve ckc_kernel_name_join to the canonical
 *   definition at link time. This TU intentionally contributes NO symbols and
 *   is kept as a documentation placeholder. */
typedef int ckc_helper_helper_spec_translation_unit_marker;
