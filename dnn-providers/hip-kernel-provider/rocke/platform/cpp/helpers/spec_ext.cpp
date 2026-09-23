// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/helpers/spec.py: kernel_name_join.
 *
 * Faithful, builder-free value producer. See the header for the original
 * Python and the contract. The goal is a byte-identical return value so the
 * downstream IR / manifest (the kernel name string baked into the manifest) is
 * byte-identical to the Python.
 */
#include "rocke/helper_helper_rocke.helpers.spec.h"

/* INTEGRATION NOTE (no symbols of its own).
 *   rocke_kernel_name_join is the canonical C99 port of
 *   rocke/helpers/spec.py:kernel_name_join and is DEFINED once in the full
 *   spec-helper translation unit
 *     helper_rocke.helpers.spec.c
 *   (declared in rocke/helper_rocke.helpers.spec.h, which
 *    rocke/helper_helper_rocke.helpers.spec.h re-exposes). Re-defining the same
 *   symbol here produced a duplicate-definition link error against that TU, so
 *   this part-file no longer carries its own copy -- callers that include the
 *   helper_helper header resolve rocke_kernel_name_join to the canonical
 *   definition at link time. This TU intentionally contributes NO symbols and
 *   is kept as a documentation placeholder. */
typedef int rocke_helper_helper_spec_translation_unit_marker;
