// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_ATTENTION_DENSE_SPEC_INTERNAL_H
#define ROCKE_ATTENTION_DENSE_SPEC_INTERNAL_H

#include "rocke/instance_attention_dense_spec.h"

/* Per-spec storage hooks. The public base C API uses two-byte K/V and no
 * storage name token; a concrete spec supplies its own answers here. */
namespace rocke_dense_detail
{
constexpr int base_kv_elem_bytes = 2;

bool validate(const rocke_attention_dense_spec_t* s,
              const rocke_attention_dense_traits_t* t,
              bool lookahead,
              int kv_elem_bytes,
              char* reason,
              size_t cap);
bool preflight(const rocke_attention_dense_spec_t* s,
               const rocke_attention_dense_traits_t* t,
               bool lookahead,
               int kv_elem_bytes,
               rocke_attention_dense_validate_fn arch_validate,
               const void* ctx,
               char* reason,
               size_t cap);
rocke_status_t kernel_name(const rocke_attention_dense_spec_t* s,
                           const rocke_attention_dense_name_parts_t* np,
                           const char* kv_storage_token,
                           char* out,
                           size_t out_cap);
} // namespace rocke_dense_detail

#endif /* ROCKE_ATTENTION_DENSE_SPEC_INTERNAL_H */
