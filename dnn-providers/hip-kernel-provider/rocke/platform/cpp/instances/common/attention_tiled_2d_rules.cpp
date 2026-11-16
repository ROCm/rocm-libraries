// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of kernels/common/attention_tiled_2d_rules.py. See
 * rocke/helper_rocke.instances.common.attention_tiled_2d_rules.h for the mapping.
 */
#include "rocke/helper_rocke.instances.common.attention_tiled_2d_rules.h"

#include <string.h>

#include "rocke/helper_rocke.helpers.spec.h" /* ROCKE_SPEC_FAIL */

typedef struct rocke_t2d_feature
{
    const char* feature;
    const char* field;
    bool enabled;
} rocke_t2d_feature_t;

/* Python: _unsupported_feature_refusal. Refuses the first enabled feature;
 * `does_not_support` is the message head ("... does not support"). */
static bool rocke_i_t2d_refuse_unsupported_feature(const char* does_not_support,
                                                   const rocke_t2d_feature_t* features,
                                                   size_t num_features,
                                                   char* reason,
                                                   size_t reason_cap)
{
    for(size_t i = 0; i < num_features; ++i)
    {
        if(features[i].enabled)
            ROCKE_SPEC_FAIL(reason,
                            reason_cap,
                            "%s %s - %s=True, expected False",
                            does_not_support,
                            features[i].feature,
                            features[i].field);
    }
    return true;
}

bool rocke_attention_tiled_2d_validate_transposed_softmax_valu_opts(
    const rocke_attention_tiled_2d_spec_t* spec, char* reason, size_t reason_cap)
{
    if(!spec->use_transposed_mask_limit)
        return true;
    if(!(spec->use_transposed_qk_32x32 && spec->use_transposed_scalar_state
         && spec->use_transposed_mask_once))
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "transposed softmax VALU opts require the R4_s1mask path - "
                        "use_transposed_qk_32x32=%s, use_transposed_scalar_state=%s, "
                        "use_transposed_mask_once=%s, expected all True",
                        spec->use_transposed_qk_32x32 ? "True" : "False",
                        spec->use_transposed_scalar_state ? "True" : "False",
                        spec->use_transposed_mask_once ? "True" : "False");
    if(spec->sliding_window > 0)
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "transposed softmax VALU opts do not support a sliding window - "
                        "sliding_window=%d, expected 0",
                        spec->sliding_window);
    const rocke_t2d_feature_t features[] = {
        {"softcap", "has_softcap", spec->has_softcap},
        {"ALiBi", "use_alibi", spec->use_alibi},
        {"QQ bias", "use_qq_bias", spec->use_qq_bias},
        {"grouped-KV2 softmax", "use_grouped_kv2_softmax", spec->use_grouped_kv2_softmax},
    };
    return rocke_i_t2d_refuse_unsupported_feature("transposed softmax VALU opts do not support",
                                                  features,
                                                  sizeof(features) / sizeof(features[0]),
                                                  reason,
                                                  reason_cap);
}

bool rocke_attention_tiled_2d_validate_grouped_kv2_and_early_v(
    const rocke_attention_tiled_2d_spec_t* spec, char* reason, size_t reason_cap)
{
    if(spec->use_grouped_kv2_softmax)
    {
        if(spec->dtype == NULL)
            ROCKE_SPEC_FAIL(reason,
                            reason_cap,
                            "use_grouped_kv2_softmax v1 is restricted to bf16 - dtype=None, "
                            "expected 'bf16'");
        if(strcmp(spec->dtype, "bf16") != 0)
            ROCKE_SPEC_FAIL(reason,
                            reason_cap,
                            "use_grouped_kv2_softmax v1 is restricted to bf16 - dtype='%s', "
                            "expected 'bf16'",
                            spec->dtype);
        if(!(spec->use_mfma_32x32 && spec->use_transposed_qk_32x32))
            ROCKE_SPEC_FAIL(reason,
                            reason_cap,
                            "use_grouped_kv2_softmax requires the transposed 32x32 R4 path - "
                            "use_mfma_32x32=%s, use_transposed_qk_32x32=%s, expected both True",
                            spec->use_mfma_32x32 ? "True" : "False",
                            spec->use_transposed_qk_32x32 ? "True" : "False");
        if(spec->block_m_per_warp != 32)
            ROCKE_SPEC_FAIL(reason,
                            reason_cap,
                            "use_grouped_kv2_softmax requires one 32-row MFMA atom per warp - "
                            "block_m_per_warp=%d, expected 32",
                            spec->block_m_per_warp);
        if(spec->sliding_window > 0)
            ROCKE_SPEC_FAIL(reason,
                            reason_cap,
                            "use_grouped_kv2_softmax v1 does not support a sliding window - "
                            "sliding_window=%d, expected 0",
                            spec->sliding_window);
        const rocke_t2d_feature_t features[] = {
            {"softcap", "has_softcap", spec->has_softcap},
            {"ALiBi", "use_alibi", spec->use_alibi},
            {"QQ bias", "use_qq_bias", spec->use_qq_bias},
        };
        if(!rocke_i_t2d_refuse_unsupported_feature("use_grouped_kv2_softmax v1 does not support",
                                                   features,
                                                   sizeof(features) / sizeof(features[0]),
                                                   reason,
                                                   reason_cap))
            return false;
        if(spec->kv_storage_dtype != NULL)
            ROCKE_SPEC_FAIL(reason,
                            reason_cap,
                            "use_grouped_kv2_softmax v1 does not support FP8 KV - "
                            "kv_storage_dtype='%s', expected None",
                            spec->kv_storage_dtype);
    }
    if(spec->use_early_v_schedule)
    {
        if(spec->use_grouped_kv2_softmax)
            ROCKE_SPEC_FAIL(reason,
                            reason_cap,
                            "use_early_v_schedule does not support grouped-KV2 softmax - "
                            "use_grouped_kv2_softmax=True, expected False");
        if(spec->kv_storage_dtype != NULL)
            ROCKE_SPEC_FAIL(reason,
                            reason_cap,
                            "use_early_v_schedule v1 does not support FP8 KV - "
                            "kv_storage_dtype='%s', expected None",
                            spec->kv_storage_dtype);
    }
    return true;
}
