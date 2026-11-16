// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of kernels/common/attention_tiled_2d_rules.py. See
 * rocke/helper_rocke.instances.common.attention_tiled_2d_rules.h for the mapping.
 */
#include "rocke/helper_rocke.instances.common.attention_tiled_2d_rules.h"

#include <string.h>

#include "rocke/helper_rocke.helpers.spec.h" /* ROCKE_SPEC_FAIL */

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
        const struct
        {
            const char* feature;
            const char* field;
            bool enabled;
        } unsupported_features[] = {
            {"softcap", "has_softcap", spec->has_softcap},
            {"ALiBi", "use_alibi", spec->use_alibi},
            {"QQ bias", "use_qq_bias", spec->use_qq_bias},
        };
        for(const auto& unsupported : unsupported_features)
        {
            if(unsupported.enabled)
                ROCKE_SPEC_FAIL(reason,
                                reason_cap,
                                "use_grouped_kv2_softmax v1 does not support %s - %s=True, "
                                "expected False",
                                unsupported.feature,
                                unsupported.field);
        }
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
