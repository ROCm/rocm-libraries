# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

FWD_DTYPE_MAP = {
    "fp16": "FmhaSparseFwdFp16",
    "bf16": "FmhaSparseFwdBf16",
}

_MASK_SIMPLIFIED_MAP = {
    "s_no": "ck_tile::SimplifiedGenericAttentionMask<false>",
    "s_mask": "ck_tile::SimplifiedGenericAttentionMask<true>",
}

_MASK_MAP = {
    "no": "FmhaMasks::NoMask",
    "causal": "FmhaMasks::CausalMask",
    "generic": "FmhaMasks::GenericMask",
}


def get_mask_map(mask: str):
    if mask == "generic":
        return _MASK_MAP
    elif mask == "simplified":
        return _MASK_SIMPLIFIED_MAP
    else:
        assert False
        return None


_MASK_CHECK_MAP = {
    "no": "t.mask_type == mask_enum::no_mask",
    "causal": "t.mask_type == mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right",
    "generic": "t.mask_type == mask_enum::window_generic",
}

_MASK_SIMPLIFIED_CHECK_MAP = {
    "s_no": "t.mask_type == mask_enum::no_mask",
    "s_mask": "t.mask_type != mask_enum::no_mask",
}


def get_mask_check_map(mask: str):
    if mask == "generic":
        return _MASK_CHECK_MAP
    elif mask == "simplified":
        return _MASK_SIMPLIFIED_CHECK_MAP
    else:
        assert False
        return None


MODE_MAP = {"batch": "false", "group": "true"}

LAYOUT_MAP = {"row": "true", "col": "false"}

PIPELINE_MAP = {
    "qr_async_jenga": "ck_tile::BlockFmhaPipelineQRKSVSAsyncJenga",
    "qr_async_vsa": "ck_tile::BlockFmhaPipelineQRKSVSAsyncVSA",
    "qr_async_sparge": "ck_tile::BlockFmhaPipelineQRKSVSAsyncSparge",
}

PIPELINE_ENUM_MAP = {
    "qr_async_jenga": "ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC",
    "qr_async_vsa": "ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC",
    "qr_async_sparge": "ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC",
}

BOOL_MAP = {
    "t": "true",
    "f": "false",
    True: "true",
    False: "false",
}

BIAS_MAP = {
    "no":          "ck_tile::BlockAttentionBiasEnum::NO_BIAS",
    "elementwise": "ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS",
    "alibi":       "ck_tile::BlockAttentionBiasEnum::ALIBI",
}

BIAS_CHECK_MAP = {
    "no":          "0",
    "elementwise": "1",
    "alibi":       "2",
}
