// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

// Hyperparams for the sparge attention forward pass. Kept as a nested struct on
// fmha_sparge_fwd_args so per-launch tensor / shape fields stay flat and the
// algorithm knobs cluster together. Defaults match the historical kPerWave
// behaviour with PV-skip disabled at runtime via the +1e30 sentinel.
struct sparge_attn_hyperparam_args
{
    // PV-skip scalar (SpargeAttn §4.4); +1e30 sentinel = skip disabled.
    float pv_threshold = 1e30f;
    // Device buffer, length == nhead_q; overrides the scalar when non-null.
    const float* pv_threshold_per_head_ptr = nullptr;

    // Per-head dispatch routing.
    // head_remap_ptr: device buffer (length == nhead_in_launch); when non-null
    // gridDim.y shrinks to nhead_in_launch and the kernel reads
    // head_remap_ptr[blockIdx.y] to recover the original head index.
    const int* head_remap_ptr = nullptr;
    int nhead_in_launch       = 0; // 0 = identity (full nhead_q grid)

    // Host-side dispatch selectors.
    // pv_skip_compile: legacy bool kept for source compat — derives
    // pv_mode_compile = (pv_skip_compile ? 1 : 0) when only the bool is set.
    bool pv_skip_compile = true;
    // pv_mode_compile: 0=kNone, 1=kPerWave, 2=kPerBlock.
    int pv_mode_compile = 1;
};
