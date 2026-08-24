# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 (CDNA3 / MI300) arch-specific instance builders (hybrid layout).

Put a kernel here only when its *algorithm* genuinely differs on gfx942 from the
shared ``instances/common/`` version (e.g. a K-loop or staging strategy that
cannot use a gfx950-only primitive such as ``ds_read_*_tr_*``). Shared,
arch-polymorphic kernels belong in ``instances/common/``.

Adding/improving a gfx942 variant here must not edit another arch's files or a
shared ``common/`` builder. See
``dsl_docs/architecture/multi_arch_data_layout.md``.
"""

from .gdn_state_scan import (  # noqa: F401
    GdnStateScanSpec,
    build_gdn_state_scan,
    gdn_state_scan_grid,
    gdn_state_scan_signature,
    is_valid_config as is_valid_gdn_state_scan_config,
    is_valid_spec as is_valid_gdn_state_scan_spec,
    pick_bf16_atom,
)
