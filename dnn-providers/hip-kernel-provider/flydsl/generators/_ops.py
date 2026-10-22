# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The op registry: what one op's objects are generated from.

Each op names its kernel directories under kernels_src/kernels/; a new op adds its
row here, and an op missing here is an error rather than an empty digest. The
registry is a table like ``_instances.py``: it is not digested as a file, each op
digests its own row instead (``_manifest.inputs_digest``), so registering an op
leaves every other op's objects current.

Ops that share a directory, as rmsnorm and rmsnorm_bwd share norm/, go stale
together: an edit to any file in it, comments included, needs a regeneration of
each.
"""

from __future__ import annotations

OP_KERNEL_DIRS = {
    "rmsnorm": ("norm",),
    "rmsnorm_bwd": ("norm",),
    "sdpa_rdna": ("attention_rdna",),
    "sdpa_rdna_decode": ("attention_rdna",),
    "sdpa_cdna": ("attention_cdna",),
    "sdpa_cdna_dualwave": ("attention_cdna",),
    "sdpa_cdna_decode": ("attention_cdna",),
    "sdpa_aiw": ("attention_rdna4",),
    "sdpa_bwd_prep": ("attention_bwd_cdna", "attention_cdna"),
    "sdpa_bwd_dkdv": ("attention_bwd_cdna", "attention_cdna"),
    "sdpa_bwd_dq": ("attention_bwd_cdna", "attention_cdna"),
    "sdpa_bwd_merged": ("attention_bwd_cdna", "attention_cdna"),
    "sdpa_bwd_dbias_reduce": ("attention_bwd_cdna", "attention_cdna"),
    "sdpa_bwd_dsink_reduce": ("attention_bwd_cdna", "attention_cdna"),
}
