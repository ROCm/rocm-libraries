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

# The split ops build gfx950 objects from the CDNA backward kernels and gfx11-generic objects from the RDNA ones, which import
# the RDNA forward's Philox.
_SDPA_BWD_SPLIT_DIRS = (
    "attention_bwd_cdna",
    "attention_cdna",
    "attention_bwd_rdna",
    "attention_rdna4",
)

# The gfx950-only ops build from the CDNA backward kernels, which import the aiw directory's Philox
# and its null-safe u64 loader (``fmha_common_gfx1201``, whose own imports reach the rest of the directory).
_SDPA_BWD_CDNA_DIRS = (
    "attention_bwd_cdna",
    "attention_cdna",
    "attention_rdna4",
)

OP_KERNEL_DIRS = {
    "rmsnorm": ("norm",),
    "rmsnorm_bwd": ("norm",),
    # The RDNA and CDNA kernels import the aiw directory's Philox (RDNA kernel modification
    # 27, CDNA 38: one source for the dropout stream), so it is an input of
    # theirs too.
    "sdpa_rdna": ("attention_rdna", "attention_rdna4/philox.py"),
    "sdpa_rdna_decode": ("attention_rdna", "attention_rdna4/philox.py"),
    "sdpa_cdna": ("attention_cdna", "attention_rdna4/philox.py"),
    "sdpa_cdna_dualwave": ("attention_cdna", "attention_rdna4/philox.py"),
    "sdpa_cdna_decode": ("attention_cdna", "attention_rdna4/philox.py"),
    "sdpa_aiw": ("attention_rdna4",),
    # The RNG dump's keep-mask writer draws the attention kernels' counters through the same
    # Philox, so that file is its input too; its own directory keeps it out of sdpa_aiw's.
    "sdpa_dropout_mask": ("dropout_mask", "attention_rdna4/philox.py"),
    # The head-dim padding copy around an SDPA launch at a head dim off 8 (FlydslSdpaHeadPad.hpp).
    "sdpa_head_pad": ("head_pad",),
    "sdpa_bwd_prep": _SDPA_BWD_SPLIT_DIRS,
    "sdpa_bwd_dkdv": _SDPA_BWD_SPLIT_DIRS,
    "sdpa_bwd_dq": _SDPA_BWD_SPLIT_DIRS,
    "sdpa_bwd_merged": _SDPA_BWD_CDNA_DIRS,
    "sdpa_bwd_dbias_reduce": _SDPA_BWD_CDNA_DIRS,
    "sdpa_bwd_dsink_reduce": _SDPA_BWD_CDNA_DIRS,
    "sdpa_bwd_tailzero": _SDPA_BWD_SPLIT_DIRS,
}
