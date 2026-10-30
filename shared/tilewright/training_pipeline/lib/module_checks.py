# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What preflight asks of the tilewright Python module, which the stages
load to score, rank and check models: that it imports, computes this
checkout's feature catalogs and builds kernel configs as this checkout's
engine does, and that it knows every serving option the models of the run
declare (the CU floor of the v3 catalog, the alignment veto of
deploy.alignment_veto, the kernel guards flag of
correctness.guard_unguarded_custom). A module that predates an option
refuses the models that declare it ('unknown serving flags'), which only a
stage after the training would show. run_orchestrator.py's preflight calls
these; each returns the errors that keep a run from starting.
"""
from __future__ import annotations

from typing import List


def guard_module_errors() -> List[str]:
    """A tilewright module that predates the kernel guards flag, which the
    models of a run with correctness.guard_unguarded_custom declare and it
    refuses ('unknown serving flags')."""
    try:
        from . import evaluate

        tilewright = evaluate.tilewright_module()
    except Exception:
        return []
    if hasattr(getattr(tilewright, "ModelInfo", None), "kernel_guards"):
        return []
    return [
        "correctness.guard_unguarded_custom: the tilewright module predates the "
        "kernel guards flag (its ModelInfo has no kernel_guards), so it refuses "
        "the models stage06 flags; reinstall it with `pip install "
        "shared/tilewright/python`"
    ]


def tilewright_errors(catalog: str = "v2", alignment_veto: bool = False) -> List[str]:
    try:
        from . import evaluate

        tilewright = evaluate.tilewright_module()
    except Exception as e:
        return [
            f"the tilewright Python module does not import ({e!r}); install it "
            f"with `pip install shared/tilewright/python`"
        ]
    try:
        from . import features
    except Exception as e:
        return [f"lib.features does not import: {e!r}"]
    have = getattr(tilewright, "feature_catalog_hash", None)
    if callable(have) and have() != features.feature_names_hash():
        return [
            f"feature catalog hash mismatch: tilewright module "
            f"{have()} vs lib/features.py {features.feature_names_hash()}"
        ]
    if catalog == "v3":
        hashes = getattr(tilewright, "feature_catalog_hashes", None)
        listed = list(hashes()) if callable(hashes) else []
        if features.v3_catalog_hash() not in listed:
            return [
                f"train.catalog v3: the tilewright module does not compute the "
                f"v3 catalog {features.v3_catalog_hash()} (it has {listed}); "
                f"reinstall it with `pip install shared/tilewright/python`"
            ]
        if not hasattr(getattr(tilewright, "ModelInfo", None), "min_effective_cus"):
            return [
                "train.catalog v3: the tilewright module predates the CU floor "
                "(its ModelInfo has no min_effective_cus), so it refuses the "
                "models stage05 writes, which declare one ('unknown serving "
                "flags'); reinstall it with `pip install shared/tilewright/python`"
            ]
    if alignment_veto and not hasattr(
        getattr(tilewright, "ModelInfo", None), "alignment_veto"
    ):
        return [
            "deploy.alignment_veto: the tilewright module predates the alignment "
            "veto (its ModelInfo has no alignment_veto), so it refuses the models "
            "stage06 stages, which declare it ('unknown serving flags'); reinstall "
            "it with `pip install shared/tilewright/python`"
        ]
    try:
        current = evaluate.module_takes_attributes(tilewright)
    except Exception as e:
        return [f"the tilewright module cannot build kernel configs: {e!r}"]
    if not current:
        return [
            "the tilewright module is older than this checkout's engine (its "
            "Config takes no kernel attributes); reinstall it with "
            "`pip install shared/tilewright/python`"
        ]
    return []
