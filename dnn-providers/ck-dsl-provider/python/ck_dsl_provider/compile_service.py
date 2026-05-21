# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Thin Python compile-service entry point for the CK DSL provider.

M1 step I-3 only ships :func:`noop_smoke`, which proves the C++ →
Python → ``ck_dsl`` import path works inside the embedded interpreter.

# TODO(I-7): add ``compile(op_kind: str, payload: dict) -> tuple[bytes, dict]``
# that instantiates the matching ``ck_dsl`` Spec dataclass via ``**payload``,
# calls ``ck_dsl.helpers.compile.compile_kernel``, and returns
# (hsaco_bytes, launch_abi_dict).
"""

from __future__ import annotations


def noop_smoke() -> dict:
    """Return a constant. Used by I-3 to prove import path + GIL handling."""
    import ck_dsl  # forces the cross-package import to succeed

    return {
        "service": "ck_dsl_provider.compile_service",
        "ck_dsl_module_path": ck_dsl.__file__,
        "smoke": "ok",
    }
