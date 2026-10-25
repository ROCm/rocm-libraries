# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The WMMA forward body lives in its own module; old import paths still work."""

from __future__ import annotations

import subprocess
import sys


def test_old_import_paths_resolve_to_new_homes():
    from rocke.helpers import _attention_shared as shared
    from rocke.helpers import mfma_attention as mfma
    from rocke.helpers import wmma_attention as wmma

    for name in (
        "_wmma_attention_fwd_inner_body",
        "_wmma_attn_op_id",
        "_WMMA_ATTN_OP_ID",
    ):
        assert getattr(mfma, name) is getattr(wmma, name)
    for name in (
        "_softmax_row_reduce",
        "_ir_type_for_dtype",
        "MFMA_ATTN_BLOCK_K",
        "_SOFTMAX_ROW_REDUCE_DIST",
    ):
        assert getattr(mfma, name) is getattr(shared, name)
        assert getattr(wmma, name, getattr(shared, name)) is getattr(shared, name)
    assert wmma._wmma_attention_fwd_inner_body.__module__ == wmma.__name__


def test_public_names_unchanged():
    import rocke.helpers as h
    from rocke.helpers.mfma_attention import (
        MFMA_ATTN_BLOCK_K,
        MFMA_ATTN_BLOCK_M,
        mfma_attention_fwd_inner_body,
    )

    assert h.mfma_attention_fwd_inner_body is mfma_attention_fwd_inner_body
    assert h.MFMA_ATTN_BLOCK_K == MFMA_ATTN_BLOCK_K == 16
    assert h.MFMA_ATTN_BLOCK_M == MFMA_ATTN_BLOCK_M == 16


def _import_in_fresh_interpreter(first: str):
    code = (
        f"import importlib, sys; importlib.import_module({first!r}); "
        "assert 'rocke.helpers.mfma_attention' in sys.modules"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_each_module_imports_first_without_cycle():
    for mod in (
        "rocke.helpers._attention_shared",
        "rocke.helpers.wmma_attention",
        "rocke.helpers.mfma_attention",
    ):
        r = _import_in_fresh_interpreter(mod)
        if mod != "rocke.helpers.mfma_attention":
            # importing the leaf/wmma module alone must not need mfma first
            code = f"import importlib; importlib.import_module({mod!r})"
            r = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True
            )
        assert r.returncode == 0, r.stderr


def test_shared_and_wmma_modules_do_not_import_mfma_module():
    import ast
    import pathlib

    import rocke.helpers as h

    root = pathlib.Path(h.__file__).parent
    for fname in ("_attention_shared.py", "wmma_attention.py"):
        tree = ast.parse((root / fname).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "mfma_attention", fname
                assert "mfma_attention" not in [a.name for a in node.names], fname
    tree = ast.parse((root / "_attention_shared.py").read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "wmma_attention" not in mods
