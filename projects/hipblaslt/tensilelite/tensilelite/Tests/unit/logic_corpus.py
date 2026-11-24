# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The one place a TensileLite unit test may locate the production logic corpus.

The corpus (hipBLASLt's tuned logic YAML) lives in the hipBLASLt product tree,
outside tensilelite, so it is absent wherever these tests run from installed
artifacts, and any tuning PR can change it. Prefer not to read it at all: test
code against a fixture under ``logic_yaml/``, and put cross-file corpus
invariants in ``tensilelite.logic.valid_corpus_consistency``, which
``TensileLogic --check-all`` runs in every kernel-generating build.

A test that still has to read the corpus carries ``@needs_logic_corpus``, so the
session summary counts it, and reaches the corpus only through
``require_logic_root`` or ``require_logic_file``. Those skip when the corpus is
absent, or fail when ``HIPBLASLT_REQUIRE_LOGIC_CORPUS`` is set, for lanes that
check the corpus out on purpose.

The logic-corpus component job (``tox -e logic-corpus``) runs every
``@needs_logic_corpus`` test with the corpus required whenever the logic
changes. This module imports neither tensilelite nor rocisa, so a test that
needs only the corpus can still run on a bare Python with ``--noconftest``.
"""

import os
from pathlib import Path

import pytest

LOGIC_ROOT_ENV = "HIPBLASLT_LOGIC_ROOT"
REQUIRE_ENV = "HIPBLASLT_REQUIRE_LOGIC_CORPUS"
MARKER = "needs_logic_corpus"

DEFAULT_LOGIC_ROOT = (
    Path(__file__).resolve().parents[4]
    / "library"
    / "src"
    / "amd_detail"
    / "rocblaslt"
    / "src"
    / "Tensile"
    / "Logic"
)

needs_logic_corpus = getattr(pytest.mark, MARKER)


def logic_root() -> Path:
    """Return ``$HIPBLASLT_LOGIC_ROOT``, else the ``Logic/`` directory of this checkout."""
    return Path(os.environ.get(LOGIC_ROOT_ENV) or DEFAULT_LOGIC_ROOT)


def corpus_required() -> bool:
    """Return whether a missing corpus is a failure rather than a skip."""
    return os.environ.get(REQUIRE_ENV, "").strip().lower() in {"1", "true", "yes"}


def require_logic_root() -> Path:
    """Return the ``Logic/`` root, skipping (or failing, if required) when it is absent."""
    root = logic_root()
    if not root.is_dir():
        message = (
            f"logic corpus not present at {root}: "
            "https://github.com/ROCm/rocm-libraries/issues/7481"
        )
        if corpus_required():
            pytest.fail(f"{message} ({REQUIRE_ENV} is set)", pytrace=False)
        pytest.skip(message)
    return root


def require_asm_full_root() -> Path:
    """Return the ``asm_full`` corpus directory under the ``Logic/`` root."""
    root = require_logic_root()
    asmFull = root / "asm_full"
    return asmFull if asmFull.is_dir() else root


def require_logic_file(rel: str) -> Path:
    """Return ``rel`` under ``asm_full``.

    Skips only when the whole corpus is absent. A corpus that is present but no
    longer has ``rel`` fails, so a retune that renames the file cannot quietly
    turn the test into a skip.
    """
    root = require_asm_full_root()
    path = root / rel
    if not path.is_file():
        pytest.fail(f"{rel} is not in the logic corpus at {root}; repoint the test", pytrace=False)
    return path
