# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The canonical form of an emitter's ck.dsl.ir output (python_batch_worker.py).

The canonical golden records this form's hash. An emitter that prints several
kernels, with a banner line before each (fused_moe_e2e), used to have no
canonical form at all, and the golden recorded its raw bytes instead. Each
kernel now canonicalizes on its own, and an output that still has no canonical
form fails the golden check rather than standing in with its raw hash.

Host-only. The inputs are real emitter outputs, run as the golden runs them.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_diff
from python_batch_worker import canonical_digest, canonical_text
from rocke.core.ir_serialize import canonicalize, parse

BANNER = "; === fused_moe_e2e stage: "


def _emit(family: str, index: int, src_dir: Path = run_diff.PARITY) -> str:
    """The ir-mode output of one config of the parity family in ``src_dir``."""
    emitter = src_dir / f"{family}_emit.py"
    return subprocess.run(
        [sys.executable, str(emitter), str(index), "ir"],
        capture_output=True,
        text=True,
        check=True,
        env=run_diff.python_emitter_env(src_dir),
        timeout=run_diff.TIMEOUT,
    ).stdout


@pytest.fixture(scope="module")
def one_kernel() -> str:
    return _emit("ir_serialize", 0)


@pytest.fixture(scope="module")
def staged() -> str:
    return _emit("fused_moe_e2e", 0)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_one_kernel_keeps_the_canonical_form_it_had(one_kernel):
    # The hashes already blessed for one-kernel outputs stay valid.
    assert canonical_text(one_kernel) == canonicalize(parse(one_kernel))
    assert canonical_digest(one_kernel.encode()) == (
        _sha(canonicalize(parse(one_kernel))),
        None,
    )


def test_each_kernel_of_a_staged_output_is_canonicalized(staged):
    banners = [line for line in staged.splitlines() if line.startswith(BANNER)]
    assert len(banners) == 3
    canonical = canonical_text(staged)
    assert [line for line in canonical.splitlines() if line.startswith(BANNER)] == (
        banners
    )
    assert canonical.count("ckdsl.ir v1\n") == staged.count("ckdsl.ir v1\n") == 3
    assert canonical_digest(staged.encode())[1] is None


def test_every_kernel_counts_when_no_banner_separates_them():
    # The 3D attention emitter prints the attention kernel and its reduce
    # kernel back to back. The parser reads one kernel and stops, so a hash
    # of the parsed text alone never covered the reduce kernel.
    if not (run_diff.LIB_PARITY / "gfx942_attention_tiled_3d_emit.py").is_file():
        pytest.skip("platform-only tree: no library parity families")
    two = _emit("gfx942_attention_tiled_3d", 0, run_diff.LIB_PARITY)
    second = two.index("ckdsl.ir v1", 1)
    first_kernel, second_kernel = two[:second], two[second:]
    assert canonical_text(two) == canonicalize(parse(first_kernel)) + canonicalize(
        parse(second_kernel)
    )
    assert canonical_text(two) != canonicalize(parse(two))


def test_a_renamed_stage_changes_the_canonical_hash(staged):
    renamed = staged.replace("stage: ROUTER", "stage: ROUTING", 1)
    assert canonical_digest(renamed.encode())[0] != canonical_digest(staged.encode())[0]


def test_a_dropped_stage_changes_the_canonical_hash(staged):
    last = staged.rindex("\n" + BANNER)
    assert (
        canonical_digest(staged[: last + 1].encode())[0]
        != canonical_digest(staged.encode())[0]
    )


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "no ck.dsl.ir kernel"),
        ("; === only a banner ===\n", "no ck.dsl.ir kernel"),
        ("define void @k() {\n}\n", "bad header"),
    ],
)
def test_output_without_a_kernel_has_no_canonical_form(text, message):
    sha, error = canonical_digest(text.encode())
    assert sha is None
    assert message in error


def test_a_kernel_that_does_not_parse_has_no_canonical_form(staged):
    broken = staged.replace("params {", "params [", 1)
    sha, error = canonical_digest(broken.encode())
    assert sha is None
    assert "expected 'params {'" in error
