################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.isRhel8``.

``isRhel8`` reads ``/etc/os-release`` and returns True iff the contents match a
RHEL-8 NAME/VERSION_ID pattern, emitting a warning in that case.

These tests pin the ACTUAL current behavior (the exact file path read and the
exact warning string emitted) so they pass on clean source and fail under the
surviving mutants that alter the path literal (2, 3, 4) or the warning argument
text (25, 26, 27, 28).
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit

# Contents that satisfy the RHEL-8 detection regex used by isRhel8.
RHEL8_OS_RELEASE = (
    'NAME="Red Hat Enterprise Linux"\n'
    'VERSION="8.6 (Ootpa)"\n'
    'VERSION_ID="8.6"\n'
)


class _FakePath:
    """Minimal stand-in for pathlib.Path used by isRhel8.

    Reports existence/contents only for the canonical ``/etc/os-release``
    argument; any other path (i.e. a mutated literal) is treated as missing.
    """

    def __init__(self, arg):
        self._arg = arg
        self._is_canonical = arg == "/etc/os-release"

    def exists(self):
        return self._is_canonical


def _install_fake_open(monkeypatch, content):
    import builtins

    real_open = builtins.open

    def fake_open(file, *args, **kwargs):
        if isinstance(file, _FakePath):
            from io import StringIO

            return StringIO(content)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_isRhel8_reads_canonical_etc_os_release_path(monkeypatch):
    """isRhel8 reads exactly ``/etc/os-release``.

    Kills mutant_2 (``Path(None)`` -> TypeError), mutant_3
    (``Path("/etc/os-release")`` literal corrupted) and mutant_4
    (``Path("/ETC/OS-RELEASE")``): under each mutant the fake Path is
    constructed with a non-canonical argument, reports ``exists()`` False, and
    isRhel8 returns False instead of the original True.
    """
    monkeypatch.setattr(U, "Path", _FakePath)
    _install_fake_open(monkeypatch, RHEL8_OS_RELEASE)
    # silence the real warning so the test does not depend on logging output
    monkeypatch.setattr(U, "printWarning", lambda *a, **k: None)

    assert U.isRhel8() is True


def test_isRhel8_emits_exact_warning_text_on_match(monkeypatch):
    """On a RHEL-8 match isRhel8 calls printWarning with the exact message.

    Kills mutant_25 (arg -> None), mutant_26 (XX-wrapped text), mutant_27
    (lowercased 'rhel8') and mutant_28 (fully upper-cased text): each changes
    the single positional argument passed to printWarning away from the
    original literal.
    """
    monkeypatch.setattr(U, "Path", _FakePath)
    _install_fake_open(monkeypatch, RHEL8_OS_RELEASE)

    calls = []
    monkeypatch.setattr(U, "printWarning", lambda *a, **k: calls.append(a))

    result = U.isRhel8()

    assert result is True
    assert calls == [
        ("Rhel8 environments may not support all tools for system queries such as amd-smi.",)
    ]
