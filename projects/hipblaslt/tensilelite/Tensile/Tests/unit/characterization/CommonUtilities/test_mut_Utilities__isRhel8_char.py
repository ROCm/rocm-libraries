################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-killing characterization test for ``Tensile.Common.Utilities.isRhel8``.

Targets survivor ``x_isRhel8__mutmut_14`` which rewrites the open call from
``open(file, "r")`` (explicit read mode passed positionally) to ``open(file, )``
(the mode argument dropped entirely). Because ``"r"`` is Python's default mode,
the two forms are *observationally identical* through the returned file object --
the only difference is the exact positional argument list handed to ``open``.

This module therefore pins that call shape directly: it installs a recording
``open`` and asserts the second positional argument is the literal ``"r"``. Under
the mutant only a single positional argument is supplied, so the recorded call
differs and the assertion fails. The remaining tests pin the surrounding branch
behavior (path existence gate, regex match/no-match, warning emission) so every
return site of the current implementation is exercised exactly as it runs today.
"""

import builtins
import importlib
from io import StringIO

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit

# Contents that satisfy the RHEL-8 detection regex used by isRhel8.
RHEL8_OS_RELEASE = (
    'NAME="Red Hat Enterprise Linux"\n'
    'VERSION="8.6 (Ootpa)"\n'
    'VERSION_ID="8.6"\n'
)

# Contents that do NOT match the RHEL-8 pattern.
NON_RHEL8_OS_RELEASE = (
    'NAME="Ubuntu"\n'
    'VERSION="22.04.3 LTS (Jammy Jellyfish)"\n'
    'VERSION_ID="22.04"\n'
)


class _FakePath:
    """Minimal stand-in for pathlib.Path used by isRhel8.

    Reports existence for the canonical ``/etc/os-release`` argument (unless
    ``force_missing`` is set); any other constructor argument (i.e. a mutated
    path literal) is treated as missing.
    """

    force_missing = False

    def __init__(self, arg):
        self.arg = arg
        self._is_canonical = arg == "/etc/os-release"

    def exists(self):
        if _FakePath.force_missing:
            return False
        return self._is_canonical


def _install_recording_open(monkeypatch, content):
    """Replace builtins.open with a recorder that returns ``content``.

    Returns the ``calls`` list; each entry is the ``(args, kwargs)`` tuple that
    isRhel8 used to invoke ``open`` for a ``_FakePath`` argument. Real files are
    passed through to the genuine ``open`` so unrelated machinery is unaffected.
    """
    calls = []
    real_open = builtins.open

    def fake_open(*args, **kwargs):
        if args and isinstance(args[0], _FakePath):
            calls.append((args, kwargs))
            return StringIO(content)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    return calls


@pytest.fixture(autouse=True)
def _reset_fake_path():
    _FakePath.force_missing = False
    yield
    _FakePath.force_missing = False


def test_isRhel8_opens_file_with_explicit_read_mode(monkeypatch):
    """isRhel8 calls ``open(file, "r")`` -- mode passed positionally as "r".

    Kills mutant_14 (``open(file, )`` drops the mode argument): under the mutant
    only one positional argument reaches ``open`` so ``args[1]`` does not exist.
    Also distinguishes mutant_12 (``open(file, None)``) and mutant_15
    (``open(file, "XXrXX")``) which change the second positional to a non-"r"
    value, and mutant_13 (``open("r")``) which supplies only ``"r"``.
    """
    monkeypatch.setattr(U, "Path", _FakePath)
    monkeypatch.setattr(U, "printWarning", lambda *a, **k: None)
    calls = _install_recording_open(monkeypatch, RHEL8_OS_RELEASE)

    result = U.isRhel8()

    assert result is True
    assert len(calls) == 1
    args, kwargs = calls[0]
    # Two positional arguments: the path object and the mode string "r".
    assert len(args) == 2
    assert isinstance(args[0], _FakePath)
    assert args[0].arg == "/etc/os-release"
    assert args[1] == "r"
    assert kwargs == {}


def test_isRhel8_returns_True_and_warns_on_match(monkeypatch, capsys):
    """A matching /etc/os-release yields True and emits the exact warning.

    Pins the match branch (``if match: return True``) and the exact positional
    warning text via the real printWarning captured on stdout.
    """
    monkeypatch.setattr(U, "Path", _FakePath)
    _install_recording_open(monkeypatch, RHEL8_OS_RELEASE)

    result = U.isRhel8()

    assert result is True
    captured = capsys.readouterr()
    assert (
        "Rhel8 environments may not support all tools for system queries such as amd-smi."
        in captured.out
    )


def test_isRhel8_warning_argument_is_exact(monkeypatch):
    """printWarning receives exactly the single canonical message string.

    Pins the warning argument so any mutation of that literal is detected.
    """
    monkeypatch.setattr(U, "Path", _FakePath)
    _install_recording_open(monkeypatch, RHEL8_OS_RELEASE)

    calls = []
    monkeypatch.setattr(U, "printWarning", lambda *a, **k: calls.append((a, k)))

    result = U.isRhel8()

    assert result is True
    assert calls == [
        (
            (
                "Rhel8 environments may not support all tools for system queries such as amd-smi.",
            ),
            {},
        )
    ]


def test_isRhel8_returns_False_on_non_matching_content(monkeypatch):
    """Non-RHEL8 content reaches the final ``return False`` with no warning.

    Pins the no-match path: the regex fails, printWarning is never called, and
    the trailing return is False.
    """
    monkeypatch.setattr(U, "Path", _FakePath)
    _install_recording_open(monkeypatch, NON_RHEL8_OS_RELEASE)

    warned = []
    monkeypatch.setattr(U, "printWarning", lambda *a, **k: warned.append(a))

    result = U.isRhel8()

    assert result is False
    assert warned == []


def test_isRhel8_reads_canonical_path_only(monkeypatch):
    """isRhel8 constructs Path("/etc/os-release") -- exact literal.

    Kills mutants that corrupt the path literal (e.g. Path(None), the XX-wrapped
    string, or the upper-cased path): the fake Path reports the canonical
    argument as present, so a mutated argument would report missing and short
    circuit to False.
    """
    seen = []
    orig_fake = _FakePath

    def recording_path(arg):
        seen.append(arg)
        return orig_fake(arg)

    monkeypatch.setattr(U, "Path", recording_path)
    monkeypatch.setattr(U, "printWarning", lambda *a, **k: None)
    _install_recording_open(monkeypatch, RHEL8_OS_RELEASE)

    result = U.isRhel8()

    assert result is True
    assert seen == ["/etc/os-release"]


def test_isRhel8_returns_False_when_file_missing(monkeypatch):
    """When /etc/os-release does not exist, isRhel8 returns False immediately.

    Pins the existence gate (``if not file.exists(): return False``): open is
    never called and the result is False. Distinguishes the mutant that flips the
    early-return value to True.
    """
    _FakePath.force_missing = True
    monkeypatch.setattr(U, "Path", _FakePath)

    calls = _install_recording_open(monkeypatch, RHEL8_OS_RELEASE)
    warned = []
    monkeypatch.setattr(U, "printWarning", lambda *a, **k: warned.append(a))

    result = U.isRhel8()

    assert result is False
    assert calls == []
    assert warned == []
