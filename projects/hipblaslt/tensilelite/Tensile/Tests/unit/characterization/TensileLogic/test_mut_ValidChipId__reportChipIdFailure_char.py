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

"""Mutation-killing characterization test for ``_reportChipIdFailure``.

The function is a single statement::

    print(f"Error: {detail} (file: {filepath})", file=sys.stderr)

The four generated mutants are ``print``-argument mutations, not literal-segment
edits. Read directly from the mutants module, they are::

    mutmut_1: print(None, file=sys.stderr)                        # message -> None
    mutmut_2: print(f"Error: {detail} (file: {filepath})", file=None)
    mutmut_3: print(file=sys.stderr)                              # no positional arg
    mutmut_4: print(f"Error: {detail} (file: {filepath})", )       # file kwarg dropped

Each is killed by capturing both streams with ``capsys`` and asserting the
*exact* text on stderr together with the *emptiness* of stdout:

* ``mutmut_1`` writes the literal ``"None\\n"`` to stderr -> ``captured.err``
  differs from the expected message.
* ``mutmut_2`` passes ``file=None``; ``print`` treats ``file=None`` as
  ``sys.stdout``, so the message lands on stdout and stderr is empty ->
  ``captured.err`` is empty and ``captured.out`` is non-empty.
* ``mutmut_3`` drops the positional argument, emitting only a bare newline to
  stderr -> ``captured.err`` is ``"\\n"``.
* ``mutmut_4`` drops the ``file`` keyword entirely; ``print`` then defaults to
  ``sys.stdout``, so (like ``mutmut_2``) the message lands on stdout and stderr
  is empty.

Asserting the exact stderr text *and* an empty stdout in the same test
distinguishes every mutant from the original in a single observation. These are
characterization assertions: they pin the message format exactly as it is
emitted today (including the trailing newline ``print`` adds) and the fact that
it goes to stderr. A future change to the wording or the stream must update this
test.
"""

from pathlib import Path

import pytest

from Tensile.TensileLogic.ValidChipId import _reportChipIdFailure

pytestmark = pytest.mark.unit


def test_reports_exact_message_to_stderr(capsys):
    """Pins the full message text, the destination stream, and the newline.

    This single assertion pair kills all four mutants: the exact ``captured.err``
    string differs under ``mutmut_1`` (``"None\\n"``) and ``mutmut_3``
    (``"\\n"``), while the empty ``captured.out`` / populated stderr expectation
    is violated by ``mutmut_2`` and ``mutmut_4`` (both route to stdout).
    """
    ret = _reportChipIdFailure(Path("some/dir/logic.yaml"), "bad placement")

    captured = capsys.readouterr()
    assert ret is None
    assert captured.err == "Error: bad placement (file: some/dir/logic.yaml)\n"
    assert captured.out == ""


def test_detail_and_filepath_are_interpolated_in_order(capsys):
    """A distinct detail/path pair pins the ordering of the two interpolations.

    Confirms the message is not a fixed constant: both ``detail`` and
    ``filepath`` flow through to stderr in the documented order and slots.
    """
    _reportChipIdFailure(Path("gfx950_id75a3/x.yaml"), "unexpected chip id")

    captured = capsys.readouterr()
    assert captured.err == "Error: unexpected chip id (file: gfx950_id75a3/x.yaml)\n"
    assert captured.out == ""


def test_empty_detail_still_emits_fixed_literals(capsys):
    """Empty detail leaves only the literal scaffolding plus the path.

    With ``detail=""`` the exact stderr text is the literal glue around the
    interpolated path, pinning that the surrounding wording is emitted verbatim.
    """
    _reportChipIdFailure(Path("logic.yaml"), "")

    captured = capsys.readouterr()
    assert captured.err == "Error:  (file: logic.yaml)\n"
    assert captured.out == ""


def test_nothing_written_to_stdout(capsys):
    """Explicitly pins that stdout stays empty (kills mutmut_2 and mutmut_4).

    ``file=None`` (mutmut_2) and a dropped ``file`` keyword (mutmut_4) both make
    ``print`` default to ``sys.stdout``; asserting stdout is empty while stderr
    carries the message catches either redirection.
    """
    _reportChipIdFailure(Path("a/b/c.yaml"), "detail here")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: detail here (file: a/b/c.yaml)\n"
