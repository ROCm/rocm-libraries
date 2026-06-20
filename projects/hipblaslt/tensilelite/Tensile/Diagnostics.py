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
################################################################################

"""TensileLite Python diagnostics.

The Python counterpart of ``client/include/Diagnostic.hpp``; the two share one
record format and one tag so a failure that crosses the Python/C++ boundary reads
the same on both sides. See ``client/DIAGNOSTICS.md`` for the full model.

Principles:

* every failure carries context (what, where, the error, a next step);
* one stable greppable tag, :data:`DIAGNOSTIC_TAG`, on every record;
* hybrid output: a single-line logfmt record (survives log truncation and is
  machine-parseable) plus a human banner;
* no bare ``print`` on a failure path -- build a :class:`Diagnostic`.

Records are written to ``stderr`` (matching the C++ client), which pytest and CI
capture verbatim.
"""

import sys

DIAGNOSTIC_TAG = "[tensilelite:diag]"


def _logfmt_value(value):
    """Quote and escape a field value for the single-line logfmt record."""
    text = str(value)
    needs_quote = text == "" or any(c in text for c in (" ", '"', "=", "\n", "\t", "\r"))
    if not needs_quote:
        return text
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )
    return '"' + escaped + '"'


class Diagnostic:
    """A tagged, contextual failure record emitted in the hybrid format.

    Build one with a severity and a short, stable category, attach context as
    ordered key/value fields, then :meth:`emit`. Mirrors the C++
    ``TensileLite::Client::Diagnostic``.
    """

    FATAL = "FATAL"
    ERROR = "ERROR"
    WARNING = "WARNING"

    def __init__(self, severity, category):
        self._severity = severity
        self._category = category
        self._fields = []

    def field(self, key, value):
        """Attach a context field; returns ``self`` for chaining."""
        self._fields.append((str(key), str(value)))
        return self

    def next(self, advice):
        """Attach the recommended next-step field; returns ``self`` for chaining."""
        return self.field("next", advice)

    def one_line(self):
        """Return the single-line logfmt record."""
        parts = [
            DIAGNOSTIC_TAG,
            "level=" + self._severity,
            "cat=" + _logfmt_value(self._category),
        ]
        for key, value in self._fields:
            parts.append(key + "=" + _logfmt_value(value))
        return " ".join(parts)

    def banner(self):
        """Return the human-readable banner block."""
        bar = "*" * 72
        key_width = max((len(k) for k, _ in self._fields), default=0)
        lines = [
            bar,
            "* TENSILELITE DIAGNOSTIC - %s  [%s]" % (self._category, self._severity),
        ]
        for key, value in self._fields:
            lines.append("* %s : %s" % (key.ljust(key_width), value))
        lines.append(bar)
        return "\n".join(lines)

    def emit(self):
        """Write the logfmt line and the banner to ``stderr``."""
        sys.stderr.write(self.one_line() + "\n")
        sys.stderr.write(self.banner() + "\n")
        sys.stderr.flush()
