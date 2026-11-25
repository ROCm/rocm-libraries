# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Custom headers as text: one `Name: value` header a line, the form of Claude Code's ANTHROPIC_CUSTOM_HEADERS, and
the mark that stands for the key in a request's headers. This is the one home of both.

- `header_lines(text)`: [(name, value)] for each line of TEXT that is not blank, in order, each stripped; a line with
  no colon comes back as ("", the line stripped), so that a caller can say it does not read as a header.
- `KEY_MARK`: what stands for the key in a binding's key_headers (rkmon_lib/bindings.py gives them to a claude binding
  reached as the Claude session is): worker/llm_client.py puts the key in its place in each request, and the round
  trip shows it in the key's place (tools/round_trip.py).

rkmon_lib/orchestrator.py reads the Claude session's headers with it, and rkmon_lib/claude_env_shape.py an older
package's session settings. It imports nothing: root's one-time move loads it by its path from the image's own code
(docker/rk_root.py), the deploy side reads it on the host, where no worker/ is, and the header values it gives hold
secrets, which no caller prints.
"""

# What stands for the key in a binding's key_headers: each request puts the key in its place.
KEY_MARK = "<key>"


def header_lines(text: str) -> list[tuple[str, str]]:
    """The headers of TEXT, `Name: value` per line, in order: see the module's docstring."""
    out = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        name, colon, value = line.partition(":")
        out.append((name.strip(), value.strip()) if colon else ("", line.strip()))
    return out
