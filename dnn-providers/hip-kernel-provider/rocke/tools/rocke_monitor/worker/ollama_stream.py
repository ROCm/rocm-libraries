# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Streamed Ollama /api/chat for the review worker, with a fast check for a wedged runner.

About one fresh model load in eight hangs on its first prompt batch. The runner logs "new prompt", then nothing, and
the request would sit until the 900 s stall timeout, three in a row costing a draft. A healthy runner logs a prompt-processing line every batch (well under a second each),
so while it waits for the first byte this reads the ollama journal: a "new prompt" line with no progress after it for
`hang_s`, or progress that stops for `hang_s`, abandons the request as RunnerHung. When the journal cannot be read, or
shows no "new prompt" line, only the stall timeout applies, as before."""

from __future__ import annotations

import http.client
import json
import re
import select
import subprocess
import time
import urllib.parse

_PROMPT = re.compile(r"\bnew prompt, n_ctx_slot\b")
# "cached n_tokens = 0" follows every new prompt, hung or not: only a nonzero count is progress.
_PROGRESS = re.compile(
    r"cached n_tokens = [1-9]|prompt processing, n_tokens|restored context checkpoint"
)


class RunnerHung(TimeoutError):
    """The runner took the prompt and then made no progress for hang_s."""


def parse_progress(text: str, since: float):
    """From `journalctl -o short-unix` text: (time of the last "new prompt" line at or after `since`, time of the last
    progress line after it, or None for either)."""
    t_prompt = t_prog = None
    for ln in text.splitlines():
        try:
            t = float(ln.split(" ", 1)[0])
        except ValueError:
            continue
        if t < since:
            continue
        if _PROMPT.search(ln):
            t_prompt, t_prog = t, None
        elif t_prompt is not None and _PROGRESS.search(ln):
            t_prog = t
    return t_prompt, t_prog


def journal_progress(since: float):
    """parse_progress over the ollama service journal since `since`; (None, None) when it cannot be read."""
    try:
        out = subprocess.run(
            [
                "journalctl",
                "-u",
                "ollama",
                "--since",
                f"@{int(since)}",
                "--no-pager",
                "-q",
                "-o",
                "short-unix",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return parse_progress(out, since)


def stream_chat(
    url: str,
    data: bytes,
    stall_s: float,
    *,
    hang_s: float,
    poll_s: float = 15,
    progress=journal_progress,
    headers: dict | None = None,
) -> dict:
    """POST `data` (an /api/chat body with "stream": true) and join the streamed chunks into one assistant message.
    Raises RunnerHung, TimeoutError (nothing for stall_s), ConnectionError or http.client errors; RuntimeError for an
    error chunk from ollama. Runner progress counts as an answer: prompt processing streams nothing, and a 234k-token
    prompt takes about 13 min, past the 900 s stall timeout, while it progresses all the way.
    """
    u = urllib.parse.urlsplit(url)
    cls = (
        http.client.HTTPSConnection
        if u.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = cls(
        u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=stall_s
    )
    try:
        t_sent = time.time()
        conn.request(
            "POST",
            u.path or "/",
            body=data,
            headers=headers or {"Content-Type": "application/json"},
        )
        if (
            u.scheme == "https"
        ):  # the TLS socket may hold decrypted bytes select() cannot see: no early peek
            poll_s = stall_s
        while not select.select([conn.sock], [], [], poll_s)[0]:
            now = time.time()
            t_prompt, t_prog = progress(t_sent)
            if t_prompt is not None and now - (t_prog or t_prompt) >= hang_s:
                where = (
                    "after a prompt batch" if t_prog else "on its first prompt batch"
                )
                raise RunnerHung(
                    f"the runner hung {where}: no progress for {now - (t_prog or t_prompt):.0f} s"
                )
            if now - max(t_sent, t_prog or 0) >= stall_s:
                raise TimeoutError(f"no answer or runner progress in {stall_s:.0f} s")
        r = conn.getresponse()
        if r.status != 200:
            raise ConnectionError(f"HTTP {r.status}: {r.read(300)!r}")
        msg = {"role": "assistant", "content": "", "thinking": "", "tool_calls": []}
        for line in r:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(f"ollama: {chunk['error']}")
            m = chunk.get("message") or {}
            msg["content"] += m.get("content") or ""
            msg["thinking"] += m.get("thinking") or ""
            msg["tool_calls"] += m.get("tool_calls") or []
            if chunk.get("done"):
                return msg
        raise ConnectionError("the stream ended before done")
    finally:
        conn.close()
