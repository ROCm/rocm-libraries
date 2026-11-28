# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Markdown to the Atlassian Document Format (ADF), for a JIRA ticket's description through the atlassian CLI.

The CLI's `--description` stores its text as it is (`##` shows as two hashes), so a formatted description goes in as
ADF: `rk atlassian --site <site> --json jira issue create ... --field-json "description=$(rk md2adf FILE)"` (or the
same `--field-json` on `jira issue update KEY`). A comment (`jira issue comment add --body`) takes plain text only:
write it without Markdown marks.

What it reads: headings (`#` to `######`), paragraphs (lines joined), bullet lists (`-`, `*`) and numbered lists
(`1.`), a list item's continuation lines, fenced code blocks (with their language), and inline `code`, **bold**,
*italic*, [text](url) links and bare http(s) URLs. Anything else is text.
"""

from __future__ import annotations

import json
import re
import sys

INLINE = re.compile(
    r"`([^`]+)`|\*\*([^*]+)\*\*|\[([^\]]+)\]\((https?://[^)\s]+)\)|(?<![\w*])\*([^*\s][^*]*)\*(?!\w)"
    r"|(https?://[^\s)<>\]]+[^\s)<>\].,;:!?'\"])"
)
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
NUMBER = re.compile(r"^\s*\d+[.)]\s+(.*)$")
FENCE = re.compile(r"^\s*(```|~~~)\s*([\w+-]*)\s*$")


def text(t: str, marks: list | None = None) -> dict:
    node = {"type": "text", "text": t}
    if marks:
        node["marks"] = marks
    return node


def inline(t: str) -> list[dict]:
    """A line's text nodes, with code, bold, italic and link marks."""
    out, pos = [], 0
    for m in INLINE.finditer(t):
        if m.start() > pos:
            out.append(text(t[pos : m.start()]))
        code, bold, label, href, italic, url = m.groups()
        if code is not None:
            out.append(text(code, [{"type": "code"}]))
        elif bold is not None:
            out.append(text(bold, [{"type": "strong"}]))
        elif label is not None:
            out.append(text(label, [{"type": "link", "attrs": {"href": href}}]))
        elif italic is not None:
            out.append(text(italic, [{"type": "em"}]))
        else:
            out.append(text(url, [{"type": "link", "attrs": {"href": url}}]))
        pos = m.end()
    if pos < len(t):
        out.append(text(t[pos:]))
    return [n for n in out if n["text"]]


def _para(t: str) -> dict:
    content = inline(t)
    return (
        {"type": "paragraph", "content": content} if content else {"type": "paragraph"}
    )


def convert(md: str) -> dict:
    """The ADF document for a Markdown text."""
    blocks: list[dict] = []
    para: list[str] = []
    items: list[str] = []
    kind = ""
    code: list[str] | None = None
    lang = ""

    def flush() -> None:
        nonlocal para, items, kind
        if para:
            blocks.append(_para(" ".join(para)))
            para = []
        if items:
            blocks.append(
                {
                    "type": "orderedList" if kind == "number" else "bulletList",
                    "content": [
                        {"type": "listItem", "content": [_para(i)]} for i in items
                    ],
                }
            )
            items, kind = [], ""

    for line in md.splitlines():
        fence = FENCE.match(line)
        if code is not None:
            if fence:
                node = {
                    "type": "codeBlock",
                    "content": [text("\n".join(code))] if code else [],
                }
                if lang:
                    node["attrs"] = {"language": lang}
                blocks.append(node)
                code = None
            else:
                code.append(line)
            continue
        if fence:
            flush()
            code, lang = [], fence.group(2)
            continue
        if not line.strip():
            flush()
            continue
        m = HEADING.match(line)
        if m:
            flush()
            blocks.append(
                {
                    "type": "heading",
                    "attrs": {"level": len(m.group(1))},
                    "content": inline(m.group(2)),
                }
            )
            continue
        b, n = BULLET.match(line), NUMBER.match(line)
        if b or n:
            new = "bullet" if b else "number"
            if para or (items and kind != new):
                flush()
            kind = new
            items.append((b or n).group(1).strip())
            continue
        if items:
            items[-1] += " " + line.strip()
        else:
            para.append(line.strip())
    if code is not None:  # an unclosed fence: its lines are code
        blocks.append(
            {"type": "codeBlock", "content": [text("\n".join(code))] if code else []}
        )
    flush()
    return {"type": "doc", "version": 1, "content": blocks}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] == "-":
        src = sys.stdin.read()
    else:
        with open(argv[0], encoding="utf-8") as f:
            src = f.read()
    print(json.dumps(convert(src), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
