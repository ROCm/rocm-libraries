#!/usr/bin/env python3
"""mutant-identity.py - Issue 15: stable mutant identity for CI fail-on-regression.

mutmut's ordinal IDs (e.g. `xǁSpinnyThingǁincrement__mutmut_7`) are position-based:
adding/removing any mutable construct earlier in a function renumbers every later
mutant. CI runs exactly when a module changed, so ordinal IDs produce false
regressions/misses. This module computes a stable identity that survives ordinal
renumbering (and, via the text anchor, line moves):

    identity = (file_path, line_anchor, diff_hash)

- file_path : repo-relative source path (path-separator normalized).
- line_anchor: the STRIPPED text of the mutated (original) line - survives line
  moves; the raw line number is kept as a secondary human field only.
- diff_hash : sha256 over the CANONICALIZED diff (see canonicalize_diff): only the
  changed (+/-) lines, with mutmut ordinal/name headers, git `index` blob hashes,
  file headers, and `@@` hunk line numbers removed, trailing whitespace and CRLF
  normalized, and LEADING indentation preserved (indentation is semantically
  meaningful in Python, so an indentation mutation is a distinct mutant). Context
  lines are excluded so the hash is robust to nearby (non-mutated) context changes.

Doc: spikes/mutant-identity.md. No mutmut is run; no source edited.

Python 3.8 compatible.
"""

import argparse
import hashlib
import json
import sys


def canonicalize_diff(diff_text):
    """Return the canonical changed-lines body used for hashing (see module doc).

    Uses proper unified-diff structure: everything BEFORE the first `@@` hunk is
    header (`diff --git`, `index <blob>`, `--- a/`, `+++ b/`, and any mutmut
    `# mutant ...__mutmut_N` name line) and is dropped; the `@@` hunk lines
    (whose line numbers churn) are dropped; INSIDE hunks only the changed (`+`/`-`)
    lines are kept - so a deletion whose own content starts with `--`/`++` is
    correctly retained, not mistaken for a file header. Context/blank lines are
    excluded (robust to nearby edits). Trailing whitespace + CRLF are normalized;
    LEADING indentation is preserved (semantic in Python).
    """
    out = []
    in_hunk = False
    for raw in diff_text.splitlines():
        line = raw.rstrip("\r\n")
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue  # pre-hunk header (git/index/---/+++/ordinal name) -> drop
        if line[:1] in ("+", "-"):
            out.append(line[0] + line[1:].rstrip())
        # context / blank / "\ No newline" lines excluded
    return "\n".join(out)


def diff_hash(diff_text, length=16):
    canon = canonicalize_diff(diff_text)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:length]


def original_line(diff_text):
    """The first removed ('-') content line, stripped - the mutated original line.

    For pure-insertion mutants (no '-' line) returns the first added line instead.
    """
    first_add = None
    in_hunk = False
    for raw in diff_text.splitlines():
        line = raw.rstrip("\r\n")
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("-"):
            return line[1:].strip()
        if line.startswith("+") and first_add is None:
            first_add = line[1:].strip()
    return first_add or ""


def normalize_path(p):
    return p.replace("\\", "/")


def identity(file_path, diff_text, line_anchor=None, line_number=None):
    """Return the stable identity dict for a mutant."""
    anchor = line_anchor if line_anchor is not None else original_line(diff_text)
    return {
        "file": normalize_path(file_path),
        "line_anchor": anchor.strip(),
        "diff_hash": diff_hash(diff_text),
        "line_number": line_number,  # secondary/human only; not part of the stable match
    }


def key_tuple(ident):
    """The tuple CI matches on (line_number deliberately excluded)."""
    return (ident["file"], ident["line_anchor"], ident["diff_hash"])


def _read(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r") as fh:
        return fh.read()


def cmd_hash(args):
    print(diff_hash(_read(args.diff)))
    return 0


def cmd_identity(args):
    diff_text = _read(args.diff)
    ident = identity(
        args.file or "",
        diff_text,
        line_anchor=args.line_text,
        line_number=args.line_number,
    )
    print(json.dumps(ident, indent=2, sort_keys=True))
    return 0


def cmd_canon(args):
    sys.stdout.write(canonicalize_diff(_read(args.diff)) + "\n")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Stable mutant identity (file + line anchor + diff-hash)"
    )
    sub = p.add_subparsers(dest="cmd")
    ph = sub.add_parser("hash", help="print the diff-hash of a diff")
    ph.add_argument("--diff", required=True)
    pi = sub.add_parser("identity", help="print the full identity JSON for a diff")
    pi.add_argument("--diff", required=True)
    pi.add_argument("--file", default="")
    pi.add_argument("--line-text", default=None)
    pi.add_argument("--line-number", type=int, default=None)
    pc = sub.add_parser(
        "canon", help="print the canonicalized diff body that gets hashed"
    )
    pc.add_argument("--diff", required=True)
    args = p.parse_args(argv)
    if args.cmd == "hash":
        return cmd_hash(args)
    if args.cmd == "identity":
        return cmd_identity(args)
    if args.cmd == "canon":
        return cmd_canon(args)
    p.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
