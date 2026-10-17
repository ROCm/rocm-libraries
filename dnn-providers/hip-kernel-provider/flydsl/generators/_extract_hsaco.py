# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Recover the device ELF from a FlyDSL IR dump.

FlyDSL has no "emit an object file" entry point. Under ``FLYDSL_DUMP_IR=1`` it
writes one ``.mlir`` per pass into ``FLYDSL_DUMP_DIR/<kernel>/``, and the dump
taken *after* the ``gpu-module-to-binary`` pass carries the finished code object
inline, as the ``bin`` string attribute of a ``gpu.binary`` op. Scraping it back
out is the only way to get a standalone HSACO.

**This is a private contract with the compiler, not an API.** Two properties it
depends on, both asserted rather than assumed:

* *The file is matched by pass name, never by the numeric stage prefix.* The
  prefix is an ordinal into whatever pipeline that build of FlyDSL runs, so it
  moves when a pass is added: the POC this is ported from documented "stage 19"
  and flydsl 0.3.4 emits ``20_gpu_module_to_binary.mlir`` out of 23 stages. The
  glob below has always keyed on the name; only the prose was stale.
* *The extracted bytes begin with ELF magic.* A pipeline change that stops
  embedding the binary, or embeds something else, then fails here rather than
  writing a plausible-looking file that will not load.

**One ``gpu.binary`` holds one object per attached target, not one per kernel.**
MLIR's own model says so -- ``#gpu.select_object<#rocdl.target>`` exists to pick
among them -- and FlyDSL uses it: the ``rocdl-attach-target`` stage attaches both
``#rocdl.target<chip = "gfx1151">`` and ``#rocdl.target<chip = "gfx1151",
flags = {no_wave64}>``, so a gfx1151 dump carries two ``bin`` attributes even
though the backend's ``gpu_module_targets()`` returns a single target. On the
objects measured they are byte-identical (same length, same SHA256), which is
what makes "take any one of them" a defensible read rather than a guess -- so
that identity is the acceptance condition below, and a dump whose objects
genuinely *differ* still fails. Choosing between two different code objects is a
selection decision, and a generator that makes it silently is how an arch ships
the wrong binary.
"""

from __future__ import annotations

from pathlib import Path

from ._manifest import sha256

# The pass whose post-dump carries the embedded code object. Name, not ordinal.
_BINARY_PASS_SUFFIX = "_gpu_module_to_binary.mlir"

# `bin = "<escaped bytes>"` inside the gpu.binary op.
_BIN_MARKER = 'bin = "'

# The `gpu.object<#rocdl.target<...>, kernels = ...>` that wraps each `bin`.
_OBJECT_MARKER = "gpu.object<"

_ELF_MAGIC = b"\x7fELF"


class HsacoExtractionError(RuntimeError):
    """The dump did not contain a recoverable code object."""


def decode_mlir_string(text: str, start: int) -> tuple[bytes, int]:
    """Decode one MLIR string literal body, returning its bytes and end offset.

    MLIR prints printable characters raw, ``\\`` as ``\\\\``, ``"`` as ``\\"``,
    and every other byte as ``\\XX`` (two uppercase hex digits). ``start`` is the
    offset of the first character *inside* the quotes; the returned offset is
    that of the closing quote.

    Decoding and terminator-finding are one pass on purpose. Scanning ahead for
    an unescaped quote by testing whether the preceding character is a backslash
    misreads a body whose final byte is 0x5C -- printed ``\\\\`` -- as an escaped
    quote and runs off the end of the literal. Consuming escapes as they are
    decoded cannot make that mistake.
    """
    out = bytearray()
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            return bytes(out), i
        if ch == "\\":
            if i + 1 >= n:
                break
            nxt = text[i + 1]
            if nxt == "\\":
                out.append(0x5C)
                i += 2
                continue
            if nxt == '"':
                out.append(0x22)
                i += 2
                continue
            if i + 2 >= n:
                break
            out.append(int(text[i + 1 : i + 3], 16))
            i += 3
            continue
        # Each code point in the dump stands for one byte; the reader must have
        # opened the file with a byte-preserving decode (see hsaco_from_dump).
        out.append(ord(ch))
        i += 1
    raise HsacoExtractionError("unterminated `bin` string literal in gpu.binary op")


def _target_attr_before(mlir_text: str, bin_at: int) -> str:
    """The ``#rocdl.target<...>`` an embedded object was compiled for.

    Diagnostic only: it names the objects in the failure message when they turn
    out to differ, so the reader learns *which* targets disagree instead of only
    that two blobs did. Read backwards from the ``bin`` attribute to the
    enclosing ``gpu.object<`` header rather than parsing MLIR.
    """
    open_at = mlir_text.rfind(_OBJECT_MARKER, 0, bin_at)
    if open_at < 0:
        return "<unknown target>"
    head = mlir_text[open_at + len(_OBJECT_MARKER) : bin_at]
    # The target attribute is everything up to the object's `kernels = ` field.
    return head.split(", kernels", 1)[0].strip().rstrip(",") or "<unknown target>"


def extract_bin_attr(mlir_text: str, where: str) -> bytes:
    """The code object embedded in `mlir_text`, across every attached target.

    One ``bin = "..."`` per ``gpu.object``, and a ``gpu.binary`` carries one
    object per attached target (see the module docstring). They must all decode
    to the same bytes; the single distinct blob is returned.
    """
    blobs: list[bytes] = []
    targets: list[str] = []

    at = mlir_text.find(_BIN_MARKER)
    if at < 0:
        raise HsacoExtractionError(
            f'{where}: no `bin = "..."` attribute; the gpu-module-to-binary pass '
            "did not embed a code object in this dump"
        )
    while at >= 0:
        body, end = decode_mlir_string(mlir_text, at + len(_BIN_MARKER))
        blobs.append(body)
        targets.append(_target_attr_before(mlir_text, at))
        at = mlir_text.find(_BIN_MARKER, end)

    distinct = {blob: target for blob, target in zip(blobs, targets)}
    if len(distinct) > 1:
        described = "\n".join(
            f"  {len(blob):>8} bytes  {sha256(blob)[:16]}  {target}"
            for blob, target in distinct.items()
        )
        raise HsacoExtractionError(
            f"{where}: the {len(blobs)} embedded objects are not identical:\n"
            f"{described}\n"
            "FlyDSL attaches several targets to one gpu.binary, and until now they "
            "have produced the same code object. Distinct objects mean the targets "
            "are no longer interchangeable, so which one to ship is a decision this "
            "generator must not make on its own: select explicitly (MLIR's "
            "#gpu.select_object) and record the choice in the manifest."
        )
    return blobs[0]


def hsaco_from_dump(dump_dir: Path) -> tuple[bytes, Path]:
    """The code object produced under `dump_dir`, with the dump file it came from."""
    candidates = sorted(dump_dir.glob(f"*/*{_BINARY_PASS_SUFFIX}"))
    if not candidates:
        raise HsacoExtractionError(
            f"{dump_dir}: no '*{_BINARY_PASS_SUFFIX}' dump. Either FLYDSL_DUMP_IR "
            "was not set for this compile, or the pass was renamed."
        )
    if len(candidates) > 1:
        names = ", ".join(str(p.relative_to(dump_dir)) for p in candidates)
        raise HsacoExtractionError(
            f"{dump_dir}: expected one compiled kernel per dump directory, found "
            f"{len(candidates)}: {names}. Give each instance its own dump dir."
        )
    dump = candidates[0]
    # surrogateescape: the dump is text with arbitrary bytes pasted through it,
    # so a strict UTF-8 decode rejects valid dumps.
    text = dump.read_text(encoding="utf-8", errors="surrogateescape")
    blob = extract_bin_attr(text, str(dump))
    if blob[: len(_ELF_MAGIC)] != _ELF_MAGIC:
        raise HsacoExtractionError(
            f"{dump}: extracted {len(blob)} bytes starting {blob[:4]!r}, which is "
            "not an ELF. The embedded-binary contract has changed."
        )
    return blob, dump
