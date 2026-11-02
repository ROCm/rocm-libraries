#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Re-stamp the shipped Origami Boost models for the current Origami sources.

An Origami Boost model (ORIBOOST, .orr) carries the hash of the Origami sources it was trained
against, and a build whose sources differ refuses it. When the sources change but the model's
features do not (the "shipped models load in this build" test of origami-tests says which), the
model is still valid and only its stamp needs updating. This tool rewrites the HEAD section's
origami_source_hash and origami_commit (and records the previous hash and file digest in
restamped_from_origami_source_hash / restamped_from_sha256); every other section (FEAT, RNGE,
TREE, MANI) is copied byte for byte. The source hash is the rule of shared/origami/CMakeLists.txt
(ORIGAMI_SOURCE_HASH).

    origami_boost_restamp.py                 # re-stamp every data/nn/boost/<arch>/*.orr in place
    origami_boost_restamp.py --check         # exit 1 if any model's stamp is out of date
    origami_boost_restamp.py --source-hash   # print the current source hash
    origami_boost_restamp.py MODEL.orr ... [--origami-root DIR] [--out-dir DIR]

Exit status: 0 success (with --check: every model current), 1 --check found a stale model, 2 error.
Needs only the Python standard library (and git for origami_commit, else "unknown").
Re-stamping never changes what a model computes; a model whose features changed must be retrained
(OrigamiML, outside this repository) instead.
"""

import argparse
import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path

MAGIC = b"ORIBOOST"
FORMAT_MAJOR = 1
REQUIRED_SECTIONS = ("HEAD", "FEAT", "RNGE", "TREE", "MANI")
T_STR, T_F64, T_I64 = 1, 2, 3

# shared/origami/CMakeLists.txt: file(GLOB_RECURSE ...) of these patterns, then
# list(FILTER ... EXCLUDE REGEX "^include/origami/nn/") and list(SORT).
STAMP_PATTERNS = (("include/origami", (".hpp",)),
                  ("src/origami", (".cpp", ".hpp")),
                  ("src/simulator", (".cpp", ".hpp")))
STAMP_EXCLUDE = "include/origami/nn/"
STAMP_KEYS = ("origami_source_hash", "origami_commit", "restamped_from_origami_source_hash",
              "restamped_from_sha256")

ORIGAMI_ROOT = Path(__file__).resolve().parent.parent


class RestampError(RuntimeError):
    pass


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def source_files(root):
    out = []
    for sub, exts in STAMP_PATTERNS:
        base = root / sub
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                rel = (Path(dirpath) / name).relative_to(root).as_posix()
                if name.endswith(exts) and not rel.startswith(STAMP_EXCLUDE):
                    out.append(rel)
    return sorted(set(out), key=lambda s: s.encode())


def source_hash(root):
    """sha256 over `relpath \\t sha256(file) \\n` lines, as CMake computes ORIGAMI_SOURCE_HASH."""
    files = source_files(root)
    if not files:
        raise RestampError(f"no Origami sources under {root}")
    lines = "".join(f"{rel}\t{sha256_hex((root / rel).read_bytes())}\n" for rel in files)
    return "sha256:" + sha256_hex(lines.encode()), len(files)


def source_commit(root):
    """ORIGAMI_SOURCE_COMMIT: the last commit touching the stamped sources (provenance only)."""
    try:
        out = subprocess.run(
            ["git", "--no-optional-locks", "log", "-1", "--format=%H", "--", "include/origami",
             "src/origami", "src/simulator", ":(exclude)include/origami/nn"],
            cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out if out and all(c in "0123456789abcdef" for c in out) else "unknown"


class Model:
    """An .orr file as its header fields and its sections, raw, in file order."""

    def __init__(self, data):
        if len(data) < 24 + 32 or data[:8] != MAGIC:
            raise RestampError("not an ORIBOOST file (bad magic)")
        self.major, self.minor, self.reserved, body_len = struct.unpack("<HHIQ", data[8:24])
        if self.major != FORMAT_MAJOR:
            raise RestampError(f"format {self.major}.{self.minor}; this tool reads "
                               f"{FORMAT_MAJOR}.x")
        if 24 + body_len + 32 != len(data):
            raise RestampError("length mismatch (truncated file?)")
        body = data[24:24 + body_len]
        if hashlib.sha256(body).digest() != data[24 + body_len:]:
            raise RestampError("checksum mismatch")
        self.sections = []
        at = 0
        while at < len(body):
            if at + 12 > len(body):
                raise RestampError("truncated section header")
            tag = body[at:at + 4].decode("ascii", "replace")
            (n,) = struct.unpack("<Q", body[at + 4:at + 12])
            if at + 12 + n > len(body):
                raise RestampError(f"truncated section {tag}")
            self.sections.append((tag, body[at + 12:at + 12 + n]))
            at += 12 + n
        tags = [t for t, _ in self.sections]
        for t in REQUIRED_SECTIONS:
            if tags.count(t) != 1:
                raise RestampError(f"section {t} must appear exactly once")
        self.head = decode_head(self.section("HEAD"))
        if encode_head(self.head) != self.section("HEAD"):
            raise RestampError("HEAD does not re-encode to the same bytes")

    def section(self, tag):
        return next(p for t, p in self.sections if t == tag)

    def head_str(self, key):
        for k, typ, v in self.head:
            if k == key:
                return v if typ == T_STR else str(v)
        return ""

    def encode(self):
        body = b"".join(t.encode("ascii") + struct.pack("<Q", len(p)) + p
                        for t, p in self.sections)
        pre = MAGIC + struct.pack("<HHIQ", self.major, self.minor, self.reserved, len(body))
        return pre + body + hashlib.sha256(body).digest()


def decode_head(payload):
    at = 0

    def take(n):
        nonlocal at
        if at + n > len(payload):
            raise RestampError("truncated HEAD")
        v = payload[at:at + n]
        at += n
        return v

    def string():
        (n,) = struct.unpack("<I", take(4))
        return take(n).decode("utf-8")

    entries = []
    (count,) = struct.unpack("<I", take(4))
    for _ in range(count):
        key = string()
        (typ,) = struct.unpack("<B", take(1))
        if typ == T_STR:
            value = string()
        elif typ == T_F64:
            (value,) = struct.unpack("<d", take(8))
        elif typ == T_I64:
            (value,) = struct.unpack("<q", take(8))
        else:
            raise RestampError(f"HEAD[{key}] has unknown type {typ}")
        entries.append((key, typ, value))
    if at != len(payload):
        raise RestampError("trailing bytes in HEAD")
    return entries


def encode_head(entries):
    def string(s):
        e = s.encode("utf-8")
        return struct.pack("<I", len(e)) + e

    out = [struct.pack("<I", len(entries))]
    for key, typ, value in entries:
        out.append(string(key) + struct.pack("<B", typ))
        if typ == T_STR:
            out.append(string(value))
        elif typ == T_F64:
            out.append(struct.pack("<d", value))
        else:
            out.append(struct.pack("<q", value))
    return b"".join(out)


def restamp(data, new_hash, new_commit):
    """The re-stamped file. Only HEAD changes; the other sections are checked byte-identical."""
    m = Model(data)
    old_hash = m.head_str("origami_source_hash")
    update = {"origami_source_hash": new_hash, "origami_commit": new_commit,
              "restamped_from_origami_source_hash": old_hash,
              "restamped_from_sha256": sha256_hex(data)}
    head = []
    for key, typ, value in m.head:
        if key in update:
            typ, value = T_STR, update.pop(key)
        head.append((key, typ, value))
    head += [(k, T_STR, update[k]) for k in STAMP_KEYS if k in update]
    m.sections = [(t, encode_head(head) if t == "HEAD" else p) for t, p in m.sections]
    out = m.encode()

    back, before = Model(out), Model(data)
    for (t0, p0), (t1, p1) in zip(before.sections, back.sections):
        if t0 != t1 or (t0 != "HEAD" and p0 != p1):
            raise RestampError(f"re-stamp changed section {t0}")
    old = {k: (t, v) for k, t, v in before.head if k not in STAMP_KEYS}
    new = {k: (t, v) for k, t, v in back.head if k not in STAMP_KEYS}
    if old != new:
        raise RestampError("re-stamp changed a HEAD entry other than the stamps")
    return out


def default_models(root):
    return sorted((root / "data" / "nn" / "boost").glob("*/*.orr"))


def write_atomic(path, data):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", type=Path,
                    help="model files (default: every data/nn/boost/<arch>/*.orr of the root)")
    ap.add_argument("--origami-root", type=Path, default=ORIGAMI_ROOT,
                    help="shared/origami whose sources the stamp is computed from "
                         "(default: this tool's tree)")
    ap.add_argument("--check", action="store_true",
                    help="only report; exit 1 if a model's stamp is out of date")
    ap.add_argument("--out-dir", type=Path,
                    help="write the re-stamped files here instead of in place")
    ap.add_argument("--commit", help="origami_commit to record (default: git log of the sources)")
    ap.add_argument("--source-hash", action="store_true",
                    help="print the current Origami source hash and exit")
    a = ap.parse_args(argv)

    try:
        root = a.origami_root.resolve()
        new_hash, n_files = source_hash(root)
        if a.source_hash:
            print(f"{new_hash} ({n_files} files under {root})")
            return 0
        models = a.models or default_models(root)
        if not models:
            raise RestampError(f"no models under {root / 'data/nn/boost'}")
        new_commit = a.commit or source_commit(root)
        print(f"Origami sources: {new_hash} ({n_files} files, commit {new_commit})")
        stale = 0
        for path in models:
            data = path.read_bytes()
            try:
                m = Model(data)
            except RestampError as exc:
                raise RestampError(f"{path}: {exc}") from exc
            old_hash = m.head_str("origami_source_hash")
            label = f"{path} ({m.head_str('arch')} {m.head_str('dtype')})"
            if old_hash == new_hash:
                print(f"  {label}: current")
                if a.out_dir:
                    write_atomic(a.out_dir / path.name, data)
                continue
            stale += 1
            if a.check:
                print(f"  {label}: STALE, stamped {old_hash}")
                continue
            out = restamp(data, new_hash, new_commit)
            dest = (a.out_dir / path.name) if a.out_dir else path
            write_atomic(dest, out)
            print(f"  {label}: re-stamped {old_hash} -> {new_hash}; FEAT, RNGE, TREE, MANI "
                  f"byte-identical; wrote {dest} (sha256 {sha256_hex(out)})")
        if a.check and stale:
            print(f"{stale} model(s) need re-stamping (run without --check), or retraining if "
                  "their features changed (see the origami-tests message)")
            return 1
        return 0
    except (OSError, RestampError, UnicodeDecodeError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
