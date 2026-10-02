#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Addition-only splice of a scratch descriptor render into a live engine directory.

The generator mints fresh UUIDs on every run, so copying a regenerated descriptor
directory over a live one replaces every retained identity, which extend.md forbids.
This tool applies only the additions:

    splice_additions.py --scratch "$GENERATED/descriptors/<producer>/<bundle>" \\
                        --live "$PROVIDER/src/engines/kernel_ingestor_engine/descriptors/<producer>/<bundle>" \\
                        [--report splice_report.json] [--check]

Pairing and checks, all before anything is written, and all run by `--check` too:

  * Documents pair by file name; kernel entries of a paired KDP pair by `name`.
    Each pair contributes `scratch id -> live id` to one UUID map.
  * Every live file and every live kernel must have a scratch twin. A scratch
    render that drops one is a removal, not an addition.
  * Every paired document, and every retained kernel entry, must equal its live
    twin after the UUID map is applied. Any other difference is a changed retained
    object and is refused.
  * Kernel entries whose name is new are appended, in scratch order, after the live
    entries of their KDP; their own new ids are kept and their references to retained
    objects are remapped. A scratch-only file (a new pack's KDP, for example) is
    written whole, remapped the same way.
  * A KDP is rewritten only if re-serialising it with the generator's serializer
    reproduces its current text, so retained bytes cannot change.

Exit codes: 0 spliced (or nothing to add), 1 refused (nothing written),
2 invalid invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: The KDP key holding the kernel entries.
KERNELS = "kernelDescriptors"


class SpliceRefused(Exception):
    """The scratch render is not an addition-only change of the live directory."""


def dump(obj: dict) -> str:
    """The generator's own serializer (codegen/generator.py `_dump`)."""
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def _read(path: Path) -> tuple[str, str]:
    """Text with LF line endings, plus the newline the file uses on disk.

    A file mixing CRLF and LF has no single newline to write back, so a rewrite
    would change retained bytes; it is refused."""
    raw = path.read_bytes().decode("utf-8")
    crlf = raw.count("\r\n")
    if crlf and crlf != raw.count("\n"):
        raise SpliceRefused(
            f"{path}: mixes CRLF and LF line endings, so a rewrite cannot keep its "
            "bytes. Normalise it to one line ending in a separate commit first."
        )
    return raw.replace("\r\n", "\n"), "\r\n" if crlf else "\n"


def _load_dir(root: Path) -> dict[str, dict]:
    if not root.is_dir():
        raise SpliceRefused(f"{root} is not a directory")
    docs = {}
    for path in sorted(root.glob("*.json")):
        try:
            docs[path.name] = json.loads(_read(path)[0])
        except json.JSONDecodeError as e:
            raise SpliceRefused(f"{path}: not JSON: {e}") from e
    if not docs:
        raise SpliceRefused(f"{root} holds no descriptor JSON")
    return docs


def _is_kdp(name: str) -> bool:
    return name.endswith(".kdp.json")


def _kernels_by_name(doc: dict, where: str) -> dict[str, dict]:
    by_name = {}
    for kernel in doc.get(KERNELS, []):
        name = kernel.get("name")
        if name in by_name:
            raise SpliceRefused(f"{where}: kernel name {name!r} appears twice")
        by_name[name] = kernel
    return by_name


def remap(value, ids: dict[str, str]):
    """`value` with every string equal to a mapped scratch UUID replaced."""
    if isinstance(value, dict):
        return {k: remap(v, ids) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v, ids) for v in value]
    if isinstance(value, str):
        return ids.get(value, value)
    return value


def first_difference(live, scratch, path: str = "") -> str:
    """A readable path to the first place two JSON values differ."""
    if isinstance(live, dict) and isinstance(scratch, dict):
        for key in list(live) + [k for k in scratch if k not in live]:
            if key not in scratch:
                return f"{path}.{key}: missing from scratch"
            if key not in live:
                return f"{path}.{key}: only in scratch"
            if live[key] != scratch[key]:
                return first_difference(live[key], scratch[key], f"{path}.{key}")
    if isinstance(live, list) and isinstance(scratch, list):
        if len(live) != len(scratch):
            return f"{path}: {len(live)} live items, {len(scratch)} scratch items"
        for i, (a, b) in enumerate(zip(live, scratch)):
            if a != b:
                return first_difference(a, b, f"{path}[{i}]")
    return f"{path or '<document>'}: live {json.dumps(live)} != scratch {json.dumps(scratch)}"


def plan(live_dir: Path, scratch_dir: Path) -> dict:
    """Check the scratch render against the live directory and return the splice.

    Raises SpliceRefused on the first conflict.
    """
    live = _load_dir(live_dir)
    scratch = _load_dir(scratch_dir)

    missing = sorted(set(live) - set(scratch))
    if missing:
        raise SpliceRefused(f"live files missing from scratch: {missing}")

    # One UUID map, scratch -> live, from every paired document and kernel.
    ids: dict[str, str] = {}

    def pair(scratch_id, live_id, what: str) -> None:
        if scratch_id is None or live_id is None:
            return
        if ids.get(scratch_id, live_id) != live_id:
            raise SpliceRefused(
                f"{what}: scratch id {scratch_id} pairs with two live ids"
            )
        ids[scratch_id] = live_id

    for name in sorted(live):
        pair(scratch[name].get("id"), live[name].get("id"), name)
        if _is_kdp(name):
            live_kernels = _kernels_by_name(live[name], f"live {name}")
            scratch_kernels = _kernels_by_name(scratch[name], f"scratch {name}")
            dropped = sorted(set(live_kernels) - set(scratch_kernels))
            if dropped:
                raise SpliceRefused(
                    f"{name}: {len(dropped)} live kernels missing from scratch, "
                    f"e.g. {dropped[:5]}"
                )
            for kname, lk in live_kernels.items():
                pair(scratch_kernels[kname].get("id"), lk.get("id"), f"{name}:{kname}")

    live_ids = {v for v in ids.values()}
    spliced: dict[str, dict] = {}
    added: list[dict] = []
    for name in sorted(live):
        mapped = remap(scratch[name], ids)
        if not _is_kdp(name):
            if mapped != live[name]:
                raise SpliceRefused(
                    f"{name} differs from live beyond UUIDs: "
                    f"{first_difference(live[name], mapped)}"
                )
            continue
        live_top = {k: v for k, v in live[name].items() if k != KERNELS}
        mapped_top = {k: v for k, v in mapped.items() if k != KERNELS}
        if mapped_top != live_top:
            raise SpliceRefused(
                f"{name} top-level fields differ from live beyond UUIDs: "
                f"{first_difference(live_top, mapped_top)}"
            )
        live_kernels = _kernels_by_name(live[name], f"live {name}")
        new = []
        for kernel in mapped.get(KERNELS, []):
            twin = live_kernels.get(kernel["name"])
            if twin is None:
                if kernel.get("id") in live_ids:
                    raise SpliceRefused(
                        f"{name}:{kernel['name']}: new kernel reuses a live id"
                    )
                new.append(kernel)
            elif kernel != twin:
                raise SpliceRefused(
                    f"{name}: retained kernel {kernel['name']} differs from live beyond "
                    f"UUIDs: {first_difference(twin, kernel)}"
                )
        if new:
            spliced[name] = {**live[name], KERNELS: live[name][KERNELS] + new}
            added += [{"file": name, "name": k["name"], "id": k["id"]} for k in new]

    new_files = {}
    for name in sorted(set(scratch) - set(live)):
        new_files[name] = remap(scratch[name], ids)
        if _is_kdp(name):
            added += [
                {"file": name, "name": k["name"], "id": k["id"]}
                for k in new_files[name].get(KERNELS, [])
            ]

    # Every refusal happens here, before anything is written, so `--check` and a
    # real run refuse the same inputs. A rewritten KDP must keep its retained bytes:
    # only a file the generator's serializer reproduces can be rewritten.
    writes = []
    for name, doc in spliced.items():
        path = live_dir / name
        text, newline = _read(path)
        if dump(json.loads(text)) != text:
            raise SpliceRefused(
                f"{name}: re-serialising the live file does not reproduce its text, so "
                "a rewrite would change retained bytes. Format it with the generator's "
                "serializer (json.dumps indent=2) in a separate commit first."
            )
        writes.append((path, dump(doc).replace("\n", newline).encode("utf-8")))
    for name, doc in new_files.items():
        writes.append((live_dir / name, dump(doc).encode("utf-8")))

    return {
        "live_dir": str(live_dir),
        "scratch_dir": str(scratch_dir),
        "spliced": spliced,
        "new_files": new_files,
        "writes": writes,
        "added_kernels": added,
        "retained_kernels": sum(
            len(live[n].get(KERNELS, [])) for n in live if _is_kdp(n)
        ),
        "uuid_map_scratch_to_live": ids,
    }


def apply(result: dict) -> None:
    """Write the splice `plan` checked; `plan` has already refused anything that
    would change retained bytes."""
    for path, data in result["writes"]:
        path.write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scratch",
        type=Path,
        required=True,
        help="the scratch render's descriptor directory for this bundle",
    )
    parser.add_argument(
        "--live",
        type=Path,
        required=True,
        help="the live engine's descriptor directory",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="write a JSON report (counts, added names/ids, UUID map)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify and report only; write nothing to --live",
    )
    args = parser.parse_args(argv)

    try:
        result = plan(args.live, args.scratch)
        if not args.check:
            apply(result)
    except SpliceRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        print("Nothing was written.", file=sys.stderr)
        return 1

    added = result["added_kernels"]
    report = {
        "mode": "check" if args.check else "spliced",
        "retained_kernels": result["retained_kernels"],
        "added_kernels": len(added),
        "total_kernels": result["retained_kernels"] + len(added),
        "added": added,
        "new_files": sorted(result["new_files"]),
        "uuid_map_scratch_to_live": result["uuid_map_scratch_to_live"],
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    verb = "would append" if args.check else "appended"
    print(
        f"retained {report['retained_kernels']} kernels, {verb} {len(added)}, "
        f"total {report['total_kernels']}; new files: {report['new_files'] or 'none'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
