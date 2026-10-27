# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Edit the grouped-convolution dispatcher's instance lists.

A list (``dispatch/grouped_convolution_instances/<arch>_<direction>.json``)
holds one JSON record per instance, in priority order: the fields of its
:class:`KernelIdentity` that leave the family's baseline (format:
:class:`dispatch.grouped_convolution.InstanceList`). Benchmark tables and logs
name kernels by their lossless labels instead; this tool moves between the two.

  labels LIST                 print the instances' labels, in priority order
  add LIST [LABEL...]         append the labelled kernels (--at N inserts them
                              at position N); --from FILE reads one label per
                              line, '#' starting a comment
  new LIST [LABEL...]         write a new list of the labelled kernels; arch
                              and direction come from the labels, and the
                              LLVM flavor and wave size they share become the
                              list's defaults
  format LIST                 rewrite LIST canonically (sparse records, one
                              per line); --check only reports whether it is

Examples:
  python instance_list.py labels ../../../dispatch/grouped_convolution_instances/gfx950_fwd.json
  python instance_list.py add gfx950_fwd.json \\
      fwd-implicit_gemm-fp16-2d-t64x64x64-w2x2-a32x32x16-v8x8x8-mem-cshuffle-grp-wave64-gfx950-llvm22
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tempfile
from pathlib import Path

from dispatch.grouped_convolution import (
    InstanceList,
    format_instance_list,
    load_instance_list,
)
from kernels.conv.kernel_identity import KernelIdentity

# Fields a new list hoists into its defaults when every instance shares them.
_HOISTED = ("llvm_flavor", "wave_size")


def _labels(args: argparse.Namespace) -> list[str]:
    labels = list(args.label)
    if args.from_file:
        for line in Path(args.from_file).read_text(encoding="utf-8").splitlines():
            name = line.split("#", 1)[0].strip()
            if name:
                labels.append(name)
    if not labels:
        raise SystemExit("error: no labels given (LABEL... or --from FILE)")
    return labels


def _identities(labels: list[str]) -> list[KernelIdentity]:
    out = []
    for name in labels:
        try:
            out.append(KernelIdentity.from_label(name))
        except ValueError as e:
            raise SystemExit(f"error: {e}") from None
    return out


def _write(path: Path, lst: InstanceList) -> None:
    """Validate ``lst`` before ``path`` ever sees it.

    The dispatcher (and this tool) load instance lists at import/labels time,
    so a broken file left on disk breaks every caller, including a later
    ``format`` meant to fix it. Write to a temp file in the same directory,
    validate that (running every check the dispatcher's loader runs), and
    only then replace ``path`` -- on a validation failure the original file
    is untouched.
    """
    text = format_instance_list(lst)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        load_instance_list(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"{path}: {len(lst.identities)} instances")


def _cmd_labels(args: argparse.Namespace) -> None:
    for identity in load_instance_list(args.list).identities:
        print(identity.label())


def _cmd_add(args: argparse.Namespace) -> None:
    lst = load_instance_list(args.list)
    current = list(lst.identities)
    added = _identities(_labels(args))
    for identity in added:
        if identity in current:
            raise SystemExit(f"error: {identity.label()} is already listed")
        current.append(identity)
    if args.at is not None:
        current = current[: -len(added)]
        current[args.at : args.at] = added
    _write(args.list, dataclasses.replace(lst, identities=tuple(current)))


def _cmd_new(args: argparse.Namespace) -> None:
    if args.list.exists():
        raise SystemExit(f"error: {args.list} exists; use add")
    identities = _identities(_labels(args))
    arch = {i.arch for i in identities}
    direction = {i.direction.removeprefix("direct_") for i in identities}
    if len(arch) != 1 or len(direction) != 1:
        raise SystemExit(
            f"error: one list holds one arch and direction, the labels name "
            f"{sorted(arch)} and {sorted(direction)}"
        )
    defaults = {}
    for f in _HOISTED:
        values = {getattr(i, f) for i in identities}
        if len(values) == 1:
            defaults[f] = values.pop()
    lst = InstanceList(
        arch=arch.pop(),
        direction=direction.pop(),
        identities=tuple(identities),
        defaults=defaults,
    )
    _write(args.list, lst)


def _cmd_format(args: argparse.Namespace) -> None:
    lst = load_instance_list(args.list)
    text = format_instance_list(lst)
    if args.list.read_text(encoding="utf-8") == text:
        return
    if args.check:
        raise SystemExit(f"{args.list} is not canonically formatted")
    _write(args.list, lst)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("labels", help="print the instances' labels")
    p.add_argument("list", type=Path)
    p.set_defaults(run=_cmd_labels)

    for name, run, helptext in (
        ("add", _cmd_add, "append labelled kernels to a list"),
        ("new", _cmd_new, "write a new list of labelled kernels"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("list", type=Path)
        p.add_argument("label", nargs="*")
        p.add_argument("--from", dest="from_file", help="file of labels")
        if name == "add":
            p.add_argument("--at", type=int, help="insert at this position")
        p.set_defaults(run=run)

    p = sub.add_parser("format", help="rewrite a list canonically")
    p.add_argument("list", type=Path)
    p.add_argument("--check", action="store_true", help="only report")
    p.set_defaults(run=_cmd_format)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except ValueError as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    sys.exit(main())
