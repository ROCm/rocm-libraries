#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Inspect stopped-wave AMDGPU values in rocGDB.

Source this file from rocGDB, then use one of these commands::

    (gdb) source tools/rocke_debug.py
    (gdb) rocke decode $v40 --dtype f32
    (gdb) rocke print
    (gdb) rocke collect --scope wave --output acc.snapshot.json

Decoding, snapshot validation, and rendering are independent of rocGDB. The
adapter below only reads stopped-wave state, invokes the pure host library, and
writes output.
"""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Sequence
from typing import Any

from rocke.debug.logical_value_reconstruction import (
    logical_snapshot,
    unavailable_status_for_error,
)
from rocke.debug.logical_value_rendering import (
    decode_logical_value,
    decode_word,
    render_readable,
    unavailable_value,
)
from rocke.debug.register_value_decoding import DTYPES, FLOAT8_FORMATS
from rocke.debug.rocgdb_value_locations import (
    bind_debug_description,
    kernel_symbol,
    symbol_address,
)
from rocke.debug.stopped_wave_snapshot import collect_selected_wave, dump_snapshot
from rocke.core.debug_manifest import (
    DEBUG_DESCRIPTION_MAGIC,
    debug_description_symbol,
)

SCHEMA = "rocke-register-v1"
MANIFEST_SCHEMA = "rocke-debug-manifest/v1"


def decode_register(
    register: str,
    raw_words: Sequence[int],
    dtype: str,
    exec_mask: int | None = None,
    float8_format: str = "ocp",
) -> list[dict[str, Any]]:
    """Build stable per-lane records for one stopped-wave register."""
    records = []
    for lane, word in enumerate(raw_words):
        raw = int(word) & 0xFFFFFFFF
        records.append(
            {
                "schema": SCHEMA,
                "register": register,
                "lane": lane,
                "active": None if exec_mask is None else bool(exec_mask & (1 << lane)),
                "raw_bits": raw,
                "raw_hex": f"0x{raw:08x}",
                "dtype": dtype,
                "float8_format": float8_format if "8" in dtype else None,
                "elements": decode_word(raw, dtype, float8_format=float8_format),
            }
        )
    return records


def records_jsonl(records: Sequence[dict[str, Any]]) -> str:
    """Serialize records as strict, deterministic JSON Lines."""
    return "\n".join(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        for record in records
    )


def records_human(records: Sequence[dict[str, Any]]) -> str:
    """Render records as a compact table for an interactive rocGDB session."""
    lines = ["register lane active raw        dtype        values"]
    for record in records:
        active = (
            "?" if record["active"] is None else ("yes" if record["active"] else "no")
        )
        values = ", ".join(element["value_text"] for element in record["elements"])
        lines.append(
            f"{record['register']:<8} {record['lane']:>4} {active:<6} "
            f"{record['raw_hex']} {record['dtype']:<12} [{values}]"
        )
    return "\n".join(lines)


def load_manifest(path: str) -> dict[str, Any]:
    """Load and minimally validate a portable rocKE debug manifest."""
    try:
        with open(path, encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load debug manifest {path!r}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        actual = manifest.get("schema") if isinstance(manifest, dict) else None
        raise ValueError(
            f"unsupported debug manifest schema {actual!r}; "
            f"expected {MANIFEST_SCHEMA!r}"
        )
    values = manifest.get("values")
    if not isinstance(values, list):
        raise TypeError("debug manifest 'values' must be a list")
    if any(not isinstance(value, dict) for value in values):
        raise TypeError("every debug manifest value must be an object")
    return manifest


def manifest_value(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one uniquely named logical-value entry from ``manifest``."""
    matches = [
        value
        for value in manifest["values"]
        if value.get("logical", {}).get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"debug manifest must contain exactly one value named {name!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocke decode", add_help=False)
    parser.add_argument("expression")
    parser.add_argument("--dtype", required=True, choices=DTYPES)
    parser.add_argument("--format", choices=("human", "jsonl"), default="human")
    parser.add_argument("--float8-format", choices=FLOAT8_FORMATS, default="ocp")
    parser.add_argument("--lane", action="append", type=int)
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--exec", dest="exec_expression", default="$exec")
    parser.add_argument("--help", action="help")
    return parser


def _print_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocke print", add_help=False)
    parser.add_argument("name", nargs="*")
    parser.add_argument("--manifest")
    parser.add_argument("--format", choices=("human", "jsonl"), default="human")
    parser.add_argument("--float8-format", choices=FLOAT8_FORMATS, default="ocp")
    parser.add_argument("--exec", dest="exec_expression", default="$exec")
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--help", action="help")
    return parser


def _collect_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocke collect", add_help=False)
    parser.add_argument("name", nargs="*")
    parser.add_argument("--manifest")
    parser.add_argument("--scope", choices=("wave", "block"), default="wave")
    parser.add_argument("--output", required=True)
    parser.add_argument("--float8-format", choices=FLOAT8_FORMATS, default="ocp")
    parser.add_argument("--exec", dest="exec_expression", default="$exec")
    parser.add_argument("--help", action="help")
    return parser


try:
    import gdb  # type: ignore
except ModuleNotFoundError:
    gdb = None


if gdb is not None:

    def _current_kernel_and_pc() -> tuple[str, int]:
        pc = int(gdb.parse_and_eval("$pc"))
        kernel = kernel_symbol(gdb.execute(f"info symbol 0x{pc:x}", to_string=True))
        return kernel, pc

    def _embedded_debug_description(kernel: str) -> dict[str, Any]:
        symbol = debug_description_symbol(kernel)
        address = symbol_address(gdb.execute(f"info address {symbol}", to_string=True))
        inferior = gdb.selected_inferior()
        header = bytes(inferior.read_memory(address, 16))
        if header[:8] != DEBUG_DESCRIPTION_MAGIC:
            raise ValueError(f"invalid rocKE debug description header for {kernel!r}")
        size = int.from_bytes(header[8:16], byteorder="little")
        if not 0 < size <= 16 * 1024 * 1024:
            raise ValueError(f"invalid rocKE debug description size {size}")
        payload = bytes(inferior.read_memory(address + 16, size))
        try:
            description = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid embedded rocKE debug description") from error
        if description.get("kernel") != kernel:
            raise ValueError(
                f"embedded debug description is for {description.get('kernel')!r}, "
                f"not stopped kernel {kernel!r}"
            )
        return description

    def _manifest_and_names(
        manifest_path: str | None, names: Sequence[str]
    ) -> tuple[dict[str, Any], list[str]]:
        if manifest_path is not None:
            manifest = load_manifest(manifest_path)
            if names:
                return manifest, list(names)
            available = [value["logical"]["name"] for value in manifest["values"]]
            if len(available) != 1:
                raise ValueError(
                    "multiple debug values are available; choose one of: "
                    + ", ".join(sorted(available))
                )
            return manifest, available
        kernel, pc = _current_kernel_and_pc()
        return bind_debug_description(
            _embedded_debug_description(kernel),
            names,
            pc=pc,
            location_text=lambda name: gdb.execute(
                f"info address {name}", to_string=True
            ),
        )

    def _gdb_words(value: Any) -> list[int]:
        try:
            lower, upper = value.type.range()
        except (gdb.error, RuntimeError):
            return [int(value)]
        return [int(value[index]) for index in range(lower, upper + 1)]

    def _collect_current_wave(
        manifest: dict[str, Any],
        names: Sequence[str],
        *,
        exec_expression: str,
        float8_format: str,
    ):
        """Collect the selected stopped wave without choosing a presentation."""
        thread = gdb.selected_thread()
        if thread is None:
            raise ValueError("no stopped thread is selected")
        if hasattr(thread, "is_stopped") and not thread.is_stopped():
            raise ValueError("the selected thread is not stopped")
        frame = gdb.selected_frame()
        try:
            architecture = frame.architecture().name()
        except (AttributeError, gdb.error, RuntimeError):
            architecture = None
        try:
            kernel = frame.name()
        except (AttributeError, gdb.error, RuntimeError):
            kernel = None
        try:
            pc = int(gdb.parse_and_eval("$pc"))
        except (gdb.error, RuntimeError):
            pc = None
        try:
            exec_mask = int(gdb.parse_and_eval(exec_expression))
        except (gdb.error, RuntimeError):
            exec_mask = None
        stop_mode = "non-stop" if bool(gdb.parameter("non-stop")) else "all-stop"
        return collect_selected_wave(
            manifest,
            names,
            read_words=lambda expression: _gdb_words(gdb.parse_and_eval(expression)),
            thread_id=str(getattr(thread, "global_num", getattr(thread, "num", "?"))),
            pc=pc,
            exec_mask=exec_mask,
            architecture=architecture,
            kernel=kernel,
            stop_mode=stop_mode,
            float8_format=float8_format,
        )

    class RockePrefix(gdb.Command):
        """rocKE commands for stopped AMDGPU waves."""

        def __init__(self) -> None:
            super().__init__("rocke", gdb.COMMAND_USER, prefix=True)

    class RockeDecode(gdb.Command):
        """Decode a physical register: rocke decode EXPR --dtype DTYPE."""

        def __init__(self) -> None:
            super().__init__("rocke decode", gdb.COMMAND_DATA)

        def invoke(self, argument: str, from_tty: bool) -> None:
            del from_tty
            try:
                args = _argument_parser().parse_args(shlex.split(argument))
                words = _gdb_words(gdb.parse_and_eval(args.expression))
                try:
                    exec_mask = int(gdb.parse_and_eval(args.exec_expression))
                except (gdb.error, RuntimeError):
                    exec_mask = None
                records = decode_register(
                    args.expression,
                    words,
                    args.dtype,
                    exec_mask=exec_mask,
                    float8_format=args.float8_format,
                )
                if args.lane is not None:
                    selected = set(args.lane)
                    records = [
                        record for record in records if record["lane"] in selected
                    ]
                if args.active_only:
                    records = [record for record in records if record["active"] is True]
                rendered = (
                    records_jsonl(records)
                    if args.format == "jsonl"
                    else records_human(records)
                )
                if rendered:
                    gdb.write(rendered + "\n")
            except (ValueError, RuntimeError, gdb.error) as error:
                raise gdb.GdbError(str(error)) from error

    class RockePrint(gdb.Command):
        """Collect and render logical values: rocke print [NAME]."""

        def __init__(self) -> None:
            super().__init__("rocke print", gdb.COMMAND_DATA)

        def invoke(self, argument: str, from_tty: bool) -> None:
            del from_tty
            try:
                args = _print_argument_parser().parse_args(shlex.split(argument))
                manifest, names = _manifest_and_names(args.manifest, args.name)
                snapshot = _collect_current_wave(
                    manifest,
                    names,
                    exec_expression=args.exec_expression,
                    float8_format=args.float8_format,
                )
                records = logical_snapshot(snapshot.to_dict())["waves"][0]["values"]
                rendered = (
                    records_jsonl(records)
                    if args.format == "jsonl"
                    else render_readable(records, show_sources=args.show_sources)
                )
                gdb.write(rendered + "\n")
            except (TypeError, ValueError, RuntimeError, gdb.error) as error:
                raise gdb.GdbError(str(error)) from error

    class RockeCollect(gdb.Command):
        """Capture stopped-wave values: rocke collect [NAME] --output PATH."""

        def __init__(self) -> None:
            super().__init__("rocke collect", gdb.COMMAND_DATA)

        def invoke(self, argument: str, from_tty: bool) -> None:
            del from_tty
            try:
                args = _collect_argument_parser().parse_args(shlex.split(argument))
                if args.scope != "wave":
                    raise ValueError(
                        "the implementation spike supports --scope wave only"
                    )
                manifest, names = _manifest_and_names(args.manifest, args.name)
                snapshot = _collect_current_wave(
                    manifest,
                    names,
                    exec_expression=args.exec_expression,
                    float8_format=args.float8_format,
                )
                dump_snapshot(snapshot, args.output)
                gdb.write(
                    f"wrote {args.output} "
                    f"(complete={str(snapshot.capture['complete']).lower()})\n"
                )
                if not snapshot.capture["complete"]:
                    raise gdb.GdbError(
                        f"snapshot is incomplete; inspect {args.output} for details"
                    )
            except (OSError, TypeError, ValueError, RuntimeError, gdb.error) as error:
                raise gdb.GdbError(str(error)) from error

    RockePrefix()
    RockeDecode()
    RockePrint()
    RockeCollect()
