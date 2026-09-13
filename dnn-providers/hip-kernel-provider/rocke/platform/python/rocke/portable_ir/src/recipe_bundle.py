# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# recipe_bundle.py -- productization plumbing for the record path: a compact,
# dependency-free CBOR codec and a recipe BUNDLE format.
#
# A bundle (schema "rocke.bundle/v1") packs many concrete (or rolled) recipes,
# keyed by (arch, cache_key), into one CBOR blob the C runtime can mmap/parse and
# serve by key (no CPython, no per-recipe JSON files). CBOR is ~2x smaller than
# the JSON and decodes into the same DOM the recipe VM already consumes.
#
# The "concrete record" path: record every production kernel's emitted IR into a
# concrete recipe (universal, byte-identical, no rolling needed) and bundle them.
# Rolled recipes can be bundled the same way (one entry covers a whole family).
#
#   python3 -m rocke.portable_ir.src.recipe_bundle --demo
import struct
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

from rocke.portable_ir.src import abi as _abi


def _provenance() -> Tuple[str, str]:
    """Best-effort (engine version, build id) of the engine on this machine.

    Soft on purpose, but only for the one expected reason: no shared library is
    built here. These strings are never compared by any reader -- only
    `min_reader` decides anything -- so building a bundle, a pure data
    operation, must not come to require a compiled engine merely to record a
    debugging aid. The catch stays narrow so that a real fault in the bindings
    surfaces instead of quietly stamping an empty provenance, which is what a
    bare `except Exception` did here at first."""
    try:
        from rocke.portable_ir.src import online

        return online.provenance()
    except (ImportError, OSError, RuntimeError):
        return ("", "")


BUNDLE_SCHEMA = "rocke.bundle/v1"


# --------------------------------------------------------------------------
# CBOR (RFC 8949) codec -- the subset recipes use: uint, negint, float64,
# text string, array, map(str keys), bool, null. No external dependency.
# --------------------------------------------------------------------------
def _enc_head(major: int, n: int) -> bytes:
    if n < 24:
        return bytes([(major << 5) | n])
    if n < 0x100:
        return bytes([(major << 5) | 24, n])
    if n < 0x10000:
        return bytes([(major << 5) | 25]) + n.to_bytes(2, "big")
    if n < 0x100000000:
        return bytes([(major << 5) | 26]) + n.to_bytes(4, "big")
    return bytes([(major << 5) | 27]) + n.to_bytes(8, "big")


def cbor_encode(o: Any) -> bytes:
    if o is None:
        return b"\xf6"
    if o is True:
        return b"\xf5"
    if o is False:
        return b"\xf4"
    if isinstance(o, bool):  # (already handled above; defensive)
        return b"\xf5" if o else b"\xf4"
    if isinstance(o, int):
        return _enc_head(0, o) if o >= 0 else _enc_head(1, -1 - o)
    if isinstance(o, float):
        return b"\xfb" + struct.pack(">d", o)
    if isinstance(o, str):
        b = o.encode("utf-8")
        return _enc_head(3, len(b)) + b
    if isinstance(o, (list, tuple)):
        return _enc_head(4, len(o)) + b"".join(cbor_encode(x) for x in o)
    if isinstance(o, dict):
        out = _enc_head(5, len(o))
        for k, v in o.items():
            out += cbor_encode(k) + cbor_encode(v)
        return out
    raise TypeError(f"cbor_encode: unsupported {type(o).__name__}")


def _dec_head(buf: bytes, i: int) -> Tuple[int, int, int]:
    """-> (major, argument, next_index)."""
    ib = buf[i]
    major, info = ib >> 5, ib & 0x1F
    i += 1
    if info < 24:
        return major, info, i
    if info == 24:
        return major, buf[i], i + 1
    if info == 25:
        return major, int.from_bytes(buf[i : i + 2], "big"), i + 2
    if info == 26:
        return major, int.from_bytes(buf[i : i + 4], "big"), i + 4
    if info == 27:
        return major, int.from_bytes(buf[i : i + 8], "big"), i + 8
    raise ValueError(f"cbor: bad additional info {info}")


def _dec(buf: bytes, i: int) -> Tuple[Any, int]:
    ib = buf[i]
    if ib == 0xF6:
        return None, i + 1
    if ib == 0xF5:
        return True, i + 1
    if ib == 0xF4:
        return False, i + 1
    if ib == 0xFB:
        return struct.unpack(">d", buf[i + 1 : i + 9])[0], i + 9
    major, arg, i = _dec_head(buf, i)
    if major == 0:
        return arg, i
    if major == 1:
        return -1 - arg, i
    if major == 3:
        return buf[i : i + arg].decode("utf-8"), i + arg
    if major == 4:
        out = []
        for _ in range(arg):
            v, i = _dec(buf, i)
            out.append(v)
        return out, i
    if major == 5:
        out = {}
        for _ in range(arg):
            k, i = _dec(buf, i)
            v, i = _dec(buf, i)
            out[k] = v
        return out, i
    raise ValueError(f"cbor: unsupported major {major}")


def cbor_decode(buf: bytes) -> Any:
    v, _ = _dec(buf, 0)
    return v


# --------------------------------------------------------------------------
# bundle format
# --------------------------------------------------------------------------
def build_bundle(
    entries: List[Dict[str, Any]], *, stamp_abi: bool = True
) -> Dict[str, Any]:
    """entries: list of {"key", "arch", "family"?, "recipe"}.

    The bundle is stamped with the oldest engine that can read it, derived from
    what its recipes actually use rather than from this generator's own version
    -- see src/abi.py for why that distinction is the whole point. The container
    takes the strictest requirement among its entries, since a reader that
    cannot handle one recipe cannot be handed the bundle and left to discover
    that at lookup time.

    `stamp_abi=False` is for tests that need to compare bundle bytes against a
    fixture recorded before stamping existed."""
    bundle = {"schema": BUNDLE_SCHEMA, "entries": entries}
    if not stamp_abi:
        return bundle
    need = max(
        [1]
        + [
            _abi.recipe_min_reader(e["recipe"], strict=False)
            for e in entries
            if isinstance(e.get("recipe"), dict)
        ]
    )
    engine, build_id = _provenance()
    return _abi.stamp(bundle, min_reader=need, engine=engine, build_id=build_id)


def write_bundle(path: str, entries: List[Dict[str, Any]]) -> int:
    blob = cbor_encode(build_bundle(entries))
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


def read_bundle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return cbor_decode(f.read())


def bundle_lookup(
    bundle: Dict[str, Any], key: str, arch: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    for e in bundle.get("entries", []):
        if e.get("key") == key and (arch is None or e.get("arch") == arch):
            return e.get("recipe")
    return None


# --------------------------------------------------------------------------
# concrete-record driver: record production kernels -> concrete recipes -> bundle
# --------------------------------------------------------------------------
def record_concrete_bundle(cases: List[Tuple[Any, str]]) -> List[Dict[str, Any]]:
    """cases: list of (build_callable, arch). Records each kernel's emitted IR
    into a concrete recipe (universal, byte-identical), keyed by kernel name."""
    from rocke.portable_ir.src.recording_builder import record_kernel

    entries = []
    for build, arch in cases:
        kernel, recipe = record_kernel(build)
        entries.append(
            {
                "key": kernel.name,
                "arch": arch,
                "family": recipe.get("kernel_name_fmt", kernel.name),
                "recipe": recipe,
            }
        )
    return entries


def _demo() -> int:
    import json

    from rocke.portable_ir.examples import export_mha

    cases = [
        (lambda D=D: export_mha.build("fp16", D, 2048, 1, 32, 1), "gfx950")
        for D in (64, 128, 256)
    ]
    entries = record_concrete_bundle(cases)
    blob = cbor_encode(build_bundle(entries))
    json_size = sum(len(json.dumps(e["recipe"])) for e in entries)
    print(f"bundled {len(entries)} concrete recipes:")
    for e in entries:
        print(f"  {e['key']}  ({e['arch']})")
    print(
        f"CBOR bundle: {len(blob)} bytes   vs JSON recipes total: {json_size} bytes "
        f"({json_size / len(blob):.2f}x)"
    )
    assert cbor_decode(blob) == build_bundle(entries), "CBOR round-trip mismatch"
    print("CBOR round-trip: OK")
    return 0


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="recipe CBOR codec + bundle plumbing")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("demo")

    enc = sub.add_parser("encode", help="JSON recipe -> CBOR")
    enc.add_argument("json_in")
    enc.add_argument("cbor_out")

    bnd = sub.add_parser(
        "bundle", help="pack recipes (json[:key[:arch]]) -> CBOR bundle"
    )
    bnd.add_argument("cbor_out")
    bnd.add_argument(
        "recipes",
        nargs="+",
        help="path/to/recipe.json[:key[:arch]] (key defaults to kernel_name_fmt)",
    )

    rd = sub.add_parser(
        "record-demo",
        help="record a production set CONCRETELY -> one CBOR bundle keyed by name",
    )
    rd.add_argument("cbor_out")
    rd.add_argument("--arch", default="gfx950")

    args = ap.parse_args(argv)

    if args.cmd == "record-demo":
        from rocke.portable_ir.examples import mini_attn, recipe_multi_result

        cases = [
            (lambda: mini_attn.build_mini_attn(0, "f32"), args.arch),
            (lambda: mini_attn.build_mini_attn(1, "f32"), args.arch),
            (lambda: recipe_multi_result.build_multi_result("i32"), args.arch),
        ]
        entries = record_concrete_bundle(cases)
        n = write_bundle(args.cbor_out, entries)
        print(f"wrote {args.cbor_out}: {n} bytes, {len(entries)} concrete recipes")
        for e in entries:
            print(f"  {e['key']}  ({e['arch']})")
        return 0

    if args.cmd == "encode":
        with open(args.json_in) as f:
            recipe = json.load(f)
        with open(args.cbor_out, "wb") as f:
            f.write(cbor_encode(recipe))
        return 0

    if args.cmd == "bundle":
        entries = []
        for spec in args.recipes:
            parts = spec.split(":")
            path = parts[0]
            with open(path) as f:
                recipe = json.load(f)
            key = (
                parts[1]
                if len(parts) > 1 and parts[1]
                else recipe.get("kernel_name_fmt", path)
            )
            arch = parts[2] if len(parts) > 2 and parts[2] else "gfx950"
            entries.append(
                {
                    "key": key,
                    "arch": arch,
                    "family": recipe.get("kernel_name_fmt", key),
                    "recipe": recipe,
                }
            )
        n = write_bundle(args.cbor_out, entries)
        print(
            f"wrote {args.cbor_out}: {n} bytes, {len(entries)} entries "
            f"({', '.join(e['key'] for e in entries)})"
        )
        return 0

    return _demo()


if __name__ == "__main__":
    raise SystemExit(_main())
