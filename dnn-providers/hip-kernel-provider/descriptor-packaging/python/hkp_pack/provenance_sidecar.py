"""Per-UKD provenance, shipped beside a packed descriptor rather than inside it.

No runtime code reads a packed UKD's `provenance`, yet on a large pack it is most
of the bytes every loading process parses at startup. The packer moves it into
one gzipped sidecar per packed descriptor file, named after it: `foo.kdp.json`
ships `foo.kdp.provenance.json.gz` and `foo.ukd.json` ships
`foo.ukd.provenance.json.gz`. The loader opens only `<name>.<type>.json`, so it
never reads the sidecar. A KDP's own header `provenance` stays inline.

The sidecar is the version 1 document

    {"version": "1.0", "kdp_id": <KDP id, or null for a standalone UKD>,
     "entries": {<UKD id>: {"ukd_sha256": <hex>, "provenance": {...}}}}

`ukd_sha256` digests the UKD as written (see `ukd_sha256`), so it binds every
kernel_source kind, even `embedded_source`, which carries no sha256 of its own.

This module is the only reader. `attach` puts each entry back onto its UKD after
checking the binding, so every check downstream reads the document the packer
digested. A descriptor is packed when the packer's empty `PACKED_MARKER` file
sits in its own directory; any other is authored. Sidecars installed apart from
their descriptors are found under a provenance root mirroring the descriptor
root (see `sidecar_path`).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import zlib
from pathlib import Path

from .errors import HkpPackError

SUFFIX = ".provenance.json.gz"
# Not a dotfile, so no install or artifact glob skips it.
PACKED_MARKER = "hkp-packed.marker"
_KDP_SUFFIX = ".kdp.json"
_UKD_SUFFIX = ".ukd.json"

VERSION = "1.0"
_READABLE_MAJOR = 1
_READABLE_MINOR = 0

# A ceiling on one sidecar's decompressed size, far above any real pack: a sidecar
# holds a few kilobytes per UKD, and a gzip body can inflate a thousandfold, so an
# uncapped read lets a small corrupt or hostile file exhaust memory.
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024


def sidecar_name(descriptor_name: str) -> str:
    """`foo.kdp.provenance.json.gz` for `foo.kdp.json`, and likewise for a UKD."""
    for suffix in (_KDP_SUFFIX, _UKD_SUFFIX):
        if descriptor_name.endswith(suffix):
            return descriptor_name[: -len(".json")] + SUFFIX
    raise HkpPackError(
        f"{descriptor_name}: only a '{_KDP_SUFFIX}' or '{_UKD_SUFFIX}' file has a "
        "provenance sidecar"
    )


def sidecar_path(descriptor_path, provenance_root=None, descriptor_root=None) -> Path:
    """`descriptor_path`'s sidecar: beside it, or at the same relative path under
    `provenance_root` as the descriptor has under `descriptor_root`."""
    path = Path(descriptor_path)
    name = sidecar_name(path.name)
    if provenance_root is None:
        return path.with_name(name)
    if descriptor_root is None:
        raise HkpPackError(
            f"{path}: a provenance root ({provenance_root}) mirrors a descriptor "
            "root, and none was named"
        )
    parent = Path(os.path.abspath(path.parent))
    try:
        rel = parent.relative_to(os.path.abspath(descriptor_root))
    except ValueError as exc:
        raise HkpPackError(
            f"{path} is not under the descriptor root {descriptor_root} that the "
            f"provenance root {provenance_root} mirrors"
        ) from exc
    return Path(provenance_root) / rel / name


def _subjects(descriptor_name: str, doc) -> tuple[str | None, list[dict]]:
    """The sidecar's `kdp_id` and the UKDs whose provenance it holds: a KDP's
    inline entries, or a standalone UKD itself."""
    if not isinstance(doc, dict):
        raise HkpPackError(f"{descriptor_name}: the descriptor is not a JSON object")
    if descriptor_name.endswith(_KDP_SUFFIX):
        entries = doc.get("kernelDescriptors", [])
        if not isinstance(entries, list):
            raise HkpPackError(
                f"{descriptor_name}: 'kernelDescriptors' is "
                f"{type(entries).__name__}, not an array"
            )
        return doc.get("id"), [e for e in entries if isinstance(e, dict)]
    return None, [doc]


def _kind(ukd: dict):
    source = ukd.get("kernel_source")
    return source.get("kind") if isinstance(source, dict) else None


def ukd_sha256(ukd: dict) -> str:
    """The digest binding a sidecar entry to its UKD, `provenance` excluded.

    Keys are sorted, so the packer digesting the dict it writes and a reader
    digesting the dict it parsed agree whatever their key order.
    """
    body = {k: v for k, v in ukd.items() if k != "provenance"}
    try:
        text = json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        raise HkpPackError(f"UKD {ukd.get('id')!r} cannot be digested: {exc}") from exc


def encode(sidecar: dict) -> bytes:
    """Compact, key-sorted JSON, gzipped with no mtime and no filename, so one
    source tree packs to the same bytes every time."""
    raw = json.dumps(sidecar, separators=(",", ":"), sort_keys=True).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(raw)
    return buf.getvalue()


def detach(descriptor_name: str, doc: dict) -> tuple[str, bytes]:
    """Move each packed UKD's `provenance` out of `doc`, in place.

    Call it on the document exactly as it is then written: each entry's
    `ukd_sha256` digests the UKD as it stands once its provenance is gone.
    Returns the sidecar's filename and its bytes; the caller writes both files.
    """
    kdp_id, ukds = _subjects(descriptor_name, doc)
    entries = {}
    for ukd in ukds:
        ident = ukd.get("id")
        if ident is not None and not isinstance(ident, str):
            raise HkpPackError(
                f"{descriptor_name}: UKD id {ident!r} is not a string, so the "
                "provenance sidecar cannot key it"
            )
        if ident in entries:
            raise HkpPackError(
                f"{descriptor_name}: UKD id {ident!r} appears twice, so its "
                "provenance sidecar cannot key both"
            )
        provenance = ukd.pop("provenance", {})
        entries[ident] = {"ukd_sha256": ukd_sha256(ukd), "provenance": provenance}
    sidecar = {"version": VERSION, "kdp_id": kdp_id, "entries": entries}
    return sidecar_name(descriptor_name), encode(sidecar)


def _inflate(data: bytes, path: Path) -> bytes:
    """One gzip member, refused past MAX_DECOMPRESSED_BYTES or with trailing bytes."""
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = inflater.decompress(data, MAX_DECOMPRESSED_BYTES + 1)
    if len(raw) > MAX_DECOMPRESSED_BYTES:
        raise HkpPackError(
            f"provenance sidecar {path} inflates past "
            f"{MAX_DECOMPRESSED_BYTES} bytes; no packed sidecar is that large"
        )
    if not inflater.eof:
        raise HkpPackError(f"provenance sidecar {path} is truncated")
    if inflater.unused_data:
        raise HkpPackError(f"provenance sidecar {path} has bytes after its gzip body")
    return raw


def _require_version(version, path: Path) -> None:
    if not isinstance(version, str):
        raise HkpPackError(
            f"provenance sidecar {path} carries no 'version' string, so its "
            "format cannot be known; repack it"
        )
    major, dot, minor = version.partition(".")
    if not (dot and major.isdigit() and minor.isdigit()):
        raise HkpPackError(
            f"provenance sidecar {path} has version {version!r}, not major.minor"
        )
    if int(major) != _READABLE_MAJOR or int(minor) > _READABLE_MINOR:
        raise HkpPackError(
            f"provenance sidecar {path} is version {version}; this reader reads "
            f"major version {_READABLE_MAJOR} only, at minor {_READABLE_MINOR} "
            "or earlier"
        )


def load_sidecar(path) -> dict:
    """The validated sidecar document at `path`; an unreadable or malformed file,
    or one of a version this reader does not read, is an HkpPackError."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise HkpPackError(f"provenance sidecar {path} does not exist") from exc
    except OSError as exc:
        raise HkpPackError(f"cannot read provenance sidecar {path}: {exc}") from exc
    try:
        sidecar = json.loads(_inflate(data, path))
    except HkpPackError:
        raise
    except (zlib.error, ValueError, RecursionError) as exc:
        raise HkpPackError(f"cannot read provenance sidecar {path}: {exc}") from exc
    if not isinstance(sidecar, dict):
        raise HkpPackError(f"provenance sidecar {path} is not a JSON object")
    _require_version(sidecar.get("version"), path)
    kdp_id = sidecar.get("kdp_id")
    if kdp_id is not None and not isinstance(kdp_id, str):
        raise HkpPackError(f"provenance sidecar {path} has a non-string 'kdp_id'")
    entries = sidecar.get("entries")
    if not isinstance(entries, dict):
        raise HkpPackError(f"provenance sidecar {path} has no 'entries' object")
    for ident, entry in entries.items():
        if not isinstance(entry, dict):
            raise HkpPackError(
                f"provenance sidecar {path}: the entry for UKD {ident!r} is not an "
                "object"
            )
        if not isinstance(entry.get("ukd_sha256"), str):
            raise HkpPackError(
                f"provenance sidecar {path}: the entry for UKD {ident!r} has no "
                "'ukd_sha256' string"
            )
        if not isinstance(entry.get("provenance"), dict):
            raise HkpPackError(
                f"provenance sidecar {path}: the entry for UKD {ident!r} has no "
                "'provenance' object"
            )
    return sidecar


def lookup(sidecar: dict, ukd: dict, where: str) -> dict:
    """The provenance a loaded `sidecar` holds for `ukd`, once its binding holds."""
    ident = ukd.get("id")
    entry = sidecar["entries"].get(ident) if isinstance(ident, str) else None
    if entry is None:
        raise HkpPackError(f"{where}: UKD {ident!r} has no entry in the sidecar")
    expected = ukd_sha256(ukd)
    if entry["ukd_sha256"] != expected:
        raise HkpPackError(
            f"{where}: the entry for UKD {ident!r} is bound to ukd_sha256 "
            f"{entry['ukd_sha256']}, but the UKD digests to {expected}. "
            "The descriptor and its sidecar come from different packs."
        )
    return entry["provenance"]


def is_packed(directory) -> bool:
    return (Path(directory) / PACKED_MARKER).is_file()


def attach(
    descriptor_path, doc: dict, *, provenance_root=None, descriptor_root=None
) -> dict:
    """Put each packed UKD's sidecar provenance back onto it, in place.

    A packed descriptor (see `is_packed`) must have its sidecar, beside it or
    under `provenance_root`. Any other is authored and left alone, sidecar unread,
    unless it holds a `kpack` UKD or a `provenance_root` is given (see
    `_require_authored`). A UKD carrying inline `provenance` beside a sidecar is
    refused rather than merged.
    """
    path = Path(descriptor_path)
    kdp_id, ukds = _subjects(path.name, doc)
    if not is_packed(path.parent):
        _require_authored(path, ukds, provenance_root)
        return doc
    side = sidecar_path(path, provenance_root, descriptor_root)
    if not side.is_file():
        hint = (
            ""
            if provenance_root is not None
            else " For an installed tree, pass --provenance-root: its sidecars "
            "install apart from its descriptors."
        )
        raise HkpPackError(
            f"{path}: packed descriptor has no provenance sidecar; expected "
            f"{side}.{hint}"
        )
    sidecar = load_sidecar(side)
    where = str(side)
    if sidecar.get("kdp_id") != kdp_id:
        raise HkpPackError(
            f"{where}: names KDP {sidecar.get('kdp_id')!r}, but {path.name} is "
            f"{kdp_id!r}"
        )
    for ukd in ukds:
        if "provenance" in ukd:
            raise HkpPackError(
                f"{path}: UKD {ukd.get('id')!r} carries inline provenance beside "
                f"{side.name}; a packed UKD carries none"
            )
        ukd["provenance"] = lookup(sidecar, ukd, where)
    return doc


def _require_authored(path: Path, ukds: list[dict], provenance_root) -> None:
    """Refuse an authored read that the descriptor or its caller contradicts."""
    if provenance_root is not None:
        raise HkpPackError(
            f"{path}: usage error: --provenance-root ({provenance_root}) relocates "
            f"a packed tree's sidecars, but {path.parent} holds no {PACKED_MARKER}, "
            "so its descriptors are authored"
        )
    kpack = [ukd.get("id") for ukd in ukds if _kind(ukd) == "kpack"]
    if kpack:
        raise HkpPackError(
            f"{path}: the descriptor looks packed (it holds kpack UKD(s) {kpack}, "
            f"which only the packer writes), but its directory holds no "
            f"{PACKED_MARKER}"
        )
