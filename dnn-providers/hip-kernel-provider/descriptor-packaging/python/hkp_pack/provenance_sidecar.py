"""Per-UKD provenance, shipped beside a packed descriptor rather than inside it.

No runtime code reads a packed UKD's `provenance`, yet on a large pack it is most
of the bytes every loading process parses at startup. The packer moves it into
one gzipped sidecar per packed descriptor file, `{stem}.provenance.json.gz` beside
`{stem}.kdp.json` or `{stem}.ukd.json`, keyed by UKD id and bound to each UKD by
its `kernel_source.sha256`. The loader opens only `<name>.<type>.json`, so it
never reads the sidecar. A KDP's own header `provenance` stays inline.

`attach` is the reader's half: it puts each entry back onto its UKD after checking
the binding, so every check downstream reads the document the packer digested.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

from .errors import HkpPackError

SUFFIX = ".provenance.json.gz"
_KDP_SUFFIX = ".kdp.json"
_UKD_SUFFIX = ".ukd.json"


def sidecar_name(descriptor_name: str) -> str:
    """`{stem}.provenance.json.gz` for `{stem}.kdp.json` or `{stem}.ukd.json`."""
    for suffix in (_KDP_SUFFIX, _UKD_SUFFIX):
        if descriptor_name.endswith(suffix):
            return descriptor_name[: -len(suffix)] + SUFFIX
    raise HkpPackError(
        f"{descriptor_name}: only a '{_KDP_SUFFIX}' or '{_UKD_SUFFIX}' file has a "
        "provenance sidecar"
    )


def sidecar_path(descriptor_path) -> Path:
    path = Path(descriptor_path)
    return path.with_name(sidecar_name(path.name))


def _subjects(descriptor_name: str, doc: dict) -> tuple[str | None, list[dict]]:
    """The sidecar's `kdp_id` and the UKDs whose provenance it holds: a KDP's
    inline entries, or a standalone UKD itself."""
    if descriptor_name.endswith(_KDP_SUFFIX):
        entries = doc.get("kernelDescriptors") or []
        return doc.get("id"), [e for e in entries if isinstance(e, dict)]
    return None, [doc]


def _kernel_source_sha256(ukd: dict):
    source = ukd.get("kernel_source")
    return source.get("sha256") if isinstance(source, dict) else None


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

    Returns the sidecar's filename and its bytes; the caller writes both files.
    """
    kdp_id, ukds = _subjects(descriptor_name, doc)
    entries = {}
    for ukd in ukds:
        ident = ukd.get("id")
        if ident in entries:
            raise HkpPackError(
                f"{descriptor_name}: UKD id {ident!r} appears twice, so its "
                "provenance sidecar cannot key both"
            )
        entries[ident] = {
            "kernel_source_sha256": _kernel_source_sha256(ukd),
            "provenance": ukd.pop("provenance", {}),
        }
    sidecar = {"kdp_id": kdp_id, "entries": entries}
    return sidecar_name(descriptor_name), encode(sidecar)


def load_sidecar(path) -> dict:
    path = Path(path)
    try:
        sidecar = json.loads(gzip.decompress(path.read_bytes()))
    except FileNotFoundError as exc:
        raise HkpPackError(f"provenance sidecar {path} does not exist") from exc
    except (OSError, EOFError, ValueError) as exc:
        raise HkpPackError(f"cannot read provenance sidecar {path}: {exc}") from exc
    if not isinstance(sidecar, dict) or not isinstance(sidecar.get("entries"), dict):
        raise HkpPackError(f"provenance sidecar {path} has no 'entries' object")
    return sidecar


def lookup(sidecar: dict, ukd: dict, where: str) -> dict:
    """The provenance `sidecar` holds for `ukd`, once its sha256 binding holds."""
    ident = ukd.get("id")
    entry = sidecar["entries"].get(ident)
    if not isinstance(entry, dict):
        raise HkpPackError(f"{where}: UKD {ident!r} has no entry in the sidecar")
    expected = _kernel_source_sha256(ukd)
    if entry.get("kernel_source_sha256") != expected:
        raise HkpPackError(
            f"{where}: the entry for UKD {ident!r} is bound to kernel_source.sha256 "
            f"{entry.get('kernel_source_sha256')!r}, but the UKD names {expected!r}. "
            "The descriptor and its sidecar come from different packs."
        )
    provenance = entry.get("provenance")
    if not isinstance(provenance, dict):
        raise HkpPackError(f"{where}: the entry for UKD {ident!r} has no provenance")
    return provenance


def attach(descriptor_path, doc: dict) -> dict:
    """Put each packed UKD's sidecar provenance back onto it, in place.

    A descriptor with no sidecar is left alone unless it holds a `kpack` UKD,
    which only the packer writes and so always ships one. A UKD carrying inline
    `provenance` beside a sidecar is refused rather than merged.
    """
    path = Path(descriptor_path)
    kdp_id, ukds = _subjects(path.name, doc)
    side = sidecar_path(path)
    if not side.is_file():
        for ukd in ukds:
            if (ukd.get("kernel_source") or {}).get("kind") == "kpack":
                raise HkpPackError(
                    f"{path}: packed UKD {ukd.get('id')!r} has no provenance "
                    f"sidecar; expected {side.name} beside it"
                )
        return doc
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
