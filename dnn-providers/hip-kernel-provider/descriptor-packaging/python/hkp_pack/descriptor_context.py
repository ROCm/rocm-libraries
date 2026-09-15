"""Read-only descriptor identities, origins and consumer bindings within one root."""

from __future__ import annotations

import dataclasses
import glob
import json
import os

from . import agreement
from .errors import HkpPackError

_DESCRIPTOR_TYPES = ("kdp", "ukd", "kmd", "ued", "umd", "udd", "uhd")


class DescriptorContextError(HkpPackError):
    """An unreadable or ambiguous descriptor tree, not an agreement finding."""


def _type_token(path: str) -> str | None:
    """The `<type>` of a `<name>.<type>.json` descriptor filename."""
    parts = os.path.basename(path).split(".")
    return parts[-2] if len(parts) >= 3 else None


@dataclasses.dataclass
class Document:
    """One descriptor file, with the type its filename declares."""

    path: str
    doc: dict
    dtype: str


class Index:
    """Every descriptor under the caller's root, indexed by its declared id.

    Duplicate ids are refused before resolution: picking either document would
    bind a consumer to a schema it may not actually use. No ancestor is searched.
    """

    def __init__(self, root: str):
        self.root = root
        self.documents: list[Document] = []
        self.by_id: dict[str, Document] = {}
        collisions: dict[str, list[str]] = {}
        for path in sorted(glob.glob(f"{glob.escape(root)}/**/*.json", recursive=True)):
            dtype = _type_token(path)
            if dtype not in _DESCRIPTOR_TYPES:
                continue
            try:
                with open(path) as fh:
                    doc = json.load(fh)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DescriptorContextError(
                    f"cannot read descriptor {path}: {exc}"
                ) from exc
            if not isinstance(doc, dict):
                raise DescriptorContextError(f"descriptor {path} is not a JSON object")
            document = Document(path, doc, dtype)
            self.documents.append(document)
            ident = doc.get("id")
            if not isinstance(ident, str) or not ident:
                continue
            if ident in self.by_id:
                collisions.setdefault(ident, [self.by_id[ident].path]).append(path)
            else:
                self.by_id[ident] = document
        if collisions:
            detail = "; ".join(
                f"id {ident!r} is claimed by {', '.join(sorted(paths))}"
                for ident, paths in sorted(collisions.items())
            )
            raise DescriptorContextError(
                f"ambiguous descriptor ids under {root}: {detail}. A reference to "
                f"one of these resolves to two different documents, so nothing here "
                f"can say which schema the bundle is actually wired to."
            )

    def of_type(self, dtype: str) -> list[Document]:
        return [d for d in self.documents if d.dtype == dtype]

    def schemas(self) -> dict:
        """id -> KMD document, for the existing declaration validator."""
        return {d.doc["id"]: d.doc for d in self.of_type("kmd") if d.doc.get("id")}

    def follow(self, document: Document, key: str, hop: str) -> Document:
        """Resolve a declared id reference, naming the hop when it dangles."""
        ref = document.doc.get(key)
        if not isinstance(ref, str) or not ref:
            raise DescriptorContextError(
                f"{os.path.basename(document.path)}: '{key}' is not an id reference "
                f"({ref!r}). The {hop} hop cannot be walked, and binding to a "
                f"same-stem sibling instead would gate a schema nothing wires this "
                f"bundle to."
            )
        target = self.by_id.get(ref)
        if target is None:
            raise DescriptorContextError(
                f"{os.path.basename(document.path)}: unresolved '{key}' reference -- "
                f"id {ref!r} matches no descriptor under {self.root}."
            )
        return target


@dataclasses.dataclass
class Entry:
    """A UKD and its runtime library origin and effective arch coverage.

    Standalone origins are the UKD's directory; inline origins are the KDP's.
    Only inline entries inherit the enclosing KDP's declaration. Empty arch is a
    wildcard; None is disjoint coverage, retained for structural-only readers.
    """

    ukd: dict
    origin_dir: str
    arch: list | None
    inline: bool = True


@dataclasses.dataclass
class Bundle:
    """One KDP resolved through the id chain to the schema that governs it."""

    kdp_path: str
    kdp_doc: dict
    engine: dict
    kmd: dict
    entries: list[Entry]


def _coverage(kdp_doc: dict, ukd: dict) -> list | None:
    """Effective coverage: [] is a wildcard, None an empty intersection."""
    kdp_arch = list(kdp_doc.get("arch") or [])
    ukd_arch = list(ukd.get("arch") or [])
    if not kdp_arch:
        return ukd_arch
    if not ukd_arch:
        return kdp_arch
    shared = [a for a in kdp_arch if a in ukd_arch]
    return shared or None


def resolve_entries(index: Index, kdp: Document) -> list[Entry]:
    """Resolve UKD references and origins without requiring an engine/schema.

    Structural desk checking needs only this hop, including entries whose arch
    coverage is disjoint. Full bundle resolution omits those unshipped entries.
    """
    entries = []
    for item in kdp.doc.get("kernelDescriptors") or []:
        if isinstance(item, str):
            target = index.by_id.get(item)
            if target is None:
                raise DescriptorContextError(
                    f"{os.path.basename(kdp.path)}: unresolved "
                    f"'kernelDescriptors' reference -- id {item!r} matches no "
                    f"descriptor under {index.root}."
                )
            ukd, origin, inline = target.doc, os.path.dirname(target.path), False
        elif isinstance(item, dict):
            ukd, origin, inline = item, os.path.dirname(kdp.path), True
        else:
            raise DescriptorContextError(
                f"{os.path.basename(kdp.path)}: a kernelDescriptors entry is "
                f"neither an inline object nor an id reference."
            )
        entries.append(Entry(ukd, origin, _coverage(kdp.doc, ukd), inline))
    return entries


def resolve_bundles(index: Index) -> list[Bundle]:
    """Every KDP under the root, walked to its engine and schema by id."""
    bundles = []
    for kdp in index.of_type("kdp"):
        engine = index.follow(kdp, "engine", "KDP -> engine (UED)")
        kmd = index.follow(engine, "metadata", "UED -> metadata (KMD)")
        if kmd.dtype != "kmd":
            raise DescriptorContextError(
                f"{os.path.basename(engine.path)}: 'metadata' resolves to "
                f"{os.path.basename(kmd.path)}, which is a "
                f"{kmd.dtype!r} document, not a KMD."
            )
        entries = [e for e in resolve_entries(index, kdp) if e.arch is not None]
        bundles.append(Bundle(kdp.path, kdp.doc, engine.doc, kmd.doc, entries))
    return bundles


def declarations(bundles: list[Bundle], schemas: dict) -> dict:
    """(engine_id, kmd_id, ukd_id) -> the declaration for that consumer.

    Agreement owns inheritance and whole-UKD overrides. This aggregate is not a
    substitute for requiring a selected entry's own resolved declaration.
    """
    found = {}
    for bundle in bundles:
        for entry in bundle.entries:
            enclosing = bundle.kdp_doc if entry.inline else None
            if agreement.resolved_contract(entry.ukd, enclosing) is None:
                continue
            declaration = agreement.select_declaration(
                entry.ukd, bundle.engine, bundle.kmd, schemas, enclosing
            )
            found[(bundle.engine["id"], bundle.kmd["id"], entry.ukd["id"])] = (
                declaration
            )
    return found


def consumer_records(bundles: list[Bundle], schemas: dict, arch: str) -> dict:
    """UKD id -> canonical records over EVERY consumer inside the caller root.

    Shared standalone UKDs carry all KDP/engine/KMD bindings, not just the selected
    bundle's. Agreement remains the sole record-construction/canonical authority.
    """
    collected: dict = {}
    for bundle in bundles:
        header = {k: v for k, v in bundle.kdp_doc.items() if k != "kernelDescriptors"}
        header["arch"] = [arch]
        for entry in bundle.entries:
            if entry.arch and arch not in entry.arch:
                continue
            enclosing = bundle.kdp_doc if entry.inline else None
            if agreement.resolved_contract(entry.ukd, enclosing) is None:
                continue
            declaration = agreement.select_declaration(
                entry.ukd, bundle.engine, bundle.kmd, schemas, enclosing
            )
            collected.setdefault(entry.ukd["id"], []).append(
                agreement.consumer_record(
                    entry.ukd, bundle.engine, bundle.kmd, header, arch, declaration
                )
            )
    return {k: agreement.canonical_records(v) for k, v in collected.items()}
