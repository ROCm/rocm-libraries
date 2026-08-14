import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import HkpPackError

KDP_TYPE = "kdp"
UKD_TYPE = "ukd"
_GENERIC_TYPES = {"kmd", "ued", "umd", "udd", "uhd"}
_ALL_TYPES = {KDP_TYPE, UKD_TYPE} | _GENERIC_TYPES

_SCALAR_TYPES = (str, int, float, bool)


def type_from_filename(path):
    """Descriptor type token from a `<name>.<type>.json` filename.

    The type is the second-to-last dot-separated segment of the file name.
    Returns None if the name has too few segments to carry a type token.
    """
    parts = Path(path).name.split(".")
    if len(parts) < 3:
        return None
    return parts[-2]


@dataclass
class Descriptor:
    """A parsed generic descriptor or KDP loaded from a flat-folder JSON file.

    UKDs are never standalone Descriptors: they live inline in a KDP's
    kernelDescriptors vector. A descriptor's type is derived from its filename
    (`<name>.<type>.json`), never from a field in the document.
    """

    path: Path
    doc: dict

    @property
    def type(self):
        return type_from_filename(self.path)

    @property
    def id(self):
        return self.doc.get("id")


@dataclass
class FlatInput:
    root: Path
    descriptors: list = field(default_factory=list)

    def by_type(self, dtype):
        return [d for d in self.descriptors if d.type == dtype]

    def kdps(self):
        return self.by_type(KDP_TYPE)

    def generics(self):
        return [d for d in self.descriptors if d.type in _GENERIC_TYPES]

    def generic_by_id(self):
        return {d.id: d for d in self.generics()}


def _read_json(path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HkpPackError(f"cannot read descriptor {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise HkpPackError(f"malformed descriptor JSON in {path.name}: {exc}") from exc


def _require(doc, keys, where):
    for key in keys:
        if key not in doc:
            raise HkpPackError(f"{where} missing required field '{key}'")


def arch_matches(kdp_doc, arch):
    """A KDP matches an arch iff its arch list is empty (wildcard) or lists it."""
    archs = kdp_doc.get("arch")
    if not archs:
        return True
    return arch in archs


def validate_hip_build(build, where):
    """A hip UKD's build block is functional; reject anything unusable.

    Rejects when build is absent/not an object or defines is present but is not
    a flat map of macro-name -> scalar. flags, when present, must be a string
    list. The failure substring is stable ('invalid build').
    """
    if not isinstance(build, dict):
        raise HkpPackError(f"{where} has invalid build (not an object)")
    defines = build.get("defines")
    if defines is not None:
        if not isinstance(defines, dict):
            raise HkpPackError(f"{where} has invalid build (defines not a map)")
        for name, val in defines.items():
            if not isinstance(name, str) or not isinstance(val, _SCALAR_TYPES):
                raise HkpPackError(
                    f"{where} has invalid build (defines must map strings to scalars)"
                )
    flags = build.get("flags")
    if flags is not None:
        if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
            raise HkpPackError(f"{where} has invalid build (flags not a string list)")


def _validate_inline_ukd(ukd, kdp_path):
    where = f"UKD '{ukd.get('id', '?')}' in {kdp_path.name}"
    if not isinstance(ukd, dict):
        raise HkpPackError(f"inline UKD in {kdp_path.name} is not a JSON object")
    _require(ukd, ["id", "name", "kernel_source", "metadata", "priority"], where)
    ks = ukd["kernel_source"]
    if not isinstance(ks, dict) or "kind" not in ks:
        raise HkpPackError(f"{where} kernel_source missing 'kind'")
    kind = ks["kind"]
    if kind == "hip":
        _require(ks, ["source", "entry"], where)
        if "build" not in ks:
            raise HkpPackError(f"{where} has invalid build (absent)")
        validate_hip_build(ks["build"], where)
    elif kind == "hsaco":
        _require(ks, ["file", "symbol"], where)
    elif kind == "kpack":
        _require(ks, ["library", "toc_key", "symbol", "sha256"], where)
    else:
        raise HkpPackError(
            f"{where} kernel_source has unsupported kind '{kind}' "
            "(expected 'hip', 'hsaco', or 'kpack')"
        )


def _validate_kdp(desc):
    doc = desc.doc
    path = desc.path
    where = f"KDP {path.name}"
    _require(
        doc,
        ["name", "arch", "matchers", "engine", "dispatch", "kernelDescriptors"],
        where,
    )
    arch = doc["arch"]
    if not isinstance(arch, list) or not all(isinstance(a, str) and a for a in arch):
        raise HkpPackError(
            f"{where} 'arch' must be a list of strings (empty = wildcard)"
        )
    kds = doc["kernelDescriptors"]
    if not isinstance(kds, list) or not kds:
        raise HkpPackError(f"{where} 'kernelDescriptors' must be a non-empty list")
    for ukd in kds:
        _validate_inline_ukd(ukd, path)


def _validate_shape(desc):
    doc = desc.doc
    path = desc.path
    if not isinstance(doc, dict):
        raise HkpPackError(f"descriptor {path.name} is not a JSON object")
    _require(doc, ["id"], f"descriptor {path.name}")
    dtype = desc.type
    if dtype not in _ALL_TYPES:
        raise HkpPackError(
            f"descriptor {path.name} has unknown type token '{dtype}' "
            "(expected <name>.<type>.json)"
        )
    if dtype == UKD_TYPE:
        raise HkpPackError(
            f"descriptor {path.name} is a standalone UKD; UKDs must be inline in a KDP"
        )
    if dtype == KDP_TYPE:
        _validate_kdp(desc)


def load_flat_input(root):
    """Load and structurally validate every *.json descriptor in a flat folder.

    root holds the authored source folder: KDP files (with inline hip UKDs) and
    the by-Id generic files (UMD/UED/UDD/KMD/UHD), plus the HIP sources the UKDs
    name. Each descriptor's type is derived from its `<name>.<type>.json`
    filename. Raises HkpPackError on any malformed / missing-field / unknown-type
    / dangling-reference descriptor.
    """
    root = Path(root)
    if not root.is_dir():
        raise HkpPackError(f"input folder does not exist: {root}")

    descriptors = []
    for jp in sorted(root.glob("*.json")):
        desc = Descriptor(path=jp, doc=_read_json(jp))
        _validate_shape(desc)
        descriptors.append(desc)

    flat = FlatInput(root=root, descriptors=descriptors)
    _validate_references(flat)
    return flat


def _validate_references(flat):
    ids = {d.id for d in flat.descriptors}
    for kdp in flat.kdps():
        doc = kdp.doc
        refs = list(doc.get("matchers", []))
        refs += [doc.get("engine"), doc.get("dispatch")]
        for ref in refs:
            if ref is not None and ref not in ids:
                raise HkpPackError(
                    f"KDP {kdp.path.name} references unknown descriptor Id '{ref}'"
                )
    for ued in flat.by_type("ued"):
        for ref in (ued.doc.get("heuristic"), ued.doc.get("metadata")):
            if ref is not None and ref not in ids:
                raise HkpPackError(
                    f"UED {ued.path.name} references unknown descriptor Id '{ref}'"
                )


def reachable_generic_ids(flat, surviving_kdps):
    """Ids of the generics reachable from a set of surviving KDPs.

    Walks KDP -> {matchers, engine, dispatch} and UED -> {heuristic, metadata}
    transitively. A generic survives pruning iff its Id is in this set.
    """
    by_id = flat.generic_by_id()
    reachable = set()
    pending = []
    for kdp in surviving_kdps:
        doc = kdp.doc
        pending += list(doc.get("matchers", []))
        pending += [doc.get("engine"), doc.get("dispatch")]
    while pending:
        rid = pending.pop()
        if rid is None or rid in reachable or rid not in by_id:
            continue
        reachable.add(rid)
        gdesc = by_id[rid]
        if gdesc.type == "ued":
            pending += [gdesc.doc.get("heuristic"), gdesc.doc.get("metadata")]
    return reachable
