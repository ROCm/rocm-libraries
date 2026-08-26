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
    """A parsed descriptor loaded from a flat-folder JSON file.

    Holds a generic descriptor, a KDP, or a standalone UKD. A UKD may be
    authored either inline in a KDP's kernelDescriptors vector or as its own
    `<name>.ukd.json` file that a KDP references by Id. A descriptor's type is
    derived from its filename (`<name>.<type>.json`), never from a field in the
    document.
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

    def ukds(self):
        return self.by_type(UKD_TYPE)

    def ukd_by_id(self):
        return {d.id: d for d in self.ukds()}


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


def _arch_subset_ok(ukd_arch, kdp_arch):
    """A UKD's arch is admissible under a referencing KDP's arch.

    An empty list on either side is a wildcard: a wildcard KDP admits any UKD,
    and a wildcard UKD is admissible under any KDP. Two explicit lists require
    the UKD's arches to be a subset of the KDP's.
    """
    if not ukd_arch or not kdp_arch:
        return True
    return set(ukd_arch) <= set(kdp_arch)


def kdp_survives(kdp_doc, flat, arch):
    """Whether a KDP ships in a given arch's shard.

    A KDP ships iff it matches the arch and at least one of its UKD entries
    (an inline dict or a standalone resolved by id) also applies to that arch.
    A KDP whose UKDs all filter out for this arch is dropped from the shard.
    """
    if not arch_matches(kdp_doc, arch):
        return False
    ukd_by_id = flat.ukd_by_id()
    for entry in kdp_doc.get("kernelDescriptors", []):
        if isinstance(entry, str):
            sdesc = ukd_by_id.get(entry)
            if sdesc is not None and arch_matches(sdesc.doc, arch):
                return True
        elif isinstance(entry, dict) and arch_matches(entry, arch):
            return True
    return False


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


def _validate_ukd_fields(ukd, where):
    """Validate the shape shared by inline and standalone UKDs.

    Both authoring forms carry the same fields; only the surrounding context
    (an entry in a KDP's kernelDescriptors vs. its own file) differs, which the
    caller conveys via `where`.
    """
    if not isinstance(ukd, dict):
        raise HkpPackError(f"{where} is not a JSON object")
    _require(ukd, ["id", "name", "kernel_source", "metadata", "priority"], where)
    if "arch" in ukd:
        arch = ukd["arch"]
        if not isinstance(arch, list) or not all(
            isinstance(a, str) and a for a in arch
        ):
            raise HkpPackError(
                f"{where} 'arch' must be a list of strings (empty = wildcard)"
            )
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


def _validate_inline_ukd(ukd, kdp_path):
    if not isinstance(ukd, dict):
        raise HkpPackError(f"inline UKD in {kdp_path.name} is not a JSON object")
    where = f"UKD '{ukd.get('id', '?')}' in {kdp_path.name}"
    _validate_ukd_fields(ukd, where)


def _validate_standalone_ukd(desc):
    """A standalone `<name>.ukd.json` is authored in the same hip form as inline.

    Its optional `arch` narrows the shards it ships in (empty/omitted = wildcard,
    applying to every referencing arch) and must be a subset of each referencing
    KDP's arch, checked in _validate_references.
    """
    doc = desc.doc
    where = f"standalone UKD {desc.path.name}"
    _validate_ukd_fields(doc, where)
    if doc["kernel_source"]["kind"] != "hip":
        raise HkpPackError(
            f"{where} must be authored in hip form "
            f"(got kind='{doc['kernel_source']['kind']}')"
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
    # Entries are heterogeneous: an inline UKD object, or a bare id string naming
    # a standalone `<name>.ukd.json` file (resolved in _validate_references).
    for ukd in kds:
        if isinstance(ukd, str):
            continue
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
        _validate_standalone_ukd(desc)
    if dtype == KDP_TYPE:
        _validate_kdp(desc)


def load_flat_input(root, log=print):
    """Load and structurally validate every *.json descriptor in a flat folder.

    root holds the authored source folder: KDP files (with inline hip UKDs), any
    standalone `<name>.ukd.json` files a KDP references by Id, and the by-Id
    generic files (UMD/UED/UDD/KMD/UHD), plus the HIP sources the UKDs name.
    Each descriptor's type is derived from its `<name>.<type>.json`
    filename. A `*.json` whose name carries no type token (not `<name>.<type>.json`)
    is not one of ours: warn and skip it rather than aborting the pack, so an
    incidental file in the source folder is tolerated. Raises HkpPackError on any
    malformed / missing-field / unknown-type / dangling-reference descriptor that
    IS type-tagged.
    """
    root = Path(root)
    if not root.is_dir():
        raise HkpPackError(f"input folder does not exist: {root}")

    descriptors = []
    for jp in sorted(root.glob("*.json")):
        if type_from_filename(jp) is None:
            log(f"skipping non-descriptor file {jp.name}")
            continue
        desc = Descriptor(path=jp, doc=_read_json(jp))
        _validate_shape(desc)
        descriptors.append(desc)

    flat = FlatInput(root=root, descriptors=descriptors)
    _validate_references(flat)
    return flat


def _validate_references(flat):
    ids = {d.id for d in flat.descriptors}
    ukd_by_id = flat.ukd_by_id()
    ukd_ids = set(ukd_by_id)
    # An inline UKD and a standalone UKD sharing an id would make a by-id KDP
    # reference ambiguous; reject the collision rather than silently pick one.
    for kdp in flat.kdps():
        for entry in kdp.doc.get("kernelDescriptors", []):
            if isinstance(entry, dict) and entry.get("id") in ukd_ids:
                raise HkpPackError(
                    f"inline UKD Id '{entry.get('id')}' in {kdp.path.name} "
                    "collides with a standalone UKD of the same Id"
                )
    for kdp in flat.kdps():
        doc = kdp.doc
        kdp_arch = doc.get("arch") or []
        refs = list(doc.get("matchers", []))
        refs += [doc.get("engine"), doc.get("dispatch")]
        for ref in refs:
            if ref is not None and ref not in ids:
                raise HkpPackError(
                    f"KDP {kdp.path.name} references unknown descriptor Id '{ref}'"
                )
        for entry in doc.get("kernelDescriptors", []):
            if isinstance(entry, str):
                if entry not in ukd_ids:
                    raise HkpPackError(
                        f"KDP {kdp.path.name} references unknown UKD Id '{entry}'"
                    )
                udoc = ukd_by_id[entry].doc
            else:
                udoc = entry
            if not _arch_subset_ok(udoc.get("arch") or [], kdp_arch):
                raise HkpPackError(
                    f"UKD '{udoc.get('id')}' arch {udoc.get('arch')} is not a "
                    f"subset of KDP {kdp.path.name} arch {doc.get('arch')}"
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
