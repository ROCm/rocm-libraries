import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import HkpPackError

KDP_TYPE = "kdp"
UKD_TYPE = "ukd"
UED_TYPE = "ued"
_GENERIC_TYPES = {"kmd", "ued", "umd", "udd", "uhd"}
_ALL_TYPES = {KDP_TYPE, UKD_TYPE} | _GENERIC_TYPES

# The per-arch archive directory, relative to an arch shard root. Reserved: the
# packer writes the .kpack here and every packed UKD's `library` resolves into
# it, so an authored folder of this name would collide with shipped output.
# Defined here rather than in pipeline.py because the loader enforces it and
# pipeline.py imports from this module.
KPACK_DIR_NAME = "kpack"

_SCALAR_TYPES = (str, int, float, bool)

# A UED engine name is a scoped 'namespace:local' identifier (loader is
# authoritative, RFC 0020 §4.2): exactly one colon, neither first nor last, with
# the name-char class on both halves.
_UED_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$")


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
    rel_dir: Path = Path(".")

    @property
    def type(self):
        return type_from_filename(self.path)

    @property
    def id(self):
        return self.doc.get("id")


@dataclass
class FlatInput:
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


def _validate_version(value, where):
    """A file-backed descriptor's version is '<major>.<minor>' with numeric halves.

    Mirrors the loader's parseDescriptorVersion (loader is authoritative); the
    tool fails fast on a malformed value rather than shipping an ungatable file.
    """
    if (
        not isinstance(value, str)
        or value.count(".") != 1
        or not all(part.isdigit() for part in value.split("."))
    ):
        raise HkpPackError(
            f"{where} has invalid version '{value}' "
            "(expected '<major>.<minor>' with numeric halves)"
        )


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
        for f in flags:
            if f.startswith("-fuse-cuid"):
                raise HkpPackError(
                    f"{where} has invalid build (-fuse-cuid is reserved; the tool "
                    "pins -fuse-cuid=none for reproducible code objects)"
                )


def validate_rocke_spec(spec, where):
    """A rocke UKD's spec block is a JSON object; nothing more is required here.

    Field-level correctness (the builder's spec dataclass) is a compile-time
    concern validated by build_spec in the producer, not at load time. The
    failure substring is stable ('invalid spec').
    """
    if not isinstance(spec, dict):
        raise HkpPackError(f"{where} has invalid spec (not an object)")


def _reject_nonbare_arch(archs, where):
    """Reject any arch entry that is not a bare gfx base target id.

    Mirrors the loader's isPlausibleArchBaseId (loader is authoritative): 'gfx'
    followed by one or more of [a-z0-9_-]. LLVM generic targets
    ('gfx9-4-generic') are legal; a feature suffix ('gfx942:xnack-') is not,
    since ':' is outside the set.

    Fatal rather than advisory: a suffixed arch matches no shard, so the KDP
    prunes from every arch and the pack exits 0 having installed nothing --
    indistinguishable from a legitimate arch skip.
    """
    for arch in archs or []:
        body = arch[3:]
        if (
            not arch.startswith("gfx")
            or not body
            or not all(c.islower() or c.isdigit() or c in "-_" for c in body)
        ):
            hint = (
                "it carries a feature suffix; name the base target (e.g. 'gfx942')"
                if ":" in arch
                else "expected a bare gfx target id (e.g. 'gfx942')"
            )
            raise HkpPackError(f"{where}: arch '{arch}' is not usable -- {hint}")


def _validate_ukd_fields(ukd, where, log=print):
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
        _reject_nonbare_arch(arch, where)
    ks = ukd["kernel_source"]
    if not isinstance(ks, dict) or "kind" not in ks:
        raise HkpPackError(f"{where} kernel_source missing 'kind'")
    kind = ks["kind"]
    if kind == "hip":
        _require(ks, ["source", "entry"], where)
        if "build" not in ks:
            raise HkpPackError(f"{where} has invalid build (absent)")
        validate_hip_build(ks["build"], where)
    elif kind == "rocke":
        _require(ks, ["source", "builder", "spec"], where)
        validate_rocke_spec(ks["spec"], where)
    elif kind == "hsaco":
        _require(ks, ["file", "symbol"], where)
    elif kind == "kpack":
        _require(ks, ["library", "toc_key", "symbol", "sha256"], where)
    else:
        raise HkpPackError(
            f"{where} kernel_source has unsupported kind '{kind}' "
            "(expected 'hip', 'rocke', 'hsaco', or 'kpack')"
        )


def _validate_inline_ukd(ukd, kdp_path, log=print):
    if not isinstance(ukd, dict):
        raise HkpPackError(f"inline UKD in {kdp_path.name} is not a JSON object")
    where = f"UKD '{ukd.get('id', '?')}' in {kdp_path.name}"
    # An inline UKD carries its own version, independent of the enclosing KDP's.
    _require(ukd, ["version"], where)
    _validate_version(ukd.get("version"), where)
    _validate_ukd_fields(ukd, where, log)


def _validate_standalone_ukd(desc, log=print):
    """A standalone `<name>.ukd.json` carries the same fields as an inline UKD.

    Kind-specific checks are delegated to _validate_ukd_fields, so a standalone
    UKD may be hip or rocke. Its optional `arch` narrows the shards it ships in
    (empty/omitted = wildcard, applying to every referencing arch) and must be a
    subset of each referencing KDP's arch, checked in _validate_references.
    """
    doc = desc.doc
    where = f"standalone UKD {desc.path.name}"
    _validate_ukd_fields(doc, where, log)


def _validate_kdp(desc, log=print):
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
    _reject_nonbare_arch(arch, where)
    kds = doc["kernelDescriptors"]
    if not isinstance(kds, list) or not kds:
        raise HkpPackError(f"{where} 'kernelDescriptors' must be a non-empty list")
    # Entries are heterogeneous: an inline UKD object, or a bare id string naming
    # a standalone `<name>.ukd.json` file (resolved in _validate_references).
    for ukd in kds:
        if isinstance(ukd, str):
            continue
        _validate_inline_ukd(ukd, path, log)


def _validate_ued(desc):
    name = desc.doc.get("name")
    if not isinstance(name, str) or not _UED_NAME_RE.match(name):
        raise HkpPackError(
            f"UED {desc.path.name} name '{name}' must be scoped 'namespace:local' "
            "matching ^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$"
        )


# The loader's enum vocabularies, mirrored so a bad spelling is a pack-time
# error rather than a runtime file-drop. Loader is authoritative:
# DescriptorLoader.hpp matchScopeFromString / heuristicKindFromString /
# metadataTypeFromString.
_MATCH_SCOPES = ("graph", "kernel")
_HEURISTIC_KINDS = ("native", "model")
_METADATA_TYPES = ("bool", "int", "float", "string", "int_list")


def _require_enum(doc, key, allowed, where):
    value = doc.get(key)
    if value not in allowed:
        raise HkpPackError(
            f"{where} has invalid {key} '{value}' "
            f"(expected one of {', '.join(allowed)})"
        )


def _validate_umd(desc):
    """UMD: scope is a closed enum and match_symbol is required.

    Mirrors parseMatchDescriptor. A bad scope drops the matcher at load, which
    cascades: a KDP naming a matcher no descriptor defines loses its pack, and
    an engine with no loadable pack is dropped entirely.
    """
    where = f"UMD {desc.path.name}"
    _require(desc.doc, ["name", "scope", "match_symbol"], where)
    _require_enum(desc.doc, "scope", _MATCH_SCOPES, where)


def _validate_udd(desc):
    """UDD: dispatch_symbol is required. Mirrors parseDispatchDescriptor."""
    _require(desc.doc, ["name", "dispatch_symbol"], f"UDD {desc.path.name}")


def _validate_uhd(desc):
    """UHD: kind is a closed enum, payload required. Mirrors
    parseHeuristicDescriptor."""
    where = f"UHD {desc.path.name}"
    _require(desc.doc, ["name", "kind", "payload"], where)
    _require_enum(desc.doc, "kind", _HEURISTIC_KINDS, where)


def _validate_kmd(desc):
    """KMD: a list of fields, each with a name and a type from the enum.

    Mirrors parseMetadataSchema. The default_value/type agreement the loader
    also checks is not duplicated: it would have to match the loader's JSON-kind
    coercion rules exactly, and a near-miss would reject descriptors the runtime
    accepts.
    """
    where = f"KMD {desc.path.name}"
    _require(desc.doc, ["name", "fields"], where)
    fields = desc.doc["fields"]
    if not isinstance(fields, list):
        raise HkpPackError(f"{where} 'fields' must be a list")
    for entry in fields:
        if not isinstance(entry, dict):
            raise HkpPackError(f"{where} has a 'fields' entry that is not an object")
        entry_where = f"{where} field '{entry.get('name')}'"
        _require(entry, ["name", "type"], entry_where)
        _require_enum(entry, "type", _METADATA_TYPES, entry_where)


def _validate_shape(desc, log=print):
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
    # Every file-backed descriptor the tool reads carries a gatable version; only
    # the inline UKD form is exempt (rejected in _validate_inline_ukd).
    _require(doc, ["version"], f"descriptor {path.name}")
    _validate_version(doc.get("version"), f"descriptor {path.name}")
    if dtype == UKD_TYPE:
        _validate_standalone_ukd(desc, log)
    if dtype == KDP_TYPE:
        _validate_kdp(desc, log)
    if dtype == UED_TYPE:
        _validate_ued(desc)
    if dtype == "umd":
        _validate_umd(desc)
    if dtype == "udd":
        _validate_udd(desc)
    if dtype == "uhd":
        _validate_uhd(desc)
    if dtype == "kmd":
        _validate_kmd(desc)


def load_flat_input(root, log=print):
    """Load and structurally validate every *.json descriptor under a root.

    Walks the root recursively: a descriptor's authored subpath is meaningful
    and is carried through to the staged and installed layouts. Loads the KDPs
    (with inline hip UKDs), standalone `<name>.ukd.json` files a KDP references
    by Id, and the by-Id generic files (UMD/UED/UDD/KMD/UHD), plus the HIP
    sources the UKDs name. Each descriptor's type is derived from its
    `<name>.<type>.json` filename. A `*.json` whose name carries no type token
    is not one of ours: warn and skip it rather than aborting the pack, so an
    incidental file in the source folder is tolerated. Raises HkpPackError on any
    malformed / missing-field / unknown-type / dangling-reference descriptor that
    IS type-tagged.

    There is exactly ONE root. Child folders under it scope the content (a
    `hip/` tree and a `rocKE/` tree, per-integration folders beneath those);
    producer selection is per-UKD on `kernel_source.kind`, never per-root. Two
    descriptors therefore cannot share a path, so the filesystem itself enforces
    the uniqueness that a multi-root merge had to check for.
    """
    root = Path(root)
    if not root.is_dir():
        raise HkpPackError(f"input folder does not exist: {root}")

    descriptors = []
    for jp in sorted(root.rglob("*.json")):
        if type_from_filename(jp) is None:
            log(f"skipping non-descriptor file {jp.relative_to(root)}")
            continue
        rel_dir = jp.parent.relative_to(root)
        # `kpack/` at the arch root is where the archive itself is written, and
        # `library` on every packed UKD is a path that ends there. An authored
        # folder of that name lands descriptors inside the reserved directory,
        # intermixed with the archive -- today they survive only because the
        # archive happens to be written last. Refuse the name rather than depend
        # on write order.
        # Compared case-insensitively. On Linux `KPACK/` and `kpack/` are
        # distinct directories and coexist harmlessly (verified), so a
        # case-sensitive check would be correct here -- but the packed tree also
        # gets built and consumed on Windows, where they are the SAME directory
        # and the collision this guard exists to prevent comes back. Rejecting
        # both spellings costs an author nothing and keeps the rule identical on
        # every platform.
        if rel_dir.parts and rel_dir.parts[0].lower() == KPACK_DIR_NAME:
            raise HkpPackError(
                f"authored folder '{KPACK_DIR_NAME}/' is reserved: it is where "
                f"the per-arch archive is written, and every packed UKD's "
                f"'library' resolves into it. Rename it "
                f"(offending descriptor: {jp.relative_to(root)})"
            )
        desc = Descriptor(
            path=jp,
            doc=_read_json(jp),
            rel_dir=rel_dir,
        )
        _validate_shape(desc, log)
        descriptors.append(desc)

    flat = FlatInput(descriptors=descriptors)
    _reject_inline_standalone_collision(flat)
    _reject_duplicate_ids(flat)
    _validate_references(flat)
    _warn_orphan_standalone_ukds(flat, log)
    return flat


def _reject_inline_standalone_collision(flat):
    """An inline UKD id colliding with a standalone UKD is ambiguous by-id.

    A subset of the global id-uniqueness rule, kept ahead of it for its more
    specific message: a by-id KDP reference cannot pick between an inline and a
    standalone UKD of the same id.
    """
    ukd_ids = set(flat.ukd_by_id())
    for kdp in flat.kdps():
        for entry in kdp.doc.get("kernelDescriptors", []):
            if isinstance(entry, dict) and entry.get("id") in ukd_ids:
                raise HkpPackError(
                    f"inline UKD Id '{entry.get('id')}' in {kdp.path.name} "
                    "collides with a standalone UKD of the same Id"
                )


def _reject_duplicate_ids(flat):
    """Every descriptor id is unique across ALL types and forms at pack time.

    Build-time invariant, deliberately stronger than the loader, which keys on
    (type, id) and permits the same id on descriptors of different types. Packing
    hard-fails on any repeat so a copy-pasted id can never ship. Iterates in
    sorted-file then authored order so the "already defined by" pointer is
    deterministic.
    """
    seen = {}

    def _claim(desc_id, source):
        if desc_id is None:
            return
        if desc_id in seen:
            raise HkpPackError(
                f"duplicate descriptor id '{desc_id}': defined by {seen[desc_id]} "
                f"and {source}"
            )
        seen[desc_id] = source

    for desc in sorted(flat.descriptors, key=lambda d: d.path.name):
        _claim(desc.id, desc.path.name)
        if desc.type == KDP_TYPE:
            for entry in desc.doc.get("kernelDescriptors", []):
                if isinstance(entry, dict):
                    _claim(entry.get("id"), f"inline UKD in {desc.path.name}")


def _warn_orphan_standalone_ukds(flat, log):
    """Warn (non-fatal) for each standalone UKD no KDP references by id.

    An orphan still packs; the warning flags a likely authoring slip (a UKD file
    that no pack pulls in).
    """
    referenced = set()
    for kdp in flat.kdps():
        for entry in kdp.doc.get("kernelDescriptors", []):
            if isinstance(entry, str):
                referenced.add(entry)
    for ukd in flat.ukds():
        if ukd.id not in referenced:
            log(
                f"standalone UKD {ukd.path.name} (id '{ukd.id}') is not referenced "
                "by any KDP"
            )


def _validate_references(flat):
    ids = {d.id for d in flat.descriptors}
    ukd_by_id = flat.ukd_by_id()
    ukd_ids = set(ukd_by_id)
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
