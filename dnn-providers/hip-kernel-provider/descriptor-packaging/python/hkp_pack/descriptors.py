import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

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
# Every kernel_source kind a UKD may declare; _validate_ukd_fields rejects others.
UKD_KINDS = ("hip", "rocke", "hsaco", "kpack", "embedded_source")

_SCALAR_TYPES = (str, int, float, bool)

# A UED engine name is a scoped 'namespace:local' identifier (loader is
# authoritative, RFC 0020 §4.2): exactly one colon, neither first nor last, with
# the name-char class on both halves.
_UED_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+")
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
# A UHD model `hash`: the artifact's SHA-256 as UhdParser and the adapters spell it.
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def canonical_id(value):
    """The key a descriptor id or reference resolves by.

    The loader parses a UUID to its bytes, so a reference matches its target in
    any letter case. Any other value is compared as written.
    """
    if isinstance(value, str) and _UUID_RE.fullmatch(value):
        return value.lower()
    return value


def type_from_filename(path):
    """Descriptor type token from a `<name>.<type>.json` filename.

    The type is the second-to-last dot-separated segment of the file name.
    Returns None if the name has too few segments to carry a type token.
    """
    parts = Path(path).name.split(".")
    if len(parts) < 3:
        return None
    return parts[-2]


@dataclass(frozen=True)
class Sidecar:
    """A non-descriptor file (e.g. a model) a descriptor names by relative path.

    `rel_dir` preserves the authored layout under the source root, so the authored
    relative path still resolves in the packed tree.
    """

    source: Path
    rel_dir: Path
    name: str


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
    sidecars: list = field(default_factory=list)

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
        """Generics keyed by canonical_id; look a reference up by the same key."""
        return {canonical_id(d.id): d for d in self.generics()}

    def ukds(self):
        return self.by_type(UKD_TYPE)

    def ukd_by_id(self):
        """Standalone UKDs keyed by canonical_id; look a reference up by the same key."""
        return {canonical_id(d.id): d for d in self.ukds()}

    def sidecars_for(self, descriptors):
        """Every sidecar the given descriptors name, in descriptor order."""
        return [sidecar for desc in descriptors for sidecar in desc.sidecars]


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

    Mirrors the loader's parseDescriptorVersion (loader is authoritative),
    including its nine-digit cap on each half; the tool fails fast on a
    malformed value rather than shipping an ungatable file.
    """
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,9}\.[0-9]{1,9}", value):
        raise HkpPackError(
            f"{where} has invalid version '{value}' "
            "(expected '<major>.<minor>' with numeric halves of at most 9 digits)"
        )


def arch_matches(kdp_doc, arch):
    """A KDP matches an arch iff its arch list is empty (wildcard) or lists it."""
    archs = kdp_doc.get("arch")
    if not archs:
        return True
    return arch in archs


def _selected_entries(doc, arch, ukd_by_id):
    """Yield the entries of doc that ship for arch.

    Yields `(entry_id, ukd_doc, sdesc)`: for a standalone-UKD id ref, the id
    string, that UKD's doc, and its Descriptor; for an inline UKD, `None`, the
    entry dict itself, and `None`. The Descriptor rather than its rel_dir,
    because the walk needs its `path.name` for the error context and for the
    shipped filename as well as its `rel_dir` for the variant key.

    All three arch filters live here and nowhere else, so the prewarm and the
    serial walk cannot select different variant sets. `ukd_by_id` arrives as a
    parameter rather than being reached for through `flat`, which leaves the
    generator no way to enumerate a standalone UKD no KDP references: an orphan
    is legal input the walk never compiles.
    """
    if not arch_matches(doc, arch):
        return
    for entry in doc["kernelDescriptors"]:
        if isinstance(entry, str):
            sdesc = ukd_by_id[canonical_id(entry)]
            if arch_matches(sdesc.doc, arch):
                yield entry, sdesc.doc, sdesc
        elif arch_matches(entry, arch):
            yield None, entry, None


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
    Requires a validated `flat`: every by-id entry resolves.
    """
    return any(True for _ in _selected_entries(kdp_doc, arch, flat.ukd_by_id()))


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


def _validate_provenance(provenance, where, *, produced=False):
    """Shape of one `provenance` block, wherever it is declared.

    A KDP and the kernels under it declare the same `specialization_contract`
    object, so one rule covers both and an unusable declaration fails at the
    document that wrote it.

    `effective_spec` is the producing compiler's statement about what it observed,
    so an authored input claiming one is refused. `produced` is true only for a
    shipped `kpack` kernel, never for a KDP, which has no payload bytes to bind.
    """
    if not isinstance(provenance, dict):
        raise HkpPackError(f"{where}: provenance must be an object")
    if not produced and "effective_spec" in provenance:
        raise HkpPackError(
            f"{where}: provenance.effective_spec is reserved for the producing "
            "compiler and cannot be authored"
        )
    if "specialization_contract" in provenance:
        contract = provenance["specialization_contract"]
        if (
            not isinstance(contract, dict)
            or set(contract) != {"schema_version", "consumers"}
            or contract["schema_version"] != 1
            or not isinstance(contract["consumers"], list)
            or not contract["consumers"]
        ):
            raise HkpPackError(
                f"{where}: specialization_contract must be "
                "{'schema_version': 1, 'consumers': [...]} with at least one consumer"
            )


def _validate_embedded_source_file(source_file, where):
    """Reject an embedded_source `source_file` that cannot act as an identity.

    The value names the source file and is not normalised anywhere. A '..'
    segment lets one file be named by two different spellings, so one file
    takes two identities. An absolute path names a location on one machine,
    and the emitted key must be the same on every machine.
    """
    if not isinstance(source_file, str) or not source_file:
        raise HkpPackError(
            f"{where} kernel_source 'source_file' must be a non-empty string"
        )
    posix = source_file.replace("\\", "/")
    if ".." in posix.split("/"):
        raise HkpPackError(
            f"{where} kernel_source source_file '{source_file}' must not "
            "contain a '..' segment"
        )
    if posix.startswith("/") or PureWindowsPath(source_file).is_absolute():
        raise HkpPackError(
            f"{where} kernel_source source_file '{source_file}' must be "
            "relative to its descriptor, not absolute"
        )


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
    if kind not in UKD_KINDS:
        raise HkpPackError(
            f"{where} kernel_source has unsupported kind '{kind}' "
            f"(expected one of {', '.join(repr(k) for k in UKD_KINDS)})"
        )
    _validate_provenance(ukd.get("provenance", {}), where, produced=kind == "kpack")
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
        for field_name in ("file", "symbol"):
            value = ks[field_name]
            if not isinstance(value, str) or not value:
                raise HkpPackError(
                    f"{where} hsaco '{field_name}' must be a non-empty string"
                )
        if not ks["symbol"].isascii():
            raise HkpPackError(f"{where} hsaco 'symbol' must be ASCII")
        # An hsaco object is built for specific archs, so a wildcard would ship
        # its bytes into every shard.
        if not ukd.get("arch"):
            raise HkpPackError(
                f"{where} hsaco requires a non-empty 'arch' naming the arch(es) "
                "the code object runs on"
            )
    elif kind == "kpack":
        _require(ks, ["library", "toc_key", "symbol", "sha256", "signature"], where)
    else:
        _require(ks, ["source_file", "entry_point"], where)
        _validate_embedded_source_file(ks["source_file"], where)


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
    UKD may be of any kind that function accepts. Its optional `arch` narrows
    the shards it ships in (empty/omitted = wildcard, applying to every
    referencing arch) and must be a subset of each referencing KDP's arch,
    checked in _validate_references.
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
    _validate_provenance(doc.get("provenance", {}), where)
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
    """Mirrors parseEngineDescriptor; knobs are checked against the KMD in
    _validate_references, once every file is loaded."""
    doc = desc.doc
    where = f"UED {desc.path.name}"
    if "heuristic" in doc:
        raise HkpPackError(
            f"{where}: a top-level 'heuristic' field is not supported; use role/arch maps"
        )
    # The loader's requireKnownKeys also admits an unprefixed `provenance`.
    _known_keys(doc, (*_UED_KEYS, "provenance"), where)
    _validate_uuid(doc["id"], f"{where}.id")
    name = doc.get("name")
    if not isinstance(name, str) or not _UED_NAME_RE.fullmatch(name):
        raise HkpPackError(
            f"{where} name '{name}' must be scoped 'namespace:local' "
            "matching ^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$"
        )
    _require(doc, ["metadata"], where)
    _validate_uuid(doc["metadata"], f"{where}.metadata")
    for key in ("knobs", "behavior_notes", "numerical_notes"):
        values = doc.get(key, [])
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise HkpPackError(f"{where}.{key} must be an array of strings")
        seen = set()
        for value in values:
            if value in seen:
                raise HkpPackError(f"{where}.{key} lists '{value}' twice")
            seen.add(value)
    for note in doc.get("behavior_notes", []):
        if note not in _BEHAVIOR_NOTES:
            raise HkpPackError(
                f"{where}.behavior_notes has unknown note '{note}' "
                f"(expected one of {', '.join(_BEHAVIOR_NOTES)})"
            )
    if "sdk_version" in doc:
        version = doc["sdk_version"]
        if not isinstance(version, str) or not _sdk_version_fits(version):
            raise HkpPackError(
                f"{where}.sdk_version {version!r} is not a MAJOR.MINOR.PATCH version"
            )
    if "graph_match" in doc:
        graph_match = doc["graph_match"]
        _known_keys(graph_match, ("native", "provenance"), f"{where}.graph_match")
        _string(graph_match.get("native"), f"{where}.graph_match.native")
    for role in _UHD_ROLES:
        if role not in doc:
            continue
        entries = doc[role]
        if not isinstance(entries, dict) or not entries:
            raise HkpPackError(f"{where}.{role} must be a nonempty arch-to-UUID map")
        for arch, value in entries.items():
            if arch != "default":
                _reject_nonbare_arch([arch], where)
            entry_where = f"{where}.{role}.{arch}"
            if isinstance(value, str) or role not in _METRIC_ROLES:
                _validate_uuid(value, entry_where)
                continue
            # RFC 0019 §3.1: a scoring role may list one UHD per metric.
            if not isinstance(value, list) or not value:
                raise HkpPackError(
                    f"{entry_where} must be a UUID or a nonempty list of UUIDs"
                )
            seen = set()
            for identity in value:
                _validate_uuid(identity, entry_where)
                if canonical_id(identity) in seen:
                    raise HkpPackError(f"{entry_where} lists {identity} twice")
                seen.add(canonical_id(identity))


# hipdnn_data_sdk::utilities::Version reads `sdk_version` with three `operator>>`
# integer extractions separated by '.': whitespace may precede each integer or
# dot, an integer may carry a sign, and text after the patch is ignored.
_SDK_VERSION_RE = re.compile(
    r"\s*([+-]?[0-9]+)\s*\.\s*([+-]?[0-9]+)\s*\.\s*([+-]?[0-9]+)", re.ASCII
)


def _sdk_version_fits(text):
    """Whether Version(text) parses: the shape above, each part within int."""
    parsed = _SDK_VERSION_RE.match(text)
    return parsed is not None and all(
        -(2**31) <= int(part) < 2**31 for part in parsed.groups()
    )


def _role_references(doc, role):
    """Every UHD id a UED role names, flattening the list form of a scoring role."""
    for value in doc.get(role, {}).values():
        if isinstance(value, list):
            yield from value
        else:
            yield value


# The loader's enum vocabularies, mirrored so a bad spelling is a pack-time
# error rather than a runtime file-drop. Loader is authoritative:
# DescriptorLoader.hpp matchScopeFromString / uhdAdapterFromString /
# metadataTypeFromString / behaviorNoteFromString. `onnx` is absent because
# uhdAdapterFromString refuses it: no adapter implements it.
_MATCH_SCOPES = ("graph", "kernel")
_UHD_ADAPTERS = (
    "static_order",
    "native",
    "tree_data",
    "table",
    "custom_library",
)
_BEHAVIOR_NOTES = ("runtime_compilation",)
# ScoreTransform.hpp SUPPORTED_TRANSFORMS, less the empty name `text()` refuses.
_SCORE_TRANSFORMS = ("identity", "log1p", "log", "exp", "sqrt")
_UHD_ROLES = ("sort_kernel_catalog", "predict_engine", "predict_applicable_kernels")
# Roles mapping an arch to one UHD per ranking metric. A candidate generator has no
# metric, so predict_applicable_kernels stays single.
_METRIC_ROLES = ("sort_kernel_catalog", "predict_engine")
_UED_KEYS = (
    "version",
    "revision",
    "id",
    "name",
    "sdk_version",
    *_UHD_ROLES,
    "metadata",
    "knobs",
    "behavior_notes",
    "numerical_notes",
    "graph_match",
)
# RFC 0019 §4.4 ranking-metric registry (RankingMetrics.hpp): metric -> required
# UHD `objective`.
_RANKING_METRIC_OBJECTIVES = {"tflops": "max", "time": "min"}
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


def _validate_uuid(value, where):
    if not isinstance(value, str) or not _UUID_RE.fullmatch(value):
        raise HkpPackError(f"{where} requires a UUID, got {value!r}")


def _known_keys(value, allowed, where):
    """Reject keys outside @p allowed; `x-` and `_` keys are extension data the
    runtime ignores at every level (UhdParser keys(), DescriptorLoader
    requireKnownKeys)."""
    if not isinstance(value, dict):
        raise HkpPackError(f"{where} must be an object")
    unknown = sorted(
        key for key in value if key not in allowed and not key.startswith(("x-", "_"))
    )
    if unknown:
        raise HkpPackError(
            f"{where} has unknown fields {unknown}; "
            "extension keys must start with 'x-' or '_'"
        )


def _string(value, where):
    if not isinstance(value, str) or not value:
        raise HkpPackError(f"{where} must be a nonempty string")


def _validate_trained_against(value, where):
    # RFC 0019 §4.1: names either the descriptor set (ued/kmd/umd, all or none) or,
    # for a model bound by provider-declared UUID, a selector_revision. Either may add
    # feature_semantics_revision (integer >= 1). Loader is authoritative: UhdParser.hpp.
    _known_keys(
        value,
        ("ued", "kmd", "umd", "selector_revision", "feature_semantics_revision"),
        where,
    )
    if "feature_semantics_revision" in value:
        semantics = value["feature_semantics_revision"]
        if (
            isinstance(semantics, bool)
            or not isinstance(semantics, int)
            or not 1 <= semantics < 2**63
        ):
            raise HkpPackError(
                f"{where}.feature_semantics_revision must be an integer >= 1"
            )
    names_descriptor_set = any(key in value for key in ("ued", "kmd", "umd"))
    if "selector_revision" in value:
        revision = value["selector_revision"]
        if not isinstance(revision, str) or not revision:
            raise HkpPackError(f"{where}.selector_revision must be a nonempty string")
    elif not names_descriptor_set:
        raise HkpPackError(
            f"{where} must name a descriptor set (ued/kmd/umd) or a selector_revision"
        )
    if not names_descriptor_set:
        return
    _require(value, ("ued", "kmd", "umd"), where)
    if not isinstance(value["umd"], list):
        raise HkpPackError(f"{where}.umd must be an array")
    for kind in ("ued", "kmd", "umd"):
        dependencies = value[kind] if kind == "umd" else [value[kind]]
        seen = set()
        for dependency in dependencies:
            entry_where = f"{where}.{kind}"
            _known_keys(dependency, ("id", "revision"), entry_where)
            _require(dependency, ("id", "revision"), entry_where)
            _validate_uuid(dependency["id"], entry_where)
            _validate_version(dependency["revision"], f"{entry_where} revision")
            identity = dependency["id"].lower()
            if identity in seen:
                raise HkpPackError(f"{entry_where} repeats dependency {identity}")
            seen.add(identity)


def _validate_uhd(desc, source_root):
    """Mirror the canonical Draft7 header, then resolve artifact sidecars safely.

    Packaging requires artifacts to exist; runtime may retain an unavailable model binding.
    """
    doc = desc.doc
    where = f"UHD {desc.path.name}"
    _known_keys(
        doc,
        (
            "version",
            "id",
            "name",
            "adapter",
            "features_signature",
            "features_hash",
            "categorical_encoding",
            "trained_against",
            "objective",
            "score",
            # Free-form authoring notes, never read; root only.
            "provenance",
            *_UHD_ADAPTERS,
        ),
        where,
    )
    _require(doc, ("version", "id", "name", "adapter"), where)
    if doc["version"] != "1.0":
        raise HkpPackError(
            f"{where}: unsupported file-format version {doc['version']!r}"
        )
    _validate_uuid(doc["id"], where)
    _string(doc["name"], f"{where}.name")
    _require_enum(doc, "adapter", _UHD_ADAPTERS, where)
    adapter = doc["adapter"]
    bodies = [key for key in _UHD_ADAPTERS if key in doc]
    if bodies != [adapter] or not isinstance(doc[adapter], dict):
        raise HkpPackError(
            f"{where} requires exactly one body matching adapter '{adapter}'"
        )
    if adapter != "static_order" or "objective" in doc:
        _require(doc, ("objective",), where)
        _require_enum(doc, "objective", ("max", "min"), where)
    if adapter in ("tree_data", "table"):
        _require(doc, ("features_signature", "features_hash", "trained_against"), where)
    if "features_signature" in doc:
        signature = doc["features_signature"]
        if not isinstance(signature, list) or not signature:
            raise HkpPackError(f"{where}.features_signature must be a nonempty array")
        for entry in signature:
            if not (
                (isinstance(entry, str) and entry.startswith("$") and len(entry) > 1)
                or (isinstance(entry, dict) and len(entry) == 1)
            ):
                raise HkpPackError(
                    f"{where}.features_signature requires bare references or inline expressions"
                )
        _require(doc, ("features_hash", "trained_against"), where)
    if "features_hash" in doc:
        value = doc["features_hash"]
        if not isinstance(value, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{16}", value
        ):
            raise HkpPackError(f"{where}.features_hash must be a sha256 digest")
    if "trained_against" in doc:
        _validate_trained_against(doc["trained_against"], f"{where}.trained_against")
    if "categorical_encoding" in doc:
        encoding = doc["categorical_encoding"]
        if not isinstance(encoding, dict):
            raise HkpPackError(f"{where}.categorical_encoding must be an object")
        for name, codes in encoding.items():
            if not name.startswith("$"):
                raise HkpPackError(
                    f"{where}.categorical_encoding key '{name}' must be a '$' reference"
                )
            if (
                not isinstance(codes, dict)
                or not codes
                or any(
                    type(code) is not int or not -(2**31) <= code < 2**31
                    for code in codes.values()
                )
            ):
                raise HkpPackError(
                    f"{where}.categorical_encoding.{name} must map values to int32 codes"
                )
    if "score" in doc:
        score = doc["score"]
        _known_keys(score, ("metric", "calibrated", "transform"), f"{where}.score")
        if "transform" in score:
            _require_enum(score, "transform", _SCORE_TRANSFORMS, f"{where}.score")
        if "metric" in score and (
            not isinstance(score["metric"], str)
            or score["metric"] not in _RANKING_METRIC_OBJECTIVES
        ):
            raise HkpPackError(
                f"{where}.score.metric {score['metric']!r} is not a registered ranking "
                f"metric (expected one of {', '.join(_RANKING_METRIC_OBJECTIVES)})"
            )
        if "calibrated" in score and not isinstance(score["calibrated"], bool):
            raise HkpPackError(f"{where}.score.calibrated must be a boolean")
        # A calibrated score is comparable across engines only in a named quantity
        # (RFC 0019 §4.4).
        if score.get("calibrated") and "metric" not in score:
            raise HkpPackError(f"{where}: a calibrated score requires score.metric")
        # The metric fixes the ranking direction; `objective` must agree with it.
        if "metric" in score:
            expected = _RANKING_METRIC_OBJECTIVES[score["metric"]]
            if doc.get("objective") != expected:
                raise HkpPackError(
                    f"{where}: score.metric '{score['metric']}' requires objective "
                    f"'{expected}', got {doc.get('objective')!r}"
                )
    body = doc[adapter]
    if adapter == "static_order":
        # static_order ranks by UKD priority, then descriptor id. Declared criteria are
        # refused, as UhdParser refuses them, rather than packed and ignored.
        if isinstance(body, dict) and "order" in body:
            raise HkpPackError(
                f"{where}.static_order.order is not supported: declared ordering criteria are "
                "not implemented; static_order ranks by priority, then descriptor id"
            )
        _known_keys(body, (), f"{where}.{adapter}")
        return
    if adapter == "native":
        _known_keys(body, ("symbol",), f"{where}.{adapter}")
        _string(body.get("symbol"), f"{where}.native.symbol")
        return
    key = "library" if adapter == "custom_library" else "artifact"
    allowed = (
        ("library", "symbol", "hash", "config")
        if adapter == "custom_library"
        else ("artifact", "hash")
    )
    _known_keys(body, allowed, f"{where}.{adapter}")
    _string(body.get(key), f"{where}.{adapter}.{key}")
    _validate_contained_path(body[key], f"{where}.{adapter}.{key}")
    # A library is loaded as code, so it must declare the digest it is verified
    # against; any other artifact without one is digested by the loader.
    if adapter == "custom_library" or "hash" in body:
        _require(body, ("hash",), f"{where}.{adapter}")
        if not isinstance(body["hash"], str) or not _SHA256_RE.fullmatch(body["hash"]):
            raise HkpPackError(
                f"{where}.{adapter}.hash must be the SHA-256 of the {key} as 64 "
                f"lowercase hexadecimal digits, got {body['hash']!r}"
            )
    if adapter == "custom_library":
        _string(body.get("symbol"), f"{where}.{adapter}.symbol")
        # The runtime supports no library configuration; only `{}` loads.
        if "config" in body and body["config"] != {}:
            raise HkpPackError(f"{where}.custom_library.config must be an empty object")
    desc.sidecars.append(_resolve_sidecar(desc, source_root, body[key]))


def _validate_contained_path(payload, where):
    """Mirror UhdParser's portable '/'-separated descriptor-relative asset names."""
    segments = payload.split("/")
    if (
        not payload
        or payload.startswith("/")
        or any(char in payload for char in ("\\", ":", "\0"))
        or ".." in segments
        or segments[-1] in ("", ".")
    ):
        raise HkpPackError(
            f"{where} {payload!r} must be a relative path inside the descriptor's directory"
        )


def _resolve_sidecar(desc, source_root, payload):
    """Resolve a descriptor-relative payload path to a carriable Sidecar.

    No root-relative fallback (matching compile_hip_variant): a typo must not bind
    to a same-named file elsewhere. The path is already known to stay inside the
    descriptor's directory (_validate_contained_path).

    The staged location is the authored path, lexically normalised as UhdParser
    normalises it before opening it. Symlinks are followed only to check
    containment and to read the bytes, so a link is staged under its own name.
    """
    where = f"UHD {desc.path.name}"
    dest = Path(os.path.normpath(Path(desc.rel_dir) / payload))
    try:
        root = Path(source_root).resolve()
        resolved = (root / dest).resolve()
        directory = (root / desc.rel_dir).resolve()
        # Symlinks must remain inside the descriptor's directory, not merely the source root.
        escapes = not resolved.is_relative_to(directory)
        found = resolved.is_file()
    except (OSError, ValueError) as exc:
        raise HkpPackError(
            f"{where} payload path {payload!r} is unusable: {exc}"
        ) from exc

    if escapes:
        raise HkpPackError(
            f"{where} payload escapes the descriptor's directory: {payload} "
            f"(from {Path(desc.rel_dir).as_posix()}, resolved to {resolved})"
        )
    # Same rule, and same case-insensitivity, as an authored descriptor folder.
    if dest.parts and dest.parts[0].lower() == KPACK_DIR_NAME:
        raise HkpPackError(
            f"{where} payload {payload} lands in the reserved '{KPACK_DIR_NAME}/' "
            "folder, where the per-arch archive is written"
        )
    if not found:
        raise HkpPackError(
            f"{where} payload source not found: {payload} (looked for {resolved}, "
            f"resolved relative to descriptor folder {Path(desc.rel_dir).as_posix()})"
        )
    return Sidecar(source=resolved, rel_dir=dest.parent, name=dest.name)


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


def _validate_shape(desc, source_root, log=print):
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
    if dtype in ("ued", "kmd", "umd") and "revision" in doc:
        _validate_version(doc["revision"], f"descriptor {path.name} revision")
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
        _validate_uhd(desc, source_root)
    if dtype == "kmd":
        _validate_kmd(desc)


def load_flat_input(root, log=print, *, exclude_folders=(), disabled_kinds=()):
    """Load and structurally validate every *.json descriptor under a root.

    Walks the root recursively: a descriptor's authored subpath is meaningful
    and is carried through to the staged and installed layouts. Loads the KDPs
    (with inline hip UKDs), standalone `<name>.ukd.json` files a KDP references
    by Id, and the by-Id generic files (UMD/UED/UDD/KMD/UHD), plus the HIP
    sources the UKDs name. Each descriptor's type is derived from its
    `<name>.<type>.json` filename. A `*.json` whose name carries no type token
    is not one of ours: warn and skip it rather than aborting the pack, so an
    incidental file in the source folder is tolerated. A hidden path -- any
    dot-prefixed segment, or a dot-prefixed filename -- is warned and skipped
    the same way, so nothing the walk passes over is invisible. Raises
    HkpPackError on any malformed / missing-field / unknown-type /
    dangling-reference descriptor that IS type-tagged.

    `exclude_folders` names top-level child folders of the root that are not
    content in this build: nothing under one is read, validated or shipped.
    A name matches case-insensitively, as the reserved `kpack/` folder does,
    and one naming no folder is ignored (a root need not carry every family).
    `disabled_kinds` names `kernel_source` kinds this build has no producer
    for: those UKDs are still read and validated, then pruned like an
    arch-pruned UKD once validation passes. An unknown kind raises HkpPackError.

    There is exactly ONE root. Child folders under it scope the content (a
    `hip/` tree and a `rocKE/` tree, per-integration folders beneath those);
    producer selection is per-UKD on `kernel_source.kind`, never per-root. Two
    descriptors therefore cannot share a path, so the filesystem itself enforces
    the uniqueness that a multi-root merge had to check for.
    """
    root = Path(root)
    if not root.is_dir():
        raise HkpPackError(f"input folder does not exist: {root}")

    unknown = sorted(set(disabled_kinds) - set(UKD_KINDS))
    if unknown:
        raise HkpPackError(
            f"cannot disable unknown kernel_source kind(s) {unknown}; "
            f"known kinds are {list(UKD_KINDS)}"
        )

    # Case-insensitive for the same reason as the `kpack/` guard below: the packed
    # tree is also consumed on Windows, where `rocKE` and `rocke` are one directory.
    excluded = {name.lower() for name in exclude_folders}
    for name in sorted(p.name for p in root.iterdir() if p.is_dir()):
        if name.lower() in excluded:
            log(f"excluding disabled family folder {name}/")

    descriptors = []
    for jp in sorted(root.rglob("*.json")):
        rel_path = jp.relative_to(root)
        # A disabled family folder is not content in this build: its files are never
        # read, so they can neither ship nor fail validation.
        if len(rel_path.parts) > 1 and rel_path.parts[0].lower() in excluded:
            continue
        # A dot-prefixed segment at any depth, or a dot-prefixed filename. The
        # source root is user-supplied and plausibly a checkout, so `.git/`,
        # `.venv/` and friends are skipped rather than refused, unlike the
        # reserved `kpack/` below -- a hidden path collides with nothing.
        if any(part.startswith(".") for part in rel_path.parts):
            log(f"skipping hidden path {rel_path}")
            continue
        if type_from_filename(jp) is None:
            log(f"skipping non-descriptor file {rel_path}")
            continue
        rel_dir = jp.parent.relative_to(root)
        # `kpack/` at the arch root is where the archive itself is written, and
        # `library` on every packed UKD is a path that ends there. An authored
        # folder of that name lands descriptors inside the reserved directory,
        # intermixed with the archive -- today they survive only because the
        # archive happens to be written last. Refuse the name rather than depend
        # on write order.
        # The comparison is case-insensitive. On Linux `KPACK/` and `kpack/` are
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
        _validate_shape(desc, root, log)
        descriptors.append(desc)

    flat = FlatInput(descriptors=descriptors)
    _reject_inline_standalone_collision(flat)
    _reject_duplicate_ids(flat)
    _validate_references(flat)
    _warn_orphan_standalone_ukds(flat, log)
    if disabled_kinds:
        _drop_disabled_kinds(flat, frozenset(disabled_kinds), log)
    return flat


def _drop_disabled_kinds(flat, disabled_kinds, log):
    """Prune every UKD whose kind this build has no producer for, as arch pruning
    prunes one that does not apply.

    Runs after validation, so a disabled UKD is still checked and an unknown kind
    is still an error. Each KDP keeps only its enabled entries; a KDP left with
    none is dropped with the disabled standalone UKDs, and the generics only it
    reached are pruned later like any other unreachable generic.
    """
    ukd_by_id = flat.ukd_by_id()

    def _disabled(entry):
        doc = ukd_by_id[canonical_id(entry)].doc if isinstance(entry, str) else entry
        return doc["kernel_source"]["kind"] in disabled_kinds

    dropped = set()
    for kdp in flat.kdps():
        entries = kdp.doc.get("kernelDescriptors", [])
        kept = [e for e in entries if not _disabled(e)]
        if len(kept) == len(entries):
            continue
        log(
            f"{kdp.path.name}: skipping {len(entries) - len(kept)} UKD(s) of a "
            f"disabled kind ({', '.join(sorted(disabled_kinds))})"
        )
        if kept:
            kdp.doc["kernelDescriptors"] = kept
        else:
            dropped.add(id(kdp))
    kept_descriptors = []
    for d in flat.descriptors:
        if id(d) in dropped:
            continue
        if d.type == UKD_TYPE:
            kind = d.doc["kernel_source"]["kind"]
            if kind in disabled_kinds:
                log(
                    f"standalone UKD {d.path.name} (id '{d.doc.get('id')}'): "
                    f"skipping disabled kind '{kind}'"
                )
                continue
        kept_descriptors.append(d)
    flat.descriptors = kept_descriptors


def _reject_inline_standalone_collision(flat):
    """An inline UKD id colliding with a standalone UKD is ambiguous by-id.

    A subset of the global id-uniqueness rule, kept ahead of it for its more
    specific message: a by-id KDP reference cannot pick between an inline and a
    standalone UKD of the same id.
    """
    ukd_ids = set(flat.ukd_by_id())
    for kdp in flat.kdps():
        for entry in kdp.doc.get("kernelDescriptors", []):
            if isinstance(entry, dict) and canonical_id(entry.get("id")) in ukd_ids:
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
        key = canonical_id(desc_id)
        if key in seen:
            raise HkpPackError(
                f"duplicate descriptor id '{desc_id}': defined by {seen[key]} "
                f"and {source}"
            )
        seen[key] = source

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
                referenced.add(canonical_id(entry))
    for ukd in flat.ukds():
        if canonical_id(ukd.id) not in referenced:
            log(
                f"standalone UKD {ukd.path.name} (id '{ukd.id}') is not referenced "
                "by any KDP"
            )


def _validate_references(flat):
    ids = {canonical_id(d.id) for d in flat.descriptors}
    ukd_by_id = flat.ukd_by_id()
    for kdp in flat.kdps():
        doc = kdp.doc
        kdp_arch = doc.get("arch") or []
        refs = list(doc.get("matchers", []))
        refs += [doc.get("engine"), doc.get("dispatch")]
        for ref in refs:
            if ref is not None and canonical_id(ref) not in ids:
                raise HkpPackError(
                    f"KDP {kdp.path.name} references unknown descriptor Id '{ref}'"
                )
        for entry in doc.get("kernelDescriptors", []):
            if isinstance(entry, str):
                sdesc = ukd_by_id.get(canonical_id(entry))
                if sdesc is None:
                    raise HkpPackError(
                        f"KDP {kdp.path.name} references unknown UKD Id '{entry}'"
                    )
                udoc = sdesc.doc
            else:
                udoc = entry
            if not _arch_subset_ok(udoc.get("arch") or [], kdp_arch):
                raise HkpPackError(
                    f"UKD '{udoc.get('id')}' arch {udoc.get('arch')} is not a "
                    f"subset of KDP {kdp.path.name} arch {doc.get('arch')}"
                )
    by_kind = {
        kind: {canonical_id(d.id): d for d in flat.by_type(kind)}
        for kind in ("kmd", "uhd")
    }
    for ued in flat.by_type("ued"):
        references = [(ued.doc["metadata"], "kmd")]
        references += [
            (ref, "uhd")
            for role in _UHD_ROLES
            for ref in _role_references(ued.doc, role)
        ]
        for ref, kind in references:
            if canonical_id(ref) not in by_kind[kind]:
                raise HkpPackError(
                    f"UED {ued.path.name} references unknown {kind.upper()} descriptor Id '{ref}'"
                )
        # The loader drops an engine exposing a knob its KMD does not declare.
        kmd = by_kind["kmd"][canonical_id(ued.doc["metadata"])]
        declared = {entry["name"] for entry in kmd.doc["fields"]}
        for knob in ued.doc.get("knobs", []):
            if knob not in declared:
                raise HkpPackError(
                    f"UED {ued.path.name} exposes knob '{knob}', which KMD "
                    f"{kmd.path.name} does not declare"
                )
        _validate_role_metrics(ued, by_kind["uhd"])


def _validate_role_metrics(ued, uhd_by_id):
    """At most one model per metric for each architecture a scoring role names.

    RFC 0019 §3.1: the loader disables a metric with two models. A `predict_engine`
    model must declare a metric.
    """
    for role in _METRIC_ROLES:
        for arch, value in ued.doc.get(role, {}).items():
            where = f"UED {ued.path.name}.{role}.{arch}"
            by_metric = {}
            for ref in [value] if isinstance(value, str) else value:
                metric = (uhd_by_id[canonical_id(ref)].doc.get("score") or {}).get(
                    "metric"
                )
                if metric is None and role == "predict_engine":
                    raise HkpPackError(
                        f"{where} names UHD '{ref}', which declares no "
                        "score.metric; an engine prediction needs one"
                    )
                if metric in by_metric:
                    what = f"metric '{metric}'" if metric else "no metric"
                    raise HkpPackError(
                        f"{where} names two UHDs with {what}: "
                        f"'{by_metric[metric]}' and '{ref}'"
                    )
                by_metric[metric] = ref


def reachable_generic_ids(flat, surviving_kdps):
    """Canonical ids (canonical_id) of the generics reachable from surviving KDPs.

    Walks KDP -> {matchers, engine, dispatch} and UED -> {role models, metadata}
    transitively. A generic survives pruning iff its canonical id is in this set.
    """
    by_id = flat.generic_by_id()
    reachable = set()
    pending = []
    for kdp in surviving_kdps:
        doc = kdp.doc
        pending += list(doc.get("matchers", []))
        pending += [doc.get("engine"), doc.get("dispatch")]
    while pending:
        rid = canonical_id(pending.pop())
        if rid is None or rid in reachable or rid not in by_id:
            continue
        reachable.add(rid)
        gdesc = by_id[rid]
        if gdesc.type == "ued":
            pending.append(gdesc.doc.get("metadata"))
            pending += [
                ref for role in _UHD_ROLES for ref in _role_references(gdesc.doc, role)
            ]
    return reachable
