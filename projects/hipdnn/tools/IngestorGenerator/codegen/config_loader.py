# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""YAML config loading and validation for the generic-kernel-ingestor generator.

Mirrors ``DescriptorGenerator/codegen/config_loader.py``'s shape: one
``ConfigError``, presence-based deprecated-key rejection, and every check
run **before** ``generator.py`` mints a single UUID. The five checks below
each catch a failure mode that the real ``DescriptorLoader.hpp`` either
lets through silently (knobs, arch shape) or only reports after dropping
an entire pack/engine with a generic message (metadata, arch-covers,
engine name collision) -- see ``Knowledge/hipdnn/ingestor/07-descriptor-
generation.md`` §3 and ``06-gotchas.md``.
"""

import warnings as _warnings
from pathlib import Path

import yaml

from .models import (
    ARCH_BASE_ID_PATTERN,
    DIALECT_DIRECT_LOAD,
    DIALECT_PACKAGED,
    DIALECTS,
    EMITTABLE_KINDS_BY_DIALECT,
    ENGINE_NAME_PATTERN,
    KERNEL_SOURCE_KIND_EMBEDDED,
    KERNEL_SOURCE_KIND_HIP,
    KERNEL_SOURCE_KIND_HSACO,
    KERNEL_SOURCE_KIND_HSACO_FILE,
    KERNEL_SOURCE_KIND_KPACK,
    KERNEL_SOURCE_KIND_ROCKE,
    KERNEL_SOURCE_KIND_ROCKE_BUILDER,
    KERNEL_SOURCE_KINDS,
    KMD_FIELD_TYPES,
    KNOWN_ARCH_BASE_IDS,
    WORKSPACE_POLICIES,
    EngineSpec,
    GraphMatchSpec,
    IngestorConfig,
    KernelSource,
    KernelSpec,
    KmdField,
    PackSpec,
)


class ConfigError(Exception):
    """Raised when a YAML config is invalid."""

    pass


def load_config(path: Path) -> IngestorConfig:
    """Load and validate a YAML config file, returning an ``IngestorConfig``.

    Raises ``ConfigError`` on any structural problem or failed pre-mint
    check. No UUID is minted here or anywhere reachable from here --
    minting happens only in ``generator.py``, after a config has fully
    survived this function.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: YAML document must be a top-level mapping.")

    _reject_deprecated_keys(raw)

    engine_raw = raw.get("engine")
    if not engine_raw or "name" not in engine_raw:
        raise ConfigError("Missing required field 'engine.name'.")

    engine = EngineSpec(
        name=engine_raw["name"],
        sdk_version=engine_raw.get("sdk_version", "1.0.0"),
        behavior_notes=list(engine_raw.get("behavior_notes", [])),
        knobs=list(engine_raw.get("knobs", [])),
        heuristic=engine_raw.get("heuristic", "native"),
    )

    kmd_fields = []
    for raw_field in raw.get("kmd_fields", []):
        for required in ("name", "type"):
            if required not in raw_field:
                raise ConfigError(
                    f"kmd_fields entry missing required key '{required}': {raw_field!r}"
                )
        kmd_fields.append(
            KmdField(
                name=raw_field["name"],
                type=raw_field["type"],
                default_value=raw_field.get("default_value"),
            )
        )

    packs = []
    for pack_raw in raw.get("packs", []):
        if "name" not in pack_raw:
            raise ConfigError(f"packs entry missing required key 'name': {pack_raw!r}")
        kernels = []
        for kernel_raw in pack_raw.get("kernels", []):
            for required in ("name", "kernel_source"):
                if required not in kernel_raw:
                    raise ConfigError(
                        f"pack '{pack_raw['name']}' kernel entry missing required key "
                        f"'{required}': {kernel_raw!r}"
                    )
            ks_raw = kernel_raw["kernel_source"]
            if "kind" not in ks_raw:
                raise ConfigError(
                    f"pack '{pack_raw['name']}' kernel '{kernel_raw['name']}': "
                    "kernel_source missing required key 'kind'."
                )
            kernels.append(
                KernelSpec(
                    name=kernel_raw["name"],
                    kernel_source=KernelSource(
                        kind=ks_raw["kind"],
                        source_file=ks_raw.get("source_file", ""),
                        entry_point=ks_raw.get("entry_point", ""),
                        source=ks_raw.get("source", ""),
                        entry=ks_raw.get("entry", ""),
                        build=dict(ks_raw.get("build", {})),
                        builder=ks_raw.get("builder", ""),
                        spec=dict(ks_raw.get("spec", {})),
                    ),
                    metadata=dict(kernel_raw.get("metadata", {})),
                    priority=kernel_raw.get("priority", 0),
                    arch=list(kernel_raw.get("arch", [])),
                )
            )
        packs.append(
            PackSpec(
                name=pack_raw["name"],
                kernels=kernels,
                arch=list(pack_raw.get("arch", [])),
                discriminator=pack_raw.get("discriminator", ""),
            )
        )

    gm_raw = raw.get("graph_match", {})
    graph_match = GraphMatchSpec(
        shape=gm_raw.get("shape", "shared_shape"),
        discriminator=gm_raw.get("discriminator", "none"),
    )

    config = IngestorConfig(
        engine=engine,
        kmd_fields=kmd_fields,
        packs=packs,
        graph_match=graph_match,
        dialect=raw.get("dialect", DIALECT_DIRECT_LOAD),
        kernel_source_kind=raw.get("kernel_source_kind", KERNEL_SOURCE_KIND_EMBEDDED),
        workspace_policy=raw.get("workspace_policy", "none"),
        delegates_to_existing_plan=bool(raw.get("delegates_to_existing_plan", False)),
        authored_subpath=raw.get("authored_subpath", ""),
    )

    _validate_config(config)

    return config


def _reject_deprecated_dict_key(
    raw_items: list, key: str, item_label_key: str, message: str
) -> None:
    """Raise ``ConfigError`` if any raw item dict contains the deprecated ``key``.

    Detection is key-*presence*, not value-truthy: ``optional: false`` still
    raises. Mirrors ``DescriptorGenerator``'s own convention.
    """
    rejected = [
        item.get(item_label_key, "<unnamed>") for item in raw_items if key in item
    ]
    if not rejected:
        return
    names = ", ".join(str(n) for n in rejected)
    raise ConfigError(f"{message} Affected entries: {names}.")


def _reject_deprecated_keys(raw: dict) -> None:
    """Reject YAML keys that look plausible -- often copied from RFC 0017's own
    worked examples -- but name nothing this loader or the runtime reads.

    Both rejected ``kmd_fields[]`` keys are exactly RFC 0017 §4's own example
    field (``{"name":"tile_m","type":"int","optional":true,"default":1}``),
    which ``Knowledge/hipdnn/ingestor/02-descriptor-format.md`` documents as
    rejected on both keys by the real loader -- only ``default_value`` is a
    real field, and there is no ``optional`` key at all, ever.
    """
    _reject_deprecated_dict_key(
        raw.get("kmd_fields", []),
        "optional",
        "name",
        "kmd_fields[].optional is not a real key -- RFC 0017 §4's own example "
        "field carries it, but the loader's MetadataField has no such member. "
        "A field is optional exactly when it has a default_value; there is no "
        "separate optional flag.",
    )
    _reject_deprecated_dict_key(
        raw.get("kmd_fields", []),
        "default",
        "name",
        "kmd_fields[].default is not a real key -- the loader spells it "
        "default_value, not default. Rename the key.",
    )
    if "schema" in raw:
        raise ConfigError(
            "Top-level 'schema' is not a real key. RFC 0020 §4.2 specifies a "
            "required 'schema' member on the UED (tag 'hipdnn.ued/v1'), but no "
            "shipped descriptor type on develop has ever carried one -- "
            "DescriptorLoader.hpp's parse*Descriptor() functions all reject it "
            "as an unknown key. Remove it."
        )


# ---------------------------------------------------------------------------
# The five config-loader pre-mint checks (run in order below, all before any
# UUID exists anywhere in this program).
# ---------------------------------------------------------------------------


def _check_engine_name_scoped(config: IngestorConfig) -> None:
    """Pre-mint check #1: engine.name matches the scoped namespace:local regex."""
    if not ENGINE_NAME_PATTERN.match(config.engine.name):
        raise ConfigError(
            f"engine.name '{config.engine.name}' must be scoped 'namespace:local' "
            f"(e.g. 'hipkernel:MyEngine'), matching "
            f"^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$. An unscoped name is exactly the "
            f"collision two vendors would both pick -- the loader hashes it "
            f"(FNV-1a) into the global 64-bit engine-id space and requires "
            f"global uniqueness."
        )
    if config.engine.heuristic not in ("native", "none"):
        raise ConfigError(
            f"engine.heuristic '{config.engine.heuristic}' must be 'native' "
            f"(emit a UHD) or 'none' (omit it -- legal; the engine falls back "
            f"to priority-then-id ranking)."
        )


def _check_knobs_int_typed(config: IngestorConfig) -> None:
    """Pre-mint check #2: every knob names a declared, int-typed KMD field.

    A non-int knob is accepted by the real loader and produces **no** knob
    at all, silently -- GenericPlanBuilder::getCustomKnobs filters to
    int64_t alternatives only (Knowledge/hipdnn/ingestor/06-gotchas.md
    "getCustomKnobs silently drops non-integer knobs"). This is the only
    point in the whole pipeline where that drop is generation-time
    reachable at all, per the plan's research findings.
    """
    declared = config.kmd_field_by_name
    for knob in config.engine.knobs:
        kmd_field = declared.get(knob)
        if kmd_field is None:
            raise ConfigError(
                f"engine.knobs names '{knob}', which no kmd_fields entry "
                f"declares. Every knob must name a declared KMD field."
            )
        if not kmd_field.is_int_typed:
            raise ConfigError(
                f"engine.knobs names '{knob}', declared in kmd_fields with "
                f"type '{kmd_field.type}'. Only int-typed fields become "
                f"usable knobs -- a non-int knob is accepted by the loader "
                f"and then silently produces no knob at all, with no error "
                f"and no warning, discovered only at plan-build time against "
                f"a real device (GenericPlanBuilder::getCustomKnobs). Retype "
                f"'{knob}' to 'int' or remove it from engine.knobs."
            )


_METADATA_TYPE_CHECKS = {
    "bool": lambda v: isinstance(v, bool),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "int_list": lambda v: isinstance(v, list)
    and all(isinstance(item, int) and not isinstance(item, bool) for item in v),
}


def _check_kernel_metadata_against_kmd(config: IngestorConfig) -> None:
    """Pre-mint check #3: every kernel's metadata type-checks against the KMD,
    with no mandatory field omitted.

    Mirrors the loader's own consequence exactly: "A wrong type, an
    undeclared field, or an omitted mandatory field drops the whole pack"
    (Knowledge/hipdnn/ingestor/02-descriptor-format.md, UKD key table).
    """
    declared = config.kmd_field_by_name
    for field_type in {f.type for f in config.kmd_fields}:
        if field_type not in KMD_FIELD_TYPES:
            raise ConfigError(
                f"kmd_fields declares type '{field_type}', which is not one of "
                f"{KMD_FIELD_TYPES}."
            )

    for pack in config.packs:
        for kernel in pack.kernels:
            where = f"pack '{pack.name}' kernel '{kernel.name}'"
            for key in kernel.metadata:
                if key not in declared:
                    raise ConfigError(
                        f"{where}: metadata names '{key}', which no kmd_fields "
                        f"entry declares. An undeclared metadata field drops "
                        f"the whole pack at resolveDescriptorSets()."
                    )
            for kmd_field in config.kmd_fields:
                if kmd_field.is_mandatory and kmd_field.name not in kernel.metadata:
                    raise ConfigError(
                        f"{where}: omits mandatory metadata field "
                        f"'{kmd_field.name}' (its kmd_fields entry has no "
                        f"default_value, so every kernel must supply it). An "
                        f"omitted mandatory field drops the whole pack."
                    )
                if kmd_field.name in kernel.metadata:
                    value = kernel.metadata[kmd_field.name]
                    check = _METADATA_TYPE_CHECKS[kmd_field.type]
                    if not check(value):
                        raise ConfigError(
                            f"{where}: metadata '{kmd_field.name}' = {value!r} "
                            f"does not match its declared kmd_fields type "
                            f"'{kmd_field.type}'."
                        )


def _check_kernel_arch_subset_of_pack(config: IngestorConfig) -> None:
    """Pre-mint check #4: a kernel's arch must be a subset of its pack's.

    Mirrors ``archCovers(pack.arch, kernel.arch)``, enforced at parse for an
    inline kernel (``DescriptorLoader.hpp:977-982``). An empty pack.arch
    covers everything (arch-independent); an empty kernel.arch inherits the
    pack's and is always covered.
    """
    for pack in config.packs:
        if not pack.arch:
            continue
        for kernel in pack.kernels:
            if not kernel.arch:
                continue
            reaching = [a for a in kernel.arch if a not in pack.arch]
            if reaching:
                raise ConfigError(
                    f"pack '{pack.name}' kernel '{kernel.name}' declares arch "
                    f"{kernel.arch}, which reaches past the pack's arch "
                    f"{pack.arch} (entries not covered: {reaching}). "
                    f"archCovers(pack.arch, kernel.arch) would fail this file "
                    f"at parse time with 'reaches past the pack's arch'."
                )


def _check_arch_shape(config: IngestorConfig) -> list[str]:
    """Pre-mint check #5: arch entries are plausible gfx-prefixed base ids.

    A shape violation (``GFX942``, ``" gfx942"``, a feature suffix) is a
    hard ``ConfigError`` -- mirrors the loader's own eager
    ``isPlausibleArchBaseId`` shape check (``DescriptorLoader.hpp:634-643``),
    which the real loader enforces at parse time for exactly this reason:
    match-time evidence (an ordinary INFO decline) is indistinguishable from
    a correctly-authored arch that legitimately excludes the running
    device. A well-formed but *unrecognized* id (e.g. ``gfx94``) is only a
    warning, per the design's own ruling -- it is not this tool's job to
    maintain an exhaustive, always-current arch list.

    Returns the list of warning messages emitted (also raised via
    ``warnings.warn`` for a caller who wants Python's own warning machinery).
    """
    messages: list[str] = []

    def check_list(archs: list, where: str) -> None:
        for arch in archs:
            if not ARCH_BASE_ID_PATTERN.match(arch):
                raise ConfigError(
                    f"{where} arch entry '{arch}' is not a plausible "
                    f"gfx-prefixed base id (lowercase 'gfx' + alnum/-/_, no "
                    f"feature suffix such as ':sramecc+', no leading/trailing "
                    f"whitespace). This mirrors the loader's own "
                    f"isPlausibleArchBaseId shape check -- a value failing it "
                    f"parses fine and then declines on every device, logging "
                    f"exactly what a healthy cross-arch install logs."
                )
            if arch not in KNOWN_ARCH_BASE_IDS:
                message = (
                    f"{where} arch entry '{arch}' is well-formed but not a "
                    f"recognized gfx target id (e.g. a typo like 'gfx94' for "
                    f"'gfx942'). It will parse and load fine, then decline on "
                    f"every real device, logged as an ordinary INFO decline "
                    f"indistinguishable from a deliberate arch exclusion."
                )
                messages.append(message)
                _warnings.warn(message, UserWarning, stacklevel=3)

    for pack in config.packs:
        check_list(pack.arch, f"pack '{pack.name}'")
        for kernel in pack.kernels:
            check_list(kernel.arch, f"pack '{pack.name}' kernel '{kernel.name}'")

    return messages


def _check_dialect(config: IngestorConfig) -> None:
    """The dialect is one of the two, and a packaged bundle names its arch.

    ``hkp_pack._validate_kdp`` REQUIRES ``arch`` on every KDP, where the
    runtime loader treats absence as a wildcard. Catching it here names the
    cause; letting it through produces a missing-key error from the packager
    about a file this tool wrote.
    """
    if config.dialect not in DIALECTS:
        raise ConfigError(
            f"dialect '{config.dialect}' must be one of {DIALECTS}. "
            f"'{DIALECT_DIRECT_LOAD}' emits descriptors the runtime loader reads "
            f"straight out of the provider's descriptors/ tree; "
            f"'{DIALECT_PACKAGED}' emits descriptors hkp_pack compiles/lowers "
            f"into a per-arch .kpack archive at build time."
        )
    if not config.is_packaged:
        return
    for pack in config.packs:
        if not pack.arch:
            raise ConfigError(
                f"pack '{pack.name}' declares no 'arch', which the packaged "
                f"dialect requires: hkp_pack validates every KDP for a non-empty "
                f"arch list and uses it to decide which per-arch shard the "
                f"descriptor ships in. (The runtime loader is laxer -- it reads "
                f"an absent arch as a wildcard -- but a packaged descriptor is "
                f"read by the packager first.)"
            )


def _check_kernel_source_kind_implemented(config: IngestorConfig) -> None:
    """Reject a ``kernel_source.kind`` the configured dialect cannot emit.

    Each rejection names the dialect, because the common mistake is a kind
    that IS real but belongs to the other dialect -- a diagnostic saying
    'unsupported' would send an author looking for a missing feature instead
    of a one-line ``dialect:`` change.
    """
    emittable = EMITTABLE_KINDS_BY_DIALECT[config.dialect]
    for kind_source, where in [(config.kernel_source_kind, "kernel_source_kind")] + [
        (
            kernel.kernel_source.kind,
            f"pack '{pack.name}' kernel '{kernel.name}'.kernel_source.kind",
        )
        for pack in config.packs
        for kernel in pack.kernels
    ]:
        if kind_source not in KERNEL_SOURCE_KINDS:
            raise ConfigError(
                f"{where} '{kind_source}' is not a recognized kernel_source kind. "
                f"Recognized: {', '.join(KERNEL_SOURCE_KINDS)}."
            )
        if kind_source in emittable:
            continue

        if kind_source in (KERNEL_SOURCE_KIND_HSACO_FILE, KERNEL_SOURCE_KIND_HSACO):
            raise ConfigError(
                f"{where} is '{kind_source}', which no adapter implements on "
                f"either path. The runtime needs supportsSourceKind() on "
                f"IKernelDispatchHandler -- a shared-SDK interface change that "
                f"does not exist on develop and wants its own sign-off."
            )
        if kind_source == KERNEL_SOURCE_KIND_KPACK:
            raise ConfigError(
                f"{where} is 'kpack', which is a PRODUCED kind, never an "
                f"authored one. hkp_pack writes it -- stamping library, "
                f"toc_key, symbol and sha256 from the artifact it actually "
                f"built -- when it lowers a 'hip' or 'rocke' descriptor. "
                f"Authoring those four by hand would be a second source of "
                f"truth that silently disagrees with the archive. Author "
                f"'{KERNEL_SOURCE_KIND_ROCKE}' or '{KERNEL_SOURCE_KIND_HIP}' "
                f"under dialect '{DIALECT_PACKAGED}' instead."
            )
        if kind_source == KERNEL_SOURCE_KIND_ROCKE_BUILDER:
            raise ConfigError(
                f"{where} is 'rocke_builder', the runtime enum spelling, which "
                f"the loader parses and nothing dispatches. A rocKE kernel never "
                f"reaches the runtime as rocKE: hkp_pack lowers it through comgr "
                f"at build time and rewrites the shipped descriptor to 'kpack'. "
                f"Author kind '{KERNEL_SOURCE_KIND_ROCKE}' under dialect "
                f"'{DIALECT_PACKAGED}'."
            )
        # A real kind, wrong dialect -- the most likely mistake, so say exactly
        # which one-line change fixes it.
        other = (
            DIALECT_PACKAGED
            if config.dialect == DIALECT_DIRECT_LOAD
            else DIALECT_DIRECT_LOAD
        )
        if kind_source in EMITTABLE_KINDS_BY_DIALECT[other]:
            raise ConfigError(
                f"{where} is '{kind_source}', which belongs to dialect "
                f"'{other}', but this config declares dialect "
                f"'{config.dialect}' (which emits "
                f"{', '.join(emittable)}). Set 'dialect: {other}', or use one "
                f"of this dialect's kinds."
            )
        raise ConfigError(
            f"{where} is '{kind_source}', which dialect '{config.dialect}' "
            f"cannot emit (it emits {', '.join(emittable)})."
        )


def _check_kernel_source_fields(config: IngestorConfig) -> None:
    """Each kernel supplies its kind's own fields, and no other kind's.

    Mirrors ``hkp_pack._validate_ukd_fields``, which requires exactly these
    per kind. Checked here so an author sees it before a comgr run rather
    than after.
    """
    required_by_kind = {
        KERNEL_SOURCE_KIND_EMBEDDED: ("source_file", "entry_point"),
        KERNEL_SOURCE_KIND_HIP: ("source", "entry"),
        KERNEL_SOURCE_KIND_ROCKE: ("source", "builder", "spec"),
    }
    for pack in config.packs:
        for kernel in pack.kernels:
            where = f"pack '{pack.name}' kernel '{kernel.name}'.kernel_source"
            ks = kernel.kernel_source
            for attr in required_by_kind.get(ks.kind, ()):
                if not getattr(ks, attr):
                    raise ConfigError(
                        f"{where} is kind '{ks.kind}' but supplies no "
                        f"'{attr}'. Kind '{ks.kind}' requires "
                        f"{', '.join(required_by_kind[ks.kind])}."
                    )
            if ks.kind == KERNEL_SOURCE_KIND_ROCKE and not isinstance(ks.spec, dict):
                raise ConfigError(f"{where}: 'spec' must be a mapping.")


def _check_workspace_policy(config: IngestorConfig) -> None:
    if config.workspace_policy not in WORKSPACE_POLICIES:
        raise ConfigError(
            f"workspace_policy '{config.workspace_policy}' must be one of "
            f"{WORKSPACE_POLICIES}."
        )


def _check_pack_discriminators(config: IngestorConfig) -> None:
    """A multi-pack engine needs a discriminator per pack to name its
    operation-scoped matcher symbol; a single-pack engine must not declare
    one (there is nothing to discriminate -- see the UMD policy)."""
    if config.is_multi_pack:
        missing = [p.name for p in config.packs if not p.discriminator]
        if missing:
            raise ConfigError(
                f"packs {missing} declare no 'discriminator', but this engine "
                f"has {len(config.packs)} packs. Every pack needs a "
                f"discriminator to name its own operation-scoped matcher "
                f"symbol (e.g. 'add' -> '<engine>.add_match')."
            )
        names = [p.discriminator for p in config.packs]
        if len(names) != len(set(names)):
            raise ConfigError(f"packs declare duplicate discriminators: {names}.")
    else:
        for pack in config.packs:
            if pack.discriminator:
                raise ConfigError(
                    f"pack '{pack.name}' declares a discriminator, but this "
                    f"engine has only one pack. A single-pack engine's "
                    f"graph_match both admits the node type and validates "
                    f"shape in one pass -- it needs no operation-scoped "
                    f"matcher, and TestConvFwdPack.cpp asserts exactly zero "
                    f"graph-scoped matchers for this shape. Remove the "
                    f"discriminator."
                )
    if not config.packs:
        raise ConfigError("packs must declare at least one pack.")
    for pack in config.packs:
        if not pack.kernels:
            raise ConfigError(f"pack '{pack.name}' declares no kernels.")


def _validate_config(config: IngestorConfig) -> list[str]:
    """Run every pre-mint check, in the order the design lists them.

    Returns any non-fatal warning messages (currently only from check #5).
    """
    # The dialect decides which kinds are legal, so it is settled first --
    # every kind diagnostic below names it.
    _check_dialect(config)

    _check_engine_name_scoped(config)  # #1
    _check_knobs_int_typed(config)  # #2
    _check_kernel_metadata_against_kmd(config)  # #3
    _check_kernel_arch_subset_of_pack(config)  # #4
    warnings_out = _check_arch_shape(config)  # #5

    # Additional structural checks needed for a config to generate at all;
    # not among the five loader-mirroring checks, but still pre-mint.
    _check_kernel_source_kind_implemented(config)
    _check_kernel_source_fields(config)
    _check_workspace_policy(config)
    _check_pack_discriminators(config)

    for note in config.engine.behavior_notes:
        from .models import BEHAVIOR_NOTES

        if note not in BEHAVIOR_NOTES:
            raise ConfigError(
                f"engine.behavior_notes names '{note}', which is not in the "
                f"closed vocabulary {BEHAVIOR_NOTES}. The loader hard-rejects "
                f"anything else."
            )

    return warnings_out
