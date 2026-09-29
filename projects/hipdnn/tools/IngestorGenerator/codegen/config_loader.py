# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""YAML config loading and validation for the generic-kernel-ingestor generator.

Every check runs before ``generator.py`` mints a UUID. The five numbered checks
catch failures ``DescriptorLoader.hpp`` either accepts silently or reports only
after dropping a whole pack or engine.
"""

import gzip
import itertools
import posixpath
import re
import warnings as _warnings
from collections.abc import Hashable
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

from .models import (
    ARCH_BASE_ID_PATTERN,
    AUTHORED_TEST_SETS,
    CXX_IDENTIFIER_PATTERN,
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
    PATH_STEM_PATTERN,
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


#: Stands in for every ``<<`` key while explicit keys are compared, so two merge
#: declarations collide with each other and never with an authored ``"<<"`` string.
_MERGE_KEY = object()


class _DuplicateKeySafeLoader(yaml.SafeLoader):
    """``SafeLoader`` that refuses a mapping declaring the same key twice.

    Stock construction keeps the last of two equal keys and drops the first
    without a word, so a key repeated while editing a long config silently
    replaces the value the author sees first. Each mapping node's own authored
    keys, ``<<`` included, are compared once, before ``SafeLoader.flatten_mapping``
    splices merged keys into it: an explicit key overriding a merged one, and
    precedence within one ``<<`` sequence, stay stock behaviour. Keys are built
    by the stock constructor, so two spellings YAML resolves to one value (``0x10``
    and ``16``) collide, and ``"off"`` and ``off`` (a string and ``False``) do not.
    """

    def __init__(self, stream):
        super().__init__(stream)
        # A merge source reached through several aliases is one node, already
        # flattened after its first visit; comparing it again would treat its
        # inherited keys as authored ones.
        self._keys_compared = set()

    def flatten_mapping(self, node):
        if node not in self._keys_compared:
            self._keys_compared.add(node)
            self._reject_repeated_keys(node)
        super().flatten_mapping(node)

    def _reject_repeated_keys(self, node) -> None:
        first_seen = {}
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                key, shown = _MERGE_KEY, "<<"
            elif key_node.tag == "tag:yaml.org,2002:value":
                # Stock flattening retags a plain ``=`` key as a string before
                # construction; no constructor exists for the value tag itself.
                key = shown = self.construct_scalar(key_node)
            else:
                key = shown = self.construct_object(key_node)
            if not isinstance(key, Hashable):
                # Stock construction reports the unhashable key itself.
                continue
            first, first_shown = first_seen.setdefault(key, (key_node, shown))
            if first is not key_node:
                raise yaml.constructor.ConstructorError(
                    f"while constructing a mapping, key {first_shown!r} is first "
                    f"declared",
                    first.start_mark,
                    f"found duplicate key {shown!r}; YAML would silently keep only "
                    f"this later value",
                    key_node.start_mark,
                )


#: Private, never authored: an expanded kernel dict carries the
#: ``(where, mapping)`` of the authored ``kernel_source`` it was built from, so
#: the closed per-kind vocabulary is applied to the author's keys and reported
#: where the author can edit. Written by `_expand_axis_kernels` and
#: `_expand_one_arm`; read and dropped by `load_config`.
_AUTHORED_KERNEL_SOURCE = "_authored_kernel_source"

#: Private, never authored: the ``kernel_defaults`` keys an expander consumed
#: while building this kernel. A consumed key is read by the expander rather
#: than written into ``kernel_source``, so the closed per-kind vocabulary does
#: not apply to it. Written by `_expand_one_arm`; read by `load_config`.
_EXPANDER_CONSUMED_DEFAULTS = "_expander_consumed_defaults"


def _unique_arch(raw_arch, where: str) -> list[str]:
    """Collapse repeats in an arch list, preserving authored order, which
    reaches the descriptor bytes. The shape guard runs first: ``dict.fromkeys``
    accepts a bare string and collapses it to characters."""
    return list(dict.fromkeys(_require_sequence(raw_arch, where, what="arch ids")))


def read_yaml(path: Path) -> Any:
    """Parse one YAML file and return the document unvalidated.

    An empty file returns ``None``. A ``.gz`` path is decompressed transparently.
    Input that is not safe YAML, or a mapping declaring one key twice, raises the
    parser's ``yaml.YAMLError`` carrying the source marks.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        # _DuplicateKeySafeLoader subclasses yaml.SafeLoader; safe_load takes no Loader.
        return yaml.load(f, Loader=_DuplicateKeySafeLoader)  # nosec B506


def load_config(path: Path) -> IngestorConfig:
    """Load and validate a YAML config file, returning an ``IngestorConfig``.

    Parses with ``read_yaml``, so its ``yaml.YAMLError`` propagates. Raises
    ``ConfigError`` on any structural problem or failed pre-mint check; no UUID
    is minted here.
    """
    raw = read_yaml(path)

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: YAML document must be a top-level mapping.")

    # Shapes first: both key walks below index into these containers, and a
    # scalar where a mapping belongs raises an uncaught AttributeError.
    _require_config_shapes(raw)
    # Deprecated before unknown: `optional`, `default` and `schema` are also
    # unknown keys, so the generic diagnostic would bury the specific one.
    _reject_deprecated_keys(raw)
    _reject_unknown_keys(raw)

    engine_raw = raw.get("engine")
    if not engine_raw or "name" not in engine_raw:
        raise ConfigError("Missing required field 'engine.name'.")

    engine = EngineSpec(
        name=engine_raw["name"],
        sdk_version=engine_raw.get("sdk_version", "1.0.0"),
        behavior_notes=_require_sequence(
            engine_raw.get("behavior_notes"), "engine.behavior_notes"
        ),
        knobs=_require_sequence(engine_raw.get("knobs"), "engine.knobs"),
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

    kmd_field_names = {f.name for f in kmd_fields}

    packs = []
    for pack_raw in raw.get("packs", []):
        if "name" not in pack_raw:
            raise ConfigError(f"packs entry missing required key 'name': {pack_raw!r}")
        # A pack-level `kernel_defaults` is merged UNDER each kernel's own
        # `kernel_source`, so a kernel overrides it by simply restating the key.
        defaults = pack_raw.get("kernel_defaults", {}) or {}
        _require_mapping(defaults, f"pack '{pack_raw['name']}' kernel_defaults")
        # One key below the mapping just guarded, and merged the same way: a
        # scalar here reaches `dict("oops")`, whose ValueError names neither
        # the pack nor the key.
        if "spec" in defaults:
            _require_mapping(
                defaults["spec"], f"pack '{pack_raw['name']}' kernel_defaults.spec"
            )
        default_spec = dict(defaults.get("spec", {}))
        # `kernel_defaults` collapses repetition across kernels; `axes`
        # collapses the multiplicative repetition within a knob-driven variant
        # set. See `_expand_axis_kernels`.
        axis_kernels_raw = _expand_axis_kernels(pack_raw, kmd_field_names)
        # `variants` collapses the other shape: a set where every shape carries
        # its own dispatcher-resolved spec, with no single kernel_template for
        # `axes` to cross. See `_expand_variant_kernels`.
        variant_kernels_raw = _expand_variant_kernels(
            pack_raw, {f.name: f.type for f in kmd_fields}
        )
        kernels = []
        for kernel_raw in (
            list(pack_raw.get("kernels", [])) + axis_kernels_raw + variant_kernels_raw
        ):
            for required in ("name", "kernel_source"):
                if required not in kernel_raw:
                    raise ConfigError(
                        f"pack '{pack_raw['name']}' kernel entry missing required key "
                        f"'{required}': {kernel_raw!r}"
                    )
            # Type-guard before the dict merges below: `dict("oops")` raises an
            # uncaught ValueError from deep inside the merge, naming neither
            # the kernel nor the key.
            _require_mapping(
                kernel_raw["kernel_source"],
                f"pack '{pack_raw['name']}' kernel '{kernel_raw['name']}' "
                f"kernel_source",
            )
            for key in ("spec", "build"):
                if key in kernel_raw["kernel_source"]:
                    _require_mapping(
                        kernel_raw["kernel_source"][key],
                        f"pack '{pack_raw['name']}' kernel '{kernel_raw['name']}' "
                        f"kernel_source.{key}",
                    )
            if "metadata" in kernel_raw:
                _require_mapping(
                    kernel_raw["metadata"],
                    f"pack '{pack_raw['name']}' kernel '{kernel_raw['name']}' metadata",
                )
            ks_raw = {**defaults, **kernel_raw["kernel_source"]}
            ks_raw.pop("spec", None)
            ks_raw["spec"] = {
                **default_spec,
                **dict(kernel_raw["kernel_source"].get("spec", {})),
            }
            if "kind" not in ks_raw:
                raise ConfigError(
                    f"pack '{pack_raw['name']}' kernel '{kernel_raw['name']}': "
                    "kernel_source missing required key 'kind' and the pack declares "
                    "no kernel_defaults.kind."
                )
            # The one gate every kernel reaches, hand-authored or expanded.
            # An expanded kernel's `kernel_source` is generated, so it names
            # the authored mapping instead -- see `_AUTHORED_KERNEL_SOURCE`.
            authored_where, authored_source = kernel_raw.get(
                _AUTHORED_KERNEL_SOURCE,
                (
                    f"pack '{pack_raw['name']}' kernel "
                    f"'{kernel_raw['name']}' kernel_source",
                    kernel_raw["kernel_source"],
                ),
            )
            # A key an expander consumed is not a kernel_source key of this
            # kernel -- see `_EXPANDER_CONSUMED_DEFAULTS`.
            consumed = kernel_raw.get(_EXPANDER_CONSUMED_DEFAULTS, frozenset())
            authored_defaults = {
                key: value for key, value in defaults.items() if key not in consumed
            }
            # The defaults are a pack-level block judged against one kernel's
            # merged kind, so the diagnostic names both: the block alone may be
            # correct for the kernel it was written for.
            for source_keys, source_where in (
                (
                    authored_defaults,
                    f"pack '{pack_raw['name']}' kernel_defaults (as merged for "
                    f"kernel '{kernel_raw['name']}')",
                ),
                (authored_source, authored_where),
            ):
                _check_kernel_source_keys(source_keys, ks_raw["kind"], source_where)
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
                    arch=_unique_arch(
                        kernel_raw.get("arch", []),
                        f"pack '{pack_raw['name']}' kernel "
                        f"'{kernel_raw['name']}' arch",
                    ),
                )
            )
        packs.append(
            PackSpec(
                name=pack_raw["name"],
                kernels=kernels,
                arch=_unique_arch(
                    pack_raw.get("arch", []), f"pack '{pack_raw['name']}' arch"
                ),
                discriminator=pack_raw.get("discriminator", ""),
            )
        )

    # Once over the whole engine rather than per pack: the scope of the check
    # is the scope of the identity it protects.
    _check_kernel_names_unique(packs)

    gm_raw = raw.get("graph_match") or {}
    _require_mapping(gm_raw, "graph_match")
    graph_match = GraphMatchSpec(
        shape=gm_raw.get("shape", "shared_shape"),
        discriminator=gm_raw.get("discriminator", "none"),
    )

    # Before the `dict()` below, which would raise a bare "dictionary update
    # sequence" ValueError from a scalar and preempt the shaped diagnostic.
    specialization_raw = raw.get("specialization") or {}
    _require_specialization_mapping(specialization_raw)

    config = IngestorConfig(
        engine=engine,
        kmd_fields=kmd_fields,
        packs=packs,
        graph_match=graph_match,
        dialect=raw.get("dialect", DIALECT_DIRECT_LOAD),
        kernel_source_kind=raw.get("kernel_source_kind", KERNEL_SOURCE_KIND_EMBEDDED),
        workspace_policy=raw.get("workspace_policy", "none"),
        authored_subpath=raw.get("authored_subpath", ""),
        specialization=dict(specialization_raw),
    )

    _validate_config(config)

    return config


def _check_kernel_names_unique(packs: list) -> None:
    """Require a unique kernel name per engine, hand-authored or expanded.

    Engine-scoped because the loader collects an engine's packs into one
    ``DescriptorSet`` by engine id, and de-duplication keys on metadata rather
    than the name.
    """
    packs_by_name: dict = {}
    for pack in packs:
        for kernel in pack.kernels:
            packs_by_name.setdefault(kernel.name, []).append(pack.name)
    collisions = {
        name: where for name, where in packs_by_name.items() if len(where) > 1
    }
    if not collisions:
        return
    # The pack list is de-duplicated for display while the count is not: a name
    # twice in one pack reads "x2 in pack(s) 'p'", which says both halves.
    shown = ", ".join(
        f"{name!r} x{len(where)} in pack(s) "
        + ", ".join(repr(p) for p in dict.fromkeys(where))
        for name, where in sorted(collisions.items())[:3]
    )
    more = f" (+{len(collisions) - 3} more)" if len(collisions) > 3 else ""
    raise ConfigError(
        f"this engine declares {len(collisions)} duplicated kernel name(s): "
        f"{shown}{more}. Kernel names must be unique across every pack of one "
        f"engine: nothing downstream catches a collision, so the entries ship as "
        f"descriptors that cannot be told apart in a log or a failure message. If "
        f"these came from a 'variants' group, its name template omits a field the "
        f"shapes differ in -- add that field to the template, or a per-arm 'tag' "
        f"that distinguishes them."
    )


def _expand_axis_kernels(pack_raw: dict, kmd_field_names: set) -> list:
    """Expand a pack's ``axes`` cross-product into ordinary kernel dicts at load
    time, indistinguishable downstream from hand-authored ones. Returns ``[]``
    if the pack declares no ``axes``."""
    axes_raw = pack_raw.get("axes")
    pack_name = pack_raw.get("name", "<unnamed>")
    template = pack_raw.get("kernel_template")
    if not axes_raw:
        if template:
            raise ConfigError(
                f"pack '{pack_name}' declares 'kernel_template' but no 'axes'. "
                f"kernel_template only has effect as the cross-product source for "
                f"axis expansion -- with no axes there is nothing to expand it "
                f"against, and it would silently produce zero kernels. Add an "
                f"'axes' mapping, or author the kernel directly under 'kernels' "
                f"and remove kernel_template."
            )
        return []
    if not isinstance(axes_raw, dict):
        raise ConfigError(
            f"pack '{pack_name}' 'axes' must be a mapping of axis name to a "
            f"non-empty list of values, got {axes_raw!r}."
        )
    if not template or "kernel_source" not in template:
        raise ConfigError(
            f"pack '{pack_name}' declares 'axes' but no 'kernel_template' with a "
            f"'kernel_source' -- axis expansion needs exactly one template kernel "
            f"to vary; without it there is nothing to cross the axes against."
        )
    _require_mapping(
        template["kernel_source"], f"pack '{pack_name}' kernel_template kernel_source"
    )

    # Sorted once: naming and value-list order below both walk axis names in
    # this fixed order, which makes the encoded name a deterministic function
    # of the combination rather than of dict iteration order.
    axis_names = sorted(axes_raw)
    for axis_name in axis_names:
        if axis_name not in kmd_field_names:
            raise ConfigError(
                f"pack '{pack_name}' axes names '{axis_name}', which no "
                f"kmd_fields entry declares. An axis expands into per-kernel "
                f"metadata, and an undeclared metadata field drops the whole "
                f"pack at resolveDescriptorSets() -- the same failure pre-mint "
                f"check #3 guards against for a hand-authored kernel; expansion "
                f"must not manufacture a config that check would have rejected "
                f"if written out by hand."
            )
        values = axes_raw[axis_name]
        if not isinstance(values, list) or not values:
            raise ConfigError(
                f"pack '{pack_name}' axis '{axis_name}' must be a non-empty "
                f"list of values, got {values!r}. An empty axis's cross-product "
                f"is empty, which would silently expand this pack to ZERO "
                f"kernels instead of failing loudly."
            )
        if len(values) == 1:
            message = (
                f"pack '{pack_name}' axis '{axis_name}' has a single value "
                f"{values!r}. A one-valued axis contributes nothing to the "
                f"cross-product -- it is enumeration wearing a costume, and "
                f"usually means either a typo (a second intended value never "
                f"added) or a value that belongs in kernel_defaults instead."
            )
            _warnings.warn(message, UserWarning, stacklevel=3)

    template_name = template.get("name", pack_name)
    template_spec = dict(template["kernel_source"].get("spec", {}) or {})
    template_metadata = dict(template.get("metadata", {}) or {})
    value_lists = [axes_raw[name] for name in axis_names]

    expanded = []
    for combo in itertools.product(*value_lists):
        axis_values = dict(zip(axis_names, combo))
        # The name encodes every axis value in this fixed order, making each
        # cross-product entry's name an injective function of its combination.
        suffix = "_".join(f"{name}{axis_values[name]}" for name in axis_names)
        kernel_name = f"{template_name}.{suffix}"

        kernel_source = {
            key: value
            for key, value in template["kernel_source"].items()
            if key != "spec"
        }
        spec = dict(template_spec)
        for axis_name, value in axis_values.items():
            spec.setdefault(axis_name, value)
        # Written for every kind and owned by `rocke` alone: a kind that reads
        # no spec drops it when the descriptor is written. This generated key
        # must not be read back as authored -- see `_AUTHORED_KERNEL_SOURCE`.
        kernel_source["spec"] = spec

        metadata = dict(template_metadata)
        for axis_name, value in axis_values.items():
            metadata.setdefault(axis_name, value)

        expanded.append(
            {
                "name": kernel_name,
                "kernel_source": kernel_source,
                "metadata": metadata,
                "priority": template.get("priority", 0),
                "arch": list(template.get("arch", [])),
                _AUTHORED_KERNEL_SOURCE: (
                    f"pack '{pack_name}' kernel_template kernel_source",
                    template["kernel_source"],
                ),
            }
        )
    return expanded


def _require_sequence(value, scope: str, what: str = "field names") -> list:
    """Require a list of names: a bare string is valid YAML and iterates as
    characters, so ``policy_knobs: use_exp2_fast`` is never recognised."""
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ConfigError(
            f"{scope} must be a list of {what}; got "
            f"{type(value).__name__} ({value!r}). A bare string iterates as "
            f"characters, which silently does nothing."
        )
    return list(value)


def _drop_tag_slot(match) -> str:
    """Replace a `{tag}` slot and ONE adjacent separator when the tag is empty.

    `_x_` -> `_`, `_x` -> `` , `x_` -> ``, `x` -> ``. Keeping one separator when the
    slot sits between two rendered parts is what stops `a_{tag}_b` becoming `ab`.
    """
    text = match.group(0)
    leading = text.startswith("_")
    trailing = text.endswith("_")
    return "_" if leading and trailing else ""


#: A ``knob_set`` arm's own keys, which are NOT spec fields. Everything else in
#: an arm is written into the kernel's ``kernel_source.spec``.
_ARM_CONTROL_KEYS = frozenset({"tag", "ordinal_offset", "metadata"})
#: A shape entry's own keys, likewise not spec fields.
_SHAPE_CONTROL_KEYS = frozenset({"knobs", "resolved", "ordinal"})


def _expand_variant_kernels(pack_raw: dict, kmd_field_names: dict) -> list:
    """Expand a pack's ``variants`` groups into ordinary kernel dicts at load
    time, crossing each shape list with a named knob set. Returns ``[]`` if the
    pack declares no ``variants``.

    A knob absent from an arm is absent from the emitted
    ``kernel_source.spec``, meaning the kernel's policy decides it at build
    time -- not the same as pinning ``false``, though both reach metadata as
    ``0`` -- so the shape records the policy's answer under ``resolved``.
    """
    groups = pack_raw.get("variants")
    pack_name = pack_raw.get("name", "<unnamed>")
    if not groups:
        return []
    if not isinstance(groups, list):
        raise ConfigError(
            f"pack '{pack_name}' 'variants' must be a list of groups, got "
            f"{type(groups).__name__}."
        )
    expanded = []
    for position, group in enumerate(groups):
        expanded.extend(
            _expand_one_variant_group(
                group,
                position,
                pack_name,
                kmd_field_names,
                (pack_raw.get("kernel_defaults") or {}).get("spec") or {},
            )
        )
    return expanded


def _expand_one_variant_group(
    group: dict,
    position: int,
    pack_name: str,
    kmd_field_names: dict,
    pack_spec_defaults: dict | None = None,
) -> list:
    """One ``variants[]`` group -> the kernel dicts it stands for."""
    where = f"pack '{pack_name}' variants[{position}]"
    if not isinstance(group, dict):
        raise ConfigError(f"{where} must be a mapping, got {type(group).__name__}.")
    unknown = sorted(set(group) - _KNOWN_VARIANT_GROUP)
    if unknown:
        raise ConfigError(
            f"{where} declares {unknown}, which this loader does not read. Known "
            f"keys: {sorted(_KNOWN_VARIANT_GROUP)}."
        )
    for required in ("name", "metadata", "knob_sets", "shapes"):
        if required not in group:
            raise ConfigError(f"{where} is missing required key '{required}'.")

    name_template = group["name"]
    metadata_fields = list(group["metadata"])
    vocabulary = dict(group.get("vocabulary") or {})
    # Knobs the kernel's own policy decides when the spec leaves them absent.
    # Naming them makes "absent" legible as a third state and obliges each
    # shape to state what the policy resolved to.
    policy_knobs = set(
        _require_sequence(group.get("policy_knobs"), f"{where} policy_knobs")
    )
    # A key naming a field the group does not emit has no effect: a mistyped
    # `vocabulary` entry leaves the builder's spelling in the metadata, which
    # loads cleanly, reconciles on every count, and matches nothing.
    for label, names in (
        ("vocabulary", sorted(vocabulary)),
        ("policy_knobs", sorted(policy_knobs)),
    ):
        stray = [n for n in names if n not in metadata_fields]
        if stray:
            raise ConfigError(
                f"{where} {label} names {stray}, which this group's 'metadata' list "
                f"does not carry, so it would have no effect. Declared metadata: "
                f"{metadata_fields}."
            )
    spec_order = list(_require_sequence(group.get("spec_order"), f"{where} spec_order"))
    # Spec fields constant across this group. Distinct from pack-level
    # `kernel_defaults.spec` because the emitted spec's key order differs
    # between groups of one set, and order is part of the descriptor bytes.
    if group.get("spec_defaults") is not None:
        _require_mapping(group["spec_defaults"], f"{where} spec_defaults")
    spec_defaults = {
        **(pack_spec_defaults or {}),
        **dict(group.get("spec_defaults") or {}),
    }
    for field_name, mapping in vocabulary.items():
        _require_mapping(mapping, f"{where} vocabulary['{field_name}']")

    undeclared = sorted(set(metadata_fields) - set(kmd_field_names))
    if undeclared:
        raise ConfigError(
            f"{where} lists {undeclared} in 'metadata', which no kmd_fields entry "
            f"declares. An undeclared metadata field drops the WHOLE pack at "
            f"resolveDescriptorSets(), so expansion must not manufacture one."
        )

    knob_sets = group["knob_sets"]
    if not isinstance(knob_sets, dict) or not knob_sets:
        raise ConfigError(
            f"{where} 'knob_sets' must be a non-empty mapping of set name to a "
            f"non-empty list of arms."
        )
    for set_name, arms in knob_sets.items():
        if not isinstance(arms, list) or not arms:
            raise ConfigError(
                f"{where} knob_set '{set_name}' must be a non-empty list of arms, "
                f"got {arms!r}. An empty set expands its shapes to ZERO kernels "
                f"instead of failing here."
            )

    expanded = []
    for index, shape in enumerate(group["shapes"]):
        shape_where = f"{where} shapes[{index}]"
        if not isinstance(shape, dict):
            raise ConfigError(
                f"{shape_where} must be a mapping, got {type(shape).__name__}."
            )
        set_name = shape.get("knobs")
        if set_name not in knob_sets:
            raise ConfigError(
                f"{shape_where} names knob_set {set_name!r}, which this group does "
                f"not declare. Declared: {sorted(knob_sets)}."
            )
        # A near-miss control key (`ordinl` for `ordinal`) would otherwise fall
        # through into the spec, changing the binary the descriptor names while
        # the config still loads. Only checkable where the group declares
        # `spec_order`, which enumerates its spec fields.
        if spec_order:
            allowed = set(spec_order) | set(spec_defaults) | _SHAPE_CONTROL_KEYS
            stray = sorted(k for k in shape if k not in allowed)
            if stray:
                raise ConfigError(
                    f"{shape_where} declares {stray}, which is neither a control key "
                    f"({sorted(_SHAPE_CONTROL_KEYS)}) nor a field named in this "
                    f"group's spec_order. A misspelled control key becomes a spec "
                    f"field silently, which changes the binary the descriptor names."
                )
        if shape.get("resolved") is not None:
            _require_mapping(shape["resolved"], f"{shape_where} resolved")
        resolved = dict(shape.get("resolved") or {})
        ordinal = shape.get("ordinal", 0)
        shape_spec = {
            **spec_defaults,
            **{
                key: value
                for key, value in shape.items()
                if key not in _SHAPE_CONTROL_KEYS
            },
        }
        for arm in knob_sets[set_name]:
            expanded.append(
                _expand_one_arm(
                    arm,
                    shape_spec,
                    resolved,
                    ordinal,
                    name_template,
                    metadata_fields,
                    vocabulary,
                    policy_knobs,
                    spec_order,
                    shape_where,
                    kmd_field_names,
                )
            )
    return expanded


def _expand_one_arm(
    arm: dict,
    shape_spec: dict,
    resolved: dict,
    ordinal: int,
    name_template: str,
    metadata_fields: list,
    vocabulary: dict,
    policy_knobs: set,
    spec_order: list,
    shape_where: str,
    kmd_types: dict,
) -> dict:
    """One (shape, arm) pair -> one kernel dict."""
    if not isinstance(arm, dict):
        raise ConfigError(
            f"{shape_where}: every knob_set arm must be a mapping, got "
            f"{type(arm).__name__}."
        )
    # Only checkable where the group declares `spec_order`, which is what enumerates
    # its spec fields; `shape_spec` carries the group's and pack's spec defaults.
    if spec_order:
        allowed = set(spec_order) | set(shape_spec) | _ARM_CONTROL_KEYS
        stray = sorted(key for key in arm if key not in allowed)
        if stray:
            raise ConfigError(
                f"{shape_where}: a knob_set arm declares {stray}, which is neither "
                f"a control key ({sorted(_ARM_CONTROL_KEYS)}) nor a field named in "
                f"this group's spec_order. A misspelled control key becomes a spec "
                f"field silently, which changes the binary the descriptor names."
            )
    if arm.get("metadata") is not None:
        # A group's `metadata` is a list of field names while an arm's is a
        # mapping of field to value: same key, one level apart.
        _require_mapping(arm["metadata"], f"{shape_where}: knob_set arm metadata")
    arm_metadata = dict(arm.get("metadata") or {})
    arm_spec = {k: v for k, v in arm.items() if k not in _ARM_CONTROL_KEYS}
    spec = {**shape_spec, **arm_spec}
    if spec_order:
        # The emitted spec's key order is part of the descriptor bytes, and the
        # shipped sets use an order that is neither the shape's nor sorted.
        # Stating it per group reproduces those bytes without reordering.
        ordered = {key: spec[key] for key in spec_order if key in spec}
        ordered.update({k: v for k, v in spec.items() if k not in ordered})
        spec = ordered

    metadata = {}
    for field_name in metadata_fields:
        if field_name in arm_metadata:
            # The arm states the matcher-visible value directly, which is how a
            # knob the spec does not carry gets swept: pinning one changes the
            # selected descriptor without changing the binary.
            value = arm_metadata[field_name]
        # Not `field_name in spec`: a spec key present with value None means
        # the kernel's policy decides at build time, the same tri-state as
        # omitting it (`dispatch_parity.build_config` dumps the builder's
        # dataclass, so every unset policy knob arrives present-and-None).
        elif spec.get(field_name) is not None:
            value = spec[field_name]
        elif field_name in resolved:
            # Absent from the spec but known: the binary is definite and the
            # shape says what it is. Otherwise the loader would use the KMD
            # default_value as the catalog key while the kernel was compiled
            # from the builder's own default.
            value = resolved[field_name]
        elif field_name in policy_knobs:
            raise ConfigError(
                f"{shape_where}: '{field_name}' is a policy knob left absent "
                f"from the spec, so the kernel's own policy decides it at build "
                f"time -- but this shape's 'resolved' block does not say what it "
                f"decided. The matcher compares metadata, and an absent knob "
                f"resolves to the KMD default_value, which can select a "
                f"different binary than the descriptor was built from."
            )
        else:
            raise ConfigError(
                f"{shape_where}: metadata field '{field_name}' is in neither the "
                f"shape, the arm, nor 'resolved'. Nothing here decides its value."
            )
        if isinstance(value, bool) and kmd_types[field_name] == "int":
            value = int(value)
        elif kmd_types[field_name] == "float" and type(value) in (int, float):
            value = float(value)
        if field_name in vocabulary and isinstance(value, str):
            # The matcher compares the hipDNN spelling; the spec carries the
            # builder's. Copying one over the other declines every graph while
            # the engine still loads and every count reconciles.
            value = vocabulary[field_name].get(value, value)
        metadata[field_name] = value

    # A bool renders as `True`/`False` under str.format, but every shipped
    # grammar spells these flags `c1`/`p0`. Normalise so a template slot reads
    # the same as the metadata mirror of the same field.
    fields = {k: int(v) if isinstance(v, bool) else v for k, v in spec.items()}
    fields.update({f"md_{k}": v for k, v in metadata.items()})
    fields["ordinal"] = ordinal + arm.get("ordinal_offset", 0)
    try:
        fields["tag"] = str(arm.get("tag", "")).format(**fields)
        template = name_template
        if not fields["tag"]:
            # An empty tag would leave `..._p0__e1` or a trailing `_`. Drop one
            # adjacent separator from the template, where the slot's position
            # is known: a rendered-name fixup cannot tell its own separator
            # from one inside a value and could collide two kernels.
            template = re.sub(r"_?\{tag\}_?", _drop_tag_slot, template, count=1)
        name = template.format(**fields)
    except KeyError as exc:
        raise ConfigError(
            f"{shape_where}: the group's name template or tag names {exc}, which "
            f"neither the shape, the arm nor the resolved metadata provides. A name "
            f"built from a field that is not there cannot be unique."
        )
    except (ValueError, IndexError, AttributeError) as exc:
        raise ConfigError(
            f"{shape_where}: the group's name template {name_template!r} (tag "
            f"{arm.get('tag', '')!r}) could not be rendered: {exc}. Every slot must "
            f"be a plain {{field}} naming a spec field, an md_<field> metadata "
            f"mirror, {{tag}} or {{ordinal}}."
        )
    # An arm authors spec values, not kernel_source keys, so the authored
    # mapping is empty; a variant kernel's only authored kernel_source keys
    # come from the pack's `kernel_defaults`, checked separately. That check
    # must skip `spec`, the one `kernel_defaults` key this expander reads for
    # every kind, or a `variants` pack would be rejected under any kind but
    # `rocke`.
    return {
        "name": name,
        "kernel_source": {"spec": spec},
        "metadata": metadata,
        _AUTHORED_KERNEL_SOURCE: (shape_where, {}),
        _EXPANDER_CONSUMED_DEFAULTS: frozenset({"spec"}),
    }


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


def _require_mapping(value, scope: str) -> None:
    """Require a mapping: the merge would otherwise raise a ``ValueError`` from
    inside a dict comprehension, naming neither the kernel nor the key, and
    generate.py catches only ``ConfigError``."""
    if not isinstance(value, dict):
        raise ConfigError(
            f"{scope} must be a mapping; got {type(value).__name__} ({value!r}). "
            f"A scalar or list here is usually a YAML indentation slip."
        )


def _require_specialization_mapping(declaration) -> None:
    """The ``specialization`` block's own shape check, shared by both its callers."""
    if not isinstance(declaration, dict):
        raise ConfigError(
            f"'specialization' must be a mapping; got "
            f"{type(declaration).__name__} ({declaration!r})."
        )


def _require_list(mapping: dict, key: str, where: str) -> None:
    """A key the loader iterates as a list of entries is one, whenever it is written.

    Absent is fine -- every reader defaults it.
    """
    if key not in mapping or isinstance(mapping[key], list):
        return
    raise ConfigError(
        f"'{where}' must be a list of entries; got "
        f"{type(mapping[key]).__name__} ({mapping[key]!r}). A key written with no "
        f"value under it is null rather than an empty list. Delete the line, or "
        f"give it entries."
    )


def _require_config_shapes(raw: dict) -> None:
    """Require every container the key walks index into to be the expected
    type. ``'name' not in 'namey'`` is False, so a mistyped scalar reaches
    ``.get`` on a str; and a null list key reads as an empty config."""
    _require_list(raw, "packs", "packs")
    _require_list(raw, "kmd_fields", "kmd_fields")
    engine_raw = raw.get("engine")
    if engine_raw:
        _require_mapping(engine_raw, "engine")
    for index, field_raw in enumerate(raw.get("kmd_fields") or []):
        _require_mapping(field_raw, f"kmd_fields[{index}]")
    for index, pack_raw in enumerate(raw.get("packs") or []):
        _require_mapping(pack_raw, f"packs[{index}]")
        where = f"pack {pack_raw.get('name', '<unnamed>')!r}"
        _require_list(pack_raw, "kernels", f"{where} kernels")
        for kernel_index, kernel_raw in enumerate(pack_raw.get("kernels") or []):
            _require_mapping(kernel_raw, f"{where} kernels[{kernel_index}]")
        # Same trap one level down, at `_expand_axis_kernels`'s
        # `'kernel_source' not in template` membership test.
        if pack_raw.get("kernel_template") is not None:
            _require_mapping(pack_raw["kernel_template"], f"{where} kernel_template")


#: Every key each level of the config understands. Closed on purpose -- see
#: `_reject_unknown_keys`.
_KNOWN_TOP = frozenset(
    {
        "engine",
        "kmd_fields",
        "packs",
        "dialect",
        "kernel_source_kind",
        "workspace_policy",
        "authored_subpath",
        "graph_match",
        "specialization",
    }
)
_KNOWN_ENGINE = frozenset(
    {
        "name",
        "sdk_version",
        "behavior_notes",
        "knobs",
        "heuristic",
    }
)
_KNOWN_KMD_FIELD = frozenset({"name", "type", "default_value"})
_KNOWN_PACK = frozenset(
    {
        "name",
        "arch",
        "kernels",
        "kernel_defaults",
        "discriminator",
        "axes",
        "kernel_template",
        "variants",
    }
)
_KNOWN_VARIANT_GROUP = frozenset(
    {
        "name",
        "metadata",
        "knob_sets",
        "shapes",
        "vocabulary",
        "policy_knobs",
        "spec_defaults",
        "spec_order",
    }
)
_KNOWN_KERNEL = frozenset({"name", "kernel_source", "metadata", "priority", "arch"})
#: Each authored kind's MANDATORY ``kernel_source`` fields, mirroring
#: ``hkp_pack._validate_ukd_fields``. Shared with `_KNOWN_KERNEL_SOURCE_BY_KIND`
#: below so the required set and the closed vocabulary cannot drift.
_REQUIRED_KERNEL_SOURCE_FIELDS: dict = {
    KERNEL_SOURCE_KIND_EMBEDDED: ("source_file", "entry_point"),
    KERNEL_SOURCE_KIND_HIP: ("source", "entry"),
    KERNEL_SOURCE_KIND_ROCKE: ("source", "builder", "spec"),
}
#: The fields a kind owns but may omit -- ``KernelSource.as_document`` writes them
#: for that kind, and nothing requires them.
_OPTIONAL_KERNEL_SOURCE_FIELDS: dict = {
    KERNEL_SOURCE_KIND_HIP: ("build",),
}
#: A ``kernel_source``'s closed key set, per kind. Only the AUTHORED kinds appear:
#: the others are rejected by `_check_kernel_source_kind_implemented` with a reason.
_KNOWN_KERNEL_SOURCE_BY_KIND: dict = {
    kind: frozenset({"kind", *required, *_OPTIONAL_KERNEL_SOURCE_FIELDS.get(kind, ())})
    for kind, required in _REQUIRED_KERNEL_SOURCE_FIELDS.items()
}
#: The top-level ``specialization`` block's own keys. ``engine_id``/``kmd_id``
#: are absent because ``generator.mint_ids`` stamps them onto the emitted
#: contract, and so is ``consumers``, which one config contributes exactly one
#: entry to.
_KNOWN_SPECIALIZATION = frozenset(
    {
        "metadata_fields",
        "matcher_only_fields",
        "bindings",
        "vocabulary",
    }
)


def _check_kernel_source_keys(keys, kind, where: str) -> None:
    """Require a ``kernel_source`` (or ``kernel_defaults``) to spell only its
    kind's keys -- the closed set ``KernelSource.as_document`` writes and the
    runtime enforces (``parseKernelSource``, ``requireKnownKeys``).

    A non-string ``kind`` is left to `_check_kernel_source_kind_implemented`.
    """
    if not isinstance(kind, str):
        return
    allowed = _KNOWN_KERNEL_SOURCE_BY_KIND.get(kind)
    if allowed is None:
        return
    unknown = sorted(set(keys) - allowed)
    if unknown:
        raise ConfigError(
            f"{where} declares {unknown}, which kind '{kind}' does not read. Known "
            f"keys for this kind: {sorted(allowed)}. A key this kind does not own "
            f"is dropped when the descriptor is written, so the bundle generates "
            f"cleanly without whatever it was meant to configure -- check for a "
            f"typo, or for a key copied from another kind's example."
        )


def _reject_unknown_keys(raw: dict) -> None:
    """Refuse a key no level of this loader reads. The vocabularies are closed:
    an unrecognised key would otherwise be dropped by ``raw.get(key, default)``
    and ``engine.knobbs`` would emit a UED with no knobs."""

    def check(scope: str, mapping, allowed: frozenset) -> None:
        if not isinstance(mapping, dict):
            return  # shape errors belong to the callers' own diagnostics
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ConfigError(
                f"{scope} declares {unknown}, which this generator does not read. "
                f"Known keys: {sorted(allowed)}. An unrecognised key is silently "
                f"ignored otherwise, so the bundle would generate cleanly without "
                f"whatever you meant to configure -- check for a typo."
            )

    check("the config's top level", raw, _KNOWN_TOP)
    check("engine", raw.get("engine"), _KNOWN_ENGINE)
    for field in raw.get("kmd_fields", []) or []:
        check(
            f"kmd_fields entry {field.get('name', '<unnamed>')!r}",
            field,
            _KNOWN_KMD_FIELD,
        )
    for pack in raw.get("packs", []) or []:
        if not isinstance(pack, dict):
            continue
        check(f"pack {pack.get('name', '<unnamed>')!r}", pack, _KNOWN_PACK)
        for kernel in pack.get("kernels", []) or []:
            check(f"kernel {kernel.get('name', '<unnamed>')!r}", kernel, _KNOWN_KERNEL)
        # `kernel_template` is a kernel envelope too, and every kernel the pack
        # ships comes out of it: a key misspelled here is dropped from the
        # whole cross-product rather than from one entry.
        template = pack.get("kernel_template")
        if template is not None:
            check(
                f"pack {pack.get('name', '<unnamed>')!r} kernel_template",
                template,
                _KNOWN_KERNEL,
            )


def _reject_deprecated_keys(raw: dict) -> None:
    """Reject YAML keys that look plausible but name nothing this loader or the
    runtime reads.

    The ``kmd_fields[]`` keys come from RFC 0017 §4's example field: only
    ``default_value`` is real, and there is no ``optional`` key. The three
    retired top-level keys are named so an out-of-tree config still setting
    them is told they were retired.
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
    if "descriptor_files_var" in raw:
        raise ConfigError(
            "Top-level 'descriptor_files_var' is retired. It was accepted as a "
            "known key when the tool shipped, but no model field, no template and "
            "no generator step ever read it -- a config that set it emitted exactly "
            "the bundle a config that omitted it emitted. Remove it; the emitted "
            "cmake_descriptor_files.txt fragment already states how this bundle's "
            "descriptors reach the build."
        )
    if "pack_kernels_var" in raw:
        raise ConfigError(
            "Top-level 'pack_kernels_var' is retired. It was accepted as a known "
            "key when the tool shipped, but no model field, no template and no "
            "generator step ever read it -- a config that set it emitted exactly "
            "the bundle a config that omitted it emitted. Remove it; the emitted "
            "cmake_target_sources.txt fragment already states which sources the "
            "pack target gains."
        )
    if "delegates_to_existing_plan" in raw:
        raise ConfigError(
            "Top-level 'delegates_to_existing_plan' is retired. It was accepted as "
            "a known key when the tool shipped, and this tool's own example configs "
            "and README set it, so a config copied from either carries it -- but no "
            "model field, no template and no generator step ever read it. Delete the "
            "line, whatever it was set to: deleting the line changes nothing about "
            "the bundle this config emits."
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


#: What ``hipdnn_data_sdk::utilities::Version``'s constructor accepts: three
#: separate integers parsed with ``istringstream >>``, which skips surrounding
#: whitespace and stops at the first character that cannot continue the number,
#: so a sign is legal and trailing text ignored. Matched exactly rather than
#: tightened: a stricter check would reject a config the runtime loads.
_SDK_VERSION_PATTERN = re.compile(r"\s*([+-]?\d+)\s*\.\s*([+-]?\d+)\s*\.\s*([+-]?\d+)")
#: The range each component is read into (``int``), and the range a descriptor's
#: ``priority`` is read into (``int64_t``, via ``requireInt64``).
_INT_MIN, _INT_MAX = -(2**31), 2**31 - 1
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


def _reject_repeats(values: list, what: str, where: str) -> None:
    """``DescriptorLoader.hpp``'s ``requireNoDuplicates`` (634-645), in Python.

    Walks the list rather than building a set, so an entry YAML read as a list
    or a mapping is reported rather than raising ``TypeError: unhashable``.
    """
    repeated: list = []
    for index, value in enumerate(values):
        if value in values[:index] and value not in repeated:
            repeated.append(value)
    if repeated:
        raise ConfigError(
            f"{where} lists {repeated} more than once. requireNoDuplicates fails "
            f"the whole descriptor on a repeated {what}, so this engine would load "
            f"nothing at all."
        )


def _check_engine_declaration(config: IngestorConfig) -> None:
    """Check the engine block as ``parseEngineDescriptor`` reads it:
    ``sdk_version`` through ``requireString`` (260-266) and the ``Version``
    constructor (770-781), ``knobs`` and ``behavior_notes`` for repeats (750,
    753). Arch lists are exempt; `_unique_arch` collapses their repeats."""
    version = config.engine.sdk_version
    if not isinstance(version, str):
        raise ConfigError(
            f"engine.sdk_version must be a string; got "
            f"{type(version).__name__} ({version!r}). An unquoted 1.0 is a YAML "
            f"float, and the loader's requireString refuses a number outright. "
            f"Quote it, with all three components: '1.0.0'."
        )
    match = _SDK_VERSION_PATTERN.match(version)
    if not match or not all(
        _INT_MIN <= int(part) <= _INT_MAX for part in match.groups()
    ):
        raise ConfigError(
            f"engine.sdk_version '{version}' is not a version the loader can "
            f"parse. hipdnn_data_sdk::utilities::Version reads three integers "
            f"'<major>.<minor>.<patch>' and throws on anything else, which fails "
            f"the engine descriptor and loads none of its packs. Use e.g. '1.0.0'."
        )
    _reject_repeats(config.engine.knobs, "knob", "engine.knobs")
    _reject_repeats(
        config.engine.behavior_notes, "behavior note", "engine.behavior_notes"
    )


def _check_knobs_int_typed(config: IngestorConfig) -> None:
    """Pre-mint check #2: every knob names a declared, int-typed KMD field.
    ``GenericPlanBuilder::getCustomKnobs`` filters to ``int64_t`` alternatives,
    so the loader silently produces no knob at all for any other type."""
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


def _int64_overflow(value, field_type: str):
    """The first part of an ``int``/``int_list`` value that does not fit int64,
    or ``None`` when every integer fits. ``metadataValueFromJson`` reads every
    integer through ``requireInt64`` (``DescriptorLoader.hpp``:511-519, 531,
    550)."""
    if field_type not in ("int", "int_list"):
        return None
    for item in value if field_type == "int_list" else [value]:
        if not _INT64_MIN <= item <= _INT64_MAX:
            return item
    return None


def _check_kernel_metadata_against_kmd(config: IngestorConfig) -> None:
    """Pre-mint check #3: every kernel's metadata type-checks against the KMD,
    with no mandatory field omitted. A wrong type, an undeclared field or an
    omitted mandatory field drops the whole pack at load."""
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
                    oversized = _int64_overflow(value, kmd_field.type)
                    if oversized is not None:
                        raise ConfigError(
                            f"{where}: metadata '{kmd_field.name}' carries "
                            f"{oversized!r}, which does not fit a signed 64-bit "
                            f"integer. requireInt64 fails the whole descriptor "
                            f"rather than reinterpreting it as a negative value."
                        )


def _check_kmd_default_values(config: IngestorConfig) -> None:
    """Check each KMD field's ``default_value`` against its declared ``type``,
    by the rules ``coerceToDeclaredType`` applies
    (``DescriptorLoader.hpp``:604-610). A field with no ``default_value`` is
    mandatory (``KmdField.is_mandatory``)."""
    for kmd_field in config.kmd_fields:
        if kmd_field.default_value is None:
            continue
        if not _METADATA_TYPE_CHECKS[kmd_field.type](kmd_field.default_value):
            raise ConfigError(
                f"kmd_fields entry '{kmd_field.name}' declares default_value "
                f"{kmd_field.default_value!r}, which contradicts its declared type "
                f"'{kmd_field.type}'. coerceToDeclaredType widens an authored "
                f"integer into a 'float' field and accepts nothing else, so this "
                f"KMD fails to parse and the engine loads no descriptors at all."
            )
        oversized = _int64_overflow(kmd_field.default_value, kmd_field.type)
        if oversized is not None:
            raise ConfigError(
                f"kmd_fields entry '{kmd_field.name}' declares a default_value "
                f"carrying {oversized!r}, which does not fit a signed 64-bit "
                f"integer. requireInt64 fails the KMD rather than reinterpreting "
                f"it, and a KMD that fails to parse loads no descriptors at all."
            )


def _check_kernel_priority(config: IngestorConfig) -> None:
    """Require every kernel ``priority`` to be a signed 64-bit integer.
    ``parseKernelDescriptor`` gates on ``is_number_integer()`` and reads
    through ``requireInt64`` (``DescriptorLoader.hpp``:977-984); nlohmann
    reports ``true`` as ``is_boolean``, so Python ``bool`` is refused."""
    for pack in config.packs:
        for kernel in pack.kernels:
            priority = kernel.priority
            where = f"pack '{pack.name}' kernel '{kernel.name}'"
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise ConfigError(
                    f"{where} declares priority {priority!r}, which must be an "
                    f"integer. The loader gates the key on is_number_integer(), "
                    f"which is false for a boolean and for any fractional value, "
                    f"and fails the whole descriptor."
                )
            if not _INT64_MIN <= priority <= _INT64_MAX:
                raise ConfigError(
                    f"{where} declares priority {priority!r}, which does not fit a "
                    f"signed 64-bit integer. requireInt64 rejects it rather than "
                    f"reinterpreting it as a negative rank."
                )


def _check_kernel_arch_subset_of_pack(config: IngestorConfig) -> None:
    """Pre-mint check #4: a kernel's arch must be a subset of its pack's.

    Mirrors ``DescriptorLoader.hpp``'s ``archCovers(pack.arch, kernel.arch)``.
    An empty pack.arch covers everything; an empty kernel.arch inherits it.
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
    ``ConfigError``, mirroring ``isPlausibleArchBaseId``; a well-formed but
    unrecognized id (``gfx94``) is a warning. Returns the warnings emitted,
    also raised via ``warnings.warn``.
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
    """Require a known dialect, and an ``arch`` on a packaged bundle:
    ``hkp_pack._validate_kdp`` requires it where the runtime loader treats
    absence as a wildcard."""
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


def _check_authored_subpath(config: IngestorConfig) -> None:
    """Check ``authored_subpath`` against the dialect's tree.

    ``direct_load`` must name one of the authored sets, each a separate pack
    target reaching a different binary. ``packaged`` is free-form but must be a
    relative subpath of the ``descriptors/`` source root: ``descriptor_dir``
    joins the two as a string, so ``..`` or an absolute path would let
    ``render()`` write above ``--output-dir``. Checked lexically, since no
    output directory exists yet.
    """
    if config.is_packaged:
        subpath = config.authored_subpath
        if not subpath:
            return
        # Read with Windows semantics on either host so a config validated on
        # Linux and generated on Windows gets the same answer: backslash is a
        # separator there, and ``C:x`` is drive-relative rather than rooted.
        stated = PureWindowsPath(subpath)
        if stated.drive or stated.root:
            raise ConfigError(
                f"engine '{config.engine.name}' is a '{DIALECT_PACKAGED}' bundle "
                f"whose 'authored_subpath' is '{subpath}', which is not relative. "
                f"It is joined under the bundle's 'descriptors/' source root, and a "
                f"rooted or drive-qualified subpath does not join -- the descriptors "
                f"land outside --output-dir. State a path relative to 'descriptors/', "
                f"e.g. '{config.kernel_source_kind}/{config.engine.slug}' (the default)."
            )
        resolved = posixpath.normpath(stated.as_posix())
        if resolved == ".." or resolved.startswith("../"):
            raise ConfigError(
                f"engine '{config.engine.name}' is a '{DIALECT_PACKAGED}' bundle "
                f"whose 'authored_subpath' '{subpath}' resolves to '{resolved}', "
                f"outside the 'descriptors/' source root it is joined into -- it "
                f"would write this bundle's files above --output-dir and install "
                f"them where hkp_pack never walks. State a path under 'descriptors/', "
                f"e.g. '{config.kernel_source_kind}/{config.engine.slug}' (the default)."
            )
        return
    if config.authored_subpath not in AUTHORED_TEST_SETS:
        stated = (
            f"'{config.authored_subpath}'" if config.authored_subpath else "nothing"
        )
        raise ConfigError(
            f"engine '{config.engine.name}' is a '{DIALECT_DIRECT_LOAD}' bundle, so "
            f"'authored_subpath' must name the authored set it is written into, and "
            f"it states {stated}. Use one of: "
            f"{', '.join(AUTHORED_TEST_SETS)}. Each is a separate pack target under "
            f"test_descriptors/, so the set decides which shard these descriptors "
            f"land in and which test binary can read them -- the consuming binary "
            f"chooses it and this tool cannot infer it."
        )


def _check_kernel_source_kind_implemented(config: IngestorConfig) -> None:
    """Reject a ``kernel_source.kind`` the configured dialect cannot emit.

    Each rejection names the dialect, because the common mistake is a real kind
    belonging to the other one, which a bare "unsupported" would hide.
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

    Mirrors ``hkp_pack._validate_ukd_fields``, checked here so an author sees it
    before a comgr run rather than after.
    """
    required_by_kind = _REQUIRED_KERNEL_SOURCE_FIELDS
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


def _check_specialization_declaration(config: IngestorConfig) -> None:
    """Check the ``specialization`` block: which metadata fields the compiler
    specialized on, and how each is read off the builder object.

    It becomes the ``provenance.specialization_contract`` the emitted KDP
    carries, which is all a machine checking a shipped bundle has; every claim
    is one ``hkp_pack.agreement.validate_consumer`` tests.

    The partition over ``kmd_fields`` is exhaustive and disjoint: each field is
    either consumed by the builder (``metadata_fields``, with a binding) or
    matcher-only. A ``rocke`` kernel's spec keys reached the compiler, so they
    cannot be matcher-only; direct-load and ``hip`` state
    ``metadata_fields: []`` explicitly. Presence is enforced at emission.
    """
    declaration = config.specialization
    if not declaration:
        return
    _require_specialization_mapping(declaration)
    unknown = sorted(set(declaration) - _KNOWN_SPECIALIZATION)
    if unknown:
        raise ConfigError(
            f"'specialization' declares {unknown}, which this loader does not "
            f"read. Known keys: {sorted(_KNOWN_SPECIALIZATION)}. 'engine_id' and "
            f"'kmd_id' are minted at generation and stamped on automatically; a "
            f"'consumers' list belongs to the emitted contract, not to a config, "
            f"which declares exactly one engine and one KMD."
        )

    declared = [f.name for f in config.kmd_fields]
    partition = {}
    for key in ("metadata_fields", "matcher_only_fields"):
        value = declaration.get(key) or []
        if not isinstance(value, list) or any(not isinstance(n, str) for n in value):
            raise ConfigError(
                f"'specialization.{key}' must be a list of kmd_fields names; got "
                f"{value!r}."
            )
        repeated = sorted({n for n in value if value.count(n) > 1})
        if repeated:
            raise ConfigError(
                f"'specialization.{key}' names {repeated} more than once."
            )
        partition[key] = list(value)

    checked = set(partition["metadata_fields"])
    matcher_only = set(partition["matcher_only_fields"])
    both = sorted(checked & matcher_only)
    if both:
        raise ConfigError(
            f"'specialization' lists {both} in BOTH 'metadata_fields' and "
            f"'matcher_only_fields'. A field is either one the compiler consumed "
            f"or one only the matcher reads; a field claiming to be both makes "
            f"the declaration self-contradictory, and a checker cannot decide "
            f"whether to demand a binding for it."
        )
    unpartitioned = sorted(set(declared) - checked - matcher_only)
    invented = sorted((checked | matcher_only) - set(declared))
    if unpartitioned or invented:
        raise ConfigError(
            f"'specialization' must partition the declared kmd_fields "
            f"{sorted(declared)} exhaustively: "
            f"{unpartitioned} are in neither 'metadata_fields' nor "
            f"'matcher_only_fields', and {invented} name no kmd_fields entry. An "
            f"unlisted field reads to a checker as one nobody specialized on, so "
            f"a value that decided the compiled binary would be passed over "
            f"unchecked."
        )

    bindings = declaration.get("bindings") or {}
    _require_mapping(bindings, "'specialization.bindings'")
    if set(bindings) != checked:
        raise ConfigError(
            f"'specialization.bindings' keys {sorted(bindings)} must equal "
            f"'metadata_fields' {sorted(checked)}. A checked field without a "
            f"binding cannot be read back off the builder object, and a binding "
            f"for an unchecked field describes a read nothing performs."
        )
    for name, binding in bindings.items():
        _require_mapping(binding, f"'specialization.bindings[{name}]'")
        if set(binding) not in ({"field"}, {"method"}):
            raise ConfigError(
                f"'specialization.bindings[{name}]' is {binding!r}; it must name "
                f"exactly one of 'field' (a direct attribute of the hydrated spec "
                f"or builder object) or 'method' (an existing zero-argument "
                f"effective accessor on that same object). Naming both leaves the "
                f"checker to choose which reading is authoritative, and naming "
                f"neither leaves it nothing to read."
            )
        accessor = next(iter(binding.values()))
        if not isinstance(accessor, str) or not accessor.isidentifier():
            raise ConfigError(
                f"'specialization.bindings[{name}]' must name one explicit "
                f"attribute; {accessor!r} is not an identifier. A computed or "
                f"guessed accessor name is how a checker ends up reading a "
                f"convention nobody implemented and reporting agreement anyway."
            )

    vocabulary = declaration.get("vocabulary") or {}
    _require_mapping(vocabulary, "'specialization.vocabulary'")
    stray = sorted(set(vocabulary) - checked)
    if stray:
        raise ConfigError(
            f"'specialization.vocabulary' translates {stray}, which "
            f"'metadata_fields' does not carry, so the translation would have no "
            f"effect -- and an untranslated builder spelling in metadata loads "
            f"cleanly, reconciles on every count, and matches nothing."
        )
    for name, spellings in vocabulary.items():
        _require_mapping(spellings, f"'specialization.vocabulary[{name}]'")

    kinds = {
        kernel.kernel_source.kind for pack in config.packs for kernel in pack.kernels
    }
    if KERNEL_SOURCE_KIND_ROCKE in kinds:
        specialized = {
            name
            for pack in config.packs
            for kernel in pack.kernels
            for name in (kernel.kernel_source.spec or {})
        } & set(declared)
        waived = sorted(specialized & matcher_only)
        if waived:
            raise ConfigError(
                f"'specialization' calls {waived} matcher-only, but a "
                f"'{KERNEL_SOURCE_KIND_ROCKE}' kernel's kernel_source.spec carries "
                f"those keys -- they are hydrated into the spec dataclass the "
                f"builder is called with, so they demonstrably reached the "
                f"compiler. Declaring a field the compiler consumed as matcher-only "
                f"removes it from the agreement check while it keeps deciding the "
                f"binary. List them in 'metadata_fields' with a binding each."
            )
    elif checked:
        raise ConfigError(
            f"'specialization.metadata_fields' names {sorted(checked)}, but no "
            f"kernel in this config is built from a compiled specialization "
            f"(kinds: {sorted(kinds)}). The direct-load and "
            f"'{KERNEL_SOURCE_KIND_HIP}' paths hydrate no builder object, so there "
            f"is nothing for a binding to read back and no agreement to check. "
            f"Declare 'metadata_fields: []' and list every field under "
            f"'matcher_only_fields'."
        )


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
    # Pack names key the pack's descriptor id and its output filename
    # (`<engine-slug>_<pack-name>.kdp.json`), so two packs sharing a name
    # collide twice: same id, and the second file overwrites the first.
    pack_names = [pack.name for pack in config.packs]
    duplicate_names = sorted({n for n in pack_names if pack_names.count(n) > 1})
    if duplicate_names:
        raise ConfigError(
            f"packs declare duplicate names: {duplicate_names}. A pack name keys "
            f"both its descriptor id and its output file, so duplicates overwrite "
            f"each other silently."
        )
    for pack in config.packs:
        if not pack.kernels:
            raise ConfigError(f"pack '{pack.name}' declares no kernels.")


#: The ``<STEM>_MATCHER_SYMBOL`` constants ``native.cpp.j2`` emits with a
#: literal stem, outside its per-pack loop. They share the matcher-symbol
#: namespace with the per-pack constants, so a discriminator folding onto one
#: redefines it. ``tests/test_config_loader.py`` compares this tuple with the
#: stems in the template.
RESERVED_MATCHER_SYMBOL_STEMS = ("GRAPH", "KERNEL")


def _check_emitted_identifiers(config: IngestorConfig) -> None:
    """Check every name spliced into a generated C++ identifier or emitted path
    for shape and collisions.

    ``native.cpp.j2`` interpolates each kmd field name uppercased into
    ``<NAME>_FIELD`` and each pack discriminator into ``<NAME>_MATCHER_SYMBOL``
    and ``<name>OperationMatches``; the two suffixes are independent
    namespaces, and a discriminator must also avoid the reserved stems.

    The engine's local name is checked against two rules, since
    `pascal_name`/`camel_name` are C++ identifiers while `slug` is a directory
    name and file stem. It is read through `EngineSpec`, so this runs after
    check #1. A pack's name is the other half of that stem (`kdp_stem` builds
    ``<engine-slug>_<pack-name>``) and is checked for every engine.
    """
    for derived, emitted_as in (
        (
            config.engine.pascal_name,
            "the classes 'Test<NAME>Packs' and '<NAME>DispatchHandler', the "
            "functions 'register<NAME>Symbols' and 'reset<NAME>ModuleCache', and "
            "the file 'packs/<NAME>Native.cpp'",
        ),
        (
            config.engine.camel_name,
            "the functions '<name>GraphMatches' and '<name>DispatchHandler'",
        ),
    ):
        if not CXX_IDENTIFIER_PATTERN.match(derived):
            raise ConfigError(
                f"engine.name '{config.engine.name}' derives '{derived}' from its "
                f"local name '{config.engine.local_name}', which must be a C++ "
                f"identifier, matching ^[A-Za-z_][A-Za-z0-9_]*$. It names "
                f"{emitted_as}. Anything else is not a declined match -- it is a "
                f"file nobody wrote failing to compile. Spell the local name in "
                f"PascalCase, snake_case or kebab-case."
            )
    if not PATH_STEM_PATTERN.match(config.engine.slug):
        raise ConfigError(
            f"engine.name '{config.engine.name}' derives the slug "
            f"'{config.engine.slug}' from its local name "
            f"'{config.engine.local_name}', which must be a single path stem, "
            f"matching ^[A-Za-z0-9_][A-Za-z0-9_-]*$. The slug is this bundle's "
            f"descriptor directory name and the stem of every descriptor file "
            f"under it, so a '.' or a separator does not name a badly-spelled "
            f"bundle -- it names a different directory, and '..' names the parent "
            f"of the one this config asked for. A hyphen is fine here and is "
            f"folded away in the C++ names. Spell the local name in PascalCase, "
            f"snake_case or kebab-case."
        )
    for kmd_field in config.kmd_fields:
        if not CXX_IDENTIFIER_PATTERN.match(kmd_field.name):
            raise ConfigError(
                f"kmd_fields entry '{kmd_field.name}' must be a C++ identifier, "
                f"matching ^[A-Za-z_][A-Za-z0-9_]*$. The name is emitted uppercased "
                f"as the constant '<NAME>_FIELD' in the generated native pack, so "
                f"any other character does not decline at match time -- it fails "
                f"to compile, in a file nobody wrote."
            )
    for pack in config.packs:
        if not PATH_STEM_PATTERN.match(pack.name):
            raise ConfigError(
                f"pack name '{pack.name}' must be a single path stem, matching "
                f"^[A-Za-z0-9_][A-Za-z0-9_-]*$. A multi-pack engine names its KDP "
                f"'<engine-slug>_<pack-name>.kdp.json' and gives the descriptor the "
                f"runtime name '<namespace>:<engine-slug>_<pack-name>', so a '.' or "
                f"a separator here does not name a badly-spelled pack -- it names a "
                f"different path, and '..' reaches out of the directory this config "
                f"asked for. A hyphen is fine: the pack name is never folded into a "
                f"C++ identifier, which is what this pack's 'discriminator' is for. "
                f"Spell it in snake_case or kebab-case."
            )
        if pack.discriminator and not CXX_IDENTIFIER_PATTERN.match(pack.discriminator):
            raise ConfigError(
                f"pack '{pack.name}' discriminator '{pack.discriminator}' must be "
                f"a C++ identifier, matching ^[A-Za-z_][A-Za-z0-9_]*$. It names "
                f"both the constant '<NAME>_MATCHER_SYMBOL' and the function "
                f"'{pack.discriminator}OperationMatches' in the generated native "
                f"pack, so any other character fails to compile there."
            )

    def claim(claims: dict, name: str, suffix: str, claimant: str) -> None:
        # Presence, not inequality: comparing descriptions would let an exact
        # duplicate claim a constant already taken, and the identifier is
        # emitted once per entry either way.
        constant = f"{name.upper()}{suffix}"
        if constant in claims:
            raise ConfigError(
                f"{claimant} and {claims[constant]} both emit the constant "
                f"'{constant}' in the generated native pack -- a redefinition the "
                f"compiler rejects. Two entries that differ only in case, or not at "
                f"all, land on one identifier. Rename or remove whichever of the two "
                f"this config owns."
            )
        claims[constant] = claimant

    field_claims: dict = {}
    for kmd_field in config.kmd_fields:
        claim(
            field_claims,
            kmd_field.name,
            "_FIELD",
            f"kmd_fields entry '{kmd_field.name}'",
        )

    matcher_claims: dict = {
        f"{stem}_MATCHER_SYMBOL": (
            f"the fixed {stem}_MATCHER_SYMBOL every generated native pack declares"
        )
        for stem in RESERVED_MATCHER_SYMBOL_STEMS
    }
    for pack in config.packs:
        if pack.discriminator:
            claim(
                matcher_claims,
                pack.discriminator,
                "_MATCHER_SYMBOL",
                f"pack '{pack.name}' discriminator '{pack.discriminator}'",
            )


def _validate_config(config: IngestorConfig) -> list[str]:
    """Run every pre-mint check, in order.

    Returns any non-fatal warning messages (currently only from check #5).
    """
    # The dialect decides which kinds are legal, so it is settled first --
    # every kind diagnostic below names it.
    _check_dialect(config)

    _check_engine_name_scoped(config)  # #1
    # After the name, before the knobs: both read the engine block, and a name
    # defect is the one an author fixes first.
    _check_engine_declaration(config)
    _check_knobs_int_typed(config)  # #2
    _check_kernel_metadata_against_kmd(config)  # #3
    # After #3, which establishes that every declared type is one of
    # KMD_FIELD_TYPES; the default_value check indexes _METADATA_TYPE_CHECKS by it.
    _check_kmd_default_values(config)
    _check_kernel_priority(config)
    _check_kernel_arch_subset_of_pack(config)  # #4
    warnings_out = _check_arch_shape(config)  # #5

    # Additional structural checks needed for a config to generate at all;
    # not among the five loader-mirroring checks, but still pre-mint.
    _check_kernel_source_kind_implemented(config)
    _check_kernel_source_fields(config)
    # After the kind checks: the declaration's obligations depend on which
    # kinds this config builds, so an unrecognized kind is named as a kind
    # problem rather than a specialization one.
    _check_specialization_declaration(config)
    _check_workspace_policy(config)
    _check_pack_discriminators(config)
    # After the discriminator check, which decides whether a pack carries one:
    # a single-pack engine emits no matcher symbol, so there is no identifier
    # to shape.
    _check_emitted_identifiers(config)
    # Last, because it is about where the bundle is written rather than what is
    # in it.
    _check_authored_subpath(config)

    for note in config.engine.behavior_notes:
        from .models import BEHAVIOR_NOTES

        if note not in BEHAVIOR_NOTES:
            raise ConfigError(
                f"engine.behavior_notes names '{note}', which is not in the "
                f"closed vocabulary {BEHAVIOR_NOTES}. The loader hard-rejects "
                f"anything else."
            )

    return warnings_out
