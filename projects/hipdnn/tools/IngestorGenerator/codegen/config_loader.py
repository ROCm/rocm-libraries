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

import gzip
import itertools
import re
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

    A ``.gz`` path is decompressed transparently. That is a capability, not the
    expected shipping form: a generated variant set belongs in the repo as plain text
    a reviewer can read, because generation is deterministic and the config -- not the
    descriptor set -- is the source of truth. See ``variants`` below for the form that
    keeps a generated set readable at that size.

    Raises ``ConfigError`` on any structural problem or failed pre-mint
    check. No UUID is minted here or anywhere reachable from here --
    minting happens only in ``generator.py``, after a config has fully
    survived this function.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: YAML document must be a top-level mapping.")

    _reject_unknown_keys(raw)
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

    kmd_field_names = {f.name for f in kmd_fields}

    packs = []
    for pack_raw in raw.get("packs", []):
        if "name" not in pack_raw:
            raise ConfigError(f"packs entry missing required key 'name': {pack_raw!r}")
        # A pack-level `kernel_defaults` is merged UNDER each kernel's own
        # `kernel_source`, so a kernel overrides it by simply restating the key.
        #
        # Generated variant sets repeat themselves enormously: a sweep over the tuning
        # axes of one engine restates `kind`, `source`, `builder` and every spec field
        # the sweep does not vary, once per kernel. Measured on the shipped gfx942
        # dense sets, five spec fields and all three kernel_source keys were identical
        # across all 2107 kernels -- roughly a third of the file saying the same thing
        # over and over, which is both large and unreadable: the fields that actually
        # differ between two variants are buried in the ones that never do.
        defaults = pack_raw.get("kernel_defaults", {}) or {}
        default_spec = dict(defaults.get("spec", {}))
        # `kernel_defaults` collapses repetition ACROSS kernels; `axes` collapses
        # repetition WITHIN a variant set driven by tuning knobs, which grows
        # multiplicatively rather than merely repeating. A five-axis, two-valued
        # sweep over a few hundred base shapes is enumerated as one YAML block per
        # variant today -- a six-figure line count that no build step reads and no
        # reviewer reads either. The axes plus the one kernel_template driving them
        # is the actual information content: about 30 lines for a small sweep,
        # however many kernels it expands to. See `_expand_axis_kernels` below for
        # the expansion itself, which runs entirely at load time and hands the
        # existing per-kernel loop ordinary kernel dicts it cannot tell apart from
        # hand-authored ones -- so it composes with `kernel_defaults` for free and
        # generator.py, the emitters and the dedup pass need no changes at all.
        axis_kernels_raw = _expand_axis_kernels(pack_raw, kmd_field_names)
        # `variants` collapses the OTHER shape of repetition: a set where every
        # shape carries its own dispatcher-resolved spec, so there is no single
        # kernel_template for `axes` to cross. See `_expand_variant_kernels`.
        variant_kernels_raw = _expand_variant_kernels(pack_raw, kmd_field_names)
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
            # Type-guard BEFORE the dict merges below. `dict("oops")` raises a
            # ValueError about "dictionary update sequence element #0" from deep
            # inside the merge -- a traceback generate.py does not catch, and one
            # that names neither the kernel nor the key. The loader HAS a proper
            # "must be a mapping" diagnostic further down; it was simply unreachable,
            # because the crash happened first. A check that cannot fire is not a
            # check.
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
        _check_kernel_names_unique(kernels, pack_raw["name"])
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


def _check_kernel_names_unique(kernels: list, pack_name: str) -> None:
    """Every kernel in a pack has its own name -- hand-authored or expanded.

    NOTHING downstream catches a collision. The loader's other checks cover PACK
    name uniqueness, and the de-duplication pass in ``generator.py`` keys on the
    resolved METADATA rather than the name, so two entries that share a name but
    differ in metadata are emitted as two descriptors the runtime cannot tell apart
    -- in a log, in a winner record, or in a failure message. Two entries that share
    a name AND metadata are worse: the survivor is whichever came first, silently.

    That is not hypothetical. A previous version of the naming code hardcoded a
    subset of attention's field names and, on any other op, found none of them and
    gave two distinct conv variants the same name.

    The check lives here rather than beside a name TEMPLATE because a name can also
    be hand-authored, and because a template that is injective over the fields it
    renders is still not injective if two shapes differ only in a field the template
    omits. Only the rendered result proves it.
    """
    seen: dict = {}
    collisions: dict = {}
    for kernel in kernels:
        if kernel.name in seen:
            collisions.setdefault(kernel.name, 1)
            collisions[kernel.name] += 1
        seen[kernel.name] = kernel
    if not collisions:
        return
    shown = ", ".join(
        f"{name!r} x{count}" for name, count in sorted(collisions.items())[:3]
    )
    more = f" (+{len(collisions) - 3} more)" if len(collisions) > 3 else ""
    raise ConfigError(
        f"pack '{pack_name}' declares {len(collisions)} duplicated kernel name(s): "
        f"{shown}{more}. Kernel names must be unique within a pack: nothing "
        f"downstream catches a collision, so the entries ship as descriptors that "
        f"cannot be told apart in a log or a failure message. If these came from a "
        f"'variants' group, its name template omits a field the shapes differ in -- "
        f"add that field to the template, or a per-arm 'tag' that distinguishes them."
    )


def _expand_axis_kernels(pack_raw: dict, kmd_field_names: set) -> list:
    """Expand a pack's ``axes`` cross-product into ordinary kernel dicts, entirely
    at load time, so ``generator.py``, the emitters, and the dedup pass need no
    changes at all -- they only ever see the same ``KernelSpec`` shape a
    hand-authored kernel produces.

    Enumeration is fine at roughly a hundred kernels. It stops being fine the
    moment a variant set is driven by tuning axes instead of hand-picked shapes:
    five two-valued knobs over a few hundred base shapes is a six-figure line
    count that no build step reads and no reviewer reads either. The actual
    information content of that sweep is the axes plus the one kernel template
    they vary -- on the order of 30 lines, whatever the expansion's kernel count
    turns out to be. ``axes`` lets a pack author that 30 lines directly instead
    of the six-figure enumeration it stands for.

    Returns ``[]`` if the pack declares no ``axes`` (the common case: an
    ordinary, hand-enumerated pack is unaffected).
    """
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

    # Sorted once, up front: naming and value-list order below both walk axis
    # names in this same fixed order, which is what makes the encoded name a
    # deterministic function of the combination rather than of dict iteration
    # order (YAML mappings do preserve insertion order, but nothing should rely
    # on that for something as load-bearing as name uniqueness).
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
        # The name must encode EVERY axis value, not a hand-picked subset -- a
        # name built from a subset is unique only by luck. That is not
        # hypothetical: dispatch_parity._kernel_name once hardcoded a subset of
        # attention's own field names, and on any other op found none of them,
        # so two distinct conv variants both landed on `conv_fwd_dtfp16`.
        # Encoding every axis, always, in this fixed sorted order, makes each
        # cross-product entry's name an injective function of its own
        # combination -- distinct by construction, not by hoping the axes
        # chosen happen to vary.
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
            }
        )
    return expanded


def _require_sequence(value, scope: str) -> list:
    """A key the loader is about to iterate as a list of names must be one.

    A bare string is the trap: `policy_knobs: use_exp2_fast` is valid YAML and
    iterates as CHARACTERS, so the knob is never recognised and the specific
    policy-knob diagnostic never fires. `spec_order: dtype` is worse -- it silently
    reorders nothing, and key order is part of the descriptor bytes.
    """
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ConfigError(
            f"{scope} must be a list of field names; got "
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


def _expand_variant_kernels(pack_raw: dict, kmd_field_names: set) -> list:
    """Expand a pack's ``variants`` groups into ordinary kernel dicts, at load
    time, so ``generator.py``, the emitters and the dedup pass see exactly the
    shape a hand-authored kernel produces and need no changes.

    WHY NOT ``axes``. ``axes`` crosses ONE ``kernel_template``, which fits a sweep
    over a single base kernel. A dispatcher-derived variant set does not look like
    that: ``dispatch_parity.py`` asks the library for a spec PER SHAPE, so every
    shape carries its own resolved values for the fields the dispatcher derives
    (``waves_per_eu`` and ``persistent`` on gfx942 attention_dense). There is no one
    template to cross. Forcing it into ``axes`` would mean either crossing those
    derived fields as axes -- which manufactures combinations the dispatcher would
    never resolve to, the exact mistake ``--report-knobs`` exists to prevent -- or
    one ``axes`` block per shape, which is the enumeration again with extra syntax.

    So a group is a shape LIST crossed per-shape with a NAMED knob set. The set is
    not a grid: on the shipped gfx942 sets most shapes carry four arms and 63 carry
    six, and a format that assumed a clean cross-product would silently drop the
    difference.

    THE TRI-STATE. A knob absent from an arm is absent from the emitted
    ``kernel_source.spec``, which is what tells the builder "your own policy decides
    this at build time". That is NOT the same as pinning it to ``false``, and both
    reach the metadata as the same ``0``. The distinction decides which binary is
    compiled, so the shape states the policy's answer under ``resolved`` and the arm
    states an override by naming the knob; a format collapsing the two into one
    boolean axis would throw the policy away silently.

    Returns ``[]`` if the pack declares no ``variants``.
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
            _expand_one_variant_group(group, position, pack_name, kmd_field_names)
        )
    return expanded


def _expand_one_variant_group(
    group: dict, position: int, pack_name: str, kmd_field_names: set
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
    # Knobs whose value the KERNEL'S OWN POLICY decides when the spec leaves them
    # absent. Naming them here is what makes "absent" legible as a third state
    # rather than as a missing key: each shape must then state what the policy
    # resolved to, so the metadata the matcher compares still describes the binary.
    policy_knobs = set(
        _require_sequence(group.get("policy_knobs"), f"{where} policy_knobs")
    )
    # A key naming a field the group does not emit has NO effect, and the silence is
    # the whole problem: a mistyped `vocabulary` entry leaves the builder's spelling
    # in the metadata, which loads cleanly, reconciles on every count, and matches
    # nothing. Cheaper to reject than to debug.
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
    # Spec fields constant across THIS group. Pack-level `kernel_defaults.spec`
    # already hoists what is constant across every kernel; a group needs its own
    # because the key ORDER of the emitted spec differs between groups of one set,
    # and order is part of the descriptor bytes.
    if group.get("spec_defaults") is not None:
        _require_mapping(group["spec_defaults"], f"{where} spec_defaults")
    spec_defaults = dict(group.get("spec_defaults") or {})
    for field_name, mapping in vocabulary.items():
        _require_mapping(mapping, f"{where} vocabulary['{field_name}']")

    undeclared = sorted(set(metadata_fields) - kmd_field_names)
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
        # A near-miss control key -- `ordinl` for `ordinal`, `resolvd` for
        # `resolved` -- would otherwise fall through into the SPEC, changing the
        # binary the descriptor names while the config still loads. When the group
        # declares `spec_order` it has already enumerated its spec fields, so anything
        # else is a typo. Groups without one are unconstrained, since nothing there
        # says what the field set should be.
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
) -> dict:
    """One (shape, arm) pair -> one kernel dict."""
    if not isinstance(arm, dict):
        raise ConfigError(
            f"{shape_where}: every knob_set arm must be a mapping, got "
            f"{type(arm).__name__}."
        )
    if arm.get("metadata") is not None:
        # Easy to get wrong: a group's `metadata` IS a list of field names, while an
        # arm's is a mapping of field to value. Same key, one level apart.
        _require_mapping(arm["metadata"], f"{shape_where}: knob_set arm metadata")
    arm_metadata = dict(arm.get("metadata") or {})
    arm_spec = {k: v for k, v in arm.items() if k not in _ARM_CONTROL_KEYS}
    spec = {**shape_spec, **arm_spec}
    if spec_order:
        # The emitted spec's KEY ORDER is part of the descriptor bytes, and the
        # shipped sets were written in an order that is neither the shape's nor
        # sorted. Stating it once per group reproduces those bytes without
        # reordering the config to match.
        ordered = {key: spec[key] for key in spec_order if key in spec}
        ordered.update({k: v for k, v in spec.items() if k not in ordered})
        spec = ordered

    metadata = {}
    for field_name in metadata_fields:
        if field_name in arm_metadata:
            # The arm states the matcher-visible value directly. This is how a knob
            # the SPEC does not carry gets swept: the dispatcher returns the shared
            # spec and leaves arch-private knobs to the kernel's policy, so pinning
            # one changes which descriptor the matcher selects without changing the
            # binary the spec builds. The two arms are genuinely different catalog
            # entries over the same kernel, and the config has to be able to say so.
            value = arm_metadata[field_name]
        elif field_name in spec:
            value = spec[field_name]
        elif field_name in resolved:
            # Absent from the spec but known: either the kernel's own policy decided
            # it at build time (a `policy_knobs` tri-state), or it is an arch-private
            # field the shared spec never carries while the builder's own default
            # does. Either way the binary is definite, and the shape says what it is.
            # Without this the loader would substitute the KMD default_value as the
            # catalog key while the kernel was compiled from something else -- two
            # independent defaults that are not required to agree, and whose
            # disagreement is silent.
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
        if isinstance(value, bool):
            value = int(value)
        if field_name in vocabulary and isinstance(value, str):
            # The matcher compares the hipDNN spelling; the spec carries the
            # builder's. Copying one over the other declines every graph while the
            # engine still loads and every count reconciles.
            value = vocabulary[field_name].get(value, value)
        metadata[field_name] = value

    # The name must be injective over everything that varies. `_check_kernel_names_
    # unique` enforces that over the whole expansion, because nothing after this does:
    # dedup keys on metadata rather than name, so a collision reaching it would ship
    # as descriptors nothing can tell apart in a log, a winner record or a failure
    # message.
    # A bool renders as `True`/`False` under str.format, but every shipped grammar
    # spells these flags `c1`/`p0`. Normalise so a template slot for `causal` reads
    # the same as the metadata mirror of the same field.
    fields = {k: int(v) if isinstance(v, bool) else v for k, v in spec.items()}
    fields.update({f"md_{k}": v for k, v in metadata.items()})
    fields["ordinal"] = ordinal + arm.get("ordinal_offset", 0)
    try:
        fields["tag"] = str(arm.get("tag", "")).format(**fields)
        template = name_template
        if not fields["tag"]:
            # An empty tag would leave `..._p0__e1` or a trailing `_`. Drop ONE
            # adjacent separator from the TEMPLATE, where the slot's position is
            # known, rather than squeezing the rendered name: a rendered-name fixup
            # cannot tell its own separator from one inside a value, so it collapses
            # a legitimate `a__b` and can land two distinct kernels on one name.
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
    return {"name": name, "kernel_source": {"spec": spec}, "metadata": metadata}


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
    """A key the loader is about to merge as a dict must actually be one.

    Without this the merge itself raises -- `ValueError: dictionary update sequence
    element #0 has length 1; 2 is required` -- from inside a dict comprehension,
    naming neither the kernel nor the key, and generate.py catches only ConfigError
    so the raw traceback reaches the author.
    """
    if not isinstance(value, dict):
        raise ConfigError(
            f"{scope} must be a mapping; got {type(value).__name__} ({value!r}). "
            f"A scalar or list here is usually a YAML indentation slip."
        )


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
        "delegates_to_existing_plan",
        "authored_subpath",
        "graph_match",
        "descriptor_files_var",
        "pack_kernels_var",
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
#: One ``variants[]`` group's keys. Closed for the same reason every other level
#: is: a typo'd key here silently expands to something other than what was meant.
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


def _reject_unknown_keys(raw: dict) -> None:
    """Refuse a key no level of this loader reads.

    Every unrecognised key was previously dropped by ``raw.get(key, default)``: the
    config generated, exit 0, a cheerful "Generated 15 files" -- and a bundle
    silently missing whatever the author thought they had configured. `engine.knobbs`
    for `engine.knobs` emits a UED with no knobs at all, and nothing anywhere says so.

    That is the worst failure this loader can have, because the author is not
    debugging: they believe the key took effect and move on. It is also the failure
    the loader already guards against for THREE specific deprecated keys -- the
    principle was accepted, just never generalised.

    Closed vocabularies, so a new key must be declared here to be honoured. That is
    the point: a typo and a genuinely new feature are indistinguishable to a loader
    that accepts anything, and only one of them should generate.
    """

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
    # Pack names must be unique. They key the pack's descriptor id AND its output
    # filename (`<engine-slug>_<pack-name>.kdp.json`), so two packs sharing a name
    # silently collide twice over: same id, and the second file overwrites the first,
    # dropping a whole pack's kernels with no error. Only `discriminator` was checked
    # for duplicates before, which does not cover a single-pack-discriminator config
    # or catch the filename collision.
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
