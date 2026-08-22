# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Data models for generic-kernel-ingestor descriptor generation.

Convention inherited from ``DescriptorGenerator/codegen/models.py``: a
dataclass field for anything a human might reasonably override, a
``@property`` for anything mechanically derivable from those fields (a
filename, a UUID cross-reference, a fragment name). Nothing here parses
YAML directly -- that is ``config_loader.py``'s job -- and nothing here
talks to Jinja2 -- that is ``generator.py``'s.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

#: KMD field types the loader accepts (``DescriptorLoader.hpp``'s ``MetadataField``).
KMD_FIELD_TYPES: tuple[str, ...] = ("bool", "int", "float", "string", "int_list")

#: The only fully-adapted ``kernel_source.kind`` on develop.
KERNEL_SOURCE_KIND_EMBEDDED = "embedded_source"
#: v1 ships no adapter; the config loader rejects this kind explicitly.
KERNEL_SOURCE_KIND_HSACO_FILE = "hsaco_file"
#: Every kind the *format* accepts, whether or not IngestorGenerator can emit it.
KERNEL_SOURCE_KINDS: tuple[str, ...] = (
    KERNEL_SOURCE_KIND_EMBEDDED,
    KERNEL_SOURCE_KIND_HSACO_FILE,
    "kpack",
    "rocke_builder",
)

WORKSPACE_POLICIES: tuple[str, ...] = ("none", "fixed", "derived")

#: RFC 0020 §4.2's closed vocabulary for UED ``behavior_notes``.
BEHAVIOR_NOTES: tuple[str, ...] = ("runtime_compilation",)

ENGINE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$")

#: ``DescriptorLoader.hpp``'s ``isPlausibleArchBaseId``: ``gfx`` + lowercase
#: alnum/``-``/``_``, no feature suffix. This is the *shape* check the loader
#: itself enforces; a well-formed-but-unrecognized id (``gfx94``) still passes
#: it and only trips the generator's own recognized-arch warning (rule 5).
ARCH_BASE_ID_PATTERN = re.compile(r"^gfx[a-z0-9_-]+$")

#: Known real base target ids, for the plausible-but-unrecognized warning
#: (config-loader rule 5). Deliberately conservative and easy to extend --
#: unrecognized is a WARNING, never an error, so a missing entry here never
#: blocks a legitimate new arch from being generated.
KNOWN_ARCH_BASE_IDS: frozenset[str] = frozenset(
    {
        "gfx900",
        "gfx906",
        "gfx908",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx950",
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1200",
        "gfx1201",
    }
)


def _to_pascal_case(snake: str) -> str:
    """Convert ``snake_case`` or ``kebab-case`` to ``PascalCase``."""
    parts = re.split(r"[_\-]", snake)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


@dataclass
class KmdField:
    """One ``fields[]`` entry of the engine's KMD (``*.kmd.json``).

    Declared verbatim from YAML -- everything here is something a human
    chooses; nothing is derivable.
    """

    name: str
    type: str
    #: Absent (``None``) means the field is *mandatory* on every kernel --
    #: the KMD's own semantics, not a sentinel this tool invents.
    default_value: object = None

    @property
    def is_mandatory(self) -> bool:
        return self.default_value is None

    @property
    def is_int_typed(self) -> bool:
        """Whether this field can back a usable UED knob.

        A non-``int`` knob is accepted by the loader and produces no
        ``KnobT`` at all, silently (``getCustomKnobs`` filters to
        ``int64_t`` alternatives only) -- config-loader check #2 exists
        because of exactly this property.
        """
        return self.type == "int"


@dataclass
class KernelSource:
    """A kernel's ``kernel_source`` object.

    Only ``embedded_source`` has a runtime adapter; the config loader
    rejects every other kind before a UUID is minted (see
    ``config_loader.validate_kernel_source_kind``).
    """

    kind: str
    source_file: str = ""
    entry_point: str = ""


@dataclass
class KernelSpec:
    """One kernel within a pack -- inline in the emitted KDP.

    ``metadata`` is a plain dict of KMD field name -> authored value;
    the config loader type-checks it against the engine's ``kmd_fields``
    before any UUID is minted (pre-mint check #3).
    """

    name: str
    kernel_source: KernelSource
    metadata: dict = field(default_factory=dict)
    priority: int = 0
    #: Empty means "inherit the pack's arch" -- the KDP/UKD convention.
    arch: list[str] = field(default_factory=list)


@dataclass
class PackSpec:
    """One ``packs[]`` entry -- becomes one ``*.kdp.json`` plus, when the
    engine has more than one pack, one operation-scoped UMD."""

    name: str
    kernels: list[KernelSpec] = field(default_factory=list)
    #: Empty means arch-independent (inherits nothing further -- the KDP's
    #: own ``arch`` is the outermost scope a kernel's arch narrows).
    arch: list[str] = field(default_factory=list)
    #: Native symbol suffix distinguishing this pack's operation-matcher,
    #: only meaningful (and only emitted) when the engine has >1 pack.
    #: e.g. "add" -> "hipkernel.<engine>.add_match".
    discriminator: str = ""

    @property
    def pascal_name(self) -> str:
        return _to_pascal_case(self.name)


@dataclass
class GraphMatchSpec:
    """The engine-level ``graph_match`` shape.

    ``shape`` and ``discriminator`` are documentation of *why* the config
    is laid out the way it is (mirroring ``07-descriptor-generation.md``
    §2's two structurally distinct shapes); they do not change what is
    emitted; the pack-level ``discriminator`` field on each ``PackSpec``
    is what actually drives whether an operation-scoped UMD is emitted.
    """

    shape: str = "shared_shape"
    discriminator: str = "none"


@dataclass
class EngineSpec:
    """The engine-level YAML block (``engine:`` in the config).

    This is also the shape the ``sources/`` adapters (Task 2A.3) produce,
    so a config built by hand and one inferred from a kernel source both
    resolve to this same normalized dataclass before generation.
    """

    name: str
    sdk_version: str = "1.0.0"
    behavior_notes: list[str] = field(default_factory=list)
    knobs: list[str] = field(default_factory=list)
    #: "native" -> emit a UHD scoring on a symbol; "none" -> omit the UHD
    #: entirely (legal: an engine may ship no ranking model).
    heuristic: str = "native"

    @property
    def namespace(self) -> str:
        return self.name.split(":", 1)[0]

    @property
    def local_name(self) -> str:
        return self.name.split(":", 1)[1]

    @property
    def slug(self) -> str:
        """Directory name under ``descriptors/`` -- snake_case of the local name."""
        s = re.sub(r"(?<!^)(?=[A-Z])", "_", self.local_name)
        return s.lower()

    @property
    def pascal_name(self) -> str:
        return _to_pascal_case(self.local_name)

    @property
    def camel_name(self) -> str:
        """The local name, lowerCamelCase -- e.g. ``ConvFwd`` -> ``convFwd``.
        Used as the prefix for this engine's free-function native symbol
        implementations (``convFwdGraphMatches``, ``convFwdDispatchHandler``)."""
        pascal = self.pascal_name
        return pascal[:1].lower() + pascal[1:]

    @property
    def has_heuristic(self) -> bool:
        return self.heuristic != "none"


@dataclass
class IngestorConfig:
    """Complete configuration for one engine's descriptor bundle.

    Every value a human might override is a declared field above this
    class's own fields (nested in ``engine``/``kmd_fields``/``packs``);
    every value below is mechanically derivable and is a ``@property``.
    """

    engine: EngineSpec
    kmd_fields: list[KmdField] = field(default_factory=list)
    packs: list[PackSpec] = field(default_factory=list)
    graph_match: GraphMatchSpec = field(default_factory=GraphMatchSpec)
    kernel_source_kind: str = KERNEL_SOURCE_KIND_EMBEDDED
    workspace_policy: str = "none"
    delegates_to_existing_plan: bool = False

    @property
    def is_multi_pack(self) -> bool:
        return len(self.packs) > 1

    def kdp_stem(self, pack: PackSpec) -> str:
        """The KDP file's stem (no ``.kdp.json``), mirroring the shipped
        convention: a single-pack engine names the file after the engine
        slug alone (``conv_fwd.kdp.json``); a multi-pack engine appends the
        pack name (``pointwise_add.kdp.json``)."""
        return (
            self.engine.slug
            if not self.is_multi_pack
            else f"{self.engine.slug}_{pack.name}"
        )

    @property
    def kmd_field_by_name(self) -> dict:
        return {f.name: f for f in self.kmd_fields}

    @property
    def int_typed_kmd_fields(self) -> list:
        return [f for f in self.kmd_fields if f.is_int_typed]

    @property
    def native_symbol_namespace(self) -> str:
        """The dotted namespace native symbols live under, e.g. ``hipkernel.conv_fwd``.

        Derived from the engine's scoped name: ``hipkernel:ConvFwd`` walks to
        ``hipkernel.conv_fwd`` -- the exact prefix every symbol in
        ``ConvNative.cpp`` shares.
        """
        local_snake = re.sub(r"(?<!^)(?=[A-Z])", "_", self.engine.local_name).lower()
        return f"{self.engine.namespace}.{local_snake}"

    @property
    def graph_match_symbol(self) -> str:
        return f"{self.native_symbol_namespace}.graph_match"

    @property
    def score_symbol(self) -> str:
        return f"{self.native_symbol_namespace}.score"

    @property
    def dispatch_symbol(self) -> str:
        return f"{self.native_symbol_namespace}.dispatch"

    @property
    def kernel_match_symbol(self) -> str:
        """Shared kernel-scoped dtype matcher, mirroring both shipped engines
        (one kernel-scoped UMD shared across every pack)."""
        return f"{self.native_symbol_namespace}.kernel_match"

    def operation_match_symbol(self, pack: PackSpec) -> str:
        """The per-pack operation-scoped matcher symbol, only meaningful when
        ``is_multi_pack`` (see UMD policy: a single-pack engine emits none)."""
        return f"{self.native_symbol_namespace}.{pack.discriminator}_match"

    @property
    def native_class_name(self) -> str:
        """The pack file's class-name stem, e.g. ``ConvFwd`` -> ``ConvFwdNative.cpp``."""
        return self.engine.pascal_name

    @property
    def register_symbols_fn(self) -> str:
        return f"register{self.engine.pascal_name}Symbols"

    @property
    def dispatch_handler_class(self) -> str:
        return f"{self.engine.pascal_name}DispatchHandler"
