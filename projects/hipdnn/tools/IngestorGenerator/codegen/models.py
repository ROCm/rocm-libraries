# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Data models for generic-kernel-ingestor descriptor generation."""

import re
from dataclasses import dataclass, field
from typing import Optional

#: KMD field types the loader accepts (``DescriptorLoader.hpp``'s ``MetadataField``).
KMD_FIELD_TYPES: tuple[str, ...] = ("bool", "int", "float", "string", "int_list")

#: The two authored dialects, not interchangeable: the dialect decides the
#: ``kernel_source`` key names, the ``kind`` vocabulary and the consumer.
#: ``direct_load`` is read by ``DescriptorLoader.hpp`` from the installed
#: ``test_descriptors/`` tree; ``packaged`` is read by ``hkp_pack``, which
#: compiles the kernel and rewrites the descriptor to ``kind: kpack``.
DIALECT_DIRECT_LOAD = "direct_load"
DIALECT_PACKAGED = "packaged"
DIALECTS: tuple[str, ...] = (DIALECT_DIRECT_LOAD, DIALECT_PACKAGED)

#: The one runtime-dispatchable kind authored directly, in ``direct_load``.
KERNEL_SOURCE_KIND_EMBEDDED = "embedded_source"
#: Runtime kind, never authored: ``hkp_pack`` stamps ``library``/``toc_key``/
#: ``symbol``/``sha256`` from the artifact it built, so a config naming this
#: kind is rejected.
KERNEL_SOURCE_KIND_KPACK = "kpack"
#: Rejected by the config loader: dispatch needs ``supportsSourceKind()``,
#: which does not exist on either path.
KERNEL_SOURCE_KIND_HSACO_FILE = "hsaco_file"
#: Legacy spelling of the runtime enum, parsed by the loader but dispatchable
#: by nothing; rocKE kernels are authored as ``rocke`` in the packaged dialect.
KERNEL_SOURCE_KIND_ROCKE_BUILDER = "rocke_builder"

#: ``packaged``-dialect authored kinds, matching ``hkp_pack``'s
#: ``_validate_ukd_fields`` vocabulary exactly.
KERNEL_SOURCE_KIND_HIP = "hip"
KERNEL_SOURCE_KIND_ROCKE = "rocke"
KERNEL_SOURCE_KIND_HSACO = "hsaco"

#: Every kind either dialect's format accepts, emittable here or not.
KERNEL_SOURCE_KINDS: tuple[str, ...] = (
    KERNEL_SOURCE_KIND_EMBEDDED,
    KERNEL_SOURCE_KIND_HSACO_FILE,
    KERNEL_SOURCE_KIND_KPACK,
    KERNEL_SOURCE_KIND_ROCKE_BUILDER,
    KERNEL_SOURCE_KIND_HIP,
    KERNEL_SOURCE_KIND_ROCKE,
    KERNEL_SOURCE_KIND_HSACO,
)

#: Kinds each dialect emits. Anything else is a ConfigError naming the dialect.
EMITTABLE_KINDS_BY_DIALECT: dict[str, tuple[str, ...]] = {
    DIALECT_DIRECT_LOAD: (KERNEL_SOURCE_KIND_EMBEDDED,),
    DIALECT_PACKAGED: (KERNEL_SOURCE_KIND_HIP, KERNEL_SOURCE_KIND_ROCKE),
}

WORKSPACE_POLICIES: tuple[str, ...] = ("none", "fixed", "derived")

#: Authored descriptor sets under the provider's ``test_descriptors/`` tree.
#: Each is its own pack target, so the named set decides which shard the
#: descriptors reach and therefore which binary can read them.
AUTHORED_TEST_SETS: tuple[str, ...] = (
    "shared",
    "unit",
    "integration",
    "archive_fixture",
)

#: RFC 0020 §4.2's closed vocabulary for UED ``behavior_notes``.
BEHAVIOR_NOTES: tuple[str, ...] = ("runtime_compilation",)

ENGINE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$")

#: ``native.cpp.j2`` builds ``<NAME>_FIELD`` from each kmd field name and
#: ``<NAME>_MATCHER_SYMBOL``/``<name>OperationMatches`` from each pack
#: discriminator, so a name outside this shape emits invalid C++.
CXX_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: A name that becomes one path component (directory name or file stem), not a
#: C++ identifier: hyphens are allowed and folded away by `_to_pascal_case`,
#: while ``.`` and path separators are rejected. Leading digits are settled
#: against the derived Pascal name.
PATH_STEM_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")

#: ``DescriptorLoader.hpp``'s ``isPlausibleArchBaseId``: ``gfx`` + lowercase
#: alnum/``-``/``_``, no feature suffix. A well-formed but unrecognized id
#: (``gfx94``) passes this shape check and only trips pre-mint check #5.
ARCH_BASE_ID_PATTERN = re.compile(r"^gfx[a-z0-9_-]+$")

#: Known real base target ids, for pre-mint check #5's warning. Unrecognized
#: is a warning, never an error, so a missing entry cannot block a new arch.
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

#: Architecture the emitted matcher-test device fixture uses when the config
#: names none. Concrete rather than empty: the fixture constructs a device by
#: value and a device with no arch matches nothing.
DEFAULT_FIXTURE_ARCH = "gfx942"

#: Base-id prefixes whose targets run a 32-lane wavefront; everything else is
#: 64. Mirrored by hand from rocKE's arch SSOT,
#: ``dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/core/arch/data/
#: arch_specs.json``; check there when a new target lands. Importing
#: ``rocke.core.arch`` would require the kernel toolchain.
WAVE32_ARCH_PREFIXES: tuple[str, ...] = ("gfx10", "gfx11", "gfx12")

WAVE32_SIZE = 32
WAVE64_SIZE = 64


def wave_size_for_arch(arch: str) -> int:
    """The wavefront width of ``arch``'s family. ``warpSize`` participates in
    ``DeviceKey``'s equality and hash
    (``plugin_sdk/include/hipdnn_plugin_sdk/ingestor/DeviceKey.hpp``), so arch
    and wave must be derived together."""
    return WAVE32_SIZE if arch.startswith(WAVE32_ARCH_PREFIXES) else WAVE64_SIZE


def _to_pascal_case(snake: str) -> str:
    """Convert ``snake_case`` or ``kebab-case`` to ``PascalCase``."""
    parts = re.split(r"[_\-]", snake)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


@dataclass
class KmdField:
    """One ``fields[]`` entry of the engine's KMD (``*.kmd.json``)."""

    name: str
    type: str
    #: ``None`` means the field is mandatory on every kernel; that is the KMD's
    #: own semantics.
    default_value: object = None

    @property
    def is_mandatory(self) -> bool:
        return self.default_value is None

    @property
    def is_int_typed(self) -> bool:
        """Whether this field can back a usable UED knob: ``getCustomKnobs``
        filters to ``int64_t`` alternatives, so a non-``int`` knob loads and
        produces no ``KnobT``."""
        return self.type == "int"


@dataclass
class KernelSource:
    """A kernel's ``kernel_source`` object, in either authored dialect.

    The fields are disjoint per ``kind`` and ``as_document`` writes only the
    ones that ``kind`` owns.
    """

    kind: str
    #: ``direct_load`` / ``embedded_source``.
    source_file: str = ""
    entry_point: str = ""
    #: ``packaged`` / both kinds. For ``hip``, a path relative to the
    #: descriptor naming it; for ``rocke``, a dotted Python module path
    #: resolved through the importable ``kernels`` package.
    source: str = ""
    #: ``packaged`` / ``hip``: the ``__global__`` entry point.
    entry: str = ""
    #: ``packaged`` / ``hip``: ``{"defines": {...}}``, the compile-time knobs.
    build: dict = field(default_factory=dict)
    #: ``packaged`` / ``rocke``: the builder function, which must take exactly
    #: ``(spec, *, arch)``; ``hkp_pack``'s ``_require_spec_arch_signature``
    #: refuses anything else.
    builder: str = ""
    #: ``packaged`` / ``rocke``: the builder's spec dataclass as a dict.
    #: ``hkp_pack`` hydrates it with ``Spec(**fields)``, so every non-defaulted
    #: field must be present.
    spec: dict = field(default_factory=dict)

    def as_document(self) -> dict:
        """The ``kernel_source`` JSON object for this kind, and nothing more.

        The runtime loader hard-fails an unknown key and ``hkp_pack`` validates
        a closed field set per kind.
        """
        if self.kind == KERNEL_SOURCE_KIND_EMBEDDED:
            return {
                "kind": self.kind,
                "source_file": self.source_file,
                "entry_point": self.entry_point,
            }
        if self.kind == KERNEL_SOURCE_KIND_HIP:
            return {
                "kind": self.kind,
                "source": self.source,
                "entry": self.entry,
                "build": self.build,
            }
        if self.kind == KERNEL_SOURCE_KIND_ROCKE:
            return {
                "kind": self.kind,
                "source": self.source,
                "builder": self.builder,
                "spec": self.spec,
            }
        raise ValueError(
            f"kernel_source kind '{self.kind}' has no emitter; the config "
            f"loader should have rejected it before generation"
        )


@dataclass
class KernelSpec:
    """One kernel within a pack, inline in the emitted KDP.

    ``metadata`` maps KMD field name to authored value, type-checked against
    the engine's ``kmd_fields`` by pre-mint check #3.
    """

    name: str
    kernel_source: KernelSource
    metadata: dict = field(default_factory=dict)
    priority: int = 0
    #: Empty inherits the pack's arch, per the KDP/UKD convention.
    arch: list[str] = field(default_factory=list)


@dataclass
class PackSpec:
    """One ``packs[]`` entry: one ``*.kdp.json``, plus one operation-scoped UMD
    when the engine has more than one pack."""

    name: str
    kernels: list[KernelSpec] = field(default_factory=list)
    #: Empty means arch-independent; the KDP's own ``arch`` is the outermost
    #: scope a kernel's arch narrows.
    arch: list[str] = field(default_factory=list)
    #: Native symbol suffix for this pack's operation-matcher, e.g. "add" ->
    #: "hipkernel.<engine>.add_match". Emitted only when the engine has >1 pack.
    discriminator: str = ""

    @property
    def pascal_name(self) -> str:
        return _to_pascal_case(self.name)


@dataclass
class GraphMatchSpec:
    """The engine-level ``graph_match`` shape.

    Documentation only: neither field changes what is emitted. The pack-level
    ``discriminator`` drives operation-scoped UMD emission.
    """

    shape: str = "shared_shape"
    discriminator: str = "none"


@dataclass
class EngineSpec:
    """The engine-level YAML block (``engine:`` in the config).

    Also the shape the ``sources/`` adapters produce, so hand-built and
    inferred configs resolve to the same dataclass.
    """

    name: str
    sdk_version: str = "1.0.0"
    behavior_notes: list[str] = field(default_factory=list)
    knobs: list[str] = field(default_factory=list)
    #: "native" emits a UHD scoring on a symbol; "none" omits the UHD, which is
    #: legal for an engine with no ranking model.
    heuristic: str = "native"

    @property
    def namespace(self) -> str:
        return self.name.split(":", 1)[0]

    @property
    def local_name(self) -> str:
        return self.name.split(":", 1)[1]

    @property
    def slug(self) -> str:
        """The bundle's directory name: snake_case of the local name."""
        s = re.sub(r"(?<!^)(?=[A-Z])", "_", self.local_name)
        return s.lower()

    @property
    def pascal_name(self) -> str:
        return _to_pascal_case(self.local_name)

    @property
    def camel_name(self) -> str:
        """The local name in lowerCamelCase (``ConvFwd`` -> ``convFwd``), the
        prefix for this engine's free-function native symbols
        (``convFwdGraphMatches``, ``convFwdDispatchHandler``)."""
        pascal = self.pascal_name
        return pascal[:1].lower() + pascal[1:]

    @property
    def has_heuristic(self) -> bool:
        return self.heuristic != "none"


@dataclass
class IngestorConfig:
    """Complete configuration for one engine's descriptor bundle.

    Human-overridable values are declared fields; anything derivable from them
    is a ``@property``.
    """

    engine: EngineSpec
    kmd_fields: list[KmdField] = field(default_factory=list)
    packs: list[PackSpec] = field(default_factory=list)
    graph_match: GraphMatchSpec = field(default_factory=GraphMatchSpec)
    dialect: str = DIALECT_DIRECT_LOAD
    kernel_source_kind: str = KERNEL_SOURCE_KIND_EMBEDDED
    workspace_policy: str = "none"
    #: Where the bundle is authored, in the tree its dialect writes into.
    #: ``packaged``: subpath under the packager's single source root, e.g.
    #: ``rocKE/gfx950_attention_dense``, preserved verbatim into the staged and
    #: installed trees; defaults to ``<kind>/<slug>``.
    #: ``direct_load``: required, one of `AUTHORED_TEST_SETS`, since the set is
    #: the consuming binary's choice.
    authored_subpath: str = ""
    specialization: dict = field(default_factory=dict)

    @property
    def is_packaged(self) -> bool:
        return self.dialect == DIALECT_PACKAGED

    @property
    def is_multi_pack(self) -> bool:
        return len(self.packs) > 1

    @property
    def descriptor_dir(self) -> str:
        """Where this bundle's descriptor files go, relative to the output dir:
        ``direct_load`` under the authored set in ``test_descriptors/``,
        ``packaged`` mirroring the packager's source root."""
        if not self.is_packaged:
            return f"test_descriptors/{self.authored_subpath}/{self.engine.slug}"
        subpath = (
            self.authored_subpath or f"{self.kernel_source_kind}/{self.engine.slug}"
        )
        return f"descriptors/{subpath}"

    def kdp_stem(self, pack: PackSpec) -> str:
        """The KDP file's stem (no ``.kdp.json``): the engine slug for a
        single-pack engine (``conv_fwd.kdp.json``), slug plus pack name for a
        multi-pack engine (``pointwise_add.kdp.json``)."""
        return (
            self.engine.slug
            if not self.is_multi_pack
            else f"{self.engine.slug}_{pack.name}"
        )

    @property
    def device_fixture_arch(self) -> str:
        """Architecture named by the emitted matcher-test device fixture: the
        first pack's first arch, or `DEFAULT_FIXTURE_ARCH` when the config
        restricts none."""
        first_pack_arch = self.packs[0].arch if self.packs else []
        return first_pack_arch[0] if first_pack_arch else DEFAULT_FIXTURE_ARCH

    @property
    def device_fixture_wave_size(self) -> int:
        """That fixture's ``warpSize``, derived from `device_fixture_arch` so a
        wave32 target never gets a hard-coded wave64 device."""
        return wave_size_for_arch(self.device_fixture_arch)

    @property
    def kmd_field_by_name(self) -> dict:
        return {f.name: f for f in self.kmd_fields}

    @property
    def int_typed_kmd_fields(self) -> list:
        return [f for f in self.kmd_fields if f.is_int_typed]

    @property
    def native_symbol_namespace(self) -> str:
        """Dotted namespace native symbols live under: ``hipkernel:ConvFwd``
        becomes ``hipkernel.conv_fwd``."""
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
        """Kernel-scoped dtype matcher, shared across every pack."""
        return f"{self.native_symbol_namespace}.kernel_match"

    def operation_match_symbol(self, pack: PackSpec) -> str:
        """Per-pack operation-scoped matcher symbol, meaningful only when
        ``is_multi_pack``; a single-pack engine emits none."""
        return f"{self.native_symbol_namespace}.{pack.discriminator}_match"

    @property
    def native_class_name(self) -> str:
        """The pack file's class-name stem, e.g. ``ConvFwd`` for
        ``ConvFwdNative.cpp``."""
        return self.engine.pascal_name

    @property
    def register_symbols_fn(self) -> str:
        return f"register{self.engine.pascal_name}Symbols"

    @property
    def dispatch_handler_class(self) -> str:
        return f"{self.engine.pascal_name}DispatchHandler"
