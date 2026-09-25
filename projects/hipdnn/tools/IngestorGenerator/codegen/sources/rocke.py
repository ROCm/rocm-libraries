# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Introspects a rocKE builder module for the descriptor fields it implies.

The builder's first non-``arch`` parameter annotation is its spec dataclass --
the same annotation ``hkp_pack._resolve_spec_class`` reads -- and that
dataclass's fields are the descriptor's ``spec`` block, its non-defaulted
fields required because the packager hydrates with ``Spec(**fields)``. The
signature must be exactly ``(spec, *, arch)``.

rocKE is imported lazily, so IngestorGenerator does not require it.
"""

import dataclasses
import importlib
import inspect
import typing
from pathlib import Path

from .base import CandidateKernel, SourceAdapterResult


class RockeIntrospectionError(RuntimeError):
    """A rocKE module/builder could not be introspected.

    Distinct from ``ConfigError``: the source could not be read, rather than
    the config being malformed.
    """


def module_path_from_source(source: str) -> str:
    """``kernels/gfx950/attention_dense.py`` -> ``kernels.gfx950.attention_dense``.
    Mirrors ``hkp_pack.rocke_compile._module_from_source``: a ``kind: rocke``
    ``source`` is a dotted module path, and slash-style is normalized."""
    text = source.strip()
    if text.endswith(".py"):
        text = text[: -len(".py")]
    return text.replace("/", ".").replace("\\", ".").strip(".")


@dataclasses.dataclass
class SpecField:
    """One field of a rocKE builder's spec dataclass."""

    name: str
    #: The annotation's readable name (``int``, ``str``, ``bool``, ...).
    type_name: str
    #: ``None`` when the field has no default, i.e. a descriptor must set it.
    default: object = None
    required: bool = False


@dataclasses.dataclass
class RockeBuilderInfo:
    """Everything the builder proves about itself."""

    module: str
    builder: str
    spec_class: str
    fields: list[SpecField] = dataclasses.field(default_factory=list)
    #: Empty when the builder satisfies ``(spec, *, arch)``; otherwise the
    #: packager's own rejection reason, raised before any config is written.
    signature_error: str = ""
    #: Arches the module's ``supports_*`` predicate accepts, when one exists
    #: and is spec-shaped. Empty means "undetermined", not "none".
    supported_arches: list[str] = dataclasses.field(default_factory=list)

    @property
    def required_fields(self) -> list[SpecField]:
        return [f for f in self.fields if f.required]


def _type_name(annotation) -> str:
    if annotation is inspect.Parameter.empty:
        return "unknown"
    return getattr(annotation, "__name__", None) or str(annotation)


def _check_spec_arch_signature(builder_fn, builder: str) -> str:
    """Mirror of ``hkp_pack._require_spec_arch_signature``, returning the
    reason instead of raising so fields and the signature problem are reported
    in one pass."""
    params = inspect.signature(builder_fn).parameters
    names = list(params)
    positional = [
        n
        for n, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if "arch" not in params or not names or names[0] == "arch" or len(positional) != 1:
        return (
            f"builder '{builder}' must take exactly (spec, *, arch); got "
            f"({', '.join(names)})"
        )
    unsuppliable = [
        n
        for n, p in params.items()
        if n != "arch" and p.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    if unsuppliable:
        return (
            f"builder '{builder}' takes keyword-only parameter(s) "
            f"{', '.join(sorted(unsuppliable))} that a descriptor cannot supply; "
            f"they would be silently frozen at their defaults. Fold them into "
            f"the spec dataclass, or drop them from the signature. "
            f"(hkp_pack refuses such a builder rather than pack it.)"
        )
    return ""


def _resolve_spec_class(module, builder_fn):
    """The builder's spec dataclass, from its first non-``arch`` annotation."""
    try:
        hints = typing.get_type_hints(builder_fn)
    except Exception:
        hints = {}
    params = [n for n in inspect.signature(builder_fn).parameters if n != "arch"]
    spec_cls = hints.get(params[0]) if params else None
    if spec_cls is None or not dataclasses.is_dataclass(spec_cls):
        raise RockeIntrospectionError(
            f"builder's first parameter is not annotated with a spec dataclass "
            f"(got {spec_cls!r}); hkp_pack resolves the spec class the same way "
            f"and would refuse this builder"
        )
    return spec_cls


def _probe_supported_arches(
    module, spec_cls, builder: str, spec_values=None
) -> list[str]:
    """Ask the module's ``supports_*`` predicate which arches it accepts, the
    only place rocKE declares arch support.

    Arch support is a property of a spec, so pass ``spec_values`` (the config's
    ``spec`` block); a synthesized placeholder usually fails the spec's
    ``__post_init__``. An empty result means "undetermined", not "unsupported".
    """
    predicate = getattr(module, builder.replace("build_", "supports_", 1), None)
    if predicate is None or not callable(predicate):
        return []
    try:
        spec = spec_cls(**spec_values) if spec_values else None
    except Exception:
        spec = None
    if spec is None:
        # No constructible spec: fall back to a required-fields-only
        # placeholder, which answers for permissive specs and reports nothing
        # otherwise.
        required = [
            f.name
            for f in dataclasses.fields(spec_cls)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
        ]
        try:
            spec = spec_cls(**{name: 1 for name in required})
        except Exception:
            return []
    accepted = []
    for arch in _CANDIDATE_ARCHES:
        try:
            ok = predicate(spec, arch=arch)
        except Exception:
            continue
        if isinstance(ok, tuple):
            ok = ok[0]
        if ok:
            accepted.append(arch)
    return accepted


#: Arches worth asking a predicate about; not a support claim.
_CANDIDATE_ARCHES = ("gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1250")


def introspect(source: str, builder: str, spec_values=None) -> RockeBuilderInfo:
    """Import ``source`` (dotted or slash-style), resolve ``builder``, and
    report what it proves. ``spec_values`` -- the config's ``spec`` block --
    makes the arch probe answerable; see `_probe_supported_arches`."""
    dotted = module_path_from_source(source)
    try:
        module = importlib.import_module(dotted)
    except Exception as exc:
        raise RockeIntrospectionError(
            f"module not importable: '{source}' (as '{dotted}'): {exc}. "
            f"The rocKE library must be on PYTHONPATH -- for a source tree that "
            f"is <provider>/rocke/library plus <provider>/rocke/platform/python."
        ) from exc

    builder_fn = getattr(module, builder, None)
    if builder_fn is None:
        available = sorted(n for n in dir(module) if n.startswith("build_"))
        raise RockeIntrospectionError(
            f"builder not found: '{builder}' in '{dotted}'. "
            f"Available: {', '.join(available) or '(none)'}"
        )

    spec_cls = _resolve_spec_class(module, builder_fn)
    fields = []
    for f in dataclasses.fields(spec_cls):
        required = (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
        )
        fields.append(
            SpecField(
                name=f.name,
                type_name=_type_name(f.type if not isinstance(f.type, str) else f.type),
                default=None if required else f.default,
                required=required,
            )
        )

    return RockeBuilderInfo(
        module=dotted,
        builder=builder,
        spec_class=spec_cls.__name__,
        fields=fields,
        signature_error=_check_spec_arch_signature(builder_fn, builder),
        supported_arches=_probe_supported_arches(
            module, spec_cls, builder, spec_values
        ),
    )


class RockeAdapter:
    """Reports a rocKE builder's spec surface as generator candidates. A rocKE
    kernel is identified by module+function, so the adapter takes the
    descriptor's ``source``/``builder`` pair and ``infer()`` ignores
    ``*sources``."""

    def __init__(self, source: str, builder: str, spec_values=None):
        self.source = source
        self.builder = builder
        self.spec_values = spec_values

    def infer(self, *sources: Path) -> SourceAdapterResult:
        info = introspect(self.source, self.builder, self.spec_values)
        if info.signature_error:
            raise RockeIntrospectionError(info.signature_error)
        # Every spec field is a candidate KMD field: it is what varies between
        # instantiations of this kernel.
        return SourceAdapterResult(
            kernels=[
                CandidateKernel(
                    entry_point=info.builder,
                    source_file=info.module,
                    template_params=[f.name for f in info.fields],
                )
            ],
            # One builder is one operation; variants differ only in spec
            # values, which is the single-pack shape.
            suggested_pack_count=1,
        )
