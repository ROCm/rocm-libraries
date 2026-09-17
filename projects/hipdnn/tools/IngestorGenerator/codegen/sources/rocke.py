# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Introspects a rocKE builder module for the descriptor fields it implies.

Unlike ``hiprtc``, this adapter does not scan text. rocKE builders carry the
answer in their own type annotations, so the extraction is exact:

1. The builder's first non-``arch`` parameter is annotated with its spec
   dataclass. ``hkp_pack``'s ``_resolve_spec_class`` reads exactly that
   annotation and hard-fails when it is not a dataclass, so reading it here
   agrees with the packager by construction rather than by convention.
2. That dataclass's fields ARE the ``spec`` block a descriptor must carry, and
   its non-defaulted fields are the ones a descriptor MUST supply -- the
   packager hydrates with ``Spec(**fields)``, so a missing required field is a
   ``TypeError`` at pack time.
3. The builder must take exactly ``(spec, *, arch)``. Checked here so the
   author learns at config time instead of after a comgr run.

Everything the design reserves for a human stays reserved: this adapter never
picks the engine name, the arch list, the knobs, or the UMD-vs-graph_match
split. It reports what the builder can prove about itself.

rocKE is imported lazily and only when this adapter runs. IngestorGenerator
does not depend on rocKE being importable; a config that names no rocKE source
never reaches this module.
"""

import dataclasses
import importlib
import inspect
import typing
from pathlib import Path

from .base import CandidateKernel, SourceAdapterResult


class RockeIntrospectionError(RuntimeError):
    """A rocKE module/builder could not be introspected.

    Distinct from ``ConfigError``: this means the *source* could not be read,
    not that the config is malformed. The caller turns it into whichever it is.
    """


def module_path_from_source(source: str) -> str:
    """``kernels/gfx950/attention_dense.py`` -> ``kernels.gfx950.attention_dense``.

    Mirrors ``hkp_pack.rocke_compile._module_from_source``. A descriptor's
    ``source`` for ``kind: rocke`` is a dotted module path resolved through the
    importable ``kernels`` package -- NOT a file under the descriptor root --
    but authors reliably write it slash-style, so both spellings are accepted
    here and normalized to the dotted form the packager imports.
    """
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
    #: ``None`` when the field has no default -- i.e. a descriptor MUST set it.
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
    #: and is spec-shaped. Empty means "could not be determined" -- NOT "none".
    #: rocKE declares arch support nowhere; it is only ever derived by asking.
    supported_arches: list[str] = dataclasses.field(default_factory=list)

    @property
    def required_fields(self) -> list[SpecField]:
        return [f for f in self.fields if f.required]


def _type_name(annotation) -> str:
    if annotation is inspect.Parameter.empty:
        return "unknown"
    return getattr(annotation, "__name__", None) or str(annotation)


def _check_spec_arch_signature(builder_fn, builder: str) -> str:
    """Mirror of ``hkp_pack._require_spec_arch_signature``, returning the reason.

    Kept as a returned string rather than a raise so the adapter can report a
    complete picture (fields AND the signature problem) in one pass instead of
    dying on the first fault.
    """
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
    """Ask the module's ``supports_*`` predicate which arches it accepts.

    rocKE declares arch support NOWHERE -- not in a spec, not in a manifest --
    so the only honest way to learn it is to construct a spec and ask.

    ``spec_values`` is the config's real ``spec`` block when there is one. That
    matters: a spec's ``__post_init__`` validates its own fields, so a
    synthesized placeholder (``1`` everywhere) usually raises and the probe
    reports nothing. Asking about the ACTUAL spec an author is packaging is
    both answerable and the only question worth asking -- arch support is a
    property of a spec, not of a builder.

    Returns empty for "could not determine", never for "unsupported". Partial
    coverage is expected and documented in ``hkp_pack.rocke_compile``: the
    tiled family's predicates take individual keyword-only args and cannot be
    called generically at all.
    """
    predicate = getattr(module, builder.replace("build_", "supports_", 1), None)
    if predicate is None or not callable(predicate):
        return []
    try:
        spec = spec_cls(**spec_values) if spec_values else None
    except Exception:
        spec = None
    if spec is None:
        # No real spec to ask about (or it did not construct). Fall back to a
        # required-fields-only placeholder, which works for the specs whose
        # __post_init__ is permissive and harmlessly reports nothing otherwise.
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


#: Arches worth asking a predicate about. Not a support claim -- just the
#: question set. A predicate that accepts none of these reports empty, which
#: the caller must treat as "unknown", never as "unsupported".
_CANDIDATE_ARCHES = ("gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1250")


def introspect(source: str, builder: str, spec_values=None) -> RockeBuilderInfo:
    """Import ``source``, resolve ``builder``, and report what it proves.

    ``source`` may be dotted or slash-style; both normalize to the dotted
    module path the packager imports. Pass ``spec_values`` -- the config's own
    ``spec`` block -- to make the arch probe answerable: arch support is a
    property of a specific spec, so asking about a synthesized one usually just
    trips the spec's own validation.
    """
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
    """Reports a rocKE builder's spec surface as generator candidates.

    Constructed with the ``source``/``builder`` pair a descriptor will name,
    because a rocKE kernel is identified by module+function, not by a file on
    disk -- so ``infer()`` takes no paths (the ``SourceAdapter`` protocol's
    ``*sources`` is accepted and ignored).
    """

    def __init__(self, source: str, builder: str, spec_values=None):
        self.source = source
        self.builder = builder
        self.spec_values = spec_values

    def infer(self, *sources: Path) -> SourceAdapterResult:
        info = introspect(self.source, self.builder, self.spec_values)
        if info.signature_error:
            raise RockeIntrospectionError(info.signature_error)
        # Every spec field is a candidate KMD field: it is exactly what varies
        # between two instantiations of this kernel. Which ones become real KMD
        # fields, and which become knobs, stays a human decision.
        return SourceAdapterResult(
            kernels=[
                CandidateKernel(
                    entry_point=info.builder,
                    source_file=info.module,
                    template_params=[f.name for f in info.fields],
                )
            ],
            # One builder is one operation: its variants differ only in spec
            # values, which is the single-pack shape.
            suggested_pack_count=1,
        )
