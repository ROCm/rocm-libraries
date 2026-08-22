import dataclasses
import inspect
import typing
from importlib import import_module
from pathlib import Path

from .errors import HkpPackError
from .variant import _hash_payload

try:
    from types import UnionType as _UnionType
except ImportError:  # pragma: no cover
    _UnionType = None


def _is_union(origin):
    if origin is typing.Union:
        return True
    return _UnionType is not None and origin is _UnionType


def _build_field(field_type, value):
    origin = typing.get_origin(field_type)
    if _is_union(origin):
        if value is None:
            return None
        non_none = [a for a in typing.get_args(field_type) if a is not type(None)]
        if len(non_none) == 1:
            return _build_field(non_none[0], value)
        raise HkpPackError(
            f"unsupported spec field type {field_type!r} (multi-arm union)"
        )
    if origin is typing.Literal:
        return value
    if origin in (list, tuple) or field_type in (list, tuple):
        raise HkpPackError(
            f"unsupported spec field type {field_type!r} (list/tuple not supported)"
        )
    if dataclasses.is_dataclass(field_type):
        return build_spec(field_type, value)
    return value


def build_spec(cls, data):
    """Construct a builder spec dataclass from a UKD spec dict, recursively.

    Walks the target dataclass's fields, resolving each field's type via
    typing.get_type_hints (not field.type, which is a string under
    `from __future__ import annotations`) and dispatching: scalar -> passthrough;
    nested dataclass -> recurse into a real instance; Optional[X] -> None or a
    built X; Literal / other plain -> passthrough. A list/tuple field type and an
    input key that is not a field are hard-rejected. Missing/mis-typed fields and
    a spec __post_init__ rejection propagate from cls(**kwargs) for the caller to
    wrap. No rocke import: directly unit-testable with local stub dataclasses.
    """
    field_names = {f.name for f in dataclasses.fields(cls)}
    extra = set(data) - field_names
    if extra:
        raise HkpPackError(
            f"unexpected spec field(s) for {cls.__name__}: {sorted(extra)}"
        )
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _build_field(hints[f.name], data[f.name])
    return cls(**kwargs)


def rocke_variant_key(source, builder, spec):
    """Stable input hash over (source, builder, spec) for a rocke variant.

    Keyed on all three: two rocke UKDs sharing source+spec but naming different
    builders produce different kernels and must not collapse to one blob, so the
    builder is part of the key. The nested spec dict hashes deterministically
    (sort_keys) regardless of key order.
    """
    return _hash_payload(
        Path(source).stem,
        {"source": source, "builder": builder, "spec": spec},
    )


def _resolve_spec_class(module, builder_fn):
    """The builder's spec dataclass, from its first-parameter type hint.

    A future UKD `spec_class` override would resolve here, ahead of the type-hint
    lookup; that seam is intentionally left unbuilt.
    """
    try:
        hints = typing.get_type_hints(builder_fn)
    except Exception:
        hints = {}
    params = [n for n in inspect.signature(builder_fn).parameters if n != "arch"]
    spec_cls = hints.get(params[0]) if params else None
    if spec_cls is None or not dataclasses.is_dataclass(spec_cls):
        raise HkpPackError(
            f"spec type not introspectable for builder '{builder_fn.__name__}' "
            "(first parameter needs a dataclass type hint)"
        )
    return spec_cls


def _require_spec_arch_signature(builder_fn, builder):
    params = inspect.signature(builder_fn).parameters
    names = list(params)
    non_arch = [
        n
        for n, p in params.items()
        if n != "arch"
        and p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if "arch" not in params or not names or names[0] == "arch" or len(non_arch) != 1:
        raise HkpPackError(f"builder signature must be (spec, *, arch) for '{builder}'")


def _load_compiler():
    """Lazy handle for the rocke compile entrypoint and its comgr error type.

    Behind a function so the hip-only path never imports rocke and tests can
    substitute a stub compiler.
    """
    from rocke.helpers import compile_kernel
    from rocke.runtime.comgr import ComgrError

    return compile_kernel, ComgrError


def _resolved_comgr_path():
    """Best-effort path of the comgr the rocke loader resolved, for diagnostics.

    Returns 'unknown' rather than raising when rocke is not importable, so a
    comgr compile error is never masked by a secondary import failure while
    reporting where comgr came from.
    """
    try:
        from rocke.runtime.comgr import resolved_lib_path

        return resolved_lib_path()
    except Exception:
        return "<unknown>"


def _module_from_source(source):
    stem = source[:-3] if source.endswith(".py") else source
    return ".".join(stem.split("/"))


def compile_rocke_variant(source, builder, spec, arch, out_dir):
    """Compile one rocke UKD variant for one arch, returning (co_path, symbol).

    Imports the builder module named by `source` — a dotted module path resolved
    through the importable `kernels` package, never a file path under the source
    root — resolves `builder`, introspects and constructs its spec dataclass from
    the UKD `spec` dict, calls the builder for a KernelDef, and lowers it via
    rocke's comgr `compile_kernel`. Writes the HSACO to <rocke_variant_key>.co and
    returns that path plus the captured launch symbol (`artifact.kernel_name`).
    Every deviation is a hard HkpPackError.
    """
    dotted = _module_from_source(source)
    try:
        module = import_module(dotted)
    except Exception as exc:
        raise HkpPackError(
            f"module not importable: '{source}' (as '{dotted}'): {exc}"
        ) from exc

    try:
        builder_fn = getattr(module, builder)
    except AttributeError as exc:
        raise HkpPackError(
            f"builder not found: '{builder}' in module '{dotted}'"
        ) from exc

    spec_cls = _resolve_spec_class(module, builder_fn)
    _require_spec_arch_signature(builder_fn, builder)

    try:
        spec_obj = build_spec(spec_cls, spec)
    except HkpPackError:
        raise
    except Exception as exc:
        raise HkpPackError(f"invalid spec for {spec_cls.__name__}: {exc}") from exc

    try:
        kernel = builder_fn(spec_obj, arch=arch)
    except NotImplementedError as exc:
        raise HkpPackError(
            f"arch not supported by builder '{builder}' @ {arch}: {exc}"
        ) from exc
    except Exception as exc:
        raise HkpPackError(
            f"builder call failed ({type(exc).__name__}): {exc}"
        ) from exc

    compile_kernel, ComgrError = _load_compiler()
    try:
        artifact = compile_kernel(kernel, arch=arch, capture_ir_text=False)
    except ComgrError as exc:
        raise HkpPackError(
            f"comgr compile failed for {source} @ {arch}: {exc} "
            f"(comgr loaded from {_resolved_comgr_path()}; set ROCKE_COMGR_LIB "
            "to override)"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    co_path = out_dir / f"{rocke_variant_key(source, builder, spec)}.co"
    co_path.write_bytes(artifact.hsaco)
    return co_path, artifact.kernel_name
