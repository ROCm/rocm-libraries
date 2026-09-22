import dataclasses
import hashlib
import inspect
import typing
from importlib import import_module
from pathlib import Path

from .errors import HkpPackError
from .variant import _hash_payload
from .agreement import OriginObserver, observe

try:
    from types import UnionType as _UnionType
except ImportError:  # pragma: no cover
    _UnionType = None


def _is_union(origin):
    if origin is typing.Union:
        return True
    return _UnionType is not None and origin is _UnionType


# Pin the backend explicitly: rocKE defaults to "cpp" (core/backend.py), but
# rocke_engine is built via CMake and excluded from the wheel (platform/pyproject.toml).
# Without that extension, a "cpp" request warns and falls back to Python. Pinning
# keeps the artifact's backend record accurate; tools/check_byte_identity.py checks
# byte identity between backends.
_BACKEND = "python"


def _reset_backend_audit():
    """Clear rocKE's fallback ledger before a compile, if it exposes one.

    Best-effort: a rocKE without the audit API loses the check rather than
    breaking packing.
    """
    try:
        from rocke.core.backend import reset_cpp_fallbacks
    except Exception:
        return
    reset_cpp_fallbacks()


def _assert_no_backend_fallback(source, builder, arch):
    """Fail if the lowering silently degraded to a different backend.

    Checked via rocKE's `cpp_fallbacks()` ledger rather than by scraping the
    warning, which pytest and CMake routinely capture.
    """
    try:
        from rocke.core.backend import cpp_fallbacks
    except Exception:
        return
    fallbacks = cpp_fallbacks()
    if not fallbacks:
        return
    detail = "; ".join(f"{name}: {reason}" for name, reason in fallbacks)
    raise HkpPackError(
        f"lowering backend fell back while compiling {builder} from {source} "
        f"@ {arch}: requested '{_BACKEND}' but rocke recorded a backend "
        f"fallback [{detail}]. A packaged kernel must record the engine that "
        "actually produced it."
    )


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
        # Validate membership; the allowed set is one call away. Without this a
        # typo'd enum-ish value constructs happily and reaches codegen: FmhaMaskMode
        # is a Literal, so {"mode": "casual"} would build a silently wrong mask.
        allowed = typing.get_args(field_type)
        if value not in allowed:
            raise HkpPackError(
                f"invalid value {value!r} for Literal field; "
                f"expected one of {list(allowed)}"
            )
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

    Field types resolve via typing.get_type_hints (not field.type, a string under
    `from __future__ import annotations`): scalar passes through, nested dataclass
    recurses, Optional[X] gives None or a built X. A list/tuple field type and an
    unknown input key are hard-rejected; missing fields and a __post_init__
    rejection propagate from cls(**kwargs). No rocke import.
    """
    field_names = {f.name for f in dataclasses.fields(cls)}
    extra = set(data) - field_names
    if extra:
        raise HkpPackError(
            f"unexpected spec field(s) for {cls.__name__}: {sorted(extra)}"
        )
    try:
        hints = typing.get_type_hints(cls)
    except Exception as exc:
        # One unresolvable forward reference must not kill the whole spec with a
        # bare NameError from deep inside typing.
        raise HkpPackError(
            f"cannot resolve type hints for {cls.__name__} "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _build_field(hints[f.name], data[f.name])
    return cls(**kwargs)


def rocke_variant_key(source, builder, spec):
    """Stable input hash over (source, builder, spec) for a rocke variant.

    All three are keyed: two rocke UKDs sharing source+spec but naming different
    builders produce different kernels and must not collapse to one blob.
    """
    return _hash_payload(
        Path(source).stem,
        {"source": source, "builder": builder, "spec": spec},
    )


def _resolve_spec_class(module, builder_fn):
    """The builder's spec dataclass, from its first-parameter type hint.

    A future UKD `spec_class` override would resolve here first; that seam is
    intentionally left unbuilt.
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
    """Require exactly `(spec, *, arch)` -- nothing the UKD cannot supply.

    Keyword-only parameters beyond `arch` are the dangerous case: a defaulted
    tuning object stays frozen at its default on every pack with nothing in the
    descriptor able to influence it and nothing in the output recording it. Having
    a default is what makes it invisible, so it is still rejected.
    """
    params = inspect.signature(builder_fn).parameters
    names = list(params)
    positional = [
        n
        for n, p in params.items()
        if n != "arch"
        and p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if "arch" not in params or not names or names[0] == "arch" or len(positional) != 1:
        raise HkpPackError(f"builder signature must be (spec, *, arch) for '{builder}'")

    unsuppliable = [
        n
        for n, p in params.items()
        if n != "arch" and p.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    if unsuppliable:
        raise HkpPackError(
            f"builder '{builder}' takes keyword-only parameter(s) "
            f"{', '.join(sorted(unsuppliable))} that a UKD cannot supply; they "
            "would be silently frozen at their defaults. Either fold them into "
            "the spec dataclass, or drop them from the builder's signature."
        )

    var_kinds = [
        n
        for n, p in params.items()
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if var_kinds:
        raise HkpPackError(
            f"builder '{builder}' takes *args/**kwargs ({', '.join(var_kinds)}); "
            "its real parameter set is not introspectable, so the packer cannot "
            "prove the UKD supplies everything that affects the kernel."
        )


def _load_compiler():
    """Lazy handle for the rocke compile entrypoint and its comgr error type, so
    the hip-only path never imports rocke and tests can substitute a stub.
    """
    from rocke.helpers import compile_kernel
    from rocke.runtime.comgr import ComgrError

    return compile_kernel, ComgrError


def _resolved_comgr_path():
    """Best-effort path of the comgr the rocke loader resolved, for diagnostics.

    Returns 'unknown' rather than raising, so a comgr compile error is never masked
    by a secondary import failure.
    """
    try:
        from rocke.runtime.comgr import resolved_lib_path

        return resolved_lib_path()
    except Exception:
        return "<unknown>"


def _module_from_source(source):
    stem = source[:-3] if source.endswith(".py") else source
    return ".".join(stem.split("/"))


# Builders whose validation predicate is not derivable from the builder name. The
# tiled family is the largest kernel group in the corpus and none of its
# predicates follow the naming convention, so name derivation alone reaches 14 of
# 30 builders.
_PREDICATE_ALIASES = {
    "build_unified_attention_2d_tiled": "supports_tiled_2d",
    "build_gfx942_4warp_gqa": "supports_tiled_2d",
    "build_unified_attention_3d_tiled": "supports_tiled_3d",
    "build_unified_attention_reduce_tiled": "supports_tiled_3d",
    "build_unified_attention_2d_fastkv_register_p": "supports_fastkv_register_p_2d",
}


def _resolve_support_predicate(module, builder):
    """Find a builder's validation predicate: is_valid_spec, derived, or alias."""
    derived = (
        "supports_" + builder[len("build_") :] if builder.startswith("build_") else None
    )
    for name in ("is_valid_spec", derived, _PREDICATE_ALIASES.get(builder)):
        if not name:
            continue
        fn = getattr(module, name, None)
        if callable(fn):
            return name, fn
    return None, None


def _check_support_predicate(module, builder, spec_obj, arch):
    """Consult the builder's own support predicate before building.

    Several builders validate only from an external launcher, so an
    out-of-envelope spec otherwise reaches codegen unchecked.

    Only spec-shaped predicates are callable generically: the tiled family's
    `supports_tiled_2d/3d` take keyword-only parameters with no spec object and are
    skipped rather than guessed at, leaving coverage partial (roughly 14 of 30
    builders) rather than validating the wrong thing.
    """
    name, predicate = _resolve_support_predicate(module, builder)
    if predicate is None:
        return

    try:
        params = inspect.signature(predicate).parameters
    except (TypeError, ValueError):
        return

    ordered = list(params.values())
    if not ordered:
        return
    first = ordered[0]
    if first.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        # kwargs-only predicate: not callable from a spec instance.
        return

    kwargs = {"arch": arch} if "arch" in params else {}
    try:
        verdict = predicate(spec_obj, **kwargs)
    except Exception:
        # A predicate that cannot run must not block packing: it is a
        # pre-flight check, and the builder itself remains the real gate.
        return

    ok, reason = verdict if isinstance(verdict, tuple) else (verdict, "")
    if not ok:
        raise HkpPackError(
            f"spec rejected by {name} for builder '{builder}' @ {arch}"
            + (f": {reason}" if reason else "")
        )


def compile_rocke_variant(
    source, builder, spec, arch, out_dir, requests=None, origins=None
):
    """Compile one variant, returning (code object, captured symbol, observations).

    Imports the builder module named by `source` -- a dotted module path resolved
    through the importable `kernels` package, never a file path -- resolves
    `builder`, constructs its spec dataclass from the UKD `spec` dict, calls the
    builder for a KernelDef, and lowers it via rocke's comgr `compile_kernel`.
    Writes the HSACO to <rocke_variant_key>.co and returns that path plus
    `artifact.kernel_name`. Every deviation is a hard HkpPackError.
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

    _check_support_predicate(module, builder, spec_obj, arch)
    # Observed BEFORE the builder runs, on the object `builder_fn` is about to be
    # handed: reading the same attributes afterwards would observe whatever the
    # builder left behind.
    origins = origins if origins is not None else OriginObserver()
    observations = observe(spec_obj, builder_fn, requests or {}, origins)

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
    _reset_backend_audit()
    try:
        artifact = compile_kernel(
            kernel, arch=arch, capture_ir_text=False, backend=_BACKEND
        )
    except ComgrError as exc:
        raise HkpPackError(
            f"comgr compile failed for {source} @ {arch}: {exc} "
            f"(comgr loaded from {_resolved_comgr_path()}; set ROCKE_COMGR_LIB "
            "to override)"
        ) from exc
    _assert_no_backend_fallback(source, builder, arch)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    co_path = out_dir / f"{rocke_variant_key(source, builder, spec)}.co"
    co_path.write_bytes(artifact.hsaco)
    origins.stable()
    # The arch, captured symbol and code object identify which compile these
    # observations came from; a reader binds all three to the shipped descriptor.
    observations["arch"] = arch
    observations["symbol"] = artifact.kernel_name
    observations["code_object_sha256"] = hashlib.sha256(artifact.hsaco).hexdigest()
    return co_path, artifact.kernel_name, observations
