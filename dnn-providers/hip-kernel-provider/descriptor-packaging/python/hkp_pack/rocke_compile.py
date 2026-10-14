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


# The lowering backend the packer pins. rocKE's own default is "cpp"
# (core/backend.py `_DEFAULT_BACKEND`), but the C++ engine's pybind extension
# `rocke_engine` is deliberately not part of the rocke wheel -- "the C++ engine
# and its rocke_engine pybind binding are built via CMake, not pip; they are
# intentionally NOT part of this wheel" (platform/pyproject.toml). So in the
# wheel venv the packer runs in, a "cpp" request cannot find its engine and
# rocKE falls back to the Python lowerer with a warning on stderr.
#
# Packaged kernels must not be produced by an accident of which engine happened
# to be importable. Pin the backend explicitly, and verify it: the two engines
# are held byte-identical by rocKE's own gate (tools/check_byte_identity.py), so
# this is about provenance being true, not about the bytes differing.
_BACKEND = "python"


def _reset_backend_audit():
    """Clear rocKE's fallback ledger before a compile, if it exposes one.

    Best-effort: a rocKE without the audit API must not break packing, it just
    loses this check. _assert_no_backend_fallback degrades to a no-op with it.
    """
    try:
        from rocke.core.backend import reset_cpp_fallbacks
    except Exception:
        return
    reset_cpp_fallbacks()


def _assert_no_backend_fallback(source, builder, arch):
    """Fail if the lowering silently degraded to a different backend.

    Checked via rocKE's `cpp_fallbacks()` ledger rather than by scraping the
    warning: pytest and CMake both capture stderr, so the warning is routinely
    invisible while the ledger is not. Confirmed empirically -- a fallback that
    printed nothing under capture still registered here.
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
        # Validate membership. The allowed set is already one call away -- the
        # union arm above uses get_args on the same field -- and without this a
        # typo'd enum-ish value constructs happily and reaches codegen. Real
        # fields are exposed to it: FmhaMaskMode is a Literal, so
        # {"mode": "casual"} would build a kernel with a silently wrong mask.
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
    try:
        hints = typing.get_type_hints(cls)
    except Exception as exc:
        # One unresolvable forward reference must not kill the whole spec with a
        # bare NameError from deep inside typing. The sibling resolution in
        # _resolve_spec_class already guards this way; encountered for real
        # during review verification.
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
    """Require exactly `(spec, *, arch)` — nothing the UKD cannot supply.

    Keyword-only parameters beyond `arch` are the dangerous case. A builder that
    takes a defaulted tuning object would leave that tuning value silently frozen
    at its default on every pack, with nothing in the descriptor able to
    influence it and nothing in the output recording that.

    That is not a hypothetical: a silently defaulted knob is exactly how a real
    regression got mis-reported in this tree. Defaulting a performance knob out
    of sight is worse than refusing to build, because the artifact looks fine.

    A parameter with a default is still rejected. Having a default is what makes
    it invisible; it does not make it unimportant.
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


# Builders whose validation predicate is not derivable from the builder name.
# The tiled ``supports_tiled_2d``/``3d`` are keyword-only, so these builders use the
# spec-first ``supports_tiled_*_spec`` wrappers in the gfx942/gfx950 modules. The
# reduce and fastkv-register-P tiled builders and all gfx1250 tiled builders have no
# spec-first predicate and are unvalidated.
_PREDICATE_ALIASES = {
    "build_unified_attention_2d_tiled": "supports_tiled_2d_spec",
    "build_gfx942_4warp_gqa": "supports_gfx942_4warp_gqa_spec",
    "build_unified_attention_3d_tiled": "supports_tiled_3d_spec",
}


def _predicate_accepts(predicate, spec_cls):
    """True if `predicate(spec, arch=...)` is a call it takes, with its first
    parameter annotated with `spec_cls` or a base.

    String annotations that cannot be resolved compare by class name.
    Unannotated or keyword-only first parameters do not match, nor does a
    predicate needing more than the spec and arch (e.g. a problem).
    """
    try:
        params = list(inspect.signature(predicate).parameters.values())
    except (TypeError, ValueError):
        return False
    if not params or params[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return False
    first = params[0]
    if any(
        p.default is inspect.Parameter.empty
        and p.name != "arch"
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in params[1:]
    ):
        return False
    try:
        ann = typing.get_type_hints(predicate).get(first.name, first.annotation)
    except Exception:
        ann = first.annotation
    if isinstance(ann, str):
        return ann.strip("'\"").rsplit(".", 1)[-1] in {
            c.__name__ for c in spec_cls.__mro__
        }
    return isinstance(ann, type) and issubclass(spec_cls, ann)


def _resolve_support_predicate(module, builder):
    """Find the builder's validation predicate: alias, derived, then is_valid*.

    A candidate counts only if its first parameter is annotated with the
    builder's spec class; one module often holds several builders and several
    predicates, each written for one spec class.
    """
    spec_cls = _resolve_spec_class(module, getattr(module, builder))
    derived = (
        "supports_" + builder[len("build_") :] if builder.startswith("build_") else None
    )
    names = [_PREDICATE_ALIASES.get(builder), derived]
    names += [n for n in vars(module) if n.startswith("is_valid")]
    for name in names:
        fn = getattr(module, name, None) if name else None
        if callable(fn) and _predicate_accepts(fn, spec_cls):
            return name, fn
    return None, None


def _check_support_predicate(module, builder, spec_obj, arch):
    """Consult the builder's own support predicate before building.

    Several builders validate only from an external launcher, not inside the
    builder itself, so an out-of-envelope spec reaches codegen unchecked and
    fails late (or worse, succeeds and ships something untested).

    Only spec-first predicates can be called generically (see
    `_PREDICATE_ALIASES`); a builder with no resolvable predicate is unvalidated.
    A predicate raising AttributeError or TypeError could not read the spec, so
    the pack fails; any other exception is ignored.
    """
    name, predicate = _resolve_support_predicate(module, builder)
    if predicate is None:
        return

    params = inspect.signature(predicate).parameters
    kwargs = {"arch": arch} if "arch" in params else {}
    try:
        verdict = predicate(spec_obj, **kwargs)
    except (AttributeError, TypeError) as exc:
        raise HkpPackError(
            f"support predicate {name} for builder '{builder}' @ {arch} "
            f"could not read the spec: {type(exc).__name__}: {exc}"
        ) from exc
    except Exception:
        # Any other exception does not block packing: the predicate is a
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
