"""Audit the Python/C++ launch-contract restatement, per surface, per op.

A rocKE kernel is launched from Python and the C++ ingestor engine restates
that launch contract by hand -- grid shape, block size, kernarg order, baked
constants, spec resolution, applicability. Nothing else compares the two
halves, and a mismatch does not fail: the kernel runs and computes something
else.

A ``launch_surface:`` block declares one entry per surface (grid, block,
kernargs, baked_constants, spec_resolution, applicability, or any further
split an op needs):

    launch_surface:
      - name: grid
        python_source: kernels/gfx942/attention_dense.py:attention_dense_grid
        cpp_mirror: dnn-providers/.../Gfx942AttentionDenseGeometry.hpp:attentionDenseGeometry
        kmd_fields: [block_m, seqlen_q, num_query_heads, batch, persistent]
        guard: attentionDenseGeometry throws on a non-positive dimension
        test: dnn-providers/.../TestGfx942AttentionDenseGeometry.cpp

Write ``guard: none`` / ``test: none`` rather than omitting the key, so an
unguarded surface is a visible choice.

``--check`` verifies, with no device and no build, that the fields a mirror
needs are declared, that the files and symbols named are real, and that every
unguarded surface is named. It never compares what the two halves compute, and
it notices a missing surface only when that surface uniquely covered a
metadata field a still-declared mirror reads through a required accessor.

    launch_surface.py --check <profile.yaml> [--allow-unguarded]
    launch_surface.py --report <profile.yaml>
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

#: Enforced eagerly, so a surface missing a key is reported as a shape error rather
#: than crashing on the first check that happens to dereference it.
REQUIRED_KEYS = ("name", "python_source", "cpp_mirror", "kmd_fields", "guard", "test")

#: The literal value meaning "nothing defends/compares this surface". A string, not
#: an absent key -- see the module docstring for why the distinction matters.
NONE_SENTINEL = "none"


class LaunchSurfaceError(RuntimeError):
    """The profile could not be read, or its launch_surface block is malformed."""


def _load_profile(path: str) -> dict:
    """Parse a profile as JSON, falling back to YAML. The mapping check covers
    both paths: a file that parses as valid JSON but is not an object would
    otherwise crash later naming neither the file nor the problem."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        import json

        loaded = json.loads(text)
    except ValueError:
        try:
            import yaml
        except ImportError:  # pragma: no cover - environment-dependent
            raise LaunchSurfaceError(f"{path} is not JSON and PyYAML is not installed.")
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise LaunchSurfaceError(
            f"profile {path} must be a mapping; got {type(loaded).__name__}."
        )
    return loaded


def load_surfaces(profile: dict) -> list[dict]:
    """The declared ``launch_surface:`` list, or an error if the profile has none.

    An absent block means the op has not been audited yet, not that there is nothing
    to audit, so it is a failure rather than a pass.
    """
    surfaces = profile.get("launch_surface")
    if not surfaces:
        raise LaunchSurfaceError(
            "profile declares no 'launch_surface:' block. Every op this engine ships "
            "restates a launch contract from Python; an absent block means it has "
            "not been audited, not that there is nothing to audit."
        )
    if not isinstance(surfaces, list):
        raise LaunchSurfaceError("'launch_surface:' must be a list of surface entries.")
    return surfaces


def _path_part(value: str) -> str:
    """The filesystem path prefix of a ``cpp_mirror``/``test`` value.

    Both fields carry a path plus a free-text locator (``Foo.cpp:functionName``,
    a pytest node id), and no path in this repo contains a ``:``.
    """
    return value.split(":", 1)[0].strip()


def find_repo_root(start: Path) -> Path:
    """The nearest ``.git`` ancestor of ``start``, or the resolved ``start``.
    Callers anchor on this script's location rather than ``Path.cwd()``, since
    every path a surface names is repo-relative. Always absolute."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


#: Metadata accessors that THROW when the named field is absent, so a field reached
#: only through one of these is one the engine dereferences unconditionally.
#: ``tryGetMetadata`` is excluded: it is how this codebase spells "this field may
#: legitimately be absent" (see Gfx950AttentionDenseNative.cpp on ``featureIsSet``).
_REQUIRED_ACCESSORS = ("getIntMetadata", "getStringMetadata")

_CONST_DEF_RE = re.compile(r'constexpr\s+std::string_view\s+(\w+)\s*=\s*"([^"]*)"\s*;')
_WRAPPER_LAMBDA_RE = re.compile(
    r"(\w+)\s*=\s*\[[^\]]*\]\s*\(\s*std::string_view\s+(\w+)\s*\)\s*\{([^}]*)\}"
)


def _strip_cpp_comments(text: str) -> str:
    """``text`` with ``//``, ``/* */`` and ``#`` comments removed, quoted
    content intact.

    Hand-rolled rather than a regex, since ``//.*$`` also eats a ``//`` inside
    a string literal. Triple quotes count as one delimiter and ``#`` as a line
    comment, so a non-C++ file named by mistake cannot open an unterminated
    string span.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if c == "#":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if c in ('"', "'") and text[i : i + 3] == c * 3:
            j = text.find(c * 3, i + 3)
            end = n if j == -1 else j + 3
            out.append(text[i:end])
            i = end
            continue
        if c in ('"', "'"):
            quote = c
            out.append(c)
            i += 1
            while i < n:
                out.append(text[i])
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _blank_cpp_string_contents(text: str) -> str:
    """Comment-stripped C++ with every string/char literal's interior blanked,
    so a call site spelled out inside an error message is not mistaken for
    code. Quote delimiters are kept so string boundaries still parse."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in ('"', "'") and text[i : i + 3] == c * 3:
            j = text.find(c * 3, i + 3)
            end = n if j == -1 else j + 3
            inner_end = n if j == -1 else j
            out.append(c * 3)
            out.append(" " * (inner_end - (i + 3)))
            out.append(text[inner_end:end])
            i = end
            continue
        if c in ('"', "'"):
            quote = c
            out.append(c)
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    out.append("  ")
                    i += 2
                    continue
                if text[i] == quote:
                    out.append(c)
                    i += 1
                    break
                out.append(" ")
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def extract_required_metadata_fields(cpp_text: str) -> set[str]:
    """KMD field names ``cpp_text`` reads through a required accessor (see
    ``_REQUIRED_ACCESSORS``), resolving a direct call and the one-line
    forwarding lambda this pack's files use (``_WRAPPER_LAMBDA_RE``).

    A regex scan, not a parse: the C++ here is one idiom, a
    ``constexpr std::string_view ..._FIELD`` per name. Comments and string
    literals are excluded first.
    """
    code = _blank_cpp_string_contents(_strip_cpp_comments(cpp_text))
    consts = dict(_CONST_DEF_RE.findall(_strip_cpp_comments(cpp_text)))

    accessor_pattern = "|".join(_REQUIRED_ACCESSORS)
    wrappers = set()
    for name, param, body in _WRAPPER_LAMBDA_RE.findall(code):
        if re.search(
            rf"\b(?:{accessor_pattern})\(std::string\({re.escape(param)}\)\)", body
        ):
            wrappers.add(name)

    fields: set[str] = set()
    direct_re = re.compile(
        rf"\.(?:{accessor_pattern})\s*\(\s*std::string\(\s*(\w+)\s*\)\s*\)"
    )
    for ident in direct_re.findall(code):
        if ident in consts:
            fields.add(consts[ident])

    for wrapper_name in wrappers:
        for ident in re.findall(
            rf"\b{re.escape(wrapper_name)}\s*\(\s*(\w+)\s*\)", code
        ):
            if ident in consts:
                fields.add(consts[ident])

    return fields


_LEADING_SYMBOL_RE = re.compile(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*")


def _leading_symbol(locator: str) -> str | None:
    """The symbol a ``cpp_mirror``/``python_source`` locator names, when its
    first whitespace-delimited token is a bare identifier or ``Class::method``;
    ``None`` when the locator is prose from the first token on
    (``"prepare() trusts persistent..."`` -> None; ``"prepare -- ..."`` ->
    ``prepare``)."""
    rest = locator.split(":", 1)[1].strip() if ":" in locator else locator.strip()
    token = rest.split(None, 1)[0] if rest.split() else ""
    return token if _LEADING_SYMBOL_RE.fullmatch(token) else None


def cpp_symbol_exists(cpp_text: str, symbol: str) -> bool:
    """Whether ``symbol`` (its ``Class::method`` qualifier stripped, since a method
    is declared under bare name inside its class body) appears followed by ``(`` --
    a declaration or a call, either way evidence the name is real -- outside comments
    and string literals.
    """
    bare = symbol.rsplit("::", 1)[-1]
    code = _blank_cpp_string_contents(_strip_cpp_comments(cpp_text))
    return re.search(rf"\b{re.escape(bare)}\s*\(", code) is not None


def python_symbol_exists(py_text: str, symbol: str) -> bool:
    """Whether ``symbol`` is defined as a function anywhere in ``py_text`` --
    module level, nested or a method -- via ``ast.parse``, so a mention in a
    comment cannot pass for a definition. A file that fails to parse reports
    the symbol as absent and the caller surfaces the reason."""
    try:
        tree = ast.parse(py_text)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
        for node in ast.walk(tree)
    )


def validate_shape(surfaces: list[dict]) -> list[str]:
    """Structural errors that make a surface unreadable, before anything else runs."""
    errors = []
    for index, surface in enumerate(surfaces):
        if not isinstance(surface, dict):
            errors.append(f"surface #{index} is not a mapping: {surface!r}")
            continue
        missing = [k for k in REQUIRED_KEYS if k not in surface]
        if missing:
            name = surface.get("name", f"#{index}")
            errors.append(f"surface '{name}' is missing required key(s): {missing}")
            continue
        if not isinstance(surface["kmd_fields"], list):
            errors.append(
                f"surface '{surface['name']}': kmd_fields must be a list of field "
                f"names, got {type(surface['kmd_fields']).__name__}"
            )
    return errors


def check(profile: dict, root: Path) -> tuple[list[str], list[str]]:
    """(failures, unguarded_or_untested) for every declared surface.

    A failure is structural: the profile's claim about itself is false. An
    honestly-declared ``guard: none`` is a separate list the caller decides on
    (``--allow-unguarded``).
    """
    surfaces = load_surfaces(profile)
    shape_errors = validate_shape(surfaces)
    if shape_errors:
        return shape_errors, []

    kmd_names = {f["name"] for f in (profile.get("kmd_fields") or [])}
    provider_root = profile.get("provider_root")
    python_root = (
        (root / provider_root / "rocke" / "library") if provider_root else root
    )

    failures: list[str] = []
    unguarded: list[str] = []

    # Per-file caches: a cpp_mirror/python_source cited by several surfaces is
    # read and scanned once, and an unreadable file is reported once.
    cpp_text_cache: dict[str, str | None] = {}
    py_text_cache: dict[str, str | None] = {}

    def read_cached(cache: dict[str, str | None], resolved: Path) -> str | None:
        # UTF-8 explicitly: read_text() would otherwise decode with the
        # platform's locale encoding, so one non-ASCII character in a comment
        # would abort the audit on a cp1252 shell and pass everywhere else.
        key = str(resolved)
        if key not in cache:
            try:
                cache[key] = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                cache[key] = None
        return cache[key]

    # (1b) prep: each cpp_mirror file's required-accessor reads, gathered before the
    # per-surface loop so a field is checked against the UNION of every surface
    # sharing that file -- only their combined kmd_fields need to cover it.
    required_by_file: dict[str, set[str]] = {}
    declared_by_file: dict[str, set[str]] = {}
    for surface in surfaces:
        cpp_path = _path_part(surface["cpp_mirror"])
        if not cpp_path:
            continue
        declared_by_file.setdefault(cpp_path, set()).update(surface["kmd_fields"])
        if cpp_path not in required_by_file:
            text = read_cached(cpp_text_cache, root / cpp_path)
            required_by_file[cpp_path] = (
                extract_required_metadata_fields(text) if text is not None else set()
            )

    for cpp_path, required in required_by_file.items():
        missing = sorted(required - declared_by_file.get(cpp_path, set()))
        if missing:
            failures.append(
                f"cpp_mirror '{cpp_path}' reads metadata field(s) {missing} through "
                f"a required accessor (getIntMetadata/getStringMetadata), but no "
                f"surface whose cpp_mirror names this file declares them in "
                f"kmd_fields -- the engine dereferences a field no surface admits "
                f"to needing"
            )

    for surface in surfaces:
        name = surface["name"]

        # (1) The headline check: a restated helper cannot branch on a field the
        # descriptor does not carry.
        undeclared = [f for f in surface["kmd_fields"] if f not in kmd_names]
        if undeclared:
            failures.append(
                f"surface '{name}' names kmd_fields {undeclared}, which the "
                f"profile's own kmd_fields does not declare -- the C++ mirror would "
                f"dereference a field the descriptor cannot carry"
            )

        # (2) cpp_mirror must point at a real file.
        cpp_path = _path_part(surface["cpp_mirror"])
        cpp_text = read_cached(cpp_text_cache, root / cpp_path) if cpp_path else None
        if cpp_path and cpp_text is None:
            # read_cached answers None for an absent file and for an
            # undecodable one; the two want different repairs, so they are
            # reported differently.
            reason = (
                "path does not exist"
                if not (root / cpp_path).is_file()
                else "is not valid UTF-8"
            )
            failures.append(f"surface '{name}' cpp_mirror {reason}: {cpp_path}")

        # (1c) cpp_mirror's leading symbol, if it names one, must be real.
        if cpp_text is not None:
            cpp_symbol = _leading_symbol(surface["cpp_mirror"])
            if cpp_symbol and not cpp_symbol_exists(cpp_text, cpp_symbol):
                failures.append(
                    f"surface '{name}' cpp_mirror names '{cpp_symbol}', which does "
                    f"not appear in {cpp_path}"
                )

        # (1c) python_source's leading symbol, resolved against provider_root's
        # rocke/library, the same base dispatch_parity.py imports modules from.
        py_path = _path_part(surface["python_source"])
        py_symbol = _leading_symbol(surface["python_source"])
        if py_path and py_symbol:
            py_text = read_cached(py_text_cache, python_root / py_path)
            if py_text is not None and not python_symbol_exists(py_text, py_symbol):
                failures.append(
                    f"surface '{name}' python_source names '{py_symbol}', which is "
                    f"not defined in {py_path}"
                )

        # (3) test must point at a real file, unless explicitly none.
        test_value = surface["test"]
        if test_value != NONE_SENTINEL:
            test_path = _path_part(test_value)
            if test_path and not (root / test_path).exists():
                failures.append(
                    f"surface '{name}' test path does not exist: {test_path}"
                )

        # (4) An unguarded or untested surface is reported by name, never silently.
        if surface["guard"] == NONE_SENTINEL or test_value == NONE_SENTINEL:
            unguarded.append(name)

    return failures, unguarded


def render_report(profile: dict) -> str:
    """The markdown table an integration owes, ready to paste into a PR."""
    surfaces = load_surfaces(profile)
    errors = validate_shape(surfaces)
    if errors:
        raise LaunchSurfaceError("; ".join(errors))

    lines = [
        "| surface | python source | C++ mirror | KMD fields | guard | test |",
        "|---|---|---|---|---|---|",
    ]
    for surface in surfaces:
        fields = ", ".join(surface["kmd_fields"]) or "(none)"
        lines.append(
            f"| {surface['name']} | {surface['python_source']} | "
            f"{surface['cpp_mirror']} | {fields} | {surface['guard']} | "
            f"{surface['test']} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the Python/C++ launch-contract restatement, per surface.",
    )
    parser.add_argument("profile", help="Path to the kernel's tool profile YAML.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify kmd_fields/cpp_mirror/test mechanically; exit non-zero on a "
        "structural failure or an unguarded surface (see --allow-unguarded).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print the surface table in markdown.",
    )
    parser.add_argument(
        "--allow-unguarded",
        action="store_true",
        help="Do not fail --check solely because a surface declares guard: none or "
        "test: none. The surfaces are still printed, by name -- this flag only "
        "changes the exit code, never the report.",
    )
    args = parser.parse_args(argv)

    if not args.check and not args.report:
        parser.error("choose --check and/or --report")

    try:
        profile = _load_profile(args.profile)
    except LaunchSurfaceError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    if args.report:
        try:
            print(render_report(profile))
        except LaunchSurfaceError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2

    if args.check:
        try:
            surfaces = load_surfaces(profile)
        except LaunchSurfaceError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2

        failures, unguarded = check(profile, find_repo_root(Path(__file__).parent))

        print("launch-surface check")
        print(f"  surfaces declared  {len(surfaces)}")
        if failures:
            print(f"  structural failures  {len(failures)}")
            for f in failures:
                print(f"      ! {f}")
        else:
            print("  structural failures  0")

        if unguarded:
            verb = "REPORTED" if args.allow_unguarded else "FAILING"
            print(f"  unguarded/untested  {len(unguarded)} ({verb})")
            for name in unguarded:
                print(f"      ? {name}")
        else:
            print("  unguarded/untested  0")

        print()
        if failures or (unguarded and not args.allow_unguarded):
            reasons = []
            if failures:
                reasons.append(f"{len(failures)} structural failure(s)")
            if unguarded and not args.allow_unguarded:
                reasons.append(
                    f"{len(unguarded)} unguarded/untested surface(s); pass "
                    f"--allow-unguarded to accept them as a deliberate choice"
                )
            print(f"CHECK FAILED ({'; '.join(reasons)})")
            return 1
        print(
            "CHECK PASSED: every surface's kmd_fields, cpp_mirror, test, and "
            "referenced symbol are real, every metadata field a mirror reads "
            "unconditionally is declared, and every unguarded surface was a "
            "declared choice."
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
