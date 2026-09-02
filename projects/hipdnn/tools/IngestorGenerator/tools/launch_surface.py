"""Make the Python/C++ launch-contract restatement AUDITABLE, per surface, per op.

A rocKE kernel is launched from Python; the C++ ingestor engine RESTATES that launch
contract by hand -- grid shape, block size, kernarg order, baked constants (strides
compiled into the binary with no kernarg), the dispatcher's spec resolution, and
applicability. Nothing in the build, the packer, the validator or the test suite
compares the two halves. A mismatch does not fail: the kernel runs and computes
something else.

Two defects have already shipped through this restatement (the persistent grid
launched with the default grid size, and a windowed causal graph served as plain
causal), and this repository just found a THIRD, structurally distinct one while
building this tool: the gfx942 attention_dense profile's ``kmd_fields`` never
declared ``block_m``, even though ``Gfx942AttentionDenseNative.cpp`` reads it via
``kernel.getIntMetadata("block_m")`` on both the matcher path (``kernelMatches``,
the ``seqlen_q % block_m == 0`` tile check) and the prepare() path
(``attentionDenseGeometry``, which sizes the launch grid from it). Every descriptor
this profile generated was missing a field its own engine dereferences unconditionally
-- a restated helper cannot branch on a field the descriptor does not carry.

A FOURTH defect class, found reviewing THIS tool: the checks above only ever look
INSIDE the profile, so a surface's ``kmd_fields`` can under-declare relative to the
profile's own list (check 1) but the profile can also just not mention a field the
C++ reads at all -- delete the whole ``kernargs`` surface, or drop ``dtype`` from
``applicability``'s ``kmd_fields``, and every check below still passed, because
nothing compared the DECLARATION against the ENGINE. Check (1b) closes that: it scans
each declared ``cpp_mirror`` file for `kernel.getIntMetadata`/`getStringMetadata`
call sites (never `tryGetMetadata` -- see its own docstring) and asserts every field
read that way is declared by SOME surface whose ``cpp_mirror`` is that file. A field
the engine dereferences unconditionally but no surface declares is exactly the
undeclared-surface signal this tool could not previously see.

WHAT A ``launch_surface:`` BLOCK DECLARES, per surface (grid, block, kernargs,
baked_constants, spec_resolution, applicability, or any further split an op needs):

    launch_surface:
      - name: grid
        python_source: kernels/gfx942/attention_dense.py:attention_dense_grid (:1803-1819)
        cpp_mirror: dnn-providers/.../Gfx942AttentionDenseGeometry.hpp:attentionDenseGeometry
        kmd_fields: [block_m, seqlen_q, num_query_heads, batch, persistent, num_persistent]
        guard: attentionDenseGeometry throws on a non-positive dimension or an
          unusable persistent/num_persistent pair (lines 82-107)
        test: dnn-providers/.../TestGfx942AttentionDenseGeometry.cpp

``guard: none`` or ``test: none`` are not omissions -- they are the honest answer for
a surface nothing defends or nothing compares, and the point of writing "none" rather
than leaving the key out is that an unguarded surface must be a DELIBERATE, VISIBLE
choice, not a blank a reviewer has to notice is missing.

WHAT ``--check`` VERIFIES, mechanically (no device, no build):

  1. Every ``kmd_fields`` entry a surface names actually exists in the profile's own
     top-level ``kmd_fields`` list (the block_m defect's exact shape).
  1b. Every metadata field a declared ``cpp_mirror`` file reads through a REQUIRED
      accessor (`getIntMetadata`, `getStringMetadata` -- both throw on a missing
      field) is declared in SOME surface's ``kmd_fields`` whose ``cpp_mirror`` names
      that same file. This is a regex scan over the mirror's own source, comments and
      string literals excluded, not a parse -- see ``extract_required_metadata_fields``.
      Fields read only through `tryGetMetadata` (the four ABI-extending features here)
      are deliberately excluded: an absent field is a valid, checked answer on that
      path, not an oversight -- see the accessor's own docstring in the mirror file.
  1c. Every ``cpp_mirror``/``python_source`` value whose LEADING TOKEN (before the
      first space, after the path/first colon) looks like a bare identifier or a
      `Class::method` qualifier is checked against the named file: the bare method
      name must appear followed by `(` somewhere in the C++ file, or -- for a Python
      source -- a `def NAME(` must exist. A locator whose first token is prose (does
      not parse as an identifier) is left alone; this tool does not attempt to divine
      intent from free text, only to catch a symbol that was clearly meant as one.
  2. Every ``cpp_mirror`` path exists on disk (resolved against the REPO ROOT, found
     by walking up from this script's own location to the nearest ``.git`` -- not the
     working directory, so ``--check`` gives the same answer run from anywhere in the
     tree).
  3. Every ``test`` path exists on disk, when it is not the literal string ``none``.
  4. Every surface with ``guard: none`` or ``test: none`` is reported BY NAME as
     unguarded/untested, and the command exits non-zero unless ``--allow-unguarded``
     is passed. Nothing here says an unguarded surface is wrong -- some genuinely
     have no cheaper defense yet -- but silence about it is what let two defects
     ship, so the check refuses to be quiet about the surfaces still not restated
     under a comparison of any kind.

WHAT IT STILL DOES NOT DO. It cannot check that ``python_source`` and ``cpp_mirror``
compute the SAME THING, only that the names they cite exist -- that is what the
``test`` column is for, and a per-surface test (the geometry header's own test, the
matcher negatives, dispatch_parity's persistent-rule check) is how that comparison
actually happens. It cannot see a field the C++ reads through `tryGetMetadata` and
never validate that the caller's optional-vs-required split is the right one -- that
is a design review question, not a regex's to answer. And it cannot catch every
shape of "this surface should exist but nobody wrote it": check 1b only fires for
fields reached through a KNOWN accessor in an ALREADY-DECLARED cpp_mirror file: a
launch surface whose C++ counterpart is a file no surface names at all (rather than
one surface's fields being incomplete) is invisible to it, same as before this
change -- see ``verify_variant_sets.py``'s coverage check for anything closer to that
question. Concretely: deleting the ``kernargs`` surface from this arch's own profile
escapes it, because ``kernargs``'s cpp_mirror is
``Gfx950AttentionDenseDispatchHandler::launch``, which reads ZERO metadata fields
through ANY accessor -- it only forwards positional device-buffer pointers. The
ABI-guard defect that surface documents (a 5-slot launch against a conditionally
6-to-8-slot Python signature) is not expressed as a metadata read at all, so there
is nothing for a metadata-field scan to notice missing. Deleting a surface whose
cpp_mirror DOES read required fields the remaining surfaces do not cover (deleting
``applicability`` here, for instance) is caught, by the same mechanism as check 1b.
In short: an ENTIRE undeclared/deleted surface is caught only when it uniquely
covered a metadata field some still-declared mirror reads through a required
accessor; a surface whose mirror reads no metadata at all (``kernargs`` here) is
invisible to this check no matter how it is mutated -- do not assume ``--check``
now covers every undeclared-surface shape.

This tool only makes the CLAIM structurally honest: the fields a mirror
needs are declared, the symbols and files named are real, and an absent guard or
test is named rather than silently missing.

    launch_surface.py --check <profile.yaml>
    launch_surface.py --check <profile.yaml> --allow-unguarded
    launch_surface.py --report <profile.yaml>
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

#: Every key a launch_surface entry must declare. Enforced eagerly so a surface
#: missing a key is reported as a shape error rather than crashing on the first
#: check that happens to dereference it.
REQUIRED_KEYS = ("name", "python_source", "cpp_mirror", "kmd_fields", "guard", "test")

#: The literal value meaning "nothing defends/compares this surface". A string, not
#: an absent key -- see the module docstring for why the distinction matters.
NONE_SENTINEL = "none"


class LaunchSurfaceError(RuntimeError):
    """The profile could not be read, or its launch_surface block is malformed."""


def _load_profile(path: str) -> dict:
    """Parse a profile as JSON, falling back to YAML.

    The mapping check covers BOTH paths. It used to sit only on the YAML branch, so a
    file that parsed as valid JSON but was not an object -- a bare list, say -- sailed
    through and crashed later with an AttributeError naming neither the file nor the
    problem.
    """
    text = Path(path).read_text()
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

    An op with NO launch_surface block is not "nothing to check" -- it is a profile
    that has not been audited yet, and reporting that as a pass would be exactly the
    silent gap this tool exists to close.
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

    Both fields carry a path plus free-text locator (``Foo.cpp:functionName (~702)``,
    a pytest node id ``test_x.py::TestY::test_z``), so a plain existence check needs
    only what precedes the first ``:`` -- no path used anywhere in this repo contains
    one.
    """
    return value.split(":", 1)[0].strip()


def find_repo_root(start: Path) -> Path:
    """The nearest ``.git`` ancestor of ``start``, or the resolved ``start`` itself if
    none is found (a profile outside any git checkout -- ``cpp_mirror``/``test``
    existence then falls back to whatever the caller passed). Both branches return an
    absolute, resolved path, so a caller passing a relative path never gets one back.

    Anchoring on this script's own location rather than ``Path.cwd()`` is what makes
    ``--check`` give the same answer run from the repo root or from three directories
    down: every path a surface names is repo-relative, and only the repo root is a
    stable base for resolving one.
    """
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


#: Metadata accessors that THROW when the named field is absent from a descriptor.
#: A field reached only through one of these is one the engine dereferences
#: unconditionally -- exactly the shape of the block_m defect. ``tryGetMetadata`` is
#: deliberately excluded: it is how this codebase spells "this field may legitimately
#: be absent", used for the four ABI-extending features a pre-fields descriptor never
#: carried (see Gfx950AttentionDenseNative.cpp's own comment on ``featureIsSet``).
#: Treating it as required would flag every one of those as a false positive.
_REQUIRED_ACCESSORS = ("getIntMetadata", "getStringMetadata")

_CONST_DEF_RE = re.compile(r'constexpr\s+std::string_view\s+(\w+)\s*=\s*"([^"]*)"\s*;')
_WRAPPER_LAMBDA_RE = re.compile(
    r"(\w+)\s*=\s*\[[^\]]*\]\s*\(\s*std::string_view\s+(\w+)\s*\)\s*\{([^}]*)\}"
)


def _strip_cpp_comments(text: str) -> str:
    """``text`` with ``//``, ``/* */``, and ``#`` comments removed, quoted content
    intact.

    A hand-rolled scanner rather than a comment-matching regex: a naive
    ``//.*$``/``/\\*.*?\\*/`` pattern also eats a ``//`` or ``/*`` that appears inside
    a string literal (a URL in a log message, an escaped ``"/* not a comment */"``),
    which would silently swallow real code after it. Walking the text character by
    character and treating a quote as an opaque span sidesteps that.

    A triple quote (``\"\"\"``/``'''``) is matched as ONE delimiter, not three single
    ones, so scanning a file that is not actually C++ (a test fixture standing in a
    real path, a stray Python source named as a cpp_mirror by mistake) cannot lose
    track of whether it is inside a string and silently drop the remainder of the
    file -- the failure mode a plain single-quote scanner has on a Python docstring.

    ``#`` is ALSO treated as a line comment, even though it starts a preprocessor
    directive rather than a comment in real C++: this scanner only cares about
    ``kernel.get*Metadata`` call sites, never about macros, so a ``#include``/``#if``
    line contributes nothing either way -- but a bare apostrophe after a ``#`` in a
    NON-C++ file scanned by mistake (a ``#:`` Sphinx-style comment in a Python
    fixture, docstring apostrophes like "file's") would otherwise open an
    unterminated single-quote span and silently swallow the rest of the file, which
    is precisely the class of false-clean scan this tool exists to prevent.
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
    """Comment-stripped C++ with every string/char literal's INTERIOR blanked.

    Applied on top of ``_strip_cpp_comments`` before the accessor-call regexes run,
    so a call site spelled out inside an error message or a doc comment's quoted
    example cannot be mistaken for real code -- only text that is actually a call
    expression matches. Quote delimiters are kept so string boundaries still parse.
    Triple quotes are treated as one delimiter, for the same reason
    ``_strip_cpp_comments`` does.
    """
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
    """KMD field names ``cpp_text`` reads through a REQUIRED accessor (see
    ``_REQUIRED_ACCESSORS``), resolving both a direct call
    (``kernel.getIntMetadata(std::string(FIELD_CONST))``) and the one-line forwarding
    lambda this pack's files use to fetch several fields through one accessor
    (``auto intField = [&kernel](std::string_view field) { return
    kernel.getIntMetadata(std::string(field)); };`` then ``intField(SOME_FIELD)``).

    A regex scan, not a parse -- deliberately: this file's C++ is a small, consistent
    idiom (a `constexpr std::string_view ..._FIELD` per metadata name, read directly
    or through one forwarding lambda), and a full parser is not warranted to recognize
    it. Comments and string literals are excluded first so neither can forge a call
    site the compiler would never see.
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
    """The symbol a ``cpp_mirror``/``python_source`` value names, if its first
    whitespace-delimited token parses as a bare identifier or a ``Class::method``
    qualifier -- ``None`` for a value whose locator is prose from the first token on
    (``"prepare() trusts persistent..."`` has no leading symbol; ``"prepare -- ..."``
    names ``prepare``).
    """
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
    """Whether ``symbol`` is defined as a function anywhere in ``py_text`` -- module
    level, nested, or a method body -- via ``ast.parse`` rather than a text scan: a
    real parser cannot be fooled by a comment or a docstring that merely mentions the
    name, and Python's ``#`` / triple-quoted comment and string forms are exactly
    what a C++-shaped regex scanner (see ``cpp_symbol_exists``) would get wrong.
    A file that fails to parse (rare -- a real syntax error, or a non-Python source
    accidentally named as one) is reported as the symbol NOT existing, and the
    reason is surfaced separately by the caller rather than raising here.
    """
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

    ``failures`` are structural: a named kmd_fields entry the KMD does not carry, a
    cpp_mirror/test path absent from disk, or a malformed entry -- each one means the
    profile's own claim about itself is false. ``unguarded_or_untested`` is a
    separate list on purpose: an honestly-declared ``guard: none`` is not a
    structural defect, it is a fact the caller must decide whether to accept
    (``--allow-unguarded``), and conflating the two would make an honest admission
    look the same as a broken path.
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

    # Per-file caches: each cpp_mirror/python_source file is read and scanned once
    # no matter how many surfaces cite it (grid and block above both cite the same
    # geometry header), and a file that fails to read is reported once per file
    # rather than once per surface referencing it.
    cpp_text_cache: dict[str, str | None] = {}
    py_text_cache: dict[str, str | None] = {}

    def read_cached(cache: dict[str, str | None], resolved: Path) -> str | None:
        key = str(resolved)
        if key not in cache:
            try:
                cache[key] = resolved.read_text()
            except OSError:
                cache[key] = None
        return cache[key]

    # (1b) prep: every cpp_mirror file's required-accessor reads, gathered before
    # the per-surface loop so a field can be checked against the UNION of every
    # surface sharing that file (grid and block both restate the same geometry
    # header; only their combined kmd_fields need to cover what the header reads).
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
        # descriptor does not carry. This is exactly how block_m shipped absent from
        # this profile's own kmd_fields while Gfx942AttentionDenseNative.cpp read it
        # unconditionally on two live paths.
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
            failures.append(
                f"surface '{name}' cpp_mirror path does not exist: {cpp_path}"
            )

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
