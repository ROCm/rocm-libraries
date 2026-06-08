#!/usr/bin/env python3
"""MicroPython-compatibility lint for the ck_dsl code shipped into the embedded
interpreter (frozen or on-disk .py/.mpy).

The provider runs ck_dsl under MicroPython, not CPython. A handful of Python
constructs compile + run fine under CPython but break under MicroPython -- they
either fail to compile in the embed's runtime compiler (.py mode) or behave
wrong at runtime (the dataclasses shim, the os shim). The build only exercises
the current frozen closure for the current test shapes, so such constructs can
creep into other modules unnoticed. This lint walks the codegen sources and
fails (exit 1) if any of the known-incompatible constructs appear:

  1. @dataclass fields without `= field(...)`. MicroPython erases bare class
     annotations and its dict iteration order is not insertion order, so the
     dataclasses shim discovers fields + their order from field()'s creation
     counter. Every field (including ones with defaults) must be `= field(...)`.
  2. PEP-448 star-unpacking in list/tuple/set displays ([a, *b]) -- unsupported
     by the embed compiler; write [a] + list(b) instead. (Starred ASSIGNMENT
     targets, a, *b = x, are fine and not flagged.)
  3. os.environ access -- MicroPython's os has no environ; use os.getenv.
  4. match statements / async def / await -- not enabled in the embed build.

The AST rules above only catch the KNOWN incompatibilities. When an mpy-cross
binary is available (--mpy-cross, or the MPY_CROSS env var), each in-scope module
is also compiled with it -- the actual MicroPython compiler -- so arbitrary
unsupported syntax is caught too, across all modules (not just the closure the
build already freezes). This is a superset-feature check: it validates the
frozen / mpy modes; the AST PEP-448 rule covers the one display construct the
reduced .py-mode runtime compiler additionally rejects.

Usage:
    check_compat.py [--mpy-cross PATH] <dir> [<dir> ...]
Each dir is walked recursively; examples/, dsl_docs/ and __pycache__ are skipped.
"""
import ast
import os
import subprocess
import sys
import tempfile

# Host-only modules: benchmark / sweep drivers that run on CPython and are never
# frozen or loaded under MicroPython, so host features (os.environ, subprocess)
# are legitimate there. Keyed by basename; keep this list short + justified.
ALLOWLIST_BASENAMES = {
    "sweep_bench.py",  # standalone benchmark driver (hipcc subprocess + os.environ)
}

# Modules that pass the AST rules but are not yet mpy-cross-compatible for a
# reason the AST rules don't model (e.g. a codegen function exceeding
# MicroPython's 255-locals-per-function limit). They are NOT in any provider
# closure today; listed here so the mpy-cross ratchet still guards every other
# module while these stay visible as "to port for MicroPython". AST rules still
# apply to them.
KNOWN_MPY_CROSS_INCOMPATIBLE = {
    "attention_tiled_2d.py",  # >255 locals in a codegen function (FMHA gfx950 instance)
}

# remote_test is a host-only remote-execution harness (SSH / SLURM / docker
# drivers); it is never frozen or loaded under MicroPython, so its host features
# (os.environ, PEP-448 star-args) are legitimate -- same rationale as the
# sweep_bench.py allowlist entry above.
SKIP_DIRS = {"__pycache__", "examples", "dsl_docs", "remote_test"}


def _is_dataclass(node):
    for dec in node.decorator_list:
        n = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(n, ast.Name) and n.id == "dataclass":
            return True
        if isinstance(n, ast.Attribute) and n.attr == "dataclass":
            return True
    return False


def _is_classvar(ann):
    t = ann.value if isinstance(ann, ast.Subscript) else ann
    if isinstance(t, ast.Name):
        return t.id == "ClassVar"
    if isinstance(t, ast.Attribute):
        return t.attr == "ClassVar"
    return False


def _is_field_call(value):
    if not isinstance(value, ast.Call):
        return False
    f = value.func
    return (isinstance(f, ast.Name) and f.id == "field") or (
        isinstance(f, ast.Attribute) and f.attr == "field"
    )


def _is_load_display_with_star(node):
    # A display (not an assignment target) that uses [*x] / (*x,) / {*x}.
    if isinstance(node, ast.Set):
        elts = node.elts
    elif isinstance(node, (ast.List, ast.Tuple)):
        if not isinstance(getattr(node, "ctx", None), ast.Load):
            return False
        elts = node.elts
    else:
        return False
    return any(isinstance(e, ast.Starred) for e in elts)


def check_source(path, src):
    """Return a list of (lineno, message) violations for one source string."""
    violations = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [(e.lineno or 0, f"SyntaxError: {e.msg}")]

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_dataclass(node):
            for stmt in node.body:
                if not isinstance(stmt, ast.AnnAssign) or not isinstance(
                    stmt.target, ast.Name
                ):
                    continue
                if _is_classvar(stmt.annotation):
                    continue
                if stmt.value is None or not _is_field_call(stmt.value):
                    violations.append(
                        (
                            stmt.lineno,
                            f"dataclass field '{stmt.target.id}' must be '= field(...)' "
                            "(MicroPython erases bare annotations; the shim needs field())",
                        )
                    )
        if _is_load_display_with_star(node):
            violations.append(
                (
                    node.lineno,
                    "PEP-448 star-unpacking in a display; use concatenation instead",
                )
            )
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "environ"
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        ):
            violations.append(
                (node.lineno, "os.environ is unavailable in MicroPython; use os.getenv")
            )
        if isinstance(node, ast.Match):
            violations.append(
                (node.lineno, "match statement not supported by the embed build")
            )
        if isinstance(
            node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)
        ):
            violations.append(
                (node.lineno, "async/await not supported by the embed build")
            )

    return violations


def mpy_cross_error(mpy_cross, path, out_dir):
    """Compile one module with mpy-cross (the real MicroPython compiler). Returns
    an error string, or None on success. Only the compile result matters (the
    .mpy output is discarded), so no special flags are needed."""
    out = os.path.join(out_dir, "out.mpy")
    proc = subprocess.run(
        [mpy_cross, "-o", out, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode == 0:
        return None
    lines = proc.stdout.strip().splitlines()
    return lines[-1] if lines else f"mpy-cross exited {proc.returncode}"


def main(argv):
    mpy_cross = os.environ.get("MPY_CROSS") or None
    dirs = []
    it = iter(argv[1:])
    for arg in it:
        if arg == "--mpy-cross":
            mpy_cross = next(it, None)
        elif arg.startswith("--mpy-cross="):
            mpy_cross = arg.split("=", 1)[1]
        else:
            dirs.append(arg)
    if not dirs:
        print(
            "usage: check_compat.py [--mpy-cross PATH] <dir> [<dir> ...]",
            file=sys.stderr,
        )
        return 2

    use_mpy_cross = bool(mpy_cross) and os.path.exists(mpy_cross)
    if mpy_cross and not use_mpy_cross:
        print(f"note: mpy-cross not found at {mpy_cross}; running AST checks only.")

    findings = []  # (path, lineno, message)
    scanned = 0
    with tempfile.TemporaryDirectory() as out_dir:
        for root in dirs:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for fn in filenames:
                    if not fn.endswith(".py") or fn in ALLOWLIST_BASENAMES:
                        continue
                    path = os.path.join(dirpath, fn)
                    with open(path, encoding="utf-8") as fh:
                        src = fh.read()
                    scanned += 1
                    for lineno, msg in check_source(path, src):
                        findings.append((path, lineno, msg))
                    if use_mpy_cross and fn not in KNOWN_MPY_CROSS_INCOMPATIBLE:
                        err = mpy_cross_error(mpy_cross, path, out_dir)
                        if err is not None:
                            findings.append(
                                (path, 0, f"mpy-cross rejects this module: {err}")
                            )

    how = "AST + mpy-cross" if use_mpy_cross else "AST"
    if findings:
        findings.sort()
        print("MicroPython-compatibility lint FAILED:\n")
        for path, lineno, msg in findings:
            print(f"  {path}:{lineno}: {msg}")
        print(
            f"\n{len(findings)} violation(s) across {scanned} files. See {__file__} for the rules."
        )
        return 1

    print(
        f"MicroPython-compatibility lint OK ({how}): {scanned} files, no incompatible constructs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
