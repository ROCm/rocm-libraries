"""Mutation check of the R1 pilot gates: each mutant reverts one piece of the
change; the named gate must FAIL on it (and pass on the restored file).

Usage: python mutants.py <rust-tree> <cargo-target-dir> [--only NAME]
Runs locally with -j 2 / 2 test threads; restores every file in a finally.
"""

from __future__ import annotations

import os
import subprocess
import sys

TREE = sys.argv[1]
TARGET = sys.argv[2]
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
CRATE = os.path.join(TREE, "platform/rust/rocke-codeobj")
PY = sys.executable
CARGO = os.path.expandvars(r"%USERPROFILE%\.cargo\bin\cargo.exe") if os.name == "nt" else "cargo"
ENV = dict(
    os.environ,
    CARGO_TARGET_DIR=TARGET,
    RUST_TEST_THREADS="2",
    PYTHONDONTWRITEBYTECODE="1",
    ROCKE_BACKEND="python",
    PYTHONPATH=os.pathsep.join([os.path.join(TREE, "platform/python"), os.path.join(TREE, "library")]),
    ROCKE_CODEOBJ_DIRECT_LIB=os.path.join(TARGET, "release", "rocke_codeobj_ffi.dll" if os.name == "nt" else "librocke_codeobj_ffi.so"),
)
PYTEST = [PY, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-x", "tests/test_qwen38_direct_pilot.py"]
RUST_TEST = [CARGO, "test", "--release", "--locked", "--offline", "--lib", "-j", "2", "-q", "rocke_ir_the_direct_backend"]
BUILD = [CARGO, "build", "--release", "--locked", "--offline", "--lib", "-j", "2"]

RS = "platform/rust/rocke-codeobj/src/rir/"
# (name, file, old, new, gates): gates run in order, each must fail.
MUTANTS = [
    ("op_names_parse", RS + "parse.rs",
     "        unsupported(format!(\"op {opname:?} is not lowered by the direct backend\"))",
     "        err(format!(\"op {opname:?} is not lowered by the direct backend\"))", ["rust", "pytest"]),
    ("asm_guard_off", RS + "validate.rs",
     "            InlineAsmIdentity if op.str_attr(\"template\") != Some(\"\") || op.str_attr(\"constraints\") != Some(\"=v,0\") => {",
     "            InlineAsmIdentity if false => {", ["rust", "pytest"]),
    ("inreg_ignored", RS + "parse.rs", "        if inreg {\n", "        if inreg && false {\n", ["rust", "pytest"]),
    ("symbol_late", RS + "parse.rs",
     "    if name.len() > 1024 || !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'$')) {",
     "    if name.len() > 1024 || !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'$' | b'-')) {",
     ["rust", "pytest"]),
    ("ptr_vec_parse", RS + "parse.rs",
     "            return Err(unsupported(format!(\"a pointer to {pointee} ({text:?})\")));",
     "            return Err(err(format!(\"a pointer to {pointee} ({text:?})\")));", ["rust", "pytest"]),
    ("mark_ignored", "platform/python/rocke/helpers/compile.py",
     "    if code_object_backend is None:\n        code_object_backend = requested_code_object_backend(kernel)\n",
     "", ["pytest"]),
    ("framework_no_mark", "library/kernels/gfx950/qwen38_mono/framework.py",
     "    if knobs.code_object_backend:\r\n        request_code_object_backend(b.kernel, knobs.code_object_backend)\r\n",
     "", ["pytest"]),
    ("knob_in_name", "library/kernels/gfx950/qwen38_mono/framework.py",
     "        return \"_\".join(parts)\r\n",
     "        if self.code_object_backend:\r\n            parts.append(\"cob\")\r\n        return \"_\".join(parts)\r\n",
     ["pytest"]),
]


def run(cmd, cwd) -> int:
    return subprocess.run(cmd, cwd=cwd, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def gate(name: str) -> int:
    if name == "rust":
        return run(RUST_TEST, CRATE)
    return run(PYTEST, os.path.join(TREE, "library"))


def main() -> int:
    bad = 0
    for name, rel, old, new, gates in MUTANTS:
        if ONLY and name != ONLY:
            continue
        path = os.path.join(TREE, rel)
        orig = open(path, "rb").read()
        text = orig.decode("utf-8")
        if text.count(old) != 1:
            print(f"{name}: anchor not found once ({text.count(old)})")
            bad += 1
            continue
        try:
            open(path, "wb").write(text.replace(old, new).encode("utf-8"))
            if rel.endswith(".rs") and run(BUILD, CRATE) != 0:
                print(f"{name}: mutant does not build")
                bad += 1
                continue
            res = {g: gate(g) for g in gates}
        finally:
            open(path, "wb").write(orig)
        killed = all(rc != 0 for rc in res.values())
        print(f"{name}: {'KILLED' if killed else 'SURVIVED'} {res}", flush=True)
        bad += not killed
    if any(rel.endswith(".rs") for _, rel, *_ in MUTANTS):
        run(BUILD, CRATE)  # restore the library of the unmutated tree
    res = {g: gate(g) for g in ("rust", "pytest")}
    print(f"restored tree: {res}")
    return 1 if bad or any(res.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
