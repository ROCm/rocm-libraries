"""Apply the minimal rust-direct overlay (research/rust-direct.md section 4.2) to the
isolated copy C:/Users/vanantha/Documents/dev/copy-rocm-lib-rust (a plain copy of the mono
tree's dnn-providers/hip-kernel-provider/rocke). Read-only on git: blobs come from
`git cat-file blob` (LF, repo-canonical bytes), merges from `git merge-file -p` on temp
files. Writes every overlaid file with LF endings and a manifest (path, class, source).

Usage: python overlay.py [--dry-run]
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

REPO = "C:/Users/vanantha/Documents/dev/copy-rocm-lib"
DST = "C:/Users/vanantha/Documents/dev/copy-rocm-lib-rust"
R = "dnn-providers/hip-kernel-provider/rocke/"
BASE = "e7d4664bda9"
TIP = "23c165f8b3f"
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlay.manifest")

P = "platform/python/rocke/"
COPY_NEW_EXTRA = [
    P + "runtime/code_object.py",
    P + "runtime/code_object_elf.py",
    P + "runtime/direct_session.py",
    P + "helpers/direct_gemm.py",
    "platform/tools/isa_tables.py",
    "platform/tools/isa_xcheck.py",
    "platform/tools/isa_burs.py",
    "platform/tests/runtime/test_code_object.py",
    "platform/tests/runtime/test_code_object_elf.py",
    "platform/tests/runtime/test_direct_gemm.py",
    "platform/tests/runtime/test_direct_session.py",
    "platform/tests/runtime/test_direct_targets.py",
    "platform/tests/tools/test_isa_burs.py",
    "platform/tests/tools/test_isa_tables.py",
    "platform/tests/tools/test_isa_xcheck.py",
    "platform/dsl_docs/runtime/direct_compiler.md",
    "platform/dsl_docs/runtime/code_object_compiler.md",
]
# Files the mono tree has unchanged from the rust branch's base: take theirs.
COPY_THEIRS = [
    P + "__init__.py",
    P + "runtime/__init__.py",
    P + "runtime/comgr.py",
    P + "helpers/compile.py",
    P + "core/ir_serialize.py",
    P + "examples/common/ck_tile_parity.py",
    P + "examples/common/distribution_2d_add_demo.py",
    P + "examples/common/distribution_reduce_demo.py",
    P + "examples/gfx1151/gemm/scripts/01_f16_verify.py",
    P + "examples/gfx942/tf32_numerics.py",
    P + "examples/gfx950/moe/test_active_tile_skip.py",
    P + "examples/gfx950/moe/test_preshuffle_b.py",
    P + "examples/gfx950/xf32/bf16x3_parity.py",
    P + "portable_ir/drivers/bench_jit_validation.py",
    P + "portable_ir/drivers/gpu_replay.py",
    P + "portable_ir/drivers/launch_from_bundle.py",
    P + "portable_ir/drivers/parity_matrix.py",
    P + "portable_ir/drivers/roll_hsaco_parity.py",
    P + "portable_ir/drivers/roll_kernel.py",
    "platform/tests/core/test_compile_target_selection.py",
    "platform/pyproject.toml",
]
# Rust-branch changes deliberately NOT overlaid (research rust-direct.md 4.2 "Do not overlay"), and
# the tests that exercise them. The two test files were first copied (mono == base) and then restored
# to the mono tree's version: their new cases test lower_llvm.py's round-trip fixes (a parsed
# ``waves_per_eu`` pair, a parsed ``tf32`` type), which stay out with lower_llvm.py.
EXCLUDED = {
    "platform/python/rocke/core/lower_llvm.py": "COMGR-path address split; the direct path never lowers to LLVM",
    "platform/tests/core/test_ir_serialize.py": "new case tests lower_llvm.py (waves_per_eu pair after parse)",
    "platform/tests/core/test_xf32_bf16x3_lowering.py": "new case tests lower_llvm.py (parsed tf32 type)",
    "platform/tests/core/test_global_address_split.py": "tests lower_llvm.py's address split",
    "C++ / CMake / CTest (cpp/**, CMakeLists.txt, tests/CMakeLists.txt, *.cpp tests, code_object_ffi.h)": "user rule: no C++ work",
    "attention_bwd*, conv_direct_grouped*, kernel_cache, reproduce_kernel, golden JSON, portable-IR gates": "not needed by the direct path",
    "helpers/fuse.py, fusion_lowering.py, sweep.py, heuristics/*": "import renames / cache-key changes not needed (comgr.py keeps build_hsaco_from_llvm_ir)",
}
MERGE = [
    P + "helpers/__init__.py",
    P + "core/backend.py",
    P + "core/isa/backend.py",
    "platform/dsl_docs/reference/env_flags.md",
]


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", REPO, *args], check=True, capture_output=True
    ).stdout


def blob(rev: str, rel: str) -> bytes:
    return git("cat-file", "blob", f"{rev}:{R}{rel}")


def blob_id(rev: str, rel: str) -> str:
    return git("rev-parse", f"{rev}:{R}{rel}").decode().strip()


def lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def main() -> int:
    dry = "--dry-run" in sys.argv
    crate = [
        ln[len(R):]
        for ln in git("ls-tree", "-r", "--name-only", TIP, R + "platform/rust/rocke-codeobj")
        .decode()
        .splitlines()
    ]
    # Sanity: every COPY_THEIRS file is unchanged mono vs base (HEAD == base blob,
    # and the working tree equals HEAD modulo CRLF).
    for rel in COPY_THEIRS + MERGE:
        same = blob_id("HEAD", rel) == blob_id(BASE, rel)
        if rel in COPY_THEIRS and not same:
            print(f"ERROR: {rel} differs mono vs base; must be merged", file=sys.stderr)
            return 2
        if rel in MERGE and same:
            print(f"note: {rel} equals base; merge degenerates to theirs")
        wt = lf(open(os.path.join(DST, rel), "rb").read())
        if wt != lf(blob("HEAD", rel)):
            print(f"ERROR: copy's {rel} differs from mono HEAD", file=sys.stderr)
            return 2
    rows = []
    out: dict[str, bytes] = {}
    for rel in crate + COPY_NEW_EXTRA:
        if os.path.exists(os.path.join(DST, rel)):
            print(f"ERROR: new file already exists: {rel}", file=sys.stderr)
            return 2
        out[rel] = blob(TIP, rel)
        rows.append((rel, "new", blob_id(TIP, rel)))
    for rel in COPY_THEIRS:
        out[rel] = blob(TIP, rel)
        rows.append((rel, "theirs(mono==base)", blob_id(TIP, rel)))
    with tempfile.TemporaryDirectory() as td:
        for rel in MERGE:
            paths = []
            for tag, data in (
                ("ours", lf(open(os.path.join(DST, rel), "rb").read())),
                ("base", blob(BASE, rel)),
                ("theirs", blob(TIP, rel)),
            ):
                p = os.path.join(td, tag)
                with open(p, "wb") as fh:
                    fh.write(data)
                paths.append(p)
            res = subprocess.run(
                ["git", "merge-file", "-p", *paths], capture_output=True
            )
            if res.returncode != 0:
                print(f"ERROR: conflicts ({res.returncode}) in {rel}", file=sys.stderr)
                return 2
            out[rel] = res.stdout
            rows.append((rel, "merge3(clean)", f"ours=mono-wt base={BASE} theirs={blob_id(TIP, rel)}"))
    for rel, data in out.items():
        assert b"\r\n" not in data or rel.endswith((".png", ".bin")), rel
        if dry:
            continue
        dst = os.path.join(DST, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(data)
    if not dry:
        with open(MANIFEST, "w", newline="\n") as fh:
            fh.write(f"# rust-direct overlay into {DST}\n")
            fh.write(f"# base {BASE} tip {TIP} (users/vanantha/rocke/rust-compiler); mono 89209ac7665\n")
            fh.write("# path<TAB>class<TAB>source blob\n")
            for row in sorted(rows):
                fh.write("\t".join(row) + "\n")
    counts: dict[str, int] = {}
    for _, cls, _ in rows:
        counts[cls] = counts.get(cls, 0) + 1
    print(("DRY " if dry else "") + f"overlay: {len(rows)} files {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
