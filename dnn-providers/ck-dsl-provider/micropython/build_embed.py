#!/usr/bin/env python3
"""Cross-platform driver for the embedded-MicroPython static library.

Replaces the bash + GNU-make pipeline (the old build_embed.sh + gen.mk, which
pulled in MicroPython's embed.mk / mkrules.mk) so the build runs anywhere Python
+ a C compiler are present -- notably Windows, which has no `make`. It does the
same work the make pipeline did, but drives MicroPython's own build tools
(makeqstrdefs.py / makeqstrdata.py / makemoduledefs.py / make_root_pointers.py /
makeversionhdr.py / makemanifest.py) directly instead of via make.

mpy-cross (needed by the frozen/mpy modes) is itself built from the pinned
MicroPython source here, make-free, by reusing the same genhdr machinery -- so
it is obtained the same way on every platform and is guaranteed to match the
pinned runtime's .mpy format + compiler (no released pip wheel matches our pin,
which needs newer-than-released f-string support for the codegen). Supply
CKDSL_MPY_CROSS_BIN / MPY_CROSS to use a prebuilt one instead.

Pipeline (all out-of-source, under $OUT_DIR):
  1. ensure a MicroPython checkout (clone @ pin if absent)
  2. (frozen/mpy) build mpy-cross from source
  3. build_bundle.py  : stage ck_dsl + ck_dsl_provider -> bundle
  4. build_frozen.py  : capture the codegen closure -> frozen_src
  5. generate the embed package (py/ sources + genhdr) -- the de-made embed.mk
  6. frozen mode only : makemanifest.py -> frozen_content.c (uses mpy-cross)
  7. compile the embed C + (frozen) frozen_content.c into libckdsl_micropython.a

Inputs come from the environment (CMake supplies them); sensible defaults let it
run standalone for debugging. See cmake/CkDslMicroPython.cmake.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROVIDER_ROOT = HERE.parent
REPO_ROOT = PROVIDER_ROOT.parent.parent
IS_WINDOWS = os.name == "nt"
EXE = ".exe" if IS_WINDOWS else ""


def env_path(name, default):
    v = os.environ.get(name)
    return Path(v) if v else Path(default)


def env_str(name, default):
    return os.environ.get(name) or default


MPY_COMMIT = env_str("MPY_COMMIT", "44a569b637")
PROVIDER_C_DIR = env_path("PROVIDER_C_DIR", PROVIDER_ROOT / "src" / "micropython")
CK_DSL_SRC = env_path(
    "CK_DSL_SRC", REPO_ROOT / "projects/composablekernel/python/ck_dsl"
)
CK_DSL_PROVIDER_SRC = env_path(
    "CK_DSL_PROVIDER_SRC", PROVIDER_ROOT / "python/ck_dsl_provider"
)
OUT_DIR = env_path("OUT_DIR", PROVIDER_ROOT / "build-micropython")
MPY_DIR = env_path("MPY_DIR", OUT_DIR / "micropython")
ROCM_PATH = env_path("ROCM_PATH", "/opt/rocm")
CC = env_str("CC", "clang" if IS_WINDOWS else "cc")
AR = env_str("AR", "llvm-ar" if IS_WINDOWS else "ar")
CKDSL_MODE = env_str("CKDSL_MODE", "frozen")
JOBS = str(os.cpu_count() or 4)

BUNDLE_DIR = OUT_DIR / "ckbundle"
FROZEN_DIR = OUT_DIR / "frozen_src"
PKG_DIR = OUT_DIR / "micropython_embed"
BUILD_DIR = OUT_DIR / "build-embed"
GENHDR = BUILD_DIR / "genhdr"
OBJ_DIR = OUT_DIR / "obj"
LIB = OUT_DIR / "libckdsl_micropython.a"
MPYX_BUILD = OUT_DIR / "mpy-cross-build"

# On Windows our clang targets the MSVC ABI (Target: *-windows-msvc, _MSC_VER
# defined), so the POSIX headers MicroPython's sources reach for -- <unistd.h>,
# <sys/time.h>, <dirent.h> -- are absent. MicroPython ships shims for exactly
# these under ports/windows/msvc and its own MSVC build (paths.props) adds that
# directory to the include path along with a couple of CRT-quieting defines.
# Mirror that here so the make-free build resolves them the same way.
#
# MICROPY_GCREGS_SETJMP: gchelper_generic.c captures callee-save registers via
# GCC-style named-register asm (`register long rbx asm("rbx")`) on the
# __x86_64__ path -- which clang defines even for the windows-msvc target, and
# then ICEs on. mpy-cross/mpconfigport.h only enables the portable setjmp-based
# capture when the arch is NOT x86_64/etc, so it misses this case. ports/windows/
# mpconfigport.h forces setjmp unconditionally; do the same here for both the
# mpy-cross and embed compiles.
#
# MICROPY_FLOAT_USE_NATIVE_FLT16: mpconfig.h auto-enables native _Float16 when
# __FLT16_MAX__ is defined (clang defines it for this target), and binary.c's
# half-float conversions then emit __extendhfsf2/__truncsfhf2 -- compiler-rt
# builtins that aren't linked on the msvc target, breaking both the mpy-cross
# and final-plugin links. Disable it so MicroPython's bundled software
# half-float path is used instead (matching what ports/unix variants do); the
# conversions are bit-identical IEEE-754, only without native instructions.
WIN_MSVC_CFLAGS = (
    [
        f"-I{MPY_DIR / 'ports' / 'windows' / 'msvc'}",
        "-D_CRT_SECURE_NO_WARNINGS",
        "-D_CRT_NONSTDC_NO_WARNINGS",
        "-DMICROPY_GCREGS_SETJMP=1",
        "-DMICROPY_FLOAT_USE_NATIVE_FLT16=0",
    ]
    if IS_WINDOWS
    else []
)


def run(cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def run_out(cmd, **kw):
    return subprocess.run(
        [str(c) for c in cmd], check=True, stdout=subprocess.PIPE, **kw
    ).stdout


def ensure_micropython():
    if not (MPY_DIR / ".git").is_dir():
        print(f"== cloning MicroPython @ {MPY_COMMIT}")
        run(
            [
                "git",
                "clone",
                "--quiet",
                "https://github.com/micropython/micropython",
                MPY_DIR,
            ]
        )
        run(["git", "-C", MPY_DIR, "checkout", "--quiet", MPY_COMMIT])


# --- genhdr generation (the de-made embed.mk / mkrules.mk qstr pipeline) -----
def generate_genhdr(
    header_build, cflags, qstr_gen_flags, qstr_sources, qstr_defs_files
):
    """Produce mpversion.h / qstrdefs.generated.h / moduledefs.h / root_pointers.h
    in header_build, by driving MicroPython's own qstr tools (no make). Shared by
    the embed package and the mpy-cross build.

    cflags          base CFLAGS (incl. -I for mpconfigport.h, and any frozen defines)
    qstr_gen_flags  scan-only extra flags (-DNO_QSTR [+ -DCKDSL_ON_DISK])
    qstr_sources    sources to scan for qstrs/modules/root-pointers (a safe superset)
    qstr_defs_files extra QSTR_DEFS to cat into qstrdefs.generated (e.g. qstrdefsport.h)
    """
    py_src = MPY_DIR / "py"
    header_build.mkdir(parents=True, exist_ok=True)

    def tool(name):
        return py_src / name

    # mpversion.h must exist BEFORE the qstr scan: it is a QSTR_GLOBAL_REQUIREMENT
    # in make (some sources #include genhdr/mpversion.h unconditionally).
    run([sys.executable, tool("makeversionhdr.py"), header_build / "mpversion.h"])

    run(
        [
            sys.executable,
            tool("makeqstrdefs.py"),
            "pp",
            CC,
            "-E",
            "output",
            header_build / "qstr.i.last",
            "cflags",
            *cflags,
            *qstr_gen_flags,
            "cxxflags",
            "sources",
            *qstr_sources,
            "dependencies",
            "changed_sources",
            *qstr_sources,
        ]
    )

    collected = {
        "qstr": header_build / "qstrdefs.collected.h",
        "module": header_build / "moduledefs.collected",
        "root_pointer": header_build / "root_pointers.collected",
        "compress": header_build / "compressed.collected",
    }
    for mode, out_file in collected.items():
        run(
            [
                sys.executable,
                tool("makeqstrdefs.py"),
                "split",
                mode,
                header_build / "qstr.i.last",
                header_build / mode,
                "_",
            ]
        )
        run(
            [
                sys.executable,
                tool("makeqstrdefs.py"),
                "cat",
                mode,
                "_",
                header_build / mode,
                out_file,
            ]
        )

    # qstrdefs.generated.h: cat(py/qstrdefs.h + <port qstrdefs> + collected) | sed |
    # cpp | sed -> makeqstrdata. The seds (regex here) wrap/unwrap Q(...) so cpp
    # leaves them intact while expanding conditionals. Uses base CFLAGS (no scan
    # flags), matching py.mk's $(CPP) $(CFLAGS).
    qd = (py_src / "qstrdefs.h").read_text()
    for extra in qstr_defs_files:
        qd += Path(extra).read_text()
    qd += collected["qstr"].read_text()
    qd = re.sub(r"(?m)^(Q\(.*)$", r'"\1"', qd)
    pre = run_out([CC, "-E", *cflags, "-"], input=qd.encode()).decode()
    pre = re.sub(r'(?m)^"(Q\(.*\))"$', r"\1", pre)
    (header_build / "qstrdefs.preprocessed.h").write_text(pre)
    (header_build / "qstrdefs.generated.h").write_bytes(
        run_out(
            [
                sys.executable,
                tool("makeqstrdata.py"),
                header_build / "qstrdefs.preprocessed.h",
            ]
        )
    )
    (header_build / "moduledefs.h").write_bytes(
        run_out([sys.executable, tool("makemoduledefs.py"), collected["module"]])
    )
    (header_build / "root_pointers.h").write_bytes(
        run_out(
            [sys.executable, tool("make_root_pointers.py"), collected["root_pointer"]]
        )
    )


# --- mpy-cross built from the pinned source (make-free, every platform) ------
def resolve_mpy_cross():
    """Path to an mpy-cross matching the pin. Prefer an explicitly supplied binary
    (CKDSL_MPY_CROSS_BIN / MPY_CROSS); otherwise build it from source. No released
    pip wheel matches our pin (the codegen needs newer-than-released f-strings),
    so building from source is the portable, guaranteed-compatible option."""
    explicit = os.environ.get("CKDSL_MPY_CROSS_BIN") or os.environ.get("MPY_CROSS")
    if explicit and Path(explicit).exists():
        return str(explicit)
    return build_mpy_cross()


def build_mpy_cross():
    mxdir = MPY_DIR / "mpy-cross"
    py_src = MPY_DIR / "py"
    prog = MPYX_BUILD / ("mpy-cross" + EXE)
    print("== building mpy-cross from source (make-free)")

    # mpy-cross compiles the py core + main.c/gccollect.c with its own config
    # (mpy-cross/mpconfigport.h, qstrdefsport.h). Include dirs mirror its Makefile:
    # -I<mpy-cross dir> -I<TOP> -I<build dir>.
    cflags = [
        f"-I{mxdir}",
        f"-I{MPY_DIR}",
        f"-I{MPYX_BUILD}",
        "-std=gnu99",
        "-Og",
        "-fno-common",
        "-Wall",
        *WIN_MSVC_CFLAGS,
    ]
    py_c = sorted(str(p) for p in py_src.glob("*.c"))
    src_c = [
        str(mxdir / "main.c"),
        str(mxdir / "gccollect.c"),
        str(MPY_DIR / "shared/runtime/gchelper_generic.c"),
    ]
    if IS_WINDOWS:
        src_c.append(str(MPY_DIR / "ports/windows/fmode.c"))

    # genhdr for mpy-cross (its qstr pool), then compile + link an executable.
    generate_genhdr(
        MPYX_BUILD / "genhdr",
        cflags,
        ["-DNO_QSTR"],
        py_c + [str(mxdir / "main.c")],
        [str(mxdir / "qstrdefsport.h")],
    )

    obj_dir = MPYX_BUILD / "obj"
    if obj_dir.exists():
        shutil.rmtree(obj_dir)
    obj_dir.mkdir(parents=True, exist_ok=True)
    objs = []
    for i, src in enumerate(py_c + src_c):
        obj = obj_dir / f"{i:03d}_{Path(src).stem}.o"
        run([CC, *cflags, "-c", src, "-o", obj])
        objs.append(obj)
    link = [CC, "-o", prog, *objs]
    if not IS_WINDOWS:
        link.append("-lm")
    run(link)
    print(f"== mpy-cross built: {prog}")
    return str(prog)


# --- embed package + genhdr (replaces embed.mk / mkrules.mk) -----------------
def embed_cflags(on_disk):
    """CFLAGS the qstr scan + qstrdefs preprocess use. Mirrors embed.mk:
    -I. (== PROVIDER_C_DIR, where mpconfigport.h lives) -I$(TOP) -I$(BUILD)
    -I$(EMBED_PORT). -Werror dropped vs embed.mk (it matters only for compiling)."""
    flags = [
        f"-I{PROVIDER_C_DIR}",
        f"-I{MPY_DIR}",
        f"-I{BUILD_DIR}",
        f"-I{MPY_DIR / 'ports' / 'embed'}",
        "-std=c99",
        *WIN_MSVC_CFLAGS,
    ]
    if not on_disk:
        # Frozen build: make adds these to global CFLAGS so the qstr scan sees the
        # frozen-only qstrs (e.g. MP_QSTR_.frozen in runtime.c). They also enable
        # mpconfigport's MICROPY_QSTR_EXTRA_POOL.
        flags += ["-DMICROPY_MODULE_FROZEN_MPY", "-DMICROPY_MODULE_FROZEN_STR"]
    return flags


def generate_embed_package(on_disk):
    py_src = MPY_DIR / "py"
    embed_port = MPY_DIR / "ports" / "embed"

    cflags = embed_cflags(on_disk)
    qstr_gen_flags = ["-DNO_QSTR"] + (["-DCKDSL_ON_DISK=1"] if on_disk else [])

    # SRC_QSTR: a SUPERSET of make's PY_CORE_O sources -- every py/*.c plus
    # extmod/modre.c and the provider's modcomgr.c. Over-scanning is safe (extra
    # qstrs are harmless; only a missing one would break the pool) and avoids
    # replicating make's config-conditional PY_CORE_O_BASENAME exactly.
    sources = sorted(str(p) for p in py_src.glob("*.c"))
    sources += [str(MPY_DIR / "extmod" / "modre.c"), str(PROVIDER_C_DIR / "modcomgr.c")]

    print("== generate embed genhdr")
    generate_genhdr(GENHDR, cflags, qstr_gen_flags, sources, [])

    # Assemble micropython_embed/ (the package the .a compiles from).
    print("== assemble embed package")
    for sub in ("py", "extmod", "shared/runtime", "genhdr", "port"):
        d = PKG_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    for f in list(py_src.glob("*.c")) + list(py_src.glob("*.h")):
        shutil.copy2(f, PKG_DIR / "py")
    shutil.copy2(MPY_DIR / "extmod" / "modplatform.h", PKG_DIR / "extmod")
    shutil.copy2(MPY_DIR / "shared/runtime/gchelper.h", PKG_DIR / "shared/runtime")
    shutil.copy2(
        MPY_DIR / "shared/runtime/gchelper_generic.c", PKG_DIR / "shared/runtime"
    )
    for h in ("moduledefs.h", "mpversion.h", "qstrdefs.generated.h", "root_pointers.h"):
        shutil.copy2(GENHDR / h, PKG_DIR / "genhdr")
    for f in list(embed_port.glob("port/*.c")) + list(embed_port.glob("port/*.h")):
        shutil.copy2(f, PKG_DIR / "port")


def freeze_modules(mpy_cross):
    """makemanifest.py -> frozen_content.c (frozen mode). Mirrors mkrules.mk."""
    print("== freeze modules (makemanifest.py)")
    e = dict(os.environ)
    e["MICROPY_MPYCROSS"] = str(mpy_cross)
    e["CKDSL_FROZEN_DIR"] = str(FROZEN_DIR)
    # Mirror make's MANIFEST_VARIABLES (MICROPY_MANIFEST_*): MPY_DIR + PORT_DIR are
    # used; MPY_LIB_DIR + BOARD_DIR must still be present (manifestfile.py indexes
    # MPY_LIB_DIR directly) -- empty disables micropython-lib loading.
    run(
        [
            sys.executable,
            MPY_DIR / "tools" / "makemanifest.py",
            "-o",
            BUILD_DIR / "frozen_content.c",
            "-v",
            f"MPY_DIR={MPY_DIR}",
            "-v",
            f"PORT_DIR={PROVIDER_C_DIR}",
            "-v",
            "MPY_LIB_DIR=",
            "-v",
            "BOARD_DIR=",
            "-b",
            BUILD_DIR,
            HERE / "manifest.py",
        ],
        env=e,
    )


# --- compile the static library ---------------------------------------------
def compile_lib(on_disk, frozen):
    if OBJ_DIR.exists():
        shutil.rmtree(OBJ_DIR)
    OBJ_DIR.mkdir(parents=True, exist_ok=True)

    cflags = [
        f"-I{PROVIDER_C_DIR}",
        f"-I{PKG_DIR}",
        f"-I{PKG_DIR / 'port'}",
        f"-I{BUILD_DIR}",
        f"-I{MPY_DIR}",
        "-fvisibility=hidden",
        "-Og",
        "-fno-common",
        "-Wall",
        *WIN_MSVC_CFLAGS,
    ]
    if not IS_WINDOWS:
        cflags.append("-fPIC")
    if frozen:
        cflags += ["-DMICROPY_MODULE_FROZEN_MPY=1", "-DMICROPY_MODULE_FROZEN_STR=1"]
    else:
        cflags.append("-DCKDSL_ON_DISK=1")

    srcs = sorted(glob.glob(str(PKG_DIR / "**" / "*.c"), recursive=True))
    srcs += [
        str(PROVIDER_C_DIR / "embed_port.c"),
        str(PROVIDER_C_DIR / "modcomgr.c"),
        str(PROVIDER_C_DIR / "comgr_compile.c"),
        str(MPY_DIR / "extmod" / "modre.c"),
    ]
    if frozen:
        srcs.append(str(BUILD_DIR / "frozen_content.c"))

    print(f"== compiling {len(srcs)} embed sources")
    objs = []
    for i, src in enumerate(srcs):
        obj = OBJ_DIR / f"{i:03d}_{Path(src).stem}.o"
        extra = (
            [f"-I{ROCM_PATH / 'include'}"]
            if Path(src).name == "comgr_compile.c"
            else []
        )
        run([CC, *cflags, *extra, "-c", src, "-o", obj])
        objs.append(obj)

    if LIB.exists():
        LIB.unlink()
    run([AR, "rcs", LIB, *objs])
    print(f"== done: {LIB} ({LIB.stat().st_size // 1024} KiB)")


def main():
    on_disk = CKDSL_MODE != "frozen"
    frozen = CKDSL_MODE == "frozen"
    print(f"== ck-dsl embed build  OUT_DIR={OUT_DIR}  MODE={CKDSL_MODE}  CC={CC}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_micropython()
    mpy_cross = resolve_mpy_cross() if CKDSL_MODE in ("frozen", "mpy") else None

    print("== build_bundle.py")
    run(
        [sys.executable, HERE / "build_bundle.py"],
        env={
            **os.environ,
            "CK_DSL_SRC": str(CK_DSL_SRC),
            "CK_DSL_PROVIDER_SRC": str(CK_DSL_PROVIDER_SRC),
            "BUNDLE_DIR": str(BUNDLE_DIR),
        },
    )
    print("== build_frozen.py")
    run(
        [sys.executable, HERE / "build_frozen.py"],
        env={
            **os.environ,
            "BUNDLE_DIR": str(BUNDLE_DIR),
            "SHIMS_DIR": str(HERE / "shims"),
            "FROZEN_DIR": str(FROZEN_DIR),
        },
    )

    # Regenerate genhdr from scratch each run (the qstr pool differs frozen vs
    # on-disk, and a stale pool would silently mismatch the compile flags).
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    generate_embed_package(on_disk)

    if frozen:
        freeze_modules(mpy_cross)
    elif CKDSL_MODE == "mpy":
        mpy_out = OUT_DIR / "frozen_src_mpy"
        if mpy_out.exists():
            shutil.rmtree(mpy_out)
        print("== compiling on-disk modules to .mpy")
        for py in FROZEN_DIR.rglob("*.py"):
            dst = mpy_out / py.relative_to(FROZEN_DIR).with_suffix(".mpy")
            dst.parent.mkdir(parents=True, exist_ok=True)
            run([mpy_cross, "-o", dst, py])

    compile_lib(on_disk, frozen)


if __name__ == "__main__":
    main()
