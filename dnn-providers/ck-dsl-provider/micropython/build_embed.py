#!/usr/bin/env python3
"""Cross-platform driver for the embedded-MicroPython static library.

Replaces the bash + GNU-make pipeline (the old build_embed.sh + gen.mk, which
pulled in MicroPython's embed.mk / mkrules.mk) so the build runs anywhere Python
+ a C compiler are present -- notably Windows, which has no `make`. It does the
same work the make pipeline did, but drives MicroPython's own build tools
(makeqstrdefs.py / makeqstrdata.py / makemoduledefs.py / make_root_pointers.py /
makeversionhdr.py / makemanifest.py) directly instead of via make.

Pipeline (all out-of-source, under $OUT_DIR):
  1. ensure a MicroPython checkout (clone @ pin if absent) + resolve mpy-cross
  2. build_bundle.py  : stage ck_dsl + ck_dsl_provider -> bundle
  3. build_frozen.py  : capture the codegen closure -> frozen_src
  4. generate the embed package (py/ sources + genhdr) -- the de-made embed.mk
  5. frozen mode only : makemanifest.py -> frozen_content.c (uses mpy-cross)
  6. compile the embed C + (frozen) frozen_content.c into libckdsl_micropython.a

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


def run(cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def run_out(cmd, **kw):
    return subprocess.run(
        [str(c) for c in cmd], check=True, stdout=subprocess.PIPE, **kw
    ).stdout


# --- 1. MicroPython source + mpy-cross -------------------------------------
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


def mpy_format_version():
    """The .mpy format version this MicroPython pin expects (MPY_VERSION)."""
    text = (MPY_DIR / "py" / "persistentcode.h").read_text()
    m = re.search(r"#define\s+MPY_VERSION\s+\(?(\d+)", text)
    return int(m.group(1)) if m else None


def assert_mpy_cross_compatible(mpy_cross_cmd):
    """Compile a trivial module and check the .mpy version byte matches the pin --
    guards the pip-wheel path (frozen/mpy reject a mismatched .mpy at load)."""
    want = mpy_format_version()
    if want is None:
        return
    probe_py = OUT_DIR / "_mpyprobe.py"
    probe_mpy = OUT_DIR / "_mpyprobe.mpy"
    probe_py.write_text("x = 1\n")
    run(mpy_cross_cmd + ["-o", probe_mpy, probe_py])
    got = probe_mpy.read_bytes()[1] if probe_mpy.stat().st_size >= 2 else None
    probe_py.unlink(missing_ok=True)
    probe_mpy.unlink(missing_ok=True)
    if got != want:
        raise SystemExit(
            f"mpy-cross .mpy version {got} != MicroPython pin {MPY_COMMIT} expects {want}. "
            f"Install an mpy-cross matching the pin (pip install 'mpy-cross==<matching>'), "
            f"or set CKDSL_MPY_CROSS_BIN to a compatible binary."
        )


def resolve_mpy_cross():
    """Return the argv prefix to invoke mpy-cross (frozen/mpy modes only).

    Tiered: explicit binary -> native build (Unix; same pinned source, guaranteed
    .mpy match) -> pip 'mpy_cross' wheel (Windows primary; version-checked) ->
    mingw32-make build -> actionable error.
    """
    explicit = os.environ.get("CKDSL_MPY_CROSS_BIN") or os.environ.get("MPY_CROSS")
    if explicit and Path(explicit).exists():
        return [explicit]

    native = MPY_DIR / "mpy-cross" / "build" / ("mpy-cross" + EXE)
    if not IS_WINDOWS and shutil.which("make"):
        print("== building mpy-cross (make)")
        run(
            ["make", "-C", MPY_DIR / "mpy-cross", "-j", JOBS], stdout=subprocess.DEVNULL
        )
        if native.exists():
            return [str(native)]
    if native.exists():
        return [str(native)]

    try:
        import mpy_cross  # noqa: F401  (the PyPI wheel)

        cmd = [sys.executable, "-m", "mpy_cross"]
        print("== using mpy-cross from the 'mpy_cross' PyPI wheel")
        assert_mpy_cross_compatible(cmd)
        return cmd
    except ImportError:
        pass

    if shutil.which("mingw32-make"):
        print("== building mpy-cross (mingw32-make)")
        run(["mingw32-make", "-C", MPY_DIR / "mpy-cross"])
        if native.exists():
            return [str(native)]

    raise SystemExit(
        "mpy-cross is required for the 'frozen'/'mpy' modes but could not be obtained.\n"
        "  - install the prebuilt wheel: pip install mpy-cross   (must match MicroPython "
        f"pin {MPY_COMMIT}), or\n"
        "  - set CKDSL_MPY_CROSS_BIN to a compatible mpy-cross binary, or\n"
        "  - install GNU make / mingw32-make so it can be built from source."
    )


# --- 4. Embed package + genhdr (replaces embed.mk / mkrules.mk) -------------
def embed_cflags(on_disk):
    """CFLAGS the qstr scan + qstrdefs preprocess use. Mirrors embed.mk:
    CFLAGS += -I. -I$(TOP) -I$(BUILD) -I$(EMBED_PORT)  (-I. == PROVIDER_C_DIR,
    where mpconfigport.h lives). -Werror is dropped vs embed.mk: it matters only
    for compiling, and the qstr scan runs cpp -E over many sources."""
    flags = [
        f"-I{PROVIDER_C_DIR}",
        f"-I{MPY_DIR}",
        f"-I{BUILD_DIR}",
        f"-I{MPY_DIR / 'ports' / 'embed'}",
        "-std=c99",
    ]
    if not on_disk:
        # Frozen build: make adds these to global CFLAGS, so the qstr scan sees
        # the frozen-only qstrs (e.g. MP_QSTR_.frozen in runtime.c, gated by
        # MICROPY_MODULE_FROZEN_MPY). They also enable mpconfigport's
        # MICROPY_QSTR_EXTRA_POOL.
        flags += ["-DMICROPY_MODULE_FROZEN_MPY", "-DMICROPY_MODULE_FROZEN_STR"]
    return flags


def generate_embed_package(on_disk):
    py_src = MPY_DIR / "py"
    embed_port = MPY_DIR / "ports" / "embed"
    GENHDR.mkdir(parents=True, exist_ok=True)

    def tool(name):
        return py_src / name

    cflags = embed_cflags(on_disk)
    qstr_gen_flags = ["-DNO_QSTR"] + (["-DCKDSL_ON_DISK=1"] if on_disk else [])

    # mpversion.h must exist BEFORE the qstr scan: it is a QSTR_GLOBAL_REQUIREMENT
    # in make (some sources, e.g. modsys.c, #include genhdr/mpversion.h
    # unconditionally, even under -DNO_QSTR).
    run([sys.executable, tool("makeversionhdr.py"), GENHDR / "mpversion.h"])

    # SRC_QSTR: a SUPERSET of make's PY_CORE_O sources -- every py/*.c plus
    # extmod/modre.c and the provider's modcomgr.c. Over-scanning is safe (extra
    # qstrs are harmless; only missing one would break the pool), and it avoids
    # having to replicate make's config-conditional PY_CORE_O_BASENAME exactly.
    sources = sorted(str(p) for p in py_src.glob("*.c"))
    sources += [str(MPY_DIR / "extmod" / "modre.c"), str(PROVIDER_C_DIR / "modcomgr.c")]

    print("== qstr scan (makeqstrdefs.py pp)")
    run(
        [
            sys.executable,
            tool("makeqstrdefs.py"),
            "pp",
            CC,
            "-E",
            "output",
            GENHDR / "qstr.i.last",
            "cflags",
            *cflags,
            *qstr_gen_flags,
            "cxxflags",
            "sources",
            *sources,
            "dependencies",
            "changed_sources",
            *sources,
        ]
    )

    # split + cat for each extraction mode -> the .collected files.
    collected = {
        "qstr": GENHDR / "qstrdefs.collected.h",
        "module": GENHDR / "moduledefs.collected",
        "root_pointer": GENHDR / "root_pointers.collected",
        "compress": GENHDR / "compressed.collected",
    }
    for mode, out_file in collected.items():
        run(
            [
                sys.executable,
                tool("makeqstrdefs.py"),
                "split",
                mode,
                GENHDR / "qstr.i.last",
                GENHDR / mode,
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
                GENHDR / mode,
                out_file,
            ]
        )

    # qstrdefs.generated.h: cat(py/qstrdefs.h + collected) | sed | cpp | sed -> makeqstrdata.
    # The two seds (reimplemented as regex) wrap/unwrap Q(...) lines so cpp leaves
    # them intact while expanding any conditionals. Uses base CFLAGS (NOT the
    # NO_QSTR/CKDSL_ON_DISK scan flags) -- matching py.mk's $(CPP) $(CFLAGS).
    qd = (py_src / "qstrdefs.h").read_text() + collected["qstr"].read_text()
    qd = re.sub(r"(?m)^(Q\(.*)$", r'"\1"', qd)
    pre = run_out([CC, "-E", *cflags, "-"], input=qd.encode()).decode()
    pre = re.sub(r'(?m)^"(Q\(.*\))"$', r"\1", pre)
    (GENHDR / "qstrdefs.preprocessed.h").write_text(pre)
    (GENHDR / "qstrdefs.generated.h").write_bytes(
        run_out(
            [
                sys.executable,
                tool("makeqstrdata.py"),
                GENHDR / "qstrdefs.preprocessed.h",
            ]
        )
    )

    (GENHDR / "moduledefs.h").write_bytes(
        run_out([sys.executable, tool("makemoduledefs.py"), collected["module"]])
    )
    (GENHDR / "root_pointers.h").write_bytes(
        run_out(
            [sys.executable, tool("make_root_pointers.py"), collected["root_pointer"]]
        )
    )

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


def freeze_modules(mpy_cross_cmd):
    """makemanifest.py -> frozen_content.c (frozen mode). Mirrors mkrules.mk."""
    print("== freeze modules (makemanifest.py)")
    e = dict(os.environ)
    e["MICROPY_MPYCROSS"] = mpy_cross_cmd[-1] if len(mpy_cross_cmd) == 1 else ""
    e["CKDSL_FROZEN_DIR"] = str(FROZEN_DIR)
    # The wheel path (python -m mpy_cross) has no single-binary path; makemanifest
    # invokes MICROPY_MPYCROSS as one executable, so a wheel needs a shim. When
    # mpy_cross_cmd isn't a lone binary, write a tiny launcher.
    if len(mpy_cross_cmd) != 1:
        shim = OUT_DIR / ("mpy_cross_shim" + (".bat" if IS_WINDOWS else ""))
        if IS_WINDOWS:
            shim.write_text(f'@echo off\r\n"{mpy_cross_cmd[0]}" -m mpy_cross %*\r\n')
        else:
            shim.write_text(
                f'#!/usr/bin/env bash\nexec "{mpy_cross_cmd[0]}" -m mpy_cross "$@"\n'
            )
            shim.chmod(0o755)
        e["MICROPY_MPYCROSS"] = str(shim)
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


# --- 6. Compile the static library -----------------------------------------
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
    mpy_cross_cmd = resolve_mpy_cross() if CKDSL_MODE in ("frozen", "mpy") else None

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
        freeze_modules(mpy_cross_cmd)
    elif CKDSL_MODE == "mpy":
        mpy_out = OUT_DIR / "frozen_src_mpy"
        if mpy_out.exists():
            shutil.rmtree(mpy_out)
        print("== compiling on-disk modules to .mpy")
        for py in FROZEN_DIR.rglob("*.py"):
            dst = mpy_out / py.relative_to(FROZEN_DIR).with_suffix(".mpy")
            dst.parent.mkdir(parents=True, exist_ok=True)
            run(mpy_cross_cmd + ["-o", dst, py])

    compile_lib(on_disk, frozen)


if __name__ == "__main__":
    main()
