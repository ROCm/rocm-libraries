# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import shutil
import subprocess
import sys
from pathlib import Path

from invoke import task

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_PATH = Path(__file__).resolve().parent
BUILD_DIR = ROOT_PATH / "build"

# ---------------------------------------------------------------------------
# Helpers – distro detection
# ---------------------------------------------------------------------------

def _os_release() -> dict:
    if sys.platform == "win32":
        return {}
    info = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, _, v = line.partition("=")
                    info[k] = v.strip('"')
    except FileNotFoundError:
        pass
    return info


def _distro_id() -> str:
    return _os_release().get("ID", "")


def _version_id() -> str:
    return _os_release().get("VERSION_ID", "")


def _supported_distros():
    if sys.platform == "win32":
        return
    supported = {"ubuntu", "centos", "almalinux", "rhel", "fedora", "sles",
                 "opensuse-leap", "mariner", "azurelinux"}
    distro = _distro_id()
    if distro not in supported:
        print(f"Unsupported distro '{distro}'. Supported: {', '.join(sorted(supported))}")
        sys.exit(2)



def _setup_msvc_env():
    """Initialize the full MSVC build environment from vcvarsall.bat."""
    if "VSINSTALLDIR" in os.environ:
        return  # Already inside a Developer Command Prompt
    # Use vswhere.exe (ships with VS 2017+ Installer) to find the latest VS installation.
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / \
        "Microsoft Visual Studio/Installer/vswhere.exe"
    vcvarsall = None
    if vswhere.exists():
        result = subprocess.run(
            [str(vswhere), "-latest", "-property", "installationPath"],
            capture_output=True, text=True,
        )
        vs_path = result.stdout.strip()
        if vs_path:
            candidate = Path(vs_path) / "VC/Auxiliary/Build/vcvarsall.bat"
            if candidate.exists():
                vcvarsall = candidate
    # Fallback: manual glob for BuildTools installs that vswhere may not index.
    if vcvarsall is None:
        vs_root = Path("C:/Program Files (x86)/Microsoft Visual Studio")
        candidates = sorted(vs_root.glob("*/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"), reverse=True)
        if candidates:
            vcvarsall = candidates[0]
    if vcvarsall is None:
        print("Warning: vcvarsall.bat not found. Ensure Visual Studio Build Tools with C++ workload is installed.")
        return
    result = subprocess.run(
        f'"{vcvarsall}" amd64 && set',
        capture_output=True, text=True, encoding="mbcs", shell=True,
    )
    original_lib = os.environ.get("LIB", "")
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ[key] = value
    # Restore original LIB entries so vcvarsall doesn't drop existing SDK paths
    if original_lib:
        existing = os.environ.get("LIB", "")
        merged = ";".join(dict.fromkeys(existing.split(";") + original_lib.split(";")))
        os.environ["LIB"] = merged

    # Ensure Windows SDK lib and include paths are present (vcvarsall.bat may omit them).
    prog86 = os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")
    sdk_lib_root = Path(prog86) / "Windows Kits/10/lib"
    sdk_versions = sorted(sdk_lib_root.glob("*/"), reverse=True) if sdk_lib_root.exists() else []
    if sdk_versions:
        sdk_ver = sdk_versions[0].name
        sdk_root = sdk_lib_root.parent

        sdk_libs = [
            str(sdk_root / "lib" / sdk_ver / "ucrt/x64"),
            str(sdk_root / "lib" / sdk_ver / "um/x64"),
        ]
        current_lib = os.environ.get("LIB", "")
        missing_libs = [p for p in sdk_libs if p not in current_lib]
        if missing_libs:
            os.environ["LIB"] = ";".join(missing_libs + ([current_lib] if current_lib else []))

        sdk_includes = [
            str(p) for p in (sdk_root / "include" / sdk_ver).iterdir() if p.is_dir()
        ]
        current_include = os.environ.get("INCLUDE", "")
        missing_includes = [p for p in sdk_includes if p not in current_include]
        if missing_includes:
            os.environ["INCLUDE"] = ";".join(missing_includes + ([current_include] if current_include else []))


def _setup_clang_shims(build_subdir: Path):
    """Create clang.exe/clang++.exe hardlinks pointing to amdclang equivalents.

    Tensile's toolchain validator looks for 'clang.exe' / 'clang++.exe' by name.
    The ROCm Windows SDK ships these as 'amdclang.exe' / 'amdclang++.exe'.
    We create hardlinks in a shims directory and prepend it to PATH.
    """
    aliases = {
        "clang.exe":   "amdclang.exe",
        "clang++.exe": "amdclang++.exe",
        # clang-offload-bundler.exe already has the right name in lib/llvm/bin/
    }
    shims_dir = build_subdir / "clang-shims"
    shims_dir.mkdir(parents=True, exist_ok=True)
    for alias, real in aliases.items():
        target = shutil.which(real)
        if not target:
            continue
        link = shims_dir / alias
        if not link.exists():
            try:
                os.link(target, link)  # hardlink — no admin required
            except OSError:
                shutil.copy2(target, link)  # fallback: copy
        print(f"clang shim: {alias} -> {target}")
    os.environ["PATH"] = str(shims_dir) + os.pathsep + os.environ.get("PATH", "")



def _rmtree(path: Path):
    """Remove a directory tree, handling read-only files on Windows."""
    def _on_error(func, path, _exc):
        os.chmod(path, 0o666)
        func(path)
    shutil.rmtree(path, onexc=_on_error)


def _find_rc_exe() -> str:
    """Find rc.exe from PATH or Windows SDK installation directory."""
    rc = shutil.which("rc") or shutil.which("rc.exe")
    if rc:
        return rc
    # Search Windows SDK bin directories (installed by Visual Studio Build Tools).
    # Use ProgramFiles(x86) env var to handle non-default install locations.
    prog_x86 = os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")
    prog = os.environ.get("ProgramFiles", "C:/Program Files")
    for sdk_root in (
        Path(prog_x86) / "Windows Kits" / "10" / "bin",
        Path(prog) / "Windows Kits" / "10" / "bin",
    ):
        if not sdk_root.exists():
            continue
        for version_dir in sorted(sdk_root.iterdir(), reverse=True):
            for arch in ("x64", "x86"):
                candidate = version_dir / arch / "rc.exe"
                if candidate.exists():
                    return candidate.as_posix()
    return ""


def _is_root() -> bool:
    if sys.platform == "win32":
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    return os.getuid() == 0


def _elevate(c, cmd: str):
    if _is_root():
        c.run(cmd)
    else:
        c.run(f"sudo {cmd}")


# ---------------------------------------------------------------------------
# Helpers – package installation
# ---------------------------------------------------------------------------

def _apt_install(c, packages: list[str]):
    for pkg in packages:
        result = subprocess.run(
            ["dpkg-query", "--show", "--showformat=${db:Status-Abbrev}\n", pkg],
            capture_output=True, text=True,
        )
        if "ii" not in result.stdout:
            print(f"\033[32mInstalling \033[33m{pkg}\033[32m via apt\033[0m")
            _elevate(c, f"apt install -y --no-install-recommends {pkg}")


def _yum_install(c, packages: list[str], extra_opts: str = ""):
    for pkg in packages:
        result = subprocess.run(
            ["yum", "list", "installed", pkg],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"\033[32mInstalling \033[33m{pkg}\033[32m via yum\033[0m")
            opts = ""
            if pkg == "openblas-devel" and _distro_id() == "centos":
                opts = "--enablerepo=crb"
            _elevate(c, f"yum -y --nogpgcheck install {pkg} {opts}")


def _dnf_install(c, packages: list[str]):
    for pkg in packages:
        result = subprocess.run(
            ["dnf", "list", "installed", pkg],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"\033[32mInstalling \033[33m{pkg}\033[32m via dnf\033[0m")
            _elevate(c, f"dnf install -y {pkg}")


def _zypper_install(c, packages: list[str]):
    for pkg in packages:
        result = subprocess.run(
            ["rpm", "-q", pkg],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"\033[32mInstalling \033[33m{pkg}\033[32m via zypper\033[0m")
            _elevate(c, f"zypper -n --no-gpg-checks install {pkg}")


def _install_msgpack_from_source(c, build_dir: Path, cxx: str, cc: str):
    msgpack_dir = build_dir / "deps" / "msgpack-c"
    if not msgpack_dir.exists():
        deps_dir = build_dir / "deps"
        deps_dir.mkdir(parents=True, exist_ok=True)
        with c.cd(str(deps_dir)):
            c.run("git clone -b cpp-3.1.0 https://github.com/msgpack/msgpack-c.git --depth 1")
        with c.cd(str(msgpack_dir)):
            c.run("git fetch --unshallow")
            c.run(f"CXX={cxx} CC={cc} cmake -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF .")
            c.run("make")
            _elevate(c, "make install")


def _install_blis(c, build_dir: Path):
    blis_paths = [
        "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libblis-mt.a",
        "/opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libblis-mt.a",
        "/opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libblis-mt.a",
        "/usr/local/lib/libblis.a",
    ]
    if any(Path(p).exists() for p in blis_paths):
        return

    blis_dir = build_dir / "deps" / "blis"
    if (blis_dir / "lib" / "libblis.a").exists():
        return

    distro = _distro_id()
    if distro in ("centos", "rhel", "sles", "opensuse-leap", "almalinux"):
        url = "https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-centos-2.0.tar.gz"
    else:
        url = "https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-ubuntu-2.0.tar.gz"

    deps_dir = build_dir / "deps"
    deps_dir.mkdir(parents=True, exist_ok=True)
    with c.cd(str(deps_dir)):
        c.run(f"wget -nv -O blis.tar.gz {url}")
        c.run("tar -xvf blis.tar.gz")
    if (deps_dir / "amd-blis-mt").exists():
        shutil.rmtree(str(deps_dir / "blis"), ignore_errors=True)
        (deps_dir / "amd-blis-mt").rename(deps_dir / "blis")
    (deps_dir / "blis.tar.gz").unlink(missing_ok=True)
    lib_dir = deps_dir / "blis" / "lib"
    symlink = lib_dir / "libblis.a"
    if not symlink.exists():
        symlink.symlink_to("libblis-mt.a")


# ---------------------------------------------------------------------------
# invoke tasks
# ---------------------------------------------------------------------------

@task(
    help={
        "install_deps": "Install build dependencies before building.",
        "install_pkg": "Install the package after building.",
        "clients": "Build library clients.",
        "jobs": "Number of parallel build jobs (default: all cores).",
        "architecture": "GPU target(s), e.g. 'all' or 'gfx90a:xnack+;gfx90a:xnack-'.",
        "cpu_ref_lib": "CPU reference library for testing: 'blis' or 'lapack'.",
        "use_system_packages": "Use system-installed msgpack/blas/lapack (requires --install-deps).",
        "debug": "Build with CMAKE_BUILD_TYPE=Debug.",
        "relwithdebinfo": "Build with CMAKE_BUILD_TYPE=RelWithDebInfo.",
        "static": "Build a static library.",
        "relocatable": "Create a relocatable ROCm package.",
        "address_sanitizer": "Build with AddressSanitizer.",
        "codecoverage": "Build with code coverage profiling.",
        "gprof": "Enable GNU gprof profiling (requires --static).",
        "no_tensile": "Build without the Tensile GEMM backend.",
        "tensile_logic": "Path for HIPBLASLT_LIBLOGIC_PATH.",
        "tensile_threads": "Parallel build threads for TensileLite (default: nproc).",
        "tensile_verbose": "TensileLite verbosity level.",
        "no_lazy_load": "Disable lazy library loading.",
        "no_msgpack": "Use YAML backend instead of msgpack.",
        "no_compress": "Don't compress TensileLite assembly objects.",
        "keep_build_tmp": "Keep the temporary build artifacts.",
        "experimental": "Include 'Experimental' logic directories.",
        "logic_filter": "Logic YAML filter (e.g. 'gfx942/Equality/*').",
        "legacy_hipblas_direct": "Enable legacy HIPBLAS_DIRECT mode.",
        "disable_marker": "Disable hipBLASLt markers.",
        "enable_tensile_marker": "Enable Tensile markers.",
        "skip_rocroller": "Skip the rocRoller backend.",
        "quiet": "Build without VERBOSE=1.",
        "enable_asm_comments": "Enable assembly comments in generated asm.",
        "build_dir": "Override the build directory.",
        "rocm_path": "Override the ROCm installation path.",
        "clean": "Remove the build directory before configuring (default: incremental).",
    }
)
def build(
    c,
    install_deps=False,
    install_pkg=False,
    clients=False,
    jobs=None,
    architecture="all",
    cpu_ref_lib="blis",
    use_system_packages=False,
    debug=False,
    relwithdebinfo=False,
    static=False,
    relocatable=False,
    address_sanitizer=False,
    codecoverage=False,
    gprof=False,
    no_tensile=False,
    tensile_logic="",
    tensile_threads=None,
    tensile_verbose="",
    no_lazy_load=False,
    no_msgpack=False,
    no_compress=False,
    keep_build_tmp=False,
    experimental=False,
    logic_filter="",
    legacy_hipblas_direct=False,
    disable_marker=False,
    enable_tensile_marker=False,
    skip_rocroller=False,
    quiet=False,
    enable_asm_comments=False,
    build_dir=None,
    rocm_path=None,
    clean=False,
):
    _supported_distros()

    distro = _distro_id()
    version_id = _version_id()
    version_major = int(version_id.split(".")[0]) if version_id else 0

    bld = Path(build_dir).resolve() if build_dir else BUILD_DIR
    if rocm_path:
        rocm = Path(rocm_path)
    elif "ROCM_PATH" in os.environ:
        rocm = Path(os.environ["ROCM_PATH"])
    elif sys.platform == "win32":
        try:
            rocm = Path(subprocess.check_output(["rocm-sdk", "path", "--root"], text=True).strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Could not find ROCm SDK. Run: pip install rocm-sdk && rocm-sdk initDevel")
            sys.exit(1)
    else:
        rocm = Path("/opt/rocm")
    # CMake requires forward slashes even on Windows to avoid backslash escape issues.
    rocm_s = rocm.as_posix()

    if tensile_threads is None:
        tensile_threads = os.cpu_count()
    jobs = jobs or os.cpu_count()

    # Determine build type
    if debug:
        build_type = "Debug"
        build_subdir = bld / "debug"
    elif relwithdebinfo:
        build_type = "RelWithDebInfo"
        build_subdir = bld / "release-debug"
    else:
        build_type = "Release"
        build_subdir = bld / "release"

    # Clean previous build (opt-in; skipping preserves FetchContent/CMake cache)
    if clean and build_subdir.exists():
        _rmtree(build_subdir)

    # Validate options
    if cpu_ref_lib not in ("blis", "lapack"):
        print("--cpu-ref-lib must be 'blis' or 'lapack'")
        sys.exit(2)

    if codecoverage and build_type == "Release":
        print("Code coverage requires Debug or RelWithDebInfo build type.")
        sys.exit(1)

    if gprof and not static:
        print("--gprof requires --static.")
        sys.exit(2)

    # PATH setup — use os.pathsep (';' on Windows, ':' on Linux)
    # lib/llvm/bin is Windows-only: the ROCm Windows SDK stores tools there
    sep = os.pathsep
    rocm_bin_dirs = [f"{rocm_s}/bin", f"{rocm_s}/hip/bin", f"{rocm_s}/llvm/bin"]
    if sys.platform == "win32":
        rocm_bin_dirs.append(f"{rocm_s}/lib/llvm/bin")
    os.environ["PATH"] = sep.join(rocm_bin_dirs) + sep + os.environ.get("PATH", "")

    # ROCm env vars are set after _setup_msvc_env() below, since vcvarsall.bat
    # would overwrite any vars we set here.

    # RocRoller
    use_rocroller = not skip_rocroller and not (distro == "rhel" and version_id == "9.1")

    # ---------------------------------------------------------------------------
    # Dependencies
    # ---------------------------------------------------------------------------
    if install_deps:
        _install_system_deps(
            c, distro, version_major, clients, use_system_packages,
            no_msgpack, use_rocroller, legacy_hipblas_direct, bld,
        )

    # ---------------------------------------------------------------------------
    # cmake options assembly
    # ---------------------------------------------------------------------------
    deps_prefix = bld / "deps" / "install"
    cmake_opts = [
        f'"-DGPU_TARGETS={architecture}"',
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_PREFIX_PATH={deps_prefix.as_posix()}",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DMSGPACK_USE_BOOST=OFF",
    ]

    if legacy_hipblas_direct:
        cmake_opts.append("-DHIPBLASLT_ENABLE_HIPBLAS_DIRECT=ON")
    if address_sanitizer:
        cmake_opts.append("-DHIPBLASLT_ENABLE_ASAN=ON")
    if codecoverage:
        cmake_opts.append("-DHIPBLASLT_ENABLE_COVERAGE=ON")
    if static:
        cmake_opts.append("-DHIPBLASLT_BUILD_SHARED_LIBS=OFF")
    if gprof:
        cmake_opts += ["-DCMAKE_CXX_FLAGS=-pg", "-DCMAKE_C_FLAGS=-pg"]
    if not use_rocroller:
        cmake_opts.append("-DHIPBLASLT_ENABLE_ROCROLLER=OFF")
    if sys.platform == "win32":
        cmake_opts.append("-DHIPBLASLT_ENABLE_OPENMP=OFF")
        # Suppress Windows SDK min/max macros that break std::numeric_limits<T>::max().
        # Pass --rocm-path explicitly: amdclang++ on Windows does not read ROCM_PATH
        # from the environment — it requires the flag on the command line to find HIP headers.
        # Append to avoid clobbering flags set earlier (e.g. -pg from --gprof).
        existing_cxx_flags = next((o.split("=", 1)[1] for o in cmake_opts if o.startswith("-DCMAKE_CXX_FLAGS=")), "")
        cmake_opts = [o for o in cmake_opts if not o.startswith("-DCMAKE_CXX_FLAGS=")]
        new_cxx_flags = f"{(existing_cxx_flags + ' ').lstrip()}-DNOMINMAX --rocm-path={rocm_s}"
        cmake_opts.append(f'"-DCMAKE_CXX_FLAGS={new_cxx_flags}"')

    # Tensile options
    if no_tensile:
        cmake_opts.append("-DHIPBLASLT_ENABLE_DEVICE=OFF")
    else:
        if tensile_logic:
            logic_path = tensile_logic if Path(tensile_logic).is_absolute() else str(ROOT_PATH / tensile_logic)
            cmake_opts.append(f"-DHIPBLASLT_LIBLOGIC_PATH={logic_path}")
        if tensile_threads != os.cpu_count():
            cmake_opts.append(f"-DTENSILELITE_BUILD_PARALLEL_LEVEL={tensile_threads}")

    cmake_opts.append(f"-DHIPBLASLT_ENABLE_YAML={'OFF' if not no_msgpack else 'ON'}")

    if build_type != "Release":
        cmake_opts.append("-DTENSILELITE_ASM_DEBUG=ON")
    if logic_filter:
        cmake_opts.append(f"-DTENSILELITE_LOGIC_FILTER={logic_filter}")
    if keep_build_tmp:
        cmake_opts.append("-DTENSILELITE_KEEP_BUILD_TMP=ON")
    if no_compress:
        cmake_opts.append("-DTENSILELITE_NO_COMPRESS=ON")
    if experimental:
        cmake_opts.append("-DTENSILELITE_EXPERIMENTAL=ON")
    if disable_marker:
        cmake_opts.append("-DHIPBLASLT_ENABLE_MARKER=OFF")
    if not enable_asm_comments:
        cmake_opts.append("-DTENSILELITE_ENABLE_ASM_COMMENTS=OFF")
    if no_lazy_load:
        cmake_opts.append("-DHIPBLASLT_ENABLE_LAZY_LOAD=OFF")

    # Client options
    if not clients:
        client_opts = ["-DHIPBLASLT_ENABLE_CLIENT=OFF"]
    else:
        if cpu_ref_lib == "blis":
            if sys.platform == "win32":
                print("Warning: BLIS is not available on Windows. Disabling BLIS for clients.")
                client_opts = ["-DHIPBLASLT_ENABLE_BLIS=OFF"]
            else:
                client_opts = ["-DHIPBLASLT_ENABLE_BLIS=ON"]
                _install_blis(c, bld)
        else:
            client_opts = ["-DHIPBLASLT_ENABLE_BLIS=OFF"]
        if not use_system_packages and install_deps:
            client_opts += [
                "-DBLAS_LIBRARIES=/usr/local/lib/libblas.a",
                '"-DLAPACK_LIBRARIES=/usr/local/lib/liblapack.a;/usr/local/lib/libblas.a"',
                "-DBLA_STATIC=ON",
            ]

    if sys.platform == "win32":
        # Use absolute paths so CMake cache stays stable regardless of PATH ordering.
        _cxx = shutil.which("amdclang++") or shutil.which("amdclang++.exe")
        _cc  = shutil.which("amdclang")   or shutil.which("amdclang.exe")
        if not _cxx or not _cc:
            print("Could not find amdclang++/amdclang on PATH. Ensure ROCm SDK is installed.")
            sys.exit(1)
        # Lowercase the filename: shutil.which returns the filesystem case (e.g. .EXE),
        # but Tensile's validator compares against lowercase extensions (.exe).
        def _normalized(p: str) -> str:
            pp = Path(p)
            return (pp.parent / pp.name.lower()).as_posix()
        compiler  = _normalized(_cxx)
        ccompiler = _normalized(_cc)
    else:
        compiler = f"{rocm_s}/bin/amdclang++"
        ccompiler = f"{rocm_s}/bin/amdclang"

    # Build subdir
    (build_subdir / "clients").mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        cmake_prefix = subprocess.check_output(["rocm-sdk", "path", "--cmake"], text=True).strip()
        prefix_path = Path(cmake_prefix).as_posix()
    else:
        prefix_path = f"{rocm_s};{rocm_s}/hcc;{rocm_s}/hip"
    module_path = f"{rocm_s}/hip/cmake"

    if relocatable:

        rocm_rpath = os.environ.get("ROCM_RPATH", "/opt/rocm/lib:/opt/rocm/lib64")
        extra = [
            "-DCPACK_SET_DESTDIR=OFF",
            f"-DCMAKE_INSTALL_PREFIX={rocm_s}",
            f"-DCPACK_PACKAGING_INSTALL_PREFIX={rocm_s}",
            f'"-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--enable-new-dtags -Wl,--rpath,{rocm_rpath}"',
            f'"-DCMAKE_PREFIX_PATH={prefix_path}"',
            f"-DCMAKE_MODULE_PATH={module_path}",
            "-DROCM_DISABLE_LDCONFIG=ON",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            f"-DROCM_PATH={rocm_s}",
            f"-DROCM_HOME={rocm_s}",
        ]
    else:
        install_prefix = (ROOT_PATH / "hipblaslt-install").as_posix()
        extra = [
            f'"-DCMAKE_PREFIX_PATH={prefix_path}"',
            f"-DCMAKE_MODULE_PATH={module_path}",
            "-DCPACK_SET_DESTDIR=OFF",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            f"-DCPACK_PACKAGING_INSTALL_PREFIX={rocm_s}",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            f"-DROCM_PATH={rocm_s}",
            f"-DROCM_HOME={rocm_s}",
        ]

    compiler_opts = [
        f"-DCMAKE_CXX_COMPILER={compiler}",
        f"-DCMAKE_C_COMPILER={ccompiler}",
    ]
    if clients:
        compiler_opts.append("-DCMAKE_Fortran_COMPILER=gfortran")
    if sys.platform == "win32":
        _setup_msvc_env()
        # Set ROCm env vars AFTER vcvarsall so it doesn't overwrite them.
        # amdclang++ reads ROCM_PATH for HIP headers and HIP_DEVICE_LIB_PATH for bitcode.
        # The pip-installed SDK stores device libs under lib/llvm/amdgcn/bitcode/ (not amdgcn/bitcode/).
        os.environ["ROCM_PATH"] = rocm_s
        os.environ["ROCM_HOME"] = rocm_s
        # The pip-installed SDK has no separate hip/ subdirectory; HIP headers
        # are at {rocm}/include/hip/. Set HIP_PATH to rocm root so amdclang++
        # finds them via {HIP_PATH}/include/hip/hip_version.h.
        os.environ["HIP_PATH"]  = rocm_s
        os.environ["HIP_DEVICE_LIB_PATH"] = (rocm / "lib" / "llvm" / "amdgcn" / "bitcode").as_posix()
        _setup_clang_shims(build_subdir)
        rc = _find_rc_exe()
        if not rc:
            print("Could not find rc.exe. Ensure the Windows SDK is installed via Visual Studio Build Tools.")
            sys.exit(1)
        compiler_opts.append(f'"-DCMAKE_RC_COMPILER={rc}"')
        # CMake uses ar-style syntax (qc flags) by default. On Windows without
        # llvm-ar, use lib.exe with MSVC-style archive rules via a toolchain file
        # (angle-bracket placeholders can't be passed cleanly as -D flags).
        lib_exe = shutil.which("lib") or shutil.which("lib.exe")
        if lib_exe:
            lib_posix = Path(lib_exe).as_posix()
            toolchain_file = build_subdir / "windows_toolchain.cmake"
            toolchain_file.parent.mkdir(parents=True, exist_ok=True)
            toolchain_file.write_text(
                f'set(CMAKE_AR "{lib_posix}" CACHE FILEPATH "" FORCE)\n'
                'set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <LINK_FLAGS> <OBJECTS>")\n'
                'set(CMAKE_C_CREATE_STATIC_LIBRARY   "<CMAKE_AR> /OUT:<TARGET> <LINK_FLAGS> <OBJECTS>")\n'
            )
            cmake_opts.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file.as_posix()}")
            print(f"lib.exe archiver: {lib_posix}")
        else:
            print("Warning: lib.exe not found. Static library archiving will fail.")
    generator_opts = ['-G "NMake Makefiles"'] if sys.platform == "win32" else []
    all_opts = " ".join(generator_opts + cmake_opts + client_opts + extra + compiler_opts)
    cmake_cmd = f"cmake {all_opts} {ROOT_PATH}"
    print(f"cmake command: {cmake_cmd}")
    with c.cd(str(build_subdir)):
        c.run(cmake_cmd)
        verbose_flag = "" if quiet else " --verbose"
        c.run(f"cmake --build . --target install -j {jobs}{verbose_flag}")

    # ---------------------------------------------------------------------------
    # Package install
    # ---------------------------------------------------------------------------
    if install_pkg:
        with c.cd(str(build_subdir)):
            c.run("make package")
        if distro == "ubuntu":
            _elevate(c, f"dpkg -i {build_subdir}/hipblaslt[-_]*.deb")
        elif distro in ("centos", "rhel", "almalinux"):
            _elevate(c, f"rpm --nodeps -U {build_subdir}/hipblaslt-*.rpm")
        elif distro == "fedora":
            _elevate(c, f"dnf install {build_subdir}/hipblaslt-*.rpm")
        elif distro in ("sles", "opensuse-leap"):
            _elevate(c, f"zypper -n --no-gpg-checks install {build_subdir}/hipblaslt-*.rpm")


# ---------------------------------------------------------------------------
# Dependency installation (split out for clarity)
# ---------------------------------------------------------------------------

def _install_system_deps(
    c, distro, version_major, build_clients, use_system_packages,
    no_msgpack, use_rocroller, legacy_hipblas_direct, bld: Path,
):
    tensile_msgpack_backend = not no_msgpack

    lib_ubuntu = ["make", "pkg-config", "libnuma1", "git", "libmsgpack-dev"]
    lib_centos = ["epel-release", "make", "gcc-c++", "rpm-build"]
    lib_centos8 = ["epel-release", "make", "gcc-c++", "rpm-build", "numactl-libs"]
    lib_fedora = ["make", "gcc-c++", "libcxx-devel", "rpm-build", "numactl-libs"]
    lib_sles = ["make", "gcc-c++", "libcxxtools9", "rpm-build"]
    lib_mariner = ["make", "rpm-build"]

    cli_ubuntu = ["python3", "python3-yaml", "libopenblas-dev"]
    cli_centos = ["python36", "python3-pip"]
    cli_centos8 = ["python39", "python3-virtualenv"]
    cli_fedora = ["python36", "PyYAML", "python3-pip"]
    cli_sles = ["pkg-config", "dpkg", "python3-pip"]
    cli_mariner = ["python3", "python3-yaml"]

    if use_system_packages:
        cli_ubuntu.append("libopenblas-dev")
        cli_centos.append("openblas-devel")
        cli_centos8.append("openblas-devel")
        cli_fedora.append("openblas-devel")
        cli_sles = ["openblas-devel"]
        cli_mariner.append("openblas-devel")
        if tensile_msgpack_backend:
            lib_centos.append("msgpack-devel")
            lib_centos8.append("msgpack-devel")
            lib_fedora.append("msgpack-devel")

    if build_clients:
        lib_ubuntu.append("gfortran")
        lib_centos.append("devtoolset-7-gcc-gfortran")
        lib_centos8.append("gcc-gfortran")
        lib_fedora.append("gcc-gfortran")
        lib_sles += ["gcc-fortran", "pkg-config", "dpkg"]

    if use_rocroller:
        lib_ubuntu += ["rocm-llvm-dev", "libzstd-dev"]
        lib_centos8 += ["rocm-llvm-devel", "zstd"]
        lib_sles += ["rocm-llvm-devel", "zstd"]

    if not legacy_hipblas_direct:
        lib_ubuntu.append("hipblas-common-dev")
        lib_centos8.append("hipblas-common-devel")

    if distro in ("centos", "rhel", "almalinux"):
        lib_centos.append("numactl" if version_major < 7 else "numactl-libs")
        cli_centos8.append("python3-pyyaml" if version_major >= 8 else "PyYAML")

    if distro == "ubuntu":
        _elevate(c, "apt update")
        _apt_install(c, lib_ubuntu)
        if build_clients:
            _apt_install(c, cli_ubuntu)
        c.run("pip3 install wheel")

    elif distro in ("centos", "rhel", "almalinux"):
        if version_major >= 8:
            _yum_install(c, lib_centos8)
            if build_clients:
                _yum_install(c, cli_centos8)
                c.run("pip3 install pyyaml")
        else:
            _yum_install(c, lib_centos)
            if build_clients:
                _yum_install(c, cli_centos)
                c.run("pip3 install pyyaml")

    elif distro == "mariner":
        _yum_install(c, lib_mariner)
        if build_clients:
            _yum_install(c, cli_mariner)
            c.run("pip3 install pyyaml")

    elif distro == "azurelinux":
        _dnf_install(c, lib_mariner)
        if build_clients:
            _dnf_install(c, cli_mariner)
            c.run("pip3 install pyyaml")

    elif distro == "fedora":
        _dnf_install(c, lib_fedora)
        if build_clients:
            _dnf_install(c, cli_fedora)
            c.run("pip3 install pyyaml")

    elif distro in ("sles", "opensuse-leap"):
        _zypper_install(c, lib_sles)
        if build_clients:
            _zypper_install(c, cli_sles)
            c.run("pip3 install pyyaml")

    # msgpack from source for non-Ubuntu RPM distros
    if distro in ("centos", "rhel", "sles", "opensuse-leap", "almalinux"):
        if tensile_msgpack_backend and not use_system_packages:
            _install_msgpack_from_source(
                c, bld,
                cxx=f"{os.environ.get('ROCM_PATH', '/opt/rocm')}/bin/amdclang++",
                cc=f"{os.environ.get('ROCM_PATH', '/opt/rocm')}/bin/amdclang",
            )

    # googletest + optional lapack
    build_lapack = "OFF" if use_system_packages else "ON"
    deps_dir = bld / "deps"
    deps_prefix = deps_dir / "install"
    deps_dir.mkdir(parents=True, exist_ok=True)
    deps_prefix.mkdir(parents=True, exist_ok=True)
    print(f"\033[32mBuilding \033[33mgoogletest\033[32m from source into {deps_prefix}\033[0m")
    with c.cd(str(deps_dir)):
        c.run(
            f"cmake -DCMAKE_INSTALL_PREFIX={deps_prefix.as_posix()}"
            f" -DCMAKE_INSTALL_LIBDIR=lib"
            f" -DBUILD_LAPACK={build_lapack}"
            f" {ROOT_PATH}/deps"
        )
        c.run(f"make -j{os.cpu_count()}")
        c.run("make install")
