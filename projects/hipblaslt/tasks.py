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
    info = {}
    with open("/etc/os-release") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, _, v = line.partition("=")
                info[k] = v.strip('"')
    return info


def _distro_id() -> str:
    return _os_release().get("ID", "")


def _version_id() -> str:
    return _os_release().get("VERSION_ID", "")


def _supported_distros():
    supported = {"ubuntu", "centos", "almalinux", "rhel", "fedora", "sles",
                 "opensuse-leap", "mariner", "azurelinux"}
    distro = _distro_id()
    if distro not in supported:
        print(f"Unsupported distro '{distro}'. Supported: {', '.join(sorted(supported))}")
        sys.exit(2)


def _elevate(c, cmd: str):
    if os.getuid() == 0:
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
    }
)
def build(
    c,
    install_deps=False,
    install_pkg=False,
    clients=False,
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
):
    _supported_distros()

    distro = _distro_id()
    version_id = _version_id()
    version_major = int(version_id.split(".")[0]) if version_id else 0

    bld = Path(build_dir).resolve() if build_dir else BUILD_DIR
    rocm = Path(rocm_path) if rocm_path else Path(os.environ.get("ROCM_PATH", "/opt/rocm"))

    if tensile_threads is None:
        tensile_threads = os.cpu_count()

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

    # Clean previous build
    if build_subdir.exists():
        shutil.rmtree(build_subdir)

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

    # PATH setup
    env_path = f"{rocm}/bin:{rocm}/hip/bin:{rocm}/llvm/bin:{os.environ.get('PATH', '')}"
    os.environ["PATH"] = env_path

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
    cmake_opts = [
        f"-DGPU_TARGETS={architecture}",
        "-DHIPBLASLT_ENABLE_FETCH=ON",
        f"-DCMAKE_BUILD_TYPE={build_type}",
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

    compiler = str(rocm / "bin" / "amdclang++")
    ccompiler = str(rocm / "bin" / "amdclang")

    # Build subdir
    (build_subdir / "clients").mkdir(parents=True, exist_ok=True)

    prefix_path = f"{rocm};{rocm}/hcc;{rocm}/hip"
    module_path = f"{rocm}/hip/cmake"

    if relocatable:
        rocm_rpath = os.environ.get("ROCM_RPATH", "/opt/rocm/lib:/opt/rocm/lib64")
        extra = [
            "-DCPACK_SET_DESTDIR=OFF",
            f"-DCMAKE_INSTALL_PREFIX={rocm}",
            f"-DCPACK_PACKAGING_INSTALL_PREFIX={rocm}",
            f'"-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--enable-new-dtags -Wl,--rpath,{rocm_rpath}"',
            f'"-DCMAKE_PREFIX_PATH={prefix_path}"',
            f"-DCMAKE_MODULE_PATH={module_path}",
            "-DROCM_DISABLE_LDCONFIG=ON",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            f"-DROCM_PATH={rocm}",
        ]
    else:
        install_prefix = ROOT_PATH / "hipblaslt-install"
        extra = [
            f'"-DCMAKE_PREFIX_PATH={prefix_path}"',
            f"-DCMAKE_MODULE_PATH={module_path}",
            "-DCPACK_SET_DESTDIR=OFF",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            f"-DCPACK_PACKAGING_INSTALL_PREFIX={rocm}",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            f"-DROCM_PATH={rocm}",
        ]

    all_opts = " ".join(cmake_opts + client_opts + extra)
    cmake_cmd = (
        f"FC=gfortran CXX={compiler} CC={ccompiler} "
        f"cmake {all_opts} {ROOT_PATH}"
    )
    print(f"cmake command: {cmake_cmd}")
    with c.cd(str(build_subdir)):
        c.run(cmake_cmd)

        verbose = "" if quiet else " VERBOSE=1"
        c.run(f"make -j{os.cpu_count()} install{verbose}")

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
    build_lapack = "OFF" if (use_system_packages or True) else "ON"  # blis path sets OFF
    print("\033[32mBuilding \033[33mgoogletest\033[32m from source into /usr/local\033[0m")
    deps_dir = bld / "deps"
    deps_dir.mkdir(parents=True, exist_ok=True)
    with c.cd(str(deps_dir)):
        c.run(f"cmake -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_LAPACK={build_lapack} {ROOT_PATH}/deps")
        c.run(f"make -j{os.cpu_count()}")
        _elevate(c, "make install")
