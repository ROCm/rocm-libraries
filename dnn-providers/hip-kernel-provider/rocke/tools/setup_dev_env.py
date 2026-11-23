#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build a rocKE development venv that compiles kernels exactly like CI.

CI compiles rocKE kernels with the COMGR -- and so the LLVM -- of the TheRock
revision pinned as ``therock-ref`` in ``.github/actions/ci-env/action.yml``. A
dev box usually compiles with something else: the system ``/opt/rocm`` COMGR,
or torch's bundled COMGR once ``import torch`` has run (rocKE then has to use
torch's copy, because two COMGRs in one process can abort LLVM).

This script makes the venv's single COMGR the CI one:

1. read ``therock-ref`` from this checkout and ask GitHub which llvm-project
   commit that TheRock revision pins (``compiler/amd-llvm``);
2. find the newest ROCm nightly whose ``rocm-sdk-core`` wheel was built from
   that same LLVM commit (its ``share/therock/therock_manifest.json`` says so);
3. install ROCm, ROCm torch and the torch device wheel for this GPU, all from
   that one nightly, into a checkout-independent base venv
   ``<root>/<gfx arch>-llvm-<llvm commit[:8]>/base``; then give this checkout
   its own small venv beside it (``<checkout dir>-<hash>``) that sees the
   base's packages and editable-installs rocKE and its test dependencies, so
   several worktrees share one base without fighting over one editable install;
4. pin rocKE to that venv's COMGR and HIP runtime (``ROCKE_COMGR_LIB`` /
   ``ROCKE_HIP_LIB``) for every process that runs the venv's Python, activated
   or not; torch from the same nightly loads those same files;
5. verify, in the venv, that torch and rocKE load that COMGR and see the GPU.

Usage -- stdout carries only the venv path, progress goes to stderr::

    VENV=$(python3 setup_dev_env.py)                 # Linux (bash)
    . "$VENV/bin/activate"

    $venv = py -3.12 setup_dev_env.py                # Windows (PowerShell)
    & "$venv\\Scripts\\Activate.ps1"

Run it on the machine that has the GPU (inside the Slurm job, not on a login
node): the device wheels are per GPU arch. Re-running is cheap: complete venvs
for the same arch and LLVM are verified and reused, never rebuilt, so a new
nightly with the same LLVM does not churn them. A CI pin with a different LLVM
gets new venvs beside the old ones, so jobs still running on the old ones are
untouched. Concurrent runs (job arrays) serialize on a lock file. If no
nightly was built with CI's LLVM the script fails rather than install a near
match.

Environment: ``ROCKE_VENV_ROOT`` (default ``~/rocke-venvs``) is where venvs
live; a base venv is several GB, so point it at larger shared storage when the
home quota is small. GITHUB_TOKEN (or GH_TOKEN) lifts the anonymous GitHub API
rate limit; the TheRock -> LLVM answer is cached under ``<root>/.cache``.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

NIGHTLY_INDEX = "https://nightly.repo.amd.com/rocm/whl-next/"
PYPI_INDEX = "https://pypi.org/simple"
THEROCK_API = "https://api.github.com/repos/ROCm/TheRock"
VENV_ROOT = Path(os.environ.get("ROCKE_VENV_ROOT") or Path.home() / "rocke-venvs")
# How far back to look for a nightly built with CI's LLVM. TheRock bumps LLVM
# every few days, so a pin older than this has no nightly worth matching.
MAX_NIGHTLIES = 60
MANIFEST = "_rocm_sdk_core/share/therock/therock_manifest.json"
STAMP = "rocke-setup.json"
PTH = "rocke_setup_env.pth"
BASE_PTH = "rocke_setup_base.pth"
HOOK_BEGIN = "# >>> rocke setup_dev_env >>>"
HOOK_END = "# <<< rocke setup_dev_env <<<"
STAMP_SCHEMA = 1

SCRIPT = Path(__file__).resolve()
ROCKE = SCRIPT.parents[1]  # tools -> rocke
PLATFORM_PKG = ROCKE / "platform"
LIBRARY_PKG = ROCKE / "library"
IS_WINDOWS = sys.platform == "win32"


class SetupError(RuntimeError):
    """A failure the user can act on; printed without a traceback."""


def log(message: str) -> None:
    print(f"[setup_dev_env] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# CI pin -> LLVM commit
# --------------------------------------------------------------------------


CI_ENV = ".github/actions/ci-env/action.yml"


def read_ci_env(start: Path) -> tuple[Path, str]:
    """(checkout root, committed ``ci-env/action.yml`` text) of the checkout holding ``start``.

    Read from git HEAD rather than the work tree: sparse checkouts of
    rocm-libraries (e.g. only ``dnn-providers``) do not have ``.github`` on disk.
    """
    try:
        top = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        text = subprocess.run(
            ["git", "-C", top, "show", f"HEAD:{CI_ENV}"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SetupError(
            f"cannot read {CI_ENV} from the git checkout holding {start} ({exc}); "
            "run this script from a rocm-libraries checkout"
        ) from exc
    return Path(top), text


def parse_therock_ref(action_yml: str) -> str:
    """The 40-hex ``therock-ref`` value from ``ci-env/action.yml`` text."""
    match = re.search(
        r"^\s*therock-ref:\s*$.*?^\s*value:\s*[\"']?([0-9a-f]{40})",
        action_yml,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise SetupError(
            "could not find a 40-hex therock-ref value in ci-env/action.yml"
        )
    return match.group(1)


def http_get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    """GET with a few retries on transient network failures."""
    request = urllib.request.Request(url, headers=headers or {})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, response.read()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == 2:
                raise SetupError(f"cannot reach {url}: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def therock_llvm_commit(therock_ref: str) -> str:
    """The llvm-project commit TheRock pins at ``therock_ref`` (``compiler/amd-llvm``)."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{THEROCK_API}/contents/compiler/amd-llvm?ref={therock_ref}"
    try:
        _, body = http_get(url, headers)
    except urllib.error.HTTPError as exc:
        hint = (
            " (GitHub API rate limit: set GITHUB_TOKEN or GH_TOKEN and re-run)"
            if exc.code in (403, 429)
            else ""
        )
        raise SetupError(f"GitHub API {exc.code} for {url}{hint}") from exc
    entry = json.loads(body)
    if entry.get("type") != "submodule" or not re.fullmatch(
        r"[0-9a-f]{40}", str(entry.get("sha", ""))
    ):
        raise SetupError(
            f"TheRock {therock_ref[:8]}: compiler/amd-llvm is not a submodule"
        )
    return entry["sha"]


# --------------------------------------------------------------------------
# Nightly wheel index
# --------------------------------------------------------------------------

_WHEEL = re.compile(
    r"^(?P<dist>[A-Za-z0-9_.]+)-(?P<ver>[^-]+)-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$"
)
_NIGHTLY = re.compile(r"^(\d+)\.(\d+)\.(\d+)a(\d{8})$")
_TORCH = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:a(\d+))?\+rocm(.+)$")


def wheel_platform() -> str:
    if IS_WINDOWS:
        return "win_amd64"
    if sys.platform.startswith("linux") and platform.machine() == "x86_64":
        return "linux_x86_64"
    raise SetupError(f"no ROCm nightly wheels for {sys.platform}/{platform.machine()}")


def python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def index_wheels(project: str) -> list[re.Match[str]]:
    """Parsed wheel filenames listed on the nightly index page of ``project``."""
    try:
        _, body = http_get(f"{NIGHTLY_INDEX}{project}/")
    except urllib.error.HTTPError as exc:
        raise SetupError(f"nightly index has no {project!r} ({exc.code})") from exc
    names = (
        urllib.parse.unquote(href.split("#", 1)[0].rsplit("/", 1)[-1])
        for href in re.findall(r'href="([^"]+)"', body.decode("utf-8", "replace"))
    )
    return [m for m in map(_WHEEL.match, names) if m]


def nightly_key(version: str) -> tuple[int, ...] | None:
    """Sort key of a ``10.3.0a20261118``-style nightly version, else None."""
    m = _NIGHTLY.match(version)
    return tuple(map(int, m.groups())) if m else None


def torch_key(version: str) -> tuple[int, int, int, int, int]:
    """Sort key of a ``2.16.0a0+rocm...`` torch version; a release outranks its alphas."""
    m = _TORCH.match(version)
    if not m:
        return (-1, -1, -1, -1, -1)
    major, minor, patch, alpha, _ = m.groups()
    return (int(major), int(minor), int(patch), 0 if alpha else 1, int(alpha or 0))


def pick_torch(
    nightly: str, torch_versions: set[str], device_versions: set[str]
) -> str | None:
    """Highest torch version built for ``nightly`` that also has the device wheel."""
    built = [
        v for v in torch_versions & device_versions if v.endswith(f"+rocm{nightly}")
    ]
    return max(built, key=torch_key) if built else None


def _range(url: str, first: int, last: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise SetupError(f"{url}: server ignored a byte-range request")
        return response.read()


def _central_directory(url: str) -> Iterator[tuple[str, int, int, int]]:
    """(name, method, compressed size, local header offset) of each file in a
    remote wheel, read via byte ranges (wheels are up to ~2 GB)."""
    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head, timeout=120) as response:
        size = int(response.headers["Content-Length"])
    tail = _range(url, max(0, size - 65536), size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise SetupError(f"{url}: no zip end-of-central-directory record")
    cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12 : eocd + 20])
    locator = tail.rfind(b"PK\x06\x07")
    if locator >= 0:  # zip64: the real offsets live in the zip64 EOCD
        (z64_offset,) = struct.unpack("<Q", tail[locator + 8 : locator + 16])
        z64 = _range(url, z64_offset, z64_offset + 55)
        cd_size, cd_offset = struct.unpack("<QQ", z64[40:56])
    directory = _range(url, cd_offset, cd_offset + cd_size - 1)
    pos = 0
    while directory[pos : pos + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", directory[pos + 10 : pos + 12])[0]
        csize, usize = struct.unpack("<II", directory[pos + 20 : pos + 28])
        name_len, extra_len, comment_len = struct.unpack(
            "<HHH", directory[pos + 28 : pos + 34]
        )
        (offset,) = struct.unpack("<I", directory[pos + 42 : pos + 46])
        name = directory[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
        extra = directory[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        csize, offset = _zip64_sizes(extra, csize, usize, offset)
        yield name, method, csize, offset
        pos += 46 + name_len + extra_len + comment_len


def read_wheel_member(url: str, member: str) -> bytes:
    """One file out of a remote wheel, via byte ranges."""
    for name, method, csize, offset in _central_directory(url):
        if name == member:
            local = _range(url, offset, offset + 29)
            skip = 30 + sum(struct.unpack("<HH", local[26:30]))
            data = _range(url, offset + skip, offset + skip + csize - 1)
            return zlib.decompress(data, -15) if method == 8 else data
    raise SetupError(f"{url} has no {member}")


def longest_member(url: str) -> int:
    """Length of the longest file path in a remote wheel."""
    return max(len(name) for name, *_ in _central_directory(url))


def _zip64_sizes(extra: bytes, csize: int, usize: int, offset: int) -> tuple[int, int]:
    """Apply a central-directory zip64 extra field to (compressed size, offset)."""
    pos = 0
    while pos + 4 <= len(extra):
        tag, length = struct.unpack("<HH", extra[pos : pos + 4])
        if tag == 0x0001:
            values = list(
                struct.unpack(f"<{length // 8}Q", extra[pos + 4 : pos + 4 + length])
            )
            if usize == 0xFFFFFFFF:
                values.pop(0)
            if csize == 0xFFFFFFFF:
                csize = values.pop(0)
            if offset == 0xFFFFFFFF:
                offset = values.pop(0)
        pos += 4 + length
    return csize, offset


def nightly_llvm(version: str, plat: str, cache: dict) -> str:
    """The llvm-project commit the ``version`` nightly was built from."""
    key = f"{plat}/{version}"
    if key not in cache:
        url = (
            f"{NIGHTLY_INDEX}rocm-sdk-core/rocm_sdk_core-{version}-py3-none-{plat}.whl"
        )
        manifest = json.loads(read_wheel_member(url, MANIFEST))
        cache[key] = next(
            s["pin_sha"]
            for s in manifest["submodules"]
            if s["submodule_name"] == "llvm-project"
        )
    return cache[key]


@dataclass(frozen=True)
class Resolution:
    therock_ref: str
    llvm: str
    nightly: str
    torch: str
    arch: str
    platform: str
    python: str


def resolve_nightly(therock_ref: str, llvm: str, arch: str, cache: dict) -> Resolution:
    """Newest nightly built with ``llvm`` that has torch for this arch and Python."""
    plat, py = wheel_platform(), python_tag()
    nightlies = sorted(
        {
            m["ver"]
            for m in index_wheels("rocm-sdk-core")
            if m["plat"] == plat and nightly_key(m["ver"])
        },
        key=nightly_key,
        reverse=True,
    )[:MAX_NIGHTLIES]
    torch_versions = {
        m["ver"] for m in index_wheels("torch") if (m["py"], m["plat"]) == (py, plat)
    }
    device_versions = {
        m["ver"]
        for m in index_wheels(f"amd-torch-device-{arch}")
        if (m["py"], m["plat"]) == (py, plat)
    }
    seen: list[str] = []
    for nightly in nightlies:
        built_with = nightly_llvm(nightly, plat, cache["nightly_llvm"])
        if built_with != llvm:
            seen.append(f"{nightly} (LLVM {built_with[:8]})")
            continue
        torch = pick_torch(nightly, torch_versions, device_versions)
        if torch:
            return Resolution(therock_ref, llvm, nightly, torch, arch, plat, py)
        seen.append(f"{nightly} (LLVM matches, but no torch for {arch}/{py}/{plat})")
    newest = "; ".join(seen[:6]) or "none listed"
    raise SetupError(
        f"no nightly in the newest {len(nightlies)} was built with CI's LLVM "
        f"{llvm[:8]} (TheRock {therock_ref[:8]}) and has torch for {arch}/{py}/{plat}. "
        f"Newest nightlies: {newest}. Refusing to install a different compiler."
    )


# --------------------------------------------------------------------------
# GPU arch
# --------------------------------------------------------------------------


def _arch_from_kfd() -> set[str]:
    """gfx targets of the GPUs in the KFD topology (Linux, no ROCm needed)."""
    archs: set[str] = set()
    for props in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"):
        try:
            text = props.read_text()
        except OSError:  # another job's GPU, not ours
            continue
        m = re.search(r"^gfx_target_version\s+(\d+)$", text, re.MULTILINE)
        if m and int(m.group(1)):
            v = int(m.group(1))  # major*10000 + minor*100 + stepping
            archs.add(f"gfx{v // 10000}{v // 100 % 100:x}{v % 100:x}")
    return archs


@contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """Send fd 1 to stderr: the Windows driver's HIP prints a banner to stdout,
    which must stay reserved for the venv path."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _arch_from_hip() -> set[str]:
    """gfx targets reported by the installed HIP runtime (the driver's, on Windows)."""
    spec = importlib.util.spec_from_file_location(
        "_rocke_hip_device_properties",
        PLATFORM_PKG / "python" / "rocke" / "runtime" / "_hip_device_properties.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = (
        ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"]
        if IS_WINDOWS
        else ["/opt/rocm/lib/libamdhip64.so", "libamdhip64.so"]
    )
    with stdout_to_stderr():
        return _probe_hip(module, names)


def _probe_hip(module, names: list[str]) -> set[str]:
    for name in names:
        try:
            hip = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        return set()
    count = ctypes.c_int(0)
    if hip.hipGetDeviceCount(ctypes.byref(count)) != 0:
        return set()
    get_props = (
        getattr(hip, "hipGetDevicePropertiesR0600", None) or hip.hipGetDeviceProperties
    )
    archs: set[str] = set()
    for device in range(count.value):
        props = module.HipDevicePropR0600()
        if get_props(ctypes.byref(props), device) == 0:
            archs.add(props.gcnArchName.decode("ascii", "replace").split(":", 1)[0])
    return archs


def detect_arch() -> str:
    archs = (_arch_from_kfd() if not IS_WINDOWS else set()) or _arch_from_hip()
    if not archs:
        raise SetupError(
            "no GPU visible; run this on the machine (or inside the Slurm job) that has the GPU"
        )
    if len(archs) > 1:
        raise SetupError(
            f"GPUs of several archs are visible ({', '.join(sorted(archs))}); "
            "request one arch (e.g. a single GPU)"
        )
    return archs.pop()


# --------------------------------------------------------------------------
# Venv
# --------------------------------------------------------------------------


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def run(cmd: list[str | Path]) -> None:
    log("$ " + " ".join(map(str, cmd)))
    rc = subprocess.call([str(c) for c in cmd], stdout=sys.stderr, stderr=sys.stderr)
    if rc:
        raise SetupError(f"command failed (exit {rc}): {cmd[0]} ... {cmd[-1]}")


def venv_query(venv: Path, code: str) -> dict:
    """Run ``code`` in the venv's Python; it must print one JSON object last."""
    proc = subprocess.run(
        [str(venv_python(venv)), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        raise SetupError(f"venv check failed:\n{proc.stderr.strip()[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


_SDK_LIBS = """
import glob, json, os, _rocm_sdk_core, sysconfig
root = os.path.dirname(_rocm_sdk_core.__file__)
pat = {"comgr": ("bin/amd_comgr*.dll", "lib/libamd_comgr.so.*"),
       "hip": ("bin/amdhip64_*.dll", "lib/libamdhip64.so.*")}
win = os.name == "nt"
out = {k: sorted(glob.glob(os.path.join(root, v[0] if win else v[1])), key=len)
       for k, v in pat.items()}
out["site"] = sysconfig.get_paths()["purelib"]
print(json.dumps(out))
"""

_SITE = """
import json, sysconfig
print(json.dumps({"site": sysconfig.get_paths()["purelib"]}))
"""

_VERIFY = """
import json, os, torch
from rocke.runtime import comgr
info = comgr.loaded_compiler_info()
n = torch.cuda.device_count()
print(json.dumps({
    "comgr": comgr.resolved_lib_path(),
    "llvm": ".".join(map(str, info.llvm_version)) if info and info.llvm_version else None,
    "torch": torch.__version__,
    "gpus": [torch.cuda.get_device_properties(i).gcnArchName for i in range(n)],
}))
"""


def require_path_room(venv: Path, res: Resolution) -> None:
    """Without Windows long paths, every file of the wheels must fit in 259
    characters under this venv's site-packages (the deepest are ~200 long)."""
    if not IS_WINDOWS:
        return
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as key:
            if winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1:
                return
    except OSError:
        pass
    torch = res.torch.replace("+", "%2B")
    wheels = [
        f"rocm-sdk-{p}/rocm_sdk_{p.replace('-', '_')}-{res.nightly}-py3-none-{res.platform}.whl"
        for p in ("core", "libraries", "devel", f"device-{res.arch}")
    ] + [
        f"{p}/{p.replace('-', '_')}-{torch}-{res.python}-{res.python}-{res.platform}.whl"
        for p in ("torch", f"amd-torch-device-{res.arch}")
    ]
    deepest = max(longest_member(NIGHTLY_INDEX + w) for w in wheels)
    site = venv / "Lib" / "site-packages"
    if len(str(site)) + 1 + deepest <= 259:
        return
    raise SetupError(
        f"Windows long paths are disabled, and the deepest wheel file ({deepest} "
        f"characters) does not fit under {site}. Either enable long paths from an "
        "Administrator PowerShell (projects/hipdnn/scripts/windows/windows_build_setup.ps1, "
        "or only Set-ItemProperty HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem "
        "LongPathsEnabled 1), or set a shorter ROCKE_VENV_ROOT such as C:\\rv"
    )


def install_base(venv: Path, res: Resolution) -> None:
    """Checkout-independent venv with ROCm + torch from the one nightly ``res`` names."""
    require_path_room(venv, res)
    run([sys.executable, "-m", "venv", venv])
    py = venv_python(venv)
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools>=61", "wheel"])
    # devel is the ROCM_PATH for the Windows hipDNN superbuild; on Linux the
    # container's /opt/rocm already provides it. --no-cache-dir: these wheels
    # are ~10 GB installed, and a second copy in the pip cache does not fit a
    # typical home quota.
    extras = ["libraries", f"device-{res.arch}"] + (["devel"] if IS_WINDOWS else [])
    run(
        [
            py, "-m", "pip", "install", "--no-cache-dir",
            "--index-url", NIGHTLY_INDEX, "--extra-index-url", PYPI_INDEX,
            f"rocm[{','.join(extras)}]=={res.nightly}",
            f"torch=={res.torch}",
            f"amd-torch-device-{res.arch}=={res.torch}",
        ]
    )  # fmt: skip


def make_overlay(venv: Path, base: Path) -> None:
    """Per-checkout venv that sees ``base``'s packages and editable-installs rocKE.

    The ROCm/torch install is several GB and checkout-independent; rocKE is an
    editable install of one checkout. Splitting them lets several worktrees share
    one base without re-pointing each other's editable install.
    """
    run([sys.executable, "-m", "venv", venv])
    base_site = venv_query(base, _SITE)["site"]
    site = Path(venv_query(venv, _SITE)["site"])
    # addsitedir (not a bare path line) so the base's own .pth files run too.
    (site / BASE_PTH).write_text(
        f"import site; site.addsitedir({base_site!r})\n", encoding="utf-8"
    )
    py = venv_python(venv)
    editable = ["-m", "pip", "install", "--config-settings", "editable_mode=compat"]
    run([py, *editable, "-e", f"{PLATFORM_PKG}[dev]"])
    run([py, *editable, "--no-deps", "-e", LIBRARY_PKG])
    # hipDNN superbuild / descriptor packaging: ctest category YAML, the
    # descriptor generator's templates, and rocm_kpack's archive format.
    run(
        [
            py,
            "-m",
            "pip",
            "install",
            "PyYAML>=6.0",
            "Jinja2>=3.1",
            "zstandard>=0.20.0",
            "msgpack",
        ]
    )


def pin_runtime(venv: Path, checkout: Path, therock_ref: str) -> dict[str, str]:
    """Point rocKE at the venv's COMGR/HIP for every process and every activation."""
    sdk = venv_query(venv, _SDK_LIBS)
    if not sdk["comgr"] or not sdk["hip"]:
        raise SetupError(f"rocm-sdk-core in {venv} has no COMGR/HIP runtime library")
    env = {"ROCKE_COMGR_LIB": sdk["comgr"][0], "ROCKE_HIP_LIB": sdk["hip"][0]}
    # A .pth line starting with "import" runs at interpreter start, so the pin
    # holds for VS Code, Slurm and subprocesses that never source activate. An
    # explicit value in the environment still wins (verify() catches a stale one).
    sets = "; ".join(f"os.environ.setdefault({k!r}, {v!r})" for k, v in env.items())
    (Path(sdk["site"]) / PTH).write_text(f"import os; {sets}\n", encoding="utf-8")
    # Activation warns when the checkout's committed CI pin has moved on.
    rerun = f"python3 {SCRIPT}" if not IS_WINDOWS else f"py -3 {SCRIPT}"
    show = f"git -C {shlex.quote(str(checkout))} show HEAD:{CI_ENV}"
    bash = [f"export {k}={shlex.quote(v)}" for k, v in env.items()] + [
        f'if _rocke_pin=$({show} 2>/dev/null) && ! printf %s "$_rocke_pin" | grep -q {therock_ref}; then',
        f"  echo '[rocke] CI pin changed since this venv was built; re-run: {rerun}' >&2",
        "fi",
        "unset _rocke_pin",
    ]
    quoted = str(checkout).replace("'", "''")
    ps = [f"$env:{k} = '{v.replace(chr(39), chr(39) * 2)}'" for k, v in env.items()] + [
        f"$_rockePin = git -C '{quoted}' show HEAD:{CI_ENV} 2>$null",
        f"if ($LASTEXITCODE -eq 0 -and -not ($_rockePin -match '{therock_ref}')) {{",
        f"  Write-Warning '[rocke] CI pin changed since this venv was built; re-run: {rerun}'",
        "}",
        "Remove-Variable _rockePin -ErrorAction SilentlyContinue",
    ]
    scripts = venv / ("Scripts" if IS_WINDOWS else "bin")
    _write_hook(scripts / "activate", bash)
    if (scripts / "Activate.ps1").is_file():
        _write_hook(scripts / "Activate.ps1", ps)
    return env


def _write_hook(script: Path, lines: list[str]) -> None:
    """(Re)place the setup block in an activate script, before any signature block."""
    text = script.read_text(encoding="utf-8")
    text = re.sub(
        rf"\n?{re.escape(HOOK_BEGIN)}.*?{re.escape(HOOK_END)}\n",
        "",
        text,
        flags=re.DOTALL,
    )
    block = "\n".join([HOOK_BEGIN, *lines, HOOK_END]) + "\n"
    signature = text.find("# SIG # Begin signature block")
    if signature >= 0:
        text = text[:signature] + block + text[signature:]
    else:
        text = text.rstrip("\n") + "\n" + block
    script.write_text(text, encoding="utf-8")


def verify(venv: Path, env: dict[str, str], arch: str) -> dict:
    """torch and rocKE load the pinned COMGR and see the GPU, in the venv."""
    found = venv_query(venv, _VERIFY)
    comgr = found["comgr"]
    if not comgr or not os.path.samefile(comgr, env["ROCKE_COMGR_LIB"]):
        raise SetupError(
            f"rocKE resolved COMGR {comgr}, expected {env['ROCKE_COMGR_LIB']}"
        )
    if not found["llvm"]:
        raise SetupError(f"could not read the LLVM version of {comgr}")
    if not any(g.split(":", 1)[0] == arch for g in found["gpus"]):
        raise SetupError(f"torch sees GPUs {found['gpus']}, expected {arch}")
    return found


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Exclusive lock across processes and nodes sharing the venv root."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as fh:
        if IS_WINDOWS:
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    log(f"waiting for {path} ...")
                    time.sleep(5)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                log(f"waiting for {path} (another setup is installing) ...")
                fcntl.lockf(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.lockf(fh, fcntl.LOCK_UN)


def load_cache(path: Path) -> dict:
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    cache.setdefault("therock_llvm", {})
    cache.setdefault("nightly_llvm", {})
    return cache


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def setup() -> Path:
    checkout, ci_env_text = read_ci_env(ROCKE)
    therock_ref = parse_therock_ref(ci_env_text)
    arch = detect_arch()
    cache_path = VENV_ROOT / ".cache" / "resolve.json"
    cache = load_cache(cache_path)
    llvm = cache["therock_llvm"].get(therock_ref)
    if llvm is None:
        llvm = cache["therock_llvm"][therock_ref] = therock_llvm_commit(therock_ref)
        save_cache(cache_path, cache)
    log(f"CI pin: TheRock {therock_ref[:8]} -> LLVM {llvm[:8]}; GPU arch {arch}")

    group = VENV_ROOT / f"{arch}-llvm-{llvm[:8]}"
    base = group / "base"
    tag = hashlib.sha1(str(checkout.resolve()).encode()).hexdigest()[:6]
    venv = group / f"{checkout.name}-{tag}"
    with file_lock(VENV_ROOT / f".{group.name}.lock"):
        stamp = read_stamp(base)
        if stamp is not None:
            log(f"reusing {base} (nightly {stamp['nightly']}, torch {stamp['torch']})")
        else:
            if base.exists():
                log(f"removing incomplete venv {base}")
                shutil.rmtree(base)
            res = resolve_nightly(therock_ref, llvm, arch, cache)
            save_cache(cache_path, cache)
            log(
                f"nightly {res.nightly} was built with LLVM {llvm[:8]}; torch {res.torch}"
            )
            install_base(base, res)
            stamp = {**asdict(res), "schema": STAMP_SCHEMA}
            write_stamp(base, stamp)
        if read_stamp(venv) is None:
            if venv.exists():
                shutil.rmtree(venv)
            make_overlay(venv, base)
        env = pin_runtime(venv, checkout, therock_ref)
        # Stamp before verifying: the install is complete either way, and a
        # verify failure (e.g. no GPU in this job) must not cost a rebuild.
        write_stamp(
            venv, {**stamp, "therock_ref": therock_ref, "checkout": str(checkout)}
        )
        found = verify(venv, env, arch)
    log(
        f"ready: COMGR {found['comgr']} (LLVM {found['llvm']}, commit {llvm[:8]}), "
        f"torch {found['torch']}, GPU {', '.join(found['gpus'])}"
    )
    return venv


def read_stamp(venv: Path) -> dict | None:
    """The completion stamp of ``venv`` if it is complete and usable here, else None."""
    try:
        stamp = json.loads((venv / STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    usable = (
        stamp.get("schema") == STAMP_SCHEMA
        and stamp.get("python") == python_tag()
        and venv_python(venv).is_file()
    )
    return stamp if usable else None


def write_stamp(venv: Path, stamp: dict) -> None:
    (venv / STAMP).write_text(
        json.dumps(stamp, indent=1, sort_keys=True), encoding="utf-8"
    )


def main() -> int:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    try:
        venv = setup()
    except SetupError as exc:
        log(f"FAILED: {exc}")
        return 1
    print(venv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
