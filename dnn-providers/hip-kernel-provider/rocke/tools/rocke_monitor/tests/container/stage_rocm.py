# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""docker/install_rocm.sh against fake installer downloads (tests/container/image.py runs it inside the image, as
root, with no network). The installer's location is the build secret rocm_package_url, which the script reads itself:
rkmon build sets it to rocm.package_url, or to the public installer of the build it found in AMD's folder
(tests/test_build_plan.py tests that part). The installer runs with the workstation's options (the core-dev component,
the HIP headers, among them) and the library targets, the installer's own when they are blank; a version that is not
one, a SHA-256 that is no digest, a blank or malformed list of HIP check targets and a build given no installer's
location stop it before any download; a SHA-256 that does not match installs nothing; a ROCm that reports another
release than rocm.version stops the build; a clang other than the pinned commit stops it; a release whose hipcc cannot
compile <hip/hip_runtime.h> for a HIP check target stops it; the release, its clang, its library targets and its hipcc
are recorded in /etc/rkmon/rocm.json, which names no installer; a failed download says why in fixed words; and no
download address, nor an installer's file name, is ever printed.

  python3 -B stages.py rocm      (as root; it writes /opt/rocm and /etc/rkmon/rocm.json in the throwaway container)
"""

import functools
import hashlib
import http.server
import json
import os
import pathlib
import shlex
import subprocess
import threading

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parent.parent / "docker" / "install_rocm.sh"

FAKE_RUN = """#!/bin/sh
# A fake ROCm runfile installer: records its arguments, lays out one core folder that reports its release, with a clang
# that prints a version and a hipcc that compiles only when the HIP headers are there; they are, with the core-dev
# component, unless this release's installer leaves them out (dev blank).
echo "$*" > {args}
v={version}
core=/opt/rocm/core-$(echo "$v" | cut -d. -f1,2)
mkdir -p "$core/lib/llvm/bin" "$core/bin" "$core/.info"
echo "${{v%%-*}}" > "$core/.info/version"
printf '#!/bin/sh\\necho "Example clang version {version}-clang (fake)"\\necho "Target: x86_64-unknown-linux-gnu"\\n' \\
    > "$core/lib/llvm/bin/clang"
chmod 755 "$core/lib/llvm/bin/clang"
cat > "$core/bin/hipcc" <<'HIPCC'
#!/bin/sh
core=$(dirname "$(dirname "$0")")
if [ ! -f "$core/include/hip/hip_runtime.h" ]; then echo "fatal error: 'hip/hip_runtime.h' file not found" >&2; exit 1; fi
out=""
while [ $# -gt 0 ]; do if [ "$1" = -o ]; then out=$2; fi; shift; done
echo obj > "$out"
HIPCC
chmod 755 "$core/bin/hipcc"
case " $* " in
    *" compo=core,core-dev "*)
        if [ -n "{dev}" ]; then mkdir -p "$core/include/hip"; : > "$core/include/hip/hip_runtime.h"; fi ;;
esac
"""
SECRET_PATH = "private-mirror-9f3c"  # the private download's folder: never printed


class Quiet(http.server.SimpleHTTPRequestHandler):
    """Serves its folder with no log line (the stage's output is its checks), and keeps the path of each request it
    answers in `requested`, so a check can tell that nothing was downloaded."""

    requested: list = []

    def do_GET(self) -> None:
        Quiet.requested.append(self.path)
        super().do_GET()

    def log_message(self, *args, **kwargs) -> None:
        pass


def serve(folder: pathlib.Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    handler = functools.partial(Quiet, directory=str(folder))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def rocm_stage(home: pathlib.Path, port: int) -> int:
    import stages
    from rkmon_lib import container_layout

    st = stages.Stage("rocm")
    if os.geteuid() != 0 or not SCRIPT.is_file():
        st.skip(
            "install_rocm.sh",
            "needs root, in the image (tests/container/image.py runs it so)",
        )
        return st.end()
    root = home / "web"
    public, private = root / "public", root / SECRET_PATH
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    args = home / "fake_rocm_args"  # where the fake installer records its arguments

    def fake(folder: pathlib.Path, v: str, dev: bool = True) -> pathlib.Path:
        f = folder / f"rocm-installer-{v}.run"
        f.write_text(
            FAKE_RUN.format(
                version=v, args=shlex.quote(str(args)), dev="1" if dev else ""
            ),
            encoding="utf-8",
        )
        return f

    # 9.6.0-1's installer leaves the HIP headers out, whatever it is asked for.
    fake(public, "9.10.0-2")
    fake(public, "9.6.0-1", dev=False)
    mine = fake(private, "9.9.0-4")
    good = hashlib.sha256(mine.read_bytes()).hexdigest()
    srv, url = serve(root)
    newest = f"{url}/public/rocm-installer-9.10.0-2.run"  # as rkmon build finds 9.10.0's newest build
    headless = f"{url}/public/rocm-installer-9.6.0-1.run"
    pkg = f"{url}/{SECRET_PATH}/rocm-installer-9.9.0-4.run"  # as rocm.package_url
    outputs = []
    try:

        def install(location: str, *argv):
            """The script with ARGV (VERSION LIBRARY_TARGETS HIP_CHECK_TARGETS CLANG_COMMIT SHA256), the installer's
            LOCATION its build secret rocm_package_url (the build gives none when blank). The fake installer's record
            of its arguments goes first, so that one there is this run's."""
            secrets = home / "secrets"
            secrets.mkdir(exist_ok=True)
            (secrets / "rocm_package_url").write_text(
                f"{location}\n" if location else "", encoding="utf-8"
            )
            if args.exists():
                args.unlink()
            p = subprocess.run(
                ["sh", str(SCRIPT), *argv],
                capture_output=True,
                text=True,
                timeout=300,
                env=dict(os.environ, SECRETS_DIR=str(secrets)),
            )
            outputs.append(p.stdout + p.stderr)
            return p.returncode, p.stdout + p.stderr

        def record() -> dict:
            return container_layout.rocm_record()

        code, text = install(newest, "9.10.0-2", "gfx942", "gfx942,gfx950", "", "")
        rec = record()
        st.check(
            code == 0
            and rec.get("version") == "9.10.0-2"
            and "ROCm reports 9.10.0" in text,
            "the installer at the build secret rocm_package_url (the public installer rkmon build found, or "
            "rocm.package_url) installs the build, which reports its release",
            text[-1200:],
        )
        st.check(
            stages.read(args).strip()
            == "deps=install gfx=gfx942 rocm compo=core,core-dev gpu-access=user assumeyes",
            "the installer runs with the workstation's options and the library targets, the core-dev component (the "
            "HIP headers) among them",
            stages.read(args),
        )
        st.check(
            "/opt/rocm/core-9.10/bin/hipcc compiles <hip/hip_runtime.h> for gfx942 gfx950"
            in text,
            "its hipcc compiles <hip/hip_runtime.h> for each HIP check target",
            text[-800:],
        )
        st.check(
            rec.get("clang") == "Example clang version 9.10.0-2-clang (fake)"
            and rec.get("llvm_bin") == "/opt/rocm/core-9.10/lib/llvm/bin"
            and rec.get("gfx") == "gfx942"
            and rec.get("components") == "core,core-dev"
            and rec.get("hipcc") == "/opt/rocm/core-9.10/bin/hipcc"
            and "installer" not in rec
            and "rocm-installer" not in json.dumps(rec),
            "the release, its LLVM tools, its clang, its library targets, its components and its hipcc are recorded "
            f"in {container_layout.ROCM_RECORD}, which names no installer",
            json.dumps(rec),
        )
        code, text = install(newest, "9.10.0-2", "", "gfx942", "", "")
        st.check(
            code == 0
            and stages.read(args).strip()
            == "deps=install rocm compo=core,core-dev gpu-access=user assumeyes",
            "blank library targets: the installer's own",
            text[-400:] + stages.read(args),
        )
        code, text = install(newest, "9.10.0-2", "gfx942,gfx950", "gfx942", "", "")
        st.check(
            code == 0 and "gfx=gfx942,gfx950 " in stages.read(args),
            "the library targets go to the installer comma-separated",
            text[-400:] + stages.read(args),
        )
        code, text = install(pkg, "9.9.0", "gfx942", "gfx942", "", good)
        rec = record()
        st.check(
            code == 0
            and "matches rocm.package_sha256" in text
            and rec.get("version") == "9.9.0"
            and rec.get("sha256") == good
            and "rocm-installer" not in json.dumps(rec)
            and SECRET_PATH not in json.dumps(rec),
            "a package URL: that exact file, checked against its SHA-256, and recorded without its address or "
            "file name",
            text[-800:],
        )
        code, text = install(pkg + "-missing", "9.9.0", "gfx942", "gfx942", "", "")
        st.check(
            code == 2
            and "the installer did not download (the server answered HTTP 404, curl exit 22)"
            in text,
            "an installer that does not download stops the build, and says why in fixed words",
            text[-400:],
        )
        code, text = install(pkg, "9.9.0", "gfx942", "gfx942", "", "0" * 64)
        st.check(
            code == 3
            and f"SHA-256 is {good}, not rocm.package_sha256" in text
            and not args.exists(),
            "a SHA-256 that does not match installs nothing (exit 3)",
            text[-800:],
        )
        code, text = install(pkg, "9.9.1", "gfx942", "gfx942", "", "")
        st.check(
            code == 3
            and "the installed ROCm reports 9.9.0" in text
            and "not rocm.version's release 9.9.1" in text,
            "a ROCm that reports another release than rocm.version stops the build (exit 3)",
            text[-800:],
        )
        code, text = install(headless, "9.6.0-1", "gfx942", "gfx942", "", "")
        st.check(
            code == 3
            and "cannot compile <hip/hip_runtime.h> for gfx942" in text
            and "'hip/hip_runtime.h' file not found" in text,
            "a release whose hipcc cannot compile <hip/hip_runtime.h> (no HIP headers) stops the build (exit 3)",
            text[-800:],
        )
        code, text = install(
            newest, "9.10.0-2", "gfx942", "gfx942", "9.10.0-2-clang", ""
        )
        st.check(
            code == 0 and "its clang is the pinned compiler" in text,
            "the pinned clang commit, when the installed clang names it, passes",
            text[-800:],
        )
        code, text = install(newest, "9.10.0-2", "gfx942", "gfx942", "d6f6cb69", "")
        st.check(
            code == 4 and "not the pinned compiler (d6f6cb69)" in text,
            "another clang than the pinned one stops the build (exit 4)",
            text[-800:],
        )
        before = len(Quiet.requested)
        refused = [
            (
                ("10.0.0-1; rm -rf /", "gfx942", "gfx942", "", ""),
                "is no release such as 10.1.0",
                "a version that is not one",
            ),
            (
                ("9.9.0", "gfx942", "gfx942", "", "not-a-digest"),
                "ROCM_PACKAGE_SHA256 must be the installer's SHA-256",
                "a SHA-256 that is no digest",
            ),
            (
                ("9.10.0-2", "gfx942", "", "", ""),
                "HIP_CHECK_TARGETS () is no list of GPU targets",
                "a blank list of HIP check targets",
            ),
            (
                ("9.10.0-2", "gfx942", "gfx942;true", "", ""),
                "HIP_CHECK_TARGETS (gfx942;true) is no list of GPU targets",
                "a malformed list of HIP check targets",
            ),
        ]
        for argv, words, what in refused:
            code, text = install(newest, *argv)
            st.check(
                code == 2
                and words in text
                and len(Quiet.requested) == before
                and not args.exists(),
                f"{what} is refused before any download (exit 2)",
                text[-400:],
            )
        code, text = install("", "9.10.0-2", "gfx942", "gfx942", "", "")
        st.check(
            code == 2
            and "the build was given no build secret rocm_package_url" in text
            and len(Quiet.requested) == before
            and not args.exists(),
            "a build given no installer's location stops before any download (exit 2): rkmon build gives one",
            text[-400:],
        )
        host = url.split("//", 1)[1]
        leaks = [
            o
            for o in outputs
            if host in o or SECRET_PATH in o or "rocm-installer-" in o
        ]
        st.check(
            not leaks,
            "no download address, nor an installer's file name, is ever printed (a package URL may name an internal "
            "host)",
            " | ".join(o[-200:] for o in leaks),
        )
    finally:
        srv.shutdown()
    return st.end()


STAGES = {"rocm": rocm_stage}
