# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""docker/install_atlassian.sh against a fake download (tests/container/image.py runs it inside the image, with no
network; it needs only sh, curl and sha256sum): the binary is installed only when its SHA-256 matches (in either case
of hex digits), a mismatch names the digest it got and installs nothing, a pinned version is checked, a binary that
prints no version is refused, a checksum that is missing or not one is refused, the URL comes from the build secret
(the script reads it with docker/param.sh, as the build's step does), no URL stops the build, a failed download says
why in fixed words, and neither the URL nor its host is ever printed.

  python3 -B stages.py atlassian   (any user: it installs into a folder of its own, never the image's)
"""

import functools
import hashlib
import http.server
import json
import pathlib
import platform
import subprocess
import threading

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parent.parent / "docker" / "install_atlassian.sh"
FAKE = "#!/bin/sh\necho 'atlassian version 7.8.9'\necho 'A new version (8.0.0) is available.' >&2\n"
NO_VERSION = "#!/bin/sh\necho 'usage: tool COMMAND'\n"  # another program, which prints no version
SECRET_PATH = "private-store/cli/linux-amd64"


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs) -> None:  # the stage's output is its checks
        pass


def atlassian_stage(home: pathlib.Path, port: int) -> int:
    import stages

    st = stages.Stage("atlassian")
    if not SCRIPT.is_file():
        st.skip("install_atlassian.sh", "the package's docker/ folder is not here")
        return st.end()
    folder = home / "store" / SECRET_PATH
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "atlassian").write_text(FAKE, encoding="utf-8")
    (folder / "other").write_text(NO_VERSION, encoding="utf-8")
    good = hashlib.sha256(FAKE.encode("utf-8")).hexdigest()
    handler = functools.partial(Quiet, directory=str(home / "store"))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/{SECRET_PATH}/atlassian"
    record = home / "atlassian.json"
    try:

        def install(dest: str, secret: str, *args) -> tuple[int, str]:
            """The script, its URL the build secret SECRET (none when blank), installing into DEST."""
            secrets = home / "secrets" / dest
            secrets.mkdir(parents=True, exist_ok=True)
            if secret:
                (secrets / "atlassian_cli_package_url").write_text(
                    secret + "\n", encoding="utf-8"
                )
            p = subprocess.run(
                ["sh", str(SCRIPT), *args, str(home / dest)],
                capture_output=True,
                text=True,
                timeout=120,
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "RKMON_ATLASSIAN_RECORD": str(record),
                    "SECRETS_DIR": str(secrets),
                },
            )
            return p.returncode, p.stdout + p.stderr

        code, text = install("ok", url, good, "")
        rec = json.loads(stages.read(record) or "{}")
        ran = subprocess.run(
            [str(home / "ok" / "atlassian"), "version"], capture_output=True, text=True
        )
        st.check(
            code == 0
            and ran.stdout.strip() == "atlassian version 7.8.9"
            and rec == {"version": "7.8.9", "sha256": good}
            and good in text,
            "a matching binary, from the build secret's URL, is installed, and its version and SHA-256 recorded",
            text[-800:] + json.dumps(rec),
        )
        code, text = install("upper", url, good.upper(), "7.8.9")
        st.check(
            code == 0 and (home / "upper" / "atlassian").is_file(),
            "the checksum in capitals, and a pinned version that matches",
            text[-600:],
        )
        wrong = "0" * 64
        code, text = install("wrong", url, wrong, "")
        st.check(
            code == 3
            and f"SHA-256 is {good}, not atlassian_cli.package_sha256 ({wrong}): nothing installed"
            in text
            and not (home / "wrong" / "atlassian").exists(),
            "a binary that does not match installs nothing, and the build names the digest it got",
            text[-600:],
        )
        code, text = install("pin", url, good, "9.9.9")
        st.check(
            code == 3
            and "reports version 7.8.9, not atlassian_cli.version (9.9.9): nothing installed. Set "
            "atlassian_cli.package_url and atlassian_cli.package_sha256 to version 9.9.9's binary, or "
            "atlassian_cli.version to 7.8.9" in text
            and not (home / "pin" / "atlassian").exists(),
            "a pinned version that differs installs nothing, and the build names the settings to change",
            text[-600:],
        )
        before = stages.read(record)
        other = hashlib.sha256(NO_VERSION.encode("utf-8")).hexdigest()
        code, text = install("noversion", url[: -len("atlassian")] + "other", other, "")
        machine = platform.machine()
        st.check(
            code == 3
            and "the binary prints no version when run as `atlassian version`, so it is not the atlassian tool, "
            f"or not one for this image's architecture ({machine}): nothing installed. Set "
            f"atlassian_cli.package_url to the atlassian CLI's Linux binary for {machine}, and "
            "atlassian_cli.package_sha256 to its SHA-256" in text
            and not (home / "noversion" / "atlassian").exists()
            and stages.read(record) == before,
            "a binary that prints no version installs nothing, though no version is pinned, and the build names "
            "the settings to change",
            text[-600:],
        )
        for name, sha in (("none", ""), ("short", good[:40]), ("nothex", "z" * 64)):
            code, text = install(name, url, sha, "")
            st.check(
                code == 2
                and "64 hex digits" in text
                and not (home / name / "atlassian").exists(),
                f"a URL with a checksum that is {'missing' if not sha else 'not one'} is refused ({name})",
                text[-400:],
            )
        code, text = install("gone", url + "-missing", good, "")
        st.check(
            code == 4
            and "the download from atlassian_cli.package_url failed (the server answered HTTP 404, curl exit 22): "
            "nothing installed" in text
            and not (home / "gone" / "atlassian").exists(),
            "a failed download installs nothing, and says why in fixed words",
            text[-400:],
        )
        code, text = install("refused", "http://127.0.0.1:1/atlassian", good, "")
        st.check(
            code == 4
            and "(no connection to the host: it refused, or cannot be reached, curl exit 7)"
            in text,
            "a host that refuses the connection: the same, in its own words",
            text[-400:],
        )
        code, text = install("blank", "", "", "")
        st.check(
            code == 5
            and "ATLASSIAN_CLI_PACKAGE_URL is blank" in text
            and "set atlassian_cli.package_url in your build settings" in text
            and not (home / "blank" / "atlassian").exists(),
            "no URL: the build stops (every image carries the CLI), naming the parameter and its settings key",
            text[-400:],
        )
        outputs = [
            install("again", url, wrong, "")[1],
            install("again2", url + "-missing", good, "")[1],
            install("again3", "http://127.0.0.1:1/atlassian", good, "")[1],
            install("again4", url, good, "")[1],
        ]
        st.check(
            not any(SECRET_PATH in o or "127.0.0.1" in o for o in outputs),
            "neither the URL nor its host is ever printed",
            " | ".join(outputs)[-600:],
        )
    finally:
        srv.shutdown()
    return st.end()


STAGES = {"atlassian": atlassian_stage}
