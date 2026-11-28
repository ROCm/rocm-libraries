# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container's code, as the package reads it: which code runs (the image's own, or the release root installed over
it in the code volume, rkmon_lib/code_volume.py), the code volume's check, the fix for code that needs a new image, and
`rk code`.

Root alone changes the code volume: it installs a release (docker/rk_root.py code-install, which `rkmon apply` and
`rkmon start` run) and goes back to the code kept (code-back, which `rkmon rollback` runs). No other user of the
container may, so `rk code install` and `rk code back` in the container only refuse, and name the host's commands
(`refuse_change`).

- IMAGE_CODE: the folder of the image's own code, where docker/Dockerfile copies it (RKMON_IMAGE_CODE names another,
  for tests); IMAGE_ID_FILE: the file that holds its digest, which the image's build writes (RKMON_IMAGE_ID_FILE names
  another, for tests).
- `code_dir(environ=os.environ) -> path`: the code volume's folder: RKMON_CODE_DIR, else container_layout.CODE_DIR.
- `image_id() -> str`: the digest of the image's own code; "" when there is none.
- `active_release(code) -> path | None`: the release that runs in the code volume's folder CODE, resolved, while it
  was installed over this image and is whole (code_volume.running); None when the image's own code runs.
- `root_steps() -> module`: docker/rk_root.py, the container's root steps, loaded once by its path from this package
  (one of code_volume.IMAGE_RUNS: a release's copy is its image's): the code volume's check, as the entry point's root
  step makes it at each start (Volume.problems), and its fix (VOLUME_FIX), which the doctor's row says too.
- `new_image_fix(then=()) -> Steps`: the fix for code that needs a new image: an image built from the new package
  (QUICK_BUILD_GUIDE.md), and image.tag set to its tag in your deploy settings file (rkmon_lib/image.py tag_step);
  THEN, the parts of more steps after those (`rkmon package update`'s next line is the state's).
- `host_commands(back, environ=os.environ) -> str`: the host's commands that change the container's code, as the
  operator types them (doctor.wrapper_name): `<wrapper> package update`, then `<wrapper> confirm` and `<wrapper>
  apply`, which install a newer package's code, or with BACK `<wrapper> rollback`, which goes back to the code kept.
  rk's usage error for `rk update` names them too (rkmon.py, ON_THE_HOST).
- `refuse_change(back, out=None, environ=os.environ) -> int`: a change of the code asked for in the container (`rk code
  install`; with BACK `rk code back`): REFUSED, exit 4, nothing changed, since root alone changes the code there, at
  the host's request; its next line names the host's commands (`host_commands`).
- `main(action, out=None) -> int`: `rk code install` and `rk code back` (ACTION "install" or "back"): refuse_change's
  refusal in the container; on a host, which has no code volume, REFUSED, exit 4: rk code works only in the container.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import pathlib
import sys

from . import code_volume, container_layout, outcome, steps
from .checks import Command, Steps

IMAGE_CODE = pathlib.Path(
    os.environ.get("RKMON_IMAGE_CODE") or container_layout.IMAGE_CODE
)
IMAGE_ID_FILE = pathlib.Path(
    os.environ.get("RKMON_IMAGE_ID_FILE") or container_layout.IMAGE_ID_FILE
)


def code_dir(environ=os.environ) -> pathlib.Path:
    """The code volume's folder: see the module's docstring."""
    return pathlib.Path(environ.get("RKMON_CODE_DIR") or container_layout.CODE_DIR)


def image_id() -> str:
    """The digest of the image's own code: see the module's docstring."""
    try:
        return IMAGE_ID_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def active_release(code) -> pathlib.Path | None:
    """The release that runs in the code volume's folder CODE: see the module's docstring."""
    return code_volume.running(code, image_id())


@functools.cache
def root_steps():
    """docker/rk_root.py, the container's root steps: see the module's docstring."""
    path = pathlib.Path(__file__).resolve().parent.parent / "docker" / "rk_root.py"
    spec = importlib.util.spec_from_file_location("rkmon_root_steps", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_image_fix(then=()) -> Steps:
    """The fix for code that needs a new image: see the module's docstring."""
    from . import image

    return Steps(
        (
            "build an image from the new package (QUICK_BUILD_GUIDE.md), then",
            image.tag_step(),
            *then,
        )
    )


def host_commands(back: bool, environ=os.environ) -> str:
    """The host's commands that change the container's code: see the module's docstring."""
    from .doctor import wrapper_name

    w = wrapper_name(environ)
    if back:
        return str(Command((w, "rollback")))
    return (
        f"{Command((w, 'package', 'update'))}, then {Command((w, 'confirm'))} and "
        f"{Command((w, 'apply'))}"
    )


def refuse_change(back: bool, out=None, environ=os.environ) -> int:
    """A change of the code asked for in the container, refused: see the module's docstring."""
    text = "in the container root alone changes the code, at the host's request; nothing changed"
    then = f"{steps.NEXT_LABEL}{host_commands(back, environ)}"
    return outcome.finish("REFUSED", text, next=then, out=out)


def main(action: str, out=None) -> int:
    """`rk code install` and `rk code back`: see the module's docstring."""
    from .doctor import deployment_image

    out = sys.stdout if out is None else out
    if not deployment_image():
        text = "rk code works only in the monitor's container; nothing changed"
        return outcome.finish("REFUSED", text, out=out)
    return refuse_change(action == "back", out=out)
