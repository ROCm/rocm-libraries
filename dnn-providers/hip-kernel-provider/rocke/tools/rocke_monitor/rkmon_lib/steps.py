# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The two numbered paths, build and deploy, and the lines that start and end a step.

- `Step(n, name, command, what, path, others=(), skip="", check="")`: step n of its path ("build" or "deploy"): its
  name, its command (the words after the program's name, as `rkmon help` shows them), what it does (its banner's
  WHAT), the other commands that do the same step (`or:` in `rkmon help`), when the step needs no command, in words
  ("" for never; `rkmon help` puts it after the others), and a command that checks the step's work and changes
  nothing ("" for none), which no next line names (`rkmon help` shows it as `(PROGRAM CHECK checks it)`).
- BUILD_STEPS (5) and DEPLOY_STEPS (8), in order; PATHS names them by path. `after(step)`: the path's next step, or
  None after its last.
- `aside(step, shown=None) -> (str, str)`: the two lines `rkmon help` and the README's table put after a step's
  command, each "" for none: its other commands, each after `or:`, then when it needs none (`or: rkmon image pull;
  nothing when the image is here`); the command that checks its work (`(rkmon image check checks it)`). SHOWN gives a
  command, the words after the program's name, as the text shows it: by default after the name rkmon was run by.
- IMAGE_TEST_STAGES: the stages of build step 4's test of the image (tests/container/image.py), in the order it runs
  them, each with what it checks: the test runs them by these names, and `rkmon image test --help` lists them.
- `banner(step, dry_run=False)`: `== build step N of 5, NAME: WHAT`, or `deploy step N of 8`; under --dry-run it ends
  with "(dry run: nothing changes)". Each step's banner fits on a line of checks.WIDTH columns, with that ending too.
  `command_banner(command, what, dry_run=False)`: `== COMMAND: WHAT`, for a command that is not a step.
- `typed(command, **words)`: a command's words as you would type them: each optional part ([...]) left out, unless a
  word is given for each placeholder in it (PROFILE="review-only" puts in `--profile review-only`), and each
  placeholder word given (FILE="~/rkmon-deploy.yaml") put in its place, one word whatever it holds; the others stay
  (IMAGE_FILE). `command_line(command, **words)`: that, after the name rkmon was run by, as a checks.Command: a fix,
  or a line that names a command. `command_name(command)`: its words before its first option, optional part or
  placeholder: "confirm" of "confirm --settings FILE", "image load" of "image load IMAGE_FILE".
- NEXT_LABEL and DONE_LABEL: `next: ` and `done: `, the words a next line starts with and those of the line a path
  ends with. Every next line and done line starts with one of them, and outcome.finish takes no other.
- `step_words(step)`: `step N, NAME:`, the words before a step's command in a fix or a next line.
- `step_fix(step, command=None, plain=False, **words)`: `step N, NAME: PROGRAM COMMAND` as a fix (checks.Steps), with
  the step's command, or the one of its others given, as command_line gives it: each word quoted as the shell needs, a
  leading ~/ outside the quotes (checks.Command); PROGRAM is the name rkmon was run by (rkmon_lib/program.py). PLAIN:
  `PROGRAM COMMAND` alone, as each deploy step's is once deploy step 8 has passed after a start
  (states.step_8_passed). A command the step does not have is a ValueError. `step_command(step, command=None,
  plain=False, **words)`: that, as text; `next_line(step, command=None, plain=False, **words)`: NEXT_LABEL and that.
- `done_line(text)`: `done: TEXT` (DONE_LABEL), the line a path ends with.
- `restart_parts(program_name="")`: the steps that run the container again, as a fix's parts (checks.Steps):
  `PROGRAM stop --drain` (the work in flight finishes, then the monitor and its container stop), then `PROGRAM start`,
  PROGRAM being PROGRAM_NAME, else the name rkmon was run by. The start's rk provision puts back a piece of
  the set-up that is missing, and probes the local review loop's server again after a failed probe, so a key set again
  or a server put right is checked; the container's Claude session reads its key (the claude login) as it starts, and rk
  orchestrate writes its settings file then.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import program
from .checks import Command, Steps

DRY_RUN = "(dry run: nothing changes)"
NEXT_LABEL = "next: "  # the words a next line starts with
DONE_LABEL = "done: "  # the words a done line starts with: the line a path ends with
OPTIONAL = re.compile(r"\s*\[[^\]]*\]")
# a word of a command's name, not an option or a placeholder
NAME_WORD = re.compile(r"[a-z][a-z-]*")
# a placeholder word: FILE, KEY, IMAGE_FILE
PLACEHOLDER = re.compile(r"[A-Z][A-Z0-9_]*")


@dataclass(frozen=True)
class Step:
    """One step of a path."""

    n: int
    name: str
    command: str
    what: str
    path: str
    others: tuple = ()
    skip: str = ""
    check: str = ""


BUILD_STEPS = (
    Step(
        1,
        "template",
        "template build FILE",
        "write a build settings file for you to fill in",
        "build",
    ),
    Step(
        2,
        "check",
        "check --settings FILE",
        "check this machine and your build settings; change nothing",
        "build",
    ),
    Step(
        3,
        "build",
        "build --settings FILE",
        "build the image from your build settings",
        "build",
    ),
    Step(
        4,
        "test",
        "image test --settings FILE",
        "test the image on this machine",
        "build",
    ),
    Step(
        5,
        "ship",
        "image save IMAGE_FILE --settings FILE",
        "save the image to a file, or push it to your registry",
        "build",
        ("image push --settings FILE",),
    ),
)
DEPLOY_STEPS = (
    Step(
        1,
        "template",
        "template deploy [--profile PROFILE] FILE",
        "write a deploy settings file for you to fill in",
        "deploy",
    ),
    Step(
        2,
        "check",
        "check --settings FILE",
        "check this host and your deploy settings; change nothing",
        "deploy",
    ),
    Step(
        3,
        "confirm",
        "confirm --settings FILE",
        "check your deploy settings, show them, and confirm them",
        "deploy",
    ),
    Step(
        4,
        "image",
        "image load IMAGE_FILE",
        "make sure the image is on this host and suits this package",
        "deploy",
        ("image pull",),
        skip="nothing when the image is here",
        check="image check",
    ),
    Step(5, "login", "login", "set each login and secret your settings need", "deploy"),
    Step(
        6,
        "start",
        "start",
        "start the container with your confirmed settings",
        "deploy",
    ),
    Step(
        7,
        "doctor",
        "doctor",
        "check the running monitor and everything it reaches",
        "deploy",
    ),
    Step(8, "first tick", "first-tick", "wait for the monitor's first tick", "deploy"),
)
PATHS = {"build": BUILD_STEPS, "deploy": DEPLOY_STEPS}
# The image's test, build step 4 (tests/container/image.py, which `image test` runs on the image): its stages, in the
# order it runs them, and what each checks, in one clause with no parentheses or semicolons, since `image test --help`
# lists them as NAME (WHAT); NAME (WHAT). `image test --help` also says the network each stage runs on.
IMAGE_TEST_STAGES = {
    "labels": "the build's labels and build record are there and agree, with no address or download location",
    "tools": "Claude Code, its SDK, gh, the atlassian CLI, git, cmake, ninja and the checks venv run",
    "units": "the package's unit tests pass inside the image, as rk selftest runs them",
    "deploy": (
        "a deployment from a settings file runs: the dashboard, a tick, the Claude session, rk doctor and the stop "
        "signal"
    ),
    "hip": "the image's hipcc compiles for each of the review's GPU targets, as rk doctor checks",
    "rocm": "docker/install_rocm.sh installs from a stand-in installers' folder and download",
    "atlassian": "docker/install_atlassian.sh installs from a stand-in download",
    "isolation": "the session cannot reach the write login, and a PR runner reads no secret and writes only its slot",
    "canary": "no PR runner finds a private value's canary, and the monitor's user cannot read the writer's volume",
    "ids": "other user ids than the image's: the start takes them on, the volumes follow, and the isolation stands",
    "start": "a container started as compose starts it ends at once on docker stop",
    "basics": (
        "a deployment with no login made ticks twice in prototype, reading GitHub only, and rk doctor fails on the "
        "logins alone"
    ),
    "misconfigured": (
        "a deployment without the alerts' webhook shows them incomplete, the tick skips them and rk teams-test "
        "refuses"
    ),
    "reviewer": (
        "each start checks the reviewer model from the container, with its key: an Ollama, a gateway and a gateway "
        "route pass, and a model whose key the gateway refuses, or that cannot call tools, is refused with the "
        "reason, which rk doctor fails on"
    ),
    "parallel-reviews": (
        "three reviews draft at once on a remote gateway, each to the end, within the shared rate budget, and a 429 "
        "makes every review wait"
    ),
    "parallel-reviews-local": "a local Ollama drafts one review at a time until its binding allows more",
    "stop": (
        "rk pause lets the drafts in flight run on, rk stop and rk kill end every process the monitor started and no "
        "other, rk stop --drain lets the drafts finish first, and rk start resumes"
    ),
    "deploy-path": (
        "the deploy path's eight steps pass as on a fresh machine, in prototype, against stand-ins for GitHub and "
        "the Claude gateway"
    ),
    "no-restart": (
        "each setting that takes effect with no restart, changed on the deploy path's running deployment, is read at "
        "its next use, with nothing restarted"
    ),
    "web-host": (
        "web-host install puts the request service on a stand-in web host over SSH as the confirmed settings say, "
        "refusing in standby and past a host key not recorded, and doctor's row of the service passes, then fails "
        "once it stops"
    ),
    "one-time-move": (
        "the first start on an older install's volumes splits its config.json by owner and copies its secrets to their "
        "own places, each with its owner and permissions, the monitor in standby, config.json and the older copies "
        "unchanged until standby off renames the one and removes the others"
    ),
    "adopt": (
        "adopt from a copy of an earlier monitor's work volume, before the first start and again after a stop: the "
        "start after each makes the one-time move, the monitor in standby until standby off ends it"
    ),
    "team-prototype": (
        "the team dashboard in prototype, served by the monitor with serve.address every address, has docker publish "
        "its port on this host's loopback alone, as docker/.env gives it and check 7 says at check and at confirm, and "
        "its page answers in the container"
    ),
    "several-deployments": (
        "two deployments of one repository run side by side under one account, no command of one reads another, and a "
        "start refuses another deployment's container name, another deployment's volume and a busy port"
    ),
    "first-apply": (
        "on a deployment that an image of the package before this one made, moved in place, a first apply whose load "
        "fails stops the new container and keeps the earlier monitor's files, one that passes runs in standby, and the "
        "way back from each runs the earlier monitor again"
    ),
}


def _path(step: Step) -> tuple:
    try:
        return PATHS[step.path]
    except KeyError:
        raise ValueError(
            f"no path {step.path!r}: a step is a build or a deploy step"
        ) from None


def after(step: Step):
    """The step after this one on its path, or None after the last."""
    path = _path(step)
    return path[step.n] if step.n < len(path) else None


def aside(step: Step, shown=None) -> tuple:
    """The two lines after a step's command, each "" for none: see the module's docstring."""
    shown = shown or (lambda command: f"{program.name()} {command}")
    others = ", ".join(f"or: {shown(c)}" for c in step.others)
    return (
        "; ".join(x for x in (others, step.skip) if x),
        f"({shown(step.check)} checks it)" if step.check else "",
    )


def _dry(text: str, dry_run: bool) -> str:
    return f"{text} {DRY_RUN}" if dry_run else text


def banner(step: Step, dry_run: bool = False) -> str:
    """The line a step's command starts with."""
    path = _path(step)
    return _dry(
        f"== {step.path} step {step.n} of {len(path)}, {step.name}: {step.what}",
        dry_run,
    )


def command_banner(command: str, what: str, dry_run: bool = False) -> str:
    """The line any other command starts with: `== COMMAND: WHAT`."""
    return _dry(f"== {command}: {what}", dry_run)


def typed(command: str, **words) -> tuple:
    """A command's words as you would type them: each optional part left out, unless a word is given for each
    placeholder in it, and each given placeholder word put in its place."""

    def optional(part: re.Match) -> str:
        inner = part.group(0).strip()[1:-1]
        names = [w for w in inner.split() if PLACEHOLDER.fullmatch(w)]
        return " " + inner if names and all(n in words for n in names) else ""

    return tuple(str(words.get(w, w)) for w in OPTIONAL.sub(optional, command).split())


def command_line(command: str, **words) -> Command:
    """The command, typed, after the name rkmon was run by."""
    return Command((program.name(), *typed(command, **words)))


def command_name(command: str) -> str:
    """A command's name: its words before its first option, optional part or placeholder."""
    words = command.split()
    end = next(
        (i for i, w in enumerate(words) if not NAME_WORD.fullmatch(w)), len(words)
    )
    return " ".join(words[:end])


def step_words(step: Step) -> str:
    """`step N, NAME:`, the words before a step's command."""
    return f"step {step.n}, {step.name}:"


def step_fix(
    step: Step, command: str | None = None, *, plain: bool = False, **words
) -> Steps:
    """`step N, NAME: PROGRAM COMMAND` as a fix, with the step's command or the one of its others given; PLAIN, the
    command alone."""
    command = step.command if command is None else command
    if command not in (step.command, *step.others):
        raise ValueError(f"{step.path} step {step.n} has no command {command!r}")
    typed = command_line(command, **words)
    return Steps((typed,) if plain else (step_words(step), typed))


def step_command(
    step: Step, command: str | None = None, *, plain: bool = False, **words
) -> str:
    """`step N, NAME: PROGRAM COMMAND`, with the step's command or the one of its others given (step_fix)."""
    return str(step_fix(step, command, plain=plain, **words))


def next_line(
    step: Step, command: str | None = None, *, plain: bool = False, **words
) -> str:
    """`next: step N, NAME: PROGRAM COMMAND`: the step's command, as step_command gives it."""
    return NEXT_LABEL + step_command(step, command, plain=plain, **words)


def done_line(text: str) -> str:
    """`done: TEXT`: the line a path ends with."""
    if not str(text).strip():
        raise ValueError("a done line says what is done")
    return f"{DONE_LABEL}{text}"


def restart_parts(program_name: str = "") -> tuple:
    """The steps that run the container again, as a fix's parts: the stop, then the start, whose rk provision repairs
    the set-up and probes a refused model's server again, and whose Claude session reads its key again (the module's
    docstring)."""
    w = program_name or program.name()
    return (Command((w, "stop", "--drain")), ", then", Command((w, "start")))
