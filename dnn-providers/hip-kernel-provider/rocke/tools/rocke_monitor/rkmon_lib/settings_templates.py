# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The settings templates the package gives you. Each one's text is settings_file.template's,
generated from its file type's schema (rkmon_lib/file_types.py), so docker/rkmon, which prints them, and the privacy
check, which reads every one of them, read the same text.

- `Template(file_type, profile=None, every_key=False)`: one template: the build file's (file_types.BUILD), the deploy
  file's for one of its profiles (file_types.DEPLOY; PROFILE, a name in its schema's x-profiles), or the deploy file's
  every-key text (EVERY_KEY: the text the section command prints a section of). `label` names it in the privacy
  check's report: "build", "deploy NAME" or "deploy every key". `text()` is its text, and `command()` the command that
  prints it: `PROGRAM template build -`, `PROGRAM template deploy --profile NAME -`, or for the every-key text the
  section command, `PROGRAM template section`.
- `templates() -> tuple`: every template the package gives you: the build file's, the deploy file's for each profile,
  in the schema's order, then the deploy file's every-key text. The schemas are read when it is called, never on
  import.
- `section_text(key=None) -> str`: what `rkmon template section [KEY]` prints: the deploy file's section KEY in
  full, as the every-key text writes it (settings_file.section); with no KEY, each section's KEY and title
  (settings_file.sections), one a line, the titles two columns after the longest KEY. An unknown KEY is a ValueError
  that names the closest section, or, when no section is close to it, the command that lists them.
- `section_command(key=None, out=None, err=None) -> int`: `rkmon template section [KEY]` (its row: commands.TABLE),
  once its arguments are parsed: `section_text(KEY)` on OUT (stdout), alone, with no banner and no outcome line, and
  exit 0. An unknown KEY is a usage error on ERR (stderr), and exit 2: `PROGRAM template section: no section KEY; did
  you mean CLOSEST?`, or, when no section is close to it, `PROGRAM template section: no section KEY; PROGRAM template
  section lists the sections`.
- `write(template, file, argv, package_dir=PACKAGE, out=None, *, next_line=None, dry_run=False, plain=False) ->
  int`: `rkmon template build FILE` and `rkmon template deploy [--profile PROFILE] FILE` once their arguments are
  parsed: TEMPLATE's text (a build template or a profile's) as FILE, a new file, yours only, through
  private_files.write. ARGV is the command line as you typed it, its program first: a refusal's fix is that command
  naming another file (private_files.again). It prints, on OUT (stdout), the WROTE line (`WROTE FILE (profile
  PROFILE; yours only, permissions 600): fill in the N values marked <...>`, N being the FAIL rows that `rkmon check`
  gives the new file, one for each <...>), then the next step's line (PLAIN: its command alone, as a deploy step's is
  once deploy step 8 has passed after a start, which the caller knows: steps.next_line), and exits 0; or a refusal's
  rows (checks.render), a blank line, `REFUSED: nothing written` and one next line, and exits 4. A refusal's next line
  is, on the deploy side, the deployment's state's: NEXT_LINE, which the caller gives (states.next_line), since
  only the command layer gathers the state's facts; on the build side, ARGV again, your command as you typed it.
  For a FILE, NEXT_LINE given on the build side is a TypeError, and so is NEXT_LINE missing on the
  deploy side. With DRY_RUN (`--dry-run`) every guard runs and nothing is written: a refusal as above, or
  `PASSED: FILE can be written (profile PROFILE; yours only, permissions 600); nothing written` and exit 0, then on
  the deploy side NEXT_LINE, the state's again, and on the build side the step to run for real, `next: step 1,
  template: PROGRAM template build FILE`. `-` as FILE prints the text alone, with no outcome line. The step's banner,
  before a FILE's lines, is the caller's. A refusal's rows show FILE, and a path beside it, as the WROTE line does
  (settings_file.shown: `~/…` in your home folder).

docker/rkmon runs this on the host: the standard library only, on the host's Python (deployment.HOST_PYTHON).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from . import (
    commands,
    file_types,
    outcome,
    private_files,
    program,
    settings_file,
    steps,
    wrapping,
)
from .checks import Command, Status
from .paths import PACKAGE

SECTION = commands.find(
    ("template", "section"), commands.TABLE
)  # the section command's row


@dataclass(frozen=True)
class Template:
    """One settings template: its file type, its profile, and whether it is the every-key text."""

    file_type: settings_file.FileType
    profile: str | None = None
    every_key: bool = False

    @property
    def label(self) -> str:
        """Its name in the privacy check's report: "build", "deploy full", "deploy every key"."""
        kind = self.file_type.noun.split()[0]
        if self.every_key:
            return f"{kind} every key"
        return f"{kind} {self.profile}" if self.profile else kind

    def text(self) -> str:
        """The template's text, as settings_file writes it."""
        return settings_file.template(self.file_type, self.profile, self.every_key)

    def command(self) -> Command:
        """The command that prints it: see the module's docstring."""
        if self.every_key:
            return steps.command_line(
                steps.command_name(self.file_type.section_command)
            )
        words = {"FILE": "-"}
        if self.profile:
            words["PROFILE"] = self.profile
        return steps.command_line(self.file_type.template_command, **words)


def templates() -> tuple:
    """Every template the package gives you: see the module's docstring."""
    deploy = file_types.DEPLOY
    return (
        (Template(file_types.BUILD),)
        + tuple(Template(deploy, p.name) for p in settings_file.profiles(deploy))
        + (Template(deploy, every_key=True),)
    )


def section_text(key: str | None = None) -> str:
    """What `rkmon template section [KEY]` prints: see the module's docstring."""
    deploy = file_types.DEPLOY
    if key is not None:
        return settings_file.section(deploy, key)
    found = settings_file.sections(deploy)
    column = 2 + max(len(k) for k, _ in found)
    return "".join(f"{k.ljust(column)}{title}".rstrip() + "\n" for k, title in found)


def section_command(key: str | None = None, out=None, err=None) -> int:
    """`rkmon template section [KEY]`: see the module's docstring."""
    try:
        text = section_text(key)
    except ValueError as exc:
        err = sys.stderr if err is None else err
        err.write(f"{program.name()} {SECTION.name}: {exc}\n")
        err.flush()
        return outcome.USAGE
    out = sys.stdout if out is None else out
    out.write(text)
    out.flush()
    return 0


def _profile_name(template: Template) -> str | None:
    """The profile TEMPLATE's text is written for: its own, else the schema's first; None for a file type without."""
    if template.profile is not None:
        return template.profile
    return next((p.name for p in settings_file.profiles(template.file_type)), None)


def _step(template: Template) -> steps.Step:
    """The step that writes TEMPLATE: step 1 of the build path or of the deploy path."""
    return next(
        s
        for path_steps in steps.PATHS.values()
        for s in path_steps
        if s.command == template.file_type.template_command
    )


def _refusal_next(step: steps.Step, argv, next_line) -> str:
    """A refusal's next line: on the deploy side NEXT_LINE, the state's, which the caller gives; on the build side
    your command again, ARGV as you typed it, and the caller gives none (see the module's docstring).
    """
    if step.path == "build":
        if next_line is not None:
            raise TypeError(
                "write() takes no next_line for the build template: a refusal's is your command again"
            )
        return f"{steps.NEXT_LABEL}{Command(tuple(argv))}"
    if next_line is None:
        raise TypeError(
            "write() takes next_line for a deploy template: a refusal's is the deployment state's "
            "(states.next_line), which only the command layer gathers"
        )
    return next_line


def write(
    template: Template,
    file: str,
    argv,
    package_dir=PACKAGE,
    out=None,
    *,
    next_line: str | None = None,
    dry_run: bool = False,
    plain: bool = False,
) -> int:
    """`rkmon template build FILE` and `rkmon template deploy [--profile PROFILE] FILE`: see the module's docstring."""
    if template.every_key:
        raise ValueError(
            "the every-key text is printed by sections, never written as a file"
        )
    out = sys.stdout if out is None else out
    text = template.text()
    if file == "-":
        out.write(text)
        out.flush()
        return 0
    step = _step(template)
    refusal_next = _refusal_next(step, argv, next_line)
    retry = private_files.again(argv, given=file)
    shown = settings_file.shown(file)
    try:
        path = private_files.write(
            file, text, package_dir, retry=retry, dry_run=dry_run, shown=shown
        )
    except private_files.Refused as refused:
        out.write(f"{refused}\n\n")
        return outcome.finish("REFUSED", "nothing written", next=refusal_next, out=out)
    profile = _profile_name(template)
    kept = f"({f'profile {profile}; ' if profile else ''}yours only, permissions 600)"
    if dry_run:
        real = (
            refusal_next if step.path == "deploy" else steps.next_line(step, FILE=shown)
        )
        what = f"{shown} can be written {kept}; nothing written"
        return outcome.finish("PASSED", what, next=real, out=out)
    doc = settings_file.read(path, template.file_type)
    left = sum(1 for r in settings_file.validate(doc) if r.status == Status.FAIL)
    what = f"{shown} {kept}"
    if left:
        what += f": fill in the {wrapping.counted(left, 'value')} marked <...>"
    following = steps.next_line(steps.after(step), FILE=shown, plain=plain)
    return outcome.finish("WROTE", what, next=following, out=out)
