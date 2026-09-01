"""Make the Python/C++ launch-contract restatement AUDITABLE, per surface, per op.

A rocKE kernel is launched from Python; the C++ ingestor engine RESTATES that launch
contract by hand -- grid shape, block size, kernarg order, baked constants (strides
compiled into the binary with no kernarg), the dispatcher's spec resolution, and
applicability. Nothing in the build, the packer, the validator or the test suite
compares the two halves. A mismatch does not fail: the kernel runs and computes
something else.

Two defects have already shipped through this restatement (the persistent grid
launched with the default grid size, and a windowed causal graph served as plain
causal), and this repository just found a THIRD, structurally distinct one while
building this tool: the gfx942 attention_dense profile's ``kmd_fields`` never
declared ``block_m``, even though ``Gfx942AttentionDenseNative.cpp`` reads it via
``kernel.getIntMetadata("block_m")`` on both the matcher path (``kernelMatches``,
the ``seqlen_q % block_m == 0`` tile check) and the prepare() path
(``attentionDenseGeometry``, which sizes the launch grid from it). Every descriptor
this profile generated was missing a field its own engine dereferences unconditionally
-- a restated helper cannot branch on a field the descriptor does not carry, and
nothing before this tool cross-referenced "fields the C++ mirror reads" against
"fields the KMD declares". That gap is check (1) below, and it is the headline check:
it is mechanical, requires no device, and would have caught the defect on the first
run.

WHAT A ``launch_surface:`` BLOCK DECLARES, per surface (grid, block, kernargs,
baked_constants, spec_resolution, applicability, or any further split an op needs):

    launch_surface:
      - name: grid
        python_source: kernels/gfx942/attention_dense.py:attention_dense_grid (:1803-1819)
        cpp_mirror: dnn-providers/.../Gfx942AttentionDenseGeometry.hpp:attentionDenseGeometry
        kmd_fields: [block_m, seqlen_q, num_query_heads, batch, persistent, num_persistent]
        guard: attentionDenseGeometry throws on a non-positive dimension or an
          unusable persistent/num_persistent pair (lines 82-107)
        test: dnn-providers/.../TestGfx942AttentionDenseGeometry.cpp

``guard: none`` or ``test: none`` are not omissions -- they are the honest answer for
a surface nothing defends or nothing compares, and the point of writing "none" rather
than leaving the key out is that an unguarded surface must be a DELIBERATE, VISIBLE
choice, not a blank a reviewer has to notice is missing.

WHAT ``--check`` VERIFIES, mechanically, from the profile alone (no device, no build):

  1. Every ``kmd_fields`` entry a surface names actually exists in the profile's own
     top-level ``kmd_fields`` list. This is the check described above.
  2. Every ``cpp_mirror`` path exists on disk (paths resolve against the current
     working directory, the same convention ``provider_root`` uses elsewhere in this
     profile format).
  3. Every ``test`` path exists on disk, when it is not the literal string ``none``.
  4. Every surface with ``guard: none`` or ``test: none`` is reported BY NAME as
     unguarded/untested, and the command exits non-zero unless ``--allow-unguarded``
     is passed. Nothing here says an unguarded surface is wrong -- some genuinely
     have no cheaper defense yet -- but silence about it is what let two defects
     ship, so the check refuses to be quiet about the surfaces still not restated
     under a comparison of any kind.

WHAT IT DOES NOT DO. It cannot check that ``python_source`` and ``cpp_mirror`` agree
-- that is what the ``test`` column is for, and a per-surface test (the geometry
header's own test, the matcher negatives, dispatch_parity's persistent-rule check)
is how that comparison actually happens. This tool only makes the CLAIM structurally
honest: the fields a mirror needs are declared, the files it and its test live in
exist, and an absent guard or test is named rather than silently missing.

    launch_surface.py --check <profile.yaml>
    launch_surface.py --check <profile.yaml> --allow-unguarded
    launch_surface.py --report <profile.yaml>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: Every key a launch_surface entry must declare. Enforced eagerly so a surface
#: missing a key is reported as a shape error rather than crashing on the first
#: check that happens to dereference it.
REQUIRED_KEYS = ("name", "python_source", "cpp_mirror", "kmd_fields", "guard", "test")

#: The literal value meaning "nothing defends/compares this surface". A string, not
#: an absent key -- see the module docstring for why the distinction matters.
NONE_SENTINEL = "none"


class LaunchSurfaceError(RuntimeError):
    """The profile could not be read, or its launch_surface block is malformed."""


def _load_profile(path: str) -> dict:
    """Parse a profile as JSON, falling back to YAML.

    The mapping check covers BOTH paths. It used to sit only on the YAML branch, so a
    file that parsed as valid JSON but was not an object -- a bare list, say -- sailed
    through and crashed later with an AttributeError naming neither the file nor the
    problem.
    """
    text = Path(path).read_text()
    try:
        import json

        loaded = json.loads(text)
    except ValueError:
        try:
            import yaml
        except ImportError:  # pragma: no cover - environment-dependent
            raise LaunchSurfaceError(f"{path} is not JSON and PyYAML is not installed.")
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise LaunchSurfaceError(
            f"profile {path} must be a mapping; got {type(loaded).__name__}."
        )
    return loaded


def load_surfaces(profile: dict) -> list[dict]:
    """The declared ``launch_surface:`` list, or an error if the profile has none.

    An op with NO launch_surface block is not "nothing to check" -- it is a profile
    that has not been audited yet, and reporting that as a pass would be exactly the
    silent gap this tool exists to close.
    """
    surfaces = profile.get("launch_surface")
    if not surfaces:
        raise LaunchSurfaceError(
            "profile declares no 'launch_surface:' block. Every op this engine ships "
            "restates a launch contract from Python; an absent block means it has "
            "not been audited, not that there is nothing to audit."
        )
    if not isinstance(surfaces, list):
        raise LaunchSurfaceError("'launch_surface:' must be a list of surface entries.")
    return surfaces


def _path_part(value: str) -> str:
    """The filesystem path prefix of a ``cpp_mirror``/``test`` value.

    Both fields carry a path plus free-text locator (``Foo.cpp:functionName (~702)``,
    a pytest node id ``test_x.py::TestY::test_z``), so a plain existence check needs
    only what precedes the first ``:`` -- no path used anywhere in this repo contains
    one.
    """
    return value.split(":", 1)[0].strip()


def validate_shape(surfaces: list[dict]) -> list[str]:
    """Structural errors that make a surface unreadable, before anything else runs."""
    errors = []
    for index, surface in enumerate(surfaces):
        if not isinstance(surface, dict):
            errors.append(f"surface #{index} is not a mapping: {surface!r}")
            continue
        missing = [k for k in REQUIRED_KEYS if k not in surface]
        if missing:
            name = surface.get("name", f"#{index}")
            errors.append(f"surface '{name}' is missing required key(s): {missing}")
            continue
        if not isinstance(surface["kmd_fields"], list):
            errors.append(
                f"surface '{surface['name']}': kmd_fields must be a list of field "
                f"names, got {type(surface['kmd_fields']).__name__}"
            )
    return errors


def check(profile: dict, root: Path) -> tuple[list[str], list[str]]:
    """(failures, unguarded_or_untested) for every declared surface.

    ``failures`` are structural: a named kmd_fields entry the KMD does not carry, a
    cpp_mirror/test path absent from disk, or a malformed entry -- each one means the
    profile's own claim about itself is false. ``unguarded_or_untested`` is a
    separate list on purpose: an honestly-declared ``guard: none`` is not a
    structural defect, it is a fact the caller must decide whether to accept
    (``--allow-unguarded``), and conflating the two would make an honest admission
    look the same as a broken path.
    """
    surfaces = load_surfaces(profile)
    shape_errors = validate_shape(surfaces)
    if shape_errors:
        return shape_errors, []

    kmd_names = {f["name"] for f in (profile.get("kmd_fields") or [])}

    failures: list[str] = []
    unguarded: list[str] = []
    for surface in surfaces:
        name = surface["name"]

        # (1) The headline check: a restated helper cannot branch on a field the
        # descriptor does not carry. This is exactly how block_m shipped absent from
        # this profile's own kmd_fields while Gfx942AttentionDenseNative.cpp read it
        # unconditionally on two live paths.
        undeclared = [f for f in surface["kmd_fields"] if f not in kmd_names]
        if undeclared:
            failures.append(
                f"surface '{name}' names kmd_fields {undeclared}, which the "
                f"profile's own kmd_fields does not declare -- the C++ mirror would "
                f"dereference a field the descriptor cannot carry"
            )

        # (2) cpp_mirror must point at a real file.
        cpp_path = _path_part(surface["cpp_mirror"])
        if cpp_path and not (root / cpp_path).exists():
            failures.append(
                f"surface '{name}' cpp_mirror path does not exist: {cpp_path}"
            )

        # (3) test must point at a real file, unless explicitly none.
        test_value = surface["test"]
        if test_value != NONE_SENTINEL:
            test_path = _path_part(test_value)
            if test_path and not (root / test_path).exists():
                failures.append(
                    f"surface '{name}' test path does not exist: {test_path}"
                )

        # (4) An unguarded or untested surface is reported by name, never silently.
        if surface["guard"] == NONE_SENTINEL or test_value == NONE_SENTINEL:
            unguarded.append(name)

    return failures, unguarded


def render_report(profile: dict) -> str:
    """The markdown table an integration owes, ready to paste into a PR."""
    surfaces = load_surfaces(profile)
    errors = validate_shape(surfaces)
    if errors:
        raise LaunchSurfaceError("; ".join(errors))

    lines = [
        "| surface | python source | C++ mirror | KMD fields | guard | test |",
        "|---|---|---|---|---|---|",
    ]
    for surface in surfaces:
        fields = ", ".join(surface["kmd_fields"]) or "(none)"
        lines.append(
            f"| {surface['name']} | {surface['python_source']} | "
            f"{surface['cpp_mirror']} | {fields} | {surface['guard']} | "
            f"{surface['test']} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the Python/C++ launch-contract restatement, per surface.",
    )
    parser.add_argument("profile", help="Path to the kernel's tool profile YAML.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify kmd_fields/cpp_mirror/test mechanically; exit non-zero on a "
        "structural failure or an unguarded surface (see --allow-unguarded).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print the surface table in markdown.",
    )
    parser.add_argument(
        "--allow-unguarded",
        action="store_true",
        help="Do not fail --check solely because a surface declares guard: none or "
        "test: none. The surfaces are still printed, by name -- this flag only "
        "changes the exit code, never the report.",
    )
    args = parser.parse_args(argv)

    if not args.check and not args.report:
        parser.error("choose --check and/or --report")

    try:
        profile = _load_profile(args.profile)
    except LaunchSurfaceError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    if args.report:
        try:
            print(render_report(profile))
        except LaunchSurfaceError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2

    if args.check:
        try:
            surfaces = load_surfaces(profile)
        except LaunchSurfaceError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2

        failures, unguarded = check(profile, Path.cwd())

        print("launch-surface check")
        print(f"  surfaces declared  {len(surfaces)}")
        if failures:
            print(f"  structural failures  {len(failures)}")
            for f in failures:
                print(f"      ! {f}")
        else:
            print("  structural failures  0")

        if unguarded:
            verb = "REPORTED" if args.allow_unguarded else "FAILING"
            print(f"  unguarded/untested  {len(unguarded)} ({verb})")
            for name in unguarded:
                print(f"      ? {name}")
        else:
            print("  unguarded/untested  0")

        print()
        if failures or (unguarded and not args.allow_unguarded):
            reasons = []
            if failures:
                reasons.append(f"{len(failures)} structural failure(s)")
            if unguarded and not args.allow_unguarded:
                reasons.append(
                    f"{len(unguarded)} unguarded/untested surface(s); pass "
                    f"--allow-unguarded to accept them as a deliberate choice"
                )
            print(f"CHECK FAILED ({'; '.join(reasons)})")
            return 1
        print(
            "CHECK PASSED: every surface's kmd_fields, cpp_mirror and test are "
            "real, and every unguarded surface was a declared choice."
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
