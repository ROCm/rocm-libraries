#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI entry point for the hipDNN generic-kernel-ingestor descriptor generator.

Exit 0 on success. Exit 1 is a gate refusal or an internal failure: a ``--strict``
budget overrun, ``--check-placeholders`` findings, an emitted file that cannot be
read back, or a template error. Exit 2 is unusable input: a missing or invalid
config, a non-existent ``--emitted-root``, an ``--output-dir`` that is not a directory,
a non-empty output directory without ``--force``, an ``--ids-from`` or
``--engine-ids-from`` that is not a directory, or a ``ValueError`` from resolving ids
or rendering (and argparse's code for a bad flag).
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

from codegen.config_loader import DEFAULT_AOT_BUDGET_PER_ARCH, ConfigError, load_config
from codegen import prior_ids
from codegen.generator import IdReport, IdResolution, IngestorGenerator, resolve_ids
from codegen.models import EVERY_ARCH

EXIT_REFUSED = 1
EXIT_BAD_INPUT = 2


def warn_over_budget(config) -> int:
    """WARN on stderr for each pack arch over ``reduction.aot_budget_per_arch``
    (default `DEFAULT_AOT_BUDGET_PER_ARCH`) kernels; returns how many are over."""
    budget = config.reduction.get("aot_budget_per_arch", DEFAULT_AOT_BUDGET_PER_ARCH)
    over = 0
    for pack in config.packs:
        per_arch = Counter(
            arch
            for kernel in pack.kernels
            for arch in (kernel.arch or pack.arch or [EVERY_ARCH])
        )
        for arch, count in per_arch.items():
            if count > budget:
                over += 1
                print(
                    f"WARN: {count} instances for {arch} exceed the AOT budget "
                    f"{budget} (red flag)",
                    file=sys.stderr,
                )
    return over


#: Where the provider keeps installed bundles, relative to the repository root.
#: Both dialects' descriptor directories (``descriptors/...`` and
#: ``test_descriptors/...``) sit directly under it.
INSTALLED_ENGINE_DIR = (
    "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine"
)


def installed_descriptor_dir(config) -> Path | None:
    """Where this bundle lives once installed in the provider, or None when this
    checkout has no provider tree. Derived from ``config.descriptor_dir``, which
    mirrors the packager's source root.

    The script's own path is not resolved: a symlinked checkout (the develop
    worktree is one) would resolve to a location outside the repository."""
    for ancestor in Path(os.path.abspath(__file__)).parents:
        engine_dir = ancestor / INSTALLED_ENGINE_DIR
        if engine_dir.is_dir():
            return engine_dir / config.descriptor_dir
    return None


def installed_siblings(config) -> list[Path]:
    """The other installed bundles of this config's engine, found beside this
    bundle's installed directory. Only packaged bundles can be satellites."""
    installed = installed_descriptor_dir(config)
    if installed is None or not config.is_packaged:
        return []
    return prior_ids.discover_siblings(installed.parent, config.engine.name, installed)


def holds_previous_version(root: Path) -> bool:
    """Whether ``root`` holds a previous version of a bundle: a UED or a KDP, or a
    ``*.dvc`` pointer for an artifact (an unfetched KDP), which reading ``root``
    then refuses."""
    return prior_ids.holds_docs(root, "ued", "kdp") or (
        root.is_dir() and any(True for _ in root.rglob("*.dvc"))
    )


def earlier_passes(
    config, output_dir: Path, used: list[Path], engine_id: str | None = None
) -> list[Path]:
    """Directories beside ``output_dir`` that hold an earlier generation of this
    engine's bundle and were not read, newest first. A new output directory per
    pass reads nothing unless told where the last pass went.

    A bundle is this engine's when its UED has the engine's name or, for a
    satellite, which has no UED, when a KDP names ``engine_id``. An entry that
    cannot be read is not an earlier pass."""
    parent = output_dir.parent
    try:
        if not parent.is_dir():
            return []
        skipped = {path.resolve() for path in used} | {output_dir.resolve()}
        found: list[tuple[float, Path]] = []
        entries = list(parent.iterdir())
    except OSError:
        return []
    for sibling in entries:
        bundle = sibling / config.descriptor_dir
        try:
            if sibling.resolve() in skipped or bundle.resolve() in skipped:
                continue
            if not prior_ids.holds_docs(bundle, "ued", "kdp"):
                continue
            if prior_ids.find_engine(bundle, config.engine.name) is None and not (
                engine_id and prior_ids.holds_kdp_of(bundle, engine_id)
            ):
                continue
            found.append((bundle.stat().st_mtime, bundle))
        except (OSError, ValueError):
            continue
    return [path for _, path in sorted(found, key=lambda item: item[0], reverse=True)]


def describe_ids(report: IdReport, fresh: bool = False) -> str:
    """The one-line account of where this run's ids came from."""
    reused = report.reused
    total = sum(reused.values())
    source = report.own_prior or report.home
    if fresh:
        return f"Ids: --new-ids; every id is new (minted {report.minted})"
    if source is None:
        return f"Ids: none found to reuse; minted {report.minted} (new bundle)"
    parts = [f"engine {reused['engine']}", f"packs {reused['packs']}"]
    if reused["operation_umds"]:
        parts.append(f"operation UMDs {reused['operation_umds']}")
    parts.append(f"kernels {reused['kernels']}")
    line = (
        f"Ids: reused {total} from {source} ({', '.join(parts)}), "
        f"minted {report.minted}, retired {len(report.retired)}"
    )
    if report.retired:
        shown = ", ".join(report.retired[:10])
        more = (
            f" (+{len(report.retired) - 10} more)" if len(report.retired) > 10 else ""
        )
        line += f": {shown}{more}"
    if report.home is not None and report.home != source:
        line += f"; engine ids from {report.home}"
    if report.arch_moved:
        shown = ", ".join(report.arch_moved[:5])
        more = (
            f" (+{len(report.arch_moved) - 5} more)"
            if len(report.arch_moved) > 5
            else ""
        )
        line += f"; {len(report.arch_moved)} kept an id across an arch change: {shown}{more}"
    return line


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a hipDNN generic-kernel-ingestor descriptor bundle "
        "from a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML config file (e.g., configs/conv_fwd.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for the generated bundle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which files would be generated without writing them or "
        "creating the output directory. It resolves the ids like a real run, so "
        "it exits 2 on a previous bundle it cannot use.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required to overwrite an existing, non-empty output directory. "
        "Without it, an existing non-empty output directory is an error -- "
        "regenerating over a live tree would silently clobber hand-filled "
        "content (e.g. a completed graph_match body). Descriptor ids are kept "
        "across a regeneration either way (see --ids-from). A run that would "
        "write a KDP beside the same pack's DVC artifact under artifacts/ is "
        "refused even with --force.",
    )
    parser.add_argument(
        "--ids-from",
        type=Path,
        help="The previous version of THIS bundle, e.g. a previous scratch "
        "directory's descriptors/rocKE/<slug>. Every descriptor and kernel whose "
        "stable key matches keeps its id; only new ones get a fresh id. Without "
        "it, the bundle already installed in the provider tree "
        "(dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/"
        "<descriptor dir>) is used when it exists. The descriptor directory "
        "under --output-dir is read as well when it holds a bundle: it is the "
        "only source when nothing is installed, and otherwise it fills in the "
        "ids of descriptors added since installation, installed ids winning.",
    )
    parser.add_argument(
        "--new-ids",
        action="store_true",
        help="Give every descriptor a new id and read no previous bundle: for an "
        "engine that is being replaced rather than extended. Without it, an "
        "installed bundle of this name is reused.",
    )
    parser.add_argument(
        "--engine-ids-from",
        action="append",
        dest="engine_ids_from",
        default=[],
        type=Path,
        help="An existing bundle of the same engine. Repeatable. Without it, the "
        "other installed bundles beside this one in the provider tree are found "
        "automatically; name them yourself for bundles outside the provider tree "
        "(e.g. scratch), and then name EVERY other bundle of the engine. The "
        "engine-level ids are taken from the bundle holding the engine's UED, and "
        "a bundle that does not hold it writes only its KDP.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 before generating when any pack arch exceeds "
        "reduction.aot_budget_per_arch, instead of only warning. Use it in a "
        "non-exploratory run or a CI step.",
    )
    parser.add_argument(
        "--check-placeholders",
        action="store_true",
        help="Do not generate. Report unfilled stub markers in the files this "
        "config's engine emits, searched under --emitted-root (default: "
        "--output-dir), and exit 1 if any remain or if any of them cannot be "
        "located. It derives the file set from the config, so it cannot miss a "
        "file a hand-written glob forgot.",
    )
    parser.add_argument(
        "--emitted-root",
        action="append",
        dest="emitted_roots",
        default=[],
        type=Path,
        help="A root the provider spliced this engine's emitted files into. "
        "Repeatable, because the provider SPLITS the bundle: packs land in the "
        "engine directory and the generated test stubs under "
        "src/tests/engines/.../packs/, which may share no ancestor worth "
        "scanning. Each root is searched only at the engine-specific relative "
        "paths this generator's own CMake fragments splice to, and a file found "
        "at the same relative path under two roots is an error rather than a "
        "pick. Omit to search --output-dir alone. A root that does not exist is "
        "an error, never an empty search.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(EXIT_BAD_INPUT)

    try:
        config = load_config(args.config)
    except (ConfigError, yaml.YAMLError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(EXIT_BAD_INPUT)

    over_budget = warn_over_budget(config)
    if over_budget and args.strict:
        print(
            f"Error: {over_budget} arch(es) exceed the AOT budget and --strict is set.",
            file=sys.stderr,
        )
        sys.exit(EXIT_REFUSED)

    if args.new_ids and (args.ids_from or args.engine_ids_from):
        parser.error("--new-ids contradicts --ids-from and --engine-ids-from")
    for flag, given in (
        ("--ids-from", [args.ids_from] if args.ids_from else []),
        ("--engine-ids-from", args.engine_ids_from),
    ):
        for path in given:
            if not path.is_dir():
                print(
                    f"Error: {flag} '{path}' does not exist or is not a directory.",
                    file=sys.stderr,
                )
                sys.exit(EXIT_BAD_INPUT)

    print(f"Loaded config for engine: {config.engine.name}")
    print(f"  Packs: {[p.name for p in config.packs]}")
    print(f"  Knobs: {config.engine.knobs}")

    template_dir = Path(__file__).parent / "templates"
    generator = IngestorGenerator(template_dir)

    sources: list[Path] = []

    def resolve() -> IdResolution:
        """Resolve ids for this run, exiting 2 on any unusable input."""
        ids_from, automatic, also_from = args.ids_from, False, None
        siblings = args.engine_ids_from
        if not args.new_ids and not siblings:
            siblings = installed_siblings(config)
            if siblings:
                automatic = True
                print(
                    "Reusing the engine's ids from the installed bundle(s) "
                    + ", ".join(str(sibling) for sibling in siblings)
                )
        if not args.new_ids and ids_from is None:
            installed = installed_descriptor_dir(config)
            if installed is not None and holds_previous_version(installed):
                ids_from, automatic = installed, True
                print(f"Reusing ids from the installed bundle {installed}")
                held = args.output_dir / config.descriptor_dir
                if prior_ids.holds_docs(held, "ued", "kdp") and held.resolve() != (
                    installed.resolve()
                ):
                    also_from = held
                    print(
                        f"Also reading {held} for the ids of descriptors added since "
                        f"installation; installed ids win wherever both have one."
                    )
        try:
            resolution = resolve_ids(
                config,
                args.output_dir,
                ids_from,
                siblings,
                fresh=args.new_ids,
                also_from=also_from,
            )
        except ValueError as e:
            # Naming another source cannot cure a refusal about the tree read.
            hint = (
                "\n(Installed bundles were picked up automatically; --new-ids "
                "ignores them, and --ids-from / --engine-ids-from override them.)"
                if automatic and not isinstance(e, prior_ids.NotOverridable)
                else ""
            )
            print(f"Error: {e}{hint}", file=sys.stderr)
            sys.exit(EXIT_BAD_INPUT)
        for warning in resolution.report.warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        sources.extend(
            path
            for path in (resolution.report.own_prior, also_from, resolution.report.home)
            if path is not None
        )
        if resolution.report.minted and not args.new_ids:
            earlier = earlier_passes(
                config, args.output_dir, sources, resolution.ids.get("ued")
            )
            if earlier:
                print(
                    f"Warning: {resolution.report.minted} descriptor(s) got new ids, "
                    f"and {earlier[0]} holds an earlier generation of this engine "
                    f"that was not read. If it is the previous pass, run again with "
                    f"--ids-from {earlier[0]}.",
                    file=sys.stderr,
                )
        return resolution

    def refuse_second_kdp() -> None:
        """Exit 2 when this run would shadow a pack's DVC artifact."""
        try:
            generator.refuse_second_kdp(config, args.output_dir)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(EXIT_BAD_INPUT)

    resolution = None
    if args.dry_run:
        resolution = resolve()
        refuse_second_kdp()
        files = generator.preview_files(config, resolution=resolution)
        print(
            f"\nDry run -- would generate {len(files)} files (output dir not created):"
        )
        for f in files:
            print(f"  {f}")
        print(
            describe_ids(resolution.report, args.new_ids).replace(
                "Ids:", "Ids (would be):", 1
            )
        )
        return

    if args.check_placeholders and (
        args.engine_ids_from or (not args.new_ids and installed_siblings(config))
    ):
        # Only a satellite's file list differs from the home layout, so ids are
        # resolved for this mode only when a sibling is in play.
        resolution = resolve()

    if args.check_placeholders:
        expected = generator.preview_files(config, resolution=resolution)
        # No --emitted-root means the output dir is the one root.
        roots = args.emitted_roots or [args.output_dir]
        shown = ", ".join(f"'{root}'" for root in roots)
        try:
            located, missing, ambiguous = generator.locate_emitted(roots, expected)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(EXIT_BAD_INPUT)
        try:
            unfilled = generator.unfilled_placeholders(roots, expected)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(EXIT_REFUSED)
        total_shippable = len(located) + len(missing) + len(ambiguous)
        print(
            f"\nLocated {len(located)} of this engine's {total_shippable} shippable "
            f"files under {shown}."
        )
        failed = False
        if ambiguous:
            # One relative path resolving twice: the scan cannot tell which is
            # yours. See `locate_emitted`.
            print(f"{len(ambiguous)} file(s) matched in more than one place:")
            for rel, paths in ambiguous.items():
                print(f"        {rel}")
                for p in paths:
                    print(f"            {p}")
            print(
                "  Narrow the roots so each file resolves once (a build tree or a "
                "stale copy under one of them is the usual cause)."
            )
            failed = True
        if missing:
            # A shippable file nobody can find is an unfinished splice, not a
            # pass: "no unfilled placeholders" over a set the scan never
            # located is a false green, so this fails even when every located
            # file was clean.
            print(
                f"{len(missing)} expected file(s) not found under any of those roots:"
            )
            for rel in missing:
                print(f"        {rel}")
            print(
                "  Add an --emitted-root for the tree holding them (the provider "
                "splits the engine and its tests; the cmake_test_sources fragment "
                "says where), or finish the splice."
            )
            failed = True
        if unfilled:
            total = sum(unfilled.values())
            print(f"{total} unfilled placeholder(s) across {len(unfilled)} file(s):")
            for rel, count in unfilled.items():
                print(f"  {count:4}  {rel}")
            failed = True
        if failed:
            sys.exit(EXIT_REFUSED)
        print(f"No unfilled placeholders across all {total_shippable} located files.")
        return

    if args.output_dir.exists() and not args.output_dir.is_dir():
        print(
            f"Error: Output directory '{args.output_dir}' exists and is not a "
            f"directory.",
            file=sys.stderr,
        )
        sys.exit(EXIT_BAD_INPUT)

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        print(
            f"Error: Output directory '{args.output_dir}' already exists and is "
            f"non-empty. Pass --force to overwrite (only safe when you intend to "
            f"regenerate over hand-filled content you have already reviewed).",
            file=sys.stderr,
        )
        sys.exit(EXIT_BAD_INPUT)

    # Resolved before anything is written: the output directory is itself a
    # source of ids.
    resolution = resolve()
    refuse_second_kdp()

    # Said before anything is written, while the files can still be dealt with.
    expected = set(generator.preview_files(config, resolution=resolution))
    stale = sorted(
        path.relative_to(args.output_dir).as_posix()
        for path in (args.output_dir / config.descriptor_dir).rglob("*.json")
        if path.relative_to(args.output_dir).as_posix() not in expected
    )
    if stale:
        print(
            f"Warning: {len(stale)} descriptor file(s) under {config.descriptor_dir} "
            f"will not be written by this run and stay there: {stale}. A removed "
            f"pack's KDP is the usual cause; the next run reads it as part of the "
            f"previous bundle, so delete what is no longer yours.",
            file=sys.stderr,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        written = generator.render(config, args.output_dir, resolution=resolution)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(EXIT_BAD_INPUT)
    except RuntimeError as e:
        print(f"Template rendering error: {e}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)

    print(f"\nGenerated {len(written)} files:")
    for f in written:
        print(f"  {f}")
    print(describe_ids(resolution.report, args.new_ids))

    # A fresh generation wrote everything into one tree, so that tree is the
    # root. A file this run wrote and cannot read back fails the run, since the
    # count below would otherwise read as the whole bundle's.
    try:
        unfilled = generator.unfilled_placeholders([args.output_dir], written)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    if unfilled:
        total = sum(unfilled.values())
        print(
            f"\n{total} unfilled placeholder(s) across {len(unfilled)} file(s) -- "
            "each one is a hook body you owe:"
        )
        for rel, count in unfilled.items():
            print(f"  {count:4}  {rel}")
        print(
            "  Re-run this command after implementing them; zero here is "
            "RUNBOOK §3's placeholder gate. Note the generated tests/ stubs "
            "count too."
        )

    print(
        "\nDone. See README.md for the CMake/registration splice points and the "
        "validate-descriptors round trip."
    )


if __name__ == "__main__":
    main()
