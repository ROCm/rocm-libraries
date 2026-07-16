# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature-agnostic developer tool for benchmarking new TensileLite codegen
solution parameters in hipBLASLt.

This tool takes a known TensileLite solution config, toggles one or more *new*
codegen solution parameter(s) on, and produces an experimental,
hipBLASLt-loadable device library (``TensileLibrary_lazy_<arch>.dat`` + code
objects). The resulting library can be loaded at runtime via
``HIPBLASLT_TENSILE_LIBPATH`` and benchmarked with ``hipblaslt-bench`` to
compare a feature across branches.

``gen-logic`` benchmarks on real hardware: winner selection during logic
analysis uses measured GFLOPS. It therefore REQUIRES a GPU of the target
``--arch`` to be present on the host and fails fast otherwise (it does not fall
back to synthetic performance data, which would make forked solutions tie and
silently drop all but the first). Cross-arch generation without benchmarking is
intentionally not supported here.

Pipeline (each stage verifies its own output artifact; nothing is produced
silently half-built):

  list-solutions
             Filter the solutions in a shipped ``3_LibraryLogic`` by parameter
             (e.g. ``--where StreamK=5``) to discover which indices to extract.
  extract    Reverse a shipped ``3_LibraryLogic`` yaml into a benchmark config
             (wraps ``Tensile.TensileLibLogicToYaml``).
  merge      Combine several per-solution ``extract`` configs into one
             multi-problem config so a whole family rebuilds into one library.
  augment    Validate ``--set NAME=v1[,v2]`` against the canonical
             ``validParameters`` registry and inject/override them into the
             config's ForkParameters; stage the result under an
             ``Experimental/<feature>/`` tree.
  gen-logic  Run the Tensile benchmark+analyze flow to emit ``3_LibraryLogic``
             and classify failures: kernel-generation error vs. all-solutions
             rejected by Solution validation. Benchmarks on the target-arch GPU
             (fails fast if that arch is not present on the host).
  build-lib  Run ``Tensile.TensileCreateLibrary --experimental`` to turn the
             staged logic into a loadable device library.
  find-index Run ``hipblaslt-bench --algo_method all`` to discover the solution
             indices reachable in the experimental library.
  bench      Run ``hipblaslt-bench --algo_method index --solution_index N``.
  pipeline   Chain augment -> gen-logic -> build-lib, short-circuiting on the
             first failing stage.

Hardware notes:
  * ``gen-logic`` passes ``--gpu-targets <arch>`` and runs the real client, so a
    GPU of ``<arch>`` must be present on the host; winner selection uses the
    measured GFLOPS. The stage refuses to run on non-matching hardware.
  * ``createLibraryLogic`` still calls ``getCUCount()`` which shells to
    ``rocminfo`` UNLESS the ``CU`` env var is set, so ``gen-logic`` sets it
    from ``--cu`` to pin the CU count used for the build predicate.

Example (StreamKFixupTreeReduction, gfx950) -- see ``--help`` of each
subcommand:

  python -m Tensile.ExperimentalLibrary extract \\
      --logic <shipped>/gfx950/.../<liblogic>.yaml --indices 0 --out base.yaml
  python -m Tensile.ExperimentalLibrary pipeline \\
      --config base.yaml --set StreamKFixupTreeReduction=1 \\
      --set StreamK=1 --feature-name streamk_treereduce \\
      --arch gfx950 --cu 256 --out work/

A/B family example -- rebuild every StreamK==5 solution from a shipped logic
with a new feature toggled OFF *and* ON inside a single library, so the two can
be contrasted by solution index (StreamKWorkStealing is illustrative; use any
real parameter, and --skip-validation for a parameter not yet in the registry):

  IDX=$(python -m Tensile.ExperimentalLibrary list-solutions \\
      --logic <shipped>/.../<liblogic>.yaml --where StreamK=5 --indices-only)
  python -m Tensile.ExperimentalLibrary extract \\
      --logic <shipped>/.../<liblogic>.yaml --indices "$IDX" --out sk5/base.yaml
  python -m Tensile.ExperimentalLibrary merge \\
      --configs sk5/base*.yaml --out sk5/merged.yaml --feature-name sk5_ws
      # (``base*.yaml`` matches both ``base.yaml`` for a single index and
      #  ``base_<idx>.yaml`` for two or more.)
  python -m Tensile.ExperimentalLibrary pipeline \\
      --config sk5/merged.yaml --set StreamKWorkStealing=0,1 \\
      --feature-name sk5_ws --arch gfx950 --cu 256 --out work/ --skip-validation

Omit the ``--set`` / use ``gen-logic`` + ``build-lib`` directly to just rebuild
the selected family as-is into one experimental library.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The Tensile package directory that contains this module. Used to locate the
# bin/Tensile launcher (``python -m Tensile.Tensile`` is intentionally disabled
# upstream).
_TENSILE_PKG_DIR = Path(__file__).resolve().parent

_SPDX_HEADER = (
    "# Copyright Advanced Micro Devices, Inc., or its affiliates.\n"
    "# SPDX-License-Identifier: MIT\n"
)


class ExperimentalLibraryError(RuntimeError):
    """Actionable, user-facing error raised by any stage of this tool."""


# ---------------------------------------------------------------------------
# Interpreter / environment helpers
# ---------------------------------------------------------------------------


def default_python() -> str:
    """Pick the interpreter to drive Tensile subprocesses.

    Prefers the repo venv under ``projects/hipblaslt/build/venv`` if present,
    otherwise falls back to the interpreter running this tool.
    """
    # tensilelite/Tensile/ -> tensilelite -> hipblaslt
    hipblaslt_root = _TENSILE_PKG_DIR.parent.parent
    candidate = hipblaslt_root / "build" / "venv" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def _format_command(cmd: Sequence[str], env_overrides: Optional[Dict[str, str]]) -> str:
    prefix = ""
    if env_overrides:
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_overrides.items()) + " "
    return prefix + " ".join(shlex.quote(str(c)) for c in cmd)


def run_command(
    cmd: Sequence[str],
    *,
    env_overrides: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
    log_path: Optional[str] = None,
    stream: bool = False,
) -> Tuple[int, str]:
    """Run a command, optionally capturing combined output to ``log_path``.

    Returns ``(returncode, captured_output)``. In ``dry_run`` mode the command
    (with env prefix) is printed and ``(0, "")`` is returned without executing.
    When ``stream`` is True, output goes straight to the console (used for the
    interactive bench run) and is not captured.
    """
    pretty = _format_command(cmd, env_overrides)
    if dry_run or verbose:
        print(f"$ {pretty}")
    if dry_run:
        return 0, ""

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    if stream:
        proc = subprocess.run(list(map(str, cmd)), env=env, cwd=cwd)
        return proc.returncode, ""

    proc = subprocess.run(
        list(map(str, cmd)),
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout or ""
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(output)
    if verbose and output:
        print(output)
    return proc.returncode, output


# ---------------------------------------------------------------------------
# Pure logic: parameter parsing / validation / config augmentation
# ---------------------------------------------------------------------------


def coerce_value(raw: str) -> Any:
    """Coerce a CLI string into the Python value used by ``validParameters``.

    Booleans are recognized first (``validParameters`` distinguishes ``bool``
    from ``int``); then ``ast.literal_eval`` handles ints, floats and lists;
    anything else is left as a bare string (e.g. ``MultipleBuffer``).
    """
    s = raw.strip()
    if s in ("True", "true", "False", "false"):
        return s.lower() == "true"
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def parse_set_arg(arg: str) -> Tuple[str, List[Any]]:
    """Parse a single ``NAME=v1[,v2,...]`` ``--set`` argument."""
    if "=" not in arg:
        raise ExperimentalLibraryError(
            f"Malformed --set '{arg}': expected NAME=value[,value...]"
        )
    name, _, values_str = arg.partition("=")
    name = name.strip()
    if not name:
        raise ExperimentalLibraryError(f"Malformed --set '{arg}': empty parameter name")
    values_str_stripped = values_str.strip()
    if values_str_stripped == "":
        raise ExperimentalLibraryError(
            f"Malformed --set '{arg}': no value(s) provided for '{name}'"
        )
    # A leading "[" means a single bracketed list value (e.g.
    # ``MatrixInstruction=[16,16,16,1]``); treat the whole thing as one token so
    # the embedded commas are not split into separate values.
    if values_str_stripped.startswith("["):
        values = [coerce_value(values_str_stripped)]
    else:
        values = [coerce_value(v) for v in values_str.split(",")]
    return name, values


def validate_sets(sets: Sequence[Tuple[str, List[Any]]]) -> None:
    """Validate parameter names and values against ``validParameters``.

    Raises :class:`ExperimentalLibraryError` with an actionable message:
    unknown names list the nearest valid candidates; bad values list the
    allowed values. A registry entry of ``-1`` means "skip value check".
    """
    try:
        from Tensile.Common.ValidParameters import validParameters
    except Exception as e:  # ModuleNotFoundError (rocisa), RuntimeError, etc.
        raise ExperimentalLibraryError(
            "Parameter validation needs Tensile.Common which requires a built "
            f"rocisa, but it could not be imported ({e}). Either build rocisa, "
            "launch this tool with the venv python via --python, or pass "
            "--skip-validation to bypass validation."
        ) from e

    for name, values in sets:
        if name not in validParameters:
            suggestions = difflib.get_close_matches(name, list(validParameters.keys()), n=5)
            hint = (
                f" Did you mean: {', '.join(suggestions)}?"
                if suggestions
                else " Run with a known solution parameter name."
            )
            raise ExperimentalLibraryError(
                f"Unknown solution parameter '{name}'.{hint}"
            )
        allowed = validParameters[name]
        if allowed == -1:
            continue
        for value in values:
            if value not in allowed:
                shown = allowed[:32] if isinstance(allowed, list) else allowed
                more = " (first 32 shown)" if isinstance(allowed, list) and len(allowed) > 32 else ""
                raise ExperimentalLibraryError(
                    f"Invalid value {value!r} for '{name}'. "
                    f"Allowed values{more}: {shown}"
                )


def _set_fork_parameter(fork_params: List[Dict[str, Any]], name: str, values: List[Any]) -> None:
    """Inject or override a single-key ForkParameters entry in place.

    Replaces an existing ``{name: ...}`` entry if present; otherwise inserts a
    new entry before any trailing ``Groups``/``MatrixInstruction`` entry (so
    the matrix-instruction block stays last), else appends.
    """
    for entry in fork_params:
        if isinstance(entry, dict) and name in entry:
            entry[name] = list(values)
            return

    new_entry = {name: list(values)}
    for idx, entry in enumerate(fork_params):
        if isinstance(entry, dict) and (
            "Groups" in entry or "MatrixInstruction" in entry
        ):
            fork_params.insert(idx, new_entry)
            return
    fork_params.append(new_entry)


def augment_config(
    config: Dict[str, Any], sets: Sequence[Tuple[str, List[Any]]]
) -> Dict[str, Any]:
    """Inject/override ForkParameters in every BenchmarkProblems group.

    Pure transform: mutates and returns ``config``. Raises
    :class:`ExperimentalLibraryError` if the config has no
    BenchmarkProblems/ForkParameters structure to augment.
    """
    problems = config.get("BenchmarkProblems")
    if not isinstance(problems, list) or not problems:
        raise ExperimentalLibraryError(
            "Config has no 'BenchmarkProblems' list; cannot augment ForkParameters."
        )

    touched = 0
    for group in problems:
        if not (isinstance(group, list) and len(group) >= 2 and isinstance(group[1], dict)):
            continue
        size_group = group[1]
        fork_params = size_group.get("ForkParameters")
        if not isinstance(fork_params, list):
            fork_params = []
            size_group["ForkParameters"] = fork_params
        for name, values in sets:
            _set_fork_parameter(fork_params, name, values)
        touched += 1

    if touched == 0:
        raise ExperimentalLibraryError(
            "No BenchmarkProblemSizeGroup found to augment "
            "(expected BenchmarkProblems[i][1] to be a dict)."
        )
    return config


def merge_configs(configs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine several per-solution ``extract`` configs into one config.

    Each ``extract`` of a single solution index yields a config with one
    ``BenchmarkProblems`` group. Merging concatenates those groups so a whole
    solution family (e.g. every StreamK==5 solution) rebuilds into a single
    library in one ``gen-logic``/``build-lib`` pass. The merged config flows
    through the existing ``augment``/``pipeline`` path unchanged.

    All inputs must target the same architecture (they normally come from the
    same shipped ``3_LibraryLogic``). ``GlobalParameters``/``LibraryLogic`` are
    taken from the first config; only ``BenchmarkProblems`` accumulate.
    """
    import copy

    cfgs = [c for c in configs if isinstance(c, dict)]
    if not cfgs:
        raise ExperimentalLibraryError("merge: no valid configs to combine.")

    merged = copy.deepcopy(cfgs[0])
    problems = list(merged.get("BenchmarkProblems") or [])
    base_arch = (merged.get("LibraryLogic") or {}).get("ArchitectureName")
    for c in cfgs[1:]:
        arch = (c.get("LibraryLogic") or {}).get("ArchitectureName")
        if arch != base_arch:
            raise ExperimentalLibraryError(
                f"merge: configs target different architectures ({base_arch!r} "
                f"vs {arch!r}); merge only combines configs from the same arch."
            )
        problems.extend(copy.deepcopy(c.get("BenchmarkProblems") or []))

    if not problems:
        raise ExperimentalLibraryError("merge: combined config has no BenchmarkProblems.")

    # GlobalParameters (incl. data-init types derived from the problem DataType)
    # come from the first config only. Warn if the merged groups span multiple
    # problem types, since those settings may not suit every group.
    ptypes = {
        (grp[0].get("OperationType"), grp[0].get("DataType"))
        for grp in problems
        if isinstance(grp, list) and grp and isinstance(grp[0], dict)
    }
    if len(ptypes) > 1:
        sys.stderr.write(
            "merge: warning: combined groups span multiple problem types "
            f"{sorted(map(str, ptypes))}; GlobalParameters are taken from the "
            "first config only and may not suit every group.\n"
        )

    merged["BenchmarkProblems"] = problems
    return merged


# ---------------------------------------------------------------------------
# YAML IO
# ---------------------------------------------------------------------------


def load_yaml(path: str) -> Any:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_config_with_header(config: Dict[str, Any], path: str, feature_name: Optional[str]) -> None:
    import yaml

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(_SPDX_HEADER)
        if feature_name:
            f.write(f"# Experimental feature config: {feature_name}\n")
        f.write("\n")
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Library-logic inspection
# ---------------------------------------------------------------------------


def count_solutions(logic_yaml_path: str) -> int:
    """Return the number of solution states in a ``3_LibraryLogic`` yaml.

    Uses the authoritative ``LibraryIO.rawLibraryLogic`` parser when possible
    and falls back to a structural scan of the raw yaml.
    """
    try:
        from Tensile import LibraryIO

        raw = LibraryIO.readYAML(logic_yaml_path)
        if not raw:
            return 0
        fields = LibraryIO.rawLibraryLogic(raw)
        all_solution_states = fields[5]
        return len(all_solution_states) if all_solution_states else 0
    except Exception:
        return _count_solutions_structural(logic_yaml_path)


def _count_solutions_structural(logic_yaml_path: str) -> int:
    try:
        data = load_yaml(logic_yaml_path)
    except Exception:
        return 0
    if not isinstance(data, list):
        return 0
    # The solutions list is the element that is a list of solution dicts.
    for element in data:
        if isinstance(element, list) and element and all(isinstance(e, dict) for e in element):
            if any(
                ("SolutionIndex" in e) or ("MacroTile0" in e) or ("ProblemType" in e)
                for e in element
            ):
                return len(element)
    return 0


_SUMMARY_KEYS = (
    "StreamK",
    "MatrixInstruction",
    "MIWaveTile",
    "DepthU",
    "PrefetchGlobalRead",
    "PrefetchLocalRead",
    "GlobalSplitU",
    "WorkGroupMapping",
)


def _value_eq(a: Any, b: Any) -> bool:
    """Equality that keeps ``bool`` distinct from ``int``.

    Python treats ``True == 1`` and ``False == 0``, so a naive ``in`` test would
    let ``--where Flag=1`` match a solution whose value is boolean ``True``.
    Require the same bool-ness before comparing so int and bool parameters do
    not cross-match.
    """
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def solution_matches(
    state: Dict[str, Any], wheres: Sequence[Tuple[str, List[Any]]]
) -> bool:
    """True when ``state`` satisfies every ``(name, values)`` predicate.

    AND across predicates, OR within each predicate's values. A solution that
    lacks a queried key does not match (so ``--where StreamK=5`` never matches a
    solution with no StreamK key).
    """
    for name, values in wheres:
        if name not in state or not any(_value_eq(state[name], v) for v in values):
            return False
    return True


def select_indices(
    states: Sequence[Dict[str, Any]], wheres: Sequence[Tuple[str, List[Any]]]
) -> List[int]:
    """Indices of solution states matching all ``wheres`` (all indices if none)."""
    return [
        i
        for i, s in enumerate(states)
        if isinstance(s, dict) and solution_matches(s, wheres)
    ]


def summarize_solution(state: Dict[str, Any]) -> str:
    """One-line digest of a solution's notable parameters for listing."""
    parts = [f"{k}={state[k]}" for k in _SUMMARY_KEYS if k in state]
    return " ".join(parts) if parts else "(no summary keys)"


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _indexed_out_path(out: str, idx: int, single: bool) -> str:
    """Per-index output path for ``extract``.

    A single index uses ``out`` verbatim. For multiple indices each gets a
    distinct file derived from ``out``'s stem/suffix (``base.yaml`` ->
    ``base_<idx>.yaml``; a suffix-less ``base`` -> ``base_<idx>.yaml``). This
    avoids the old ``str.replace('.yaml', ...)`` no-op that silently collided
    when ``--out`` had no ``.yaml`` suffix.
    """
    if single:
        return out
    p = Path(out)
    if p.suffix:
        return str(p.with_name(f"{p.stem}_{idx}{p.suffix}"))
    return str(p.with_name(f"{p.name}_{idx}.yaml"))


def _extract_snippet() -> str:
    return (
        "import sys\n"
        "from pathlib import Path\n"
        "from Tensile.TensileLibLogicToYaml import TensileLibLogicToYaml\n"
        "inp, out, skip = sys.argv[1], sys.argv[2], sys.argv[3] == '1'\n"
        "ids = [int(x.strip()) for x in sys.argv[4].split(',')]\n"
        "def out_for(idx):\n"
        "    if len(ids) == 1:\n"
        "        return out\n"
        "    p = Path(out)\n"
        "    if p.suffix:\n"
        "        return str(p.with_name(p.stem + '_' + str(idx) + p.suffix))\n"
        "    return str(p.with_name(p.name + '_' + str(idx) + '.yaml'))\n"
        "for idx in ids:\n"
        "    f = out_for(idx)\n"
        "    res = TensileLibLogicToYaml(inp, idx, f, skip)\n"
        "    if not res:\n"
        "        raise SystemExit(f'TensileLibLogicToYaml failed for index {idx}')\n"
        "    print(f'WROTE {f}')\n"
    )


def cmd_extract(args: argparse.Namespace) -> int:
    logic = os.path.realpath(args.logic)
    if not args.dry_run and not os.path.isfile(logic):
        raise ExperimentalLibraryError(f"Library logic file not found: {logic}")
    out = os.path.realpath(args.out)
    ids = [s.strip() for s in args.indices.split(",") if s.strip()]
    if not ids:
        raise ExperimentalLibraryError("--indices must contain at least one index")

    cmd = [
        args.python,
        "-c",
        _extract_snippet(),
        logic,
        out,
        "1" if args.skip_mi else "0",
        ",".join(ids),
    ]
    rc, output = run_command(cmd, dry_run=args.dry_run, verbose=args.verbose)
    if args.dry_run:
        return 0
    if rc != 0:
        raise ExperimentalLibraryError(
            f"extract failed (rc={rc}). Output:\n{output}"
        )

    produced = []
    single = len(ids) == 1
    for idx in ids:
        f = _indexed_out_path(out, int(idx), single)
        if not os.path.isfile(f) or os.path.getsize(f) == 0:
            raise ExperimentalLibraryError(
                f"extract reported success but config is missing/empty: {f}\n{output}"
            )
        produced.append(f)
    print("Extracted config(s):")
    for f in produced:
        print(f"  {f}")
    return 0


_INDICES_SENTINEL = "INDICES:"


def _list_snippet() -> str:
    # --indices-only output is machine-consumed (IDX=$(... --indices-only)), so
    # the index list is tagged with a sentinel and the match summary goes to
    # stderr; the parent then forwards only the sentinel payload to stdout. This
    # keeps any import-time banners on stdout/stderr from corrupting $IDX.
    return (
        "import sys\n"
        "from Tensile import LibraryIO\n"
        "from Tensile.ExperimentalLibrary import "
        "parse_set_arg, select_indices, summarize_solution\n"
        "logic, indices_only = sys.argv[1], sys.argv[2] == '1'\n"
        "wheres = [parse_set_arg(a) for a in sys.argv[3:]]\n"
        "states = LibraryIO.rawLibraryLogic(LibraryIO.readYAML(logic))[5] or []\n"
        "idxs = select_indices(states, wheres)\n"
        "if indices_only:\n"
        "    print('INDICES:' + ','.join(str(i) for i in idxs))\n"
        "else:\n"
        "    for i in idxs:\n"
        "        print(str(i) + '\\t' + summarize_solution(states[i]))\n"
        "sys.stderr.write(str(len(idxs)) + ' / ' + str(len(states)) "
        "+ ' solution(s) matched\\n')\n"
    )


def cmd_list_solutions(args: argparse.Namespace) -> int:
    logic = os.path.realpath(args.logic)
    if not args.dry_run and not os.path.isfile(logic):
        raise ExperimentalLibraryError(f"Library logic file not found: {logic}")
    # Parse --where up front so a malformed predicate fails before the subprocess.
    for w in args.where:
        parse_set_arg(w)

    cmd = [
        args.python,
        "-c",
        _list_snippet(),
        logic,
        "1" if args.indices_only else "0",
        *args.where,
    ]
    if args.dry_run or args.verbose:
        print(f"$ {_format_command(cmd, None)}")
    if args.dry_run:
        return 0

    # Capture stdout (the data) and stderr (banners + match summary) separately
    # so --indices-only is never polluted by Tensile import noise.
    proc = subprocess.run(
        list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise ExperimentalLibraryError(
            f"list-solutions failed (rc={proc.returncode}). Output:\n"
            f"{proc.stdout}{proc.stderr}"
        )

    if args.indices_only:
        payload = next(
            (
                line[len(_INDICES_SENTINEL):]
                for line in reversed(proc.stdout.splitlines())
                if line.startswith(_INDICES_SENTINEL)
            ),
            None,
        )
        if payload is None:
            raise ExperimentalLibraryError(
                f"list-solutions did not emit an index list.\n{proc.stdout}"
            )
        print(payload)
    else:
        out = proc.stdout
        sys.stdout.write(out if out.endswith("\n") or not out else out + "\n")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    out = os.path.realpath(args.out)
    if args.dry_run:
        print(f"[dry-run] would merge {len(args.configs)} config(s) into {out}")
        return 0
    configs = []
    for path in args.configs:
        if not os.path.isfile(path):
            raise ExperimentalLibraryError(f"merge: config not found: {path}")
        data = load_yaml(path)
        if not isinstance(data, dict):
            raise ExperimentalLibraryError(f"merge: {path} did not parse as a mapping.")
        configs.append(data)
    merged = merge_configs(configs)
    dump_config_with_header(merged, out, args.feature_name)
    n = len(merged.get("BenchmarkProblems") or [])
    print(
        f"merge OK: combined {len(configs)} config(s) into "
        f"{n} BenchmarkProblems group(s):\n  {out}"
    )
    return 0


def _do_augment(args: argparse.Namespace) -> str:
    """Shared augment implementation; returns the output config path."""
    sets = [parse_set_arg(s) for s in args.set]
    if not sets:
        raise ExperimentalLibraryError("At least one --set NAME=value is required")
    if not getattr(args, "skip_validation", False):
        validate_sets(sets)

    config = load_yaml(args.config)
    if not isinstance(config, dict):
        raise ExperimentalLibraryError(
            f"Config {args.config} did not parse as a mapping."
        )
    augment_config(config, sets)

    if args.out:
        out_path = os.path.realpath(args.out)
    else:
        staging = os.path.realpath(args.staging)
        base = os.path.basename(args.config)
        out_path = os.path.join(
            staging, "Logic", args.arch, "Experimental", args.name, base
        )
    dump_config_with_header(config, out_path, args.name)
    return out_path


def cmd_augment(args: argparse.Namespace) -> int:
    if not getattr(args, "name", None):
        raise ExperimentalLibraryError("augment requires --feature-name (or --name).")
    if args.dry_run:
        sets = [parse_set_arg(s) for s in args.set]
        if not args.skip_validation:
            validate_sets(sets)
        print(f"[dry-run] would inject {sets} into {args.config}")
        return 0
    out_path = _do_augment(args)
    print(f"Augmented config written to:\n  {out_path}")
    return 0


def _stage_logic(workdir: str, arch: str, feature_name: str) -> str:
    """Copy produced 3_LibraryLogic yaml(s) into an Experimental staging tree.

    Returns the staging root to pass as TensileCreateLibrary's LogicPath; the
    staged path contains an ``Experimental`` component so the library builder
    keeps the files only when ``--experimental`` is set.
    """
    import shutil

    src_dir = Path(workdir) / "3_LibraryLogic"
    logic_files = sorted(src_dir.glob("*.yaml"))
    staging_root = Path(workdir) / "experimental_logic"
    dest_dir = staging_root / arch / "Experimental" / feature_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in logic_files:
        shutil.copy2(f, dest_dir / f.name)
    return str(staging_root)


def cmd_gen_logic(args: argparse.Namespace) -> int:
    config = os.path.realpath(args.config)
    if not args.dry_run and not os.path.isfile(config):
        raise ExperimentalLibraryError(f"Config not found: {config}")
    workdir = os.path.realpath(args.out)
    if not args.dry_run:
        os.makedirs(workdir, exist_ok=True)
    feature_name = args.feature_name or "feature"

    # Fail fast (before any kernel generation) if the target arch is not a GPU
    # present on this host. gen-logic runs REAL benchmarking so winner selection
    # uses measured GFLOPS; on non-matching hardware there is no benchmark to run,
    # and falling back to uniform/synthetic GFLOPS would make every forked
    # solution tie -- the winner-take-all logic analysis then keeps only the
    # first-listed solution and silently drops the others. Refuse rather than
    # emit a misleading library. Skipped under --dry-run (must stay HW-independent).
    # NOTE: mixed-arch hosts -- we only check that the target arch is present
    # somewhere on the host, not that the benchmarked device (config `Device`)
    # is that arch. Device pinning on mixed-arch hosts is a follow-up.
    if not args.dry_run:
        from Tensile.Common.Architectures import detectHostGfxArchs, hostHasArch

        if not hostHasArch(args.arch):
            detected = detectHostGfxArchs()
            detected_str = ", ".join(detected) if detected else "none"
            raise ExperimentalLibraryError(
                f"Target arch '{args.arch}' is not present on this host "
                f"(detected GPU archs: {detected_str}). gen-logic runs real "
                "hipblaslt-bench benchmarking so winner selection uses measured "
                "GFLOPS; it will not run on non-matching hardware because there is "
                "no benchmark to run, and synthetic/uniform GFLOPS would make every "
                "forked solution tie -- the winner-take-all analysis would then keep "
                "only the first-listed solution and silently drop the rest. Run this "
                f"tool on a host with a '{args.arch}' GPU. (Cross-arch generation "
                "without benchmarking is intentionally not supported by this tool.)"
            )

    bin_tensile = _TENSILE_PKG_DIR / "bin" / "Tensile"
    if not args.dry_run and not bin_tensile.is_file():
        raise ExperimentalLibraryError(f"Tensile launcher not found: {bin_tensile}")

    cmd = [
        args.python,
        str(bin_tensile),
        config,
        workdir,
        "--gpu-targets",
        args.arch,
    ]
    env_overrides = {"CU": str(args.cu)}
    log_path = os.path.join(workdir, "gen_logic.log")
    rc, output = run_command(
        cmd,
        env_overrides=env_overrides,
        dry_run=args.dry_run,
        verbose=args.verbose,
        log_path=log_path,
    )
    if args.dry_run:
        return 0

    logic_dir = Path(workdir) / "3_LibraryLogic"
    logic_files = sorted(logic_dir.glob("*.yaml")) if logic_dir.is_dir() else []
    total_solutions = sum(count_solutions(str(f)) for f in logic_files)

    # SUCCESS: clean exit, at least one logic file, at least one counted solution.
    if rc == 0 and logic_files and total_solutions >= 1:
        staging_root = _stage_logic(workdir, args.arch, feature_name)
        print(f"gen-logic OK: {total_solutions} solution(s) across {len(logic_files)} logic file(s).")
        print(f"Tensile log: {log_path}")
        print(f"Staged experimental logic dir (pass to build-lib --logic-dir):\n  {staging_root}")
        return 0

    # rc==0 but the solution count came back 0. The all-rejected case exits
    # NON-zero upstream (BenchmarkProblems printExit on 0 valid solutions), so a
    # clean exit with present-but-uncounted logic is a parse hiccup, not a
    # rejection -- report it as its own diagnostic rather than "rejected".
    if rc == 0:
        non_empty = [f for f in logic_files if f.stat().st_size > 0]
        if non_empty:
            raise ExperimentalLibraryError(
                "gen-logic produced 3_LibraryLogic but could not be parsed to "
                f"count solutions: {non_empty[0]}\n"
                "  (clean exit, so this is inconclusive -- not a validation "
                "rejection; inspect the file and the log)\n"
                f"  Tensile log: {log_path}"
            )
        raise ExperimentalLibraryError(
            "gen-logic exited 0 but produced no 3_LibraryLogic.\n"
            f"  Tensile log: {log_path}"
        )

    # rc != 0 below. Only now consult rejection markers, and only when NO
    # 3_LibraryLogic was produced, using a TIGHT marker set so real codegen
    # crashes are not misreported as a validation rejection.
    low = output.lower()
    rejection_markers = ("0 valid solutions", "resulted in 0 valid solutions")
    if not logic_files and any(m in low for m in rejection_markers):
        raise ExperimentalLibraryError(
            "NO VALID SOLUTIONS (rejected by Solution validation): your "
            "parameter combination left 0 valid solutions. Check the "
            "constraints for the parameter(s) you toggled (e.g. "
            "StreamKForceDPOnly requires StreamK==3 and is incompatible with "
            "StreamKAtomic==1).\n"
            f"  produced logic files: {len(logic_files)}; total solutions: {total_solutions}\n"
            f"  Tensile log: {log_path}"
        )

    raise ExperimentalLibraryError(
        f"KERNEL GENERATION FAILED (codegen error): Tensile exited rc={rc}. "
        "This is a codegen error rather than a validation rejection. See the "
        f"captured log:\n  {log_path}"
    )


def cmd_build_lib(args: argparse.Namespace) -> int:
    logic_dir = os.path.realpath(args.logic_dir)
    if not args.dry_run and not os.path.isdir(logic_dir):
        raise ExperimentalLibraryError(f"Logic dir not found: {logic_dir}")
    libdir = os.path.realpath(args.out)
    if not args.dry_run:
        os.makedirs(libdir, exist_ok=True)

    cmd = [
        args.python,
        "-m",
        "Tensile.TensileCreateLibrary",
        logic_dir,
        libdir,
        "HIP",
        f"--architecture={args.arch}",
        "--code-object-version=default",
        "--library-format=msgpack",
        "--no-enumerate",
    ]
    if args.experimental:
        cmd.append("--experimental")

    log_path = os.path.join(libdir, "build_lib.log")
    rc, output = run_command(
        cmd, dry_run=args.dry_run, verbose=args.verbose, log_path=log_path
    )
    if args.dry_run:
        return 0
    if rc != 0:
        raise ExperimentalLibraryError(
            f"TensileCreateLibrary failed (rc={rc}). See log:\n  {log_path}"
        )

    # TensileCreateLibrary writes the master lazy file to a PER-ARCH subdir:
    #   <out>/library/<base-arch>/TensileLibrary_lazy_<base-arch>.dat
    # (target features after the first ':' are stripped from the dir name).
    base_arch = args.arch.split(":")[0]
    lib_root = Path(libdir) / "library"
    arch_dir = lib_root / base_arch
    # The msgpack writer always emits a zlib-compressed master file
    # (``TensileLibrary_lazy_<arch>.dat.zlib``); the runtime loader probes
    # ``.dat.zlib`` first and falls back to a plain ``.dat``. Accept either so
    # verification matches what is actually produced/loadable.
    stem = f"TensileLibrary_lazy_{base_arch}.dat"
    candidates = [arch_dir / stem, arch_dir / f"{stem}.zlib"]
    produced = next((c for c in candidates if c.is_file()), None)
    if produced is None and lib_root.is_dir():
        # Recursive fallback in case the arch dir name differs slightly.
        hits = sorted(lib_root.rglob("TensileLibrary_lazy_*.dat")) + sorted(
            lib_root.rglob("TensileLibrary_lazy_*.dat.zlib")
        )
        if hits:
            produced = hits[0]
    if produced is None or produced.stat().st_size == 0:
        raise ExperimentalLibraryError(
            f"Library build produced no '{stem}[.zlib]' (or it is empty) under "
            f"{arch_dir}. See log:\n  {log_path}"
        )

    # HIPBLASLT_TENSILE_LIBPATH is used verbatim by tensile_host.cpp (it loads
    # <env>/TensileLibrary_lazy_<arch>.dat without appending the arch subdir), so
    # it MUST point at the per-arch directory that actually holds the master file.
    libpath_dir = produced.parent
    print(f"build-lib OK. Library file: {produced}")
    print("Export this to load the library at runtime:")
    print(f"  export HIPBLASLT_TENSILE_LIBPATH={libpath_dir}")
    return 0


def _bench_problem_args(extra: List[str]) -> List[str]:
    # Strip a leading "--" separator if argparse REMAINDER captured it.
    if extra and extra[0] == "--":
        return extra[1:]
    return extra


def _resolve_lib_dir(lib: str, arch: str, *, must_exist: bool) -> str:
    """Resolve ``--lib`` to the per-arch dir holding the master lazy file.

    Accepts the build ``<out>`` root, the ``<out>/library`` dir, or the per-arch
    ``<out>/library/<base-arch>`` dir, and returns the directory that actually
    contains ``TensileLibrary_lazy_<base-arch>.dat`` -- which is what
    HIPBLASLT_TENSILE_LIBPATH must point at (tensile_host.cpp uses it verbatim).
    Searches recursively if the obvious candidates miss. When ``must_exist`` is
    False (e.g. dry-run before the library is built) a best-effort per-arch path
    is returned instead of raising.
    """
    base_arch = arch.split(":")[0]
    target = f"TensileLibrary_lazy_{base_arch}.dat"
    # The msgpack master file is canonically zlib-compressed (``.dat.zlib``);
    # the runtime loads either, so resolve against both forms.
    targets = (target, target + ".zlib")
    root = Path(os.path.realpath(lib))

    candidates = [
        root / "library" / base_arch,
        root / base_arch,
        root,
        root / "library",
    ]
    for d in candidates:
        if any((d / t).is_file() for t in targets):
            return str(d)

    if root.is_dir():
        hits = sorted(root.rglob(target)) + sorted(root.rglob(target + ".zlib"))
        if hits:
            return str(hits[0].parent)
        # Fall back to any arch's master file if the requested one is absent.
        any_hits = sorted(root.rglob("TensileLibrary_lazy_*.dat")) + sorted(
            root.rglob("TensileLibrary_lazy_*.dat.zlib")
        )
        if any_hits:
            return str(any_hits[0].parent)

    if must_exist:
        raise ExperimentalLibraryError(
            f"Could not find '{target}' under {root}. Pass --lib pointing at the "
            "build <out> root, its 'library/' dir, or the per-arch "
            "'library/<arch>' dir produced by build-lib."
        )

    # Best-effort guess for dry-run / not-yet-built libraries.
    if root.name == base_arch:
        return str(root)
    if root.name == "library":
        return str(root / base_arch)
    return str(root / "library" / base_arch)


def cmd_find_index(args: argparse.Namespace) -> int:
    # Only require the library to exist when we are actually going to run it;
    # the dry-run and the "print the command yourself" helper just need a path.
    will_execute = bool(args.bench) and not args.dry_run
    lib_subdir = _resolve_lib_dir(args.lib, args.arch, must_exist=will_execute)
    problem_args = _bench_problem_args(args.extra)

    if not args.bench:
        full = ["hipblaslt-bench", "--algo_method", "all", *problem_args]
        print("No --bench provided. Run this command yourself:")
        print(f"  HIPBLASLT_TENSILE_LIBPATH={lib_subdir} {' '.join(shlex.quote(c) for c in full)}")
        return 0

    cmd = [args.bench, "--algo_method", "all", *problem_args]
    env_overrides = {"HIPBLASLT_TENSILE_LIBPATH": lib_subdir}
    rc, output = run_command(
        cmd, env_overrides=env_overrides, dry_run=args.dry_run, verbose=args.verbose
    )
    if args.dry_run:
        return 0
    if rc != 0:
        raise ExperimentalLibraryError(
            f"hipblaslt-bench failed (rc={rc}). Output:\n{output}"
        )

    print(output)
    candidates = []
    for line in output.splitlines():
        low = line.lower()
        if "solution" in low and "index" in low:
            candidates.append(line.strip())
    print("\nCandidate solution lines:")
    if candidates:
        for c in candidates:
            print(f"  {c}")
    else:
        print("  (could not parse solution indices; inspect the full output above)")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    lib_subdir = _resolve_lib_dir(args.lib, args.arch, must_exist=not args.dry_run)
    problem_args = _bench_problem_args(args.extra)

    cmd = [
        args.bench,
        "--algo_method",
        "index",
        "--solution_index",
        str(args.solution_index),
        *problem_args,
    ]
    env_overrides = {"HIPBLASLT_TENSILE_LIBPATH": lib_subdir}
    rc, _ = run_command(
        cmd,
        env_overrides=env_overrides,
        dry_run=args.dry_run,
        verbose=args.verbose,
        stream=True,
    )
    if args.dry_run:
        return 0
    if rc != 0:
        raise ExperimentalLibraryError(f"hipblaslt-bench failed (rc={rc}).")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    out_root = os.path.realpath(args.out)
    if not args.dry_run:
        os.makedirs(out_root, exist_ok=True)

    # Stage 1: augment -> write into the Experimental staging tree.
    augment_ns = argparse.Namespace(
        config=args.config,
        set=args.set,
        name=args.feature_name,
        arch=args.arch,
        out=None,
        staging=out_root,
        skip_validation=args.skip_validation,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    if args.dry_run:
        sets = [parse_set_arg(s) for s in args.set]
        if not args.skip_validation:
            validate_sets(sets)
        print(f"[dry-run] augment: inject {sets} into {args.config}")
        augmented = os.path.join(
            out_root, "Logic", args.arch, "Experimental", args.feature_name,
            os.path.basename(args.config),
        )
    else:
        augmented = _do_augment(augment_ns)
        print(f"[1/3] augment OK -> {augmented}")

    # Stage 2: gen-logic.
    gen_workdir = os.path.join(out_root, "gen", args.feature_name)
    gen_ns = argparse.Namespace(
        config=augmented,
        arch=args.arch,
        out=gen_workdir,
        cu=args.cu,
        feature_name=args.feature_name,
        python=args.python,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    cmd_gen_logic(gen_ns)
    if not args.dry_run:
        print("[2/3] gen-logic OK")
    staging_root = os.path.join(gen_workdir, "experimental_logic")

    # Stage 3: build-lib.
    libdir = os.path.join(out_root, "lib", args.feature_name)
    build_ns = argparse.Namespace(
        logic_dir=staging_root,
        arch=args.arch,
        out=libdir,
        experimental=True,
        python=args.python,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    cmd_build_lib(build_ns)
    if not args.dry_run:
        print("[3/3] build-lib OK")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_global_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--python",
        default=None,
        help="Interpreter to drive Tensile (default: repo venv if present, else current).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands/env without executing.")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Tensile.ExperimentalLibrary",
        description="Developer tool to build experimental hipBLASLt device libraries "
        "for benchmarking new TensileLite codegen parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list-solutions
    pl = sub.add_parser(
        "list-solutions",
        help="List/filter solution indices in a 3_LibraryLogic by parameter.",
    )
    pl.add_argument("--logic", required=True, help="Shipped 3_LibraryLogic yaml.")
    pl.add_argument(
        "--where", action="append", default=[], metavar="NAME=v1[,v2]",
        help="Keep solutions whose NAME is one of the values "
        "(repeatable; AND across keys, OR within values). Values match raw "
        "solution-state shapes, e.g. StreamK=5.",
    )
    pl.add_argument(
        "--indices-only", action="store_true",
        help="Print only a comma-separated index list (feed to extract --indices).",
    )
    _add_global_flags(pl)
    pl.set_defaults(func=cmd_list_solutions)

    # extract
    pe = sub.add_parser("extract", help="Reverse a 3_LibraryLogic yaml into a benchmark config.")
    pe.add_argument("--logic", required=True, help="Shipped 3_LibraryLogic yaml.")
    pe.add_argument("--indices", default="0", help="Comma-separated solution indices, e.g. 0,3.")
    pe.add_argument("--out", required=True, help="Output config yaml path.")
    pe.add_argument("--skip-mi", action="store_true", help="Skip the MatrixInstruction field.")
    _add_global_flags(pe)
    pe.set_defaults(func=cmd_extract)

    # merge
    pm = sub.add_parser(
        "merge",
        help="Merge per-solution extract configs into one multi-problem config.",
    )
    pm.add_argument(
        "--configs", nargs="+", required=True,
        help="Config yamls to merge (e.g. base_0.yaml base_3.yaml or base_*.yaml).",
    )
    pm.add_argument("--out", required=True, help="Output merged config yaml path.")
    pm.add_argument(
        "--feature-name", default=None, help="Optional header annotation."
    )
    _add_global_flags(pm)
    pm.set_defaults(func=cmd_merge)

    # augment
    pa = sub.add_parser("augment", help="Validate and inject --set params into ForkParameters.")
    pa.add_argument("--config", required=True, help="Base benchmark config yaml.")
    pa.add_argument(
        "--set", action="append", default=[], metavar="NAME=v1[,v2]",
        help="Parameter to toggle (repeatable).",
    )
    pa.add_argument(
        "--feature-name", dest="name", default=None,
        help="Feature name (staging dir component).",
    )
    # Back-compat hidden alias for --feature-name.
    pa.add_argument("--name", dest="name", default=None, help=argparse.SUPPRESS)
    pa.add_argument("--arch", default="gfx950", help="Target arch for staging path (default gfx950).")
    pa.add_argument("--out", default=None, help="Explicit output path (overrides staging layout).")
    pa.add_argument(
        "--staging", default="experimental_staging",
        help="Staging root when --out is not given.",
    )
    pa.add_argument(
        "--skip-validation", action="store_true",
        help="Skip --set validation against validParameters (no rocisa needed; "
        "pure config editing).",
    )
    _add_global_flags(pa)
    pa.set_defaults(func=cmd_augment)

    # gen-logic
    pg = sub.add_parser("gen-logic", help="Benchmark on the target-arch GPU and validate the produced logic (requires that GPU present).")
    pg.add_argument("--config", required=True, help="Augmented benchmark config yaml.")
    pg.add_argument("--arch", default="gfx950", help="GPU target (default gfx950).")
    pg.add_argument("--out", required=True, help="Work directory for Tensile output.")
    pg.add_argument("--cu", type=int, default=304, help="CU count pinned via the CU env for the build predicate.")
    pg.add_argument("--feature-name", default=None, help="Feature name for the Experimental dir.")
    _add_global_flags(pg)
    pg.set_defaults(func=cmd_gen_logic)

    # build-lib
    pb = sub.add_parser("build-lib", help="Run TensileCreateLibrary to build a loadable library.")
    pb.add_argument("--logic-dir", required=True, help="Dir containing Experimental/... logic.")
    pb.add_argument("--arch", default="gfx950", help="GPU target (default gfx950).")
    pb.add_argument("--out", required=True, help="Output library directory.")
    pb.add_argument(
        "--experimental", dest="experimental", action="store_true", default=True,
        help="Include Experimental/ logic (default on).",
    )
    pb.add_argument(
        "--no-experimental", dest="experimental", action="store_false",
        help="Disable Experimental/ inclusion.",
    )
    _add_global_flags(pb)
    pb.set_defaults(func=cmd_build_lib)

    # find-index
    pf = sub.add_parser(
        "find-index",
        help="Discover solution indices via hipblaslt-bench --algo_method all.",
    )
    pf.add_argument(
        "--lib", required=True,
        help="Build <out> root, its 'library/' dir, or the per-arch "
        "'library/<arch>' dir.",
    )
    pf.add_argument("--arch", default="gfx950", help="GPU target (default gfx950).")
    pf.add_argument("--bench", default=None, help="Path to hipblaslt-bench.")
    pf.add_argument("extra", nargs=argparse.REMAINDER, help="Problem args after --.")
    _add_global_flags(pf)
    pf.set_defaults(func=cmd_find_index)

    # bench
    pbn = sub.add_parser("bench", help="Run hipblaslt-bench against a specific solution index.")
    pbn.add_argument(
        "--lib", required=True,
        help="Build <out> root, its 'library/' dir, or the per-arch "
        "'library/<arch>' dir.",
    )
    pbn.add_argument("--arch", default="gfx950", help="GPU target (default gfx950).")
    pbn.add_argument("--bench", required=True, help="Path to hipblaslt-bench.")
    pbn.add_argument("--solution-index", required=True, type=int, help="Solution index to run.")
    pbn.add_argument("extra", nargs=argparse.REMAINDER, help="Problem args after --.")
    _add_global_flags(pbn)
    pbn.set_defaults(func=cmd_bench)

    # pipeline
    pp = sub.add_parser("pipeline", help="Chain augment -> gen-logic -> build-lib.")
    pp.add_argument("--config", required=True, help="Base benchmark config yaml.")
    pp.add_argument(
        "--set", action="append", default=[], metavar="NAME=v1[,v2]",
        help="Parameter to toggle (repeatable).",
    )
    pp.add_argument("--feature-name", required=True, help="Feature name.")
    pp.add_argument("--arch", default="gfx950", help="GPU target (default gfx950).")
    pp.add_argument("--cu", type=int, default=304, help="CU count pinned via the CU env for the build predicate.")
    pp.add_argument("--out", required=True, help="Output root for all stages.")
    pp.add_argument(
        "--skip-validation", action="store_true",
        help="Skip --set validation against validParameters (no rocisa needed).",
    )
    _add_global_flags(pp)
    pp.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "python", None) is None:
        args.python = default_python()
    try:
        return args.func(args)
    except ExperimentalLibraryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
