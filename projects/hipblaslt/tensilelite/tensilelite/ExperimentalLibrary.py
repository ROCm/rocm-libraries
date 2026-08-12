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
             Filter the solutions in one or more shipped ``3_LibraryLogic``
             yaml file(s)/dir(s) by parameter (e.g. ``--where StreamK=5``) to
             discover which indices to extract.
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
             rejected by Solution validation vs. an unedited placeholder
             ``ProblemSizes`` left over from extracting a ``Prediction``-type
             source. Benchmarks on the target-arch GPU (fails fast if that
             arch is not present on the host).
  build-lib  Run ``Tensile.TensileCreateLibrary --experimental`` to turn the
             staged logic into a loadable device library.
  patch-logic
             Override ``--set`` params on shipped ``3_LibraryLogic`` solutions
             (optionally narrowed by ``--where``) and build a matched
             baseline/patched library pair for an A/B experiment -- no
             re-benchmark (works for Origami/learned-rule logic that
             ``gen-logic`` cannot reproduce).
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
      --set StreamK=3 --feature-name streamk_treereduce \\
      --arch gfx950 --cu 256 --out work/

A/B family example -- rebuild every StreamK==5 solution from a shipped logic
with a new feature toggled OFF *and* ON inside a single library, so the two can
be contrasted by solution index (StreamKWorkStealing is illustrative; use any
real parameter, and --skip-validation for a parameter not yet in the registry):

  IDX=$(python -m Tensile.ExperimentalLibrary list-solutions \\
      --logic-src <shipped>/.../<liblogic>.yaml --where StreamK=5 --indices-only)
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

Origami/Prediction-logic A/B example -- toggle a parameter on every matching
solution across several shipped files (e.g. one per transpose) and get back
two libraries that differ only in that parameter, with no re-benchmark:

  python -m Tensile.ExperimentalLibrary patch-logic \\
      --logic-src <shipped>/gfx950/.../gfx950_Cijk_Ailk_Bjlk_*.yaml \\
                  <shipped>/gfx950/.../gfx950_Cijk_Ailk_Bljk_*.yaml \\
      --where StreamK=5 --set PrefetchAcrossPersistent=1 \\
      --matched-pair --skip-unbuildable \\
      --arch gfx950 --out work/pap --feature-name sk5_pap1
      # prints both HIPBLASLT_TENSILE_LIBPATH exports and writes
      # work/pap/patch_manifest.csv (per-solution applied/skipped + reason)

``--where`` is optional: dropping it from the command above selects every
solution under ``--logic-src`` instead of just the ``StreamK=5`` family, so
the same ``--set`` is applied library-wide. ``--set`` itself is always
required.
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


_PLACEHOLDER_PROBLEM_SIZES = [{"Exact": [1, 1, 1, 1]}]


def _placeholder_problem_size_groups(config: Dict[str, Any]) -> List[int]:
    """Indices (within ``BenchmarkProblems``) of groups whose ``ProblemSizes``
    is still ``TensileLibLogicToYaml``'s placeholder (``[{'Exact': [1, 1, 1,
    1]}]``), emitted when ``extract`` has no real per-size table to recover
    from a ``Prediction``-type source and left unedited by the user.
    """
    placeholder_groups: List[int] = []
    for i, group in enumerate(config.get("BenchmarkProblems") or []):
        if not (isinstance(group, list) and len(group) >= 2 and isinstance(group[1], dict)):
            continue
        for params in group[1].get("BenchmarkFinalParameters") or []:
            if isinstance(params, dict) and params.get("ProblemSizes") == _PLACEHOLDER_PROBLEM_SIZES:
                placeholder_groups.append(i)
                break
    return placeholder_groups


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


def _scan_for_solutions_element(
    data: Any,
) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
    """Locate the solution-states element inside a parsed ``3_LibraryLogic``.

    A shipped logic yaml parses to a raw LIST, one element of which is itself a
    list of solution-state dicts. Returns ``(index_in_data, that_list)``, or
    ``None`` if ``data`` is not a list or no such element is found.
    """
    if not isinstance(data, list):
        return None
    for i, element in enumerate(data):
        if isinstance(element, list) and element and all(isinstance(e, dict) for e in element):
            if any(
                ("SolutionIndex" in e) or ("MacroTile0" in e) or ("ProblemType" in e)
                for e in element
            ):
                return i, element
    return None


def _library_type(raw: Any) -> Optional[str]:
    """Read the LibraryType discriminator at position 11 of a parsed
    ``3_LibraryLogic`` list, if present (mirrors ``LibraryIO.parseLibraryLogicList``).
    """
    if isinstance(raw, list) and len(raw) > 11 and raw[11]:
        return raw[11]
    return None


def _count_solutions_structural(logic_yaml_path: str) -> int:
    try:
        data = load_yaml(logic_yaml_path)
    except Exception:
        return 0
    found = _scan_for_solutions_element(data)
    return len(found[1]) if found else 0


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


def _dedup_keys(*key_lists: Sequence[str]) -> List[str]:
    """Flatten several key lists into one, keeping first occurrence order."""
    seen = set()
    result = []
    for keys in key_lists:
        for k in keys:
            if k not in seen:
                seen.add(k)
                result.append(k)
    return result


def summarize_solution(state: Dict[str, Any], extra_keys: Sequence[str] = ()) -> str:
    """One-line digest of a solution's notable parameters for listing.

    ``extra_keys`` (e.g. the active ``--where`` names) are shown first, followed
    by ``_SUMMARY_KEYS``, de-duplicated; only keys present in ``state`` appear.
    """
    keys = _dedup_keys(extra_keys, _SUMMARY_KEYS)
    parts = [f"{k}={state[k]}" for k in keys if k in state]
    return " ".join(parts) if parts else "(no summary keys)"


def _resolve_logic_sources(sources: Sequence[str]) -> List[str]:
    """Expand ``--logic-src`` entries (files and/or dirs) into logic yaml paths.

    A directory contributes every ``*.yaml`` found recursively within it. The
    result is de-duplicated (by realpath) while preserving first-seen order.
    """
    files: List[str] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            files.extend(str(f) for f in sorted(p.rglob("*.yaml")))
        elif p.is_file():
            files.append(str(p))
        else:
            raise ExperimentalLibraryError(
                f"--logic-src not found (no such file or dir): {src}"
            )
    resolved: List[str] = []
    seen = set()
    for f in files:
        rp = os.path.realpath(f)
        if rp not in seen:
            seen.add(rp)
            resolved.append(rp)
    if not resolved:
        raise ExperimentalLibraryError(
            "--logic-src resolved to no *.yaml logic files."
        )
    return resolved


def _find_solutions_element(raw: Any, path: str) -> Tuple[int, List[Dict[str, Any]]]:
    """Locate the solution-states element inside a parsed ``3_LibraryLogic``.

    Uses the same detection as :func:`_count_solutions_structural` (via
    :func:`_scan_for_solutions_element`), but on an already-parsed ``raw``
    object, and raises instead of silently returning a fallback.
    """
    if not isinstance(raw, list):
        raise ExperimentalLibraryError(
            f"{path} did not parse as a LibraryLogic list."
        )
    found = _scan_for_solutions_element(raw)
    if found is None:
        raise ExperimentalLibraryError(
            f"could not find a solution-states list in {path}."
        )
    return found


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _indexed_out_path(out: str, idx: int, single: bool) -> str:
    """Per-index output path for ``extract``.

    A single index uses ``out`` verbatim. For multiple indices each gets a
    distinct file derived from ``out``'s stem/suffix (``base.yaml`` ->
    ``base_<idx>.yaml``; a suffix-less ``base`` -> ``base_<idx>.yaml``), so a
    suffix-less ``--out`` still yields one distinct file per index instead of
    every index writing the same path.
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
        "from Tensile.TensileLibLogicToYaml import TensileLibLogicToYaml\n"
        "from Tensile.ExperimentalLibrary import _indexed_out_path\n"
        "inp, out, skip = sys.argv[1], sys.argv[2], sys.argv[3] == '1'\n"
        "ids = [int(x.strip()) for x in sys.argv[4].split(',')]\n"
        "for idx in ids:\n"
        "    f = _indexed_out_path(out, idx, len(ids) == 1)\n"
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

    # A structural fact about the source file, independent of whether the
    # subprocess actually runs, so this check applies under --dry-run too
    # (skipped only if the file doesn't exist yet).
    if os.path.isfile(logic) and _library_type(load_yaml(logic)) == "Prediction":
        sys.stderr.write(
            f"extract: warning: {logic} is a 'Prediction'-type library logic "
            "with no real per-size table; the extracted config's ProblemSizes "
            "will be a placeholder ([{'Exact': [1, 1, 1, 1]}]) that must be "
            "hand-edited to real sizes before benchmarking with gen-logic. To "
            "toggle a parameter and rebuild the same library instead, use "
            "patch-logic.\n"
        )

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


def cmd_list_solutions(args: argparse.Namespace) -> int:
    wheres = [parse_set_arg(w) for w in args.where]
    files = _resolve_logic_sources(args.logic_src)

    # Indices are per-file (not global), so a flat comma-separated list is only
    # unambiguous when exactly one logic file is in play.
    if args.indices_only and len(files) > 1:
        raise ExperimentalLibraryError(
            f"list-solutions: --logic-src resolved {len(files)} logic files, "
            "but --indices-only requires exactly one (its output is a single "
            "file's index list, and indices are not comparable across files); "
            "narrow --logic-src to one file."
        )

    per_file: List[Tuple[str, List[Dict[str, Any]], List[int], Optional[str]]] = []
    for path in files:
        raw = load_yaml(path)
        _, states = _find_solutions_element(raw, path)
        idxs = select_indices(states, wheres)
        per_file.append((path, states, idxs, _library_type(raw)))

    if args.indices_only:
        _, _, idxs, _ = per_file[0]
        print(",".join(str(i) for i in idxs))
        return 0

    for path, states, idxs, library_type in per_file:
        print(f"== {path} [{library_type or 'unknown'}] ({len(idxs)}/{len(states)} match) ==")
        for i in idxs:
            print(f"{i}\t{summarize_solution(states[i], extra_keys=[name for name, _ in wheres])}")

    total_matched = sum(len(idxs) for _, _, idxs, _ in per_file)
    total_states = sum(len(states) for _, states, _, _ in per_file)
    if len(per_file) > 1:
        print(
            f"list-solutions: {total_matched}/{total_states} solution(s) "
            f"matched across {len(per_file)} file(s)."
        )
    else:
        print(f"list-solutions: {total_matched}/{total_states} solution(s) matched.")
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


def _augment_staging_path(staging: str, arch: str, feature_name: str, config_basename: str) -> str:
    """Default augment output path when ``--out`` is not given."""
    return os.path.join(staging, "Logic", arch, "Experimental", feature_name, config_basename)


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
        out_path = _augment_staging_path(staging, args.arch, args.name, base)
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


def _experimental_staging_path(root: str, arch: str, feature_name: str, filename: str) -> str:
    """Build the ``<root>/<arch>/Experimental/<feature_name>/<filename>`` staging
    path shared by gen-logic's output staging and patch-logic's logic trees.
    """
    return os.path.join(root, arch, "Experimental", feature_name, filename)


def _stage_logic(workdir: str, arch: str, feature_name: str) -> str:
    """Copy produced 3_LibraryLogic yaml(s) into an Experimental staging tree.

    Returns the staging root to pass as TensileCreateLibrary's LogicPath; the
    staged path contains an ``Experimental`` component so the library builder
    keeps the files only when ``--experimental`` is set.
    """
    import shutil

    src_dir = Path(workdir) / "3_LibraryLogic"
    logic_files = sorted(src_dir.glob("*.yaml"))
    staging_root = str(Path(workdir) / "experimental_logic")
    dest_dir = Path(_experimental_staging_path(staging_root, arch, feature_name, ""))
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in logic_files:
        dest = _experimental_staging_path(staging_root, arch, feature_name, f.name)
        shutil.copy2(f, dest)
    return staging_root


def cmd_gen_logic(args: argparse.Namespace) -> int:
    config = os.path.realpath(args.config)
    if not args.dry_run and not os.path.isfile(config):
        raise ExperimentalLibraryError(f"Config not found: {config}")
    workdir = os.path.realpath(args.out)
    if not args.dry_run:
        os.makedirs(workdir, exist_ok=True)
    feature_name = args.feature_name or "feature"

    # A structural fact about the config file's contents; skipped under
    # --dry-run when the config doesn't exist yet (e.g. a pipeline dry-run
    # whose augment stage hasn't actually written it).
    if os.path.isfile(config):
        placeholder_groups = _placeholder_problem_size_groups(load_yaml(config))
        if placeholder_groups:
            raise ExperimentalLibraryError(
                f"Config {config} has BenchmarkProblems group(s) "
                f"{placeholder_groups} with an unedited placeholder "
                f"ProblemSizes ({_PLACEHOLDER_PROBLEM_SIZES}), left over from "
                "extracting a 'Prediction'-type source with no real per-size "
                "table. Set real ProblemSizes for those group(s) before "
                "benchmarking, or use patch-logic instead to toggle a "
                "parameter and rebuild the same library without benchmarking."
            )

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


def _tensile_create_library_cmd(
    python: str,
    logic_dir: str,
    out_dir: str,
    arch: str,
    *,
    experimental: bool = True,
    jobs: Optional[int] = None,
) -> List[str]:
    """Build the ``python -m Tensile.TensileCreateLibrary ...`` argv shared by
    build-lib and patch-logic's buildability probe.
    """
    cmd = [
        python,
        "-m",
        "Tensile.TensileCreateLibrary",
        logic_dir,
        out_dir,
        "HIP",
        f"--architecture={arch}",
        "--code-object-version=default",
        "--library-format=msgpack",
        "--no-enumerate",
    ]
    if experimental:
        cmd.append("--experimental")
    if jobs is not None:
        cmd += ["--jobs", str(jobs)]
    return cmd


def cmd_build_lib(args: argparse.Namespace) -> int:
    logic_dir = os.path.realpath(args.logic_dir)
    if not args.dry_run and not os.path.isdir(logic_dir):
        raise ExperimentalLibraryError(f"Logic dir not found: {logic_dir}")
    libdir = os.path.realpath(args.out)
    if not args.dry_run:
        os.makedirs(libdir, exist_ok=True)

    cmd = _tensile_create_library_cmd(
        args.python,
        logic_dir,
        libdir,
        args.arch,
        experimental=args.experimental,
        jobs=getattr(args, "jobs", None),
    )

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
        augmented = _augment_staging_path(
            out_root, args.arch, args.feature_name, os.path.basename(args.config)
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
        jobs=None,
    )
    cmd_build_lib(build_ns)
    if not args.dry_run:
        print("[3/3] build-lib OK")
    return 0


# ---------------------------------------------------------------------------
# patch-logic: override params on matching shipped-logic solutions and build a
# matched baseline/patched library pair (no re-benchmark).
# ---------------------------------------------------------------------------


# Diagnostic-context columns for patch_manifest.csv, appended after whatever
# --set/--where names the run actually queried (see cmd_patch_logic).
_PATCH_MANIFEST_PARAMS = (
    "StreamK",
    "MacroTile0",
    "MacroTile1",
    "DepthU",
    "PrefetchGlobalRead",
    "1LDSBuffer",
    "DirectToLdsA",
    "DirectToLdsB",
)

# A probe builds exactly one solution, so its internal TensileCreateLibrary -j
# gains nothing from scaling with probe concurrency (that would multiply out
# to concurrency * this many processes).
_PROBE_INTERNAL_JOBS = 1


def _apply_overrides(state: Dict[str, Any], sets: Sequence[Tuple[str, List[Any]]]) -> None:
    """Set each ``NAME=value`` override directly on a solution-state dict.

    Each ``--set`` carries exactly one value (enforced by the caller), so the
    scalar is written straight onto the state; rebuilding re-derives the kernel
    because ``solutionStateToSolution`` clears the derived-parameter flags.
    """
    for name, values in sets:
        state[name] = values[0]


def _unique_staged_name(base: str, assigned: set) -> str:
    """Pick a collision-free staged filename for a source basename."""
    if base not in assigned:
        assigned.add(base)
        return base
    stem, dot, ext = base.rpartition(".")
    n = 1
    while True:
        cand = f"{stem}_{n}.{ext}" if dot else f"{base}_{n}"
        if cand not in assigned:
            assigned.add(cand)
            return cand
        n += 1


def _probe_override(
    args: argparse.Namespace,
    raw: Any,
    sol_pos: int,
    override_state: Dict[str, Any],
    probe_dir: str,
) -> Tuple[bool, str]:
    """Build a single-solution logic (with the override) to test buildability.

    Writes ``raw`` with its solution element replaced by ``[override_state]``
    into an ``Experimental`` staging tree, then runs TensileCreateLibrary and
    classifies the outcome. Returns ``(passed, reason)``: ``reason`` is empty on
    success, else ``reject`` / ``codegen`` / ``other`` from scanning output.
    """
    import copy

    probe_raw = copy.deepcopy(raw)
    probe_raw[sol_pos] = [copy.deepcopy(override_state)]
    # The truncated one-solution list invalidates any winner table that
    # references other positions; force FreeSize's trivial [0, len(solutions)]
    # dispatch instead of carrying over the source's real indexOrder/
    # exactLogic/rangeLogic/LibraryType, which would otherwise be
    # inconsistent with the truncated list.
    for i in (6, 7, 8):  # indexOrder, exactLogic, rangeLogic
        if i < len(probe_raw):
            probe_raw[i] = None
    if len(probe_raw) > 11:
        probe_raw[11] = "FreeSize"
    logic_root = Path(probe_dir) / "logic"
    dest = _experimental_staging_path(
        str(logic_root), args.arch, args.feature_name, "probe.yaml"
    )
    dump_config_with_header(probe_raw, dest, args.feature_name)
    out_dir = Path(probe_dir) / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = _tensile_create_library_cmd(
        args.python,
        str(logic_root),
        str(out_dir),
        args.arch,
        experimental=True,
        jobs=_PROBE_INTERNAL_JOBS,
    )

    log_path = str(Path(probe_dir) / "probe.log")
    # The caller only invokes _probe_override when not args.dry_run, so this
    # always runs for real.
    rc, output = run_command(cmd, verbose=args.verbose, log_path=log_path)
    if rc == 0:
        return True, ""
    low = output.lower()
    if "rejection of a librarylogic" in low:
        return False, "reject"
    if "resulted in error" in low:
        return False, "codegen"
    return False, "other"


def cmd_patch_logic(args: argparse.Namespace) -> int:
    import copy
    import csv
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    if not args.feature_name:
        raise ExperimentalLibraryError("patch-logic requires --feature-name.")
    wheres = [parse_set_arg(w) for w in args.where]
    if not wheres:
        sys.stderr.write(
            "patch-logic: warning: no --where given; --set will be applied to "
            "every solution in --logic-src.\n"
        )
    sets = [parse_set_arg(s) for s in args.set]
    if not sets:
        raise ExperimentalLibraryError(
            "patch-logic requires at least one --set NAME=value."
        )
    for name, values in sets:
        if len(values) != 1:
            raise ExperimentalLibraryError(
                f"patch-logic --set '{name}' takes exactly one value (got "
                f"{values}); patch-logic writes a single scalar per parameter. "
                "Use --matched-pair to contrast this value against baseline."
            )
    if not args.skip_validation:
        validate_sets(sets)

    # Manifest columns: the --set/--where names the user actually queried come
    # first (what they care about), _PATCH_MANIFEST_PARAMS fills in the rest as
    # diagnostic context, with no duplicate columns for overlapping names.
    manifest_params = _dedup_keys(
        [name for name, _ in sets], [name for name, _ in wheres], _PATCH_MANIFEST_PARAMS
    )

    files = _resolve_logic_sources(args.logic_src)
    out_root = os.path.realpath(args.out)

    # Pass 1: parse each file, locate its solution list, and select matches.
    # ``raw`` here is the untouched BASELINE for each file; the patched copy is
    # derived later so both trees stay identical except for applied overrides.
    file_infos: List[Dict[str, Any]] = []
    assigned_names: set = set()
    total_matched = 0
    for path in files:
        raw = load_yaml(path)
        sol_pos, solutions = _find_solutions_element(raw, path)
        matched = [
            i
            for i, s in enumerate(solutions)
            if isinstance(s, dict) and solution_matches(s, wheres)
        ]
        total_matched += len(matched)
        staged_name = _unique_staged_name(os.path.basename(path), assigned_names)
        file_infos.append(
            {
                "path": path,
                "raw": raw,
                "sol_pos": sol_pos,
                "solutions": solutions,
                "matched": matched,
                "staged_name": staged_name,
            }
        )
        print(
            f"patch-logic: {path}: {len(matched)}/{len(solutions)} solution(s) "
            f"match --where"
        )

    if total_matched == 0:
        sys.stderr.write(
            "patch-logic: warning: no solutions matched --where; the patched "
            "library will be identical to baseline.\n"
        )

    # Pass 2: decide, per matching solution, whether the override is applied.
    # Default: apply to all matches. With --skip-unbuildable, probe each match
    # (single-solution build) and skip the override on any that fail, keeping
    # that solution as baseline in BOTH libraries so dispatch never diverges.
    probe_root = os.path.join(out_root, "_probe")
    if args.skip_unbuildable and not args.dry_run:
        os.makedirs(probe_root, exist_ok=True)
        tasks = []  # (fi_index, sol_index, override_state, probe_dir)
        for fi_idx, fi in enumerate(file_infos):
            for pos in fi["matched"]:
                override_state = copy.deepcopy(fi["solutions"][pos])
                _apply_overrides(override_state, sets)
                probe_dir = os.path.join(probe_root, f"{fi_idx}_{pos}")
                tasks.append((fi_idx, pos, override_state, probe_dir))
        results: Dict[Tuple[int, int], Tuple[bool, str]] = {}
        max_workers = args.jobs if (args.jobs and args.jobs > 0) else (os.cpu_count() or 1)
        if tasks:
            print(
                f"patch-logic: probing {len(tasks)} override(s) for buildability "
                f"(up to {max_workers} concurrent)..."
            )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _probe_override,
                    args,
                    file_infos[fi_idx]["raw"],
                    file_infos[fi_idx]["sol_pos"],
                    override_state,
                    probe_dir,
                ): (fi_idx, pos)
                for (fi_idx, pos, override_state, probe_dir) in tasks
            }
            for fut, key in futures.items():
                results[key] = fut.result()
        # Attach per-solution decisions.
        for fi_idx, fi in enumerate(file_infos):
            decisions = {}
            for pos in fi["matched"]:
                passed, reason = results.get((fi_idx, pos), (True, ""))
                decisions[pos] = (passed, reason)
            fi["decisions"] = decisions
        shutil.rmtree(probe_root, ignore_errors=True)
    else:
        # No probing: every match gets the override (a build failure surfaces
        # directly). Dry-run also lands here and assumes all-applied for planning.
        for fi in file_infos:
            fi["decisions"] = {pos: (True, "") for pos in fi["matched"]}

    # Pass 3: emit the patched (and, for --matched-pair, baseline) logic trees.
    patched_root = os.path.join(out_root, "patched_logic")
    baseline_root = os.path.join(out_root, "baseline_logic")
    manifest_rows: List[List[Any]] = []
    applied_total = 0
    skipped_total = 0
    for fi in file_infos:
        raw = fi["raw"]
        sol_pos = fi["sol_pos"]
        staged_name = fi["staged_name"]
        patched_raw = copy.deepcopy(raw)
        patched_solutions = patched_raw[sol_pos]
        for pos in fi["matched"]:
            passed, reason = fi["decisions"][pos]
            base_state = fi["solutions"][pos]
            if passed:
                _apply_overrides(patched_solutions[pos], sets)
                applied_total += 1
                status, row_reason = "applied", ""
            else:
                skipped_total += 1
                status, row_reason = "skipped", reason
            sol_index = base_state.get("SolutionIndex", pos)
            manifest_rows.append(
                [fi["path"], sol_index, status, row_reason]
                + [base_state.get(k, "") for k in manifest_params]
            )

        if not args.dry_run:
            patched_dest = _experimental_staging_path(
                patched_root, args.arch, args.feature_name, staged_name
            )
            dump_config_with_header(patched_raw, patched_dest, args.feature_name)
            if args.matched_pair:
                baseline_dest = _experimental_staging_path(
                    baseline_root, args.arch, args.feature_name, staged_name
                )
                dump_config_with_header(raw, baseline_dest, args.feature_name)

    print(
        f"patch-logic: {applied_total} override(s) applied, "
        f"{skipped_total} skipped across {len(file_infos)} file(s)."
    )

    # Manifest CSV (one row per matching solution).
    if not args.dry_run:
        manifest_path = os.path.join(out_root, "patch_manifest.csv")
        os.makedirs(out_root, exist_ok=True)
        with open(manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["logic_file", "solution_index", "status", "reason"] + manifest_params
            )
            writer.writerows(manifest_rows)
        print(f"patch-logic: manifest written to {manifest_path}")

    # Pass 4: build the loadable library/libraries via the existing build path.
    def _build_tree(logic_root: str, lib_out: str, label: str) -> str:
        print(f"=== patch-logic: building {label} library ===")
        ns = argparse.Namespace(
            logic_dir=logic_root,
            arch=args.arch,
            out=lib_out,
            experimental=True,
            python=args.python,
            dry_run=args.dry_run,
            verbose=args.verbose,
            jobs=args.jobs,
        )
        cmd_build_lib(ns)
        return _resolve_lib_dir(lib_out, args.arch, must_exist=not args.dry_run)

    patched_libpath = _build_tree(
        patched_root, os.path.join(out_root, "patched"), "PATCHED"
    )
    baseline_libpath = None
    if args.matched_pair:
        baseline_libpath = _build_tree(
            baseline_root, os.path.join(out_root, "baseline"), "BASELINE"
        )

    print("\npatch-logic OK. Export to load a library at runtime:")
    print(f"  PATCHED : export HIPBLASLT_TENSILE_LIBPATH={patched_libpath}")
    if baseline_libpath is not None:
        print(f"  BASELINE: export HIPBLASLT_TENSILE_LIBPATH={baseline_libpath}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_global_flags(p: argparse.ArgumentParser, *, include_python: bool = True) -> None:
    if include_python:
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
        help="List/filter solution indices in 3_LibraryLogic yaml file(s)/dir(s) by parameter.",
    )
    pl.add_argument(
        "--logic-src", nargs="+", required=True,
        help="Shipped 3_LibraryLogic yaml file(s) and/or dir(s) (dirs are "
        "globbed for *.yaml recursively).",
    )
    pl.add_argument(
        "--where", action="append", default=[], metavar="NAME=v1[,v2]",
        help="Keep solutions whose NAME is one of the values "
        "(repeatable; AND across keys, OR within values). Values match raw "
        "solution-state shapes, e.g. StreamK=5.",
    )
    pl.add_argument(
        "--indices-only", action="store_true",
        help="Print only a comma-separated index list for the single resolved "
        "logic file (feed to extract --indices); errors if --logic-src "
        "resolves to more than one file, since indices are per-file.",
    )
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
    pm.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be merged without writing the output.",
    )
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
    pa.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be injected without writing the output.",
    )
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
    pb.add_argument(
        "--jobs", "-j", dest="jobs", type=int, default=None,
        help="Parallel jobs for TensileCreateLibrary (passed through as -j).",
    )
    _add_global_flags(pb)
    pb.set_defaults(func=cmd_build_lib)

    # patch-logic
    ppl = sub.add_parser(
        "patch-logic",
        help="Override --set params on shipped-logic solutions (optionally "
        "narrowed by --where) and build a matched baseline/patched library "
        "pair (no re-benchmark).",
    )
    ppl.add_argument(
        "--logic-src", nargs="+", required=True,
        help="Shipped 3_LibraryLogic yaml file(s) and/or dir(s) (dirs are "
        "globbed for *.yaml recursively).",
    )
    ppl.add_argument(
        "--where", action="append", default=[], metavar="NAME=v1[,v2]",
        help="Select solutions to patch (repeatable; AND across keys, OR within "
        "values), e.g. --where StreamK=5. Omit to select every solution in "
        "--logic-src.",
    )
    ppl.add_argument(
        "--set", action="append", default=[], required=True, metavar="NAME=value",
        help="Override to apply to matching solutions (repeatable; one value "
        "each), e.g. --set PrefetchAcrossPersistent=1.",
    )
    ppl.add_argument(
        "--matched-pair", action="store_true",
        help="Also emit an unmodified baseline library alongside the patched "
        "one for A/B comparison.",
    )
    ppl.add_argument(
        "--skip-unbuildable", action="store_true",
        help="Probe each override; if a solution's override fails to build, skip "
        "the override (keep it as baseline in both libraries) instead of aborting.",
    )
    ppl.add_argument("--arch", default="gfx950", help="GPU target (default gfx950).")
    ppl.add_argument("--out", required=True, help="Output root for logic/libraries/manifest.")
    ppl.add_argument("--feature-name", required=True, help="Experimental staging dir component.")
    ppl.add_argument(
        "--jobs", "-j", dest="jobs", type=int, default=None,
        help="Parallel jobs: TensileCreateLibrary -j for the final build(s), and "
        "max concurrent probes with --skip-unbuildable (each probe builds a "
        "single solution regardless of this value). Default: all CPUs for "
        "probing, whatever TensileCreateLibrary's own default is for builds.",
    )
    ppl.add_argument(
        "--skip-validation", action="store_true",
        help="Skip --set validation against validParameters (no rocisa needed).",
    )
    _add_global_flags(ppl)
    ppl.set_defaults(func=cmd_patch_logic)

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
    _add_global_flags(pf, include_python=False)
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
    _add_global_flags(pbn, include_python=False)
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
