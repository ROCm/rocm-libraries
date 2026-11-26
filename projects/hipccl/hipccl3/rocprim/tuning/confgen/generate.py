#!/usr/bin/env python3

import argparse
import csv
import datetime
import glob
import json
import logging
import math
import os
import pathlib
import re
import subprocess

import typing_extensions as ty
import jinja2

log = logging.getLogger("confgen.generate")

# Used when no block_sort config is found for a block_merge type.
MERGE_SORT_DEFAULT_SORTED_TILE = 1024

# Benchmarks whose name differs from the tuned algorithm (see benchmark/CMakeLists.txt).
BENCHMARK_NAMES = {
    "device_adjacent_difference_inplace": ["device_adjacent_difference"],
    "device_lower_bound": ["device_binary_search"],
    "device_upper_bound": ["device_binary_search"],
    "device_run_length_encode_non_trivial": [
        "device_run_length_encode_non_trivial_runs"
    ],
    "device_segmented_radix_sort": [
        "device_segmented_radix_sort_keys",
        "device_segmented_radix_sort_pairs",
    ],
}


def round_up(value, alignment):
    """Rounds value up to a multiple of alignment."""
    return -(-value // alignment) * alignment


def annotate_type(type_id, type_name):
    """Creates annotations for types to simplify jinja templating."""
    is_floating = False
    is_arithmetic = True
    size = 0
    size_min = None

    # Floating point types
    if type_id == "rocprim::half":
        is_floating = True
        size = 2
        # Overwrite size_min as there is no 1-byte floating point type
        size_min = 0
    elif type_id == "float":
        is_floating = True
        size = 4
    elif type_id == "double":
        is_floating = True
        size = 8
    # Integer types
    elif type_id in ("int8_t", "char"):
        size = 1
    elif type_id == "short":
        size = 2
    elif type_id == "int":
        size = 4
    elif type_id == "int64_t":
        size = 8
    elif type_id == "rocprim::int128_t":
        size = 16
    # Aggregate types: 'custom_type<T,U>' is 'struct { T x; U y; }'
    elif custom_type_match := re.fullmatch(r"custom_type<(.+),(.+)>", type_id):
        is_arithmetic = False
        members = [
            annotate_type(m.strip(), type_name) for m in custom_type_match.groups()
        ]
        if all(m["size"] > 0 for m in members):
            # Scalar members are aligned to their size
            offset = 0
            for member in members:
                offset = round_up(offset, member["size"]) + member["size"]
            size = round_up(offset, max(m["size"] for m in members))
        else:
            log.warning(f"Could not derive the size of {type_id}!")

    # Types without a known size (e.g. 'empty_type', channel counts) are not arithmetic
    if size == 0:
        is_arithmetic = False

    # Derive size_min from size if not set
    if size_min is None:
        size_min = size // 2

    return {
        "name": type_name,
        "type": type_id,
        "size": size,
        "size_min": size_min,
        "is_floating": is_floating,
        "is_arithmetic": is_arithmetic,
        "check_arithmetic": False,
    }


def make_config(
    type_data: dict[str, ty.Any], config_data: dict[str, ty.Any] | None = None
):
    """Constructs a dictionary that contains a config that can be used in jinja templating."""
    if config_data is None:
        config_data = type_data

    # Select the following type names
    # TODO: derive this from alg name maybe?
    type_names = [
        "key_type",
        "value_type",
        "data_type",
        "flag_type",
        "input_type",
        "output_type",
        "channels",
        "active_channels",
    ]
    # Only select relevant types
    type_info = {k: type_data[k] for k in type_names if k in type_data}

    # Annotate with extra info for jinja
    type_details = {k: annotate_type(type_info[k], k) for k in type_info}
    type_hash = frozenset(type_info.items())
    return (type_hash, {"type": type_details, "config": type_info | config_data})


def copyright_year(existing_file: pathlib.Path | None) -> str:
    """Keeps the first year of an existing header's copyright notice."""
    year = datetime.datetime.now().year
    if existing_file is not None and existing_file.exists():
        with open(existing_file, "r") as file:
            match = re.search(r"Copyright \(c\) (\d{4})", file.readline())
        if match and int(match.group(1)) < year:
            return f"{match.group(1)}-{year}"
    return str(year)


def annotate_fp_check(targets):
    """Marks types that need a floating-point guard.

    A type only gets the guard on targets whose tuning data contains both
    floating-point and non-floating-point types for it; otherwise the types of
    the missing kind would match no config at all.
    """
    for configs in targets.values():
        types_by_name: dict[str, list[dict[str, ty.Any]]] = {}
        for c in configs.values():
            for name, t in c["type"].items():
                types_by_name.setdefault(name, []).append(t)
        for types in types_by_name.values():
            is_floating = {
                t["is_floating"]
                for t in types
                if t["type"] not in ("empty_type", "rocprim::empty_type")
            }
            for t in types:
                t["check_fp"] = len(is_floating) > 1


def guard_signature(type_details: dict[str, dict[str, ty.Any]]) -> tuple:
    """Returns what the generated guard checks for the types of a config."""

    def type_signature(t: dict[str, ty.Any]) -> tuple:
        if t["type"] in ("empty_type", "rocprim::empty_type"):
            return ("empty",)
        if t["size"] == 0:
            return ("type", t["type"])
        is_arithmetic = t["is_arithmetic"] if t["check_arithmetic"] else None
        is_floating = t["is_floating"] if t.get("check_fp") else None
        return (is_arithmetic, is_floating, t["size_min"], t["size"])

    return tuple(sorted((name, type_signature(t)) for name, t in type_details.items()))


def resolve_shadowed_configs(targets, new_type_hashes):
    """Makes sure every config of a target has a distinct guard.

    The generated picker returns the first config whose guard matches, so a
    config with the same guard as another one could never be selected.
    Configs whose guards collide get an arithmetic guard on the types where
    they differ, e.g. 'rocprim::int128_t' and a 16-byte 'custom_type'. Configs
    that still collide are dropped; newly tuned configs take precedence over
    existing ones.
    """
    for target_hash, configs in targets.items():
        groups: dict[tuple, list[dict[str, ty.Any]]] = {}
        for config in configs.values():
            groups.setdefault(guard_signature(config["type"]), []).append(config)
        for group in groups.values():
            for name in group[0]["type"]:
                types = [config["type"][name] for config in group]
                if len({t["is_arithmetic"] for t in types}) > 1:
                    for t in types:
                        t["check_arithmetic"] = True

        new = new_type_hashes.get(target_hash, set())
        kept = {}
        dropped = set()
        # Visit newly tuned configs first, so they win over existing ones.
        for type_hash in sorted(configs, key=lambda h: h not in new):
            signature = guard_signature(configs[type_hash]["type"])
            if signature in kept:
                log.warning(
                    f"Dropping config {dict(type_hash)} for {dict(target_hash)}: "
                    f"same guard as {dict(kept[signature])}"
                )
                dropped.add(type_hash)
            else:
                kept[signature] = type_hash
        targets[target_hash] = {h: c for h, c in configs.items() if h not in dropped}


def keep_baseline_if_close(
    data: dict[str, ty.Any],
    candidates: list[dict[str, ty.Any]],
    winner: dict[str, ty.Any] | None,
    margin: float,
) -> dict[str, ty.Any] | None:
    """Returns the fastest baseline entry (the configs the library used on this arch
    when tuning, recorded by the tuner) unless the tuned winner is more than 'margin'
    faster than it. Times are in ms/GiB, so lower is better. Without a winner, returns
    the fastest baseline entry, or None if the tuner recorded none."""
    keys = data.get("tune_params_keys", [])
    best_name = None
    best = None
    for name, baseline in data.get("baselines", {}).items():
        for c in candidates:
            if all(str(c.get(k)) == str(baseline["config"].get(k)) for k in keys):
                if best is None or c["time"] < best["time"]:
                    best_name = name
                    best = c
    if best is None or winner is None:
        return best if winner is None else winner

    speedup = best["time"] / winner["time"] - 1
    if speedup > margin:
        log.info(f"Tuned config is {speedup:.1%} faster than the {best_name} config")
        return winner
    log.info(
        f"Keeping the {best_name} config: the tuned config is only {speedup:.1%} faster"
    )
    return best


def get_sorted_tile(
    algs, existing_block_sort, target_info: dict[str, str], data: dict[str, ty.Any]
) -> int:
    """Returns the items per block of the block_sort config the library uses with a
    block_merge config: the newly tuned one if block_sort was tuned for this target
    too, otherwise the existing one. The merge tiles must fit in it
    (device_merge_sort.hpp)."""
    type_hash, _ = make_config(data)
    tuned = (
        algs.get("device_merge_sort_block_sort", {})
        .get(frozenset(target_info.items()), {})
        .get(type_hash)
    )
    config = tuned["config"] if tuned else None
    if config is None and existing_block_sort is not None:
        import parse as config_parser

        types = {k: data[k] for k in ("key_type", "value_type") if k in data}
        config = config_parser.find_config(*existing_block_sort, target_info, types)
    if config is None:
        log.warning(
            f"No block_sort config found, limiting merge tiles to "
            f"{MERGE_SORT_DEFAULT_SORTED_TILE} items"
        )
        return MERGE_SORT_DEFAULT_SORTED_TILE
    return int(config["block_size_x"]) * int(config["ipt"])


def get_benchmark_names(alg: str) -> list[str]:
    """Returns the rocPRIM benchmarks that measure a tuned algorithm."""
    if alg in BENCHMARK_NAMES:
        return BENCHMARK_NAMES[alg]
    for family in ("device_partition", "device_select"):
        if alg.startswith(family):
            return [family]
    return [alg]


def strip_cfg(name: str) -> str:
    """Removes the 'cfg' entry from a primbench benchmark name, so the results of a
    tuned config match the baseline results of the same case."""
    start = name.find("cfg: ")
    if start < 0:
        return name
    end = start + len("cfg: ")
    if name.startswith("{", end):
        depth = 0
        for end in range(end, len(name)):
            depth += {"{": 1, "}": -1}.get(name[end], 0)
            if depth == 0:
                break
        end += 1
    else:
        end = name.find(",", end)
        end = len(name) if end < 0 else end
    # Also remove the separator next to the entry.
    if name.startswith(", ", end):
        end += 2
    elif name[:start].endswith(", "):
        start -= 2
    return name[:start] + name[end:]


def get_case_results_path(
    file_path: str, data: dict[str, ty.Any], entry: dict[str, ty.Any]
) -> pathlib.Path | None:
    """Returns the per-case primbench results of a tuned config, written by the tuner
    to <output>/cases/<cache name>/<name>=<value>_..._size<bytes>.json."""
    cache_path = pathlib.Path(file_path)
    case_dir = cache_path.parent / "cases" / cache_path.name.removesuffix("_cache.json")
    prefix = "_".join(f"{k}={entry.get(k)}" for k in data.get("tune_params_keys", []))
    paths = glob.glob(f"{glob.escape(str(case_dir / prefix))}_size*.json")
    return pathlib.Path(paths[0]) if len(paths) == 1 else None


def read_case_results(path: pathlib.Path) -> dict[str, float]:
    """Returns the bytes per second of each case in a primbench JSON output."""
    with open(path, "r") as f:
        specializations = json.load(f).get("specializations", [])
    return {strip_cfg(s["name"]): s["bytes_per_second"] for s in specializations}


def read_baseline_csv(path: pathlib.Path) -> dict[str, float]:
    """Returns the bytes per second of each case in a primbench CSV output."""
    with open(path, "r", newline="") as f:
        return {
            strip_cfg(row["name"]): float(row["bytes_per_second"])
            for row in csv.DictReader(f)
            if row.get("noise_timeout", "0") in ("0", "false")
        }


def run_baseline_benchmark(
    benchmark_dir: pathlib.Path,
    benchmark: str,
    arch: str,
    names: set[str],
    sizes: set[int],
    csv_path: pathlib.Path,
) -> bool:
    """Runs the cases of a rocPRIM benchmark that the tuner measured, and writes the
    results to csv_path. The benchmark must be built from the baseline sources."""
    executable = benchmark_dir / f"benchmark_rocprim_{benchmark}"
    if not executable.exists():
        log.warning(f"Could not find benchmark executable: {executable}")
        return False

    json_path = csv_path.parent.parent / "json" / f"{csv_path.stem}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    args = [str(executable), "--csv-out", str(csv_path), "--json-out", str(json_path)]
    # Run the cases at the size the tuner used.
    if len(sizes) == 1:
        args += ["--size", str(next(iter(sizes)))]
    elif sizes:
        log.warning(f"{benchmark}: the tuner used several sizes {sorted(sizes)}")
    # Only run the cases the tuner measured. Long filters are left out, since a single
    # argument is limited to 128 KiB on Linux.
    escape = lambda n: re.sub(r"([\\^$.|?*+()\[\]{}])", r"\\\1", n)
    case_filter = "^(" + "|".join(escape(n) for n in sorted(names)) + ")$"
    if names and len(case_filter) < 100_000:
        args += ["--filter", case_filter]

    log.info(f"Running baseline benchmark {executable.name} for {arch}")
    try:
        subprocess.check_call(args)
    except (OSError, subprocess.CalledProcessError) as error:
        log.warning(f"Could not run {executable}: {error}")
        return False
    return csv_path.exists()


def load_baselines(
    cases: dict[tuple[str, str], dict[str, ty.Any]],
    baseline_dir: pathlib.Path,
    benchmark_dir: pathlib.Path | None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Returns the baseline results (bytes per second by case name) of each
    (benchmark, arch) in 'cases', read from primbench CSV files named
    benchmark_rocprim_<benchmark>_<arch>.csv in baseline_dir. Missing files are created
    by running the benchmarks from benchmark_dir, if it's set."""
    baselines = {}
    for (benchmark, arch), needed in cases.items():
        csv_path = baseline_dir / f"benchmark_rocprim_{benchmark}_{arch}.csv"
        if not csv_path.exists() and benchmark_dir is not None:
            baseline_dir.mkdir(parents=True, exist_ok=True)
            run_baseline_benchmark(
                benchmark_dir,
                benchmark,
                arch,
                needed["names"],
                needed["sizes"],
                csv_path,
            )
        if csv_path.exists():
            baselines[(benchmark, arch)] = read_baseline_csv(csv_path)
        else:
            log.warning(f"No baseline results for {benchmark} on {arch}: {csv_path}")
    return baselines


def compare_to_baseline(
    file_path: str,
    data: dict[str, ty.Any],
    candidates: list[dict[str, ty.Any]],
    baseline: dict[str, float],
    margin: float,
) -> tuple[bool, dict[str, ty.Any] | None]:
    """Compares each candidate to the baseline results, case by case, using the
    per-case results the tuner saved. A candidate is eligible if the geometric mean of
    its speedups is above 1 + margin and none of its cases is slower than 1 - margin;
    the eligible candidate with the highest geometric mean wins.

    Returns (False, None) if no case could be compared, otherwise (True, winner), where
    winner is None if no candidate is eligible."""

    def speedups(c: dict[str, ty.Any]) -> tuple[list[float], int]:
        path = get_case_results_path(file_path, data, c)
        if path is None:
            return [], 0
        results = read_case_results(path)
        return [
            results[name] / baseline[name]
            for name in results
            if name in baseline and baseline[name] > 0
        ], len(results)

    def geomean(values: list[float]) -> float:
        return math.exp(sum(math.log(v) for v in values) / len(values))

    best = None
    best_score = 0.0
    compared = 0
    for c in candidates:
        values, total = speedups(c)
        if not values:
            continue
        compared += 1
        score = geomean(values)
        if score > 1 + margin and min(values) >= 1 - margin and score > best_score:
            best = c
            best_score = score
            best_cases = (len(values), total, min(values))

    if compared == 0:
        return False, None

    # The tuner also measured the configs the library used (see --tune-baseline). If
    # its results differ from the baseline results for the same config, the two were
    # measured differently (build, session, commit) and the comparison is unreliable.
    keys = data.get("tune_params_keys", [])
    for name, recorded in data.get("baselines", {}).items():
        entry = next(
            (
                c
                for c in candidates
                if all(str(c.get(k)) == str(recorded["config"].get(k)) for k in keys)
            ),
            None,
        )
        values = speedups(entry)[0] if entry else []
        if values and abs(geomean(values) - 1) > margin:
            log.warning(
                f"The tuner measured the {name} config {geomean(values) - 1:+.1%} "
                f"faster than the baseline results, check that both come from the same "
                f"sources and build: {file_path}"
            )
    if best is None:
        log.info(
            f"No config is faster than the baseline on every case ({compared} compared)"
        )
    else:
        matched, total, worst = best_cases
        log.info(
            f"Tuned config is {best_score - 1:.1%} faster than the baseline "
            f"(worst case {worst - 1:+.1%}, {matched} of {total} cases compared)"
        )
    return True, best


def derive_gen(arch: str) -> str:
    """Derives the hardware generation from a given architecture."""
    arch_mapping = {
        "gfx803": "gcn3",
        "gfx90[06]": "gcn5",
        "gfx908": "cdna1",
        "gfx90a": "cdna2",
        "gfx94.": "cdna3",
        "gfx95.": "cdna4",
        "gfx101.": "rdna1",
        "gfx103.": "rdna2",
        "gfx11..": "rdna3",
        "gfx12..": "rdna4",
    }
    for pattern, gen in arch_mapping.items():
        if re.match(pattern, arch):
            return gen
    log.warning(f"Could not derive hardware generation from {arch}!")
    return "unknown"


def derive_target(data: dict[str, ty.Any], mapping: dict[str, str]) -> dict[str, str]:
    """Derives the target tuple from a data-dictionary"""
    arch = data["arch_name"]
    gpu = "generic"
    if "gpu" in data:
        gpu = data["gpu"]
    if arch in mapping:
        gpu = mapping[arch]

    if gpu == "generic":
        log.warning(
            f"No gpu name specified and no matching gfx-architecture mapping was specified!"
        )

    return {"gen": derive_gen(arch), "arch": arch, "gpu": gpu, "rep": "amdgcn"}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--input",
        "-i",
        required=True,
        help="Directory or file glob-pattern that points to kernel tuner results",
    )
    cli.add_argument(
        "--output", "-o", required=True, help="The output directory for config headerse"
    )
    cli.add_argument(
        "--existing",
        "-e",
        help="The directory with current config headers to merge new configs with.",
    )
    cli.add_argument(
        "--map-gfx",
        "-g",
        action="append",
        nargs=2,
        help="The mapping from gfx-id to gpu name.",
    )
    cli.add_argument(
        "--default-tuning",
        choices=["keep", "base", "fastest"],
        default="keep",
        help="Config for unknown targets: 'keep' keeps what the existing header uses "
        "(the default; same as 'base' without --existing), 'base' uses the "
        "algorithm's default config (*_config_params_base), 'fastest' aliases to the "
        "tuned configs of the lowest tuned architecture.",
    )
    cli.add_argument(
        "--baseline-margin",
        type=float,
        default=0.05,
        help="Keep the config the library used on the tuned arch (current or "
        "fallback, whichever is faster) unless the tuned config is faster by more "
        "than this fraction (default: 0.05). When comparing to baseline benchmark "
        "results, a tuned config is also rejected if any case is slower than the "
        "baseline by more than this.",
    )
    cli.add_argument(
        "--no-compare-to-base",
        action="store_true",
        help="Don't compare the tuned configs case by case to baseline benchmark "
        "results, only to the baselines measured by the tuner (--tune-baseline).",
    )
    cli.add_argument(
        "--baseline-path",
        help="Directory with primbench CSV results of the baseline, named "
        "benchmark_rocprim_<benchmark>_<arch>.csv. If not set, the benchmarks are run "
        "from --benchmark-dir and their results stored in <output>/baseline-bench/csv.",
    )
    cli.add_argument(
        "--benchmark-dir",
        help="Directory with the rocPRIM benchmark executables, built from the baseline "
        "sources. Needed if --baseline-path is not set, unless --no-compare-to-base.",
    )

    args = cli.parse_args()

    gfx_mapping = {}
    if args.map_gfx:
        gfx_mapping = dict(args.map_gfx)

    # Read '--input' argument and suffix with '.json' if it's not '.json'.
    input_glob = args.input
    if not input_glob.endswith(".json"):
        input_glob = f"{input_glob}/*.json"

    # Find all files. block_sort goes first, because block_merge configs must fit in
    # its sorted tile.
    file_paths = sorted(
        glob.glob(input_glob),
        key=lambda p: "device_merge_sort_block_merge" in os.path.basename(p),
    )

    # Find existing configs
    existing_dir = None
    if args.existing:
        existing_dir = pathlib.Path(args.existing)
        if not existing_dir.exists():
            log.error(f"Could not find directory: {existing_dir}")
            quit()

    existing_block_sort = None
    if existing_dir and (existing_dir / "device_merge_sort_block_sort.hpp").exists():
        import parse as config_parser

        header = (existing_dir / "device_merge_sort_block_sort.hpp").read_text()
        existing_block_sort = (
            config_parser.parse_lines(header.splitlines()),
            config_parser.parse_fallback_target(header),
        )

    # Load the baseline results of the benchmarks that measure the tuned algorithms.
    baselines = {}
    compare_to_base = not args.no_compare_to_base
    if not compare_to_base and (args.baseline_path or args.benchmark_dir):
        log.warning(
            "--baseline-path and --benchmark-dir are ignored with --no-compare-to-base"
        )
    if compare_to_base:
        if args.baseline_path:
            baseline_dir = pathlib.Path(args.baseline_path)
            benchmark_dir = None
        elif args.benchmark_dir:
            baseline_dir = pathlib.Path(args.output) / "baseline-bench" / "csv"
            benchmark_dir = pathlib.Path(args.benchmark_dir)
        else:
            log.error(
                "Comparing to a baseline needs --baseline-path or --benchmark-dir. "
                "Pass --no-compare-to-base to skip the comparison."
            )
            quit(1)

        # Collect the cases and sizes the tuner measured for each benchmark.
        cases: dict[tuple[str, str], dict[str, ty.Any]] = {}
        for file_path in file_paths:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except json.decoder.JSONDecodeError:
                continue
            entry = next((c for c in data["cache"].values() if "time" in c), None)
            path = get_case_results_path(file_path, data, entry) if entry else None
            for benchmark in get_benchmark_names(data["algo_name"]):
                needed = cases.setdefault(
                    (benchmark, data["arch_name"]), {"names": set(), "sizes": set()}
                )
                if path is not None:
                    needed["names"] |= set(read_case_results(path))
                    needed["sizes"].add(int(re.search(r"_size(\d+)", path.name)[1]))
        baselines = load_baselines(cases, baseline_dir, benchmark_dir)

    algs: dict[str, dict[frozenset, dict[frozenset, dict[str, ty.Any]]]] = {}
    for file_path in file_paths:
        log.debug(f"Parsing: {file_path}")
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError:
                log.warning(f"Skipping due to JSONDecodeError: {file_path}")
                continue

            # Record the target: gen, arch, gpu, rep
            target_info = derive_target(data, gfx_mapping)
            # Create hashable key
            target_hash = frozenset(target_info.items())

            # Filter/skip data entries that have no "time" column
            data_filter = [c for c in data["cache"].values() if "time" in c]
            filter_dif = len(data["cache"]) - len(data_filter)
            if filter_dif > 0:
                log.warning(f"Skipping failed compilations: {filter_dif}")

            # Skip configs that still hit the noise timeout after the tuner's retries
            noise_timeouts = set(data.get("noise_timeouts", []))
            if noise_timeouts:
                data_filter = [
                    c
                    for k, c in data["cache"].items()
                    if "time" in c and k not in noise_timeouts
                ]
                log.warning(
                    f"Skipping configs that hit the noise timeout: {len(noise_timeouts)}"
                )

            if data["algo_name"] == "device_merge_sort_block_merge":
                sorted_tile = get_sorted_tile(
                    algs, existing_block_sort, target_info, data
                )
                temp = [
                    c
                    for c in data_filter
                    if int(c["block_size_x"]) * int(c["ipt"]) <= sorted_tile
                    and int(c.get("odd_even_block_size", 0))
                    * int(c.get("odd_even_ipt", 0))
                    <= sorted_tile
                ]
                if len(temp) < len(data_filter):
                    log.info(
                        f"Dropped {len(data_filter) - len(temp)} constraint-violating configs: {file_path}"
                    )
                data_filter = temp

            if not data_filter:
                log.warning(f"No valid configs left, skipping: {file_path}")
                continue

            # Find best config: compared to the baseline benchmark results if possible,
            # otherwise to the baselines measured by the tuner
            compared = False
            if compare_to_base:
                baseline = {}
                for benchmark in get_benchmark_names(data["algo_name"]):
                    baseline |= baselines.get((benchmark, data["arch_name"]), {})
                compared, config_data = compare_to_baseline(
                    file_path, data, data_filter, baseline, args.baseline_margin
                )
                if compared and config_data is None:
                    # No tuned config beats the baseline, keep the config it used.
                    config_data = keep_baseline_if_close(
                        data, data_filter, None, math.inf
                    )
                    if config_data is None:
                        log.warning(
                            f"No tuned config beats the baseline and the tuner recorded "
                            f"no baseline config, so no config is generated: {file_path}"
                        )
                        continue
            if not compared:
                config_data = min(data_filter, key=lambda c: c["time"])
                config_data = keep_baseline_if_close(
                    data, data_filter, config_data, args.baseline_margin
                )
            # Drop unrelated entries
            config_ignored_names = [
                "time",
                "times",
                "compile_time",
                "verification_time",
                "framework_time",
                "benchmark_time",
                "strategy_time",
                "timestamp",
            ]
            config_data = {
                k: config_data[k] for k in config_data if k not in config_ignored_names
            }

            # Ensure algorithm entry exists
            alg: str = data["algo_name"]
            if alg not in algs:
                algs[alg] = {}

            # Ensure target entry exists
            if target_hash not in algs[alg]:
                algs[alg][target_hash] = {}

            type_hash, config = make_config(data, config_data)

            # Add config entry
            algs[alg][target_hash][type_hash] = config

    log.info(f"Parsed {len(algs)} different algorithm(s)!")

    script_dir = pathlib.Path(__file__).parent.absolute()
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(script_dir / "templates"),
        autoescape=jinja2.select_autoescape(),
    )

    # Remember which configs were newly tuned, they take precedence over existing ones
    new_type_hashes = {
        alg_name: {target_hash: set(target) for target_hash, target in targets.items()}
        for alg_name, targets in algs.items()
    }

    # Merge newly generated configs to existing configs
    existing_fallback = {}
    if existing_dir:
        import parse as config_parser

        for alg_name, targets in algs.items():
            config_file = existing_dir / f"{alg_name}.hpp"
            log.debug(f"Parsing: {config_file}")
            with open(config_file, "r") as file:
                header = file.read()
                existing_data = config_parser.parse_lines(header.splitlines())
                existing_fallback[alg_name] = config_parser.parse_fallback_target(
                    header
                )
                merged_targets = {
                    target_hash: dict(make_config(entry) for entry in target)
                    for target_hash, target in existing_data.items()
                }

                for target_hash, target in targets.items():
                    if target_hash not in merged_targets:
                        new_target = dict(target_hash)
                        for existing_hash in merged_targets:
                            existing_target = dict(existing_hash)
                            if (
                                existing_target["arch"] == new_target["arch"]
                                and existing_target["gpu"] != new_target["gpu"]
                            ):
                                log.warning(
                                    f"{alg_name}: adding {new_target['arch']} configs for "
                                    f"{new_target['gpu']} next to the existing ones for "
                                    f"{existing_target['gpu']}. Check --map-gfx if they "
                                    "should be the same GPU."
                                )
                        merged_targets[target_hash] = {}

                    for type_hash, entry in target.items():
                        merged_targets[target_hash][type_hash] = entry
            algs[alg_name] = merged_targets

    # Output generated configs
    output_dir = pathlib.Path(args.output)
    os.makedirs(output_dir, exist_ok=True)
    for alg_name, targets in algs.items():
        annotate_fp_check(targets)
        resolve_shadowed_configs(targets, new_type_hashes[alg_name])

        default_target = None
        if args.default_tuning == "keep" and existing_fallback.get(alg_name):
            default_target = dict(existing_fallback[alg_name])
        elif args.default_tuning == "fastest":
            default_target = min(
                (dict(t) for t in targets.keys() if dict(t)["gen"] != "gcn5"),
                key=lambda t: int(re.sub("gfx", "", t["arch"]), base=16),
            )

        template_name = f"{alg_name}.h.jinja2"
        template = env.get_template(template_name)
        rendered = template.render(
            {
                "year": copyright_year(
                    existing_dir / f"{alg_name}.hpp" if existing_dir else None
                ),
                "targets": targets,
                "alg_name": alg_name,
                "alg_short_name": re.sub(r"^device_", "", alg_name),
                "unknown_target": {
                    "gen": "unknown",
                    "arch": "unknown",
                    "gpu": "generic",
                    "rep": "amdgcn",
                },
                "default_target": default_target,
            }
        )

        with open(output_dir / f"{alg_name}.hpp", "w") as file:
            file.write(rendered)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
