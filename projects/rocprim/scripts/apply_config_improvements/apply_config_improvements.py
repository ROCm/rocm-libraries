#!/usr/bin/env python3

# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import json
import re
import sys
import unittest
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

EMPTY_TYPENAME = "empty_type"
warnings = []
total_specializations = 0
improvement_count = 0
new_specialization_count = 0
noisy_rejected_count = 0
slower_rejected_count = 0
marginal_improvements_rejected_count = 0


class colors:
    OK = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    END_COLOR = "\033[0m"


class Contender(NamedTuple):
    instance: Dict[str, Any]
    string: str


def print_results():
    parts = []

    if improvement_count:
        parts.append(f"{improvement_count} improved")
    if new_specialization_count:
        parts.append(f"{new_specialization_count} new")
    if noisy_rejected_count:
        parts.append(f"{noisy_rejected_count} noisy rejected")
    if slower_rejected_count:
        parts.append(f"{slower_rejected_count} slower rejected")
    if marginal_improvements_rejected_count:
        parts.append(
            f"{marginal_improvements_rejected_count} marginal improvements rejected"
        )

    print(f"{total_specializations} specializations: {', '.join(parts)}")


def strip_ansi(s: str) -> str:
    # Remove ANSI escape sequences for color
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# Takes: Instance(key_type='double', value_type='empty_type')
# Returns: key_type=double, value_type=empty_type
def stringify_instance_key(instance_key) -> str:
    s = str(instance_key)
    s = s.removeprefix("Instance(").removesuffix(")")
    # Remove quotes around values
    s = re.sub(r"'([^']*)'", r"\1", s)
    return s


def add_new_contenders(
    algorithm_name: str,
    new_config: Dict[str, Any],
    new_alg_data: Dict[str, Any],
    score_assigner: Callable[[List[Dict[str, Any]]], None],
    contenders: Dict[Tuple[Any, str], Contender],
    improvement_threshold_percentage: float,
) -> bool:
    global warnings, total_specializations, improvement_count, new_specialization_count, noisy_rejected_count, slower_rejected_count, marginal_improvements_rejected_count

    improved = False
    rows = []
    Z_SCORE = 1.96  # 95% confidence

    # Collect all rows first to determine max widths
    for arch, new_arch_specializations in new_config["specializations"].items():
        if arch not in new_alg_data and (
            arch != "unknown" or "gfx908" not in new_alg_data
        ):
            warnings.append(
                f"{colors.WARN}The new JSON data is missing {arch} for {algorithm_name}{colors.END_COLOR}"
            )
            continue

        # create_optimization.py its create_config_file_content() chose to make "unknown" a copy of "gfx908"
        new_arch_data = new_alg_data["gfx908" if arch == "unknown" else arch]
        for instance_key, new_instance_data in new_arch_specializations.items():
            if instance_key not in new_arch_data:
                sys.exit(
                    f"{colors.FAIL}The new JSON data is missing {arch} specialization '{stringify_instance_key(instance_key)}' for {algorithm_name}{colors.END_COLOR}"
                )
            new_instances = new_arch_data[instance_key]

            add_base_args(new_instance_data["base_args"], new_instances)
            score_assigner(new_instances)
            new_best_instance = get_best_instance(new_instances)

            total_specializations += 1

            row = {}
            row["arch"] = str(arch)
            row["key"] = stringify_instance_key(instance_key)
            row["new_family_index"] = new_best_instance["family_index"]
            row["new_bps"] = f"{new_best_instance['bytes_per_second']:.2e}"

            key = (instance_key, arch)

            # If there is no old config specialization, we always accept the new one.
            if key not in contenders:
                status = "New"
                colored_status = f"{colors.OK}{status}{colors.END_COLOR}"
                row["status"] = colored_status
                row["status_raw"] = status

                row["old_noise"] = ""

                new_noise_val = new_best_instance.get("cv", 0.0) * 100
                row["new_noise"] = f"{new_noise_val:.1f}%"

                rows.append(row)
                new_specialization_count += 1
                improved = True
                contenders[key] = Contender(
                    instance=new_best_instance, string=new_instance_data["string"]
                )
                continue

            old_best_instance = contenders[key].instance

            row["old_family_index"] = old_best_instance.get("family_index", "-")
            row["old_bps"] = (
                f"{old_best_instance['bytes_per_second']:.2e}"
                if old_best_instance
                else "-"
            )

            best_instance = (
                get_best_instance([old_best_instance, new_best_instance])
                if old_best_instance
                else new_best_instance
            )

            # If there was no improvement, skip this contender.
            if best_instance != new_best_instance:
                slower_rejected_count += 1
                continue

            new_score = new_best_instance["score"]
            old_score = old_best_instance.get("score", 0.0)
            improvement = (
                ((new_score - old_score) / old_score * 100.0) if old_score > 0 else 0.0
            )

            # If improvement is below CLI threshold, skip (no coloring)
            if improvement < improvement_threshold_percentage:
                marginal_improvements_rejected_count += 1
                continue

            # Adaptive z-score threshold.
            # This ensures that if the old or new results were very noisy,
            # that the improvement threshold is raised accordingly.
            cv_old = old_best_instance.get("cv", 0.0)
            cv_new = new_best_instance.get("cv", 0.0)
            adaptive_threshold = Z_SCORE * ((cv_old**2 + cv_new**2) ** 0.5) * 100.0
            threshold = max(improvement_threshold_percentage, adaptive_threshold)

            old_noise_val = cv_old * 100
            new_noise_val = cv_new * 100

            # If improvement is below adaptive threshold, print noise values in red and skip
            if improvement < threshold:
                noisy_rejected_count += 1
                status = f"Rejected: {improvement:.1f}% < {threshold:.1f}% faster"
                colored_status = f"{colors.FAIL}{status}{colors.END_COLOR}"
                row["status"] = colored_status
                row["status_raw"] = status
                row["old_noise"] = f"{old_noise_val:.1f}%"
                row["new_noise"] = f"{new_noise_val:.1f}%"
                rows.append(row)
                continue

            status = f"Improved: {improvement:.1f}%"
            colored_status = f"{colors.OK}{status}{colors.END_COLOR}"
            row["status"] = colored_status
            row["status_raw"] = status

            row["old_noise"] = f"{old_noise_val:.1f}%"
            row["new_noise"] = f"{new_noise_val:.1f}%"

            rows.append(row)
            improvement_count += 1
            improved = True
            contenders[key] = Contender(
                instance=new_best_instance, string=new_instance_data["string"]
            )

    columns = [
        ("status", f"Status of {algorithm_name}"),
        ("noise", "Noise (old/new)"),
        ("bps", "Bytes/sec (old/new)"),
        ("arch", "Arch"),
        ("key", "Specialization"),
        ("family_index", "Family index (old/new)"),
    ]

    # Prepare compact row values
    for row in rows:
        # Compose compact columns, using '-' for missing old values
        old_noise = row.get("old_noise", "")
        new_noise = row.get("new_noise", "")
        row["noise"] = f"-/{new_noise}" if not old_noise else f"{old_noise}/{new_noise}"
        row["bps"] = f"{row.get('old_bps', '-')}/{row.get('new_bps', '-')}"
        row["family_index"] = (
            f"{row.get('old_family_index', '-')}/{row.get('new_family_index', '-')}"
        )

    # Calculate column widths based on raw (uncolored) content
    col_widths = {}
    for key, title in columns:
        if key == "status":
            raw_key = "status_raw"
            max_content = max(
                [len(str(row.get(raw_key, ""))) for row in rows] + [len(title)]
            )
        else:
            max_content = max(
                [len(strip_ansi(str(row.get(key, "")))) for row in rows] + [len(title)]
            )
        col_widths[key] = max_content

    # Print header
    header_line = "  ".join(f"{title:<{col_widths[key]}}" for key, title in columns)
    print(header_line)

    # Print horizontal line made of dashes under the header
    print("-" * len(header_line))

    # Print rows, using colored text but padding based on uncolored width
    for row in rows:
        line = []
        for key, _ in columns:
            value = row.get(key, "")
            pad = col_widths[key] + len(value) - len(strip_ansi(value))
            line.append(f"{value:<{pad}}")
        print("  ".join(line))

    print("")

    return improved


def get_old_contenders(
    algorithm_name: str,
    old_config: Dict[str, Any],
    old_alg_data: Dict[str, Any],
    score_assigner: Callable[[List[Dict[str, Any]]], None],
) -> Dict[Tuple[Any, str], Contender]:
    global warnings

    contenders: Dict[Tuple[Any, str], Contender] = {}

    old_config.setdefault("specializations", {})

    for arch, old_arch_specializations in old_config["specializations"].items():
        # Always keep every old specialization, even if missing in old_alg_data
        if arch not in old_alg_data and (
            arch != "unknown" or "gfx908" not in old_alg_data
        ):
            warnings.append(
                f"{colors.WARN}The old JSON data is missing {arch} for {algorithm_name}{colors.END_COLOR}"
            )
            old_arch_data = {}
        else:
            old_arch_data = old_alg_data["gfx908" if arch == "unknown" else arch]

        for instance_key, old_instance_data in old_arch_specializations.items():
            if instance_key not in old_arch_data:
                # If old_arch_data is falsy, then a warning was already printed
                if old_arch_data:
                    warnings.append(
                        f"{colors.WARN}The old JSON data is missing {arch} specialization '{stringify_instance_key(instance_key)}' for {algorithm_name}{colors.END_COLOR}"
                    )
                old_instances = []
            else:
                old_instances = old_arch_data[instance_key]

            add_base_args(old_instance_data["base_args"], old_instances)
            score_assigner(old_instances)

            # If old_instances is empty, just use the config string as-is
            old_best_instance = (
                get_best_instance(old_instances) if old_instances else {}
            )

            key = (instance_key, arch)
            contenders[key] = Contender(
                instance=old_best_instance, string=old_instance_data["string"]
            )

    return contenders


def get_best_instance(instances):
    return max(instances, key=lambda x: x["score"])


def add_base_args(base_args, instances):
    """
    Adds base arguments from a config to instances, for get_score_assigner().
    """
    for instance in instances:
        if instance["algo"] in {"merge_sort_block_sort", "radix_sort_block_sort"}:
            instance["bs"] = int(base_args[0])
            instance["ipt"] = int(base_args[1])


def score_assigner_default(instances: List[Dict[str, Any]]) -> None:
    """
    Default formula to assign scores, only looking at items_per_second.
    """
    for instance in instances:
        instance["score"] = instance["items_per_second"]


def score_assigner_block_sort(instances: List[Dict[str, Any]]) -> None:
    """
    Formula to assign scores for block sort algorithms.
    If we can double the sorted items_per_block and items_per_second
    does not degrade more than ~10%, it is considered superior.
    """
    for instance in instances:
        instance["score"] = instance["items_per_second"] * (
            (instance["bs"] * instance["ipt"]) ** (1 / 4)
        )


def get_score_assigner(algorithm_name: str) -> Callable[[List[Dict[str, Any]]], None]:
    """
    Retrieves the appropriate scoring function for the specified algorithm.

    This function ensures consistent configuration scoring between
    **autotuning** and **benchmarking** contexts, accounting for algorithm-specific
    performance trade-offs.

    For most algorithms, `score_assigner_default` is used.
    For block-sorting algorithms (e.g., merge sort, radix sort), the
    specialized `score_assigner_block_sort` is used to reward larger block
    sizes when throughput remains acceptable.

    `adjacent_difference` uses only `value_type` as its config selector,
    even though its benchmark also includes the keys `"is_left"`.
    So while adjacent_difference is benchmarked with both is_left=True
    and is_left=False, the config selection doesn't distinguish between them.
    """
    if algorithm_name in {"merge_sort_block_sort", "radix_sort_block_sort"}:
        return score_assigner_block_sort
    return score_assigner_default


def generate_improved_configs(
    improvement_threshold_percentage: float,
    old_data: Dict[str, Any],
    new_data: Dict[str, Any],
    old_configs: Dict[str, Any],
    new_configs: Dict[str, Any],
    improved_configs_dir: Path,
) -> None:
    improved_configs_dir.mkdir(exist_ok=True)
    for algorithm_name, new_config in new_configs.items():
        old_config = old_configs.get(algorithm_name)
        old_alg_data = old_data.get(algorithm_name)
        new_alg_data = new_data.get(algorithm_name, {})

        score_assigner = get_score_assigner(algorithm_name)
        contenders = get_old_contenders(
            algorithm_name, old_config or {}, old_alg_data or {}, score_assigner
        )
        improved = add_new_contenders(
            algorithm_name,
            new_config,
            new_alg_data,
            score_assigner,
            contenders,
            improvement_threshold_percentage,
        )
        if not improved:
            continue

        improved_config_path = improved_configs_dir / new_config["full_name"]
        with improved_config_path.open("w", encoding="utf-8") as f:
            f.write(
                old_config["start_of_config"]
                if old_config
                else new_config["start_of_config"]
            )
            for contender in contenders.values():
                f.write(contender.string)
            # Remove leading newlines from end_of_config to avoid triple newlines
            end = (
                old_config["end_of_config"]
                if old_config
                else new_config["end_of_config"]
            ).lstrip("\n")
            if not end.endswith("\n"):
                end += "\n"
            f.write(end)


def extract_template_arguments(code: str) -> list[str]:
    """
    Extracts the top-level template arguments from a base config class in a C++ struct specialization.

    Assumes the input contains a line ending in `: XXX_config<...> {`, where `XXX_config` is
    the name of the base config class and the angle brackets contain template arguments.

    Args:
        code (str): A string containing the C++ struct definition.

    Returns:
        List[str]: A list of top-level template arguments as strings.
                   Nested templates like `kernel_config<1024, 1>` are preserved as single items.

    Example:
        Input:
            'struct foo : some_config<256, kernel_config<128, 2>, 1 << 2> {'
        Output:
            ['256', 'kernel_config<128, 2>', '1 << 2']
    """
    # Match the base class ending in `_config<...> {`
    match = re.search(r":\s*\w+_config<(.+?)>\s*{", code, re.DOTALL)
    if not match:
        return []

    template_args = match.group(1)

    # Split top-level arguments by comma, keeping nested templates intact
    args = []
    current = ""
    depth = 0
    i = 0
    while i < len(template_args):
        if template_args[i : i + 2] in ("<<", ">>"):
            current += template_args[i : i + 2]
            i += 2
            continue

        char = template_args[i]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1

        if char == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += char
        i += 1

    if current.strip():
        args.append(current.strip())

    return args


def get_specialization_key(specialization_string: str) -> Any:
    matches = dict(re.findall(r"(\w+) = ([\w:]+)", specialization_string))
    Instance = namedtuple("Instance", matches.keys())
    return Instance(*matches.values())


# This is the format of the returned dictionary:
# {
#     'select_flag': {
#         'full_name': 'device_select_flag.hpp',
#         'start_of_config': '// Copyright ...',
#         'end_of_config': '} // end namespace detail ...',
#         'specializations': {
#             'gfx1200': {
#                 Instance(key_type='double'): {
#                     'string': '// Based on key_type = double\n ... select_config<512, kernel_config<128, 2>>\n{};\n\n',
#                     'base_args': [
#                         '512',
#                         'kernel_config<128, 2>'
#                     ]
#                 },
#                 Instance(key_type='float'): {
#                     'string': '// Based on key_type = float\n ... select_config<1024, kernel_config<128, 2>>\n{};\n\n',
#                     'base_args': [
#                         '1024',
#                         'kernel_config<128, 2>'
#                     ]
#                 }
#             }
#         }
#     }
# }
def read_configs(dir_path: Path) -> Dict[str, Any]:
    configs: Dict[str, Any] = {}

    for hpp_path in dir_path.rglob("*.hpp"):
        name = hpp_path.name.removeprefix("device_").removesuffix(".hpp")
        config = configs.setdefault(name, {})
        config["full_name"] = hpp_path.name

        text = hpp_path.read_text()
        matches = re.split(r"(// Based on .*?{\s*};\n\n)", text, flags=re.DOTALL)
        matches = list(filter(None, matches))
        config["start_of_config"] = matches[0]
        config["end_of_config"] = matches[-1]

        all_specializations = config.setdefault("specializations", {})

        specialization_strings = matches[1:-1]
        for specialization_string in specialization_strings:
            arch_match = re.search(r"target_arch::(.*?)\)", specialization_string)
            if not arch_match:
                sys.exit(
                    f"{colors.FAIL}Could not find arch in specialization: {specialization_string}{colors.END_COLOR}"
                )
            arch = arch_match.group(1)

            specialization_key = get_specialization_key(specialization_string)
            arch_specializations = all_specializations.setdefault(arch, {})
            if specialization_key in arch_specializations:
                sys.exit(
                    f"{colors.FAIL}Specialization key duplicate '{specialization_key}'{colors.END_COLOR}"
                )
            arch_specializations[specialization_key] = {
                "string": specialization_string,
                "base_args": extract_template_arguments(specialization_string),
            }

    return configs


def get_instance_key(instanced_types: Dict[str, Any], selectors: List[str]) -> Any:
    """
    Creates a hashable namedtuple based on selectors and instanced_types.
    Missing types are filled with EMPTY_TYPENAME.
    """
    Instance = namedtuple("Instance", selectors)
    return Instance(
        *(instanced_types.get(field, EMPTY_TYPENAME) for field in selectors)
    )


# This is the format of the returned dictionary:
# {
#     'select_flag': {
#         'gfx1200': {
#             Instance(key_type='double'): [
#                 { 'items_per_second': 200, 'segment_count': 10 },
#                 { 'items_per_second': 300, 'segment_count': 20 }
#             ],
#             Instance(key_type='float'): [
#                 { 'items_per_second': 400, 'segment_count': 10 },
#                 { 'items_per_second': 600, 'segment_count': 20 }
#             ]
#         }
#     }
# }
def read_data(dir_path: Path, selectors: Dict[str, List[str]]) -> Dict[str, Any]:
    all_data: Dict[str, Any] = {}

    for json_path in dir_path.rglob("*.json"):
        json_data = json.loads(json_path.read_text())
        arch = json_data["context"]["hdp_gcn_arch_name"].split(":")[0]

        for benchmark in json_data["benchmarks"]:
            name = benchmark["name"]
            name_filtered = re.match(r"{.*}", name)
            if not name_filtered:
                raise RuntimeError(f"ERROR: cannot parse JSON from: '{name}'")

            data = json.loads(name_filtered.group(0))
            if not data:
                raise RuntimeError(f"ERROR: cannot parse JSON from: '{name}'")

            data["items_per_second"] = benchmark["items_per_second"]
            data["cv"] = benchmark["cv"]
            data["family_index"] = benchmark["family_index"]
            data["bytes_per_second"] = benchmark["bytes_per_second"]

            algorithm_name = data["algo"]
            if "subalgo" in data:
                algorithm_name += "_" + data["subalgo"]

            alg_data = all_data.setdefault(algorithm_name, {})
            arch_data = alg_data.setdefault(arch, defaultdict(list))

            if algorithm_name not in selectors:
                sys.exit(
                    f"{colors.FAIL}No selectors found for algorithm '{algorithm_name}' in the selectors JSON{colors.END_COLOR}"
                )
            instance_key = get_instance_key(data, selectors[algorithm_name])
            arch_data[instance_key].append(data)

    return all_data


def get_selectors() -> Dict[str, List[str]]:
    """
    Reads selectors from a JSON file at selectors_path.
    The JSON should be a dict mapping algorithm names to lists of selector strings.
    """
    selectors_path = Path(__file__).parent / "selectors.json"
    with selectors_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class SingleUseAction(argparse.Action):
    """Custom action that forbids setting an argument multiple times."""

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"argument {option_string} specified multiple times")
        setattr(namespace, self.dest, values)


class StrictArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that automatically uses SingleUseAction for all args."""

    def add_argument(self, *args, **kwargs):
        # Don't override actions like 'store_true', 'append', etc.
        if "action" not in kwargs or kwargs["action"] == "store":
            kwargs["action"] = SingleUseAction
        return super().add_argument(*args, **kwargs)


def existing_path(path_str):
    """Validate that a path exists (file or directory) and return it as a Path."""
    p = Path(path_str)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"{path_str!r} does not exist")
    return p


def add_arguments(parser: StrictArgumentParser) -> None:
    parser.add_argument(
        "--improvement_threshold_percentage",
        help="Minimum improvement percentage required for acceptance (acts as a floor for adaptive threshold)",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--old_json_dir",
        help="The input directory of old JSON files running the benchmarks produced",
        required=True,
        type=existing_path,
    )
    parser.add_argument(
        "--new_json_dir",
        help="The input directory of new JSON files running the benchmarks produced",
        required=True,
        type=existing_path,
    )
    parser.add_argument(
        "--old_configs_dir",
        help="The input directory of old config files",
        required=True,
        type=existing_path,
    )
    parser.add_argument(
        "--new_configs_dir",
        help="The input directory of new config files",
        required=True,
        type=existing_path,
    )
    parser.add_argument(
        "--improved_configs_dir",
        help="The output directory of improved config files",
        required=True,
        type=Path,  # Not using existing_path, since the dir is created.
    )


# Run this with `python3 -m unittest scripts/apply_config_improvements/apply_config_improvements.py`
class TestExtractTemplateArguments(unittest.TestCase):
    def test_complex(self):
        specialization = """
        struct default_radix_sort_onesweep_config<
            static_cast<unsigned int>(target_arch::gfx1030),
            key_type,
            value_type,
            std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                              && (sizeof(key_type) > 4) && (sizeof(value_type) <= 16)
                              && (sizeof(value_type) > 8))>>
            : radix_sort_onesweep_config<kernel_config<1024, 1>, (1 << 17) + 1 >> 2 + 70000,
                                         8,
                                         block_radix_rank_algorithm::match>
        {};
        """

        expected = [
            "kernel_config<1024, 1>",
            "(1 << 17) + 1 >> 2 + 70000",
            "8",
            "block_radix_rank_algorithm::match",
        ]

        self.assertEqual(extract_template_arguments(specialization), expected)


def main() -> None:
    parser = StrictArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    selectors = get_selectors()

    old_data = read_data(args.old_json_dir, selectors)
    new_data = read_data(args.new_json_dir, selectors)

    old_configs = read_configs(args.old_configs_dir)
    new_configs = read_configs(args.new_configs_dir)
    if len(new_configs) == 0:
        sys.exit(f"{colors.FAIL}No new configs{colors.END_COLOR}")

    generate_improved_configs(
        args.improvement_threshold_percentage,
        old_data,
        new_data,
        old_configs,
        new_configs,
        args.improved_configs_dir,
    )

    if warnings:
        print("\n".join(warnings))
        print("")

    print_results()


if __name__ == "__main__":
    main()
