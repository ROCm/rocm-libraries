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
from dataclasses import dataclass
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

TARGET_GPUS_DICT = {
    "MI350X": "mi350x",
    "MI325X": "mi325x",
    "MI308X": "mi308x",
    "MI300A": "mi300a",
    "MI300X": "mi300x",
    "MI210": "mi210",
    "MI100": "mi100",
    "RX 9060": "rx9060",
    "RX 9070": "rx9070",
    "V620": "v620",
    "RX 7900": "rx7900",
    "RX 6900": "rx6900"
}

def get_gen_from_architecture(arch):
    match arch:
        case "target_arch::gfx803":
            return "gen::gcn3"
        case "target_arch::gfx900" | "target_arch::gfx906":
            return "gen::gcn5"
        case "target_arch::gfx908":
            return "gen::cdna1"
        case "target_arch::gfx90a":
            return "gen::cdna2"
        case "target_arch::gfx942":
            return "gen::cdna3"
        case "target_arch::gfx950":
            return "gen::cdna4"
        case (
            "target_arch::gfx1010"
            | "target_arch::gfx1011"
            | "target_arch::gfx1012"
        ):
            return "gen::rdna1"
        case "target_arch::gfx1030":
            return "gen::rdna2"
        case (
            "target_arch::gfx1100"
            | "target_arch::gfx1101"
            | "target_arch::gfx1102"
            | "target_arch::gfx1103"
            | "target_arch::gfx1150"
            | "target_arch::gfx1151"
            | "target_arch::gfx1152"
            | "target_arch::gfx1153"
        ):
            return "gen::rdna3"
        case "target_arch::gfx1200" | "target_arch::gfx1201":
            return "gen::rdna4"
        case "target_arch::unknown" | "target_arch::invalid":
            return "gen::unknown"
        case _:
            return "gen::unknown"


def get_target_gpu_from_context(context):
    """
    Uses the benchmark run context embedded into the benchmark output json to retrieve the targeted gpu
    """

    gpu_from_context = context['hdp_name']

    ret = "gpu::generic"

    for gpu in TARGET_GPUS_DICT.keys():
        if gpu in gpu_from_context:
            ret = f'gpu::{TARGET_GPUS_DICT[gpu]}'

    if ret == "gpu::generic":
        print(f"WARNING: Unrecognized GPU '{gpu_from_context}', so using gpu::generic", file=sys.stderr, flush=True)
    return ret

@dataclass(frozen=True, eq=True)
class Target:
    """
    Data class describing the target of the benchmark
    """
    gen: str = "gen::unknown"
    arch: str = "target_arch::unknown"
    gpu: str = "gpu::generic"
    rep: str = "rep::amdgcn"

    def __iter__(self):
        yield from (self.gen, self.arch, self.gpu, self.rep)

    def as_str(self) -> str:
        return ", ".join(str(v) for v in self)
    
    @classmethod
    def from_string(self, raw: str) -> "Target":
        """Create Target instance from a raw string with key::value pairs"""
        # Split by commas and strip whitespace/newlines
        parts = [p.strip() for p in raw.split(",") if p.strip()]

        # Default data
        data = {
            "gen": self.gen,
            "arch": self.arch,
            "gpu": self.gpu,
            "rep": self.rep,
        }

        # Assign fields based on prefixes
        for part in parts:
            if part.startswith("gen::"):
                data["gen"] = part
            elif part.startswith("target_arch::"):
                data["arch"] = part
            elif part.startswith("gpu::"):
                data["gpu"] = part
            elif part.startswith("rep::"):
                data["rep"] = part

        return self(**data)
    

def get_target(context):
    arch = f"target_arch::{context["hdp_gcn_arch_name"].split(":")[0]}"
    gpu = get_target_gpu_from_context(context)
    gen = get_gen_from_architecture(arch)
    rep = "rep::amdgcn"
    return Target(gen, arch, gpu, rep)

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
    contenders: Dict[Target, Dict[Any, Contender]],
    picker_strings: Dict[Tuple[Target, str], str],
    improvement_threshold_percentage: float,
) -> bool:
    global warnings, total_specializations, improvement_count, new_specialization_count, noisy_rejected_count, slower_rejected_count, marginal_improvements_rejected_count

    improved = False
    rows = []
    Z_SCORE = 1.96  # 95% confidence

    # Collect all rows first to determine max widths
    for target, new_target_specializations in new_config["specializations"].items():
        if Target() == target:
            continue
        if target not in new_alg_data:
            warnings.append(
                f"{colors.WARN}The new JSON data is missing {target} for {algorithm_name}{colors.END_COLOR}"
            )
            continue

        new_target_data = new_alg_data[target]
        for instance_key, new_instance_data in list(new_target_specializations.items()):
            if instance_key in {"begin_of_picker", "end_of_picker"}:
                picker_strings[(target, instance_key)] = new_instance_data
                continue
            if instance_key not in new_target_data:
                sys.exit(
                    f"{colors.FAIL}The new JSON data is missing {target} specialization '{stringify_instance_key(instance_key)}' for {algorithm_name}{colors.END_COLOR}"
                )
            new_instances = new_target_data[instance_key]

            add_base_args(new_instance_data["base_args"], new_instances)
            score_assigner(new_instances)
            new_best_instance = get_best_instance(new_instances)

            total_specializations += 1

            row = {}
            row["target"] = target.as_str()
            row["key"] = stringify_instance_key(instance_key)
            row["new_family_index"] = new_best_instance["family_index"]
            row["new_bps"] = f"{new_best_instance['bytes_per_second']:.2e}"

            # If there is no old config specialization, we always accept the new one.
            if target not in contenders or instance_key not in contenders[target]:
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
                if target not in contenders:
                    contenders[target] = {}
                contenders[target][instance_key] = Contender(
                    instance=new_best_instance, string=new_instance_data["string"]
                )
                continue

            old_best_instance = contenders[target][instance_key].instance

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
            contenders[target][instance_key] = Contender(
                instance=new_best_instance, string=new_instance_data["string"]
            )

    columns = [
        ("status", f"Status of {algorithm_name}"),
        ("noise", "Noise (old/new)"),
        ("bps", "Bytes/sec (old/new)"),
        ("target", "Target"),
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
):
    global warnings

    contenders: Dict[Target, Dict[Any, Contender]] = {}
    picker_strings: Dict[Tuple[str, str], str] = {}

    old_config.setdefault("specializations", {})

    for target, old_target_specializations in old_config["specializations"].items():
        if Target() == target:
            continue
        contenders[target] = {}
        # Always keep every old specialization, even if missing in old_alg_data
        if target not in old_alg_data:
            warnings.append(
                f"{colors.WARN}The old JSON data is missing {target} for {algorithm_name}{colors.END_COLOR}"
            )
            old_target_data = {}
        else:
            old_target_data = old_alg_data[target]

        for instance_key, old_instance_data in old_target_specializations.items():
            if instance_key in {"begin_of_picker", "end_of_picker"}:
                picker_strings[(target, instance_key)] = old_instance_data
                continue
            if instance_key not in old_target_data:
                # If old_target_data is falsy, then a warning was already printed
                if old_target_data:
                    warnings.append(
                        f"{colors.WARN}The old JSON data is missing {target} specialization '{stringify_instance_key(instance_key)}' for {algorithm_name}{colors.END_COLOR}"
                    )
                old_instances = []
            else:
                old_instances = old_target_data[instance_key]

            add_base_args(old_instance_data["base_args"], old_instances)
            score_assigner(old_instances)

            # If old_instances is empty, just use the config string as-is
            old_best_instance = (
                get_best_instance(old_instances) if old_instances else {}
            )

            contenders[target][instance_key] = Contender(
                instance=old_best_instance, string=old_instance_data["string"]
            )

    return (contenders, picker_strings)


def get_comp_targets(old_config: Dict[str, Any], new_config: Dict[str, Any], algorithm_name: str):
    ret = f"// All existing configs\nusing {algorithm_name}_targets = comp_targets<"

    unique_targets = []
    for target in new_config["specializations"].keys():
        if target != Target() and target not in unique_targets:
            unique_targets.append(target)
    for target in old_config["specializations"].keys():
        if target != Target() and target not in unique_targets:
            unique_targets.append(target)
    
    for unique_target in unique_targets:
        ret += f"comp_target<{unique_target.as_str()}>,"

    ret += f"comp_target<{Target().as_str()}>>;"
    return ret

def get_best_instance(instances):
    return max(instances, key=lambda x: x["score"])

def parse_args(base_args):
    numbers = []
    for item in base_args:
        numbers.extend(map(int, re.findall(r'\d+', item)))
    return numbers

def add_base_args(base_args, instances):
    """
    Adds base arguments from a config to instances, for get_score_assigner().
    """
    for instance in instances:
        if instance["algo"] in {"merge_sort_block_sort", "radix_sort_block_sort"}:
            args = parse_args(base_args)
            instance["bs"] = int(args[0])
            instance["ipt"] = int(args[1])


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
        contenders, picker_strings = get_old_contenders(
            algorithm_name, old_config or {}, old_alg_data or {}, score_assigner
        )
        improved = add_new_contenders(
            algorithm_name,
            new_config,
            new_alg_data,
            score_assigner,
            contenders,
            picker_strings,
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
            # Add every picker for every different target
            for target, target_contenders in contenders.items():
                f.write(picker_strings[(target, "begin_of_picker")])
                for contender in target_contenders.values():
                    f.write(contender.string)
                f.write(picker_strings[(target, "end_of_picker")])

            # Add unknown target fallback case
            unknown_target_contender = (
                old_config["specializations"][Target()]["begin_of_picker"]
                if old_config
                else new_config["specializations"][Target()]["begin_of_picker"]
            ).lstrip("\n")

            f.write(unknown_target_contender)

            # Add comp_targets
            comp_targets = get_comp_targets(old_config, new_config, algorithm_name)
            f.write(comp_targets)
            
            # Remove leading newlines from end_of_config to avoid triple newlines
            end = (
                old_config["end_of_config"]
                if old_config
                else new_config["end_of_config"]
            )
            if not end.endswith("\n"):
                end += "\n"
            f.write(end)


def extract_template_arguments(code: str) -> list[str]:
    """
    Extracts the top-level template arguments from a base config class in a C++ struct specialization.

    Assumes the input contains a line ending in `: XXX_config_params{...} {`, where `XXX_config_params` is
    the name of the base config params class and the curly brackets contain template arguments.

    Args:
        code (str): A string containing the C++ struct definition.

    Returns:
        List[str]: A list of top-level template arguments as strings.
                   Nested structs like `kernel_config_params{1024, 1}` are preserved as single items.

    Example:
        Input:
            ```
            if constexpr(true)
            {
                return some_config_params{256, kernel_config_params{128, 2}, 1 << 2};
            }
            ```
        Output:
            ['256', 'kernel_config_params{128, 2}', '1 << 2']
    """
    # Match the base class ending in `_config<...> {`
    match = re.search(r"return\s+\w+_config_params\s*\{\s*([\s\S]*?)\s*\};", code, re.DOTALL)

    if not match:
        return []

    template_args = match.group(1)

    # Split top-level arguments by comma, keeping nested templates intact
    args = []
    current = ""
    depth = 0
    i = 0
    while i < len(template_args):
        if template_args[i : i + 2] in ("{{", "}}"):
            current += template_args[i : i + 2]
            i += 2
            continue

        char = template_args[i]
        if char == "{":
            depth += 1
        elif char == "}":
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
#         'comp_targets': 'using select_flag_targets = comp_targets< ... >;'
#         'end_of_config': '} // end namespace detail ...',
#         'specializations': {
#             Target(gen::rdna2, target_arch::gfx1030, gpu::v620, rep::amdgcn): {
#                 'begin_of_picker': 'template<class Target, class data_type> constexpr auto select_flag_config_picker() ... {\n',
#                 'end_of_picker': `// Default case if none of the conditions match\nreturn partition_config_params_base<data_type>();}\n`,
#                 Instance(key_type='double'): {
#                     'string': '// Based on key_type = double\n ... partition_config_params{512, kernel_config_params{128, 2}};\n',
#                     'base_args': [
#                         '512',
#                         'kernel_config_params{128, 2}'
#                     ]
#                 },
#                 Instance(key_type='float'): {
#                     'string': '// Based on key_type = float\n ... partition_config_params{1024, kernel_config_params{128, 2}};\n',
#                     'base_args': [
#                         '1024',
#                         'kernel_config_params{128, 2}'
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
        pickers = re.split(r"(template\s*<[\s\S]*?>\s*\{[\s\S]*?}\n\n)", text, flags=re.DOTALL)
        pickers = list(filter(None, pickers))
        # Remove spaces from picker functions to have a more consistent input.
        pickers = [picker for picker in pickers if picker.strip() != ""]

        # All the code before the picker functions.
        config["start_of_config"] = pickers[0]

        # The code after the picker functions need to be divided in comp_targets and the end.
        comp_target = re.split(r"(// .*\nusing[\s\S]*?;)", pickers[-1], flags=re.DOTALL)
        comp_target = list(filter(None, comp_target))
        config["comp_targets"] = comp_target[0]
        config["end_of_config"] = comp_target[-1]

        all_specializations = config.setdefault("specializations", {})

        picker_strings = pickers[1:-1]
        for picker_string in picker_strings:
            target_match = re.search(r"comp_target<((.||\s)*?)>", picker_string)

            if not target_match:
                sys.exit(
                    f"{colors.FAIL}Could not find arch in specialization: {picker_string}{colors.END_COLOR}"
                )

            target = Target.from_string(target_match.group(1))

            specializations = re.split(r"(\s*// Based on[\s\S]*?\}\n(?=\s*//))", picker_string, flags=re.DOTALL)
            specializations = list(filter(None, specializations))
            target_specializations = all_specializations.setdefault(target, {})
            target_specializations["begin_of_picker"] = specializations[0]
            target_specializations["end_of_picker"] = specializations[-1]

            for specialization_string in specializations[1:-1]:
                specialization_key = get_specialization_key(specialization_string)

                if specialization_key in target_specializations:
                    sys.exit(
                        f"{colors.FAIL}Specialization key duplicate '{specialization_key}'{colors.END_COLOR}"
                    )
                args = extract_template_arguments(specialization_string)
                if args is []:
                    sys.exit(
                        f"{colors.FAIL}Arguments are empty '{specialization_key}'{colors.END_COLOR}"
                    )
                target_specializations[specialization_key] = {
                    "string": specialization_string,
                    "base_args": args,
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
#         Target(gen::rdna2, target_arch::gfx1030, gpu::v620, rep::amdgcn): {
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
        target = get_target(json_data["context"])

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
            target_data = alg_data.setdefault(target, defaultdict(list))

            if algorithm_name not in selectors:
                sys.exit(
                    f"{colors.FAIL}No selectors found for algorithm '{algorithm_name}' in the selectors JSON{colors.END_COLOR}"
                )
            instance_key = get_instance_key(data, selectors[algorithm_name])
            target_data[instance_key].append(data)

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
        // Based on key_type = double, value_type = rocprim::int128_t
        if constexpr((bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                    && (sizeof(key_type) > 4) && (sizeof(value_type) <= 16)
                    && (sizeof(value_type) > 8)))
        {
            return radix_sort_onesweep_config_params{
                kernel_config_params{1024, 1},
                (1 << 17) + 1 >> 2 + 70000,
                8,
                block_radix_rank_algorithm::match
            };
        }
        // Needs a comment after it.
        """

        expected = [
            "kernel_config_params{1024, 1}",
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

    if total_specializations == 0:
        warnings.append(
            f"{colors.WARN}No specializations are found in the config files!{colors.END_COLOR}"
        )

    if warnings:
        print("\n".join(warnings))
        print("")

    print_results()


if __name__ == "__main__":
    main()
