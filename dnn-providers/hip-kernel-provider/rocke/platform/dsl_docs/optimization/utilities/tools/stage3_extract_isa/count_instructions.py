# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Count key ISA instructions in AMD GCN/CDNA assembly.

Provides instruction counting, pattern analysis, and ISA statistics
for performance debugging and kernel optimization.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ISAInstructionCounter:
    """Count and categorize instructions in AMD GCN/CDNA assembly."""

    # Instruction patterns grouped by category
    INSTRUCTION_PATTERNS = {
        "compute": [
            ("v_mfma_*", r"v_mfma"),
            ("v_fma_*", r"v_fma_"),
            ("v_mac_*", r"v_mac_"),
            ("v_dot_*", r"v_dot"),
        ],
        "lds": [
            ("ds_read_*", r"ds_read"),
            ("ds_write_*", r"ds_write"),
        ],
        "global_memory": [
            ("buffer_load_*", r"buffer_load"),
            ("buffer_store_*", r"buffer_store"),
            ("global_load_*", r"global_load"),
            ("global_store_*", r"global_store"),
            ("flat_load_*", r"flat_load"),
            ("flat_store_*", r"flat_store"),
        ],
        "synchronization": [
            ("s_waitcnt", r"s_waitcnt"),
            ("s_barrier", r"s_barrier"),
        ],
        "scheduling": [
            ("s_sched_barrier", r"s_sched_barrier"),
            ("s_sched_group_barrier", r"s_sched_group_barrier"),
            ("s_setprio", r"s_setprio"),
        ],
        "control_flow": [
            ("s_cbranch_*", r"s_cbranch"),
            ("s_branch", r"s_branch"),
        ],
        "register_spilling": [
            ("v_accvgpr_write", r"v_accvgpr_write"),
            ("v_accvgpr_read", r"v_accvgpr_read"),
        ],
    }

    def __init__(self, isa_file: Path):
        """Initialize counter with ISA file.

        Args:
            isa_file: Path to AMD GCN/CDNA assembly file (.s)
        """
        self.isa_file = Path(isa_file)
        if not self.isa_file.exists():
            raise FileNotFoundError(f"ISA file not found: {isa_file}")

        with open(self.isa_file, "r") as f:
            self.lines = f.readlines()

    def count_all(self) -> Dict[str, Dict[str, int]]:
        """Count all instruction categories.

        Returns:
            Dictionary mapping category -> {instruction_pattern: count}
        """
        results = {}

        for category, patterns in self.INSTRUCTION_PATTERNS.items():
            results[category] = {}
            for pattern_name, regex in patterns:
                count = sum(1 for line in self.lines if re.search(regex, line))
                results[category][pattern_name] = count

        return results

    def find_mfma_contexts(
        self, context_lines: int = 10
    ) -> List[Tuple[int, List[str]]]:
        """Find MFMA instructions with surrounding context.

        Args:
            context_lines: Number of lines before/after to include

        Returns:
            List of (line_number, context_lines) tuples
        """
        contexts = []

        for i, line in enumerate(self.lines):
            if re.search(r"v_mfma", line):
                start = max(0, i - context_lines)
                end = min(len(self.lines), i + context_lines + 1)
                context = self.lines[start:end]
                contexts.append((i + 1, context))  # 1-indexed line number

        return contexts

    def find_loop_structure(self) -> List[Tuple[int, str]]:
        """Find loop labels and branch instructions.

        Returns:
            List of (line_number, line_content) tuples
        """
        loop_structure = []

        # Pattern for basic block labels (e.g., BB0_1:)
        label_pattern = r"BB\d+_\d+:"

        for i, line in enumerate(self.lines):
            if re.search(label_pattern, line) or re.search(r"s_cbranch", line):
                loop_structure.append((i + 1, line.strip()))

        return loop_structure

    def extract_kernel_metadata(self) -> Dict[str, any]:
        """Extract kernel metadata from assembly header.

        Returns:
            Dictionary with VGPR, SGPR, LDS, occupancy info
        """
        metadata = {
            "vgpr_count": None,
            "sgpr_count": None,
            "agpr_count": None,
            "lds_bytes": None,
            "workgroup_size_x": None,
            "workgroup_size_y": None,
            "workgroup_size_z": None,
        }

        # AMDHSA metadata patterns
        patterns = {
            "vgpr_count": r"amdhsa_next_free_vgpr\s+(\d+)",
            "sgpr_count": r"amdhsa_next_free_sgpr\s+(\d+)",
            "agpr_count": r"amdhsa_accum_offset\s+(\d+)",
            "lds_bytes": r"amdhsa_group_segment_fixed_size\s+(\d+)",
            "workgroup_size_x": r"amdhsa_system_vgpr_workitem_id\s+(\d+)",
        }

        for key, pattern in patterns.items():
            for line in self.lines[:200]:  # Check first 200 lines for metadata
                match = re.search(pattern, line)
                if match:
                    metadata[key] = int(match.group(1))
                    break

        return metadata

    def print_summary(self, name: Optional[str] = None):
        """Print formatted instruction count summary.

        Args:
            name: Optional name for the kernel (defaults to filename)
        """
        if name is None:
            name = self.isa_file.name

        print(f"{'=' * 80}")
        print(f"ISA Analysis: {name}")
        print(f"{'=' * 80}")
        print(f"Total lines: {len(self.lines)}")
        print()

        counts = self.count_all()

        # Print each category
        for category, instructions in counts.items():
            category_name = category.replace("_", " ").title()
            print(f"{category_name}:")
            for instr_name, count in instructions.items():
                print(f"  {instr_name:<25} {count}")
            print()

        # Print metadata
        metadata = self.extract_kernel_metadata()
        if any(v is not None for v in metadata.values()):
            print("Kernel Metadata:")
            for key, value in metadata.items():
                if value is not None:
                    print(f"  {key:<25} {value}")
            print()


def compare_isa_files(file1: Path, file2: Path, name1: str = None, name2: str = None):
    """Compare instruction counts between two ISA files.

    Args:
        file1: First ISA file
        file2: Second ISA file
        name1: Optional name for first file
        name2: Optional name for second file
    """
    counter1 = ISAInstructionCounter(file1)
    counter2 = ISAInstructionCounter(file2)

    name1 = name1 or file1.name
    name2 = name2 or file2.name

    counts1 = counter1.count_all()
    counts2 = counter2.count_all()

    print(f"{'=' * 80}")
    print(f"ISA Comparison: {name1} vs {name2}")
    print(f"{'=' * 80}")
    print()

    print(f"{'Instruction':<30} {name1:<20} {name2:<20} {'Delta':<15}")
    print(f"{'-' * 85}")

    for category in counts1.keys():
        print(f"\n{category.replace('_', ' ').title()}:")
        for instr_name in counts1[category].keys():
            c1 = counts1[category][instr_name]
            c2 = counts2[category][instr_name]
            delta = c2 - c1
            delta_str = f"{delta:+d}" if delta != 0 else "same"

            print(f"  {instr_name:<28} {c1:<20} {c2:<20} {delta_str:<15}")

    print()


def main():
    """Command-line interface for ISA analysis."""
    parser = argparse.ArgumentParser(
        description="Count and analyze AMD GCN/CDNA assembly instructions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Count instructions in a single file
  %(prog)s kernel.s

  # Count with custom name
  %(prog)s kernel.s --name "CK Tile v1"

  # Compare two kernels
  %(prog)s kernel1.s kernel2.s --compare

  # Extract MFMA contexts
  %(prog)s kernel.s --mfma-context 15
        """,
    )

    parser.add_argument("isa_file", type=Path, help="ISA file to analyze (.s)")
    parser.add_argument(
        "isa_file2", type=Path, nargs="?", help="Second ISA file for comparison"
    )
    parser.add_argument("--name", help="Name for the kernel")
    parser.add_argument("--name2", help="Name for second kernel (comparison mode)")
    parser.add_argument("--compare", action="store_true", help="Compare two ISA files")
    parser.add_argument(
        "--mfma-context",
        type=int,
        metavar="N",
        help="Show N lines of context around each MFMA instruction",
    )
    parser.add_argument(
        "--loop-structure", action="store_true", help="Show loop labels and branches"
    )

    args = parser.parse_args()

    if args.compare or args.isa_file2:
        if not args.isa_file2:
            parser.error("--compare requires two ISA files")
        compare_isa_files(args.isa_file, args.isa_file2, args.name, args.name2)
    else:
        counter = ISAInstructionCounter(args.isa_file)
        counter.print_summary(args.name)

        if args.mfma_context:
            print(f"{'=' * 80}")
            print(f"MFMA Contexts (±{args.mfma_context} lines)")
            print(f"{'=' * 80}")
            contexts = counter.find_mfma_contexts(args.mfma_context)
            for line_num, context in contexts[:3]:  # Show first 3 MFMAs
                print(f"\nAround line {line_num}:")
                print("".join(context))

        if args.loop_structure:
            print(f"{'=' * 80}")
            print("Loop Structure")
            print(f"{'=' * 80}")
            structure = counter.find_loop_structure()
            for line_num, line_content in structure[:20]:  # Show first 20
                print(f"  {line_num:5d}: {line_content}")


if __name__ == "__main__":
    main()
