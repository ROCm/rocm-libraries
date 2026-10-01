# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Vector-width fallback shared by the GEMM full-benchmark drivers.

A problem whose contiguous A/B/C extent is not a multiple of the native vector
width (e.g. K=257 for a row-major A) cannot run any native kernel. With the
fallback on, the sweep also builds kernels with narrower fixed widths, reports
how many (tile, width) combinations were rejected or failed to compile, and
pairs every problem only with the kernels whose widths divide its own.
"""

import itertools

from codegen_common import (
    CommonTypeMappings,
    VECTOR_SIZE_VARIANTS,
    gemm_problem_vector_sizes,
    gemm_vector_size_sweep,
)


def add_vector_fallback_arg(parser):
    parser.add_argument(
        "--no-vector-fallback",
        action="store_true",
        help="Build only native-vector-width kernels. By default, problems whose "
        "contiguous A/B/C extents are not a multiple of the native width also get "
        "kernels with narrower fixed widths (gcd of extent and native width)",
    )


class VectorFallback:
    """Per-run fallback state: problem widths, sweep kwargs and reject counts."""

    def __init__(self, problems, layout, dtype, variant, disabled=False):
        out_dtype = CommonTypeMappings.get_output_dtype(dtype)
        # Per-problem widest legal A/B/C widths; a kernel may run a problem iff
        # each of its effective widths divides the problem's.
        self.prob_vecs = [
            gemm_problem_vector_sizes(
                int(p["M"]), int(p["N"]), int(p["K"]), layout[:3], dtype, dtype, out_dtype
            )
            for p in problems
        ]
        self.enabled = not disabled and variant in VECTOR_SIZE_VARIANTS
        self.rejects = {}
        self.expand_kwargs = {}
        if self.enabled:
            # Every power-of-two width <= the problem's, for misaligned tensors
            # only; the per-problem winner among them is the width's cost.
            sweep = {
                t
                for v in self.prob_vecs
                for t in gemm_vector_size_sweep(v, dtype, dtype, out_dtype)
            }
            self.expand_kwargs = dict(
                vector_sizes=sorted({(0, 0, 0), *sweep}), rejects=self.rejects
            )
            print(f"  Vector-width fallback triples: {self.expand_kwargs['vector_sizes']}")

    @staticmethod
    def limit_base_kernels(configs, max_kernels):
        """First ``max_kernels`` native kernels, each with its fixed-width variants.

        expand_sweep emits every native config right before its fixed-width
        variants, so --max-kernels counts tiles and a small limit still keeps the
        kernels misaligned problems need; without the fallback it is a plain slice.
        """
        if max_kernels <= 0:
            return configs
        n_native = itertools.accumulate(not any(c.vector_sizes) for c in configs)
        return [c for c, n in zip(configs, n_native) if n <= max_kernels]

    def report_rejects(self):
        for reason, n in sorted(self.rejects.items()):
            print(f"  Vector-width reject ({n}x): {reason}")

    def report_builds(self, configs, lib_paths):
        failed = [
            c.name for c, lib in zip(configs, lib_paths) if lib is None and any(c.vector_sizes)
        ]
        for name in failed:
            print(f"  Build FAILED for fixed vector widths: {name}")
        if self.enabled:
            n_vec = sum(1 for c in configs if any(c.vector_sizes))
            n_rej = sum(self.rejects.values())
            print(
                f"  Vector-width summary: {n_vec + n_rej} fixed-width kernels requested, "
                f"{n_rej} rejected before compile, "
                f"{len(failed)} failed to compile, {n_vec - len(failed)} built"
            )

    def pairs(self, problems, built_kernels):
        """Kernel indices each problem may run (all kernels when disabled)."""
        if self.enabled:
            kernel_vecs = [cfg.effective_vector_sizes for cfg, _ in built_kernels]
            pairs = [
                [i for i, kv in enumerate(kernel_vecs) if all(p % k == 0 for p, k in zip(pv, kv))]
                for pv in self.prob_vecs
            ]
        else:
            pairs = [list(range(len(built_kernels)))] * len(problems)
        n_meas = sum(map(len, pairs))
        print(f"  Problems: {len(problems)}")
        print(
            f"  Total measurements: {n_meas} "
            f"({len(built_kernels) * len(problems) - n_meas} vector-width-incompatible "
            f"pairs skipped)"
        )
        for prob, pv, idx in zip(problems, self.prob_vecs, pairs):
            if not idx:
                print(
                    f"  WARNING: no built kernel supports problem "
                    f"{prob['M']}x{prob['N']}x{prob['K']} (widths {pv})"
                )
        return pairs
