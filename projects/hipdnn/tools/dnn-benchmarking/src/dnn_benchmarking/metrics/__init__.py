# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Metric probes and derivations for dnn-benchmarking.

Always-on probes (zero overhead, wrapped around the timed loop):
    * Analytical FLOPs / IO bytes from graph JSON (:mod:`analytical`).
    * Host CPU rusage delta and host RAM snapshot (:mod:`host`).
    * GPU telemetry snapshot via amdsmi (:mod:`gpu_smi`).
    * One-shot machine metadata (:mod:`machine_info`).

Opt-in profiling sources (separate workload re-run, orchestrated via
:mod:`profiling_orchestrator`):
    * rocprofv3 PMC counters (:mod:`rocprof_pmc`).
    * rocprofv3 kernel/memory trace (:mod:`rocprof_trace`).
    * Linux ``perf stat`` CPU counters (:mod:`perf`).
    * ``rocprof-compute --roof-only`` roofline plot (:mod:`roofline`).

Each opt-in source runs after the timed pass so PMC sampling and roof
replay can't pollute the headline timing. Results land in
``ProviderEngineResult.extra_metrics`` under per-source keys.
"""

from .analytical import (
    compute_flops,
    compute_io_bytes,
    derive_throughputs,
    list_unsupported_node_types,
)
from .gpu_smi import GpuSmiProbe, is_amdsmi_available
from .host import CpuTimeDelta, CpuTimeProbe, host_memory_snapshot, is_psutil_available
from .machine_info import collect_machine_info

__all__ = [
    "compute_flops",
    "compute_io_bytes",
    "derive_throughputs",
    "list_unsupported_node_types",
    "GpuSmiProbe",
    "is_amdsmi_available",
    "CpuTimeDelta",
    "CpuTimeProbe",
    "host_memory_snapshot",
    "is_psutil_available",
    "collect_machine_info",
]
