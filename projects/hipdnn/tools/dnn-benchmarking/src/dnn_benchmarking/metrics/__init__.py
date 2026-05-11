# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Always-on metric probes and derivations for dnn-benchmarking.

Phase 1 surface:
    * Analytical FLOPs / IO bytes from graph JSON (:mod:`analytical`).
    * Host CPU rusage delta and host RAM snapshot (:mod:`host`).
    * GPU telemetry snapshot via amdsmi (:mod:`gpu_smi`).
    * One-shot machine metadata (:mod:`machine_info`).

Phase 2/3 sources (rocprofv3 PMC + trace, perf, rocprof-compute roofline)
are documented in ``docs/metrics-phase2.md`` and
``docs/metrics-phase3.md`` and will land in the ``extra_metrics`` bag on
``ProviderEngineResult`` without breaking the Phase 1 schema.
"""

from .analytical import (
    compute_flops,
    compute_io_bytes,
    derive_throughputs,
    list_unsupported_node_types,
)
from .gpu_smi import GpuSmiProbe, is_amdsmi_available
from .host import RusageDelta, RusageProbe, host_memory_snapshot, is_psutil_available
from .machine_info import collect_machine_info

__all__ = [
    "compute_flops",
    "compute_io_bytes",
    "derive_throughputs",
    "list_unsupported_node_types",
    "GpuSmiProbe",
    "is_amdsmi_available",
    "RusageDelta",
    "RusageProbe",
    "host_memory_snapshot",
    "is_psutil_available",
    "collect_machine_info",
]
