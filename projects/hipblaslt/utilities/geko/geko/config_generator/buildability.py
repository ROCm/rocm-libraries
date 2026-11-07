# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which candidate MatrixInstructions can TensileLite build? Code generation only (CPU, no GPU).

An MI is *buildable* for a GEMM type when TensileLite code generation builds at least one kernel
for it with every other parameter inside geko's generic (GA) tuning ranges, i.e. a kernel a geko
GA run can produce. Kernels fail over a resource limit (registers, LDS, occupancy) or on a validity rule.
Lists are generated with StreamK off, which also covers StreamK-on runs; heuristic runs read the
same list. Subtile kernels are not covered.

Search per MI, stopping at the first kernel that builds:
  1. Presets: settings that build most tiles.
  2. Factorial: every combination of the parameters that decide register and LDS use, over the
     values the architecture's space tunes.
  R. Repair: the nearest settings TensileLite accepts to each refused one (validation only), code-generated.
  3. One parameter at a time, starting from the kernels closest to fitting.
Each buildable MI is then searched at every DepthU of the space (buildable_depthus).
"""

from __future__ import annotations

import contextlib
import copy
import io
import itertools
import json
import math
import multiprocessing as mp
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from geko.config_generator.constants import dataSize, library_logic_architecture
from geko.config_generator.fork_params import get_profile_class
from geko.config_generator.fork_params.hw_profiles.gfx942.optimization_param import GFX942GAParams
from geko.config_generator.fork_params.hw_profiles.gfx950.optimization_param import GFX950GAParams
from geko.config_generator.fork_params.hw_profiles.gfx1250.optimization_param import GFX1250GAParams
from geko.config_generator.mi_candidates import MI, enumerate_candidate_mis

# Any ordinary size: the generic ranges do not depend on it (M or N = 1 drops DepthU on gfx942).
PROBE_SIZE = [4096, 4096, 1, 4096]

# Settings tried first, per generic (GA) profile: DepthU "K" = the smallest legal DepthU, positive read widths in
# 2-byte elements. They only order the search; a profile without an entry starts from the factorial.
_GFX9_PRESETS = [
    {
        "DepthU": "K",
        "PrefetchGlobalRead": 2,
        "1LDSBuffer": 0,
        "ClusterLocalRead": 1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "DirectToLds": 1,
        "UseSgprForGRO": 0,
    },
    {
        "DepthU": "K",
        "PrefetchGlobalRead": 1,
        "1LDSBuffer": 1,
        "ClusterLocalRead": 1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "DirectToLds": 0,
        "UseSgprForGRO": 0,
    },
    {
        "DepthU": "K",
        "PrefetchGlobalRead": 1,
        "1LDSBuffer": 0,
        "ClusterLocalRead": 1,
        "GlobalReadVectorWidthA": 2,
        "GlobalReadVectorWidthB": 2,
        "DirectToLds": 1,
        "UseSgprForGRO": 0,
    },
]
LIST_PRESETS = {
    GFX950GAParams: _GFX9_PRESETS,
    GFX942GAParams: _GFX9_PRESETS,
    # HalfPLR x ClusterLocalRead: the only register levers gfx1250's space tunes (sweeps over 220 MIs, 6 non-FP4 types).
    GFX1250GAParams: [{"DepthU": "K", "HalfPLR": h, "ClusterLocalRead": c} for c in (0, 1) for h in (0, 2, 1, 3)],
}

# Base value of each parameter outside the factorial: the cheapest the space allows (parameters a space does not tune
# are ignored). Every kernel the search tries starts from these values, so they shape what it reaches, not just order.
LOW_RESOURCE = {
    "MIArchVgpr": False,
    "TransposeLDS": -1,
    "StoreVectorWidth": -1,
    "SourceSwap": False,
    "TailloopInNll": False,
    "UnrollLoopSwapGlobalReadOrder": 0,
    "WaveSeparateGlobalReadA": 0,
    "WaveSeparateGlobalReadB": 0,
    "NumElementsPerBatchStore": 0,
    "ScheduleGROverBarrier": 0,
    "ExtraMiLatencyLeft": -1,
    "StorePriorityOpt": False,
    "StoreSyncOpt": 0,
    "WorkGroupMapping": 0,
    "PrefetchGL2": 0,
    "TDMIterateMode": 0,
    "LDSSegmentInterleave": 0,
    "AdaptiveGemm": 0,
}


OFAT_PASSES = 2
MULTI_STARTS = 4

# Repair limits: refusals repaired per MI, accepted settings kept per repair, validations for pairs of changes.
REPAIRS = 8
REPAIR_KEEP = 4
REPAIR_PAIRS = 2000

# Stop after _initKernel when a register pool is this far over its cap: pools only grow, so the
# kernel cannot build. Near misses are generated in full so stage 3 can rank them by overflow.
EARLY_EXIT_MARGIN = 1.2


# The search space whose ranges define "buildable": the GA's.
LIST_SEARCH_SPACE = "generic"

# StreamK kernels need slightly more registers, so a StreamK-off list also covers StreamK-on runs.
LIST_STREAMK = False


# ---------------------------------------------------------------------------- the search space
@dataclass
class TuningSpace:
    """Fork-parameter ranges of a GEMM type's generic tuning config, as TensileLite sees them."""

    config_path: Path
    fork: Dict[str, List[Any]]  # fork parameter -> its values, in config order
    groups: Dict[str, List[Dict[str, Any]]]  # non-MI group name -> its entries
    mi_group: str  # group carrying MatrixInstruction
    wavefront: int = 64  # lanes per wave: MI Designer's WorkGroup and the DirectToVgpr read widths depend on it


def write_probe_config(
    config: Dict[str, Any], hipblaslt_path: Path, workdir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """geko's generic (GA) tuning config for PROBE_SIZE, MI Designer on every candidate MI.

    ``config`` is an input config dict (TRANSA, TRANSB, DataType, DestDataType,
    ComputeDataType, ARCH and other keys); ARCH defaults apply.
    Returns (config path, the prepared config, with GemmProblems).
    """
    from geko.config_generator.config_generator import run
    from geko.config_generator.constants import ENV_UPDATABLE_KEYS
    from geko.config_generator.load_input_config import (
        apply_input_config_defaults,
        get_gemm_problem,
        validate_input_config,
    )

    cfg = dict(
        copy.deepcopy(config),
        backend="ductile",
        search_space=LIST_SEARCH_SPACE,
        Sizes=[PROBE_SIZE],
        SIZE_OPTION=0,
        ONE_SIZE_PER_CONFIG=True,
        CLUSTER=0,
        BUILDABLE_MIS=False,
        StreamK=LIST_STREAMK,
    )
    # geko lets same-name environment variables override these keys (StreamK, BUILDABLE_MIS, ...): hide them so
    # the probe holds exactly the settings above, whatever the shell exports.
    hidden = {k: os.environ.pop(k) for k in ENV_UPDATABLE_KEYS if k in os.environ}
    try:
        validate_input_config(cfg)
        apply_input_config_defaults(cfg)
        get_gemm_problem(cfg)
        workdir.mkdir(parents=True, exist_ok=True)
        run(cfg, hipblaslt_path, workdir, write_shell_scripts=False)
    finally:
        os.environ.update(hidden)
    name = cfg["GemmProblems"][0].name
    paths = list(workdir.glob(f"{name}_*.yaml"))
    if len(paths) != 1:
        raise RuntimeError(f"expected 1 tuning config for {name} in {workdir}, found {len(paths)}")
    return paths[0], cfg


def _group_signature(entries: Sequence[Dict[str, Any]]) -> str:
    return "group_" + "_".join(sorted({k for e in entries for k in e}))


def _is_cms(entry: Dict[str, Any]) -> bool:
    return entry.get("UseCustomMainLoopSchedule", 0) not in (0, None)


# ---------------------------------------------------------------------------- code generation
@dataclass
class Measurement:
    err: int  # 0 builds; -1 refused before code generation; N overflowedResources code; -4 other exception
    vgpr: Optional[int] = None
    agpr: Optional[int] = None
    sgpr: Optional[int] = None
    lds: Optional[int] = None
    vgpr_cap: Optional[int] = None
    sgpr_cap: Optional[int] = None
    lds_cap: Optional[int] = None
    occ_regs: Optional[int] = None  # registers per lane as the occupancy check counts them (_occupancy)
    occ_cap: Optional[int] = None  # per-lane register budget for one workgroup per execution unit
    reason: str = ""

    def _usage(self) -> Dict[str, float]:
        """Each modelled resource limit (TensileLite overflow codes 1, 2, 4, 8) as the fraction of its cap in use."""
        return {
            "vgpr": self.vgpr / self.vgpr_cap,
            "sgpr": self.sgpr / self.sgpr_cap,
            "lds": self.lds / self.lds_cap,
            "occupancy": self.occ_regs / self.occ_cap,
        }

    def overflow(self) -> float:
        """Fraction by which the most over-used resource exceeds its cap; <= 0 fits, inf when no
        resource counts exist."""
        if self.vgpr is None:
            if self.lds is not None and self.lds_cap:  # LDS rejection before code generation
                return self.lds / self.lds_cap - 1
            return math.inf
        return max(self._usage().values()) - 1

    def resource(self) -> str:
        """The most over-used resource ("" when no resource counts exist)."""
        if self.vgpr is None:
            return "lds" if self.lds is not None and self.lds_cap else ""
        usage = self._usage()
        return max(usage, key=usage.get)


def _occupancy(kw, sol) -> Tuple[int, int]:
    """Registers per lane and their budget, as TensileLite's occupancy check (checkResources, code 4) counts them."""
    v, a = kw.vgprPool.size(), kw.agprPool.size()
    regs = math.ceil(v / 8) * 8 + a if kw.states.archCaps["ArchAccUnifiedRegs"] else max(v, a)
    per_unit = kw.states.regCaps["PhysicalMaxVgprCU"] * (2 if kw.states.asmCaps["HasWMMA"] else 1)
    return regs, per_unit // sol["NumThreads"]


class CodegenChecker:
    """TensileLite set up as ``Tensile <config> --cpu-only --gpu-targets <arch>`` does, for one GEMM type."""

    def __init__(self, config_path: Path, arch: str):
        from tensilelite import LibraryIO
        from tensilelite.BenchmarkStructs import BenchmarkProcess
        from tensilelite.Common import makeDebugConfig
        from tensilelite.Common.Architectures import gfxToIsa
        from tensilelite.Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
        from tensilelite.Common.Constants import coVersionMap
        from tensilelite.Common.GlobalParameters import (
            assignGlobalParameters,
            globalParameters,
            restoreDefaultGlobalParameters,
        )
        from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
        from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain

        self.arch = library_logic_architecture(arch)
        restoreDefaultGlobalParameters()
        globalParameters["CpuOnly"] = True
        globalParameters["CodeObjectVersion"] = coVersionMap["4"]
        cxx, bundler = validateToolchain(
            ToolchainDefaults.CXX_COMPILER, ToolchainDefaults.OFFLOAD_BUNDLER
        )
        self.assembler = makeAssemblyToolchain(cxx, bundler, "4", debug=False).assembler
        with contextlib.redirect_stdout(io.StringIO()):
            self.isa_info = makeIsaInfoMap([gfxToIsa(self.arch)], cxx)
            applyArchCapOverrides(self.isa_info, [self.arch])

        cfg = LibraryIO.read(str(config_path))
        gp = dict(cfg.get("GlobalParameters", {}))
        gp["PrintSolutionRejectionReason"] = True  # reasons for MIs rejected before code generation
        assignGlobalParameters(gp, self.isa_info)
        self.debug = makeDebugConfig(gp)
        with contextlib.redirect_stdout(io.StringIO()):
            bp = BenchmarkProcess(
                cfg["BenchmarkProblems"][0][0],
                cfg["BenchmarkProblems"][0][1],
                False,
                keyPathPrefix="BenchmarkProblems[0][1]",
                srcFile=str(config_path),
                gfxName=self.arch,
            )
        step = bp[0]
        self.problem_type = bp.problemType
        self.constant = dict(step.constantParams)
        self.space = self._space(config_path, step)
        pt = self.problem_type
        cap = self.constant.get("MaxLDS", -1)
        self.lds_cap = (
            cap
            if isinstance(cap, int) and cap > 0
            else self.isa_info[gfxToIsa(self.arch)].archCaps["DeviceLDS"]
        )
        # Operand bytes for lds_lower_bound; None when sparse metadata or MX scales also use LDS.
        plain = not (pt["Sparse"] or pt["MXBlockA"] or pt["MXBlockB"])
        nbytes = (pt["DataTypeA"].numBytes(), pt["DataTypeB"].numBytes())
        self.lds_bytes = nbytes if plain else None

    def lds_lower_bound(self, perm: Dict[str, Any]) -> Optional[int]:
        """LDS bytes any kernel for ``perm`` needs at least: one DepthU slice of A and of B, except
        an operand DirectToVgpr loads straight into registers. Only the LDS distance of a setting TensileLite
        refuses without counts; TensileLite decides every setting."""
        du = perm.get("DepthU", self.constant.get("DepthU"))
        if self.lds_bytes is None or not isinstance(du, int):
            return None
        from geko.config_generator.mi_designer import MFMA, MIDesign

        mt = MIDesign.calculate_mfma_parameters(
            MFMA.from_list(perm[self.space.mi_group]["MatrixInstruction"])
        )
        dtv = {
            ab: perm.get(f"DirectToVgpr{ab}", self.constant.get(f"DirectToVgpr{ab}")) for ab in "AB"
        }
        a = 0 if dtv["A"] else mt.MT0 * du * self.lds_bytes[0]
        b = 0 if dtv["B"] else mt.MT1 * du * self.lds_bytes[1]
        return int(a + b)

    def _space(self, config_path: Path, step) -> TuningSpace:
        fork = {k: list(v) for k, v in dict(step.forkParams).items()}
        groups: Dict[str, List[Dict[str, Any]]] = {}
        mi_group = None
        for entries in step.paramGroups:
            entries = [dict(e) for e in entries]
            if any("MatrixInstruction" in e for e in entries):
                mi_group = "group_MatrixInstruction"
                # MI Designer sets WorkGroup (LSU > 1) and GlobalSplitU per MI and size (SettingSearch.mi_entries
                # rebuilds them per candidate); a per-MI DepthU (MX / subtile) is a DepthU the space allows.
                for e in entries:
                    if (
                        not _is_cms(e)
                        and "DepthU" in e
                        and e["DepthU"] not in fork.setdefault("DepthU", [])
                    ):
                        fork["DepthU"].append(e["DepthU"])
                continue
            groups[_group_signature(entries)] = entries
        if mi_group is None:
            raise RuntimeError("tuning config carries no MatrixInstruction group")
        for k in list(fork):  # a key also set by a group belongs to the group
            if any(k in e for entries in groups.values() for e in entries):
                del fork[k]
        for k in fork:
            self.constant.pop(k, None)
        wavefront = int((fork.get("WavefrontSize") or [self.constant.get("WavefrontSize", -1)])[0])
        if wavefront == -1:  # auto, resolved as TensileLite does (SolutionStructs/Solution.py): wave32 if the ISA has it
            from tensilelite.Common.Architectures import gfxToIsa

            wavefront = 32 if self.isa_info[gfxToIsa(self.arch)].archCaps["HasWave32"] else 64
        return TuningSpace(config_path, fork, groups, mi_group, wavefront)

    def _initialised(self, perm: Dict[str, Any], buf: io.StringIO):
        """Validate ``perm`` and set up its kernel writer, as Ductile's ``_validate_solution`` does without code
        generation. Returns (solution, writer) or a refusal."""
        from tensilelite.backends.ductile_backend import _generate_single_solution_with_groups
        from tensilelite.KernelWriterAssembly import KernelWriterAssembly

        lb = self.lds_lower_bound(perm)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                sol = _generate_single_solution_with_groups(
                    perm,
                    self.problem_type,
                    self.constant,
                    self.assembler,
                    self.debug,
                    self.isa_info,
                    silent=True,
                )
            except SystemExit:
                sol = None
            if sol is None:
                m = _rejected(buf.getvalue())
                if m.lds is None and lb is not None and lb > self.lds_cap:  # e.g. "no LDS-feasible pair"
                    m.lds, m.lds_cap = lb, self.lds_cap
                return m
            kw = KernelWriterAssembly(self.assembler, self.debug)
            try:
                kw._initKernel(sol, {}, {})
            except (RuntimeError, SystemExit):
                return _rejected(buf.getvalue())
            except Exception as e:  # e.g. an internal assert: this setting yields no kernel
                return Measurement(
                    err=-4, reason=f"TensileLite {type(e).__name__} in _initKernel: {e}"
                )
        return sol, kw

    def accepts(self, perm: Dict[str, Any]) -> bool:
        """Whether TensileLite accepts ``perm`` before code generation (milliseconds; a full measure takes seconds)."""
        return not isinstance(self._initialised(dict(perm), io.StringIO()), Measurement)

    def measure(self, perm: Dict[str, Any]) -> Measurement:
        """Code-generate one kernel. ``perm`` maps fork parameters to values and group names
        (including ``space.mi_group``) to entries, as a Ductile GA sample does."""
        buf = io.StringIO()
        init = self._initialised(dict(perm), buf)
        if isinstance(init, Measurement):
            return init
        sol, kw = init
        counts = dict(
            vgpr_cap=kw.states.regCaps["MaxVgpr"],
            sgpr_cap=kw.states.regCaps["MaxSgpr"],
            lds_cap=sol["MaxLDS"],  # TensileLite's LDS limit for this solution (DeviceLDS unless the config lowers it)
        )
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            over_v = kw.vgprPool.size() > EARLY_EXIT_MARGIN * counts["vgpr_cap"]
            if over_v or kw.sgprPool.size() > EARLY_EXIT_MARGIN * counts["sgpr_cap"]:
                # Register counts are lower bounds: the body would only add to them.
                regs, budget = _occupancy(kw, sol)
                return Measurement(
                    err=1 if over_v else 2,
                    vgpr=kw.vgprPool.size(),
                    agpr=kw.agprPool.size(),
                    sgpr=kw.sgprPool.size(),
                    lds=sol["LdsNumBytes"],
                    occ_regs=regs,
                    occ_cap=budget,
                    reason=f"too many {'vgprs' if over_v else 'sgprs'} after _initKernel (body skipped)",
                    **counts,
                )
            try:
                kw._getKernelSource(sol)
                err = 0
            except RuntimeError:
                err = kw.states.overflowedResources or -3  # the code TensileLite's error message reports
            except SystemExit:
                err = -3
            except Exception as e:  # e.g. an internal assert: this setting yields no kernel
                return Measurement(
                    err=-4, reason=f"TensileLite {type(e).__name__} in _getKernelSource: {e}"
                )
            # checkResources records an overflow and generation finishes the body, so the pools
            # hold the counts that decided the verdict.
            regs, budget = _occupancy(kw, sol)
            return Measurement(
                err=err,
                vgpr=kw.vgprPool.size(),
                agpr=kw.agprPool.size(),
                sgpr=kw.sgprPool.size(),
                lds=sol["LdsNumBytes"],
                occ_regs=regs,
                occ_cap=budget,
                reason=_last_reason(buf.getvalue()),
                **counts,
            )


def _last_reason(text: str) -> str:
    for line in reversed(text.splitlines()):
        for marker in ("reject: ", "overflowed resources."):
            if marker in line:
                return line[
                    line.index(marker) + (len(marker) if marker == "reject: " else 0) :
                ].strip()
    return ""


def _rejected(text: str) -> Measurement:
    reason = _last_reason(text)
    m = re.search(r"Kernel Uses (\d+) > (\d+) bytes of LDS", reason)
    if m:
        return Measurement(err=-1, lds=int(m.group(1)), lds_cap=int(m.group(2)), reason=reason)
    return Measurement(err=-1, reason=reason)


# ---------------------------------------------------------------------------- search
class SettingSearch:
    """The kernels to try for one MI, in order: presets, core factorial, then changes of one or two parameters.

    Each kernel is a full GA sample: every fork parameter, one entry of every group, and one of the MI
    group entries MI Designer can emit for the MI (``mi_entries``).
    """

    def __init__(self, space: TuningSpace, config: Dict[str, Any]):
        self.space = space
        self.gemm_type = config["GemmProblems"][0].gemm_type
        self.elem_bytes = dataSize[self.gemm_type.data_type]
        self.depthu: Optional[int] = None  # set by pinned(): the one DepthU every kernel uses
        self.presets_table = LIST_PRESETS.get(get_profile_class(config["ARCH"], LIST_SEARCH_SPACE), [])

    def pinned(self, depthu: int) -> "SettingSearch":
        """This search with every kernel at ``depthu``: the space's only DepthU, and only the MI
        entries for which it is legal."""
        s = copy.copy(self)
        s.space = replace(self.space, fork={**self.space.fork, "DepthU": [depthu]})
        s.depthu = depthu
        return s

    @staticmethod
    def _min_depthu(entry: Dict[str, Any]) -> int:
        """Smallest DepthU with at least one loop iteration: MFMA K x local split U (TensileLite rejects
        LoopIters = DepthU / LSU / MatrixInstK < 1, SolutionStructs/Solution.py)."""
        return entry["MatrixInstruction"][2] * entry.get("WorkGroup", [0, 0, 1])[2]

    def read_width(self, bf16_elements: int) -> int:
        """Global-read vector width moving the same bytes as ``bf16_elements`` 2-byte elements."""
        return max(1, int(bf16_elements * 2 // self.elem_bytes))

    def mi_entries(self, mi: MI) -> List[Dict[str, Any]]:
        """MI group entries MI Designer can emit for ``mi``, one per local split U (1, 2, 4 up to 4 / waves), LSU 1 first.

        Which LSU MI Designer picks also depends on MI_FILTER, StreamK and the size (MIDesign._find_mi_for_size);
        the list must hold for any of them, so every LSU is tried and the MI is buildable if any entry builds.
        GlobalSplitU (size-dependent, non-StreamK only) is left at its default of 1.
        """
        from geko.config_generator.mi_designer import MFMA, MIDesign

        m = MFMA.from_list(mi)
        p = MIDesign.calculate_mfma_parameters(m, waveFrontSize=self.space.wavefront)
        entries = []
        for lsu in range(1, int(4 / (m.waveM * m.waveN)) + 1):
            if lsu == 3:
                continue
            entry = {"MatrixInstruction": list(mi)}
            if lsu > 1:
                entry["WorkGroup"] = [p.WG0, p.WG1, lsu]
            entries.append(entry)
        if self.depthu is not None:
            entries = [e for e in entries if self.depthu >= self._min_depthu(e)]
        return entries

    def _legal_depthu(self, entry: Dict[str, Any]) -> List[int]:
        """DepthU values the space allows with at least one loop iteration."""
        k = self._min_depthu(entry)
        return sorted(d for d in self.space.fork.get("DepthU", []) if isinstance(d, int) and d >= k)

    def _base(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Every parameter at its lowest-resource allowed value (else its first), every group at its first entry."""
        s = {}
        for name, values in self.space.fork.items():
            low = LOW_RESOURCE.get(name)
            s[name] = low if name in LOW_RESOURCE and low in values else values[0]
        for g, entries in self.space.groups.items():
            s[g] = entries[0]
        s[self.space.mi_group] = entry
        return s

    def _group_with(self, name: str) -> Optional[str]:
        return next(
            (g for g, entries in self.space.groups.items() if any(name in e for e in entries)), None
        )

    def presets(self, mi: MI, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        legal_du = self._legal_depthu(entry)
        for p in self.presets_table:
            s = self._base(entry)
            ok = True
            for name, v in p.items():
                if name == "DepthU":
                    v = legal_du[0] if legal_du else None
                elif name.startswith("GlobalReadVectorWidth") and v > 0:
                    v = self.read_width(v)
                if name in self.space.fork:
                    if v not in self.space.fork[name]:
                        ok = False
                        break
                    s[name] = v
                    continue
                g = self._group_with(name)
                if g is None:
                    continue  # not tuned in this space: its default applies
                match = next(
                    (e for e in self.space.groups[g] if all(e[k] == p[k] for k in e if k in p)),
                    None,
                )
                if match is None:
                    ok = False
                    break
                s[g] = match
            if ok and s not in out:
                out.append(s)
        return out

    def core(self, mi: MI, entry: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        env = self.space.fork
        width = [-1, self.read_width(2)]
        wanted = {
            "DepthU": self._legal_depthu(entry)[:2],
            "PrefetchGlobalRead": [2, 1],
            "1LDSBuffer": [0, 1],
            "ClusterLocalRead": [1, 0],
            "GlobalReadVectorWidthA": width,
            "GlobalReadVectorWidthB": width,
            "TransposeLDS": [-1, 0, 2],
            "DirectToVgprA": [False, True],
            "DirectToVgprB": [False, True],
            "MIArchVgpr": [False],
            "HalfPLR": [0, 1, 2, 3],  # bit mask: halve the prefetch-local-read buffer of A = 1, B = 2, both = 3
        }
        values = {
            name: ([v for v in vals if v in env[name]] or [env[name][0]])
            for name, vals in wanted.items()
            if name in env
        }
        # DirectToVgpr loads an operand straight into the registers its MFMA reads, so that operand's
        # global-read width must match the MFMA's input per thread (M or N x K x B / wavefront lanes), or twice
        # it for small-K FP32 MFMAs; neither factorial width gives it. Tried only with DirectToVgpr on.
        m, lanes = entry["MatrixInstruction"], self.space.wavefront
        dtv_widths = {
            ab: [w for w in (per, 2 * per) if w in env.get(f"GlobalReadVectorWidth{ab}", [])]
            for ab, per in (("A", m[0] * m[2] * m[3] // lanes), ("B", m[1] * m[2] * m[3] // lanes))
        }
        names = list(values) + list(self.space.groups)
        choices = list(values.values()) + list(self.space.groups.values())
        base = self._base(entry)
        for combo in itertools.product(*choices):
            s = dict(base)
            s.update(zip(names, combo))
            yield s
            for ab in ("A", "B"):
                grvw = f"GlobalReadVectorWidth{ab}"
                if s.get(f"DirectToVgpr{ab}") and grvw in values and s[grvw] == values[grvw][0]:
                    for w in dtv_widths[ab]:
                        if w not in values[grvw]:
                            yield dict(s, **{grvw: w})

    def changes(
        self, start: Dict[str, Any], entries: Sequence[Dict[str, Any]]
    ) -> Iterable[Tuple[str, Any]]:
        """Every single change of ``start`` within the space: (fork parameter or group, new value)."""
        dims = (
            list(self.space.fork.items())
            + list(self.space.groups.items())
            + [(self.space.mi_group, entries)]
        )
        for name, values in dims:
            for v in values:
                if v != start.get(name):
                    yield name, v

    def one_at_a_time(
        self, start: Dict[str, Any], entries: Sequence[Dict[str, Any]]
    ) -> Iterable[Dict[str, Any]]:
        for name, v in self.changes(start, entries):
            yield {**start, name: v}


@dataclass
class Verdict:
    mi: MI
    buildable: bool
    stage: str  # preset / core / repair / one-at-a-time / "" (unbuildable)
    checks: int
    witness: Dict[str, Any] = field(default_factory=dict)
    overflow: float = math.inf  # closest kernel's Measurement.overflow() (unbuildable MIs)
    reason: str = ""
    depthu: List[int] = field(default_factory=list)  # DepthUs it builds at (buildable_depthus)
    validations: int = 0  # validation-only checks (repair), besides the ``checks`` code generations
    unmodelled: Dict[int, int] = field(default_factory=dict)  # failures with every resource fitting, per error code
    # Resources over their cap in the kernel that built: TensileLite no longer enforces the limit as modelled
    # (the early exit and the overflow measure would then misjudge kernels).
    over_cap: List[str] = field(default_factory=list)


def _starts(failures: Sequence[Tuple[float, str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Up to MULTI_STARTS failed kernels to start the one-at-a-time stage from: the closest to fitting
    for each over-used resource, then the next closest overall. Refusals with every resource under its
    cap (negative overflow) rank last: they never led to a build. Refused settings are never starts."""
    ranked = sorted((f for f in failures if f[0] != math.inf), key=lambda f: (f[0] < 0, f[0]))
    starts, resources = [], set()
    for ov, res, s in ranked:
        if ov >= 0 and res not in resources:
            resources.add(res)
            starts.append(s)
    for ov, res, s in ranked:
        if len(starts) >= MULTI_STARTS:
            break
        if not any(s is t for t in starts):
            starts.append(s)
    starts.sort(key=lambda s: next((f[0] < 0, f[0]) for f in ranked if f[2] is s))
    return starts[:MULTI_STARTS]


def _key(setting: Dict[str, Any]) -> str:
    return json.dumps(setting, sort_keys=True, default=str)


def _named(name: str, value: Any, reason: str) -> bool:
    """Whether a refusal message names the parameter a change sets (for a group, a key of its new entry)."""
    return any(k in reason for k in (value if isinstance(value, dict) else (name,)))


def repair(
    checker: CodegenChecker,
    search: SettingSearch,
    refused: Dict[str, Any],
    reason: str,
    entries: Sequence[Dict[str, Any]],
    accepted: Dict[str, bool],
) -> List[Dict[str, Any]]:
    """Up to REPAIR_KEEP accepted settings nearest ``refused`` (validation only): single changes, those to a
    parameter ``reason`` names first, else pairs within REPAIR_PAIRS validations. ``accepted`` caches verdicts."""

    def ok(s: Dict[str, Any]) -> bool:
        k = _key(s)
        if k not in accepted:
            accepted[k] = checker.accepts(s)
        return accepted[k]

    singles = sorted(search.changes(refused, entries), key=lambda c: not _named(*c, reason))
    found: List[Dict[str, Any]] = []
    for name, v in singles:
        s = {**refused, name: v}
        if ok(s):
            found.append(s)
            if len(found) == REPAIR_KEEP:
                return found
    if found:
        return found
    budget = len(accepted) + REPAIR_PAIRS
    for (n1, v1), (n2, v2) in itertools.combinations(singles, 2):
        if len(accepted) >= budget:
            break
        if n1 == n2:
            continue
        s = {**refused, n1: v1, n2: v2}
        if ok(s):
            found.append(s)
            if len(found) == REPAIR_KEEP:
                break
    return found


def classify(checker: CodegenChecker, search: SettingSearch, mi: MI) -> Verdict:
    """Try kernels for one MI until one builds."""
    entries = search.mi_entries(mi)
    if not entries:
        return Verdict(mi, False, "", 0, reason="MI Designer emits no entry for this MI")
    checks, best, best_m, seen = 0, None, None, set()
    failures: List[Tuple[float, str, Dict[str, Any]]] = []  # (overflow, resource, setting)
    refused: List[Tuple[Dict[str, Any], str]] = []  # (setting, reason): failures without resource counts
    accepted: Dict[str, bool] = {}  # the repair stage's validation verdicts
    unmodelled: Dict[int, int] = {}

    def attempt(perms: Iterable[Dict[str, Any]], stage: str) -> Optional[Verdict]:
        nonlocal checks, best, best_m
        for s in perms:
            key = _key(s)
            if key in seen:
                continue
            seen.add(key)
            m = checker.measure(s)
            checks += 1
            if m.err == 0:
                v = Verdict(mi, True, stage, checks, s)
                if m.vgpr is not None:
                    v.over_cap = sorted(r for r, u in m._usage().items() if u > 1)
                return v
            failures.append((m.overflow(), m.resource(), s))
            if m.overflow() == math.inf:
                refused.append((s, m.reason))
            if best_m is None or m.overflow() < best_m.overflow():
                best, best_m = s, m
            if m.vgpr is not None and m.overflow() <= 0:
                unmodelled[m.err] = unmodelled.get(m.err, 0) + 1
        return None

    def stages() -> Optional[Verdict]:
        for stage, perms in (
            ("preset", lambda e: search.presets(mi, e)),
            ("core", lambda e: search.core(mi, e)),
        ):
            for entry in entries:
                v = attempt(perms(entry), stage)
                if v:
                    return v
        # Refused settings give stage 3 no distance: repair one per (MI entry, DepthU, rule), numbers masked
        # so one rule broken with different values counts once.
        repaired, before_repair = set(), len(failures)
        for s, reason in list(refused):
            group = (_key(s[search.space.mi_group]), s.get("DepthU"), re.sub(r"\d+", "#", reason))
            if group in repaired:
                continue
            if len(repaired) == REPAIRS:
                break
            repaired.add(group)
            v = attempt(repair(checker, search, s, reason, entries, accepted), "repair")
            if v:
                return v
        # The failures stages 1-2 measured come first, so repair-generated kernels never crowd out a start
        # the search would have used without them; repair-generated ones fill the remaining slots.
        starts = _starts(failures[:before_repair])
        starts += [s for s in _starts(failures[before_repair:]) if not any(s is t for t in starts)]
        for start in starts[:MULTI_STARTS]:
            for _ in range(OFAT_PASSES):
                n = len(failures)
                v = attempt(search.one_at_a_time(start, entries), "one-at-a-time")
                if v:
                    return v
                closer = min(failures[n:], key=lambda f: f[0], default=None)
                start_ov = next(f[0] for f in failures if f[2] is start)
                if closer is None or closer[0] >= start_ov:
                    break
                start = closer[2]
        return None

    v = stages() or Verdict(
        mi,
        False,
        "",
        checks,
        best or {},
        best_m.overflow() if best_m else math.inf,
        best_m.reason if best_m else "",
    )
    v.validations, v.unmodelled = len(accepted), unmodelled
    return v


def searched_depthu(space: TuningSpace) -> List[int]:
    """The DepthUs buildable_depthus tries: every integer DepthU of the space, ascending."""
    return sorted(d for d in space.fork.get("DepthU", []) if isinstance(d, int))


def buildable_depthus(
    checker: CodegenChecker, search: SettingSearch, mi: MI, witness: Dict[str, Any]
) -> List[int]:
    """Every DepthU of the space at which ``mi`` builds, for an MI whose ``witness`` setting built.

    Searches from the witness's DepthU upward, then downward, stopping in each direction at the first
    DepthU where classify finds no kernel. That assumes an MI's buildable DepthUs are contiguous, which
    held for all 900 gfx950 MIs checked at every DepthU (300 each of BBS NN, F8BS NT, SSS NN): a larger
    DepthU stages more of A and B in LDS and registers. Each stop is a full search that fails, which
    roughly doubles a list's generation time. On gfx950, 0 of 481 sampled stops built under a much
    deeper search (3,000 attempts each).
    """
    dus = searched_depthu(search.space)
    w = witness.get("DepthU")
    if w not in dus:
        return []
    i = dus.index(w)
    found = [w]
    for d in dus[i + 1 :]:
        if not classify(checker, search.pinned(d), mi).buildable:
            break
        found.append(d)
    for d in reversed(dus[:i]):
        if not classify(checker, search.pinned(d), mi).buildable:
            break
        found.insert(0, d)
    return found


# ---------------------------------------------------------------------------- driver
_WORKER: Tuple[Optional[CodegenChecker], Optional[SettingSearch]] = (None, None)


def _classify_in_worker(mi: MI) -> Verdict:
    checker, search = _WORKER
    v = classify(checker, search, mi)
    if v.buildable:
        v.depthu = buildable_depthus(checker, search, mi, v.witness)
    return v


class SubtileOnly(Exception):
    """The GEMM type is tuned only as subtile kernels, which MI Designer never filters with a list."""


def classify_candidates(
    config: Dict[str, Any],
    hipblaslt_path: Path,
    workers: int,
    progress: Callable[[int, int, int], None] = lambda done, total, buildable: None,
) -> Tuple[List[Verdict], TuningSpace, str]:
    """Verdict for every candidate MI of one GEMM type, in enumeration order; every buildable MI's
    verdict also lists the DepthUs it builds at (buildable_depthus).

    Returns (verdicts, tuning space, GEMM name).
    """
    global _WORKER
    from geko.config_generator.config_generator import mi_design_mx_options

    with tempfile.TemporaryDirectory(prefix="geko_buildable_") as tmp:
        config_path, cfg = write_probe_config(config, hipblaslt_path, Path(tmp))
        gemm = cfg["GemmProblems"][0]
        if mi_design_mx_options({**cfg, "GemmProblem": gemm})[1]:
            raise SubtileOnly(f"{gemm.name} on {cfg['ARCH']} is tuned as subtile kernels (gfx950 MX), "
                              "which do not read buildable-MI lists")
        checker = CodegenChecker(config_path, cfg["ARCH"])
    search = SettingSearch(checker.space, cfg)
    mis = enumerate_candidate_mis(cfg["ARCH"], gemm.gemm_type.data_type, LIST_SEARCH_SPACE)

    _WORKER = (checker, search)  # inherited by forked workers
    verdicts: Dict[MI, Verdict] = {}
    # A fresh worker per MI: code generation for a large kernel leaves several GB in the process, and long-lived
    # workers accumulate it (gfx1250 FP4 at 352 workers exhausted 3 TB). Verdicts do not depend on the process.
    with mp.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        for v in pool.imap_unordered(_classify_in_worker, mis, chunksize=1):
            verdicts[v.mi] = v
            progress(len(verdicts), len(mis), sum(x.buildable for x in verdicts.values()))
    _WORKER = (None, None)
    return [verdicts[mi] for mi in mis], checker.space, gemm.name


def write_buildable_list(
    path: Path,
    verdicts: Sequence[Verdict],
    *,
    arch: str,
    gemm_name: str,
    source: str,
    depthu: Sequence[int],
) -> None:
    """Write the list MI Designer reads: the DepthUs searched, then the buildable MIs in enumeration order with
    the DepthUs each builds at, then the unbuildable ones as comments."""
    from geko.config_generator.mi_designer import MFMA, MIDesign

    def mt(mi: MI) -> str:
        p = MIDesign.calculate_mfma_parameters(MFMA.from_list(mi))
        return f"MT{p.MT0}x{p.MT1}"

    buildable = [v for v in verdicts if v.buildable]
    lines = [
        f"# Buildable MatrixInstructions for {gemm_name} on {library_logic_architecture(arch)}:\n",
        "# MIs with at least one kernel TensileLite builds within geko's generic (GA) ranges.\n",
        "# MI Designer proposes only these, each only at the DepthUs it builds at (DepthUs the search did\n",
        "# not try stay allowed).\n",
        f"# {len(buildable)} of {len(verdicts)} candidates buildable; the rest are at the end.\n",
        "# From scripts/generate_buildable_mis.py; rerun when TensileLite or geko changes.\n",
        "\n",
        f"Architecture: {library_logic_architecture(arch)}\n",
        f"GemmType: {gemm_name}\n",
        f"Source: {source}\n",
        f"DepthU: [{', '.join(map(str, depthu))}]  # searched\n",
        "MatrixInstructions:\n" if buildable else "MatrixInstructions: []\n",
    ]

    def entry(v: Verdict) -> str:
        du = f", DepthU: [{', '.join(map(str, v.depthu))}]" if v.depthu else ""
        return f"  - {{MatrixInstruction: [{', '.join(map(str, v.mi))}]{du}}}  # {mt(v.mi)}\n"

    lines += [entry(v) for v in buildable]
    lines.append(
        "\n# Not buildable (closest kernel: overflow of its most over-used resource, and why):\n"
    )
    for v in verdicts:
        if not v.buildable:
            reason = v.reason.replace("\n", " ")[:100]
            over = "no resource counts" if math.isinf(v.overflow) else f"{v.overflow:+.1%}"
            lines.append(f"#   [{', '.join(map(str, v.mi))}]  {mt(v.mi)}  {over}  {reason}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def write_report(path: Path, verdicts: Sequence[Verdict]) -> None:
    """Every MI's verdict, the stage that decided it, code generations and validation-only checks used,
    the witness setting (the setting that built, or for unbuildable MIs the one closest to fitting),
    when recorded the DepthUs it builds at, and the failures no modelled resource explains (per error code)."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["mi", "buildable", "stage", "checks", "overflow", "reason", "witness", "depthu", "validations", "unmodelled",
             "over_cap"]
        )
        for v in verdicts:
            w.writerow(
                [
                    list(v.mi),
                    v.buildable,
                    v.stage,
                    v.checks,
                    "" if v.buildable else v.overflow,
                    v.reason,
                    json.dumps(v.witness, sort_keys=True, default=str),
                    ",".join(map(str, v.depthu)),
                    v.validations,
                    json.dumps(v.unmodelled, sort_keys=True),
                    ",".join(v.over_cap),
                ]
            )
