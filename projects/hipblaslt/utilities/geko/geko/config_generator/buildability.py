# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which candidate MatrixInstructions can TensileLite build? Code generation only (CPU, no GPU).

An MI is *buildable* for a GEMM type when at least one kernel for it, with every other
parameter inside the generic (GA) search space's fork ranges, passes TensileLite code
generation: Solution validation, then KernelWriterAssembly._initKernel and _getKernelSource,
which is where a tuning run's "Generating kernel source resulted in error N" failures
(VGPR / AGPR / SGPR / LDS / occupancy overflow) are raised. Nothing is assembled or run.
One list serves every search space but subtile: the heuristic space (exhaustive Tensile
search, rarely used) is narrower than generic, so it can build no MI generic cannot.
One list also serves both StreamK settings: lists are generated with StreamK off (LIST_STREAMK).

The fork ranges are geko's own: the generic tuning config geko's config generator writes for
PROBE_SIZE (MI Designer on every candidate), read back with TensileLite's own loader. So every
value used to prove an MI buildable is one a GA tuning run for that GEMM type can sample.

Search, per MI and per MI group entry MI Designer can emit for it (one per local split U), stopping
at the first kernel that builds. DepthU is "smallest legal" relative to the MFMA (>= K x LSU) and
global-read widths move the same bytes whatever the element size:
  1. PRESETS: three settings found on gfx950 BBS TN; on gfx950 they decide 1171 of the 1171
     buildable BBS TN MIs and 822 of the 838 buildable F8BS TN MIs.
  2. CORE factorial: the parameters that decide register and LDS use (DepthU, prefetch,
     LDS buffering, local-read clustering, global-read widths, LDS transpose, MIArchVgpr) and
     every entry of every non-MI group (e.g. DirectToLds / UseSgprForGRO pairs), at their
     lowest-resource values the envelope allows. TransposeLDS 0 (vs auto) trades VGPRs for LDS;
     with PrefetchGlobalRead 2 it is what builds large gfx950 FP32 NN tiles. TransposeLDS 2 is
     what builds most LDS-limited gfx950 BF16 / FP8 NN, NT and TT tiles.
  3. One parameter at a time: every value of every parameter and group in the envelope, and
     every MI entry, for up to OFAT_PASSES passes from each of up to MULTI_STARTS of the
     kernels closest to fitting (the closest per over-used resource first). A single start can
     sit where no one change helps while a slightly worse kernel is two changes from building.
An MI none of these build is reported unbuildable, with how far the closest kernel overflows.

Every buildable MI is then also searched at each DepthU of the space, climbing from the DepthU its
witness used (buildable_depthus), and its list line records the DepthUs at which it builds.
"""

from __future__ import annotations

import contextlib
import copy
import io
import itertools
import json
import math
import multiprocessing as mp
import re
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from geko.config_generator.constants import dataSize, library_logic_architecture
from geko.config_generator.mi_candidates import MI, enumerate_candidate_mis

# Any ordinary size: geko's generic profiles do not read the problem size, so every such size yields
# the same fork ranges (checked on gfx950 / gfx942, 10 data types, 4 layouts, 40 sizes). Avoid
# M = 1 or N = 1, for which geko's gfx942 BBS / HHS generic config carries no DepthU.
PROBE_SIZE = [4096, 4096, 1, 4096]

# The minimal settings of the gfx950 BBS TN study (see the module docstring for how many MIs they
# decide). DepthU "K" is the smallest legal DepthU (>= MFMA K x LSU); a positive global-read width is
# in 2-byte elements and is scaled to the data type (2 BF16 elements = 4 FP8 elements).
PRESETS = [
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

# Base value of parameters outside the core factorial, where the envelope allows it: usually the
# lowest-resource one (gfx950 BF16 study: the alternatives cost registers). Not always: for tiles at the
# VGPR cap, SourceSwap=True (gfx950 FP8 NN/NT) and TailloopInNll=True (FP32, StreamK off) save the last
# few VGPRs. The one-at-a-time stage finds those; adding them to the factorial would quadruple it for no
# further MI (checked on every VGPR-limited gfx950 rejection within 20% of fitting).
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
}

OFAT_PASSES = 2
MULTI_STARTS = 4

# Stop after _initKernel when a register pool is already this far over its cap. The pools only grow
# during body generation (rocisa RegisterPool) and checkResources fails a kernel whose pool exceeds the
# cap, so such a kernel cannot build; skipping the body saves ~0.5 s. Kernels within the margin are
# generated in full so the one-at-a-time stage still ranks near misses by their exact overflow.
EARLY_EXIT_MARGIN = 1.2


# The search space whose fork ranges define "buildable": the GA's.
LIST_SEARCH_SPACE = "generic"

# Lists are generated with StreamK off. StreamK changes the kernel kind (persistent StreamK kernels
# vs one workgroup per tile), and the persistent kernels need slightly more VGPRs: on gfx950 the
# StreamK-off lists hold every MI the StreamK-on ones do, plus 95 VGPR-edge tiles (FP8 TN, SSS). So
# a StreamK-off list excludes no MI a run with either setting can build; StreamK-on runs carry the
# few edge tiles that fail for them.
LIST_STREAMK = False


# ---------------------------------------------------------------------------- the search space
@dataclass
class TuningSpace:
    """Fork-parameter ranges of a GEMM type's generic tuning config, as TensileLite sees them."""

    config_path: Path
    fork: Dict[str, List[Any]]  # fork parameter -> its values, in config order
    groups: Dict[str, List[Dict[str, Any]]]  # non-MI group name -> its entries
    mi_group: str  # group carrying MatrixInstruction


def write_probe_config(
    config: Dict[str, Any], hipblaslt_path: Path, workdir: Path
) -> Tuple[Path, Dict[str, Any]]:
    """geko's generic (GA) tuning config for PROBE_SIZE, MI Designer on every candidate MI.

    ``config`` is an input config dict (TRANSA, TRANSB, DataType, DestDataType,
    ComputeDataType, ARCH and other keys); ARCH defaults apply.
    Returns (config path, the prepared config, with GemmProblems).
    """
    from geko.config_generator.config_generator import run
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
    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    get_gemm_problem(cfg)
    workdir.mkdir(parents=True, exist_ok=True)
    run(cfg, hipblaslt_path, workdir, write_shell_scripts=False)
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
    err: int  # 0 builds; -1 rejected before code generation; N KernelWriter error; -4 other exception
    vgpr: Optional[int] = None
    agpr: Optional[int] = None
    sgpr: Optional[int] = None
    lds: Optional[int] = None
    vgpr_cap: Optional[int] = None
    sgpr_cap: Optional[int] = None
    lds_cap: Optional[int] = None
    reason: str = ""

    def _usage(self) -> Dict[str, float]:
        """Each resource's use as a fraction of its cap (VGPRs, SGPRs, LDS, VGPR + AGPR file of 512 per lane)."""
        return {
            "vgpr": self.vgpr / self.vgpr_cap,
            "sgpr": self.sgpr / self.sgpr_cap,
            "lds": self.lds / self.lds_cap,
            "vgpr+agpr": (math.ceil(self.vgpr / 8) * 8 + self.agpr) / 512,
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
        return TuningSpace(config_path, fork, groups, mi_group)

    def measure(self, perm: Dict[str, Any]) -> Measurement:
        """Code-generate one kernel. ``perm`` maps fork parameters to values and group names
        (including ``space.mi_group``) to entries, as a Ductile GA sample does."""
        from tensilelite.backends.ductile_backend import _generate_single_solution_with_groups
        from tensilelite.KernelWriterAssembly import KernelWriterAssembly

        perm = dict(perm)
        buf = io.StringIO()
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
                return _rejected(buf.getvalue())
            kw = KernelWriterAssembly(self.assembler, self.debug)
            try:
                kw._initKernel(sol, {}, {})
            except (RuntimeError, SystemExit):
                return _rejected(buf.getvalue())
            except Exception as e:  # e.g. an internal assert: this setting yields no kernel
                return Measurement(
                    err=-4, reason=f"TensileLite {type(e).__name__} in _initKernel: {e}"
                )
            vcap, scap = kw.states.regCaps["MaxVgpr"], kw.states.regCaps["MaxSgpr"]
            over_v = kw.vgprPool.size() > EARLY_EXIT_MARGIN * vcap
            if over_v or kw.sgprPool.size() > EARLY_EXIT_MARGIN * scap:
                # Register counts are lower bounds: the body would only add to them.
                return Measurement(
                    err=1 if over_v else 2,
                    vgpr=kw.vgprPool.size(),
                    agpr=kw.agprPool.size(),
                    sgpr=kw.sgprPool.size(),
                    lds=sol["LdsNumBytes"],
                    vgpr_cap=vcap,
                    sgpr_cap=scap,
                    lds_cap=kw.states.archCaps["DeviceLDS"],
                    reason=f"too many {'vgprs' if over_v else 'sgprs'} after _initKernel (body skipped)",
                )
            try:
                kw._getKernelSource(sol)
                err = 0
            except RuntimeError as e:
                tail = str(e).rsplit(" ", 1)[-1]
                err = int(tail) if tail.lstrip("-").isdigit() else -3
            except SystemExit:
                err = -3
            except Exception as e:  # e.g. an internal assert: this setting yields no kernel
                return Measurement(
                    err=-4, reason=f"TensileLite {type(e).__name__} in _getKernelSource: {e}"
                )
            # checkResources records an overflow and generation finishes the body, so the pools
            # hold the counts that decided the verdict.
            return Measurement(
                err=err,
                vgpr=kw.vgprPool.size(),
                agpr=kw.agprPool.size(),
                sgpr=kw.sgprPool.size(),
                lds=sol["LdsNumBytes"],
                vgpr_cap=kw.states.regCaps["MaxVgpr"],
                sgpr_cap=kw.states.regCaps["MaxSgpr"],
                lds_cap=kw.states.archCaps["DeviceLDS"],
                reason=_last_reason(buf.getvalue()),
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
    """The kernels to try for one MI, in order: presets, core factorial, then one parameter at a time.

    Each kernel is a full GA sample: every fork parameter, one entry of every group, and one of the MI
    group entries MI Designer can emit for the MI (``mi_entries``).
    """

    def __init__(self, space: TuningSpace, config: Dict[str, Any]):
        self.space = space
        self.gemm_type = config["GemmProblems"][0].gemm_type
        self.elem_bytes = dataSize[self.gemm_type.data_type]
        self.depthu: Optional[int] = None  # set by pinned(): the one DepthU every kernel uses

    def pinned(self, depthu: int) -> "SettingSearch":
        """This search with every kernel at ``depthu``: the space's only DepthU, and only the MI
        entries for which it is legal."""
        s = copy.copy(self)
        s.space = replace(self.space, fork={**self.space.fork, "DepthU": [depthu]})
        s.depthu = depthu
        return s

    @staticmethod
    def _min_depthu(entry: Dict[str, Any]) -> int:
        """Smallest DepthU with at least one loop iteration: MFMA K x local split U."""
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
        p = MIDesign.calculate_mfma_parameters(m)
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
        for p in PRESETS:
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
            "MIArchVgpr": [False],
        }
        values = {
            name: ([v for v in vals if v in env[name]] or [env[name][0]])
            for name, vals in wanted.items()
            if name in env
        }
        names = list(values) + list(self.space.groups)
        choices = list(values.values()) + list(self.space.groups.values())
        base = self._base(entry)
        for combo in itertools.product(*choices):
            s = dict(base)
            s.update(zip(names, combo))
            yield s

    def one_at_a_time(
        self, start: Dict[str, Any], entries: Sequence[Dict[str, Any]]
    ) -> Iterable[Dict[str, Any]]:
        dims = (
            list(self.space.fork.items())
            + list(self.space.groups.items())
            + [(self.space.mi_group, entries)]
        )
        for name, values in dims:
            for v in values:
                if v != start.get(name):
                    s = dict(start)
                    s[name] = v
                    yield s


@dataclass
class Verdict:
    mi: MI
    buildable: bool
    stage: str  # preset / core / one-at-a-time / "" (unbuildable)
    checks: int
    witness: Dict[str, Any] = field(default_factory=dict)
    overflow: float = math.inf  # closest kernel's Measurement.overflow() (unbuildable MIs)
    reason: str = ""
    depthu: List[int] = field(default_factory=list)  # DepthUs it builds at (buildable_depthus)


def _starts(failures: Sequence[Tuple[float, str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Up to MULTI_STARTS failed kernels to start the one-at-a-time stage from: the closest to fitting
    for each over-used resource, then the next closest overall."""
    ranked = sorted((f for f in failures if f[0] != math.inf), key=lambda f: f[0])
    starts, resources = [], set()
    for ov, res, s in ranked:
        if res not in resources:
            resources.add(res)
            starts.append(s)
    for ov, res, s in ranked:
        if len(starts) >= MULTI_STARTS:
            break
        if not any(s is t for t in starts):
            starts.append(s)
    starts.sort(key=lambda s: next(f[0] for f in ranked if f[2] is s))
    return starts[:MULTI_STARTS]


def classify(checker: CodegenChecker, search: SettingSearch, mi: MI) -> Verdict:
    """Try kernels for one MI until one builds."""
    entries = search.mi_entries(mi)
    if not entries:
        return Verdict(mi, False, "", 0, reason="MI Designer emits no entry for this MI")
    checks, best, best_m, seen = 0, None, None, set()
    failures: List[Tuple[float, str, Dict[str, Any]]] = []  # (overflow, resource, setting)

    def attempt(perms: Iterable[Dict[str, Any]], stage: str) -> Optional[Verdict]:
        nonlocal checks, best, best_m
        for s in perms:
            key = json.dumps(s, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            m = checker.measure(s)
            checks += 1
            if m.err == 0:
                return Verdict(mi, True, stage, checks, s)
            failures.append((m.overflow(), m.resource(), s))
            if best_m is None or m.overflow() < best_m.overflow():
                best, best_m = s, m
        return None

    for stage, perms in (
        ("preset", lambda e: search.presets(mi, e)),
        ("core", lambda e: search.core(mi, e)),
    ):
        for entry in entries:
            v = attempt(perms(entry), stage)
            if v:
                return v
    for start in _starts(failures):
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
    return Verdict(
        mi,
        False,
        "",
        checks,
        best or {},
        best_m.overflow() if best_m else math.inf,
        best_m.reason if best_m else "",
    )


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
    dus = sorted(d for d in search.space.fork.get("DepthU", []) if isinstance(d, int))
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
    with tempfile.TemporaryDirectory(prefix="geko_buildable_") as tmp:
        config_path, cfg = write_probe_config(config, hipblaslt_path, Path(tmp))
        checker = CodegenChecker(config_path, cfg["ARCH"])
    gemm = cfg["GemmProblems"][0]
    search = SettingSearch(checker.space, cfg)
    mis = enumerate_candidate_mis(cfg["ARCH"], gemm.gemm_type.data_type, LIST_SEARCH_SPACE)

    _WORKER = (checker, search)  # inherited by forked workers
    verdicts: Dict[MI, Verdict] = {}
    with mp.get_context("fork").Pool(workers) as pool:
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
) -> None:
    """Write the list MI Designer reads: buildable MIs in enumeration order, unbuildable ones as comments."""
    from geko.config_generator.mi_designer import MFMA, MIDesign

    def mt(mi: MI) -> str:
        p = MIDesign.calculate_mfma_parameters(MFMA.from_list(mi))
        return f"MT{p.MT0}x{p.MT1}"

    buildable = [v for v in verdicts if v.buildable]
    lines = [
        f"# Buildable MatrixInstructions for {gemm_name} on {library_logic_architecture(arch)}.\n",
        "# Generated by scripts/generate_buildable_mis.py; do not edit by hand. Regenerate when TensileLite\n",
        "# code generation or geko's fork-parameter ranges change.\n",
        "#\n",
        "# An MI is listed when TensileLite code generation builds at least one kernel for it with its other\n",
        "# parameters inside geko's widest tuning ranges (the generic search space) and StreamK off, which\n",
        "# together build the most MIs. One list therefore serves every search space but subtile, and tuning\n",
        "# runs with StreamK on or off. MI Designer reads this list instead of enumerating MIs.\n",
    ]
    if any(v.depthu for v in buildable):
        lines.append(
            "# Each MI's comment gives its macro tile and every DepthU in those ranges at which it builds.\n"
        )
    lines += [
        f"# {len(buildable)} of {len(verdicts)} candidate MIs are buildable; the others are listed at the end.\n",
        "\n",
        f"Architecture: {library_logic_architecture(arch)}\n",
        f"GemmType: {gemm_name}\n",
        f"Source: {source}\n",
        "MatrixInstructions:\n",
    ]

    def note(v: Verdict) -> str:
        return mt(v.mi) + (f"  DepthU {','.join(map(str, v.depthu))}" if v.depthu else "")

    lines += [f"  - [{', '.join(map(str, v.mi))}]  # {note(v)}\n" for v in buildable]
    lines.append(
        "\n# Not buildable within those ranges (closest kernel: how far its most over-used resource\n"
        "# exceeds the cap, and TensileLite's reason):\n"
    )
    for v in verdicts:
        if not v.buildable:
            reason = v.reason.replace("\n", " ")[:100]
            over = "no resource counts" if math.isinf(v.overflow) else f"{v.overflow:+.1%}"
            lines.append(f"#   [{', '.join(map(str, v.mi))}]  {mt(v.mi)}  {over}  {reason}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def write_report(path: Path, verdicts: Sequence[Verdict]) -> None:
    """Every MI's verdict, the stage that decided it, code generations used, the witness setting
    (the setting that built, or for unbuildable MIs the one closest to fitting) and, when recorded,
    the DepthUs it builds at."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["mi", "buildable", "stage", "checks", "overflow", "reason", "witness", "depthu"]
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
                ]
            )
