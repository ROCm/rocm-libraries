# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""TensileLite set up as ``Tensile --cpu-only`` for one GEMM type: the probe tuning config, the search space it
defines, and code generation of one kernel."""

from __future__ import annotations

import contextlib
import copy
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geko.config_generator.buildability.constants import (
    EARLY_EXIT_MARGIN,
    LIST_SEARCH_SPACE,
    LIST_STREAMK,
    PROBE_SIZE,
)
from geko.config_generator.buildability.measurement import Measurement, _last_reason, _occupancy, _rejected
from geko.config_generator.constants import library_logic_architecture


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
