# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The kernels to try for one MI: presets, the core factorial, single changes (stage 3), and repair of settings
TensileLite refused."""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from geko.config_generator.buildability.codegen import CodegenChecker, TuningSpace
from geko.config_generator.buildability.constants import (
    LIST_PRESETS,
    LIST_SEARCH_SPACE,
    LOW_RESOURCE,
    REPAIR_KEEP,
    REPAIR_PAIRS,
)
from geko.config_generator.constants import dataSize
from geko.config_generator.fork_params import get_profile_class
from geko.config_generator.mi_candidates import MI


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
