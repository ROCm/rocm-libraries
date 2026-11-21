# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What code generation measured for one kernel: resource use against the caps, or why TensileLite refused it."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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
