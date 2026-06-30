# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Import-by-path loader for a curated ``level_NN_<name>.py`` snapshot module.

Each curated level snapshot is a SELF-CONTAINED copy of the fp8 mega-kernel
emitter as it existed at that optimization step (its own ``FusedMegaKernelSpecFp8``
+ ``build_moe_fused_mega_gemm_fp8``). To reproduce a structural level we load that
snapshot module *by file path* under a throwaway module name -- so the production
``rocke.instances.common.moe_fused_mega_fp8`` on disk is never touched or
shadowed -- and build from it.

Generalized from the campaign's ``build_predtla_fp8.py`` helper (which hard-coded
a single ``/tmp`` path); this version takes any level snapshot path.

Build-only by default; the caller supplies the GPU launch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[5]  # .../composablekernel/python
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_level_module(path: str | Path, *, mod_name: str | None = None) -> ModuleType:
    """Load a curated ``level_NN_*.py`` snapshot as a throwaway module.

    The loaded module exposes (at minimum) ``FusedMegaKernelSpecFp8`` and
    ``build_moe_fused_mega_gemm_fp8`` -- the same surface as the production file
    at that snapshot's point in the campaign. The production file is NOT
    imported or shadowed (the throwaway name lives under
    ``rocke.instances.common._level_<stem>``).
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"level snapshot not found: {path}")
    if mod_name is None:
        mod_name = f"rocke.instances.common._level_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load level snapshot at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_level_kernel(
    path: str | Path,
    *,
    name: str = "rocke_fused_moe_mega_fp8",
    arch: str = "gfx950",
    **spec_kwargs,
):
    """Load + build a level snapshot -> ``(module, spec, KernelDef)``.

    ``spec_kwargs`` are forwarded to that snapshot's ``FusedMegaKernelSpecFp8``
    (e.g. ``tile_m=32`` for the level-1 toggle row).
    """
    mod = load_level_module(path)
    spec = mod.FusedMegaKernelSpecFp8(name=name, **spec_kwargs)
    kd = mod.build_moe_fused_mega_gemm_fp8(spec, arch=arch)
    return mod, spec, kd
