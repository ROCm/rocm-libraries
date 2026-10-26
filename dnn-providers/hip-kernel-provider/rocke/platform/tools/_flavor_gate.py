# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Did an instance decline to lower because the LLVM it was asked for is too old?
#
# Both corpus-wide gates have to ask: the corpus is flavor-wide, but neither a
# host nor a single sweep is. The refusal is the emitter working correctly, so a
# gate that counts it as a validity failure goes red on a clean tree. Shared
# rather than duplicated so `check_ir_validity` and `check_arch_domain` cannot
# drift on what the question is, even though they act on the answer differently.
#
# Not a CLI entry point -- hence the leading underscore, as with _hostcaps.py.

from __future__ import annotations

import re


def requires_newer_flavor(exc: BaseException) -> bool:
    """Whether ``exc`` is an instance refusing a too-old LLVM flavor.

    Matches the emitter's wording -- ``requires llvm23 (ROCm 7.13+), got
    llvm22`` from the instruction emitters, ``requires LLVM flavor llvm23, got
    llvm22`` from the llvm23-gated target ops (TDM tensor moves, clusters,
    ...) -- not the exception type: the refusal is a bare
    ``NotImplementedError``/``ValueError``, which the lowerer also raises for
    genuine defects. Anything not saying "requires [LLVM flavor] llvmNN" stays
    fatal.
    """
    return bool(re.search(r"requires (?:LLVM flavor )?llvm\d+", str(exc)))
