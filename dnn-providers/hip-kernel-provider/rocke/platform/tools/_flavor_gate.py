# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Did an instance decline to lower because the LLVM it was asked for is too old?
#
# Both corpus-wide gates have to ask. The corpus is flavor-wide, but neither a
# host nor a single sweep is: `check_ir_validity` lowers at the host's flavor,
# `check_arch_domain` lowers at every committed flavor in turn, and an instance
# gated on a newer vintage refuses in both. That refusal is the emitter working
# correctly -- it is the only reason this is not a validity failure -- so a gate
# that counts it as one goes red on a clean tree for a case it was never going
# to be able to measure.
#
# Shared rather than duplicated for the same reason as _llvm_identity.py: the
# two callers do different things with the answer (one reports UNVALIDATED and
# keeps its exit code, the other reports the case as skipped), but they must
# agree on what the question *is*. Two copies of this regex would drift the
# moment the gating message was reworded.
#
# Not a CLI entry point -- hence the leading underscore, as with _hostcaps.py.

from __future__ import annotations

import re


def requires_newer_flavor(exc: BaseException) -> bool:
    """Whether ``exc`` is an instance refusing a too-old LLVM flavor.

    Matches the emitter's own wording -- ``requires llvm23 (ROCm 7.13+), got
    llvm22`` -- rather than the exception type, deliberately. The refusal is
    raised as a bare ``NotImplementedError``/``ValueError``, and those come out
    of the lowerer for genuine defects too; keying on the type would swallow
    every one of them. Anything that does not say "requires llvmNN" stays
    fatal, which is the direction that cannot hide a real failure.
    """
    return bool(re.search(r"requires llvm\d+", str(exc)))
