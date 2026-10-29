# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The two entry points. See ``hipdnn_sdpa/__init__.py`` for the contract."""

from __future__ import annotations

from . import _problem, _runtime


def sdpa_fwd(
    q,
    k,
    v,
    *,
    o,
    stats,
    scale,
    bias=None,
    mask=None,
    dropout=None,
    varlen=None,
    engine=None,
) -> str:
    """Forward SDPA on hipDNN; writes ``o`` and ``stats`` in place.

    Returns the engine name used. Raises :class:`EngineDeclined` when hipDNN cannot
    serve the call.
    """
    prob = _problem.forward_problem(
        q, k, v, o, stats, scale, bias, mask, dropout, varlen
    )
    return _runtime.run(prob, engine)


def sdpa_bwd(
    q,
    k,
    v,
    o,
    do,
    stats,
    *,
    dq,
    dk,
    dv,
    scale,
    dbias=None,
    bias=None,
    mask=None,
    dropout=None,
    varlen=None,
    engine=None,
) -> str:
    """Backward SDPA on hipDNN; writes ``dq``, ``dk``, ``dv`` (and ``dbias``) in place.

    Returns the engine name used. Raises :class:`EngineDeclined` when hipDNN cannot
    serve the call.
    """
    prob = _problem.backward_problem(
        q, k, v, o, do, stats, dq, dk, dv, scale, dbias, bias, mask, dropout, varlen
    )
    return _runtime.run(prob, engine)
