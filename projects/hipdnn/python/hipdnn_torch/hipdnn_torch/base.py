# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.base -- the generic op-override machinery.

:class:`OpOverride` monkeypatches one ``torch.nn.functional`` entry point to route
onto the hipDNN engine, with everything the three concrete overrides
(:mod:`~hipdnn_torch.linear`, :mod:`~hipdnn_torch.rmsnorm`, :mod:`~hipdnn_torch.sdpa`)
share:

  * ``install()`` / ``uninstall()`` -- patch/restore the functional symbol
    (patching ``torch.nn.functional.<op>`` covers ``F.<op>`` too, since ``F`` *is*
    that module; ``nn.Module`` forwards resolve the functional by attribute lookup
    at call time, so one patch catches them all);
  * a per-shape graph cache and the identical build -> pin-the-engine ->
    ``check_support`` -> ``build_plans`` sequence, factored into :meth:`_cached_graph`;
  * hard-pinned execution (:meth:`_execute`);
  * the **native fallback with logging** the whole layer exists to give you: every
    call the engine can't claim goes back to real PyTorch, is counted, and is
    logged with the *reason* -- either the failed gate predicate or the caught
    exception. That per-reason tally (:meth:`format_report`) is the actionable
    "what is hipDNN still missing" list.

A concrete override subclasses this, sets :attr:`op_name`, and implements:

  * ``_call(self, real, *args, **kwargs)`` -- the drop-in replacement. It gates,
    then either builds/executes a graph (calling :meth:`note_aot`) or defers to
    ``real(...)`` (calling :meth:`note_native` with a reason).
  * ``_graph(self, ...)`` -- build and return a hipDNN ``Graph`` with tensors, the
    op, and the output uid set (but NOT yet ``build_operation_graph``'d);
    :meth:`_cached_graph` finalises and caches it.
"""

import logging

from . import bootstrap as _bootstrap

log = logging.getLogger("hipdnn_torch")


class NotApplicable(RuntimeError):
    """Internal: the engine cannot serve this graph/shape. Always caught and
    turned into a native fallback -- never escapes to the model."""


class OpOverride:
    #: ``torch.nn.functional`` attribute this override patches, e.g. ``"linear"``.
    op_name = None

    def __init__(self):
        self._installed = False
        self._real = None
        self._graph_cache = {}
        self._census = {}      # census-key -> {"aot": int, "native": int, **extras}
        self._fallbacks = {}   # reason -> count (the "gaps" tally)
        self.state = None      # bootstrap.State, set on install()

    # -- convenience --------------------------------------------------------
    @property
    def installed(self) -> bool:
        return self._installed

    def _tok(self, dtype) -> str:
        torch = self.state.torch
        return {
            torch.float16: "f16",
            torch.bfloat16: "bf16",
            torch.float32: "f32",
        }.get(dtype, str(dtype))

    # -- graph build / execute (shared across every op) ---------------------
    def _cached_graph(self, key, build, describe):
        """Return a cached ``{"graph", "ws"}`` for ``key``; on a miss, call
        ``build()`` (subclass returns a wired-but-unbuilt ``Graph``), run the
        shared finalise sequence, and cache it. ``describe`` is a short shape
        string for the not-applicable message."""
        entry = self._graph_cache.get(key)
        if entry is not None:
            return entry

        st = self.state
        g = build()

        err = g.build_operation_graph(st.handle)
        if err.is_bad():
            raise NotApplicable(f"build_operation_graph: {err.get_message()}")

        ranked = g.get_ranked_engine_ids([st.hipdnn.HeuristicMode.FALLBACK])
        if st.engine_id not in ranked:
            raise NotApplicable(f"{st.engine_name} not applicable for {describe}")

        err = g.create_execution_plan_ext(st.engine_id)
        if err.is_bad():
            raise NotApplicable(f"create_execution_plan_ext: {err.get_message()}")
        err = g.check_support()
        if err.is_bad():
            raise NotApplicable(f"check_support: {err.get_message()}")
        err = g.build_plans()
        if err.is_bad():
            raise NotApplicable(f"build_plans: {err.get_message()}")

        entry = {"graph": g, "ws": g.get_workspace_size()}
        self._graph_cache[key] = entry
        return entry

    def _execute(self, entry, variant_pack, device) -> None:
        """Allocate the workspace (if any) and run the pinned plan. ``variant_pack``
        maps uid -> device pointer (int); the workspace is an int pointer, 0 == none."""
        st = self.state
        ws = entry["ws"]
        workspace = (
            st.torch.empty(ws, dtype=st.torch.uint8, device=device) if ws > 0 else None
        )
        ws_ptr = workspace.data_ptr() if workspace is not None else 0
        err = entry["graph"].execute(st.handle, variant_pack, ws_ptr)
        if err.is_bad():
            raise NotApplicable(f"execute: {err.get_message()}")

    # -- census + fallback logging ------------------------------------------
    def _row(self, key) -> dict:
        return self._census.setdefault(key, {"aot": 0, "native": 0})

    def note_aot(self, key, **extras) -> None:
        """Count a call served by the engine. ``extras`` are extra integer
        counters folded into the row (e.g. ``biased=1``, ``weightless=1``)."""
        row = self._row(key)
        row["aot"] += 1
        for name, val in extras.items():
            row[name] = row.get(name, 0) + int(val)

    def note_native(self, key, reason, level=logging.INFO) -> None:
        """Count + log a native fallback. ``reason`` is a short human string (the
        failed gate or the exception); it is tallied for :meth:`format_report`.
        Gate declines log at INFO, unexpected exceptions at WARNING."""
        self._row(key)["native"] += 1
        self._fallbacks[reason] = self._fallbacks.get(reason, 0) + 1
        log.log(level, "%s -> native fallback [%s]: %s", self.op_name, key, reason)

    # -- install / uninstall ------------------------------------------------
    def install(self) -> None:
        if self._installed:
            return
        self.state = _bootstrap.bootstrap()
        functional = self.state.torch.nn.functional
        self._real = getattr(functional, self.op_name)
        real = self._real

        def wrapper(*args, **kwargs):
            return self._call(real, *args, **kwargs)

        setattr(functional, self.op_name, wrapper)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        setattr(self.state.torch.nn.functional, self.op_name, self._real)
        self._installed = False

    def _call(self, real, *args, **kwargs):
        raise NotImplementedError

    # -- reporting ----------------------------------------------------------
    def reset(self) -> None:
        self._census.clear()
        self._fallbacks.clear()

    def census(self) -> dict:
        return {k: dict(v) for k, v in self._census.items()}

    def totals(self):
        aot = sum(r["aot"] for r in self._census.values())
        native = sum(r["native"] for r in self._census.values())
        return aot, native

    def fallback_reasons(self) -> dict:
        return dict(self._fallbacks)

    def format_report(self) -> str:
        """Human-readable per-shape census + the ranked fallback-reason tally."""
        if not self._census:
            return f"{self.op_name}: (no intercepted calls)"

        aot, native = self.totals()
        extras = sorted(
            {k for r in self._census.values() for k in r if k not in ("aot", "native")}
        )
        lines = [f"{self.op_name} intercept census (shape -> aot / native):"]
        for key in sorted(self._census):
            row = self._census[key]
            tail = "".join(f"  {e}={row[e]}" for e in extras if row.get(e))
            lines.append(
                f"  {key:34s}  aot={row['aot']:5d}  native={row['native']:5d}{tail}"
            )
        lines.append(f"  {'TOTAL':34s}  aot={aot:5d}  native={native:5d}")
        if self._fallbacks:
            lines.append("  fallback reasons (why calls went native -- gaps to close):")
            for reason, cnt in sorted(self._fallbacks.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {cnt:5d}  {reason}")
        return "\n".join(lines)
