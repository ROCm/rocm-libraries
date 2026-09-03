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

# Reverse id->name for the engines we co-load. The shipped bindings'
# engine_id_to_name() registry can predate a plugin engine and return '' for it,
# so we resolve the selected engine's id through this local table first.
_KNOWN_ENGINES = (
    "AOT_CATALOG_ENGINE",
    "ASM_SDPA_ENGINE",
    "HIPBLASLT_ENGINE",
    "HIP_MLOPS_ENGINE",
    "MIOPEN_ENGINE",
    "MIOPEN_ENGINE_DETERMINISTIC",
)
_ID_TO_NAME = {_bootstrap._fnv1a64(n): n for n in _KNOWN_ENGINES}


def _engine_name(hipdnn, engine_id) -> str:
    """Best-effort id -> engine name: local table, then the bindings' registry,
    then the raw hex id."""
    name = _ID_TO_NAME.get(engine_id)
    if name:
        return name
    try:
        got = hipdnn.engine_id_to_name(engine_id)
        if got:
            return got
    except Exception:  # noqa: BLE001 -- registry lookup is best-effort
        pass
    return hex(engine_id & 0xFFFFFFFFFFFFFFFF)


def _ws(graph, getter) -> int:
    """A workspace getter's value, or 0 when it does not apply / is unbound.

    The plan-spec and compiled-plan paths each have their own sizer, and a
    getter for the path not taken may be absent or may raise rather than
    return 0. Callers want the max across both, so a miss must read as 0.
    """
    fn = getattr(graph, getter, None)
    if fn is None:
        return 0
    try:
        return int(fn() or 0)
    except Exception:  # noqa: BLE001 -- wrong path for this getter
        return 0


class NotApplicable(RuntimeError):
    """Internal: the engine cannot serve this graph/shape. Always caught and
    turned into a native fallback -- never escapes to the model."""


class OpOverride:
    #: ``torch.nn.functional`` attribute this override patches, e.g. ``"linear"``.
    op_name = None

    def __init__(self):
        self._installed = False
        self._real = None
        # graph-key -> {"graph", "ws"} for shapes the engine serves. Grows once
        # per distinct shape and holds hipDNN Graph objects (GPU resources); it
        # is never evicted. Fine for a bring-up tool whose shape set is bounded,
        # but a very-many-shapes, long-running process would accumulate memory.
        self._graph_cache = {}
        self._nope_cache = {}  # graph-key -> reason, for shapes the engine rejects
        self._census = {}  # census-key -> {"aot": int, "native": int, **extras}
        self._fallbacks = {}  # reason -> count (the "gaps" tally)
        self._last_engine = None  # winning engine name from the last _cached_graph
        self._tune_log = []  # one record per exhaustive sweep actually run
        self.state = None  # bootstrap.State, set on install()

    # -- convenience --------------------------------------------------------
    @property
    def installed(self) -> bool:
        return self._installed

    def _tok(self, dtype) -> str:
        """Short, stable census token for a dtype. Derived generically from the
        dtype's name (``torch.bfloat16`` -> ``bf16``, ``torch.float8_e4m3fn`` ->
        ``f8_e4m3fn``) so any dtype -- including ones added later -- renders sensibly
        without a hardcoded table."""
        name = getattr(dtype, "__name__", None) or str(dtype)
        if name.startswith("torch."):
            name = name[len("torch.") :]
        # float32->f32, float16->f16, bfloat16->bf16, float8_e4m3fn->f8_e4m3fn
        name = name.replace("bfloat", "bf").replace("float", "f")
        return name

    def _tuning_requested(self) -> bool:
        """Whether this graph should be swept before it is first executed.

        PyTorch already has the exact concept: ``torch.backends.cudnn.benchmark``
        means "spend time measuring the available algorithms for this shape, pick
        the fastest, and cache it for later calls of the same shape". hipDNN's
        exhaustive sweep plus its exact-match ranking cache is that same contract,
        so the flag drives it and models that already opt in get tuned selection
        with no hipDNN-specific code:

            torch.backends.cudnn.benchmark = True   # sweeps, caches

        It is read HERE, per graph build, not once at bootstrap -- the flag is
        routinely toggled at runtime (and around a warmup region), and reading it
        early would freeze whatever value happened to be set at import.

        ``HIPDNN_TORCH_TUNE`` overrides the flag in both directions, for a run
        that must tune without touching the model's own backend settings (or must
        not tune despite them).
        """
        st = self.state
        # A forced engine is a candidate set of one: the sweep would measure a
        # field of one and the cache write is refused as a partial sweep.
        if st.select_mode != "default":
            return False
        if st.tune_mode == "tune":
            return True
        if st.tune_mode == "off_explicit":
            return False
        try:
            return bool(st.torch.backends.cudnn.benchmark)
        except Exception:  # noqa: BLE001 -- no cudnn shim on this build
            return False

    # -- graph build / execute (shared across every op) ---------------------
    def _cached_graph(self, key, build, describe):
        """Return a cached ``{"graph", "ws"}`` for ``key``; on a miss, call
        ``build()`` (subclass returns a wired-but-unbuilt ``Graph``), run the
        shared finalise sequence, and cache it. ``describe`` is a short shape
        string for the not-applicable message.

        **Cache-key invariant:** since the overrides build from each tensor's actual
        dims + strides + per-tensor dtype (no forced layout/dtype), the ``key`` a
        subclass passes *must* encode all three -- dim tuples, stride tuples, and
        dtypes (plus any op params baked into the graph). Two calls with identical
        dims but different layout/dtype build *different* graphs; a coarser key would
        collide and run the first graph against the second's mismatched pointers.

        A shape the engine rejects is remembered too (:attr:`_nope_cache`): the
        Python gate is coarse, so a key can pass the gate yet fail here, and
        re-running the (non-trivial) ``build_operation_graph`` /
        ``get_ranked_engine_ids`` probe on every call would be wasteful. The
        rejection reason is cached and re-raised immediately on later calls, so
        each such shape probes once and then falls back cheaply."""
        entry = self._graph_cache.get(key)
        if entry is not None:
            self._last_engine = entry["engine"]
            return entry
        nope = self._nope_cache.get(key)
        if nope is not None:
            raise NotApplicable(nope)

        st = self.state
        try:
            g = build()

            err = g.build_operation_graph(st.handle)
            if err.is_bad():
                raise NotApplicable(f"build_operation_graph: {err.get_message()}")

            # No engine claiming the graph is a clean decline, not an error. Most
            # backends signal that with an empty ranking; some (this gfx1151
            # nightly's binding) instead RAISE "Failed to get ranked engine ids:
            # No engine configurations available for the graph". Translate either
            # form to NotApplicable so the op's fallback ladder (e.g. linear's
            # fused-bias -> matmul-only rung) runs instead of aborting to full
            # native -- an unclaimed fused graph must not sink the matmul.
            try:
                ranked = g.get_ranked_engine_ids([st.hipdnn.HeuristicMode.FALLBACK])
            except (
                Exception
            ) as err:  # noqa: BLE001 -- ranking probe failure == no engine
                raise NotApplicable(f"get_ranked_engine_ids raised: {err}")
            if not ranked:
                raise NotApplicable(f"no engine applicable for {describe}")

            # THE TWO PATHS ARE MUTUALLY EXCLUSIVE. hipDNN offers either the
            # compiled-plan path (create_execution_plans -> build_plans) or the
            # plan-spec path (add_engine*() -> build_plans(ALL) -> autotune);
            # calling add_all_engines() after create_execution_plans() is
            # rejected outright. So tuning REPLACES heuristic selection here, it
            # does not follow it.
            #
            # The sweep itself cannot run at this point -- it needs the real
            # device pointers, which only exist at execute time -- so it is
            # deferred to the first _execute() for this graph (see _maybe_tune).
            tuning = self._tuning_requested()
            if tuning:
                # Plan-spec path: collect every applicable engine as a candidate.
                err = g.add_all_engines()
                if err.is_bad():
                    raise NotApplicable(f"add_all_engines: {err.get_message()}")
            elif st.select_mode == "default":
                # Compiled-plan path. Hand the graph to hipDNN and let it select
                # across every loaded engine. The backend Config policy
                # (HIPDNN_HEUR_CONFIG_PATH) participates here, so a rule file --
                # including one a previous tuning run wrote -- can decide the
                # engine per graph/shape without any code change.
                err = g.create_execution_plans([st.hipdnn.HeuristicMode.FALLBACK])
                if err.is_bad():
                    raise NotApplicable(f"create_execution_plans: {err.get_message()}")
            else:  # force: pin the configured engine, bypassing ranking
                if st.engine_id not in ranked:
                    raise NotApplicable(
                        f"{st.engine_name} not applicable for {describe}"
                    )
                err = g.create_execution_plan_ext(st.engine_id)
                if err.is_bad():
                    raise NotApplicable(
                        f"create_execution_plan_ext: {err.get_message()}"
                    )

            if tuning:
                # ALL, not the default: the sweep ranks compiled plans, so every
                # candidate has to be compiled before it can be timed. check_support()
                # belongs to the compiled-plan path and is skipped here -- an
                # unsupported candidate simply fails to compile and drops out.
                err = g.build_plans(st.hipdnn.BuildPlanPolicy.ALL)
            else:
                err = g.check_support()
                if err.is_bad():
                    raise NotApplicable(f"check_support: {err.get_message()}")
                err = g.build_plans()
            if err.is_bad():
                raise NotApplicable(f"build_plans: {err.get_message()}")

            # Ground-truth winner of the (built) plan -- the engine that will run.
            # Under tuning this is provisional; _maybe_tune re-reads it after the
            # sweep leaves the measured winner active.
            try:
                engine = _engine_name(st.hipdnn, g.get_execution_plan_engine_id())
            except Exception:  # noqa: BLE001 -- fall back to the pinned name
                engine = st.engine_name if st.select_mode == "force" else "?"
        except NotApplicable as na:
            self._nope_cache[key] = str(na)
            raise

        entry = {
            "graph": g,
            # Under tuning the workspace must cover EVERY candidate, not just the
            # heuristic pick: a short workspace holds an applicable engine out of
            # the timing loop while leaving it a live candidate for a later run,
            # and the cache write is then refused as a partial sweep. The two
            # getters cover the two paths -- get_estimated_max_workspace_size()
            # sizes the plan-spec path this build used, get_autotune_workspace_size()
            # the compiled-plan path -- so take the max and let a zero from the
            # inapplicable one fall away.
            "ws": (
                max(_ws(g, "get_estimated_max_workspace_size"),
                    _ws(g, "get_autotune_workspace_size"),
                    _ws(g, "get_workspace_size"))
                if tuning
                else g.get_workspace_size()
            ),
            "engine": engine,
            "tuned": not tuning,  # False => _execute runs the sweep once
        }
        self._graph_cache[key] = entry
        self._last_engine = engine
        return entry

    def _maybe_tune(self, entry, variant_pack, ws_ptr) -> None:
        """Run the exhaustive sweep once for this graph, then mark it tuned.

        This is the whole point of tune mode: hipDNN benchmarks every applicable
        engine on the REAL tensors, ranks them on ``robust_time_ms``, writes that
        order to the exact-match cache (keyed on graph + device), and leaves the
        winning plan active. Every later run of the same graph -- including a
        later process with ``HIPDNN_TORCH_TUNE`` unset -- inherits the ranking
        without re-measuring.

        Failure is deliberately non-fatal: a graph that cannot be swept still has
        a built heuristic plan, so it executes as it would have without tuning.
        """
        st = self.state
        entry["tuned"] = True  # once, whatever happens -- never re-sweep
        cfg = st.hipdnn.AutotuneConfig()
        storage = st.hipdnn.AutotuneStorageConfig()
        if st.tune_config_path:
            # Also emit the portable JSON rules file. The exact-match cache is
            # keyed on graph+device and is the fast path; this file is the
            # shareable artifact, replayable via HIPDNN_HEUR_CONFIG_PATH.
            storage.file_path = st.tune_config_path
            storage.delete_all_existing_file_content = False
        try:
            results, outcome = entry["graph"].autotune_exhaustive_sweep(
                st.handle, variant_pack, ws_ptr, entry["ws"], cfg, storage
            )
        except Exception as ex:  # noqa: BLE001 -- a failed sweep must not break the model
            log.warning("%s: exhaustive sweep failed, using heuristic plan: %s",
                        self.op_name, ex)
            return

        ranked = [r for r in results if r.succeeded]
        ranked.sort(key=lambda r: r.robust_time_ms)
        if ranked:
            try:
                entry["engine"] = _engine_name(
                    st.hipdnn, entry["graph"].get_execution_plan_engine_id()
                )
                self._last_engine = entry["engine"]
            except Exception:  # noqa: BLE001
                pass
            # Re-read: the winning plan is now active and may want more workspace
            # than the heuristic pick did.
            entry["ws"] = max(entry["ws"], entry["graph"].get_workspace_size())
        self._tune_log.append(
            {
                "op": self.op_name,
                "outcome": getattr(outcome, "name", str(outcome)),
                "winner": entry["engine"],
                "candidates": len(results),
                "benchmarked": sum(1 for r in results if r.benchmarked),
                "best_ms": ranked[0].robust_time_ms if ranked else None,
                "ranking": [(r.engine_name, r.robust_time_ms) for r in ranked[:8]],
            }
        )
        # A refused write means later runs will NOT inherit this ranking, which
        # silently turns "tuned" into "tuned once, per process". Say so loudly.
        name = getattr(outcome, "name", str(outcome))
        if name in ("WRITTEN", "UNCHANGED"):
            log.info("%s: swept %d candidates, winner=%s (%s)",
                     self.op_name, len(results), entry["engine"], name)
        else:
            log.warning(
                "%s: swept %d candidates but the ranking was NOT cached (%s) -- "
                "later runs will re-decide from the heuristic",
                self.op_name, len(results), name)

    def _execute(self, entry, variant_pack, device) -> None:
        """Allocate the workspace (if any) and run the pinned plan. ``variant_pack``
        maps uid -> device pointer (int); the workspace is an int pointer, 0 == none."""
        st = self.state
        ws = entry["ws"]
        workspace = (
            st.torch.empty(ws, dtype=st.torch.uint8, device=device) if ws > 0 else None
        )
        ws_ptr = workspace.data_ptr() if workspace is not None else 0
        if not entry.get("tuned", True):
            # First execution of this graph under tune mode: sweep on these very
            # pointers, then fall through and execute with the winner active.
            self._maybe_tune(entry, variant_pack, ws_ptr)
            ws = entry["ws"]
            if ws > (workspace.numel() if workspace is not None else 0):
                workspace = st.torch.empty(ws, dtype=st.torch.uint8, device=device)
                ws_ptr = workspace.data_ptr()
        err = entry["graph"].execute(st.handle, variant_pack, ws_ptr)
        if err.is_bad():
            raise NotApplicable(f"execute: {err.get_message()}")

    # -- census + fallback logging ------------------------------------------
    def _row(self, key) -> dict:
        return self._census.setdefault(key, {"aot": 0, "native": 0})

    def note_aot(self, key, **extras) -> None:
        """Count a call served by the engine. ``extras`` are extra integer
        counters folded into the row (e.g. ``biased=1``, ``weightless=1``). The
        winning engine name from the just-run :meth:`_cached_graph` is recorded in
        the row's ``engines`` set so ``default`` runs show *which* engine served
        each shape."""
        row = self._row(key)
        row["aot"] += 1
        if self._last_engine:
            row.setdefault("engines", set()).add(self._last_engine)
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
            {
                k
                for r in self._census.values()
                for k in r
                if k not in ("aot", "native", "engines")
            }
        )
        lines = [f"{self.op_name} intercept census (shape -> aot / native):"]
        for key in sorted(self._census):
            row = self._census[key]
            tail = "".join(f"  {e}={row[e]}" for e in extras if row.get(e))
            engines = row.get("engines")
            if engines:
                tail += "  engine=" + ",".join(sorted(engines))
            lines.append(
                f"  {key:34s}  aot={row['aot']:5d}  native={row['native']:5d}{tail}"
            )
        lines.append(f"  {'TOTAL':34s}  aot={aot:5d}  native={native:5d}")
        if self._fallbacks:
            lines.append("  fallback reasons (why calls went native -- gaps to close):")
            for reason, cnt in sorted(self._fallbacks.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {cnt:5d}  {reason}")
        return "\n".join(lines)
