"""Parity adapter for rocKE's TILED attention path.

`dispatch_parity.py` (and the sibling tools that share its profile) drive three
profile-declared symbols:

    request = request_cls(**fields)          # dispatch_parity.py:178
    spec    = factory(request)               # dispatch_parity.py:179
    ok, why = predicate(spec, arch=arch)     # dispatch_parity.py:188

For the DENSE kernel those three land directly on rocKE symbols, because
`AttentionRequest` accepts `arch=`, `dense_spec_for_request` takes a request, and
`supports_attention_dense` takes a spec. **None of the three holds for the tiled
path**, and each failure is silent or misleading rather than loud:

1. **`UnifiedAttentionProblem` cannot be the `request.class`.** `resolve_shapes`
   injects `fields["arch"] = profile["arch"]` (dispatch_parity.py:171-172) whenever
   the profile declares an arch, and the problem dataclass has no `arch` field:

       TypeError: UnifiedAttentionProblem.__init__() got an unexpected
                  keyword argument 'arch'

   Every shape would land in the `rejected` bucket with a TypeError that reads like
   a corpus defect. Dropping `arch:` from the profile is not a fix -- five other
   tools read it.

2. **`supports_native_unified_attention_tiled` takes a PROBLEM, not a spec.**
   Verified by execution: passing it the spec `_tiled_spec_from_problem` returns
   raises `AttributeError: 'UnifiedAttention2DTiledSpec' object has no attribute
   'use_fp8'`. The dense predicate takes a spec; this one does not, and the
   signature difference is invisible from the profile.

3. **The predicate does NOT enforce the 2D/3D split, and that is the dangerous
   one.** `supports_native_unified_attention_tiled` answers for the tiled FAMILY.
   Measured over a 12-shape decode grid (num_seqs x max_seqlen_k), it returned
   `(True, 'supported')` for **12 of 12** shapes that `select_path()` routes to
   **3d**. Taking it at face value would size a 2D-only variant set from a corpus
   that includes every 3D shape -- descriptors built for a path this engine does
   not ship, matching graphs it cannot serve. This adapter enforces
   `select_path() == "2d"` as a first-class decline with a named reason, which is
   decision D4's scope made mechanical instead of assumed.

Nothing here reimplements rocKE. The spec still comes from
`kernels.common.attention_unified._tiled_spec_from_problem` -- the PRODUCTION
wrapper, which adds `_resolve_lds_budget` on top of the builders-layer function
(decision D5) -- and applicability still comes from rocKE's own predicate. This
module only translates the calling convention and adds the scope gate.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

__all__ = [
    "TiledAttentionRequest",
    "tiled_spec_for_request",
    "supports_tiled_2d_for_spec",
    "problem_for_spec",
]


#: Where the originating problem is parked on the spec the factory returns.
#:
#: The predicate needs the PROBLEM but the tool hands it the SPEC, and the two are
#: not interconvertible: `_tiled_spec_from_problem` is lossy (a 22-field problem
#: becomes a 46-field spec that drops `num_cus`, `target_ctas`, `max_seqlen_k`, ...).
#: Recomputing the problem from the spec would be a second, hand-written
#: implementation of the mapping -- exactly the restatement that ships wrong.
#:
#: The underscore prefix is load-bearing, not style: `dataclasses.fields()` reports
#: only declared fields, so an attribute set through `object.__setattr__` is
#: invisible to every consumer that enumerates the spec (`knob_partition`,
#: `build_config`, the metadata writer). Verified: 46 fields before and after.
_PROBLEM_ATTR = "_parity_problem"


@dataclasses.dataclass(frozen=True)
class TiledAttentionRequest:
    """One corpus row, in the vocabulary `UnifiedAttentionProblem` actually needs.

    Deliberately NOT `dispatch.attention.common.AttentionRequest`. That class's
    `_problem()` hard-codes `total_q = batch * seqlen_q` and `num_seqs = batch`, so
    every problem it builds is uniform and non-varlen -- which is the one thing a
    paged/varlen corpus must be able to vary. It also sources `block_size` from a
    `kv_block_size` field defaulting to 16, so a corpus that omits the key silently
    gets block_size=16 for every paged shape rather than an error.

    Field names match `UnifiedAttentionProblem`'s so a corpus row maps across
    without a rename table. `arch` is accepted and dropped: the tool injects it,
    the problem class rejects it (see module docstring, failure 1).
    """

    # --- required by UnifiedAttentionProblem (9) ---
    total_q: int
    num_seqs: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    max_seqlen_q: int
    max_seqlen_k: int
    dtype: str

    # --- optional, defaults mirroring UnifiedAttentionProblem's own ---
    q_dtype: Optional[str] = None
    sliding_window: int = 0
    softcap: float = 0.0
    use_sinks: bool = False
    use_alibi: bool = False
    use_qq_bias: bool = False
    use_fp8: bool = False
    fp8_fnuz: bool = False
    num_cus: int = 120
    target_ctas: int = 0
    waves_per_eu: Optional[int] = None
    compile_backend: Optional[str] = None
    num_kv_blocks: int = 0

    # --- the target arch ---
    #: Injected by dispatch_parity.py:171-172 from the profile's `arch:`. Never
    #: forwarded to the problem (which has no such field -- see failure 1), but NOT
    #: inert either: it is what `_pin_attention_arch` pins the module memo to. See
    #: that function for why a host-side tool MUST state the arch explicitly.
    arch: Optional[str] = None

    def problem(self):
        """The `UnifiedAttentionProblem` this row denotes.

        Imported lazily so the module is importable without the rocKE library on
        `sys.path` -- `dispatch_parity.py` calls `_bind_provider()` (which extends
        `sys.path`) AFTER the profile is loaded but BEFORE any symbol is imported,
        and a module-level import here would run in the wrong order.
        """
        from kernels.common.attention_unified import UnifiedAttentionProblem

        fields = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "arch"
        }
        return UnifiedAttentionProblem(**fields)


def _pin_attention_arch(arch: Optional[str]) -> None:
    """Pin rocKE's memoized attention arch. **Load-bearing, not defensive.**

    `_resolve_attention_arch()` (`attention_unified.py:237-263`) queries the LOCAL
    DEVICE via `get_device_arch()`, memoizes the answer in a module global, and falls
    back to `'gfx950'` only when the query FAILS. There is no environment variable.

    **The fallback is not the common case on a dev box, and assuming it was cost this
    run a diagnosis.** The prior plan recorded that "host tooling works today only
    because the fallback happens to be our arch" -- that is true only on a machine with
    no GPU at all. This workstation has a real gfx90a, so `get_device_arch()` SUCCEEDS
    and returns `'gfx90a'`; the fallback is never reached. Measured:

        get_device_arch() -> 'gfx90a'

    Unpinned, `dispatch_parity.py` therefore resolved **0 of 79** shapes servable --
    73 declined with "neither path is available on gfx90a" and 6 raised LDS-budget
    RuntimeErrors against gfx90a's 64 KB cap. Every one of those reads like a real
    applicability finding about the tiled kernel. It is not: it is this memo.

    Pinning here rather than in a wrapper script is deliberate -- the profile's `arch:`
    is the single declaration of intent, five tools read it, and any of them that
    imports this adapter gets the pin for free. A tool that forgot a shell export would
    otherwise silently measure a different GPU.
    """
    if not arch:
        return
    from kernels.common import attention_unified

    current = getattr(attention_unified, "_RESOLVED_ATTENTION_ARCH", None)
    if current == arch:
        return
    if current is not None:
        # Never silently re-pin: specs already built under the old arch would be
        # inconsistent with those built after, and the resulting set would be a
        # silent mixture rather than an error.
        raise RuntimeError(
            f"attention arch already resolved to {current!r}, cannot re-pin to "
            f"{arch!r}. Specs built under two arches would silently mix. Run one "
            f"arch per process."
        )
    attention_unified._RESOLVED_ATTENTION_ARCH = arch


def tiled_spec_for_request(request: TiledAttentionRequest):
    """The profile's `dispatch:` entry point. Returns the spec rocKE would build.

    Calls `kernels.common.attention_unified._tiled_spec_from_problem` -- the
    PRODUCTION wrapper (decision D5), which applies `_resolve_lds_budget` on top of
    `builders.common.attention_spec_builder._tiled_spec_from_problem`. The
    builders-layer function omits that pass, so a descriptor built from it bakes a
    pre-LDS-budget spec.

    No overrides are passed, and that is deliberate. `_spec_gfx950_generic` ALREADY
    folds `_d256_gfx950_spec_overrides()` in via a tail `replace()` for exactly the
    D256 cohort. Applying them again off-cohort silently builds a DIFFERENT binary
    -- 7 fields differ and the kernel name changes -- with no error. The resolver's
    output is baked unmodified; the override dict is never hand-transcribed and none
    of its 7 fields may be passed to `--knobs`.
    """
    # BEFORE the import that triggers resolution. `_tiled_spec_from_problem` reaches
    # `_resolve_attention_arch()` on its first call and memoizes whatever it finds,
    # so pinning afterwards is too late for the whole run.
    _pin_attention_arch(request.arch)

    from kernels.common.attention_unified import _tiled_spec_from_problem

    problem = request.problem()
    spec = _tiled_spec_from_problem(problem)
    # Frozen dataclass; `object.__setattr__` is the only way in. See _PROBLEM_ATTR.
    object.__setattr__(spec, _PROBLEM_ATTR, problem)
    return spec


def problem_for_spec(spec):
    """The problem a spec was built from, or None if it did not come from here."""
    return getattr(spec, _PROBLEM_ATTR, None)


def supports_tiled_2d_for_spec(spec, arch: Optional[str] = None) -> Tuple[bool, str]:
    """The profile's `predicate:` entry point, in the (spec, arch=) shape the tool calls.

    Two gates, in this order, and the order matters: rocKE's own applicability
    answer comes first so a shape this kernel genuinely cannot build is reported
    with the KERNEL's reason rather than being masked by our scope decision. Only
    then does the 2D-only scope apply.
    """
    problem = problem_for_spec(spec)
    if problem is None:
        # Never silently pass. A spec built by some other factory carries no
        # problem, and answering "supported" for it would inflate the servable
        # count with shapes nothing verified.
        return False, (
            "no originating problem on this spec; it was not built by "
            "tiled_spec_for_request, so tiled applicability cannot be evaluated"
        )

    from kernels.common.attention_unified import supports_native_unified_attention_tiled

    supported, why = supports_native_unified_attention_tiled(problem)
    if not supported:
        return False, str(why)

    # Decision D4: this engine ships the 2D path only. The rocKE predicate above
    # answers for the tiled FAMILY and returns True for 3D shapes too (measured:
    # 12 of 12 on a decode grid), so without this gate the parity set would carry
    # descriptors for a path this engine does not ship.
    #
    # A NAMED, source-derived decline -- `select_path()` delegates to
    # `rocke.helpers.attention.use_2d_kernel` -- not a vague gap. These shapes route
    # to 3D precisely because the 2D grid under-fills the device, so declining them
    # is a CORRECT decline rather than a silently worse path.
    path = problem.select_path()
    if path != "2d":
        return False, (
            f"select_path()=={path!r}: routed to the 3D split-KV path, which this "
            f"engine does not ship (decision D4). Long-context small-batch decode; "
            f"3D needs a different spec class, two coupled kernels and a non-zero "
            f"workspace, so it is a separate engine."
        )

    return True, "supported by tiled 2D on the 2D-routed path"
