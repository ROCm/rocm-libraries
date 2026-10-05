# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
FoldShortPathPass -- merge the `T < M` arm's value order into the emitted prologue/drain scaffold,
then remove the arm.
"""

from __future__ import annotations

from ..nodes import CondGoto, Goto, LoopBack
from ..nodes import terminator_targets
from ..analyses.short_path import ShortPathFold, FOLD, SPLIT, UNSOUND, NA, VACUOUS
from .base import Pass


def folded(prog) -> bool:
    """Did the `T < M` arm get folded into the shared prologue/drain path?

    The question a consumer must ASK rather than assume.  True also when there was no arm to begin
    with (M == 0): the shared shape is then trivially what the Program has."""
    sl = prog.meta.get("short_loop")
    return sl is None or sl.get("verdict") == "folded"


class FoldShortPathPass(Pass):
    """See module docstring.  The short arm is the value-order authority; this pass first transfers
    any ordering constraints exposed by the def-use proof, then deletes its blocks and retargets
    the CFG to the physical scaffold."""

    def run(self, prog, am):
        v = am.get(ShortPathFold(), prog)
        sl = prog.meta.get("short_loop")
        if v.verdict == NA:
            return

        # A SPLIT is not a kernel rejection: it is the proof identifying an early folded-path write
        # and every old-value consumer that write crossed. Transfer those exact constraints into
        # the drain bodies, then rerun the independent value proof before deleting the arm.
        limit = 1 + sum(len(block.body) for block in prog.blocks.values())
        for _round in range(limit):
            if v.verdict != SPLIT or not self._merge_short_ordering(prog, v):
                break
            am.invalidate()
            v = am.get(ShortPathFold(), prog)
        else:
            raise RuntimeError("short-path ordering merge did not converge")

        if v.verdict == UNSOUND:
            # EVERY unserved consumer, not just the first: which OPERANDS are unserved is the whole
            # diagnosis: A and B can be served while the MX scales are not.
            worst = "".join("      %r\n" % (u,) for u in (v.unbound or ())[:8])
            raise RuntimeError(
                f"the `T < M` path cannot be emitted in EITHER shape: {v.reason}.\n"
                f"  unserved consumers ({len(v.unbound or ())}):\n{worst}"
                f"  shape: short_loop={sl!r} peel_depth={prog.meta.get('peel_depth')!r} "
                f"inner_order={tuple(prog.meta.get('inner_axis_order') or ())} "
                f"mma_inputs={tuple(prog.meta.get('mma_inputs') or ())}\n"
                f"Folding would hide it behind a path that happens to work; emitting the arm would "
                f"ship known-broken code.  Fix the arm (it is built by `emit.build_ir`, not here) -- "
                f" for the two ways this has happened.")

        if sl is None:                       # blocks exist but nothing recorded them: a lowering bug
            raise RuntimeError(
                "short blocks are present but `meta['short_loop']` is missing -- the arm's verdict "
                "must be carried, not inferred from block names")

        record = dict(sl, fold=v.verdict, reason=v.reason, obligations=v.obligations,
                      extras=v.extras)

        self._fold_arm(prog, record, v)

    @staticmethod
    def _merge_short_ordering(prog, verdict):
        """Sink each proven early write after its last old-value consumer.

        `ShortPathFold` supplies concrete producer/consumer nodes from the failed value pairs, so
        this transformation does not reconstruct logical values from register names or read
        metadata.  That distinction matters for a carrier read which fills several logical units:
        the next chunk has the same within-chunk coordinates but is still a different value.
        """
        by_block = {}
        for ordering in verdict.ordering:
            block = prog.blocks.get(ordering.block)
            if block is None:
                continue
            positions = {id(node): pos for pos, node in enumerate(block.body)}
            producer_pos = positions.get(id(ordering.producer))
            consumer_pos = positions.get(id(ordering.consumer))
            if producer_pos is None or consumer_pos is None or producer_pos >= consumer_pos:
                continue
            by_block.setdefault(ordering.block, []).append(
                (producer_pos, consumer_pos, ordering.producer, ordering.consumer))

        changed = False
        for label, constraints in by_block.items():
            block = prog.blocks[label]
            positions = {id(node): pos for pos, node in enumerate(block.body)}
            latest = {}
            for _producer_pos, consumer_pos, producer, consumer in constraints:
                prior = latest.get(id(producer))
                if prior is None or positions[id(prior)] < consumer_pos:
                    latest[id(producer)] = consumer

            movers = set(latest)
            after = {}
            for producer_id, consumer in latest.items():
                producer = next(node for node in block.body if id(node) == producer_id)
                after.setdefault(id(consumer), []).append(producer)
            for nodes in after.values():
                nodes.sort(key=lambda node: positions[id(node)])

            merged = []
            for node in block.body:
                if id(node) not in movers:
                    merged.append(node)
                merged.extend(after.get(id(node), ()))
            if any(left is not right for left, right in zip(block.body, merged)):
                block.body[:] = merged
                changed = True

        if changed:
            prog.bump()
        return changed

    def _fold_arm(self, prog, record, v):
        """FOLD/VACUOUS: rewire every edge into the short arm, then drop its blocks."""
        assert v.verdict in (FOLD, SPLIT, VACUOUS), v.verdict
        short = set(v.short_blocks)
        first_drain = v.folded_blocks[1] if len(v.folded_blocks) > 1 else "end"

        # THE FOLDED EDGE INHERITS THE ARM'S FRAME.  Rewiring `-> short0` into `-> drain0`
        # does not merely change a target: the two land at DIFFERENT positions on the summation
        arm_base = prog.blocks[v.short_blocks[0]].chunk_base if v.short_blocks else 0

        def _rewire(src_lab):
            # `first_drain` is "end" when the coverage leaves no drain block to land in; there is then
            # no frame to state.
            if first_drain in prog.blocks:
                prog.blocks[first_drain].path_chunk_base[src_lab] = arm_base

        for lab, blk in prog.blocks.items():
            if lab in short:
                continue
            t = blk.term
            if isinstance(t, CondGoto) and t.f_target in short:
                _rewire(lab)
                blk.term = CondGoto(t.pred, t.t_target, first_drain)
                blk.succs = (t.t_target, first_drain)
            elif isinstance(t, Goto) and t.target in short:
                _rewire(lab)
                blk.term = Goto(first_drain)
                blk.succs = (first_drain,)
            elif isinstance(t, LoopBack) and t.exit_target in short:
                _rewire(lab)
                blk.term = LoopBack(t.trips, t.body, first_drain, t.label)
                blk.succs = (t.body, first_drain)

        for lab in short:
            del prog.blocks[lab]

        # preds are a DECLARED fact checked by G-TERM, so recompute them from the rewritten
        # terminators rather than patching drain0 by name.
        computed = {lab: set() for lab in prog.blocks}
        for lab, blk in prog.blocks.items():
            for tgt in terminator_targets(blk.term):
                if tgt in prog.blocks:
                    computed[tgt].add(lab)
        for lab, blk in prog.blocks.items():
            if blk.preds:
                blk.preds = tuple(sorted(computed[lab]))

        record["verdict"] = "folded"
        # the arm is GONE, so the model-only declaration goes with it: a record naming deleted
        # blocks would leave G-EMIT checking a count against nothing.
        record.pop("model_only", None)
        prog.meta["short_loop"] = record
        prog.bump()


