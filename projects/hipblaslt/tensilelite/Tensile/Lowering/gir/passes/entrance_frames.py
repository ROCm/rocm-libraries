# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Make every frame phi's predecessor inputs explicit."""

from __future__ import annotations

from .base import Pass
from ..analyses.cfg import BackEdges
from ..nodes import GenPhi, Move, Mma


class EntranceFramePhisPass(Pass):
    """Generalize loop-header phis and add phis at path-conditioned drain joins."""

    def run(self, prog, am):
        back = am.get(BackEdges(), prog)
        changed = False
        for block in prog.blocks.values():
            by_gen = {phi.gen.id: phi for phi in block.phis}
            order = [phi.gen.id for phi in block.phis]

            for gid, phi in list(by_gen.items()):
                incoming = dict(phi.incomings)
                for pred in block.preds:
                    if pred in incoming:
                        continue
                    incoming[pred] = (
                        None
                        if back.is_back_edge(pred, block.label)
                        else phi.entry_val
                    )
                by_gen[gid] = GenPhi(
                    gen=phi.gen,
                    entry_val=phi.entry_val,
                    incomings=tuple((pred, incoming[pred]) for pred in block.preds),
                )

            for pred, chunk_base in sorted(block.path_chunk_base.items()):
                if pred not in block.preds:
                    raise RuntimeError(
                        "entrance frame names non-predecessor %s->%s"
                        % (pred, block.label)
                    )
                if block.gen_rel is None:
                    raise RuntimeError(
                        "entrance frame %s->%s has no relative frame"
                        % (pred, block.label)
                    )
                required = {}
                for inst in block.body:
                    if not isinstance(inst, (Move, Mma)):
                        continue
                    for ref in tuple(inst.srcs) + tuple(inst.dsts):
                        gen = getattr(ref, "gen", None)
                        if ref.tile.space != "shared" or gen is None:
                            continue
                        ring = max(1, int(gen.ring))
                        phase = (int(chunk_base) - int(block.gen_rel)) % ring
                        prior = required.setdefault(gen.id, (gen, phase))
                        if prior[1] != phase:
                            raise RuntimeError(
                                "entrance frame %s->%s requires conflicting phases %d/%d "
                                "for Gen %d"
                                % (pred, block.label, prior[1], phase, gen.id)
                            )
                for gid, (gen, phase) in sorted(required.items()):
                    phi = by_gen.get(gid)
                    incoming = (
                        dict(phi.incomings)
                        if phi is not None
                        else {name: None for name in block.preds}
                    )
                    prior = incoming.get(pred)
                    if prior is not None and int(prior) != phase:
                        raise RuntimeError(
                            "entrance frame %s->%s conflicts with phi phase %d/%d "
                            "for Gen %d"
                            % (pred, block.label, int(prior), phase, gid)
                        )
                    incoming[pred] = phase
                    by_gen[gid] = GenPhi(
                        gen=gen,
                        entry_val=(phi.entry_val if phi is not None else None),
                        incomings=tuple(
                            (name, incoming.get(name)) for name in block.preds
                        ),
                    )
                    if gid not in order:
                        order.append(gid)

            phis = [by_gen[gid] for gid in order]
            if block.phis != phis:
                block.phis = phis
                changed = True
        if changed:
            prog.bump()
