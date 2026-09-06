# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.varlen_aten -- the ATen-level route for paged varlen attention.

:mod:`~hipdnn_torch.varlen` patches the public ``varlen_attn`` wrapper, which is
the ergonomic seam. This module registers the *same* mapping one layer down, on the
dispatcher, so callers that build the op call themselves -- vLLM-style serving
stacks, traced/compiled graphs, anything reaching ``torch.ops.aten`` directly --
are routed too.

**Which op, and why not the obvious one.** The plan named
``aten::_cudnn_attention_forward``. That is the wrong target on ROCm, for two
independent reasons, both verified against the installed wheel rather than the
PyTorch source tree:

  1. **Its schema has no paged arguments.** In every published ROCm wheel the op
     ends at ``scale`` -- no ``seqused_k``, no ``block_table``. The paged
     signature exists on ``ROCm/pytorch@develop`` but has not reached an artifact.
  2. **ROCm never selects that backend anyway.** ``varlen.py``'s
     ``_should_use_cudnn`` is compiled down to ``return False`` on a HIP build, so
     the cuDNN branch is unreachable regardless of its schema.

The op paged traffic actually flows through is ``aten::_flash_attention_forward``,
which *does* carry ``seqused_k`` and ``block_table``. That is what is registered
here.

**Registration is CUDA-key, not CompositeImplicitAutograd.** The op's paged path
only ever runs on device tensors, and overriding the composite key would also
capture meta/tracing calls, where there is no pointer to execute against.
"""

import logging

from .base import NotApplicable
from .varlen import VarlenSdpaOverride

log = logging.getLogger("hipdnn_torch")

_OP_NAME = "_flash_attention_forward"

#: Arguments the paged mapping cannot express. Any of these present and non-default
#: sends the call back to the native kernel, counted with a reason.
_UNMAPPED = ("alibi_slopes",)


class AtenVarlenRoute:
    """Registers a hipDNN implementation of ``aten::_flash_attention_forward``.

    Not an :class:`~hipdnn_torch.base.OpOverride`: that class patches a Python
    attribute, while this one installs a dispatcher kernel. It *reuses* the
    override's graph building and census so both routes report through one tally
    and neither can drift from the other's mapping."""

    def __init__(self, override=None):
        # Share one override instance (and therefore one census + graph cache)
        # with the module-level route when the caller passes theirs in.
        self._override = override if override is not None else VarlenSdpaOverride()
        self._lib = None
        self._installed = False

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def override(self):
        return self._override

    def install(self) -> None:
        if self._installed:
            return
        ov = self._override
        if ov.state is None:
            from . import bootstrap as _bootstrap

            ov.state = _bootstrap.bootstrap()
        torch = ov.state.torch

        if not hasattr(torch.ops.aten, _OP_NAME):
            raise ImportError(
                f"aten::{_OP_NAME} is not available in torch {torch.__version__}"
            )
        schema_args = {
            a.name for a in getattr(torch.ops.aten, _OP_NAME).default._schema.arguments
        }
        missing = {"block_table", "seqused_k"} - schema_args
        if missing:
            # Older wheels carry the op but not its paged arguments. Registering
            # against that schema would silently never see a page table, which is
            # the failure mode this check exists to make loud.
            raise ImportError(
                f"aten::{_OP_NAME} in torch {torch.__version__} lacks "
                f"{sorted(missing)}; paged routing needs torch >= 2.12"
            )

        op = getattr(torch.ops.aten, _OP_NAME)
        # The dispatch key we register on. Falling back cannot simply re-call the
        # op: that re-enters THIS kernel and recurses until the stack dies
        # (verified -- it is not theoretical). Excluding our own key before
        # redispatching sends the call to the kernel we displaced.
        key = torch._C.DispatchKey.CUDA
        key_set = torch._C.DispatchKeySet(key)

        def _impl(
            query,
            key_t,
            value,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            return_debug_mask,
            scale=None,
            window_size_left=None,
            window_size_right=None,
            seqused_k=None,
            alibi_slopes=None,
            block_table=None,
            num_splits=None,
        ):
            def _native():
                with torch._C._ExcludeDispatchKeyGuard(key_set):
                    return op(
                        query,
                        key_t,
                        value,
                        cum_seq_q,
                        cum_seq_k,
                        max_q,
                        max_k,
                        dropout_p,
                        is_causal,
                        return_debug_mask,
                        scale=scale,
                        window_size_left=window_size_left,
                        window_size_right=window_size_right,
                        seqused_k=seqused_k,
                        alibi_slopes=alibi_slopes,
                        block_table=block_table,
                        num_splits=num_splits,
                    )

            census_key = "aten"
            try:
                census_key = (
                    f"aten:Tq={int(query.shape[0])},Hq={int(query.shape[1])},"
                    f"D={int(query.shape[-1])},dtype={ov._tok(query.dtype)}"
                )
            except Exception:  # noqa: BLE001 -- census must never break dispatch
                pass

            # dropout and the debug mask have no expression in the graph, and
            # alibi_slopes is a scoring term the tiled engine declines outright.
            if dropout_p:
                ov.note_native(census_key, "dropout_p != 0")
                return _native()
            if return_debug_mask:
                ov.note_native(census_key, "return_debug_mask requested")
                return _native()
            if alibi_slopes is not None:
                ov.note_native(census_key, "alibi_slopes not served")
                return _native()

            left = -1 if window_size_left is None else int(window_size_left)
            right = (
                0
                if is_causal
                else (-1 if window_size_right is None else int(window_size_right))
            )

            out = ov.run_paged(
                query,
                key_t,
                value,
                cum_seq_q,
                max_k,
                scale=scale,
                window=(left, right),
                seqused_k=seqused_k,
                block_table=block_table,
                num_splits=num_splits,
                census_key=census_key,
            )
            if out is None:
                return _native()

            # The op returns (output, softmax_logsumexp, rng_state, unused,
            # debug_attn_mask). The graph produces only the output, so the rest
            # are returned as correctly-shaped empties -- the same thing the
            # native kernel yields when dropout is 0 and no debug mask is asked
            # for. logsumexp is [H, total_q]; a caller that wants a real one is
            # already declined above via return_aux/backward paths.
            empty = torch.empty(0, device=query.device, dtype=query.dtype)
            lse = torch.empty(
                (int(query.shape[1]), int(query.shape[0])),
                device=query.device,
                dtype=torch.float32,
            )
            rng = torch.empty(2, device=query.device, dtype=torch.uint64)
            return out, lse, rng, empty, empty

        self._lib = torch.library.Library("aten", "IMPL")
        self._lib.impl(_OP_NAME, _impl, "CUDA")
        self._installed = True
        log.info("hipdnn_torch: registered aten::%s (CUDA)", _OP_NAME)

    def uninstall(self) -> None:
        """Drop the registration. ``torch.library.Library`` releases its
        registrations when the object is destroyed, so releasing our reference is
        the supported way to undo this."""
        if not self._installed:
            return
        self._lib = None
        self._installed = False

    # -- reporting: delegate so both routes share one tally ------------------
    def census(self) -> dict:
        return self._override.census()

    def fallback_reasons(self) -> dict:
        return self._override.fallback_reasons()

    def format_report(self) -> str:
        return self._override.format_report()

    def reset(self) -> None:
        self._override.reset()
