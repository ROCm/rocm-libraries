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

#: The op paged varlen ACTUALLY arrives on. Measured on device, not read:
#: ``varlen_attn`` delegates immediately to this custom op, so patching the
#: Python wrapper is bypassed by every caller -- including the wrapper's own
#: body. Verified in Spur job 469, where the monkeypatch provably installed and
#: the call still reached the native kernel with no frame of ours in the
#: traceback. See Results/paged-real-seam-is-torch-attn-varlen-op.md.
_OP_NAMESPACE = "torch_attn"
_OP_NAME = "_varlen_attn"

#: The op is registered when ``torch.nn.attention.varlen`` is imported, not at
#: torch import. Touching that module first is what makes the namespace exist.
_OP_DEFINING_MODULE = "torch.nn.attention.varlen"

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
            # importlib, NOT `from . import bootstrap`. Inside a function that
            # runs after the package is initialised, `from . import bootstrap`
            # resolves the ATTRIBUTE on the package -- and __init__.py has
            # rebound that name to the bootstrap() FUNCTION
            # (`from .bootstrap import bootstrap`). The same statement works at
            # module top-level, before __init__ finishes, which is why base.py
            # gets away with it and this did not:
            #   AttributeError: 'function' object has no attribute 'bootstrap'
            import importlib

            _bootstrap = importlib.import_module("hipdnn_torch.bootstrap")
            ov.state = _bootstrap.bootstrap()
        torch = ov.state.torch

        import importlib
        importlib.import_module(_OP_DEFINING_MODULE)  # registers the custom op
        ns = getattr(torch.ops, _OP_NAMESPACE)
        if not hasattr(ns, _OP_NAME):
            raise ImportError(
                f"{_OP_NAMESPACE}::{_OP_NAME} is not available in torch {torch.__version__}"
            )
        schema_args = {
            a.name for a in getattr(ns, _OP_NAME).default._schema.arguments
        }
        missing = {"block_table", "seqused_k"} - schema_args
        if missing:
            # Older wheels carry the op but not its paged arguments. Registering
            # against that schema would silently never see a page table, which is
            # the failure mode this check exists to make loud.
            raise ImportError(
                f"{_OP_NAMESPACE}::{_OP_NAME} in torch {torch.__version__} lacks "
                f"{sorted(missing)}; paged routing needs torch >= 2.12"
            )

        op = getattr(ns, _OP_NAME)
        # The dispatch key we register on. Falling back cannot simply re-call the
        # op: that re-enters THIS kernel and recurses until the stack dies
        # (verified -- it is not theoretical). Excluding our own key before
        # redispatching sends the call to the kernel we displaced.
        key = torch._C.DispatchKey.CUDA
        key_set = torch._C.DispatchKeySet(key)

        def _impl(
            query,
            key,
            value,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            is_causal=False,
            scale=None,
            window_size=None,
            seqused_k=None,
            block_table=None,
            num_splits=None,
        ):
            """hipDNN's implementation of ``torch_attn::_varlen_attn``.

            Signature mirrors the op EXACTLY (13 args, 3 returns). It is not the
            same shape as ``aten::_flash_attention_forward`` -- no dropout_p, no
            return_debug_mask, no alibi_slopes, and the window arrives as one
            ``window_size`` pair rather than two scalars."""

            def _native():
                with torch._C._ExcludeDispatchKeyGuard(key_set):
                    return op(
                        query, key, value, cu_seq_q, cu_seq_k, max_q, max_k,
                        is_causal, scale, window_size, seqused_k, block_table,
                        num_splits,
                    )

            census_key = "varlen_op"
            try:
                census_key = (
                    f"op:Tq={int(query.shape[0])},Hq={int(query.shape[1])},"
                    f"D={int(query.shape[-1])},dtype={ov._tok(query.dtype)}"
                )
            except Exception:  # noqa: BLE001 -- census must never break dispatch
                pass

            # torch's (left, right) window; is_causal is the right edge at 0.
            left, right = (-1, -1)
            if window_size is not None:
                try:
                    left, right = int(window_size[0]), int(window_size[1])
                except Exception:  # noqa: BLE001 -- malformed window -> native
                    ov.note_native(census_key, "window_size not a (left, right) pair")
                    return _native()
            if is_causal:
                right = 0

            out = ov.run_paged(
                query, key, value, cu_seq_q, max_k, scale=scale,
                window=(left, right), seqused_k=seqused_k,
                block_table=block_table, num_splits=num_splits,
                census_key=census_key,
            )
            if out is None:
                return _native()

            # The op returns (output, softmax_lse, rng_state). The graph produces
            # only the output; lse is [H, total_q] and rng_state is the 2-element
            # zero tensor the native path hardcodes because dropout is always 0.
            lse = torch.empty(
                (int(query.shape[1]), int(query.shape[0])),
                device=query.device, dtype=torch.float32,
            )
            rng = torch.zeros(2, device=query.device, dtype=torch.uint64)
            return out, lse, rng

        self._lib = torch.library.Library(_OP_NAMESPACE, "IMPL")
        self._lib.impl(_OP_NAME, _impl, "CUDA")
        self._installed = True
        log.info("hipdnn_torch: registered %s::%s (CUDA)", _OP_NAMESPACE, _OP_NAME)

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
