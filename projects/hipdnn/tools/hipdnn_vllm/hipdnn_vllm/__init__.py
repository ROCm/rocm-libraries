# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""hipDNN drop-in backend for vLLM.

vLLM calls :func:`register` through the ``vllm.general_plugins`` entry point in
every process it starts (API server, EngineCore, each worker). Registration
does no HIP or torch device work; ``_C`` and the hipDNN runtime load lazily on
the first op call in a worker (see :mod:`hipdnn_vllm._bootstrap`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading

__all__ = ["BACKEND_NAME", "BACKEND_CLASS_PATH", "COMPILE_FACTOR_NAME", "register"]

BACKEND_NAME = "HIPDNN"
BACKEND_CLASS_PATH = "hipdnn_vllm.attention.backend.HipdnnAttentionBackend"
# Name of the entry register() adds to vllm.envs.environment_variables.
COMPILE_FACTOR_NAME = "HIPDNN_VLLM_COMPILE_FACTOR"

logger = logging.getLogger("hipdnn_vllm")

_register_lock = threading.Lock()
_registered = False


def _package_fingerprint() -> str:
    """Identity of the installed plugin files (Python sources, ``_C``, bundled
    providers): the path, size and mtime of each, hashed."""
    root = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            st = os.stat(path)
            digest.update(f"{os.path.relpath(path, root)}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:16]


def _register_compile_factor(config) -> None:
    """vLLM keys its torch.compile and AOT-compile caches on
    ``vllm.envs.compile_factors()`` (every getter in
    ``environment_variables``) plus the vLLM config, and nothing there reflects
    this plugin. A run with the plugin would otherwise load graphs compiled
    without it, and the other way round. One entry carries the plugin's files
    and every setting that changes what a traced graph dispatches to."""
    from vllm import envs

    value = json.dumps(
        {
            "files": _package_fingerprint(),
            "ops": sorted(config.ops),
            "strict": config.strict,
            "attn_debug_route": config.attn_debug_route,
            "plugin_mode": config.plugin_mode,
            "provider_dirs": list(config.provider_dirs),
        },
        sort_keys=True,
    )
    envs.environment_variables[COMPILE_FACTOR_NAME] = lambda: value


def _is_rocm() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_rocm()


def _attention_backend_member():
    """Unpickling target of the ``HIPDNN`` ``AttentionBackendEnum`` member.

    A spawned EngineCore unpickles the vLLM config before it loads the general
    plugins, so the stock ``AttentionBackendEnum(value)`` lookup would fail on
    the dynamic member. Pickles of the member name this function instead, which
    registers the backend in the unpickling process first. Imports only the
    vLLM registry (no torch device work)."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    _register_attention_backend()
    return AttentionBackendEnum[BACKEND_NAME]


def _reduce_attention_backend_member(protocol):
    return _attention_backend_member, ()


def _register_attention_backend() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    # AttentionBackendEnum.register raises if the name exists, so a second call
    # in the same process (or a re-imported registry) must not re-register.
    if BACKEND_NAME in AttentionBackendEnum.__members__:
        member = AttentionBackendEnum[BACKEND_NAME]
        if not (member.is_overridden() and member.get_path() == BACKEND_CLASS_PATH):
            register_backend(member, BACKEND_CLASS_PATH)
    else:
        register_backend(BACKEND_NAME, BACKEND_CLASS_PATH)
        member = AttentionBackendEnum[BACKEND_NAME]
    # Pickle looks __reduce_ex__ up on the instance first; Enum's own reduce
    # would pickle AttentionBackendEnum(<class path>), which only resolves
    # after this function has run in the unpickling process.
    member.__reduce_ex__ = _reduce_attention_backend_member


def _register_ir_providers() -> None:
    """Register the ``hipdnn`` vLLM IR providers (rms_norm, fused_add_rms_norm).

    Done in every process and even when disabled or off ROCm (then with
    ``supported=False``): vLLM asserts that every provider named in
    ``--ir-op-priority`` is registered, and hashes their sources into the
    compile cache key in the engine process."""
    try:
        import vllm.ir  # noqa: F401
    except ImportError:
        logger.debug("hipdnn_vllm: this vLLM has no vllm.ir; IR providers not registered")
        return
    from .ops import norm_ir  # noqa: F401  (registers on import)


def register() -> None:
    """vLLM general-plugin entry point. Idempotent and re-entrant."""
    global _registered
    with _register_lock:
        if _registered:
            return
        from . import stats
        from .config import load_config

        config = load_config()
        _register_ir_providers()
        if config.disable:
            logger.info("hipdnn_vllm: disabled by HIPDNN_VLLM_DISABLE")
            _registered = True
            return
        if not _is_rocm():
            logger.info("hipdnn_vllm: not a ROCm platform; nothing registered")
            _registered = True
            return
        stats.configure(config)
        _register_compile_factor(config)
        # Before any device work: comgr fixes its cache settings at HIP's first
        # kernel launch (see hipdnn_vllm.comgr_cache).
        from . import comgr_cache

        comgr_cache.configure()
        # The attention backend is registered even when HIPDNN_VLLM_OPS excludes
        # "attention": selecting it with --attention-backend is the user's act,
        # and the impl then routes every call to fallback with DISABLED.
        _register_attention_backend()
        # The linear patch goes in before any model is built
        # (UnquantizedLinearMethod binds its gemm in __init__). Off unless
        # HIPDNN_VLLM_OPS names linear/linear_all; a vLLM symbol mismatch raises
        # LinearPatchError.
        if config.op_enabled("linear") or config.op_enabled("linear_all"):
            from .ops.linear import install_patch

            install_patch(config)
        if config.op_enabled("encoder_attention"):
            from .ops.encoder_attention import register_oot as _register_encoder_attention

            _register_encoder_attention()
        from .layers import register_layers

        layers = register_layers(config)
        if layers:
            logger.debug("hipdnn_vllm: registered OOT layers %s", layers)
        _registered = True
        logger.debug("hipdnn_vllm: registered attention backend %s", BACKEND_NAME)
