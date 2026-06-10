#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate HSTU jagged-fwd kernel + dispatch header for Python ctypes JIT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[1]

_DTYPE_MAP = {
    "fp16": ("ck_tile::fp16_t", "half"),
    "bf16": ("ck_tile::bf16_t", "bfloat16"),
}


def _kernel_name(cfg: Dict[str, Any]) -> str:
    sig = cfg["signature"]
    alg = cfg["algorithm"]
    name = (
        f"hstu_{sig['data_type']}_causal{int(sig['use_causal'])}"
        f"_maxk{sig['max_k']}_mtile{alg['mtile']}"
        f"_splitkv{int(sig['use_splitkv'])}"
    )
    # Distinguishing tile tokens for overridden (nonzero) block-tile dims only,
    # so base-tile kernels keep the legacy name (and byte-identical cpp).
    for tok, key in (("km0", "km0"), ("n0", "kn0"), ("n0s", "kn0sub"),
                     ("n1", "kn1"), ("k1", "kk1"), ("wk", "warp_k")):
        val = int(alg.get(key, 0))
        if val:
            name += f"_{tok}{val}"
    return name


def _render_kernel_cpp(name: str, cfg: Dict[str, Any]) -> str:
    sig = cfg["signature"]
    alg = cfg["algorithm"]
    ctype, dtype_file = _DTYPE_MAP[sig["data_type"]]
    causal = "true" if sig["use_causal"] else "false"
    max_k = sig["max_k"]
    mtile = alg["mtile"]
    use_splitkv = sig["use_splitkv"]
    km0 = int(alg.get("km0", 0))
    kn0 = int(alg.get("kn0", 0))
    kn0sub = int(alg.get("kn0sub", 0))
    kn1 = int(alg.get("kn1", 0))
    kk1 = int(alg.get("kk1", 0))
    warp_k = int(alg.get("warp_k", 0))
    # Any explicit override (tile dim or warp_k) needs the full positional
    # template form; a fully-base config keeps the legacy 7-arg form so its cpp
    # stays byte-identical.
    override_active = any((km0, kn0, kn0sub, kn1, kk1, warp_k))
    # WarpK is positional, so once we emit the full form we must give it a value;
    # 0 == "dispatch default" (16).
    warpk_arg = warp_k if warp_k else 16

    if use_splitkv:
        dispatch = "jagged_forward_splitkv_causal_softmax_bias_dropout_dispatch"
    else:
        dispatch = "jagged_forward_causal_softmax_bias_dropout_dispatch"

    include_splitkv = ""
    if use_splitkv:
        include_splitkv = '#include "hstu_attention_jagged_forward_splitkv_dispatch.hpp"\n'

    if override_active and not use_splitkv:
        # Thread the block-tile + warp_k overrides through the dispatch template.
        # All other variant params keep their template defaults. KM0/KN0/kN0Sub/
        # KN1/KK1 == 0 means "use base dim"; WarpK selects 16x16x{16,32}.
        dispatch_args = (
            f"""        {ctype},
        {causal},
        false,
        false,
        false,
        {max_k},
        {mtile},
        /*kUseAsyncPipeline=*/false,
        /*WarpK=*/{warpk_arg},
        /*kUseAgpr=*/false,
        /*kUsePingPong=*/false,
        /*kUseSchedGroup=*/false,
        /*kUseTrLoad=*/false,
        /*Occupancy=*/-1,
        /*KM0=*/{km0},
        /*KN0=*/{kn0},
        /*KN0Sub=*/{kn0sub},
        /*KN1=*/{kn1},
        /*KK1=*/{kk1}"""
        )
    else:
        dispatch_args = (
            f"""        {ctype},
        {causal},
        false,
        false,
        false,
        {max_k},
        {mtile}"""
        )

    return f"""// SPDX-License-Identifier: MIT
// Auto-generated HSTU kernel: {name}

#include <ck_tile/core/numeric/{dtype_file}.hpp>
#include "hstu_attention_jagged_forward_dispatch.hpp"
{include_splitkv}
#include "hstu_attention_params.hpp"

namespace hstu_jit {{

void run_jagged_fwd(HstuAttentionNoGroupFwdParams& param, hipStream_t stream)
{{
    {dispatch}<
{dispatch_args}>::Run(param, stream);
}}

}} // namespace hstu_jit
"""


def _render_dispatch_header(name: str, cfg: Dict[str, Any]) -> str:
    sig = cfg["signature"]
    alg = cfg["algorithm"]
    return f"""// Auto-generated HSTU dispatch header for Python ctypes library
#pragma once

#include "hstu_attention_params.hpp"
#include "ck_tile/dispatcher/hstu_registry.hpp"

namespace hstu_jit {{
void run_jagged_fwd(HstuAttentionNoGroupFwdParams& param, hipStream_t stream);
}}

#ifndef HSTU_RUN_JAGGED_FWD
#define HSTU_RUN_JAGGED_FWD(param, stream) hstu_jit::run_jagged_fwd(param, stream)
#endif

namespace generated {{

inline void register_hstu_python_kernels(
    ck_tile::dispatcher::HstuRegistry& registry, const std::string& /*arch*/)
{{
    ck_tile::dispatcher::HstuKernelKey key;
    key.name = "{name}";
    key.data_type = "{sig['data_type']}";
    key.use_causal = {str(sig['use_causal']).lower()};
    key.use_softmax = false;
    key.has_bias = false;
    key.max_k = {sig['max_k']};
    key.mtile = {alg['mtile']};
    key.use_splitkv = {str(sig['use_splitkv']).lower()};

    registry.register_kernel(key, [](void* stream_ptr) -> float {{
        (void)stream_ptr;
        return 0.f;
    }});
}}

}} // namespace generated

#ifndef REGISTER_GENERATED_KERNELS
#define REGISTER_GENERATED_KERNELS(registry, arch) \\
    ::generated::register_hstu_python_kernels(registry, arch)
#endif

static const int HSTU_KERNEL_COUNT = 1;
static const char* HSTU_KERNEL_NAMES[] = {{"{name}"}};
"""


def _canonicalize(raw: Dict[str, Any], arch: str) -> Dict[str, Any]:
    sig = raw.get("signature", raw)
    alg = raw.get("algorithm", {})
    return {
        "arch": raw.get("arch", arch),
        "signature": {
            "family": "jagged_fwd",
            "data_type": sig.get("data_type", "bf16"),
            "use_causal": bool(sig.get("use_causal", True)),
            "use_softmax": False,
            "has_bias": False,
            "max_k": int(sig.get("max_k", 128)),
            "use_splitkv": bool(sig.get("use_splitkv", False)),
        },
        "algorithm": {
            "mtile": int(alg.get("mtile", sig.get("mtile", 128))),
            # Block-tile overrides (0 == use base dim). Carried through so the
            # generated cpp can pin sequence<kM0,kN0,kN0Sub,kN1,kK1,...>.
            "km0": int(alg.get("km0", 0)),
            "kn0": int(alg.get("kn0", 0)),
            "kn0sub": int(alg.get("kn0sub", 0)),
            "kn1": int(alg.get("kn1", 0)),
            "kk1": int(alg.get("kk1", 0)),
            "warp_k": int(alg.get("warp_k", 0)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HSTU JIT kernel for ctypes")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-target", default="gfx950")
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    raw = json.loads(args.config_json)
    cfg = _canonicalize(raw, args.gpu_target)
    name = _kernel_name(cfg)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cpp_path = out / f"{name}.cpp"
    cpp_path.write_text(_render_kernel_cpp(name, cfg))

    dispatch_path = out / "hstu_python_dispatch.hpp"
    dispatch_path.write_text(_render_dispatch_header(name, cfg))

    print(f"  Generated {cpp_path.name}, {dispatch_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
