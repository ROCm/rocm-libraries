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
    return (
        f"hstu_{sig['data_type']}_causal{int(sig['use_causal'])}"
        f"_maxk{sig['max_k']}_mtile{alg['mtile']}"
        f"_splitkv{int(sig['use_splitkv'])}"
    )


def _render_kernel_cpp(name: str, cfg: Dict[str, Any]) -> str:
    sig = cfg["signature"]
    alg = cfg["algorithm"]
    ctype, dtype_file = _DTYPE_MAP[sig["data_type"]]
    causal = "true" if sig["use_causal"] else "false"
    max_k = sig["max_k"]
    mtile = alg["mtile"]
    use_splitkv = sig["use_splitkv"]

    if use_splitkv:
        dispatch = "jagged_forward_splitkv_causal_softmax_bias_dropout_dispatch"
    else:
        dispatch = "jagged_forward_causal_softmax_bias_dropout_dispatch"

    include_splitkv = ""
    if use_splitkv:
        include_splitkv = '#include "hstu_attention_jagged_forward_splitkv_dispatch.hpp"\n'

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
        {ctype},
        {causal},
        false,
        false,
        false,
        {max_k},
        {mtile}>::Run(param, stream);
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
