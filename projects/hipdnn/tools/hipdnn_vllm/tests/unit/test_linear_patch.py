# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The unquantized-linear patch against the pinned vLLM.

Each case runs register() in a fresh interpreter, because the patch rebinds
module globals of vLLM for the life of the process. Cases skip when the pinned
vLLM is not importable or the platform is not ROCm. None of them touches a
device: register() must not load _C.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.vllm

TOOL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRELUDE = f"""
import sys
sys.path.insert(0, {TOOL_DIR!r})
try:
    from vllm.platforms import current_platform
except Exception as ex:
    print("SKIP: vllm not importable:", ex)
    sys.exit(0)
if not current_platform.is_rocm():
    print("SKIP: not ROCm")
    sys.exit(0)
import hipdnn_vllm
SITES = {{
    "linear": "vllm.model_executor.layers.linear",
    "lm_head": "vllm.model_executor.layers.vocab_parallel_embedding",
    "moe_router_gate": "vllm.model_executor.layers.fused_moe.runner.moe_runner",
}}
"""


def _run(body: str, ops: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HIPDNN_VLLM_")}
    if ops is not None:
        env["HIPDNN_VLLM_OPS"] = ops
    code = PRELUDE + textwrap.dedent(body)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.stdout.startswith("SKIP"):
        pytest.skip(r.stdout.strip())
    assert r.returncode == 0, out
    assert "CASE_OK" in r.stdout, out
    return out


@pytest.mark.parametrize("ops", [None, "attention,rms_norm,fused_add_rms_norm,layer_norm,conv"])
def test_linear_off_installs_no_patch(ops):
    _run(
        """
        hipdnn_vllm.register()
        assert "hipdnn_vllm.ops.linear" not in sys.modules
        import importlib
        utils = importlib.import_module("vllm.model_executor.layers.utils")
        for site, name in SITES.items():
            mod = importlib.import_module(name)
            assert mod.dispatch_unquantized_gemm is utils.dispatch_unquantized_gemm, site
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        assert UnquantizedLinearMethod()._gemm_impl is utils.rocm_unquantized_gemm
        import torch
        assert not hasattr(torch.ops.hipdnn_vllm, "unquantized_gemm")
        import json
        import vllm.envs
        factor = json.loads(vllm.envs.compile_factors()[hipdnn_vllm.COMPILE_FACTOR_NAME])
        assert not {"linear", "linear_all"} & set(factor["ops"]), factor
        print("CASE_OK")
        """,
        ops,
    )


def test_linear_patches_linear_site_only():
    _run(
        """
        hipdnn_vllm.register()
        from hipdnn_vllm.ops import linear as lin
        assert lin.patch_state() == {"linear": True, "lm_head": False, "moe_router_gate": False}, lin.patch_state()
        import importlib
        utils = importlib.import_module("vllm.model_executor.layers.utils")
        for site in ("lm_head", "moe_router_gate"):
            mod = importlib.import_module(SITES[site])
            assert mod.dispatch_unquantized_gemm is utils.dispatch_unquantized_gemm, site
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        assert UnquantizedLinearMethod()._gemm_impl is lin.hipdnn_unquantized_gemm
        # register() did no device work: _C (and the hipDNN runtime) not loaded.
        assert "hipdnn_vllm._C" not in sys.modules
        import torch
        assert hasattr(torch.ops.hipdnn_vllm, "unquantized_gemm")
        import json
        import vllm.envs
        factor = json.loads(vllm.envs.compile_factors()[hipdnn_vllm.COMPILE_FACTOR_NAME])
        assert "linear" in factor["ops"], factor
        print("CASE_OK")
        """,
        "linear",
    )


def test_linear_all_patches_every_site():
    _run(
        """
        hipdnn_vllm.register()
        hipdnn_vllm._registered = False
        hipdnn_vllm.register()  # idempotent
        from hipdnn_vllm.ops import linear as lin
        assert lin.patch_state() == {"linear": True, "lm_head": True, "moe_router_gate": True}, lin.patch_state()
        from vllm.model_executor.layers import vocab_parallel_embedding as vpe
        assert vpe.dispatch_unquantized_gemm() is lin.hipdnn_unquantized_gemm
        lin.uninstall_patch()
        assert lin.patch_state() == {"linear": False, "lm_head": False, "moe_router_gate": False}
        print("CASE_OK")
        """,
        "linear_all",
    )


_MISMATCHES = {
    "dispatch_signature": """
        def bad(linear_backend="auto", extra=None):
            return utils.rocm_unquantized_gemm
        utils.dispatch_unquantized_gemm = bad
        linear_mod.dispatch_unquantized_gemm = bad
    """,
    "gemm_signature": """
        orig = utils.rocm_unquantized_gemm
        def rocm_unquantized_gemm(layer, x, weight):
            return orig(layer, x, weight)
        utils.rocm_unquantized_gemm = rocm_unquantized_gemm
    """,
    "site_replaced": """
        linear_mod.dispatch_unquantized_gemm = lambda linear_backend="auto": None
    """,
    "init_no_longer_dispatches": """
        def __init__(self):
            self._gemm_impl = None
        linear_mod.UnquantizedLinearMethod.__init__ = __init__
    """,
}


@pytest.mark.parametrize("case", sorted(_MISMATCHES))
def test_symbol_mismatch_fails_loudly(case):
    _run(
        textwrap.dedent(
            """
            import importlib
            utils = importlib.import_module("vllm.model_executor.layers.utils")
            linear_mod = importlib.import_module("vllm.model_executor.layers.linear")
            """
        )
        + textwrap.dedent(_MISMATCHES[case])
        + """
before = {s: importlib.import_module(n).dispatch_unquantized_gemm for s, n in SITES.items()}
from hipdnn_vllm.ops.linear import LinearPatchError
try:
    hipdnn_vllm.register()
except LinearPatchError as ex:
    print("RAISED:", ex)
else:
    raise SystemExit("register() did not raise")
after = {s: importlib.import_module(n).dispatch_unquantized_gemm for s, n in SITES.items()}
assert before == after, "a failed install modified vLLM"
print("CASE_OK")
""",
        "linear_all",
    )
