# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""heuristic() and matmul() against NumPy, and the quiet failures matmul() now refuses.

hipBLASLt is column-major. A column-major matrix S (r x c) is stored here as the C-contiguous
host array S.T, with MatrixLayout(type, r, c, ld=r).

Tolerances are a forward error bound, not a fudge factor: for each element of D, a K-term dot
product in the accumulation type is within K * eps(acc) * (|A| @ |B|) of the exact result, and
storing it adds one rounding, eps(out) * |D|. fp16 and bf16 products are exact in fp32.
"""

import ml_dtypes
import numpy as np
import pytest

import hipblaslt

c = hipblaslt._core
pytestmark = pytest.mark.gpu

_DT = {
    np.dtype(np.float32): c.DataType.R_32F,
    np.dtype(np.float16): c.DataType.R_16F,
    np.dtype(ml_dtypes.bfloat16): c.DataType.R_16BF,
    np.dtype(np.float64): c.DataType.R_64F,
}


def _eps(dtype) -> float:
    dtype = np.dtype(dtype)
    return float(
        ml_dtypes.finfo(dtype).eps
        if dtype == ml_dtypes.bfloat16
        else np.finfo(dtype).eps
    )


def _assert_within_bound(out, a, b, ref, out_dtype, acc_dtype, k):
    bound = (
        k * _eps(acc_dtype) * (np.abs(a) @ np.abs(b))
        + _eps(out_dtype) * np.abs(ref)
        + 1e-300
    )
    err = np.abs(out.astype(np.float64) - ref)
    worst = float(np.max(err / bound))
    assert (
        worst <= 1.0
    ), f"error exceeds the bound by {worst:.2f}x (max error {float(err.max()):.3g})"


def _stored(x):
    return np.ascontiguousarray(x.T)


class Problem:
    """D = alpha * op(A) @ op(B) + beta * C, set up for hipBLASLt."""

    def __init__(
        self, dtype, m, n, k, op_a="N", op_b="N", seed=0, compute=None, scale=None
    ):
        rng = np.random.default_rng(seed)
        self.dtype = np.dtype(dtype)
        self.t = _DT[self.dtype]
        self.m, self.n, self.k = m, n, k
        self.a = rng.uniform(-1, 1, (m, k)).astype(self.dtype)
        self.b = rng.uniform(-1, 1, (k, n)).astype(self.dtype)
        self.cmat = rng.uniform(-1, 1, (m, n)).astype(self.dtype)
        sa = self.a if op_a == "N" else self.a.T
        sb = self.b if op_b == "N" else self.b.T
        self.da, self.db = hipblaslt.from_numpy(_stored(sa)), hipblaslt.from_numpy(
            _stored(sb)
        )
        self.dc = hipblaslt.from_numpy(_stored(self.cmat))
        self.dd = c.DeviceArray.empty([n, m], self.t)
        self.la = c.MatrixLayout(self.t, *sa.shape, sa.shape[0])
        self.lb = c.MatrixLayout(self.t, *sb.shape, sb.shape[0])
        self.lc = c.MatrixLayout(self.t, m, n, m)
        self.ld = c.MatrixLayout(self.t, m, n, m)
        if self.dtype == np.float64:
            compute, scale = (
                compute or c.ComputeType.COMPUTE_64F,
                scale or c.DataType.R_64F,
            )
        self.acc = np.float64 if self.dtype == np.float64 else np.float32
        self.desc = c.MatmulDesc(
            compute or c.ComputeType.COMPUTE_32F, scale or c.DataType.R_32F
        )
        ops = {"N": c.Operation.N, "T": c.Operation.T}
        self.desc.set_attribute(c.MatmulDescAttr.TRANSA, ops[op_a])
        self.desc.set_attribute(c.MatmulDescAttr.TRANSB, ops[op_b])
        self.pref = c.Preference()
        self.pref.set_attribute(c.PreferenceAttr.MAX_WORKSPACE_BYTES, 64 * 1024 * 1024)

    def heuristic(self, handle, max_results=16):
        args = (self.desc, self.la, self.lb, self.lc, self.ld, self.pref, max_results)
        return c.heuristic(handle, *args)

    def run(self, handle, result, alpha=1.0, beta=0.0, workspace="auto"):
        if workspace == "auto":
            n = result.workspace_size
            workspace = c.DeviceArray.empty([n], c.DataType.R_8I) if n else None
        c.matmul(
            handle, self.desc, alpha, self.da, self.la, self.db, self.lb, beta,
            self.dc, self.lc, self.dd, self.ld, result, workspace,
        )  # fmt: skip
        return self.dd.to_numpy().T

    def reference(self, alpha=1.0, beta=0.0):
        a, b = self.a.astype(np.float64), self.b.astype(np.float64)
        return alpha * (a @ b) + beta * self.cmat.astype(np.float64)

    def check(self, out, alpha=1.0, beta=0.0):
        a = abs(alpha) * np.abs(self.a.astype(np.float64))
        ref = self.reference(alpha, beta)
        # beta * C adds one more term to each sum.
        _assert_within_bound(
            out, a, self.b.astype(np.float64), ref, self.dtype, self.acc, self.k + 1
        )


@pytest.fixture
def handle():
    with c.Handle() as h:
        yield h


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float16, ml_dtypes.bfloat16], ids=["f32", "f16", "bf16"]
)
@pytest.mark.parametrize("ops", ["NN", "NT", "TN", "TT"])
def test_numbers_match_numpy(handle, dtype, ops):
    p = Problem(dtype, 48, 40, 64, op_a=ops[0], op_b=ops[1], seed=1)
    results = p.heuristic(handle)
    assert results, "the heuristic returned no solution"
    p.check(p.run(handle, results[0]))


def test_heuristic_ranks_are_positions(handle):
    results = Problem(np.float32, 64, 64, 64).heuristic(handle, max_results=8)
    assert [r.algo.heuristic_rank for r in results] == list(range(len(results)))


def test_beta_and_c(handle):
    p = Problem(np.float32, 32, 24, 40, seed=2)
    out = p.run(handle, p.heuristic(handle)[0], alpha=1.5, beta=0.5)
    p.check(out, alpha=1.5, beta=0.5)


def test_compute_64f_takes_double_alpha_and_beta(handle):
    # 1/3 and 1/7 aren't representable in fp32. Passed as float (the prototype's bug), the fp64
    # GEMM reads 8 bytes from a 4-byte value and returns garbage.
    p = Problem(np.float64, 64, 48, 32, seed=3)
    alpha, beta = 1 / 3, 1 / 7
    out = p.run(handle, p.heuristic(handle)[0], alpha=alpha, beta=beta)
    p.check(out, alpha=alpha, beta=beta)


@pytest.mark.parametrize(
    "dtype,alpha",
    [(np.float32, 2.0), (np.float32, 2), (np.float64, 1 / 3), (np.float64, 3)],
    ids=["fp32-float", "fp32-int", "fp64-float", "fp64-int"],
)
def test_alpha_as_a_python_number(handle, dtype, alpha):
    # An fp64 alpha of 2.0 into an fp32 scale gives D = 0 if written as a double; a Python int
    # read as a pointer segfaults. Both go through the scale type here.
    p = Problem(dtype, 16, 16, 16, seed=4)
    p.check(p.run(handle, p.heuristic(handle)[0], alpha=alpha), alpha=alpha)


def test_alpha_must_be_a_number(handle):
    p = Problem(np.float32, 8, 8, 8)
    with pytest.raises(TypeError, match="alpha must be a number"):
        p.run(handle, p.heuristic(handle)[0], alpha="1")
    with pytest.raises(TypeError, match="beta must be a number"):
        p.run(handle, p.heuristic(handle)[0], beta=True)


def _result_needing_workspace(handle):
    # fp16 problems with a long K choose solutions that split K and need a workspace.
    for m, n, k in [(16, 16, 65536), (1024, 1024, 1024), (256, 256, 65536)]:
        p = Problem(np.float16, m, n, k, seed=5)
        for r in p.heuristic(handle, max_results=64):
            if r.workspace_size:
                return p, r
    pytest.fail("no solution needing a workspace was found to test the size check with")


def test_workspace_too_small_is_refused(handle):
    p, r = _result_needing_workspace(handle)
    small = c.DeviceArray.empty([r.workspace_size - 1], c.DataType.R_8I)
    with pytest.raises(
        ValueError, match=f"needs a workspace of {r.workspace_size} bytes"
    ):
        p.run(handle, r, workspace=small)
    with pytest.raises(ValueError, match="none was given"):
        p.run(handle, r, workspace=None)


def test_a_bigger_workspace_is_fine(handle):
    p, r = _result_needing_workspace(handle)
    big = c.DeviceArray.empty([r.workspace_size * 2 + 4096], c.DataType.R_8I)
    p.check(p.run(handle, r, workspace=big))


def test_every_heuristic_candidate_is_right(handle):
    # The debugging story in one test: run every solution the heuristic offers.
    p = Problem(np.float32, 64, 48, 32, seed=6)
    results = p.heuristic(handle, max_results=64)
    assert results
    for r in results:
        p.check(p.run(handle, r))


def test_relu_epilogue(handle):
    p = Problem(np.float32, 32, 32, 32, seed=7)
    p.desc.set_attribute(c.MatmulDescAttr.EPILOGUE, c.Epilogue.RELU)
    out = p.run(handle, p.heuristic(handle)[0])
    ref = np.maximum(p.reference(), 0)
    _assert_within_bound(
        out,
        np.abs(p.a.astype(np.float64)),
        p.b.astype(np.float64),
        ref,
        np.float32,
        np.float32,
        p.k,
    )


def test_bias_epilogue(handle):
    p = Problem(np.float32, 32, 24, 32, seed=8)
    bias = np.random.default_rng(8).uniform(-1, 1, p.m).astype(np.float32)
    dbias = hipblaslt.from_numpy(bias)
    p.desc.set_attribute(c.MatmulDescAttr.EPILOGUE, c.Epilogue.BIAS)
    p.desc.set_attribute(c.MatmulDescAttr.BIAS_POINTER, dbias.ptr)
    out = p.run(handle, p.heuristic(handle)[0])
    ref = p.reference() + bias.astype(np.float64)[:, None]
    _assert_within_bound(
        out,
        np.abs(p.a.astype(np.float64)),
        p.b.astype(np.float64),
        ref,
        np.float32,
        np.float32,
        p.k + 1,
    )


def test_operand_type_must_match_its_layout(handle):
    p = Problem(np.float32, 8, 8, 8)
    p.da = hipblaslt.from_numpy(np.zeros((8, 8), np.float16))
    with pytest.raises(
        ValueError,
        match="A is a DataType.R_16F DeviceArray, but its layout is DataType.R_32F",
    ):
        p.run(handle, p.heuristic(handle)[0])


def test_operand_must_be_big_enough_for_its_layout(handle):
    # Spike 2 saw a too-large size overwrite the next allocation; a too-small buffer is the same
    # bug from the other side.
    p = Problem(np.float32, 8, 8, 8)
    p.dd = c.DeviceArray.empty([7, 8], p.t)
    with pytest.raises(
        ValueError, match="D's layout reaches 256 bytes, but its DeviceArray has 224"
    ):
        p.run(handle, p.heuristic(handle)[0])


def test_device_pointer_mode_is_refused(handle):
    p = Problem(np.float32, 8, 8, 8)
    result = p.heuristic(handle)[0]
    p.desc.set_attribute(c.MatmulDescAttr.POINTER_MODE, c.PointerMode.DEVICE)
    with pytest.raises(ValueError, match="POINTER_MODE must be HOST"):
        p.run(handle, result)


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float16, ml_dtypes.bfloat16, np.float64],
    ids=lambda d: np.dtype(d).name,
)
def test_gemm_matches_numpy(dtype):
    rng = np.random.default_rng(9)
    a = rng.uniform(-1, 1, (33, 17)).astype(dtype)
    b = rng.uniform(-1, 1, (17, 9)).astype(dtype)
    out = hipblaslt.gemm(a, b)
    assert out.dtype == np.dtype(dtype) and out.shape == (33, 9)
    acc = np.float64 if np.dtype(dtype) == np.float64 else np.float32
    ref = a.astype(np.float64) @ b.astype(np.float64)
    _assert_within_bound(
        out,
        np.abs(a.astype(np.float64)),
        np.abs(b.astype(np.float64)),
        ref,
        dtype,
        acc,
        17,
    )
