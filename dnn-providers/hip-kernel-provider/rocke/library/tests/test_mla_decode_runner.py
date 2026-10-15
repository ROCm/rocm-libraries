# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU checks for :class:`builders.mla.mla_decode_runner.MlaDecodeRunner`.

The runner's host-side refusals are what keep a mismatched tensor from being
read or written out of bounds on the device, so they are tested without one:
the input checks run on CPU tensors through a runner that never compiled, and
the ``num_seg`` override is refused before anything is compiled.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch required")

from builders.mla.mla_decode_runner import MlaDecodeRunner  # noqa: E402
from dispatch.mla import MLARequest, dispatch_mla, mla_decode_launch  # noqa: E402

H, T, B, MAX_K = 128, 3, 4, 100  # one sequence has no query
PAGES = -(-MAX_K // 16)


@pytest.fixture(autouse=True)
def _llvm22(monkeypatch):
    """Decode admission refuses llvm20; pin a flavor it serves."""
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm22")


def _result(**overrides):
    req = dict(
        num_heads=H,
        total_q=T,
        num_seqs=B,
        max_seqlen_q=1,
        max_seqlen_k=MAX_K,
        arch="gfx942",
        num_cus=304,
    )
    req.update(overrides)
    return dispatch_mla(MLARequest(**req))


def _uncompiled_runner(result) -> MlaDecodeRunner:
    """A runner holding ``result``'s contract, without compiling a kernel."""
    runner = object.__new__(MlaDecodeRunner)
    runner.launch = mla_decode_launch(result)
    runner.request = result.request
    runner.spec = result.spec
    return runner


def _inputs(**overrides) -> dict:
    t = {
        "qa": torch.zeros(T, H, 576, dtype=torch.bfloat16),
        "c_kv": torch.zeros(B * PAGES, 16, 512, dtype=torch.bfloat16),
        "k_rope": torch.zeros(B * PAGES, 16, 64, dtype=torch.bfloat16),
        "cu_seqlens_q": torch.tensor([0, 1, 1, 2, 3], dtype=torch.int32),
        "cu_seqlens_k": torch.tensor([0, 100, 150, 160, 260], dtype=torch.int32),
        "block_table": torch.zeros(B, PAGES, dtype=torch.int32),
        "out": torch.zeros(T, H, 512, dtype=torch.bfloat16),
        "lse": torch.zeros(T, H, dtype=torch.float32),
    }
    t.update(overrides)
    return t


def _check(runner, t):
    runner._check_inputs(
        t["qa"],
        t["c_kv"],
        t["k_rope"],
        t["cu_seqlens_q"],
        t["cu_seqlens_k"],
        t["block_table"],
        t["out"],
        t["lse"],
    )


class TestInputChecks:
    def test_well_formed_inputs_pass(self):
        _check(_uncompiled_runner(_result()), _inputs())

    @pytest.mark.parametrize(
        "name, bad, match",
        [
            ("qa", torch.zeros(T + 1, H, 576, dtype=torch.bfloat16), "qa has shape"),
            ("qa", torch.zeros(T, H, 576, dtype=torch.float16), "qa must be"),
            (
                "cu_seqlens_q",
                torch.tensor([0, 1, 1, 2, 3], dtype=torch.int64),
                "cu_seqlens_q must be",
            ),
            ("out", torch.zeros(T, H, 128, dtype=torch.bfloat16), "out has shape"),
            ("out", torch.zeros(T, H, 512, dtype=torch.float32), "out must be"),
            (
                "out",
                torch.zeros(H, T, 512, dtype=torch.bfloat16).transpose(0, 1),
                "out must be contiguous",
            ),
            ("lse", torch.zeros(T, H + 1, dtype=torch.float32), "lse has shape"),
            (
                "c_kv",
                torch.zeros(4, 16, 256, dtype=torch.bfloat16),
                "c_kv must be paged",
            ),
            ("k_rope", torch.zeros(4, 16, 64, dtype=torch.float32), "k_rope must be"),
            (
                "block_table",
                torch.zeros(B, PAGES, dtype=torch.int64),
                "block_table must be torch.int32",
            ),
            # Too narrow for max_seqlen_k -- and a zero-width table, whose
            # slot 0 the kernel would read for every sequence.
            (
                "block_table",
                torch.zeros(B, PAGES - 1, dtype=torch.int32),
                "block_table has 6 columns",
            ),
            (
                "block_table",
                torch.zeros(B, 0, dtype=torch.int32),
                "block_table has 0 columns",
            ),
            (
                "block_table",
                torch.zeros(B - 1, PAGES, dtype=torch.int32),
                "rows < num_seqs=4",
            ),
            (
                "block_table",
                torch.zeros(PAGES, B, dtype=torch.int32).t(),
                "unit column stride",
            ),
            (
                "qa",
                torch.zeros(H, T, 576, dtype=torch.bfloat16).transpose(0, 1),
                "qa must be contiguous",
            ),
            (
                "block_table",
                torch.zeros(B, PAGES, dtype=torch.int32, device="meta"),
                "block_table is on meta",
            ),
        ],
    )
    def test_mismatched_inputs_are_refused(self, name, bad, match):
        with pytest.raises(ValueError, match=match):
            _check(_uncompiled_runner(_result()), _inputs(**{name: bad}))

    def test_contents_must_respect_the_request(self):
        runner = _uncompiled_runner(_result())
        t = _inputs()
        runner.check_contents(t["cu_seqlens_q"], t["cu_seqlens_k"])
        two = torch.tensor([0, 2, 2, 2, 3], dtype=torch.int32)
        with pytest.raises(ValueError, match="max_seqlen_q=1"):
            runner.check_contents(two, t["cu_seqlens_k"])
        long_k = torch.tensor([0, 101, 150, 160, 260], dtype=torch.int32)
        with pytest.raises(ValueError, match="max_seqlen_k=100"):
            runner.check_contents(t["cu_seqlens_q"], long_k)
        decreasing = torch.tensor([0, 1, 0, 2, 3], dtype=torch.int32)
        with pytest.raises(ValueError, match="non-decreasing"):
            runner.check_contents(decreasing, t["cu_seqlens_k"])
        # Every length <= 1, but the offsets address rows past total_q.
        shifted = torch.tensor([2, 3, 3, 4, 5], dtype=torch.int32)
        with pytest.raises(ValueError, match="total_q=3"):
            runner.check_contents(shifted, t["cu_seqlens_k"])

    def test_contents_check_used_page_ids_only(self):
        runner = _uncompiled_runner(_result())
        t = _inputs()
        bt = torch.zeros(B, PAGES, dtype=torch.int32)
        bt[2, 1:] = -1  # sequence 2 (10 keys) owns one page: a poisoned tail
        runner.check_contents(t["cu_seqlens_q"], t["cu_seqlens_k"], bt, num_pages=8)
        bt[0, 3] = 8  # a used slot past the cache
        with pytest.raises(ValueError, match="row 0 names a page outside"):
            runner.check_contents(t["cu_seqlens_q"], t["cu_seqlens_k"], bt, num_pages=8)


class TestNumSegOverride:
    def test_override_past_the_i32_workspace_is_refused(self):
        # 2000 tokens x 128 heads x 512 x 64 segments overflows i32 offsets;
        # dispatch would have clamped it, and the override must not bypass that.
        result = _result(total_q=2000, num_seqs=2000)
        with pytest.raises(ValueError, match="i32 offsets"):
            MlaDecodeRunner(result, num_seg=64)

    @pytest.fixture
    def no_compile(self, monkeypatch):
        """Construct a runner without comgr or a device."""
        import types

        import rocke
        import rocke.runtime

        monkeypatch.setattr(
            rocke,
            "compile_kernel",
            lambda kernel, **kw: types.SimpleNamespace(
                hsaco=b"", kernel_name=kernel.name
            ),
        )
        monkeypatch.setattr(
            rocke.runtime, "KernelLauncher", lambda **kw: types.SimpleNamespace(**kw)
        )

    @pytest.mark.parametrize("num_seg", [1, 7])
    def test_override_rebuilds_grid_signature_and_identity(self, no_compile, num_seg):
        result = _result(max_seqlen_k=8192)
        assert result.spec.num_seg not in (1, 7)  # the override changes it
        runner = MlaDecodeRunner(result, num_seg=num_seg)
        forced = runner.launch.result
        assert runner.spec.num_seg == forced.spec.num_seg == num_seg
        assert runner.grid == (2, 4, num_seg)
        assert f"num_seg forced to {num_seg}" in forced.explanation
        names = [a["name"] for a in runner._latent.signature]
        if num_seg == 1:
            # Split -> single pass is a different binary: a new identity.
            assert forced.kernel_id.compile_key != result.kernel_id.compile_key
            assert runner._combine is None
            assert runner.launch.workspace_shapes() == {}
            assert "num_seg" not in names
        else:
            assert runner._combine is not None
            assert runner.launch.workspace_shapes()["part_acc_ptr"] == (
                T,
                num_seg,
                128,
                512,
            )
            assert names[-1] == "num_seg"
