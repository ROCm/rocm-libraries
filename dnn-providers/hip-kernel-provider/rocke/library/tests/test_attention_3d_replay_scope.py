# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fixed-binding scope lifetime, validation and whole-pair ownership."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from kernels.common import attention_unified as au
from test_attention_3d_replay_bindings import Tensor
from test_attention_3d_replay_bindings import replay as _replay
from test_attention_workspace import runtime as _runtime

runtime = _runtime
replay = _replay


@pytest.fixture
def scoped(replay, runtime, monkeypatch):
    args, graphs, operations = replay
    calls = []
    cfg = au.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1))

    class Stage:
        def bind(self, values, *, config):
            calls.append(("bind", values, config))
            return lambda: operations.append(("stage", values))

    prepared = SimpleNamespace(
        pipeline=SimpleNamespace(stages=(Stage(), Stage())),
        seg_config=cfg,
        red_config=cfg,
    )
    monkeypatch.setattr(au, "_get_3d_pipeline", lambda *a, **kw: prepared)

    def run(**kw):
        calls.append(("validate", kw))
        return au._run_3d_tiled(**kw)

    monkeypatch.setattr(au, "run_unified_attention_torch", run)
    owner = au.prepare_unified_attention_torch(**args, stream=29)
    return owner, args, operations, calls


@pytest.mark.parametrize("graph", [False, True])
def test_scope_prepares_once_and_drains_before_restoring(scoped, runtime, graph):
    owner, args, operations, calls = scoped
    owner._arguments["use_graph"] = graph
    before = len(calls)
    with owner.replay_scope() as replay:
        assert runtime.stream == 29
        after = len(calls)
        assert after > before
        contexts = list(runtime.contexts)
        operations.clear()
        for _ in range(5):
            replay()
        assert len(calls) == after
        assert runtime.contexts == contexts
        assert len(operations) == (5 if graph else 10)
        assert all(x[0] == ("replay" if graph else "stage") for x in operations)
        assert owner._lock.locked()
        for action in (owner.launch, owner.close, owner.rebind):
            with pytest.raises(RuntimeError, match="already submitting"):
                action()
        with pytest.raises(RuntimeError, match="already submitting"):
            with owner.replay_scope():
                pass
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="thread inside"):
                pool.submit(replay).result()
            with pytest.raises(RuntimeError, match="already submitting"):
                pool.submit(owner.close).result()
    assert operations[-1] == ("wait", 29)
    assert runtime.stream == 17 and not owner._lock.locked()
    assert owner._scope_operation is None
    with pytest.raises(RuntimeError, match="inside replay_scope"):
        replay()
    owner.close()


def test_scope_rebind_changes_storage_and_scalars(scoped):
    owner, args, operations, calls = scoped
    with owner.replay_scope() as replay:
        replay()
    original = next(iter(owner._graphs.values()))
    replacement = Tensor(512)
    owner.rebind(out=replacement, softmax_scale=0.25)
    assert owner._arguments["out"] is replacement
    assert owner._arguments["softmax_scale"] == 0.25
    assert next(iter(owner._graphs.values())) is not original
    with owner.replay_scope() as replay:
        replay()
    with pytest.raises(ValueError, match="prepare again"):
        owner.rebind(stream=37)
    assert owner._arguments["stream"] == 29
    replacement.strides = (512, 128, 1)
    with pytest.raises(ValueError, match="prepare again"):
        with owner.replay_scope():
            pass
    assert owner._scope_setup_thread is None and not owner._lock.locked()
    owner.close()


def test_scope_body_exception_invalidates_callable(scoped, runtime):
    owner, args, operations, calls = scoped
    with pytest.raises(ValueError, match="body"):
        with owner.replay_scope() as replay:
            replay()
            raise ValueError("body")
    assert runtime.stream == 17 and not owner._lock.locked()
    with pytest.raises(RuntimeError, match="inside replay_scope"):
        replay()
    with owner.replay_scope() as again:
        again()
    owner.close()


@pytest.mark.parametrize("action", ["scope", "rebind"])
def test_setup_failure_drains_and_preserves_saved_bindings(
    scoped, runtime, monkeypatch, action
):
    owner, args, operations, calls = scoped
    saved = dict(owner._arguments)
    original = au._run_3d_tiled_on_stream

    def partial(**kw):
        result = original(**kw)
        with pytest.raises(RuntimeError, match="already submitting"):
            owner.launch()
        raise ValueError("setup failed")

    monkeypatch.setattr(au, "_run_3d_tiled_on_stream", partial)
    with pytest.raises(ValueError, match="setup failed"):
        if action == "scope":
            with owner.replay_scope():
                pytest.fail("failed setup must not yield")
        else:
            owner.rebind(out=Tensor(768))
    assert operations[-1] == ("wait", 29)
    assert owner._arguments == saved
    assert owner._scope_operation is None
    assert owner._scope_setup_thread is None and not owner._scope_setup_entered
    assert runtime.stream == 17 and not owner._lock.locked()
    monkeypatch.setattr(au, "_run_3d_tiled_on_stream", original)
    with owner.replay_scope() as replay:
        replay()
    owner.close()


def test_rebind_drain_failure_retains_operation(scoped, monkeypatch):
    owner, args, operations, calls = scoped
    owner._arguments["use_graph"] = False
    original = au.wait_stream_and_release
    count = 0

    def fail_second(stream):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("drain failed")
        original(stream)

    monkeypatch.setattr(au, "wait_stream_and_release", fail_second)
    # Suppress the ordinary launch's optional fence to test the rebind drain.
    with au.no_fence(), pytest.raises(RuntimeError, match="drain failed"):
        owner.rebind(out=Tensor(768))
    assert owner._scope_operation is not None
    assert not owner._lock.locked()
    with pytest.raises(RuntimeError, match="previous scope did not drain"):
        owner.launch()
    monkeypatch.setattr(au, "wait_stream_and_release", original)
    owner.close()


def test_scope_drain_failure_retains_buffers_until_close(scoped, monkeypatch):
    owner, args, operations, calls = scoped
    original = au.wait_stream_and_release

    def fail(stream):
        raise RuntimeError("drain failed")

    with pytest.raises(RuntimeError, match="drain failed"):
        with owner.replay_scope() as replay:
            monkeypatch.setattr(au, "wait_stream_and_release", fail)
            replay()
    assert owner._scope_operation is not None
    assert not owner._lock.locked()
    with pytest.raises(RuntimeError, match="inside replay_scope"):
        replay()
    with pytest.raises(RuntimeError, match="previous scope did not drain"):
        with owner.replay_scope():
            pass
    monkeypatch.setattr(au, "wait_stream_and_release", original)
    owner.close()
    assert owner._scope_operation is None and owner._workspace is None


def test_scope_rejects_capture_closed_and_unsupported_target(
    scoped, runtime, monkeypatch
):
    owner, args, operations, calls = scoped
    runtime.capturing = True
    for action in (owner.rebind, owner.replay_scope):
        with pytest.raises(RuntimeError, match="capture"):
            result = action()
            if hasattr(result, "__enter__"):
                with result:
                    pass
    runtime.capturing = False
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx1250")
    with pytest.raises(NotImplementedError, match="gfx942/gfx950"):
        with owner.replay_scope():
            pass
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        with owner.replay_scope():
            pass
