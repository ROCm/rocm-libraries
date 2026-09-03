"""The wheel-digest stamp's contract: rewrite ONLY on a content change.

The stamp gates venv reprovisioning and repacking. Its whole value is the
negative case -- a wheel rebuilt to identical bytes must leave the stamp's mtime
untouched, because CMake and Ninja key on mtime and would otherwise recompile
every kernel for every arch on every build. A test that only checked "digest
changes when content changes" would pass against an unconditional write, which
is precisely the implementation this exists to rule out.
"""

import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "hkp_wheel_digest.py"


def _run(stamp, *wheels):
    cmd = [sys.executable, str(TOOL), "--stamp", str(stamp)]
    for w in wheels:
        cmd += ["--wheel", str(w)]
    return subprocess.run(cmd, capture_output=True, text=True)


def _wheels(tmp_path, a=b"alpha", b=b"beta"):
    w1 = tmp_path / "rocke-0.1.0-py3-none-any.whl"
    w2 = tmp_path / "rocke_library-0.1.0-py3-none-any.whl"
    w1.write_bytes(a)
    w2.write_bytes(b)
    return w1, w2


@pytest.mark.quick
def test_creates_stamp_on_first_run(tmp_path):
    w1, w2 = _wheels(tmp_path)
    stamp = tmp_path / "stamp"

    proc = _run(stamp, w1, w2)

    assert proc.returncode == 0, proc.stderr
    assert stamp.is_file()
    assert len(stamp.read_text().strip()) == 64


@pytest.mark.quick
def test_identical_content_leaves_mtime_untouched(tmp_path):
    """The load-bearing case: a rebuilt-but-identical wheel must not restage.

    Asserts on mtime, not on the digest value: an unconditional write would keep
    the digest identical while bumping mtime, and mtime is what the build reads.
    """
    w1, w2 = _wheels(tmp_path)
    stamp = tmp_path / "stamp"
    _run(stamp, w1, w2)
    before = stamp.stat().st_mtime_ns

    # Rewrite both wheels with the same bytes, as `pip wheel` does every build.
    w1.write_bytes(b"alpha")
    w2.write_bytes(b"beta")
    proc = _run(stamp, w1, w2)

    assert proc.returncode == 0, proc.stderr
    assert stamp.stat().st_mtime_ns == before, "identical wheels must not restamp"
    assert "unchanged" in proc.stdout


@pytest.mark.quick
def test_content_change_rewrites_stamp(tmp_path):
    w1, w2 = _wheels(tmp_path)
    stamp = tmp_path / "stamp"
    _run(stamp, w1, w2)
    first = stamp.read_text().strip()

    w2.write_bytes(b"beta-modified")
    proc = _run(stamp, w1, w2)

    assert proc.returncode == 0, proc.stderr
    assert stamp.read_text().strip() != first
    assert "updated" in proc.stdout


@pytest.mark.quick
def test_digest_is_order_independent(tmp_path):
    # CMake passes the wheels in a fixed order today, but the digest must not
    # silently depend on that: an order-sensitive hash would restage on a
    # reordering that changes nothing.
    w1, w2 = _wheels(tmp_path)
    s1, s2 = tmp_path / "s1", tmp_path / "s2"

    _run(s1, w1, w2)
    _run(s2, w2, w1)

    assert s1.read_text() == s2.read_text()


@pytest.mark.quick
def test_digest_covers_wheel_names_not_just_bytes(tmp_path):
    # Two wheels swapping contents must not hash the same; the name is part of
    # the identity, otherwise a rename would go unnoticed.
    stamp_a = tmp_path / "a" / "stamp"
    stamp_b = tmp_path / "b" / "stamp"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    a1, a2 = _wheels(tmp_path / "a", a=b"one", b=b"two")
    b1, b2 = _wheels(tmp_path / "b", a=b"two", b=b"one")

    _run(stamp_a, a1, a2)
    _run(stamp_b, b1, b2)

    assert stamp_a.read_text() != stamp_b.read_text()


@pytest.mark.quick
def test_missing_wheel_is_a_hard_error(tmp_path):
    """A missing wheel means the caller wired the dependency wrong.

    Digesting nothing would make the staleness chain silently inert -- the same
    class of failure the chain exists to prevent -- so it must fail loudly.
    """
    stamp = tmp_path / "stamp"

    proc = _run(stamp, tmp_path / "absent.whl")

    assert proc.returncode == 1
    assert "not found" in proc.stderr
    assert not stamp.exists()
