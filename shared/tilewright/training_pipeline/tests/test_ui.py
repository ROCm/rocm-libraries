# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from lib import ui


@pytest.mark.parametrize(
    "seconds, text",
    [
        (0, "0s"),
        (59.4, "59s"),
        (60, "1m00s"),
        (3599, "59m59s"),
        (3600, "1h00m"),
        (3725, "1h02m"),
    ],
)
def test_fmt_dur(seconds, text):
    assert ui.fmt_dur(seconds) == text


def test_fmt_int():
    assert ui.fmt_int(1234567) == "1,234,567"
    assert ui.fmt_int(0) == "0"


def test_log_lines_carry_tag_and_message(capsys):
    ui.ok("t1", "fine")
    ui.info("t2", "note")
    ui.warn("t3", "careful")
    ui.err("t4", "broken")
    ui.grey("t5", "quiet")
    ui.banner("Title")
    captured = capsys.readouterr()
    for stream, tag, msg in (
        (captured.out, "t1", "fine"),
        (captured.out, "t2", "note"),
        (captured.err, "t3", "careful"),
        (captured.err, "t4", "broken"),
        (captured.out, "t5", "quiet"),
    ):
        assert f"[{tag}]" in stream and msg in stream
    assert "careful" not in captured.out and "broken" not in captured.out
    assert "Title" in captured.out and "=" * 80 in captured.out
