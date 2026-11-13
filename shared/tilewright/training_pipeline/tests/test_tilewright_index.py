# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import random

import pytest

from lib import tilewright_index as twi


def test_rows_follow_the_engine():
    idx = twi.parse(
        "# header\r\n"
        "S\ta.bin\tchip=0x1A2B\n"
        "S a0.bin   # trailing comment\r\n"
        "S\tb.bin\tchip=0x1a2b\n"
        "S c.bin\n"
        "T\tt.bin\tchip=0xf\n"
        "U u.bin chip=0x0\n"
        "U v.bin\n"
        "V x\n"
        "V\n"
        "W w.bin chip=0x12 extra\n"
        "\n"
    )
    assert idx.rows == [
        ("S", "a.bin", 0x1A2B),
        ("S", "a0.bin", None),
        ("T", "t.bin", 0xF),
        ("U", "v.bin", None),
        ("V", "x", None),
    ]
    assert idx.malformed == ["U", "W"]
    assert idx.malformed_rows == [
        (7, "U", "U u.bin chip=0x0"),
        (11, "W", "W w.bin chip=0x12 extra"),
    ]
    assert idx.dropped == [
        "S\tb.bin\tchip=0x1a2b",
        "S c.bin",
        "U u.bin chip=0x0",
        "V",
        "W w.bin chip=0x12 extra",
    ]
    assert idx.comments == ["# header"]
    assert idx.model_of("S") == "a0.bin"
    assert idx.model_of("S", 0x1A2B) == "a.bin"
    assert idx.model_of("S", 0x1A2C) == "a0.bin"
    assert idx.model_of("T") is None and idx.model_of("T", 0xF) == "t.bin"
    assert idx.model_of("Absent", 0xF) is None
    for stem in ("U", "W"):
        with pytest.raises(ValueError, match="malformed"):
            idx.model_of(stem)


@pytest.mark.parametrize(
    "text, chip",
    [
        ("chip=0x1", 1),
        ("chip=0xFFFF", 0xFFFF),
        ("chip=0x00ab", 0xAB),
        ("chip=0x", None),
        ("chip=0x0", None),
        ("chip=0x0000", None),
        ("chip=0x10000", None),
        ("chip=0X1234", None),
        ("chip=1234", None),
        ("chip=0x12g4", None),
        ("chip=+0x12", None),
        ("CHIP=0x1234", None),
        ("0x1234", None),
        ("chip=0x1234 ", None),
    ],
)
def test_chip_fields(text, chip):
    assert twi.chip_of(text) == chip
    if chip is not None:
        assert twi.chip_of(twi.chip_field(chip)) == chip


def test_chip_field_values():
    assert twi.chip_field(0x1A2B) == "chip=0x1a2b"
    assert twi.chip_field(1) == "chip=0x0001"
    for bad in (0, -1, 0x10000, True, 1.0, "0x1a2b"):
        with pytest.raises(ValueError):
            twi.chip_field(bad)


def test_rendered_rows_put_a_stems_row_without_a_chip_id_first():
    rows = [("S", "c.bin", 0x20), ("A", "a.bin", None), ("S", "s.bin", None)]
    rows.append(("S", "b.bin", 0x3))
    assert twi.render_rows(rows) == [
        "A\ta.bin",
        "S\ts.bin",
        "S\tb.bin\tchip=0x0003",
        "S\tc.bin\tchip=0x0020",
    ]
    assert [twi.row_line(r) for r in twi.in_stem_order(rows)] == [
        "S\ts.bin",
        "S\tb.bin\tchip=0x0003",
        "S\tc.bin\tchip=0x0020",
        "A\ta.bin",
    ]
    assert twi.parse("\n".join(twi.render_rows(rows))).rows == twi.in_stem_order(
        sorted(rows)
    )


def _random_index(rng, files):
    """A random tilewright_index text over stems S0..S2 and `files`, with
    duplicates, comments, blank and one-field lines and malformed rows, its
    fields separated by any of the blanks of C's isspace()."""
    lines = []
    for _ in range(rng.randrange(0, 9)):
        stem = rng.choice(["S0", "S1", "S2"])
        sep = rng.choice(["\t", " ", "  \t", "\v", "\f", "\r", " \f\v"])
        roll = rng.random()
        if roll < 0.08:
            lines.append(rng.choice(["", "# comment", stem, "   "]))
            continue
        line = f"{stem}{sep}{rng.choice(files)}"
        if roll < 0.55:
            chip = rng.choice([0x1A2B, 0x77, 0xFFFF])
            line += f"{sep}chip=0x{chip:0{rng.choice([1, 4])}{rng.choice('xX')}}"
        elif roll < 0.62:
            line += sep + rng.choice(["chip=0x0", "extra", "chip=0x12345", "id=0x1"])
        elif roll < 0.66:
            line += f"{sep}chip=0x10{sep}more"
        if rng.random() < 0.2:
            line += rng.choice([" # note", "\r", "# chip=0x77"])
        lines.append(line)
    return "\n".join(lines) + rng.choice(["", "\n"])


def test_the_engine_reads_an_index_as_parse_says(tmp_path):
    """Random index texts: for every stem and chip ID, the model the engine
    loads (known by its provenance) is the one `Index.model_of` names, and
    both refuse the same stems."""
    tw = pytest.importorskip("tilewright")
    from test_v3 import LARGE, v3_bundle, write

    files = [f"m{i}.bin" for i in range(4)]
    for i, name in enumerate(files):
        data = write(v3_bundle(labels=[LARGE]), provenance={"config_id": name})
        (tmp_path / name).write_bytes(data)
    rng = random.Random(12345)
    checked = refused = 0
    for _ in range(300):
        text = _random_index(rng, files + ["missing.bin"])
        (tmp_path / "tilewright_index").write_bytes(text.encode())
        idx = twi.parse(twi.read_text(tmp_path / "tilewright_index"))
        for stem in ("S0", "S1", "S2", "S9"):
            for chip in (0, 0x1A2B, 0x77, 0xFFFF, 0x1234):
                try:
                    want = idx.model_of(stem, chip)
                except ValueError:
                    with pytest.raises(ValueError, match="malformed row"):
                        tw.load_model_by_index(stem, str(tmp_path), chip)
                    refused += 1
                    continue
                if want == "missing.bin":
                    with pytest.raises(ValueError, match="missing.bin"):
                        tw.load_model_by_index(stem, str(tmp_path), chip)
                    continue
                got = tw.load_model_by_index(stem, str(tmp_path), chip)
                name = None if got is None else tw.describe(got).provenance.config_id
                assert name == want, (text, stem, chip)
                checked += 1
    assert checked > 1000 and refused > 100


def test_read_text_ends_lines_where_the_engine_does(tmp_path):
    """The engine ends a line only at a line feed; a carriage return
    elsewhere separates fields, which Path.read_text would end a line at."""
    path = tmp_path / "tilewright_index"
    path.write_bytes(b"S a.bin\rchip=0x12\r\nS b.bin\n")
    assert twi.read_text(path) == "S a.bin\rchip=0x12\r\nS b.bin\n"
    idx = twi.parse(twi.read_text(path))
    assert idx.rows == [("S", "a.bin", 0x12), ("S", "b.bin", None)]
    assert twi.parse(path.read_text()).rows != idx.rows
