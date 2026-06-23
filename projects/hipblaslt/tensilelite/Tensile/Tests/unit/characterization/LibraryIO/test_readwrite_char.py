################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for the ``LibraryIO`` read/write primitives:
``write`` (format dispatch), ``writeYAML`` / ``writeJson`` / ``writeMsgPack``,
``read`` / ``readYAML`` / ``readJson``, and the ``StrictTypeLoader`` 0/1-vs-bool
behaviour. All over ``tmp_path``; round-trips snapshot the parsed structure and
the written YAML/JSON text (deterministic for plain data — no version/timestamp
/ absolute path leaks).
"""

import yaml

import pytest

import Tensile.LibraryIO as L

pytestmark = pytest.mark.unit


# A small, plain document covering the scalar types the writers handle.
_DATA = {
    "Name": "example",
    "Count": 3,
    "Flag": True,
    "Ratio": 1.5,
    "Tags": ["a", "b"],
    "Nested": {"x": 1, "y": 0},
}
_DATA_LIST = [{"SolutionIndex": 0, "Name": "k0"}, {"SolutionIndex": 1, "Name": "k1"}]


# ===========================================================================
# writeYAML / readYAML round-trip + written text
# ===========================================================================

def test_write_yaml_text(tmp_path, snapshot):
    p = tmp_path / "out.yaml"
    L.writeYAML(str(p), _DATA)
    assert p.read_text() == snapshot


def test_yaml_roundtrip(tmp_path, snapshot):
    p = tmp_path / "out.yaml"
    L.writeYAML(str(p), _DATA)
    assert L.readYAML(str(p)) == snapshot


def test_write_yaml_kwargs_override(tmp_path, snapshot):
    # Caller-supplied kwargs are respected (explicit_start/end suppressed).
    p = tmp_path / "out.yaml"
    L.writeYAML(str(p), _DATA, explicit_start=False, explicit_end=False)
    assert p.read_text() == snapshot


# ===========================================================================
# writeJson / readJson round-trip + written text
# ===========================================================================

def test_write_yaml_flow_style_override(tmp_path, snapshot):
    # Supplying default_flow_style takes the "already set" branch (no default).
    p = tmp_path / "out.yaml"
    L.writeYAML(str(p), _DATA, default_flow_style=False)
    assert p.read_text() == snapshot


def test_write_json_text(tmp_path, snapshot):
    p = tmp_path / "out.json"
    L.writeJson(str(p), _DATA)
    assert p.read_text() == snapshot


def test_json_roundtrip(tmp_path, snapshot):
    p = tmp_path / "out.json"
    L.writeJson(str(p), _DATA)
    assert L.readJson(str(p)) == snapshot


# ===========================================================================
# writeMsgPack — binary; round-trip through msgpack.unpack
# ===========================================================================

def test_msgpack_roundtrip(tmp_path, snapshot):
    p = tmp_path / "out.dat"
    L.writeMsgPack(str(p), _DATA_LIST)
    import msgpack
    with open(p, "rb") as f:
        loaded = msgpack.unpack(f, raw=False)
    assert loaded == snapshot


# ===========================================================================
# write() format dispatch (appends extension)
# ===========================================================================

def test_write_dispatch_yaml(tmp_path, snapshot):
    base = tmp_path / "art"
    L.write(str(base), _DATA, format="yaml")
    assert (tmp_path / "art.yaml").read_text() == snapshot


def test_write_dispatch_json(tmp_path, snapshot):
    base = tmp_path / "art"
    L.write(str(base), _DATA, format="json")
    assert (tmp_path / "art.json").read_text() == snapshot


def test_write_dispatch_msgpack(tmp_path):
    base = tmp_path / "art"
    L.write(str(base), _DATA_LIST, format="msgpack")
    assert (tmp_path / "art.dat").exists()


def test_write_dispatch_unrecognized(tmp_path):
    with pytest.raises(SystemExit):
        L.write(str(tmp_path / "art"), _DATA, format="xml")


# ===========================================================================
# read() dispatch by extension
# ===========================================================================

def test_read_dispatch_yaml(tmp_path, snapshot):
    p = tmp_path / "doc.yaml"
    L.writeYAML(str(p), _DATA)
    assert L.read(str(p)) == snapshot


def test_read_dispatch_json(tmp_path, snapshot):
    p = tmp_path / "doc.json"
    L.writeJson(str(p), _DATA)
    assert L.read(str(p)) == snapshot


def test_read_dispatch_customized_loader(tmp_path, snapshot):
    # customizedLoader=True routes .yaml through load_yaml_stream(StrictTypeLoader).
    p = tmp_path / "doc.yaml"
    p.write_text("a: 1\nb: 0\nc: true\n")
    assert L.read(str(p), customizedLoader=True) == snapshot


def test_read_dispatch_unrecognized(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("nope")
    with pytest.raises(SystemExit):
        L.read(str(p))


# ===========================================================================
# StrictTypeLoader — 0/1 stay int; true/false -> bool; yes/no stay str
# ===========================================================================

def test_strict_type_loader_values(snapshot):
    doc = "a: 1\nb: 0\nc: true\nd: false\ne: True\nf: False\ng: yes\nh: no\ni: hello\n"
    assert yaml.load(doc, L.StrictTypeLoader) == snapshot


def test_strict_type_loader_types(snapshot):
    doc = "a: 1\nb: 0\nc: true\nd: false\ne: yes\nf: no\n"
    parsed = yaml.load(doc, L.StrictTypeLoader)
    assert {k: type(v).__name__ for k, v in parsed.items()} == snapshot
