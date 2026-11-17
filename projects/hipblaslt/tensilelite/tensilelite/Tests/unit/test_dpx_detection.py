# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only coverage of the DPX detection the `dpx` tests are gated on.

The partitions of one GPU share the PCI domain, bus and device and differ in the function, which
the KFD topology shows as ``location_id = bus << 8 | device << 3 | function``. These tests build
small fake topologies, so they need no GPU.
"""

import pytest

from gpu_detection import dpx_pairs, has_dpx, kfd_gpu_nodes

pytestmark = pytest.mark.unit


def _location(bus, device=0, function=0):
    return (bus << 8) | (device << 3) | function


def _topology(tmp_path, nodes):
    """A fake /sys/class/kfd/kfd/topology/nodes with one directory per ``(simd_count, domain, loc)``."""
    for index, (simd, domain, location) in enumerate(nodes):
        node = tmp_path / str(index)
        node.mkdir()
        (node / "properties").write_text(
            f"cpu_cores_count 0\nsimd_count {simd}\ndomain {domain}\nlocation_id {location}\n"
        )
    return str(tmp_path)


CPU = (0, 0, 0)


def test_a_gpu_in_dpx_mode_is_a_pair(tmp_path):
    # The DPX gfx1250 of the test box: 0001:01:00.0 and 0001:01:00.1.
    nodes = [CPU, CPU, (512, 1, _location(1, 0, 0)), (512, 1, _location(1, 0, 1))]
    path = _topology(tmp_path, nodes)
    assert kfd_gpu_nodes(path) == [(1, 256), (1, 257)]
    assert dpx_pairs(path, environ={}) == [(0, 1)]
    assert has_dpx(path, environ={})


def test_two_separate_cards_are_not_dpx(tmp_path):
    # Two SPX cards: different buses. They cannot reach each other, so they are not partitions.
    path = _topology(tmp_path, [CPU, (512, 1, _location(1)), (512, 1, _location(0x41))])
    assert dpx_pairs(path, environ={}) == []
    assert not has_dpx(path, environ={})


def test_one_spx_card_is_not_dpx(tmp_path):
    path = _topology(tmp_path, [CPU, (512, 1, _location(1))])
    assert not has_dpx(path, environ={})


def test_the_same_bus_in_another_domain_is_another_gpu(tmp_path):
    path = _topology(tmp_path, [(512, 1, _location(1)), (512, 2, _location(1))])
    assert not has_dpx(path, environ={})


def test_more_than_two_partitions_is_another_mode(tmp_path):
    # CPX (eight partitions) and QPX (four) are not DPX.
    for count in (3, 4, 8):
        sub = tmp_path / str(count)
        sub.mkdir()
        path = _topology(sub, [(64, 1, _location(1, 0, f)) for f in range(count)])
        assert dpx_pairs(path, environ={}) == [], count


def test_two_dpx_gpus_give_two_pairs(tmp_path):
    nodes = [
        (512, 1, _location(1, 0, 0)),
        (512, 1, _location(1, 0, 1)),
        (512, 1, _location(0x41, 0, 0)),
        (512, 1, _location(0x41, 0, 1)),
    ]
    path = _topology(tmp_path, nodes)
    assert dpx_pairs(path, environ={}) == [(0, 1), (2, 3)]


@pytest.mark.parametrize(
    "environ, expected",
    [
        ({"HIP_VISIBLE_DEVICES": "0"}, []),  # only one partition is visible
        ({"ROCR_VISIBLE_DEVICES": "1"}, []),
        ({"HIP_VISIBLE_DEVICES": "0,1"}, [(0, 1)]),
        ({"ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "1,0"}, [(0, 1)]),
        ({"HIP_VISIBLE_DEVICES": "2,3"}, [(0, 1)]),  # the second GPU, renumbered
        ({"HIP_VISIBLE_DEVICES": "0,2"}, []),  # one partition of each GPU
        ({"HIP_VISIBLE_DEVICES": "GPU-0123456789abcdef"}, []),  # a UUID cannot be matched: skip
    ],
)
def test_visibility_masks_are_applied(tmp_path, environ, expected):
    nodes = [
        (512, 1, _location(1, 0, 0)),
        (512, 1, _location(1, 0, 1)),
        (512, 1, _location(0x41, 0, 0)),
        (512, 1, _location(0x41, 0, 1)),
    ]
    path = _topology(tmp_path, nodes)
    assert dpx_pairs(path, environ=environ) == expected


def test_a_machine_without_kfd_is_not_dpx(tmp_path):
    assert kfd_gpu_nodes(str(tmp_path / "missing")) == []
    assert not has_dpx(str(tmp_path / "missing"), environ={})
