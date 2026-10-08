# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute emitted input-bias workspace addresses and tree accumulation on CPU."""

import re
from types import SimpleNamespace

import pytest
from rocisa.container import sgpr

from Tensile.Component import Component
from Tensile.Components.StreamK import StreamKMemoryOrderingDefault
from _persistent_isa import Machine
from test_streamk_spatial_cluster import _setup

pytestmark = pytest.mark.unit


class _PayloadMachine(Machine):
    """Single-lane LDS/VMEM extension; ordering is checked by assembly tests."""

    def run_payload(self, module, lds, workspace):
        for raw in str(module).splitlines():
            line = raw.split("//", 1)[0].strip()
            if not line or line.startswith(("/*", "*")):
                continue
            opcode, _, operands = line.partition(" ")
            args = [part.strip().split()[0] for part in operands.split(",") if part.strip()]
            if opcode == "s_mov_b64" and re.fullmatch(r"s\[\d+:\d+\]", args[0]):
                first = int(args[0][2:].split(":")[0])
                value = self.value(args[1])
                self.registers[f"s{first}"] = value & 0xffffffff
                self.registers[f"s{first + 1}"] = value >> 32
            elif opcode.startswith("s_wait") or opcode == "s_barrier":
                continue
            elif opcode == "v_lshlrev_b32":
                if self.exec:
                    self.registers[args[0]] = self.value(args[2]) << self.value(args[1])
            elif opcode == "v_cmpx_lt_u32":
                self.exec = self.value(args[-2]) < self.value(args[-1])
            elif opcode in ("ds_read_b32", "ds_load_b32", "ds_write_b32", "ds_store_b32"):
                if not self.exec:
                    continue
                if opcode in ("ds_read_b32", "ds_load_b32"):
                    self.registers[args[0]] = lds[self.value(args[1]) // 4]
                else:
                    lds[self.value(args[0]) // 4] = self.value(args[1])
            elif opcode.startswith(("buffer_load", "buffer_store")):
                if not self.exec:
                    continue
                first = int(args[2][2:].split(":")[0])
                base = self.value(f"s{first}") + (self.value(f"s{first + 1}") << 32)
                address = base + self.value(args[1])
                if opcode.startswith("buffer_load"):
                    self.registers[args[0]] = workspace[address]
                else:
                    workspace[address] = self.value(args[0])
            elif opcode == "v_add_f32":
                if self.exec:
                    self.registers[args[0]] = self.f32(self.value(args[1]) + self.value(args[2]))
            else:
                super().run(line)


@pytest.mark.parametrize("source,transpose", [("A", False), ("B", False), ("A", True), ("B", True)])
@pytest.mark.parametrize("shape", [(2, 1), (2, 2), (4, 4)])
def test_emitted_input_bias_tree_retains_every_producer_and_peer(monkeypatch, source, transpose, shape,
                                                               _isolate_rocisa_state):
    from gpu_test_helpers import init_rocisa
    init_rocisa(target="gfx942", wavesize=64)
    kernel, processing, _, writer = _setup(monkeypatch, shape)
    kernel.update(MacroTile0=16, MacroTile1=32, NumThreads=64)
    axis = int(source == "B") ^ int(transpose)
    kernel["ProblemType"].update(Gradient=True, UseBias=True, BiasSrc=source,
                                 Index01A=int(transpose), Index01B=1 ^ int(transpose))
    writer.states.version = (9, 4, 2)
    writer.states.bpeCinternal = 4
    writer.states.lraTileProperties = [SimpleNamespace(maxKId=2), SimpleNamespace(maxKId=4)]
    writer.states.archCaps.update(RequiresXCntForVolatileVMEM=False, EnableXnackReplay=False)
    monkeypatch.setattr(Component.StreamKMemoryOrdering, "find", lambda _: StreamKMemoryOrderingDefault())
    publish = processing.inputBiasPartials(writer, kernel, sgpr("Producer"))
    reduce = processing.inputBiasPartials(writer, kernel, sgpr("Producer"), reduce=True)
    peers = shape[0] * shape[1]
    count = kernel[f"MacroTile{axis}"] * writer.states.lraTileProperties[axis].maxKId
    slot_bytes = kernel[f"MacroTile{axis}"] * kernel["WavefrontSize"] * 4
    base = (1 << 32) - 512
    bias_base = base + 4 * peers * 16 * 32 * 4
    workspace = {}
    payloads = {(worker, peer): [float(100 * worker + 10 * peer + i) for i in range(count)]
                for worker in range(4) for peer in range(peers)}
    expected = {peer: [sum(payloads[worker, peer][i] for worker in range(4)) for i in range(count)]
                for peer in range(peers)}

    def execute(module, worker, peer, partner=None):
        for serial in range(kernel["NumThreads"]):
            machine = _PayloadMachine(Producer=worker if partner is None else partner,
                                      StreamKClusterPeer=peer, skGrid=4)
            machine.registers[str(sgpr("AddressWS", 2))] = base
            machine.registers["Srd127_96"] = 0
            machine.registers["v[vgprSerial]"] = serial
            machine.run_payload(module, payloads[worker, peer], workspace)

    for peer in range(peers):
        execute(publish, 1, peer)
        execute(publish, 3, peer)
        execute(reduce, 2, peer, 3)
        execute(publish, 2, peer)
        execute(reduce, 0, peer, 1)
        execute(reduce, 0, peer, 2)
        assert payloads[0, peer] == expected[peer]
    assert len(workspace) == 3 * peers * count
    for address in workspace:
        slot, element = divmod(address - bias_base, slot_bytes)
        assert slot // peers in (1, 2, 3)
        assert element % 4 == 0 and element // 4 < count
