# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Small single-lane interpreter for emitted persistent-control instructions.

This checks scalar control flow and tile arithmetic on the CPU. It does not
model GPU memory ordering, wave scheduling, or device floating-point accuracy.
Unexpected opcodes fail instead of being silently ignored.
"""

import struct


def reg(name):
    return "s[sgpr%s]" % name


class Machine:
    def __init__(self, **registers):
        self.registers = {reg(k): v for k, v in registers.items()}
        self.scc = False
        self.exec = True
        self.barriers = 0

    def __getitem__(self, name):
        return self.registers.get(reg(name), 0)

    def __setitem__(self, name, value):
        self.registers[reg(name)] = value

    def value(self, operand):
        if operand in self.registers:
            return self.registers[operand]
        try:
            return int(operand, 0)
        except ValueError:
            if operand.startswith(("s", "v", "exec")):
                return 0
            raise AssertionError("Unknown operand " + operand)

    @staticmethod
    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    def run(self, module):
        lines = []
        for raw in str(module).splitlines():
            line = raw.split("//", 1)[0].strip()
            if not line or line.startswith(("/*", "*", "//")):
                continue
            lines.append(line)
        labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(":")}
        pc = 0
        steps = 0
        while pc < len(lines):
            line = lines[pc]
            pc += 1
            steps += 1
            assert steps < 10000, "Unexpected control-flow loop"
            if line.endswith(":"):
                continue
            opcode, _, operands = line.partition(" ")
            args = [a.strip() for a in operands.split(",")]
            if opcode in ("s_branch", "s_cbranch_scc0", "s_cbranch_scc1"):
                taken = opcode == "s_branch" or self.scc == (opcode == "s_cbranch_scc1")
                if taken:
                    if args[0] not in labels:
                        return args[0]
                    pc = labels[args[0]]
                continue
            if opcode in ("s_barrier", "s_barrier_signal", "s_barrier_wait"):
                self.barriers += 1
                continue
            if opcode in ("s_nop", "s_waitcnt"):
                continue
            if opcode.startswith("v_") and not self.exec and opcode != "v_readfirstlane_b32":
                continue
            vals = [self.value(a) for a in args]
            result = None
            if opcode in ("s_mov_b32", "s_mov_b64", "v_mov_b32", "v_readfirstlane_b32"):
                result = vals[1]
            elif opcode == "s_cmov_b32":
                if self.scc:
                    result = vals[1]
            elif opcode == "s_cselect_b32":
                result = vals[1] if self.scc else vals[2]
            elif opcode.startswith("s_cmp_"):
                predicate = opcode.split("_")[2]
                self.scc = {"eq": vals[0] == vals[1], "lt": vals[0] < vals[1],
                            "ge": vals[0] >= vals[1], "gt": vals[0] > vals[1]}[predicate]
            elif opcode == "s_bitcmp1_b32":
                self.scc = bool(vals[0] & (1 << vals[1]))
            elif opcode in ("s_add_u32", "v_add_u32"):
                result = vals[-2] + vals[-1]
                if opcode == "s_add_u32":
                    self.scc = result > 0xffffffff
                result &= 0xffffffff
            elif opcode in ("s_sub_u32", "v_sub_u32"):
                result = (vals[-2] - vals[-1]) & 0xffffffff
            elif opcode in ("s_mul_i32", "v_mul_u32_u24"):
                result = (vals[1] * vals[2]) & 0xffffffff
            elif opcode == "s_and_b32":
                result = vals[1] & vals[2]
                self.scc = bool(result)
            elif opcode == "s_or_b32":
                result = vals[1] | vals[2]
                self.scc = bool(result)
            elif opcode == "s_xor_b32":
                result = vals[1] ^ vals[2]
                self.scc = bool(result)
            elif opcode in ("s_lshr_b32", "s_lshl_b32"):
                result = vals[1] >> vals[2] if opcode == "s_lshr_b32" else (vals[1] << vals[2]) & 0xffffffff
            elif opcode == "v_cvt_f32_u32":
                result = self.f32(vals[1])
            elif opcode == "v_rcp_iflag_f32":
                result = self.f32(1.0 / vals[1])
            elif opcode == "v_mul_f32":
                result = self.f32(vals[1] * vals[2])
            elif opcode == "v_cvt_u32_f32":
                result = int(vals[1]) & 0xffffffff
            elif opcode.startswith("v_cmp_") or opcode in ("v_cmpx_eq_u32", "v_cmpx_gt_u32"):
                result = int(vals[1] == vals[2] if "_eq_" in opcode else vals[1] > vals[2])
            else:
                raise AssertionError("Uninterpreted instruction: " + line)
            if result is not None:
                self.registers[args[0]] = result
                if args[0].startswith("exec"):
                    self.exec = bool(result)
        return None
