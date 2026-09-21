# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import re

import gdb

FUNCTION = os.environ["KERNEL_NAME"]

gdb.execute("set pagination off")
gdb.execute("set breakpoint pending on")

gdb.Breakpoint(FUNCTION + "_exec_begin", temporary=True)
gdb.execute("run")

disasm = gdb.execute(f"disassemble {FUNCTION}", to_string=True)

ds_instructions = []
for line in disasm.splitlines():
    # Match lines like:  <+1156>:   ds_write_b32 v3, v1
    m = re.search(r"<\+(\d+)>:\s+(ds_\S+)\s+(.*)", line)
    if not m:
        continue
    offset = int(m.group(1))
    mnemonic = m.group(2)
    operands = [op.strip().split()[0] for op in m.group(3).split(",")]
    if mnemonic.startswith("ds_write"):
        addr_reg = operands[0]
    elif mnemonic.startswith("ds_read"):
        addr_reg = operands[1]
    else:
        raise ValueError(f"Unhandled ds_ instruction: {mnemonic}")
    ds_instructions.append({"offset": offset, "reg": addr_reg, "mnemonic": mnemonic})


class WG0Breakpoint(gdb.Breakpoint):
    """Breakpoint that only stops when the current wave is in workgroup (0,0,0)."""

    def stop(self):
        pos = str(gdb.parse_and_eval("$_dispatch_pos"))
        return "(0,0,0)/" in pos


for instr in ds_instructions:
    spec = f"*{FUNCTION}+{instr['offset']}"
    print(f"# ${instr['reg']} ({instr['mnemonic']} @ +{instr['offset']})")

    WG0Breakpoint(spec, temporary=True)
    gdb.execute("continue")

    # Discover the 4 wave threads of workgroup (0,0,0) by iterating all threads
    # and checking $_dispatch_pos.
    threads = []
    current = gdb.selected_thread()
    for t in gdb.inferiors()[0].threads():
        t.switch()
        if "(0,0,0)/" in str(gdb.parse_and_eval("$_dispatch_pos")):
            threads.append(t.num)
        if len(threads) == 4:
            break
    current.switch()

    for t in sorted(threads):
        gdb.execute(f"thread {t}")
        gdb.execute("x/i $pc")
        gdb.execute(f"p ${instr['reg']}")

gdb.execute("set confirm off")
gdb.execute("quit")
