################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

# pylint: disable=E501
"""
This script is intended to be used from within rocGDB.
e.g.: rocgdb -ex "source trace_memory.py" -ex "gpu_memory_trace -h /work/tensile/Tensile/working/0_Build/client/tensile_client
"""

import argparse
import gdb
import re
import shlex

from dataclasses import dataclass


@dataclass
class MemoryTraceLocation:
    address: int
    buffer_descriptor: int
    offset_register: int


def mkBreakpoint(address, thread_id):
    bp = gdb.Breakpoint(f"*{hex(address)}")
    bp.thread = thread_id
    return bp


def workgroupArgument(s):
    try:
        x, y, z = map(int, s.split(","))
        return x, y, z
    except Exception:
        raise argparse.ArgumentTypeError("Workgroup coordinates must be: x,y,z")


def workCoordinates():
    """Return (workgroup, workitem) coordiantes."""
    # see https://rocm.docs.amd.com/projects/ROCgdb/en/latest/ROCgdb/gdb/doc/gdb/AMD-GPU.html
    position = gdb.convenience_variable("_dispatch_pos").string()
    m = re.match(r"\((\d+),(\d+),(\d+)\)/(\d+)", position)
    if m is not None:
        return (int(m[1]), int(m[2]), int(m[3])), int(m[4])
    return None, None


class MemoryTrace(gdb.Command):
    """Traces GPU memory access for a kernel in rocGDB"""

    def __init__(self):
        super(MemoryTrace, self).__init__("gpu_memory_trace", gdb.COMMAND_USER)
        self.parser = argparse.ArgumentParser(
            description="Extract memory access from GDB"
        )
        subparsers = self.parser.add_subparsers(dest="command", required=True)

        full_subparse = subparsers.add_parser(
            "full_memory_trace",
            help="Perform full memory trace.",
            description="Perform full memory trace.",
        )

        kernel_subparse = subparsers.add_parser(
            "get_to_kernel",
            help="Only do steps necesary to get to the kernel.",
            description="Only do steps necesary to get to the kernel.",
        )

        for subparser in [full_subparse, kernel_subparse]:
            subparser.add_argument(
                "--instruction",
                help="Buffer instruction kind to inspect: 'load' or 'store'",
                type=str,
                required=True,
            )
            subparser.add_argument(
                "--kernel_label",
                help="Name label of the kernel.",
                type=str,
                required=True,
            )
            subparser.add_argument(
                "--run_command",
                type=str,
                help="Command to launch the executable.",
                required=True,
            )

        for subparser in [full_subparse]:
            subparser.add_argument(
                "--workgroup",
                help="Workgroup to track memory access in, e.g.: x,y,z",
                type=workgroupArgument,
                action="append",
                required=True,
            )
            subparser.add_argument(
                "--trace_count",
                type=int,
                help="The maximum number of times to break (0 means no-limit -- yolo!).",
                default=0,
            )
            subparser.add_argument(
                "--base_address",
                type=str,
                help="The base address of the matrix.",
                required=True,
            )
            subparser.add_argument(
                "--csv_file",
                help="File to write csv to.",
                type=str,
            )

        self.bp_thread_loads = dict()
        self.offset_registers = dict()
        self.bd_registers = dict()

    def getToKernel(self, args):
        gdb.execute("set pagination off")
        gdb.execute("set breakpoint pending on")
        gdb.Breakpoint(args.kernel_label)
        gdb.execute(f"run {args.run_command}")

    def automagicLocations(self, loadstore):
        r = re.compile(r"buffer_(\w+)_(\S+) v\S+, v([0-9]+), s\[([0-9]+):")

        self.locations = []

        arch = gdb.selected_frame().architecture()
        pc = gdb.selected_frame().pc()
        while True:
            inst = arch.disassemble(pc)[0]
            if "buffer_" in inst["asm"]:
                m = r.match(inst["asm"])
                if m is not None:
                    if m[1] == loadstore:
                        address = inst["addr"]
                        buffer_descriptor = int(m[4])
                        offset = int(m[3])
                        self.locations.append(
                            MemoryTraceLocation(address, buffer_descriptor, offset)
                        )

            # this sometimes walks past the last "s_endprgm", which is harmless
            if "illegal" in inst["asm"]:
                break

            pc += inst["length"]

        print(f"Found {len(self.locations)} matching instructions:")
        for location in self.locations:
            print(location)

        for location in self.locations:
            self.offset_registers[location.address] = location.offset_register
            self.bd_registers[location.address] = location.buffer_descriptor

    def traceMemory(self, args):
        csv_file = None
        if args.csv_file is not None:
            columns = [
                "Address",
                "ThreadID",
                "BasePointer",
                "AdjustedBase",
                "Offset",
                "Size",
                "BufferDescriptor",
                "WorkgroupX",
                "WorkgroupY",
                "WorkgroupZ",
                "Workitem",
            ]
            csv_file = open(args.csv_file, "w")
            csv_file.write(",".join(columns) + "\n")

        inferior = gdb.selected_inferior()
        uint32 = gdb.selected_frame().architecture().integer_type(32, False)
        base_address = int(args.base_address, 16)

        count = 0
        gdb.execute("continue")
        while inferior.is_valid() and (
            args.trace_count == 0 or count < args.trace_count
        ):
            for thread in gdb.selected_inferior().threads():
                thread_id = thread.global_num
                thread.switch()
                if thread_id not in self.bp_thread_loads:
                    # if thread is not in self.bp_thread_loads yet, we
                    # probably broke at the top of the kernel in a new
                    # wavefront.  try adding all the breakpoints for
                    # this wavefront.
                    wg, wf = workCoordinates()
                    if wg in args.workgroup:
                        self.bp_thread_loads[thread_id] = [
                            mkBreakpoint(location.address, thread_id)
                            for location in self.locations
                        ]

                if thread_id not in self.bp_thread_loads:
                    continue

                frame = gdb.selected_frame()

                wg, wf = workCoordinates()

                pc_info = frame.pc()
                if all(location.address != pc_info for location in self.locations):
                    continue

                # read registers (`gdb.Value`), cast them to uint32, then convert to `int`
                offsets = frame.read_register(f"v{self.offset_registers[pc_info]}")
                offsets = [int(offsets[i].cast(uint32)) for i in range(64)]
                s = [
                    int(
                        frame.read_register(f"s{self.bd_registers[pc_info] + i}").cast(
                            uint32
                        )
                    )
                    for i in range(4)
                ]

                buffer_descriptor = hex(
                    (s[3] << 96) + (s[2] << 64) + (s[1] << 32) + s[0]
                )
                base_pointer = (s[1] << 32) + s[0]
                size = s[2]

                for lane, offset in enumerate(offsets):
                    address = base_pointer - base_address + offset
                    if csv_file is not None:
                        csv_file.write(
                            f"{address},"
                            f"{thread_id},"
                            f"{base_pointer},"
                            f"{base_pointer - base_address},"
                            f"{offset},"
                            f"{size},"
                            f"{buffer_descriptor},"
                            f"{wg[0]},"
                            f"{wg[1]},"
                            f"{wg[2]},"
                            f"{wf * 64 + lane}\n"
                        )

            count += 1
            gdb.execute("continue")

    def complete(self, text, word):
        # Disable autocomplete.
        return gdb.COMPLETE_NONE

    def invoke(self, args, from_tty):
        print(f"Args Passed: {args}")
        args = self.parser.parse_args(shlex.split(args))
        print(f"Args Parsed: {args}")

        self.getToKernel(args)
        self.automagicLocations(args.instruction)

        if args.command in ["get_to_kernel"]:
            return

        try:
            self.traceMemory(args)
        except Exception as e:
            print(f"Stopping early due to exception:\n{e}")


MemoryTrace()
