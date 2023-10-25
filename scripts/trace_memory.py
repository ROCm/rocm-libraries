# pylint: disable=E501
"""
This script is intended to be used from within rocGDB.
e.g.: rocgdb -ex "source trace_memory.py" -ex "gpu_memory_trace -h /work/tensile/Tensile/working/0_Build/client/tensile_client
"""

import gdb

import argparse
import numpy as np
import shlex
import pathlib
import sys

repo_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(repo_dir / "scripts" / "lib"))

import matrix_utils  # noqa: E402


class MemoryTrace(gdb.Command):
    """Traces GPU memory access for a kernel in rocGDB"""

    UNACCESSED = -250

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
                help="Workgroup to track memory access in, e.g.: '(x,y,z)'",
                type=str,
                required=True,
            )
            subparser.add_argument(
                "--instruction_address",
                help="Addresses of the load instructions to inspect.",
                nargs="*",
            )
            subparser.add_argument(
                "--buffer_descriptor",
                type=int,
                help="The starting sgpr of the buffer descrtiptor. "
                "Either specify one to use for every instruction, or one for each.",
                nargs="*",
            )
            subparser.add_argument(
                "--offset",
                type=int,
                help="The vgpr of the offset. Either specify one to use for every instruction, or one for each.",
                nargs="*",
            )
            subparser.add_argument(
                "--instruction_width",
                type=int,
                help="The number of bytes accessed by each memory instruction.",
                default=16,
            )
            subparser.add_argument(
                "--trace_count",
                type=int,
                help="The maximum number of times to break.",
                default=16,
            )
            subparser.add_argument(
                "--base_address",
                type=str,
                help="The base address of the matrix.",
                required=True,
            )
            subparser.add_argument(
                "--matrix_m",
                type=int,
                help="The first dimension of the matrix in bytes.",
                required=True,
            )
            subparser.add_argument(
                "--matrix_n",
                type=int,
                help="The second dimension of the matrix in bytes.",
                required=True,
            )
            subparser.add_argument(
                "--image_file",
                help="File to write image to.",
                type=str,
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
        bp_kernel = gdb.Breakpoint(args.kernel_label)
        gdb.execute(f"run {args.run_command}")
        bp_kernel.enabled = False

    def setupBreakpoints(self, args):
        for thread in gdb.selected_inferior().threads():
            thread.switch()
            thread_id = thread.global_num
            wg_info = str(gdb.convenience_variable("_dispatch_pos"))
            if args.workgroup in wg_info:
                self.bp_thread_loads[thread_id] = list()
                for address in args.instruction_address:
                    self.bp_thread_loads[thread_id].append(
                        gdb.Breakpoint(f"*{address}")
                    )
                    self.bp_thread_loads[thread_id][-1].thread = thread_id

        for i, address in enumerate(args.instruction_address):
            self.offset_registers[int(address, 16)] = (
                args.offset[0] if len(args.offset) == 1 else args.offset[i]
            )
            self.bd_registers[int(address, 16)] = (
                args.buffer_descriptor[0]
                if len(args.buffer_descriptor) == 1
                else args.buffer_descriptor[i]
            )

    def traceMemory(self, args):
        csv_file = None
        if args.csv_file is not None:
            csv_file = open(args.csv_file, "w")
            csv_file.write(
                "Address,ThreadID,BasePointer,AdjustedBase,Offset,Size,BufferDescriptor\n"
            )

        inferior = gdb.selected_inferior()

        count = 0
        gdb.execute("continue")
        while inferior.is_valid() and count < args.trace_count:
            for thread in gdb.selected_inferior().threads():
                thread_id = thread.global_num
                if thread_id not in self.bp_thread_loads:
                    continue

                thread.switch()

                pc_info = gdb.selected_frame().pc()
                if all(
                    int(address, 16) != pc_info for address in args.instruction_address
                ):
                    continue

                wf = int(
                    str(gdb.convenience_variable("_lane_workgroup_pos"))[2:-2].split(
                        ","
                    )[0]
                )

                offsets = gdb.execute(
                    f"info register v{self.offset_registers[pc_info]}", to_string=True
                )
                offsets = (
                    offsets.replace(f"v{self.offset_registers[pc_info]}", "")
                    .strip()[1:-1]
                    .split(", ")
                )
                s0 = gdb.execute(
                    f"info register s{self.bd_registers[pc_info]}", to_string=True
                ).split()[1]
                s1 = gdb.execute(
                    f"info register s{self.bd_registers[pc_info] + 1}", to_string=True
                ).split()[1]
                s2 = gdb.execute(
                    f"info register s{self.bd_registers[pc_info] + 2}", to_string=True
                ).split()[1]
                s3 = gdb.execute(
                    f"info register s{self.bd_registers[pc_info] + 3}", to_string=True
                ).split()[1]

                buffer_descriptor = f"{s3[2:]}{s2[2:]}{s1[2:]}{s0[2:]}"
                base_pointer = f"{s1[2:]}{s0[2:]}"
                size = f"{s2[2:]}"

                if (
                    self.data[
                        int(base_pointer, 16)
                        - int(args.base_address, 16)
                        + int(offsets[0], 16)
                    ]
                    == MemoryTrace.UNACCESSED
                ):
                    count += 1
                    for offset in offsets:
                        address = (
                            int(base_pointer, 16)
                            - int(args.base_address, 16)
                            + int(offset, 16)
                        )
                        if csv_file is not None:
                            csv_file.write(
                                f"{address},"
                                f"{thread_id},"
                                f"{int(base_pointer,16)},"
                                f"{int(base_pointer,16) - int(args.base_address,16)},"
                                f"{int(offset,16)},"
                                f"{int(size,16)},"
                                f"{buffer_descriptor}\n"
                            )
                        for i in range(args.instruction_width):
                            self.data[address + i] = wf
                        wf += 1
            gdb.execute("continue")

    def complete(self, text, word):
        # Disable autocomplete.
        return gdb.COMPLETE_NONE

    def invoke(self, args, from_tty):
        print(f"Args Passed: {args}")
        args = self.parser.parse_args(shlex.split(args))
        print(f"Args Parsed: {args}")

        self.getToKernel(args)

        if args.command in ["get_to_kernel"]:
            return

        self.setupBreakpoints(args)

        self.data = np.zeros(args.matrix_m * args.matrix_n) + MemoryTrace.UNACCESSED
        try:
            self.traceMemory(args)
        except Exception as e:
            print(f"Stopping early due to exception:\n{e}")

        if args.image_file is not None:
            matrix_utils.matrixIO.writeNumpyToImage(
                self.data,
                args.matrix_n,
                args.matrix_m,
                args.image_file,
                MemoryTrace.UNACCESSED,
            )


MemoryTrace()
