#!/usr/bin/env python

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


import argparse
import re

import utils

DESCRIPTION = """
This script converts a file containing AMD GPU machine code to C++ code using the
Instruction class. It accepts a file name as an argument. This file should
contain AMD GPU machine code. The generated C++ code will be printed to
standard output.

This does not handle all possible cases. Notably, it does not handle the case where:
v[2:5] is used somewhere in the file and v[2:3] is also used somewhere in the file.
It DOES handle the case where:
v[2:5] is used somewhere in the file and v2 and v3 are also used somewhere in the file.
"""

special_registers = {
    "vcc": "m_context->getVCC()",
    "scc": "m_context->getSCC()",
    "exec": "m_context->getExec()",
}


# Generate C++ code for initializing all Registers that will be used
def initialize_registers(registers):
    retval = ""
    for register in registers.values():
        (name, index, register_type, children) = register
        if index is not None:
            continue
        retval += """auto {} = std::make_shared<Register::Value>(m_context,
        Register::Type::{}, DataType::Int32, {});\n""".format(
            name, register_type, str(children)
        )
        if children > 1:
            retval += "co_yield " + name + "->allocate();\n"
    return retval


# Generate C++ code for declaring all Registers that will be used
def declare_registers(registers):
    retval = ""
    for register in registers.values():
        (name, index, register_type, children) = register
        if index is not None:
            continue
        retval += "Register::ValuePtr {};\n".format(name)
    return retval


# Generate C++ code for defining all Registers that will be used
def define_registers(registers):
    retval = ""
    for register in registers.values():
        (name, index, register_type, children) = register
        if index is not None:
            continue
        retval += """{} = std::make_shared<Register::Value>(m_context,
        Register::Type::{}, DataType::Int32, {});\n""".format(
            name, register_type, str(children)
        )
    return retval


# Generate C++ code for allocating all Registers that need to be allocated
def allocate_registers(struct_name, registers):
    retval = ""
    for register in registers.values():
        (name, index, register_type, children) = register
        if index is not None:
            continue
        if children > 1:
            retval += "co_yield {}.{}->allocate();\n".format(struct_name, name)
    return retval


# Generate C++ code for initializing all Labels that will be used
def initialize_labels(labels):
    retval = ""
    for label in labels:
        value = labels[label]
        retval += 'auto {} = m_context->labelAllocator()->label("{}");\n'.format(
            value, label
        )
    return retval


# Generate C++ code for declaring all Labels that will be used
def declare_labels(labels):
    retval = ""
    for label in labels:
        value = labels[label]
        retval += "Register::ValuePtr  {};\n".format(value)
    return retval


# Generate C++ code defining all Labels that will be used
def define_labels(labels):
    retval = ""
    for label in labels:
        value = labels[label]
        retval += '{}  = m_context->labelAllocator()->label("{}");\n'.format(
            value, label
        )
    return retval


# Find all of the registers and labels within a file and create
# dictionaries to map between registers and labels within machine code
# to values of type Register::Value
def find_registers(my_lines):
    multi_register = re.compile(r"[sv]\[(\d+):(\d+)\]")
    counter = 0
    registers = dict()
    for my_line in my_lines:
        line_split = my_line.split(" ", 1)
        # instruction = line_split[0]
        if len(line_split) > 1:
            args = line_split[1]
            split_args = args.split(",")
            for arg in split_args:
                arg = arg.strip()
                if arg in special_registers:
                    continue
                elif arg[0] == "v":
                    register_type = "Vector"
                elif arg[0] == "s":
                    register_type = "Scalar"
                else:
                    continue
                if arg not in registers:
                    multi = multi_register.match(arg)
                    complete_register = arg[0] + "_" + str(counter)
                    # Handle registers written like "v[2:4]"
                    if multi:
                        sub_registers = range(
                            int(multi.group(1)), int(multi.group(2)) + 1
                        )
                        for i, val in enumerate(sub_registers):
                            registers[arg[0] + str(val)] = (
                                complete_register,
                                i,
                                register_type,
                                1,
                            )
                        registers[arg] = (
                            complete_register,
                            None,
                            register_type,
                            len(sub_registers),
                        )
                    else:
                        registers[arg] = (complete_register, None, register_type, 1)
                    counter += 1
    return registers


def find_labels(my_lines, cli_args):
    counter = 0
    labels = dict()
    found_name = False
    for my_line in utils.clean_lines(my_lines, False):
        # If line is a label, add it to the labels dictionary
        if my_line[-1] == ":":
            if cli_args.remove_name_label and not found_name:
                found_name = True
            else:
                labels[my_line[:-1]] = "label_" + str(counter)
                counter += 1
    return labels


# Convert an argument from a machine code instruction to a Register::Value
def convert_arg(arg, registers, labels):
    arg = arg.strip()
    if arg in special_registers:
        return special_registers[arg]
    elif arg in registers:
        (name, index, register_type, children) = registers[arg]
        if index is None:
            return name
        else:
            return name + "->subset({" + str(index) + "})"
    elif arg in labels:
        return labels[arg]
    else:
        return 'Register::Value::Label("' + arg + '")'


# Generate a CPP file with struct definition to generate the input machine code.
def machineCodeToInstructionList(lines, cli_args, registers, labels):
    result = """
#include <string>

#pragma GCC optimize("O0")

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include "GemmGuidePostKernels.hpp"

using namespace rocRoller;

namespace rocRollerTest
{{
    struct {name}
    {{
        ContextPtr m_context;
""".format(
        name=cli_args.function_name
    )

    result += declare_labels(labels)
    result += declare_registers(registers)

    result += """

        {name}(ContextPtr context)
            : m_context(context)
        {{
""".format(
        name=cli_args.function_name
    )

    result += define_labels(labels)
    result += define_registers(registers)

    block_count = 0
    result += """
        }}

        std::vector<Instruction> block{}()
        {{

            // clang-format off
return {{
""".format(
        block_count
    )

    line_count = 0
    in_macro = False
    for line in lines:
        in_macro, new_instruction = handleLine(in_macro, line, registers, labels)
        if new_instruction:
            result += new_instruction + ",\n"
            line_count += 1
        if line_count >= 100:
            block_count += 1
            line_count = 0
            result += """
}};
            // clang-format on
        }}

        std::vector<Instruction> block{}()
        {{
            // clang-format off
return {{
""".format(
                block_count
            )

    result += """
}};
            // clang-format on
        }}
    }};

    Generator<Instruction> {name}_Program(ContextPtr context)
    {{
        {name} gen(context);
""".format(
        name=cli_args.function_name
    )

    result += allocate_registers(cli_args.function_name, registers)

    for i in range(block_count):
        result += (
            "        for(auto const& inst : gen.block{}()) co_yield inst;\n".format(i)
        )

    result += """
    }
}
"""
    return result


# Convert machine code to a sequence of co_yield statements.
def machineCodeToCoYields(lines, cli_args, registers, labels):
    result = ""
    result += initialize_labels(labels)
    result += initialize_registers(registers)
    in_macro = False
    for my_line in lines:
        in_macro, new_instruction = handleLine(in_macro, my_line, registers, labels)
        result += "co_yield_(" + new_instruction + ");\n"
    return result


# Convert a single line of machine code to an instruction.
def handleLine(in_macro, my_line, registers, labels):  # noqa: C901
    if in_macro or my_line.startswith(".macro"):
        in_macro = True
        new_instruction = 'Instruction("' + my_line + '", {}, {}, {}, "")'
        if my_line == ".endm":
            in_macro = False
    elif my_line.startswith(".set"):
        new_instruction = 'Instruction("' + my_line + '", {}, {}, {}, "")'
    elif not my_line or my_line.startswith("."):
        return in_macro, ""
    else:
        if "//" in my_line:
            code = my_line[0 : my_line.find("//")].strip()
            comment = my_line[my_line.find("//") + 2 :]
        elif "/*" in my_line:
            code = utils.remove_between(my_line, "/*", "*/").strip()
            comment = utils.get_between(my_line, "/*", "*/")
        else:
            code = my_line
            comment = ""
        if len(code) == 0:
            if comment:
                new_instruction = 'Instruction::Comment("' + comment + '")'
            else:
                return in_macro, ""
        elif code[-1] == ":":
            if code[:-1] in labels:
                new_instruction = "Instruction::Label(" + labels[code[:-1]] + ")"
            else:
                new_instruction = 'Instruction::Comment("' + code + comment + '")'
        else:
            line_split = code.split(" ", 1)
            instruction = line_split[0]
            new_instruction = 'Instruction("' + instruction + '", '
            if len(line_split) > 1:
                args = line_split[1]
                split_args = args.split(",")
                converted_args = [
                    convert_arg(arg, registers, labels) for arg in split_args
                ]
                converted_modifiers = ['"{}"'.format(arg) for arg in split_args]
                # If a label is the first argument, there will be no "dest" argument
                if converted_args[0] in labels.values():
                    new_instruction += "{{}}, {{{}}}, {{{}}}, ".format(
                        ", ".join(converted_args[0:4]),
                        ", ".join(converted_modifiers[4:]),
                    )
                else:
                    new_instruction += "{{{}}}, {{{}}}, {{{}}}, ".format(
                        converted_args[0],
                        ", ".join(converted_args[1:5]),
                        ", ".join(converted_modifiers[5:]),
                    )
            else:
                new_instruction += "{}, {}, {}, "
            new_instruction += '"' + comment + '")'
    return in_macro, new_instruction


# Take a file containing AMD GPU machine code and print out C++ code using the
# Instruction class.
def machineCodeToInstructions(inputFile, cli_args):
    with open(inputFile) as f:
        my_lines = f.readlines()
        my_lines = utils.clean_lines(my_lines, cli_args.leave_comments)
        result = ""

        labels = dict()
        if not cli_args.ignore_labels:
            labels = find_labels(my_lines, cli_args)

        registers = dict()
        if not cli_args.ignore_registers:
            registers = find_registers(my_lines)

        if cli_args.instruction_list:
            result += machineCodeToInstructionList(
                my_lines, cli_args, registers, labels
            )
        else:
            result += machineCodeToCoYields(my_lines, cli_args, registers, labels)

        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "input_file", type=str, default="a.s", help="File with AMD GPU machine code"
    )
    parser.add_argument(
        "--ignore_labels", action="store_true", help="Skip label detection."
    )
    parser.add_argument(
        "--ignore_registers", action="store_true", help="Skip register detection."
    )
    parser.add_argument(
        "--leave_comments", action="store_true", help="Dont remove comments."
    )
    parser.add_argument(
        "--instruction_list",
        action="store_true",
        help="Convert to list of instructions.",
    )
    parser.add_argument(
        "--function_name",
        default="Program",
        help="The name of the generated C++ function",
    )
    parser.add_argument(
        "--remove_name_label",
        action="store_true",
        help="Ignore the program name label.",
    )
    args = parser.parse_args()

    machineCodeToInstructions(args.input_file, args)
