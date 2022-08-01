#!/usr/bin/env python

import argparse
import itertools
import re
import utils

"""
This script converts a file containing AMD GPU machine code to C++ code using the
Instruction class. It accepts a file name as an argument. This file should
contain AMD GPU machine code. The generated C++ code will be printed to
standard output.

This does not handle all possible cases. Notably, it does not handle the case where:
v[2:5] is used somewhere in the file and v[2:3] is also used somewhere in the file.
It DOES handle the case where:
v[2:5] is used somewhere in the file and v2 and v3 are also used somewhere in the file.
"""

special_registers = {"vcc" : "m_context->getVCC()",
                     "exec" : "m_context->getExec()"}

# Generate C++ code for initializing all Registers that will be used
def initialize_register(register):
    (name, index, register_type, children) = register
    if index is not None:
        return ""
    result = "auto " + name +  " = std::make_shared<Register::Value>(m_context, Register::Type::" + register_type + ", DataType::Int32, " + str(children) + ");\n"
    if children > 1:
        result += "co_yield " + name + "->allocate();\n"
    return result

# Generate C++ code for initializing all Labels that will be used
def initialize_label(label, value):
    return "auto " + value + " = m_context->labelAllocator()->label(\"" + label + "\");\n"

# Find all of the registers and labels within a file and create
# dictionaries to map between registers and labels within machine code
# to values of type Register::Value
def find_registers(my_lines):
    multi_register = re.compile(r"[sv]\[(\d+):(\d+)\]")
    counter = 0
    registers = dict()
    for my_line in my_lines:
        line_split = my_line.split(" ", 1)
        instruction = line_split[0]
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
                if not arg in registers:
                    multi = multi_register.match(arg)
                    complete_register = arg[0] + "_" + str(counter)
                    # Handle registers written like "v[2:4]"
                    if multi:
                        sub_registers = range(int(multi.group(1)), int(multi.group(2)) + 1)
                        for i, val in enumerate(sub_registers):
                            registers[arg[0] + str(val)] = (complete_register, i, register_type, 1)
                        registers[arg] = (complete_register, None, register_type, len(sub_registers))
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
        if my_line[-1] == ':':
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
        return "Register::Value::Special(\"" + arg + "\")"

# Take a file containing AMD GPU machine code and print out C++ code using the
# Instruction class.
def machineCodeToInstructions(inputFile, cli_args):
    with open(inputFile) as f:
        my_lines = f.readlines()
        my_lines = utils.clean_lines(my_lines, cli_args.leave_comments)
        result = ""

        if cli_args.instruction_list:
            result += """
#pragma once

#include <string>

#include <rocRoller/Context.hpp>
#include <rocRoller/LabelAllocator.hpp>
#include <rocRoller/Instruction.hpp>
#include <rocRoller/Register.hpp>

using namespace rocRoller;

inline std::vector<Instruction> {name}(std::shared_ptr<rocRoller::Context> m_context){{
""".format(name = cli_args.function_name)

        labels = dict()
        if not cli_args.ignore_labels:
            labels = find_labels(my_lines, cli_args)

            for label in labels:
               result += initialize_label(label, labels[label])
            result += "\n"

        registers = dict()
        if not cli_args.ignore_registers:
            registers = find_registers(my_lines)

            for register in registers.values():
               result += initialize_register(register)
            result += "\n"

        if cli_args.instruction_list:
            result += "// clang-format off\n"
            result += "std::vector<Instruction> rv = {\n"

        in_macro = False

        for my_line in my_lines:
            if in_macro or my_line.startswith(".macro"):
                in_macro = True
                new_instruction = "Instruction(\"" + my_line + "\", {}, {}, {}, \"\")"
                if my_line == ".endm":
                    in_macro = False
            elif my_line.startswith(".set"):
                new_instruction = "Instruction(\"" + my_line + "\", {}, {}, {}, \"\")"
            elif not my_line or my_line.startswith('.'):
                continue
            #elif cli_args.direct_translate:
            #    new_instruction = "Instruction(\"" + my_line + "\", {}, {}, {}, \"\")"
            else:
                if "//" in my_line:
                    code = my_line[0:my_line.find("//")].strip()
                    comment = my_line[my_line.find("//") + 2:]
                elif "/*" in my_line:
                    code = utils.remove_between(my_line, "/*", "*/").strip()
                    comment = utils.get_between(my_line, "/*", "*/")
                else:
                    code = my_line
                    comment = ""
                if len(code) == 0:
                    if comment:
                        new_instruction = "Instruction::Comment(\"" + comment + "\")"
                    else:
                        continue
                elif code[-1] == ":":
                    if code[:-1] in labels:
                        new_instruction = "Instruction::Label(" + labels[code[:-1]] + ")"
                    else:
                        new_instruction = "Instruction::Comment(\"" + code + comment + "\")"
                else:
                    line_split = code.split(" ", 1)
                    instruction = line_split[0]
                    new_instruction = "Instruction(\"" + instruction + "\", "
                    if len(line_split) > 1:
                        args = line_split[1]
                        converted_args = [convert_arg(arg, registers, labels) for arg in args.split(",")]
                        # If a label is the first argument, there will be no "destination" argument
                        if converted_args[0] in labels.values():
                            new_instruction += "{}, {"
                            new_instruction += ", ".join(converted_args)
                            new_instruction += "}, "
                        else:
                            new_instruction += "{" + converted_args[0] + "}, {"
                            if len(converted_args) > 1:
                                new_instruction += ", ".join(converted_args[1:])
                            new_instruction += "}, "
                    else:
                        new_instruction += "{}, {}, "
                    new_instruction += "{}, \"" + comment + "\")"
            if cli_args.instruction_list:
                result += new_instruction + ",\n"
            else:
                result += "co_yield_(" + new_instruction + ");\n"

        if cli_args.instruction_list:
            result += "};\n"
            result += "// clang-format on\n"
            result += "return rv;\n"
            result += "}\n"

        print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert AMD GPU machine code to C++ code using the Instruction class')
    parser.add_argument('input_file', type=str, default='a.s', help='File with AMD GPU machine code')
    parser.add_argument('--ignore_labels', action="store_true", help='Skip label detection.')
    parser.add_argument('--ignore_registers', action="store_true", help='Skip register detection.')
    parser.add_argument('--leave_comments', action="store_true", help='Dont remove comments.')
    parser.add_argument('--instruction_list', action="store_true", help='Convert to list of instructions instead of co_yields.')
    parser.add_argument('--function_name', default="Program", help='The name of the generated C++ function')
    parser.add_argument('--remove_name_label', action="store_true", help='Ignore the program name label.')
    args = parser.parse_args()

    machineCodeToInstructions(args.input_file, args)
