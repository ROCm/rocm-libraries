#!/usr/bin/python
# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

import itertools
import random
import sys
import getopt
import yaml
import os

class KernelArguments:
    SolutionNameMin = ""
    DataType = 4
    DestDataType = 4
    ComputeDataType = 0
    TransposeA = False
    TransposeB = False
    WorkGroup = [1, 1, 1]
    ThreadTile = [1, 1, 1]
    MacroTile = [1, 1, 0]
    StaggerU = 0
    DepthU = 32
    GlobalSplitU = 0
    StaggerStrideShift =0
    WorkGroupMapping = 0
    PackBatchDims = 0
    UseInitialStridesAB = False
    UseInitialStridesCD = False
    ActivationFused = False
    GlobalAccumulation = 0
    Activation = False
    ActivationHPA = False
    ActivationType = ""

def writefile(filename, kernel_maps):
    with open(filename, 'w') as f:
        try:
            f.write("#include <map>\n")
            f.write("#include <vector>\n")
            f.write("#include <string>\n")
            f.write("#include \"hip/hip_runtime.h\"\n")
            f.write("\n")
            f.write("std::map<std::string, int> kernel_count = \n{\n")
            count_keys = len(kernel_maps.keys())
            for key in kernel_maps.keys():
                count = len(kernel_maps[key])
                count_keys  = count_keys - 1
                if count_keys == 0:
                    f.write("    {}\"{}\", {}{}\n".format("{", key, count, "}"))
                else:
                    f.write("    {}\"{}\", {}{},\n".format("{", key, count, "}"))

            f.write("};\n")
            f.write("extern \"C\" size_t get_kernel_counts(const char* name)\n")
            f.write("{\n")
            f.write("    auto it = kernel_count.find(name);\n")
            f.write("    if(it != kernel_count.end())\n")
            f.write("        return it->second;\n")
            f.write("    else\n        return 0;\n")
            f.write("};\n")

            f.write("struct KernelParams\n")
            f.write("{\n")
            f.write("    char SolutionNameMin[256];\n")
            f.write("    int DataType;\n")
            f.write("    int DestDataType;\n")
            f.write("    int ComputeDataType;\n")
            f.write("    bool TransposeA;\n")
            f.write("    bool TransposeB;\n")
            f.write("    unsigned int WorkGroup[3];\n")
            f.write("    unsigned int ThreadTile[3];\n")
            f.write("    unsigned int MacroTile[3];\n")
            f.write("    size_t StaggerU;\n")
            f.write("    size_t DepthU;\n")
            f.write("    size_t GlobalSplitU;\n")
            f.write("    size_t StaggerStrideShift;\n")
            f.write("    int WorkGroupMapping;\n")
            f.write("    size_t PackBatchDims;\n")
            f.write("    bool UseInitialStridesAB;\n")
            f.write("    bool UseInitialStridesCD;\n")
            f.write("    bool ActivationFused;\n")
            f.write("    int GlobalAccumulation;\n")
            f.write("    bool Activation;\n")
            f.write("    bool ActivationHPA;\n")
            f.write("    char ActivationType[32];\n")
            f.write("};\n")

            f.write("std::map<std::string, std::vector<KernelParams>> kernel_params = \n{\n")
            count_keys = len(kernel_maps.keys())
            for key in kernel_maps.keys():
                f.write("{}\"{}\", {}".format("{", key, "{"))
                for ka in kernel_maps[key]:
                    wg = "{} {}, {}, {}{}".format("{", ka.WorkGroup[0], ka.WorkGroup[1], ka.WorkGroup[2], "}")
                    tt = "{} {}, {}, {}{}".format("{", ka.ThreadTile[0], ka.ThreadTile[1], ka.ThreadTile[2], "}")
                    mt = "{} {}, {}, {}{}".format("{", ka.MacroTile[0], ka.MacroTile[1], ka.MacroTile[2], "}")
                    values = "\"{}\", {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, \"{}\"".format(
                            ka.SolutionNameMin, ka.DataType, ka.DestDataType, ka.ComputeDataType,
                            "true" if ka.TransposeA else "false", "true" if ka.TransposeB else "false",
                            wg, tt, mt,
                            ka.StaggerU, ka.DepthU, ka.GlobalSplitU, ka.StaggerStrideShift, ka.WorkGroupMapping, ka.PackBatchDims,
                            "true" if ka.UseInitialStridesA else "false", "true" if ka.UseInitialStridesCD else "false",
                            "true" if ka.ActivationFused else "false", 0 if not ka.GlobalAccumulation else ka.GlobalAccumulation,
                            "true" if ka.Activation else "false", "true" if ka.ActivationHPA else "false", ka.ActivationType)
                    f.write("{}{}{},\n".format("{", values, "}"))
                f.write("}},\n")
            f.write("};\n")
            f.write("extern \"C\" KernelParams* get_kernel_params(const char* name)\n")
            f.write("{\n")
            f.write("    auto it = kernel_params.find(name);\n")
            f.write("    if(it != kernel_params.end())\n")
            f.write("        return it->second.data();\n")
            f.write("    else\n        return NULL;\n")
            f.write("};\n")
        except Exception as e:
            print(e)
        f.close()

def main(args):

    kernel_maps={}

    #float, half, int, bf16, int8
    dataTypes = [0, 4, 6, 7, 8]

    (opts, rem) = getopt.getopt(args, '', ['filename=', 'yaml=', 'v'])
    optDict = dict(opts)
    filename    = optDict.get('--filename', '')

    if '--yaml' in optDict:
        path = optDict['--yaml']
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for file_name in files:
            file_name_u = file_name.upper()
            if ".YAML" not in file_name_u :
                continue
            with open(os.path.join(path, file_name), 'r') as f:
                try:
                    contents_a = yaml.safe_load(f)
                    contents4 = contents_a[4]
                    contents5 = contents_a[5]
                    for c_index in range(0, len(contents5)):
                        contents_p = contents5[c_index].get('ProblemType')
                        ka = KernelArguments()
                        #append K1 (Kernel True) at the buttom for single-source compilation
                        #TODO do not appened this key word.
                        ka.SolutionNameMin = contents5[c_index].get('SolutionNameMin') + "K1"
                        ka.DataType = contents4.get('DataType')
                        ka.DestDataType = contents4.get('DestDataType')
                        ka.ComputeDataType = contents4.get('ComputeDataType')
                        ka.TransposeA = contents4.get('TransposeA')
                        ka.TransposeB = contents4.get('TransposeB')
                        ka.WorkGroup = contents5[c_index].get('WorkGroup')
                        ka.ThreadTile[0] = contents5[c_index].get('ThreadTile')[0]
                        ka.ThreadTile[1] = contents5[c_index].get('ThreadTile')[1]
                        ka.MacroTile[0] = contents5[c_index].get('MacroTile0')
                        ka.MacroTile[1] = contents5[c_index].get('MacroTile1')
                        ka.StaggerU = contents5[c_index].get('StaggerU')
                        ka.DepthU = contents5[c_index].get('DepthU')
                        ka.GlobalSplitU = contents5[c_index].get('GlobalSplitU')
                        ka.StaggerStrideShift = contents5[c_index].get('_staggerStrideShift')
                        ka.WorkGroupMapping = contents5[c_index].get('WorkGroupMapping')
                        ka.PackBatchDims = contents5[c_index].get('PackBatchDims')
                        ka.UseInitialStridesA = contents4.get('UseInitialStridesAB')
                        ka.UseInitialStridesC = contents4.get('UseInitialStridesCD')
                        ka.ActivationFused = contents5[c_index].get('ActivationFused')
                        ka.GlobalAccumulation = contents5[c_index].get('_GlobalAccumulation')
                        ka.Activation = contents_p.get('Activation')
                        ka.ActivationHPA = contents_p.get('ActivationHPA')
                        ka.ActivationType = contents_p.get('ActivationType')

                        if '--v' in optDict:
                            print("SolutionNameMin=", ka.SolutionNameMin)
                            print("DataType=", ka.DataType)
                            print("DestDataType=", ka.DestDataType)
                            print("ComputeDataType=", ka.ComputeDataType)
                            print("TransposeA=", ka.TransposeA)
                            print("TransposeB=", ka.TransposeB)
                            print("WorkGroup=", ka.WorkGroup)
                            print("ThreadTile=", ka.ThreadTile)
                            print("MacroTile=", ka.MacroTile)
                            print("StaggerU=", ka.StaggerU)
                            print("DepthU=", ka.DepthU)
                            print("GlobalSplitU=", ka.GlobalSplitU)
                            print("StaggerStrideShift=", ka.StaggerStrideShift)
                            print("WorkGroupMapping=", ka.WorkGroupMapping)
                            print("PackBatchDims=", ka.PackBatchDims)
                            print("UseInitialStridesA=", ka.UseInitialStridesA)
                            print("UseInitialStridesC=", ka.UseInitialStridesC)
                            print("ActivationFused=", ka.ActivationFused)
                            print("GlobalAccumulation=", ka.GlobalAccumulation)
                            print("Activation=", ka.Activation)
                            print("ActivationHPA=", ka.ActivationHPA)
                            print("ActivationType=", ka.ActivationType)
                        key="{}_{}_{}_{}_{}".format(ka.DataType, ka.DestDataType, ka.ComputeDataType, 'T' if ka.TransposeA else 'N', 'T' if ka.TransposeB else 'N')
                        if key in kernel_maps :
                            kernel_maps[key].append(ka)
                        else:
                            kas= [ka]
                            kernel_maps[key] = kas
                except Exception as e:
                    print('Failed to read file: {}'.format(filename))
                    print(e)
                    return

    writefile(filename, kernel_maps)

if __name__=="__main__":
    main(sys.argv[1:])
