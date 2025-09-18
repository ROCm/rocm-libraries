################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import pytest
import subprocess
from collections import namedtuple

from YamlBuilder.YamlBuilder import YamlBuilder
from YamlBuilder.YamlBuilder import Solutions

TestConfig=namedtuple("TestConfig", ["solution", "problem_func"])
Runner=namedtuple("Runner", ["level", "func", "solution"])
TensileState=namedtuple("TensileState", ["args"])

# this control the default solutions used for each test.
solutions = ["src1", "src5_gsu", "asm3_pbd", "asm3_splitu", "asm3_mi"]
#solutions = ["asm3_pbd"] # run a smaller set of tests

@pytest.fixture(scope="function")
def file_with_test_name(request, tmp_path):
    def get(suffix):
        filename = request.node.name + suffix
        return tmp_path.joinpath(filename)
    return get

@pytest.fixture
def run_nothing():
    def run(conv, problemType, solution=Solutions.defaultSolution(), problemFunc=None, problemLevel=-1, dataType='s'):
        pass
    return run

@pytest.fixture
def run_generate_yaml(request, file_with_test_name):
    def run(conv, problemType, solution=Solutions.defaultSolution(), problemFunc=None, problemLevel=-1, dataType='s'):
        if problemFunc == None:
            problemFunc = YamlBuilder.ProblemSizes
        if problemLevel==-1:
            problemLevel = request.config.getoption("--problem-level")
        config = YamlBuilder.ConvolutionContraction(conv, problemType, solution, dataType, problemFunc, True, problemLevel)
        configFile = file_with_test_name(".contraction.yaml")
        print("Generate_YAML output:", configFile)
        config.write(configFile)
        return configFile
    return run

@pytest.fixture
def run_contraction(tensile_args, tmp_path, run_generate_yaml, request, tensile_script_path):
    def run(conv, problemType, solution, problemFunc=None, problemLevel=-1, dataType='s'):
        configFile = run_generate_yaml(conv, problemType, solution, problemFunc, problemLevel, dataType)
        args = [str(configFile), str(tmp_path), *tensile_args]
        # change to use  check_output to force windows cmd block util command finish
        subprocess.check_output([tensile_script_path] + args, stderr=subprocess.STDOUT)
    return run

@pytest.fixture
def run_convolution_vs_contraction(tensile_args, tmp_path, file_with_test_name, tensile_script_path):
    def run(conv, problemType={}, solution=Solutions.src1, dataType='s'):
        config = YamlBuilder.ConvolutionVsContraction(conv, solution, dataType)
        configFile = file_with_test_name(".conv.yaml")
        config.write(configFile)

        args = [str(configFile), str(tmp_path), *tensile_args]
        # change to use  check_output to force windows cmd block util command finish
        subprocess.check_output([tensile_script_path] + args, stderr=subprocess.STDOUT)
    return run

level_params = [pytest.param((0, Solutions.src1), id="Convolution_Class"),
                pytest.param((1, Solutions.defaultSolution()), id="Generate_YAML-" + Solutions.defaultSolution().__name__)] + \
               [pytest.param((2, getattr(Solutions,s)), id="Run_Contraction-"+s) for s in solutions]
               #[pytest.param((3, Solutions.defaultSolution()), id="Run_Convolution_vs_Contraction:" + Solutions.defaultSolution().__name__)]

@pytest.fixture(params=level_params)
def run_convolution_level(request,
                          pytestconfig,
                          run_nothing,
                          run_generate_yaml,
                          run_contraction,
                          run_convolution_vs_contraction):
    level_fixtures = [run_nothing,
                      run_generate_yaml,
                      run_contraction,
                      run_convolution_vs_contraction]
    curLevel = request.param[0]
    argLevel = request.config.getoption("--test-level")
    if curLevel > argLevel:
        pytest.skip()
    return Runner(curLevel, level_fixtures[curLevel], request.param[1])

def pytest_addoption(parser):
    parser.addoption(
        "--no-conv-assertions", action="store_true", default=1,
        help='''
        Disable assertiong on Convolution class.
        ''')

    parser.addoption(
        "--test-level", action="store", type=int, default=2,
        help='''
        0= test Tensile.Convolution class;
        1= 0 plus generate YAML;
        2= 1 plus run tensile client and compare generated contraction vs CPU;
        3= 2 plus run tensile_client with convolution-vs-contraction (only forward conv tests which define Spatial will PASS )
        '''
        )
    parser.addoption(
        "--problem-level", action="store", type=int, default=2,
        help='''
        How many exact configurations to generate for contraction testing for non-src solutions (typically asm).
        1= single problem
        2= tens of problems
        3= hundreds of problems
        '''
        )

@pytest.fixture()
def tensile_state(request):
    """
    Shared tensile state, including args.
    """
    args={}
    try:
        args["no_conv_assertions"]= request.config.getoption('--no-conv-assertions')
    except ValueError:
        args["no_conv_assertions"]=1
    return TensileState(args=args)
