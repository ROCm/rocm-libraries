# Copyright (c) 2018 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import getopt
import sys
import subprocess
from subprocess import call
import datetime

now = datetime.datetime.now()
today = str(now.year) + '-' + str(now.month) + '-' + str(now.day)

opts, args = getopt.getopt(sys.argv[1:], 'n:f:a:t:c:p:o:')

#Get VGG16 Caffe model
cmd = 'wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
if os.system(cmd) != 0:
    print('ERROR: Could not download VGG16 Caffe model. Exiting\n')
    exit()

dumpFile = 'vgg16-perf-dump.txt'
network = 'vgg16'
FP16 = 0
argmaxSwitch = 0
testMode = 0
runMode = 0 #0 is default, 1 is with fuse-ops enabled
pythonMode = 0
outputDirFlag = 0
precision_str = 'FP32'

for opt, arg in opts:
    if opt =='-n':
    	network = arg
    elif opt == '-f':
    	FP16 = int(arg)
    elif opt == '-a':
    	argmaxSwitch = int(arg)
    elif opt == '-t':
    	testMode = int(arg)
    elif opt == '-c':
        runMode = int(arg)
    elif opt == '-p':
        pythonMode = int(arg)
    elif opt == '-o':
        outputDirFlag = int(arg)

if outputDirFlag == 1:

    outputDirName = 'output-inference-tests-' + today
    fullpath_outputDirName = outputDirName
    if os.path.exists(fullpath_outputDirName) == 0:
        cmd = 'mkdir ' + fullpath_outputDirName
        if os.system(cmd) != 0:
            print('ERROR: Failed to make dir named ' + fullpath_outputDirName + '. Exiting.\n')

if testMode == 0:
	batchsizes = [1,64]
elif testMode > 0:
	batchsizes = [1]
	
if FP16 == 1:
    dumpFile = network + '-FP16'
    summaryFile = network + '-fp16'
    testSummaryDir = network + '-fp16'
    precision_str = 'FP16'
else:
    dumpFile = network + '-FP32'
    summaryFile = network + '-fp32'
    testSummaryDir = network + '-fp32'

if pythonMode == 1:
    dumpFile = dumpFile + '-python'
    summaryFile = summaryFile + '-python'
    testSummaryDir = testSummaryDir + '-python'

if runMode == 1:
    dumpFile = dumpFile + '-fuseOn-perf-dump.txt'
    summaryFile = summaryFile + '-fuseOn-perf-summary.csv'
    testSummaryDir = testSummaryDir + '-fuseOn-' + today
else:
    dumpFile = dumpFile + '-perf-dump.txt'
    summaryFile = summaryFile + '-perf-summary.csv'
    testSummaryDir = testSummaryDir + '-' + today

if os.path.exists(testSummaryDir):
    cmd = 'rm -rf ' + testSummaryDir
    if os.system(cmd) != 0:
        print('ERROR: Unable to delete directory ' + testSummaryDir + '. Exiting.\n')
cmd = 'mkdir ' + testSummaryDir
if os.system(cmd) != 0:
    print('ERROR: Unable to create directory ' + testSummaryDir + '. Exiting.\n')

fp = open(dumpFile, 'w')
fp.write(today + "\n")
fp.close()

for batchsize in batchsizes:
    cmd = 'python3 inference-performance-accuracy.py -n ' + network + ' -b ' + str(batchsize) + ' -f ' + str(FP16) + ' -c ' + str(runMode) + ' -a ' + str(argmaxSwitch) + ' -p ' + str(pythonMode) + ' >> ' + dumpFile
    if os.system(cmd) != 0:
        print('ERROR: inference-performance-accuracy.py failed to run. Exiting.\n')
        exit()
    print('RAN (Caffe): ' + network + ', precision - ' + precision_str + ', batch size - ' + str(batchsize) + '\n')

cmd = 'python3 inference-parse-perf-dump.py ' + dumpFile + ' > ' + summaryFile
if os.system(cmd) != 0:
    print('ERROR: inference-parse-perf-dump.py failed to run. Exiting.\n')
    exit()

cmd = 'mv ' + dumpFile + ' ' + testSummaryDir
if os.system(cmd) != 0:
    print('ERROR: Could not move ' + dumpFile + ' into ' + testSummaryDir + '\n')
cmd = 'mv ' + summaryFile + ' ' + testSummaryDir
if os.system(cmd) != 0:
    print('ERROR: Could not move ' + summaryFile + ' into ' + testSummaryDir + '\n')
    
for batchsize in batchsizes:
    if FP16 == 1:
        currentTensor = network + '-' + str(batchsize) + '-fp16'
        if pythonMode == 1:
            currentTensor = currentTensor + '-python'
        currentTensor = currentTensor + '.fp'
        failToSplit = 0
        if batchsize > 1:
            cmd = './split-tensor-fp16 ' + currentTensor + ' ' + str(batchsize)
            if os.system(cmd) != 0:
                print('ERROR: Could not split tensor ' + currentTensor + '\n')
                failToSplit = 1
            if failToSplit == 0:
                splitTensorDir = network + '-' + str(batchsize) + '-fp16'
                cmd = 'mkdir ' + splitTensorDir
                if os.system(cmd) != 0:
                    print('ERROR: Could not make dir: ' + splitTensorDir + '\n')
                i = 0
                while i < batchsize:
                    if pythonMode == 0:
                        currentSplitTensor = network + '-' + str(batchsize) + '-fp16-' + str(i) + '.fp'
                    elif pythonMode == 1:
                        currentSplitTensor = network + '-' + str(batchsize) + '-fp16-python-' + str(i) + '.fp'
                    cmd = 'mv ' + currentSplitTensor + ' ' + splitTensorDir
                    if os.system(cmd) != 0:
                        print('ERROR: Could not move ' + currentSplitTensor + ' into ' + splitTensorDir + '\n')
                    i = i + 1
                cmd = 'mv ' + splitTensorDir + ' ' + testSummaryDir
                if os.system(cmd) != 0:
                    print('ERROR: Could not move ' + splitTensorDir + ' into ' + testSummaryDir + '\n')
    else:
        currentTensor = network + '-' + str(batchsize)
        if pythonMode == 1:
            currentTensor = currentTensor + '-python'
        currentTensor = currentTensor + '.fp'
        failToSplit = 0
        if batchsize > 1:
            cmd = './split-tensor ' + currentTensor + ' ' + str(batchsize)
            if os.system(cmd) != 0:
                print('ERROR: Could not split tensor ' + currentTensor + '\n')
                failToSplit = 1
            if failToSplit == 0:
                splitTensorDir = network + '-' + str(batchsize)
                cmd = 'mkdir ' + splitTensorDir
                if os.system(cmd) != 0:
                    print('ERROR: Could not make dir: ' + splitTensorDir + '\n')
                i = 0
                while i < batchsize:
                    if pythonMode == 0:
                        currentSplitTensor = network + '-' + str(batchsize) + '-' + str(i) + '.fp'
                    elif pythonMode == 1:
                        currentSplitTensor = network + '-' + str(batchsize) + '-python-' + str(i) + '.fp'
                    cmd = 'mv ' + currentSplitTensor + ' ' + splitTensorDir
                    if os.system(cmd) != 0:
                        print('ERROR: Could not move ' + currentSplitTensor + ' into ' + splitTensorDir + '\n')
                    i = i + 1
                cmd = 'mv ' + splitTensorDir + ' ' + testSummaryDir
                if os.system(cmd) != 0:
                    print('ERROR: Could not move ' + splitTensorDir + ' into ' + testSummaryDir + '\n')
        
    cmd = 'mv ' + currentTensor + ' ' + testSummaryDir
    if os.system(cmd) != 0:
        print('ERROR: Could not move ' + currentTensor + ' into ' + testSummaryDir + '\n')
    
#Run accuracy script
if pythonMode == 1:
    cmd = 'python3 inference-measure-accuracy.py -d ' + testSummaryDir + ' -p 1' 
    if os.system(cmd) != 0:
        print('ERROR: Could not run inference-measure-accuracy.py on ' + testSummaryDir + ' folder.\n')
else:
    cmd = 'python3 inference-measure-accuracy.py -d ' + testSummaryDir
    if os.system(cmd) != 0:
        print('ERROR: Could not run inference-measure-accuracy.py on ' + testSummaryDir + ' folder.\n')

#Copy to output dir
if outputDirFlag == 1:
    cmd = 'cp -r ' + testSummaryDir + ' ' + fullpath_outputDirName
    if os.system(cmd) != 0:
        print('ERROR: Could not copy ' + testSummaryDir + ' to ' + fullpath_outputDirName + '\n')
