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

opts, args = getopt.getopt(sys.argv[1:], 'd:m:n:b:f:s:a:c:p:')

buildDir = '../../../model_compiler/python/'
modelDir = 'VGG_ILSVRC_16_layers.caffemodel'
network = 'vgg16'
batchsize = 1
model_width = 224
model_height = 224
input_tensor = ''
fp_16 = 0
argmaxOn = 0
normalized = 0
runMode = 0
pythonMode = 0

for opt, arg in opts:
    if opt == '-d':
    	buildDir = arg
    elif opt =='-m':
    	modelDir = arg
    elif opt =='-n':
    	network = arg
    elif opt == '-b':
    	batchsize = int(arg)
    elif opt == '-f':
    	fp_16 = int(arg)
    elif opt == '-s':
    	dataset = int(arg)
    elif opt == '-a':
    	argmaxOn = int(arg)
    elif opt == '-c':
    	runMode = int(arg)
    elif opt == '-p':
    	pythonMode = int(arg)


network_folder = network
nnir_output_folder = network + '-nnir'
network_build_folder = network + '-build'

if fp_16 == 1:
	output_tensor = network + '-' + str(batchsize) + '-fp16'
	output_tensor_accuracy = network + '-' + str(batchsize) + '-fp16'
	output_accuracy_summary = network + '-' + str(batchsize) + '-fp16'

else:
	output_tensor = network + '-' + str(batchsize)
	output_tensor_accuracy = network + '-' + str(batchsize)
	output_accuracy_summary = network + '-' + str(batchsize)

if pythonMode == 1:
	output_tensor = output_tensor + '-python'
	output_tensor_accuracy = output_tensor_accuracy + '-python'
	output_accuracy_summary = output_accuracy_summary + '-python'

if runMode == 1:
	output_tensor = output_tensor + '.fp'
	output_tensor_accuracy = output_tensor_accuracy + '-fuseOn-accuracy.txt'
	output_accuracy_summary = output_accuracy_summary + '-fuseOn-accuracy-summary.txt'
else:
	output_tensor = output_tensor + '.fp'
	output_tensor_accuracy = output_tensor_accuracy + '-accuracy.txt'
	output_accuracy_summary = output_accuracy_summary + '-accuracy-summary.txt'



if fp_16 == 0 and normalized == 0:
	input_tensor = 'tensors-' + str(model_width) + 'x' + str(model_height)
elif fp_16 == 0 and normalized == 1:
	input_tensor = 'tensors-' + str(model_width) + 'x' + str(model_height) + '-norm'
elif fp_16 == 1 and normalized == 0:
	input_tensor = 'tensors-fp16-' + str(model_width) + 'x' + str(model_height)
elif fp_16 == 1 and normalized == 1:
	input_tensor = 'tensors-fp16-' + str(model_width) + 'x' + str(model_height) + '-norm'




input_tensor += '/tensor-' + str(batchsize) + '-' + str(model_width) + 'x' + str(model_height) + '.fp'


print('Network path = ' + modelDir + '\n')
print('buildDir = ' + buildDir + '\n')
print('Input_tensor = ' + input_tensor + '\n')

cmd = 'python3 ' + buildDir + 'caffe_to_nnir.py ' + modelDir + ' ' + network_folder + ' --input-dims 1,3,' + str(model_width) + ',' + str(model_height)
if os.system(cmd) != 0:
	print('ERROR: caffe2nnir.py failed to run. Exiting.\n')
	exit()

cmd = 'python3 ' + buildDir + 'nnir_update.py --batch-size ' + str(batchsize) + ' ' + network_folder + ' ' + nnir_output_folder
if os.system(cmd) != 0:
	print('ERROR: nnir_update.py failed to run. Exiting.\n')
	exit()

if runMode == 1:
	old_nnir_output_folder = nnir_output_folder
	nnir_output_folder = nnir_output_folder + '-fuseOn'
	cmd = 'python3 ' + buildDir + 'nnir_update.py --fuse-ops 1 '+ old_nnir_output_folder + ' ' + nnir_output_folder
	if os.system(cmd) != 0:
		print('ERROR: nnir_update.py failed to run. Exiting.\n')
		exit()

if fp_16 == 1:
	old_nnir_output_folder = nnir_output_folder
	nnir_output_folder = nnir_output_folder + '-FP16'
	cmd = 'python3 ' + buildDir + 'nnir_update.py --convert-fp16 1 '+ old_nnir_output_folder + ' ' + nnir_output_folder
	if os.system(cmd) != 0:
		print('ERROR: nnir_update.py failed to run. Exiting.\n')
		exit()

if argmaxOn == 0:
	cmd = 'python3 ' + buildDir + 'nnir_to_openvx.py ' + nnir_output_folder + '/ ' + network_build_folder
	if os.system(cmd) != 0:
		print('ERROR: nnir2openvx.py failed to run. Exiting\n')
		exit()
elif argmaxOn > 0:
	cmd = 'python3 ' + buildDir + 'nnir_to_openvx.py --argmax UINT16 ' + nnir_output_folder + '/ ' + network_build_folder
	if os.system(cmd) != 0:
		print('ERROR: nnir2openvx.py (WITH argmax) failed to run. Exiting\n')
		exit()

os.chdir(network_build_folder)

cmd = 'cmake .'
if os.system(cmd) != 0:
	print('ERROR: Cmake failed to run. Exiting.\n')
	exit()

cmd = 'make'
if os.system(cmd) != 0:
	print('ERROR: Failed to make anntest. Exiting.\n')
	exit()

if pythonMode == 0:
	cmd = './anntest weights.bin ../' + input_tensor + ' ../' + output_tensor
	if os.system(cmd) != 0:
		print('ERROR: Failed to run anntest. Exiting.\n')
		exit()
elif pythonMode == 1:
	cmd = 'pwd'
	pathToLib = subprocess.check_output(cmd, shell=True)
	sys.path.append(pathToLib)

	cmd = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:' + pathToLib + ' python anntest.py libannpython.so weights.bin ../' + input_tensor + ' ../' + output_tensor
	if os.system(cmd) != 0:
		print('ERROR: Failed to run anntest.py. Exiting.\n')
		exit()

if argmaxOn > 0:
	cmd = 'od -t u2 -v ../' + output_tensor + ' > ../' + output_tensor_accuracy
	if os.system(cmd) != 0:
		print('ERROR: Failed to run od on output tensor. Exiting.\n')
		exit()

	os.chdir('../')

	cmd = 'python3 mivisionx-labels.py ' + output_tensor_accuracy + ' > ' + output_accuracy_summary
	if os.system(cmd) != 0:
		print('ERROR: Failed to run mivisionx-labels.py. Exiting.\n')
		exit()
