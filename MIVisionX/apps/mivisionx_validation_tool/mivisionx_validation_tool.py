__author__      = "Kiriti Nagesh Gowda"
__copyright__   = "Copyright 2019, AMD MIVisionX"
__credits__     = ["Hansel Yang; Lakshmi Kumar;"]
__license__     = "MIT"
__version__     = "0.9.5"
__maintainer__  = "Kiriti Nagesh Gowda"
__email__       = "mivisionx.support@amd.com"
__status__      = "ALPHA"
__script_name__ = "MIVisionX Validation Tool"

import sys
import argparse
from PyQt5 import QtWidgets
from inference_control import *

# MIVisionX Classifier
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    if len(sys.argv) == 1:
        panel = InferenceControl()
        panel.show()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_format',		type=str, required=True,	help='pre-trained model format, options:caffe/onnx/nnef [required]')
        parser.add_argument('--model_name',			type=str, required=True,	help='model name                             [required]')
        parser.add_argument('--model',				type=str, required=True,	help='pre_trained model file/folder          [required]')
        parser.add_argument('--model_batch_size',	type=str, required=True,	help='n - batch size			             [required]')
        parser.add_argument('--rocal_mode',			type=str, required=True,	help='rocal mode (1/2/3)			             [required]')
        parser.add_argument('--model_input_dims',	type=str, required=True,	help='c,h,w - channel,height,width           [required]')
        parser.add_argument('--model_output_dims',	type=str, required=True,	help='c,h,w - channel,height,width           [required]')
        parser.add_argument('--label',				type=str, required=True,	help='labels text file                       [required]')
        parser.add_argument('--output_dir',			type=str, required=True,	help='output dir to store ADAT results       [required]')
        parser.add_argument('--image_dir',			type=str, required=True,	help='image directory for analysis           [required]')
        parser.add_argument('--image_val',			type=str, default='',		help='image list with ground truth           [optional]')
        parser.add_argument('--hierarchy',			type=str, default='',		help='AMD proprietary hierarchical file      [optional]')
        parser.add_argument('--add',				type=str, default='', 		help='input preprocessing factor [optional - default:[0,0,0]]')
        parser.add_argument('--multiply',			type=str, default='',		help='input preprocessing factor [optional - default:[1,1,1]]')
        parser.add_argument('--fp16',				type=str, default='no',		help='quantize to FP16 			[optional - default:no]')
        parser.add_argument('--replace',			type=str, default='no',		help='replace/overwrite model   [optional - default:no]')
        parser.add_argument('--verbose',			type=str, default='no',		help='verbose                   [optional - default:no]')
        parser.add_argument('--loop',				type=str, default='yes',	help='verbose                   [optional - default:yes]')
        parser.add_argument('--gui',				type=str, default='yes',	help='verbose                   [optional - default:yes]')
        parser.add_argument('--cpu_name',			type=str, default='epyc',	help='verbose                   [optional - default:epyc]')
        parser.add_argument('--gpu_name',			type=str, default='vega20',	help='verbose                   [optional - default:vega20]')
        parser.add_argument('--fps_file',			type=str, default='',		help='verbose                   [optional]')
        args = parser.parse_args()
        
        # get arguments
        modelFormat = args.model_format
        modelName = args.model_name
        modelLocation = args.model
        modelBatchSize = args.model_batch_size
        rocalMode = (int)(args.rocal_mode)
        modelInputDims = args.model_input_dims
        modelOutputDims = args.model_output_dims
        label = args.label
        outputDir = args.output_dir
        imageDir = args.image_dir
        imageVal = args.image_val
        hierarchy = args.hierarchy
        inputAdd = args.add
        inputMultiply = args.multiply
        fp16 = args.fp16
        replaceModel = args.replace
        verbose = args.verbose
        loop = args.loop
        gui = args.gui
        fps_file = args.fps_file
        cpu_name = args.cpu_name
        gpu_name = args.gpu_name
        container_logo = 0

        viewer = InferenceViewer(modelName, modelFormat, imageDir, modelLocation, label, hierarchy, imageVal, modelInputDims, modelOutputDims, 
                                    modelBatchSize, outputDir, inputAdd, inputMultiply, verbose, fp16, replaceModel, loop, rocalMode, gui, container_logo, fps_file, cpu_name,
                                    gpu_name, parent=None)
    app.exec_()