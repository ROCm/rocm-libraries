__author__      = "Kiriti Nagesh Gowda"
__copyright__   = "Copyright 2019 - 2024, AMD MIVisionX"
__credits__     = ["Mike Schmit; Hansel Yang; Lakshmi Kumar;"]
__license__     = "MIT"
__version__     = "1.1"
__maintainer__  = "Kiriti Nagesh Gowda"
__email__       = "mivisionx.support@amd.com"
__status__      = "Shipping"
__script_name__ = "MIVisionX Inference Analyzer"

import argparse
import os
import sys
import ctypes
import cv2
import time
import numpy
import statistics
import numpy as np
from numpy.ctypeslib import ndpointer
from inference_control import inference_control
from PyQt5 import QtWidgets

# global variables
FP16inference = False
verbosePrint = False
labelNames = None
colors =[
        (0,153,0),        # Top1
        (153,153,0),      # Top2
        (153,76,0),       # Top3
        (0,128,255),      # Top4
        (255,102,102),    # Top5
        ]
# error check calls
def ERROR_CHECK(call):
    status = call
    if(status != 0):
        exit(status)

# AMD Neural Net python wrapper
class AnnAPI:
    def __init__(self,library):
        self.lib = ctypes.cdll.LoadLibrary(library)
        self.annQueryInference = self.lib.annQueryInference
        self.annQueryInference.restype = ctypes.c_char_p
        self.annQueryInference.argtypes = []
        self.annCreateInference = self.lib.annCreateInference
        self.annCreateInference.restype = ctypes.c_void_p
        self.annCreateInference.argtypes = [ctypes.c_char_p]
        self.annReleaseInference = self.lib.annReleaseInference
        self.annReleaseInference.restype = ctypes.c_int
        self.annReleaseInference.argtypes = [ctypes.c_void_p]
        self.annCopyToInferenceInput = self.lib.annCopyToInferenceInput
        self.annCopyToInferenceInput.restype = ctypes.c_int
        self.annCopyToInferenceInput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_bool]
        self.annCopyFromInferenceOutput = self.lib.annCopyFromInferenceOutput
        self.annCopyFromInferenceOutput.restype = ctypes.c_int
        self.annCopyFromInferenceOutput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
        self.annRunInference = self.lib.annRunInference
        self.annRunInference.restype = ctypes.c_int
        self.annRunInference.argtypes = [ctypes.c_void_p, ctypes.c_int]
        print('OK: AMD VX NN API found "' + self.annQueryInference().decode("utf-8") + '" as configuration in ' + library)

# classifier definition
class annieObjectWrapper():
    def __init__(self, annpythonlib, weightsfile):
        self.api = AnnAPI(annpythonlib)
        input_info,output_info,empty = self.api.annQueryInference().decode("utf-8").split(';')
        input,name,n_i,c_i,h_i,w_i = input_info.split(',')
        outputCount = output_info.split(",")
        stringcount = len(outputCount)
        if stringcount == 6:
            output,opName,n_o,c_o,h_o,w_o = output_info.split(',')
        else:
            output,opName,n_o,c_o= output_info.split(',')
            h_o = '1'; w_o  = '1'
        self.hdl = self.api.annCreateInference(weightsfile.encode('utf-8'))
        self.dim = (int(w_i),int(h_i))
        self.outputDim = (int(n_o),int(c_o),int(h_o),int(w_o))

    def __del__(self):
        self.api.annReleaseInference(self.hdl)

    def runInference(self, img, out):
        # create input.f32 file
        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]
        img_t = np.concatenate((img_r, img_g, img_b), 0)    
        # copy input f32 to inference input
        status = self.api.annCopyToInferenceInput(self.hdl, np.ascontiguousarray(img_t, dtype=np.float32), (img.shape[0]*img.shape[1]*3*4), 0)
        if(status):
                print('ERROR: AMD VX NN CopyToInferenceInput Failed ')
        # run inference
        status = self.api.annRunInference(self.hdl, 1)
        if(status):
                print('ERROR: AMD VX NN RunInference Failed ')
        # copy output f32
        status = self.api.annCopyFromInferenceOutput(self.hdl, np.ascontiguousarray(out, dtype=np.float32), out.nbytes)
        if(status):
                print('ERROR: AMD VX NN CopyFromInferenceOutput Failed ')
        return out

    def classify(self, img):
        # create output.f32 buffer
        out_buf = bytearray(self.outputDim[0]*self.outputDim[1]*self.outputDim[2]*self.outputDim[3]*4)
        out = np.frombuffer(out_buf, dtype=numpy.float32)
        # run inference & receive output
        output = self.runInference(img, out)
        return output

# process classification output function
def processClassificationOutput(inputImage, modelName, modelOutput):
    # post process output file
    start = time.time()
    softmaxOutput = np.float32(modelOutput)
    topIndex = []
    topLabels = []
    topProb = []
    for x in softmaxOutput.argsort()[-5:]:
        topIndex.append(x)
        topLabels.append(labelNames[x])
        topProb.append(softmaxOutput[x])
    end = time.time()
    if(verbosePrint):
        print('%30s' % 'Processed results in ', str((end - start)*1000), 'ms')

    # display output
    start = time.time()
    # initialize the result image
    resultImage = np.zeros((250, 525, 3), dtype="uint8")
    resultImage.fill(255)
    cv2.putText(resultImage, 'MIVisionX Object Classification', (25,  25),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    topK = 1   
    for i in reversed(range(5)):
        txt =  topLabels[i]
        conf = topProb[i]
        txt = 'Top'+str(topK)+':'+txt+' '+str(int(round((conf*100), 0)))+'%' 
        size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        t_height = size[0][1]
        textColor = (colors[topK - 1])
        cv2.putText(resultImage,txt,(45,t_height+(topK*30+40)),cv2.FONT_HERSHEY_SIMPLEX,0.5,textColor,1)
        topK = topK + 1
    end = time.time()
    if(verbosePrint):
        print('%30s' % 'Processed results image in ', str((end - start)*1000), 'ms')

    return resultImage, topIndex, topProb

# MIVisionX Classifier
if __name__ == '__main__':
    
    if len(sys.argv) == 1:
        app = QtWidgets.QApplication(sys.argv)
        panel = inference_control()
        app.exec_()
        modelFormat = (str)(panel.model_format)
        modelName = (str)(panel.model_name)
        modelLocation = (str)(panel.model)
        modelInputDims = (str)(panel.input_dims)
        modelOutputDims = (str)(panel.output_dims)
        label = (str)(panel.label)
        outputDir = (str)(panel.output)
        imageDir = (str)(panel.image)
        imageVal = (str)(panel.val)
        hierarchy = (str)(panel.hier)
        inputAdd = (str)(panel.add)
        inputMultiply = (str)(panel.multiply)
        fp16 = (str)(panel.fp16)
        replaceModel = (str)(panel.replace)
        verbose = (str)(panel.verbose)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_format',       type=str, required=True,    help='pre-trained model format, options:caffe/onnx/nnef [required]')
        parser.add_argument('--model_name',         type=str, required=True,    help='model name                             [required]')
        parser.add_argument('--model',              type=str, required=True,    help='pre_trained model file/folder          [required]')
        parser.add_argument('--model_input_dims',   type=str, required=True,    help='c,h,w - channel,height,width           [required]')
        parser.add_argument('--model_output_dims',  type=str, required=True,    help='c,h,w - channel,height,width           [required]')
        parser.add_argument('--label',              type=str, required=True,    help='labels text file                       [required]')
        parser.add_argument('--output_dir',         type=str, required=True,    help='output dir to store ADAT results       [required]')
        parser.add_argument('--image_dir',          type=str, required=True,    help='image directory for analysis           [required]')
        parser.add_argument('--image_val',          type=str, default='',       help='image list with ground truth           [optional]')
        parser.add_argument('--hierarchy',          type=str, default='',       help='AMD proprietary hierarchical file      [optional]')
        parser.add_argument('--add',                type=str, default='',       help='input preprocessing factor [optional - default:[0,0,0]]')
        parser.add_argument('--multiply',           type=str, default='',       help='input preprocessing factor [optional - default:[1,1,1]]')
        parser.add_argument('--fp16',               type=str, default='no',     help='quantize to FP16          [optional - default:no]')
        parser.add_argument('--replace',            type=str, default='no',     help='replace/overwrite model   [optional - default:no]')
        parser.add_argument('--verbose',            type=str, default='no',     help='verbose                   [optional - default:no]')
        args = parser.parse_args()
        
        # get arguments
        modelFormat = args.model_format
        modelName = args.model_name
        modelLocation = args.model
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
    # set verbose print
    if(verbose != 'no'):
        verbosePrint = True

    # set fp16 inference turned on/off
    if(fp16 != 'no'):
        FP16inference = True

    # set paths
    modelCompilerPath = '/opt/rocm/libexec/mivisionx/model_compiler/python'
    ADATPath= '/opt/rocm/libexec/mivisionx/toolkit/amd_data_analysis_toolkit/classification'
    setupDir = '~/.mivisionx-inference-analyzer'
    analyzerDir = os.path.expanduser(setupDir)
    modelDir = analyzerDir+'/'+modelName+'_dir'
    nnirDir = modelDir+'/nnir-files'
    openvxDir = modelDir+'/openvx-files'
    modelBuildDir = modelDir+'/build'
    adatOutputDir = os.path.expanduser(outputDir)
    inputImageDir = os.path.expanduser(imageDir)
    trainedModel = os.path.expanduser(modelLocation)
    labelText = os.path.expanduser(label)
    hierarchyText = os.path.expanduser(hierarchy)
    imageValText = os.path.expanduser(imageVal)
    pythonLib = modelBuildDir+'/libannpython.so'
    weightsFile = openvxDir+'/weights.bin'
    finalImageResultsFile = modelDir+'/imageResultsFile.csv'

    # get input & output dims
    str_c_i, str_h_i, str_w_i = modelInputDims.split(',')
    c_i = int(str_c_i); h_i = int(str_h_i); w_i = int(str_w_i)
    str_c_o, str_h_o, str_w_o = modelOutputDims.split(',')
    c_o = int(str_c_o); h_o = int(str_h_o); w_o = int(str_w_o)

    # input pre-processing values
    Ax=[0,0,0]
    if(inputAdd != ''):
        Ax = [float(item) for item in inputAdd.strip("[]").split(',')]
    Mx=[1,1,1]
    if(inputMultiply != ''):
        Mx = [float(item) for item in inputMultiply.strip("[]").split(',')]

    # check pre-trained model
    if(not os.path.isfile(trainedModel) and modelFormat != 'nnef' ):
        print("\nPre-Trained Model not found, check argument --model\n")
        quit()

    # check for label file
    if (not os.path.isfile(labelText)):
        print("\nlabels.txt not found, check argument --label\n")
        quit()
    else:
        fp = open(labelText, 'r')
        labelNames = fp.readlines()
        labelNames = [line.rstrip('\n') for line in labelNames]
        fp.close()

    # MIVisionX setup
    if(os.path.exists(analyzerDir)):
        print("\nMIVisionX Inference Analyzer\n")
        # replace old model or throw error
        if(replaceModel == 'yes'):
            ERROR_CHECK(os.system('rm -rf '+modelDir))
        elif(os.path.exists(modelDir)):
            print("OK: Model exists")

    else:
        print("\nMIVisionX Inference Analyzer Created\n")
        ERROR_CHECK(os.system('(cd ; mkdir .mivisionx-inference-analyzer)'))

    # Setup Text File for Demo
    if (not os.path.isfile(analyzerDir + "/setupFile.txt")):
        f = open(analyzerDir + "/setupFile.txt", "w")
        f.write(modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose)
        f.close()
    else:
        count = len(open(analyzerDir + "/setupFile.txt").readlines())
        if count < 10:
            with open(analyzerDir + "/setupFile.txt", "r") as fin:
                data = fin.read().splitlines(True)
                modelList = []
                for i in range(len(data)):
                    modelList.append(data[i].split(';')[1])
                if modelName not in modelList:
                    f = open(analyzerDir + "/setupFile.txt", "a")
                    f.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose)
                    f.close()
        else:
            with open(analyzerDir + "/setupFile.txt", "r") as fin:
                data = fin.read().splitlines(True)
            delModelName = data[0].split(';')[1]
            delmodelPath = analyzerDir + '/' + delModelName + '_dir'
            if(os.path.exists(delmodelPath)): 
                ERROR_CHECK(os.system('rm -rf ' + delmodelPath))
            with open(analyzerDir + "/setupFile.txt", "w") as fout:
                fout.writelines(data[1:])
            with open(analyzerDir + "/setupFile.txt", "a") as fappend:
                fappend.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose)
                fappend.close()

    # Compile Model and generate python .so files
    if (replaceModel == 'yes' or not os.path.exists(modelDir)):
        ERROR_CHECK(os.system('mkdir '+modelDir))
        if(os.path.exists(modelDir)):
            # convert to NNIR
            if(modelFormat == 'caffe'):
                ERROR_CHECK(os.system('(cd '+modelDir+'; python3 '+modelCompilerPath+'/caffe_to_nnir.py '+trainedModel+' nnir-files --input-dims 1,'+modelInputDims+' )'))
            elif(modelFormat == 'onnx'):
                ERROR_CHECK(os.system('(cd '+modelDir+'; python3 '+modelCompilerPath+'/onnx_to_nnir.py '+trainedModel+' nnir-files --input-dims 1,'+modelInputDims+' )'))
            elif(modelFormat == 'nnef'):
                ERROR_CHECK(os.system('(cd '+modelDir+'; python3 '+modelCompilerPath+'/nnef_to_nnir.py '+trainedModel+' nnir-files )'))
            else:
                print("ERROR: Neural Network Format Not supported, use caffe/onnx/nnef in arugment --model_format")
                quit()
            # convert the model to FP16
            if(FP16inference):
                ERROR_CHECK(os.system('(cd '+modelDir+'; python3 '+modelCompilerPath+'/nnir_update.py --convert-fp16 1 --fuse-ops 1 nnir-files nnir-files)'))
                print("\nModel Quantized to FP16\n")
            # convert to openvx
            if(os.path.exists(nnirDir)):
                ERROR_CHECK(os.system('(cd '+modelDir+'; python3 '+modelCompilerPath+'/nnir_to_openvx.py nnir-files openvx-files)'))
            else:
                print("ERROR: Converting Pre-Trained model to NNIR Failed")
                quit()
            
            # build model
            if(os.path.exists(openvxDir)):
                ERROR_CHECK(os.system('mkdir '+modelBuildDir))
            else:
                print("ERROR: Converting NNIR to OpenVX Failed")
                quit()
    ERROR_CHECK(os.system('(cd '+modelBuildDir+'; cmake ../openvx-files; make; ./anntest ../openvx-files/weights.bin )'))
    print("\nSUCCESS: Converting Pre-Trained model to MIVisionX Runtime successful\n")
    
    #else:
        #print("ERROR: MIVisionX Inference Analyzer Failed")
        #quit()

    #logos
    AMD_Epyc_Black_resize = cv2.imread("../../samples/images/amd-epyc-black-resize.png")
    AMD_ROCm_Black_resize = cv2.imread("../../samples/images/rocm-black-resize.png")

    # opencv display window
    windowInput = "MIVisionX Inference Analyzer - Input Image"
    windowResult = "MIVisionX Inference Analyzer - Results"
    windowProgress = "MIVisionX Inference Analyzer - Progress"
    cv2.namedWindow(windowInput, cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(windowInput, 800, 800)

    # create inference classifier
    classifier = annieObjectWrapper(pythonLib, weightsFile)

    # check for image val text
    totalImages = 0
    if(imageVal == ''):
        print("\nFlow without Image Validation Text..Creating a file with no ground truths\n")
        imageList = os.listdir(inputImageDir)
        imageList.sort()
        imageValText = os.getcwd() + '/imageValTxt.txt'
        fp = open(imageValText , 'w')
        for imageFile in imageList:
            fp.write(imageFile + " -1" + "\n")

    
    if (not os.path.isfile(imageValText)):
        print("\nImage Validation Text not found, check argument --image_val\n")
        quit()
    else:
        fp = open(imageValText, 'r')
        imageValidation = fp.readlines()
        fp.close()
        totalImages = len(imageValidation)

    # original std out location 
    orig_stdout = sys.stdout
    # setup results output file
    sys.stdout = open(finalImageResultsFile,'w')
    print('Image File Name,Ground Truth Label,Output Label 1,Output Label 2,Output Label 3,\
            Output Label 4,Output Label 5,Prob 1,Prob 2,Prob 3,Prob 4,Prob 5')
    sys.stdout = orig_stdout

    # process images
    correctTop5 = 0; correctTop1 = 0; wrong = 0; noGroundTruth = 0
    totalTimeForInference = 0
    totalCorners = 0
    listOfCorners = []
    for x in range(totalImages):
        imageFileName,grountTruth = imageValidation[x].split(' ')
        groundTruthIndex = int(grountTruth)
        imageFile = os.path.expanduser(inputImageDir+'/'+imageFileName)
        if (not os.path.isfile(imageFile)):
            print('Image File - '+imageFile+' not found')
            quit()
        else:
            # read image
            start = time.time()
            frame = cv2.imread(imageFile)
            end = time.time()
            totalTimeForInference += ((end - start)*1000)
            if(verbosePrint):
                print('%30s' % 'Read Image in ', str((end - start)*1000), 'ms')
                
            # image complexity detection
            start = time.time()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray,2,3,0.04)
            numCorners = np.sum(dst > 0.01 * dst.max())
            listOfCorners.append(numCorners)
            totalCorners += numCorners
            end = time.time()
            if(verbosePrint):
                print('%30s' % 'Image complexity detection in ', str((end - start)*1000), 'ms')

            # resize and process frame
            start = time.time()
            if(numCorners <= 3500):
                resizedFrame = cv2.resize(frame, (w_i,h_i), interpolation = cv2.INTER_LINEAR)
            else:
                resizedFrame = cv2.resize(frame, (w_i,h_i), interpolation = cv2.INTER_AREA)
            RGBframe = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
            if(inputAdd != '' or inputMultiply != ''):
                pFrame = np.zeros(RGBframe.shape).astype('float32')
                for i in range(RGBframe.shape[2]):
                    pFrame[:,:,i] = RGBframe.copy()[:,:,i] * Mx[i] + Ax[i]
                RGBframe = pFrame
            end = time.time()
            totalTimeForInference += ((end - start)*1000)
            if(verbosePrint):
                print('%30s' % 'Input pre-processed in ', str((end - start)*1000), 'ms')

            # run inference
            start = time.time()
            output = classifier.classify(RGBframe)
            end = time.time()
            totalTimeForInference += ((end - start)*1000)
            if(verbosePrint):
                print('%30s' % 'Executed Model in ', str((end - start)*1000), 'ms')

            # process output and display
            resultImage, topIndex, topProb = processClassificationOutput(resizedFrame, modelName, output)
            # copy logos to output window
            frameWindow = cv2.resize(frame, (800,800))
            frameWindow[0:AMD_ROCm_Black_resize.shape[0], 0:AMD_ROCm_Black_resize.shape[1]] = AMD_ROCm_Black_resize
            frameWindow[0:AMD_Epyc_Black_resize.shape[0], 607: 607 + AMD_Epyc_Black_resize.shape[1]] = AMD_Epyc_Black_resize
            start = time.time()
            cv2.imshow(windowInput, frameWindow)
            cv2.imshow(windowResult, resultImage)
            end = time.time()
            if(verbosePrint):
                print('%30s' % 'Processed display in ', str((end - start)*1000), 'ms')

            # write image results to a file
            start = time.time()
            sys.stdout = open(finalImageResultsFile,'a')
            print(imageFileName+','+str(groundTruthIndex)+','+str(topIndex[4])+
                ','+str(topIndex[3])+','+str(topIndex[2])+','+str(topIndex[1])+','+str(topIndex[0])+','+str(topProb[4])+
                ','+str(topProb[3])+','+str(topProb[2])+','+str(topProb[1])+','+str(topProb[0]))
            sys.stdout = orig_stdout
            end = time.time()
            if(verbosePrint):
                print('%30s' % 'Image result saved in ', str((end - start)*1000), 'ms')

            # create progress image
            start = time.time()
            progressImage = np.zeros((400, 500, 3), dtype="uint8")
            progressImage.fill(255)
            cv2.putText(progressImage, 'Inference Analyzer Progress', (25,  25),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            size = cv2.getTextSize(modelName, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            t_width = size[0][0]
            t_height = size[0][1]
            headerX_start = int(250 -(t_width/2))
            cv2.putText(progressImage,modelName,(headerX_start,t_height+(20+40)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
            txt = 'Processed: '+str(x+1)+' of '+str(totalImages)
            size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(progressImage,txt,(50,t_height+(60+40)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            # progress bar
            cv2.rectangle(progressImage, (50,150), (450,180), (192,192,192), -1)
            progressWidth = int(50+ ((400*(x+1))/totalImages))
            cv2.rectangle(progressImage, (50,150), (progressWidth,180), (255,204,153), -1)
            percentage = int(((x+1)/float(totalImages))*100)
            pTxt = 'progress: '+str(percentage)+'%'
            cv2.putText(progressImage,pTxt,(175,170),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

            if(groundTruthIndex == topIndex[4]):
                correctTop1 = correctTop1 + 1
                correctTop5 = correctTop5 + 1
            elif(groundTruthIndex == topIndex[3] or groundTruthIndex == topIndex[2] or groundTruthIndex == topIndex[1] or groundTruthIndex == topIndex[0]):
                correctTop5 = correctTop5 + 1
            elif(groundTruthIndex == -1):
                noGroundTruth = noGroundTruth + 1
            else:
                wrong = wrong + 1

            # top 1 progress
            cv2.rectangle(progressImage, (50,200), (450,230), (192,192,192), -1)
            progressWidth = int(50 + ((400*correctTop1)/totalImages))
            cv2.rectangle(progressImage, (50,200), (progressWidth,230), (0,153,0), -1)
            percentage = int((correctTop1/float(totalImages))*100)
            pTxt = 'Top1: '+str(percentage)+'%'
            cv2.putText(progressImage,pTxt,(195,220),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            # top 5 progress
            cv2.rectangle(progressImage, (50,250), (450,280), (192,192,192), -1)
            progressWidth = int(50+ ((400*correctTop5)/totalImages))
            cv2.rectangle(progressImage, (50,250), (progressWidth,280), (0,255,0), -1)
            percentage = int((correctTop5/float(totalImages))*100)
            pTxt = 'Top5: '+str(percentage)+'%'
            cv2.putText(progressImage,pTxt,(195,270),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            # wrong progress
            cv2.rectangle(progressImage, (50,300), (450,330), (192,192,192), -1)
            progressWidth = int(50+ ((400*wrong)/totalImages))
            cv2.rectangle(progressImage, (50,300), (progressWidth,330), (0,0,255), -1)
            percentage = int((wrong/float(totalImages))*100)
            pTxt = 'Mismatch: '+str(percentage)+'%'
            cv2.putText(progressImage,pTxt,(175,320),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            # no ground truth progress
            cv2.rectangle(progressImage, (50,350), (450,380), (192,192,192), -1)
            progressWidth = int(50+ ((400*noGroundTruth)/totalImages))
            cv2.rectangle(progressImage, (50,350), (progressWidth,380), (0,255,255), -1)
            percentage = int((noGroundTruth/float(totalImages))*100)
            pTxt = 'Ground Truth unavailable: '+str(percentage)+'%'
            cv2.putText(progressImage,pTxt,(125,370),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            
            cv2.imshow(windowProgress, progressImage)
            end = time.time()
            if(verbosePrint):
                print('%30s' % 'Progress image created in ', str((end - start)*1000), 'ms')
                print('%30s' % 'Images Processed: ', str(x), '\n\n')

            # exit on ESC
            key = cv2.waitKey(2)
            if key == 27: 
                break

    # Inference Analyzer Successful
    print("\nSUCCESS: Images Inferenced with the Model\n")
    timePerInference = float (totalTimeForInference / totalImages)
    print('%30s' % 'Time per image Inference ', str(timePerInference), 'ms')
    avgCornersPerImage = float(totalCorners / totalImages )
    print('%30s' % 'Avg number of corners per image: ', str(avgCornersPerImage))
    print('%30s' % 'Mean number of corners per image: ', str(statistics.mean(listOfCorners)))
    cv2.destroyWindow(windowInput)
    cv2.destroyWindow(windowResult)

    # Create ADAT folder and file
    print("\nADAT tool called to create the analysis toolkit\n")
    if(not os.path.exists(adatOutputDir)):
        ERROR_CHECK(os.system('mkdir -p ' + adatOutputDir))
    
    if(hierarchy == ''):
        ERROR_CHECK(os.system('python '+ADATPath+'/generate-visualization.py --inference_results '+finalImageResultsFile+
        ' --image_dir '+inputImageDir+' --label '+labelText+' --model_name '+modelName+' --output_dir '+adatOutputDir+' --output_name '+modelName+'-ADAT'))
    else:
        ERROR_CHECK(os.system('python '+ADATPath+'/generate-visualization.py --inference_results '+finalImageResultsFile+
        ' --image_dir '+inputImageDir+' --label '+labelText+' --hierarchy '+hierarchyText+' --model_name '+modelName+' --output_dir '+adatOutputDir+' --output_name '+modelName+'-ADAT'))
    print("\nSUCCESS: Image Analysis Toolkit Created\n")
    print("Press ESC to exit or close progress window\n")

    # Wait to quit
    while True:
        key = cv2.waitKey(2)
        if key == 27:
            cv2.destroyAllWindows()
            break        
        if cv2.getWindowProperty(windowProgress,cv2.WND_PROP_VISIBLE) < 1:        
            break

    outputHTMLFile = os.path.expanduser(adatOutputDir+'/'+modelName+'-ADAT-toolKit/index.html')
    ERROR_CHECK(os.system('firefox '+outputHTMLFile))
