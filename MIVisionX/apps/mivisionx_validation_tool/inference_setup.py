import sys
import os
import ctypes
import time
import numpy as np
import cv2
from numpy.ctypeslib import ndpointer
from PyQt5 import QtCore
from rocal_setup import *
import amd.rocal.types as types
from amd.rocal.plugin.pytorch import ROCAL_iterator
from amd.rocal.pipeline import Pipeline

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
        print('OK: AnnAPI found "' + self.annQueryInference().decode("utf-8") + '" as configuration in ' + library)

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

    def runInference(self, img_tensor, out):
        # copy input f32 to inference input
        status = self.api.annCopyToInferenceInput(self.hdl, np.ascontiguousarray(img_tensor, dtype=np.float32), img_tensor.nbytes, 0)
        if(status):
                print('ERROR: annCopyToInferenceInput Failed ')
        # run inference
        status = self.api.annRunInference(self.hdl, 1)
        if(status):
                print('ERROR: annRunInference Failed ')
        # copy output f32
        status = self.api.annCopyFromInferenceOutput(self.hdl, np.ascontiguousarray(out, dtype=np.float32), out.nbytes)
        if(status):
                print('ERROR: annCopyFromInferenceOutput Failed ')
        return out

    def classify(self, img_tensor):
        # create output.f32 buffer
        out_buf = bytearray(self.outputDim[0]*self.outputDim[1]*self.outputDim[2]*self.outputDim[3]*4)
        out = np.frombuffer(out_buf, dtype=np.float32)
        # run inference & receive output
        output = self.runInference(img_tensor, out)
        return output

class modelInference(QtCore.QObject):
    def __init__(self, modelName, modelFormat, imageDir, modelLocation, label, hierarchy, imageVal, modelInputDims, modelOutputDims, 
                modelBatchSize, outputDir, inputAdd, inputMultiply, verbose, fp16, replaceModel, loop, rocal_mode, origQueue, augQueue, gui, totalImages, fps_file, parent=None):

        super(modelInference, self).__init__(parent)
        self.modelCompilerPath = '/opt/rocm/libexec/mivisionx/model_compiler/python'
        self.ADATPath = '/opt/rocm/libexec/mivisionx/toolkit/analysis_and_visualization/classification'
        self.setupDir = '~/.mivisionx-validation-tool'

        self.analyzerDir = os.path.expanduser(self.setupDir)
        self.modelName = modelName
        self.modelDir = self.analyzerDir+'/'+modelName+'_dir'
        self.inputImageDir = os.path.expanduser((str)(imageDir))
        self.trainedModel = os.path.expanduser((str)(modelLocation))
        self.labelText = os.path.expanduser(label)
        self.hierarchy = hierarchy
        self.hierarchyText = os.path.expanduser(hierarchy)
        self.imageValText = os.path.expanduser(imageVal)
        self.adatOutputDir = os.path.expanduser(outputDir)
        self.nnirDir = self.modelDir+'/nnir-files'
        self.openvxDir = self.modelDir+'/openvx-files'
        self.modelBuildDir = self.modelDir+'/build'
        self.pythonLib = self.modelBuildDir+'/libannpython.so'
        self.weightsFile = self.openvxDir+'/weights.bin'
        self.finalImageResultsFile = self.modelDir+'/imageResultsFile.csv'
        if(fps_file != ''):
            self.fps_fileName = str(self.analyzerDir +"/"+ fps_file + '.txt')
        else:
            self.fps_fileName = str(self.analyzerDir+"/fps.txt")
        self.modelBatchSize = modelBatchSize
        self.replaceModel = replaceModel
        self.verbosePrint = False
        self.FP16inference = False
        self.loop = False
        self.classifier = None
        self.labelNames = []
        self.rocalEngine = None
        self.rocal_mode = rocal_mode
        self.origQueue = origQueue
        self.augQueue = augQueue
        self.imgCount = 0
        self.adatFlag = False
        self.totalImages = totalImages
        str_c_i, str_h_i, str_w_i = modelInputDims.split(',')
        self.c_i = int(str_c_i); self.h_i = int(str_h_i); self.w_i = int(str_w_i)
        str_c_o, str_h_o, str_w_o = modelOutputDims.split(',')
        self.c_o = int(str_c_o); self.h_o = int(str_h_o); self.w_o = int(str_w_o)
        self.totalStats = [0,0,0]
        self.augStats = []
        self.setupDone = False
        self.pauseState = False

        # set verbose print
        if(verbose != 'no'):
            self.verbosePrint = True

        # set replace model
        self.replaceModel = False
        if(replaceModel != 'no'):
            self.replaceModel = True

        # set fp16 inference turned on/off
        self.tensor_dtype = types.FLOAT
        if(fp16 != 'no'):
            self.FP16inference = True
            self.tensor_dtype=types.FLOAT16

        #set loop parameter based on user input
        if loop == 'yes':
            self.loop = True
        else:
            self.loop = False
        
        #set gui parameter based on user input
        self.gui = gui
        # get input & output dims
        self.modelBatchSizeInt = int(modelBatchSize)
        # input pre-processing values
        self.Ax=[0,0,0]
        if(inputAdd != ''):
            self.Ax = [float(item) for item in inputAdd.strip("[]").split(',')]
        self.Mx=[1,1,1]
        if(inputMultiply != ''):
            self.Mx = [float(item) for item in inputMultiply.strip("[]").split(',')]

        # MIVisionX setup
        if(os.path.exists(self.analyzerDir)):
            print("\nMIVisionX Validation Tool\n")
            # replace old model or throw error
            if(self.replaceModel):
                os.system('rm -rf '+self.modelDir)
            elif(os.path.exists(self.modelDir)):
                print("OK: Model exists")
        else:
            print("\nMIVisionX Validation Tool Created\n")
            os.system('(cd ; mkdir .mivisionx-validation-tool)')

        # Setup Text File for Demo
        if (not os.path.isfile(self.analyzerDir + "/setupFile.txt")):
            f = open(self.analyzerDir + "/setupFile.txt", "w")
            f.write(modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelBatchSize + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(self.Ax).strip('[]').replace(" ","") + ';' + str(self.Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose + ';' + loop)
            f.close()
        else:
            count = len(open(self.analyzerDir + "/setupFile.txt").readlines())
            if count < 10:
                with open(self.analyzerDir + "/setupFile.txt", "r") as fin:
                    data = fin.read().splitlines(True)
                    modelList = []
                    for i in range(len(data)):
                        if data[i] != '\n':
                            modelList.append(data[i].split(';')[1])
                    if modelName not in modelList:
                        f = open(self.analyzerDir + "/setupFile.txt", "a")
                        f.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelBatchSize + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(self.Ax).strip('[]').replace(" ","") + ';' + str(self.Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose + ';' + loop)
                        f.close()
            else:
                with open(self.analyzerDir + "/setupFile.txt", "r") as fin:
                    data = fin.read().splitlines(True)
                    delModelName = data[0].split(';')[1]
                delmodelPath = self.analyzerDir + '/' + delModelName + '_dir'
                if(os.path.exists(delmodelPath)): 
                    os.system('rm -rf ' + delmodelPath)
                with open(self.analyzerDir + "/setupFile.txt", "w") as fout:
                    fout.writelines(data[1:])
                with open(self.analyzerDir + "/setupFile.txt", "a") as fappend:
                    fappend.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelBatchSize + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(self.Ax).strip('[]').replace(" ","") + ';' + str(self.Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose + ';' + loop)
                    fappend.close()

        self.modelFormat = modelFormat
        self.modelInputDims = modelInputDims
        self.modelOutputDims = modelOutputDims
        self.imageVal = imageVal
        self.stdout = None
        #to calculate FPS
        self.msFrame = 0.0
        self.totalFPS = 0.0
        # get correct list for augmentations
        self.rocalList = []
        self.setupInference()

    def setupInference(self):
        # check pre-trained model
        if(not os.path.isfile(self.trainedModel) and self.modelFormat != 'nnef' ):
            print("\nPre-Trained Model not found, check argument --model\n")
            quit()

        # check for label file
        if (not os.path.isfile(self.labelText)):
            print("\nlabels.txt not found, check argument --label\n")
            quit()
        else:
            fp = open(self.labelText, 'r')
            self.labelNames = [x.strip('\n') for x in fp.readlines()]
            fp.close()

        # Compile Model and generate python .so files
        if (self.replaceModel or not os.path.exists(self.modelDir)):
            os.system('mkdir '+self.modelDir)
            if(os.path.exists(self.modelDir)):
                # convert to NNIR
                if(self.modelFormat == 'caffe'):
                    os.system('(cd '+self.modelDir+'; python3 '+self.modelCompilerPath+'/caffe_to_nnir.py '+self.trainedModel+' nnir-files --input-dims 1,' + self.modelInputDims + ')')
                    os.system('(cd '+self.modelDir+'; python3 '+self.modelCompilerPath+'/nnir_update.py --batch-size ' + self.modelBatchSize + ' nnir-files nnir-files)')
                elif(self.modelFormat == 'onnx'):
                    os.system('(cd '+self.modelDir+'; python3 '+self.modelCompilerPath+'/onnx_to_nnir.py '+self.trainedModel+' nnir-files --input_dims 1,' + self.modelInputDims + ')')
                    os.system('(cd '+self.modelDir+'; python3 '+self.modelCompilerPath+'/nnir_update.py --batch-size ' + self.modelBatchSize + ' nnir-files nnir-files)')
                elif(self.modelFormat == 'nnef'):
                    os.system('(cd '+self.modelDir+'; python3 '+self.modelCompilerPath+'/nnef_to_nnir.py '+self.trainedModel+' nnir-files --batch-size ' + self.modelBatchSize + ')')
                    #os.system('(cd '+self.modelDir+'; python3 '+self.modelCompilerPath+'/nnir_update.py --batch-size ' + self.modelBatchSize + ' nnir-files nnir-files)')
                else:
                    print("ERROR: Neural Network Format Not supported, use caffe/onnx/nnef in arugment --model_format")
                    quit()
                # convert the model to FP16
                if(self.FP16inference):
                    os.system('(cd '+self.modelDir+'; python3 '+self.modelCompilerPath+'/nnir_update.py --convert-fp16 1 --fuse-ops 1 nnir-files nnir-files)')
                    print("\nModel Quantized to FP16\n")
                # convert to openvx
                if(os.path.exists(self.nnirDir)):
                    os.system('(cd '+self.modelDir+'; python3 '+self.modelCompilerPath+'/nnir_to_openvx.py nnir-files openvx-files)')
                else:
                    print("ERROR: Converting Pre-Trained model to NNIR Failed")
                    quit()
                
                # build model
                if(os.path.exists(self.openvxDir)):
                    os.system('mkdir '+self.modelBuildDir)
                else:
                    print("ERROR: Converting NNIR to OpenVX Failed")
                    quit()

        os.system('(cd '+self.modelBuildDir+'; cmake ../openvx-files; make; ./anntest ../openvx-files/weights.bin )')
        print("\nSUCCESS: Converting Pre-Trained model to MIVisionX Runtime successful\n")

        # create inference classifier
        self.classifier = annieObjectWrapper(self.pythonLib, self.weightsFile)

        # check for image val text
        if(self.imageVal != ''):
            if (not os.path.isfile(self.imageValText)):
                print("\nImage Validation Text not found, check argument --image_val\n")
                quit()
            else:
                fp = open(self.imageValText, 'r')
                imageValidation = fp.readlines()
                fp.close()
        else:
            print("\nFlow without Image Validation Text not implemented, pass argument --image_val\n")
            quit()

        # original std out location 
        self.orig_stdout = sys.stdout
        # setup results output file
        sys.stdout = open(self.finalImageResultsFile,'w')	
        print('Image File Name,Ground Truth Label,Output Label 1,Output Label 2,Output Label 3,Output Label 4,Output Label 5,Prob 1,Prob 2,Prob 3,Prob 4,Prob 5')
        sys.stdout = self.orig_stdout

        # Setup rocAL Data Loader. 
        rocal_batch_size = 1
        device_id = 0
        self.pipe = Pipeline(batch_size=rocal_batch_size, num_threads=1, device_id=0, seed=12 + device_id, rocal_cpu=True, tensor_layout = types.NCHW, tensor_dtype=self.tensor_dtype)
        self.rocalEngine = InferencePipe(self.pipe, imageValidation, self.modelBatchSizeInt, self.rocal_mode, self.c_i, 
                                    self.h_i, self.w_i, rocal_batch_size, self.tensor_dtype, self.Mx, self.Ax, 
                                    tensor_layout = types.NCHW, num_threads=1, device_id=0, 
                                    data_dir=self.inputImageDir, crop=224, rocal_cpu=True)
        self.imageIterator = ROCAL_iterator(self.pipe)
        self.rocalList = self.rocalEngine.get_rocal_list(self.rocal_mode, self.modelBatchSizeInt)
        for i in range(self.modelBatchSizeInt):
            self.augStats.append([0,0,0])
        self.setupDone = True

    # process classification output function
    def processClassificationOutput(self, modelOutput):#, labelNames):
        # post process output file
        softmaxOutput = np.float32(modelOutput)
        outputList = np.split(softmaxOutput, self.modelBatchSizeInt)
        topIndex = []
        #topLabels = []
        topProb = []
        for i in range(len(outputList)):
            for x in outputList[i].argsort()[-5:]:
                topIndex.append(x)
                #topLabels.append(labelNames[x])
                topProb.append(softmaxOutput[x])

        return topIndex, topProb

    def setIntensity(self, intensity):
        self.rocalEngine.updateAugmentationParameter(intensity)
    
    def pauseInference(self):
        self.pauseState = not self.pauseState

    def terminate(self):
        self.pauseState = True

    def runInference(self):
        while self.setupDone:
            while not self.pauseState:
                msFrame = 0.0
                start = time.time()
                image_RGB_it, image_tensor = self.rocalEngine.get_next_augmentation(self.imageIterator)
                image_RGB = image_RGB_it[0]
                image_batch = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)
                original_image = image_batch[0:self.h_i, 0:self.w_i]
                cloned_image = np.copy(image_batch)
                frame = image_tensor
                #get image file name and ground truth
                imageFileName = self.rocalEngine.get_input_name()
                groundTruthIndex = self.rocalEngine.get_ground_truth()
                groundTruthIndex = int(groundTruthIndex)
                groundTruthLabel = self.labelNames[groundTruthIndex]
                groundTruthLabel = groundTruthLabel.split(" ", 1)[1]
                groundTruthLabel = groundTruthLabel.split(",", 1)[0]

                end = time.time()
                msFrame += (end-start)*1000
                if (self.verbosePrint):
                    print ('%30s' % 'Get next image from rocal took', str((end - start)*1000), 'ms')
    
                if self.gui:
                    text_width, text_height = cv2.getTextSize(groundTruthLabel, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    text_off_x = int((self.w_i/2) - (text_width/2))
                    text_off_y = int(self.h_i-7)
                    box_coords = ((text_off_x, text_off_y), (text_off_x + text_width - 2, text_off_y - text_height - 2))
                    color = (245, 197, 66)
                    cv2.rectangle(original_image, (text_off_x, text_off_y), (text_off_x + text_width + 15, text_off_y - text_height), color, cv2.FILLED)
                    cv2.putText(original_image, groundTruthLabel, (text_off_x, text_off_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
                    self.origQueue.put(original_image)
                
                # call python inference. Returns output tensor with 1000 class probabilites
                start = time.time()
                output = self.classifier.classify(frame)
                end = time.time()
                msFrame += (end-start)*1000
                if (self.verbosePrint):
                    print ('%30s' % 'inference took', str((end - start)*1000), 'ms' )
                start = time.time()
                topIndex, topProb = self.processClassificationOutput(output)
                end = time.time()
                msFrame += (end-start)*1000
                if (self.verbosePrint):
                    print ('%30s' % 'Processing inference output took', str((end - start)*1000), 'ms' )
                # Process output for each of the 64 images
                for i in range(self.modelBatchSizeInt):
                    #process the output tensor
                    start = time.time()
                    correctResult = self.processOutput(groundTruthIndex, topIndex, topProb, i, imageFileName)
                    end = time.time()
                    msFrame += (end-start)*1000
                    if (self.verbosePrint):
                        print ('%30s' % 'Processing top 5 results took ', str((end - start)*1000), 'ms' )

                    if self.gui:
                        augmentationText = self.rocalList[i].split('+')
                        textCount = len(augmentationText)
                        for cnt in range(0,textCount):
                            currentText = augmentationText[cnt]
                            text_width, text_height = cv2.getTextSize(currentText, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                            text_off_x = (int)((self.w_i/2) - (text_width/2))
                            text_off_y = ((int)(i*self.h_i)+self.h_i-7-(cnt*text_height))
                            box_coords = ((text_off_x, text_off_y), (text_off_x + text_width - 2, text_off_y - text_height - 2))
                            cv2.rectangle(cloned_image, box_coords[0], box_coords[1], (245, 197, 66), cv2.FILLED)
                            cv2.putText(cloned_image, currentText, (text_off_x, text_off_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2) 
                        # put augmented image result
                        if not correctResult:
                            cv2.rectangle(cloned_image, (0,(i*(self.h_i-1)+i)),((self.w_i-1),(self.h_i-1)*(i+1) + i), (255,0,0), 4, cv2.LINE_8, 0)
                        else:      
                            cv2.rectangle(cloned_image, (0,(i*(self.h_i-1)+i)),((self.w_i-1),(self.h_i-1)*(i+1) + i), (0,255,0), 4, cv2.LINE_8, 0)

                if self.gui:
                    #split image as needed
                    if self.modelBatchSizeInt == 64:
                            image_batch = np.vsplit(cloned_image, 16)
                            final_image_batch = np.hstack((image_batch))
                    elif self.modelBatchSizeInt == 16:
                        image_batch = np.vsplit(cloned_image, 4)
                        final_image_batch = np.hstack((image_batch))
                    self.augQueue.put(final_image_batch)
                self.updateFPS(msFrame) 
                self.imgCount +=  1
                if self.imgCount == self.totalImages:
                    if self.adatFlag == False:
                        self.generateADAT(self.modelName, self.hierarchy)
                        self.adatFlag = True
                    self.resetStats()
                    self.imageIterator.reset()

    def updateFPS(self, msFrame):
        self.totalFPS = 1000/(msFrame/self.modelBatchSizeInt)
        if not self.gui:
            fpsText = open(self.fps_fileName, "w")
            fpsText.write(str(int(self.totalFPS)))
            fpsText.close()
            print('FPS: %d\n' % self.totalFPS)
    
    def getFPS(self):
        return self.totalFPS

    def getTotalStats(self):
        return self.totalStats

    def getAugStats(self, augmentation):
        return self.augStats[augmentation]

    def getAugName(self, index):
        return self.rocalList[index]

    def processOutput(self, groundTruthIndex, topIndex, topProb, i, imageFileName):
        sys.stdout = open(self.finalImageResultsFile,'a')
        print(imageFileName+','+str(groundTruthIndex)+','+str(topIndex[4 + i*4])+
        ','+str(topIndex[3 + i*4])+','+str(topIndex[2 + i*4])+','+str(topIndex[1 + i*4])+','+str(topIndex[0 + i*4])+','+str(topProb[4 + i*4])+
        ','+str(topProb[3 + i*4])+','+str(topProb[2 + i*4])+','+str(topProb[1 + i*4])+','+str(topProb[0 + i*4]))
        sys.stdout = self.orig_stdout
        
        #data collection for individual augmentation scores
        countPerAugmentation = self.augStats[i]

        correctResult = False
        #print groundTruthIndex, topIndex
        # augmentedResults List: 0 = wrong; 1-5 = TopK; -1 = No Ground Truth
        if(groundTruthIndex == topIndex[4 + i*4]):
            self.totalStats[0] += 1
            self.totalStats[1] += 1
            correctResult = True
            countPerAugmentation[0] += 1
            countPerAugmentation[1] += 1
        elif(groundTruthIndex == topIndex[3 + i*4] or groundTruthIndex == topIndex[2 + i*4] or groundTruthIndex == topIndex[1 + i*4] or groundTruthIndex == topIndex[0 + i*4]):
            self.totalStats[1] += 1
            countPerAugmentation[1] += 1
            correctResult = True
        else:
            self.totalStats[2] += 1
            countPerAugmentation[2] += 1

        self.augStats[i] = countPerAugmentation
        return correctResult

    def resetStats(self):
        self.imgCount = 0
        self.totalStats = [0,0,0]
        self.augStats = []
        for i in range(self.modelBatchSizeInt):
            self.augStats.append([0,0,0])
        self.totalFPS = 0.0

    def generateADAT(self, modelName, hierarchy):
        # Create ADAT folder and file
        print("\nADAT tool called to create the analysis toolkit\n")
        if(not os.path.exists(self.adatOutputDir)):
            os.system('mkdir ' + self.adatOutputDir)
        
        if(hierarchy == ''):
            os.system('python3 '+self.ADATPath+'/generate-visualization.py --inference_results '+self.finalImageResultsFile+
            ' --image_dir '+self.inputImageDir+' --label '+self.labelText+' --model_name '+modelName+' --output_dir '+self.adatOutputDir+' --output_name '+modelName+'-ADAT')
        else:
            os.system('python3 '+self.ADATPath+'/generate-visualization.py --inference_results '+self.finalImageResultsFile+
            ' --image_dir '+self.inputImageDir+' --label '+self.labelText+' --hierarchy '+self.hierarchyText+' --model_name '+modelName+' --output_dir '+self.adatOutputDir+' --output_name '+modelName+'-ADAT')
        print("\nSUCCESS: Image Analysis Toolkit Created\n")
