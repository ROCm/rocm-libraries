import pyqtgraph as pg
import queue
from PyQt5 import QtGui, uic
from PyQt5.QtGui import QPixmap
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTime, QTimer, QThread
from inference_setup import *

class InferenceViewer(QtWidgets.QMainWindow):
    def __init__(self, model_name, model_format, image_dir, model_location, label, hierarchy, image_val, input_dims, output_dims, batch_size, output_dir, 
                                        add, multiply, verbose, fp16, replace, loop, rocal_mode, gui, container_logo, fps_file, cpu_name, gpu_name, parent):
        super(InferenceViewer, self).__init__(parent)
        self.parent = parent

        self.model_name = model_name
        self.model_format = model_format 
        self.image_dir = image_dir
        self.model_location = model_location
        self.label = label
        self.hierarchy = hierarchy
        self.image_val = image_val
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batch_size = batch_size
        self.batch_size_int = (int)(batch_size)
        self.output_dir = output_dir
        self.add = add
        self.multiply = multiply
        self.gui = True if gui == 'yes' else False
        self.fp16 = fp16
        self.replace = replace
        self.verbose = verbose
        self.loop = loop
        self.rocal_mode = rocal_mode
        inputImageDir = os.path.expanduser(image_dir)
        self.total_images = len(os.listdir(inputImageDir))
        self.origImageQueue = queue.Queue()
        self.augImageQueue = queue.Queue()
        self.fps_file = fps_file
        self.inferenceEngine = None
        self.receiver_thread = None

        self.initEngines()
        
        if self.gui:
            self.container_index = (int)(container_logo)
            self.cpu_name = cpu_name
            self.gpu_name = gpu_name
            self.pauseState = False
            self.showAug = False
            self.elapsedTime = QTime.currentTime()
            self.lastTime = 0
            self.totalElapsedTime = 0.0
            self.totalAccuracy = 0
            self.progIndex = 0
            self.augIntensity = 0.0
            self.imgCount = 0
            self.frameCount = 9
            self.lastIndex = self.frameCount - 1
            self.x = [0] 
            self.y = [0]
            self.augAccuracy = []
            self.totalCurve = None
            self.augCurve = None
            self.graph = None
            self.legend = None
            self.lastAugName = None
            self.pen = pg.mkPen('w', width=4)
            self.AMD_Radeon_pixmap = QPixmap("./data/images/AMD_Radeon.png")
            self.AMD_Radeon_white_pixmap = QPixmap("./data/images/AMD_Radeon-white.png")
            self.MIVisionX_pixmap = QPixmap("./data/images/MIVisionX-logo.png")
            self.MIVisionX_white_pixmap = QPixmap("./data/images/MIVisionX-logo-white.png")
            self.EPYC_pixmap = QPixmap("./data/images/EPYC-blue.png")
            self.EPYC_white_pixmap = QPixmap("./data/images/EPYC-blue-white.png")
            self.docker_pixmap = QPixmap("./data/images/Docker.png")
            self.singularity_pixmap = QPixmap("./data/images/Singularity.png")
            self.rocal_pixmap = QPixmap("./data/images/rocal.png")
            self.rocal_white_pixmap = QPixmap("./data/images/rocal-white.png")
            self.graph_image_pixmap = QPixmap("./data/images/Graph-image.png")
            self.graph_image_white_pixmap = QPixmap("./data/images/Graph-image-white.png")
            self.initUI()
            self.updateTimer = QTimer()
            self.updateTimer.timeout.connect(self.update)
            self.updateTimer.timeout.connect(self.plotGraph)
            self.updateTimer.timeout.connect(self.setProgressBar)
            self.updateTimer.start(40)
            self.FPSTimer = QTimer()
            self.FPSTimer.timeout.connect(self.displayFPS)
            self.FPSTimer.start(3000)
       
    def initUI(self):
        uic.loadUi("inference_viewer.ui", self)
        self.setStyleSheet("background-color: white")
        self.name_label.setText("Model: %s" % (self.model_name))
        self.cpuName_label.setText(self.cpu_name)
        self.gpuName_label.setText(self.gpu_name)
        self.dataset_label.setText("Augmentation set - %d" % (self.rocal_mode))
        self.imagesFrame.setStyleSheet(".QFrame {border-width: 20px; border-image: url(./data/images/filmStrip.png);}")
        self.total_progressBar.setStyleSheet("QProgressBar::chunk { background: lightblue; }")
        self.top1_progressBar.setStyleSheet("QProgressBar::chunk { background: green; }")
        self.top5_progressBar.setStyleSheet("QProgressBar::chunk { background: lightgreen; }")
        self.mis_progressBar.setStyleSheet("QProgressBar::chunk { background: red; }")
        self.total_progressBar.setMaximum(self.total_images*self.batch_size_int)
        self.graph = pg.PlotWidget(title="Accuracy vs Time")
        self.graph.setLabel('left', 'Accuracy', '%')
        self.graph.setLabel('bottom', 'Time', 's')
        self.graph.setYRange(0, 100, padding=0)
        pg.setConfigOptions(antialias=True)
        self.totalCurve = self.graph.plot(pen=self.pen)
        self.augCurve = self.graph.plot(pen=pg.mkPen('b', width=4))
        self.legend = pg.LegendItem(offset=(370,1))
        self.legend.setParentItem(self.graph.plotItem)
        self.legend.addItem(self.totalCurve, 'Cumulative')
        self.graph.setBackground(None)
        self.graph.setMaximumWidth(550)
        self.graph.setMaximumHeight(300)
        self.verticalLayout_2.addWidget(self.graph)
        self.level_slider.setMaximum(100)
        self.level_slider.valueChanged.connect(self.setIntensity)
        self.pause_pushButton.setStyleSheet("color: white; background-color: darkBlue")
        self.stop_pushButton.setStyleSheet("color: white; background-color: darkRed")
        self.pause_pushButton.clicked.connect(self.pauseView)
        self.stop_pushButton.clicked.connect(self.terminate)
        self.dark_checkBox.stateChanged.connect(self.setBackground)
        self.verbose_checkBox.stateChanged.connect(self.showVerbose)
        self.rocal_checkBox.stateChanged.connect(self.showRocal)
        self.dark_checkBox.setChecked(True)
        self.graph_imageLabel.setPixmap(self.graph_image_pixmap)
        if self.container_index == 1:
            self.container_logo.setPixmap(self.docker_pixmap)
        elif self.container_index == 2:
            self.container_logo.setPixmap(self.singularity_pixmap)
        else:
            self.container_logo.hide()
        for augmentation in range(self.batch_size_int):
            self.augAccuracy.append([0])

        self.showVerbose()
        self.showRocal()

    def initEngines(self):
        self.receiver_thread = QThread()
        # Creating an object for inference.
        self.inferenceEngine = modelInference(self.model_name, self.model_format, self.image_dir, self.model_location, self.label, self.hierarchy, self.image_val,
                                                self.input_dims, self.output_dims, self.batch_size, self.output_dir, self.add, self.multiply, self.verbose, self.fp16, 
                                                self.replace, self.loop, self.rocal_mode, self.origImageQueue, self.augImageQueue, self.gui, self.total_images, self.fps_file)
        
        self.inferenceEngine.moveToThread(self.receiver_thread)
        self.receiver_thread.started.connect(self.inferenceEngine.runInference)
        self.receiver_thread.finished.connect(self.inferenceEngine.deleteLater)
        self.receiver_thread.start()
        self.receiver_thread.terminate()

    def paintEvent(self, event):
        self.showAugImage()
        self.showImage()
        if self.imgCount == self.total_images:
            if self.loop == 'yes':
                self.resetViewer()
            else:
                self.terminate()
                

    def resetViewer(self):
        self.imgCount = 0
        del self.x[:]
        self.x.append(0)
        del self.y[:]
        self.y.append(0)
        del self.augAccuracy[:]
        for augmentation in range(self.batch_size_int):
            self.augAccuracy.append([0])

        self.lastTime = 0
        self.elapsedTime = QTime.currentTime()
        self.totalElapsedTime = 0.0
        self.progIndex = 0
        self.showAug = False
        self.lastIndex = self.frameCount - 1
        self.totalCurve.clear()
        self.augCurve.clear()
        self.name_label.setText("Model: %s" % (self.model_name))
        self.legend.removeItem(self.lastAugName)

    def setProgressBar(self):
        if self.showAug:
            self.setAugProgress(self.progIndex)
        else:
            self.setTotalProgress()

    def setTotalProgress(self):
        totalStats = self.inferenceEngine.getTotalStats()
        top1 = totalStats[0]
        top5 = totalStats[1]
        mis = totalStats[2]
        totalCount = top5 + mis
        if totalCount == 0:
            totalCount = totalCount + 1 # to avoid division by zero
        self.totalAccuracy = (float)(top5) / (totalCount+1) * 100
        self.total_progressBar.setValue(totalCount)
        self.total_progressBar.setMaximum(self.total_images*self.batch_size_int)
        self.imgProg_label.setText("Processed: %d of %d" % (totalCount, self.total_images*self.batch_size_int))
        self.top1_progressBar.setValue(top1)
        self.top1_progressBar.setMaximum(totalCount)
        self.top5_progressBar.setValue(top5)
        self.top5_progressBar.setMaximum(totalCount)
        self.mis_progressBar.setValue(mis)
        self.mis_progressBar.setMaximum(totalCount)

    def setAugProgress(self, augmentation):
        augStats = self.inferenceEngine.getAugStats(augmentation)
        top1 = augStats[0]
        top5 = augStats[1]
        mis = augStats[2]
        totalCount = top5 + mis
        self.total_progressBar.setValue(totalCount)
        self.total_progressBar.setMaximum(self.total_images)
        self.imgProg_label.setText("Processed: %d of %d" % (totalCount, self.total_images))
        self.top1_progressBar.setValue(top1)
        self.top1_progressBar.setMaximum(totalCount)
        self.top5_progressBar.setValue(top5)
        self.top5_progressBar.setMaximum(totalCount)
        self.mis_progressBar.setValue(mis)
        self.mis_progressBar.setMaximum(totalCount)

    def plotGraph(self):
        if not self.pauseState:
            curTime = self.elapsedTime.elapsed()/1000.0
            if (curTime - self.lastTime > 0.01):
                self.x.append(curTime+self.totalElapsedTime)
                self.y.append(self.totalAccuracy)
                self.totalCurve.setData(x=self.x, y=self.y, pen=self.pen)
                for augmentation in range(self.batch_size_int):
                    augStats = self.inferenceEngine.getAugStats(augmentation)
                    top5 = augStats[1]
                    mis = augStats[2]
                    totalCount = top5 + mis
                    totalAccuracy = (float)(top5) / (totalCount+1) * 100
                    self.augAccuracy[augmentation].append(totalAccuracy)
            
                if self.showAug:
                    self.augCurve.setData(x=self.x, y=self.augAccuracy[self.progIndex], pen=pg.mkPen('b', width=4))
                
            self.lastTime = curTime

    def showImage(self):
        if not self.origImageQueue.empty():
            origImage = self.origImageQueue.get()    
            origWidth = origImage.shape[1]
            origHeight = origImage.shape[0]
            qOrigImage = QtGui.QImage(origImage, origWidth, origHeight, origWidth*3, QtGui.QImage.Format_RGB888)
            qOrigImageResized = qOrigImage.scaled(self.image_label.width(), self.image_label.height(), QtCore.Qt.IgnoreAspectRatio)  
            index = self.imgCount % self.frameCount
            self.origImage_layout.itemAt(index).widget().setPixmap(QtGui.QPixmap.fromImage(qOrigImageResized))
            self.origImage_layout.itemAt(index).widget().setStyleSheet("border: 5px solid yellow;")
            self.origImage_layout.itemAt(self.lastIndex).widget().setStyleSheet("border: 0")
            self.imgCount += 1
            self.lastIndex = index
        

    def showAugImage(self):
        if not self.augImageQueue.empty():
            augImage = self.augImageQueue.get()
            augWidth = augImage.shape[1]
            augHeight = augImage.shape[0]
            qAugImage = QtGui.QImage(augImage, augWidth, augHeight, augWidth*3, QtGui.QImage.Format_RGB888)
            if self.batch_size_int == 64:
                qAugImageResized = qAugImage.scaled(self.aug_label.width(), self.aug_label.height(), QtCore.Qt.IgnoreAspectRatio)              
            else:
                qAugImageResized = qAugImage.scaled(self.aug_label.width(), self.aug_label.height(), QtCore.Qt.KeepAspectRatio)
            self.aug_label.setPixmap(QtGui.QPixmap.fromImage(qAugImageResized))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.terminate()
            
        if event.key() == QtCore.Qt.Key_Space:
            self.pauseView()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            mousePos = event.pos()
            if self.aug_label.geometry().contains(mousePos):
                index = self.calculateIndex(mousePos.x(), mousePos.y())
                self.progIndex = index
                self.showAug = True
                augName = self.inferenceEngine.getAugName(index)
                self.name_label.setText(augName)
                self.augCurve.setData(x=self.x, y=self.augAccuracy[self.progIndex], pen=pg.mkPen('b', width=4))
                self.legend.removeItem(self.lastAugName)
                self.legend.addItem(self.augCurve, augName)
                self.lastAugName = augName
            else:
                self.showAug = False
                self.name_label.setText("Model: %s" % (self.model_name))
                self.augCurve.clear()
                self.legend.removeItem(self.lastAugName)
                self.legend.removeItem('Cumulative')
                self.legend.addItem(self.totalCurve, 'Cumulative')
            if not self.pauseState:
                self.totalCurve.clear()
                self.augCurve.clear()

    def setBackground(self):
        if self.dark_checkBox.isChecked():
            self.setStyleSheet("background-color: #25232F;")
            self.pen = pg.mkPen('w', width=4)
            self.graph.setBackground(None)
            self.origTitle_label.setStyleSheet("color: #C82327;")
            self.controlTitle_label.setStyleSheet("color: #C82327;")
            self.progTitle_label.setStyleSheet("color: #C82327;")
            self.graphTitle_label.setStyleSheet("color: #C82327;")
            self.augTitle_label.setStyleSheet("color: #C82327;")
            self.name_label.setStyleSheet("color: white;")
            self.dataset_label.setStyleSheet("color: white;")
            self.imgProg_label.setStyleSheet("color: white;")
            self.fps_label.setStyleSheet("color: #C82327;")
            self.dark_checkBox.setStyleSheet("color: white;")
            self.verbose_checkBox.setStyleSheet("color: white;")
            self.rocal_checkBox.setStyleSheet("color: white;")
            self.level_label.setStyleSheet("color: white;")
            self.low_label.setStyleSheet("color: white;")
            self.high_label.setStyleSheet("color: white;")
            self.cpu_label.setStyleSheet("color: #C82327;")
            self.gpu_label.setStyleSheet("color: #C82327;")
            self.cpuName_label.setStyleSheet("color: white;")
            self.gpuName_label.setStyleSheet("color: white;")
            self.AMD_logo.setPixmap(self.AMD_Radeon_white_pixmap)
            if self.rocal_checkBox.isChecked():
                self.MIVisionX_logo.setPixmap(self.rocal_white_pixmap)
                self.graph_imageLabel.setPixmap(self.graph_image_white_pixmap)
            else:
                self.MIVisionX_logo.setPixmap(self.MIVisionX_white_pixmap)
            self.EPYC_logo.setPixmap(self.EPYC_white_pixmap)
            self.totalCurve.setData(x=self.x, y=self.y, pen=self.pen)
        else:
            self.setStyleSheet("background-color: white;")
            self.pen = pg.mkPen('k', width=4)
            self.graph.setBackground(None)
            self.origTitle_label.setStyleSheet("color: 0;")
            self.controlTitle_label.setStyleSheet("color: 0;")
            self.progTitle_label.setStyleSheet("color: 0;")
            self.graphTitle_label.setStyleSheet("color: 0;")
            self.augTitle_label.setStyleSheet("color: 0;")
            self.name_label.setStyleSheet("color: 0;")
            self.dataset_label.setStyleSheet("color: 0;")
            self.imgProg_label.setStyleSheet("color: 0;")
            self.fps_label.setStyleSheet("color: 0;")
            self.dark_checkBox.setStyleSheet("color: 0;")
            self.verbose_checkBox.setStyleSheet("color: 0;")
            self.rocal_checkBox.setStyleSheet("color: 0;")
            self.level_label.setStyleSheet("color: 0;")
            self.low_label.setStyleSheet("color: 0;")
            self.high_label.setStyleSheet("color: 0;")
            self.cpu_label.setStyleSheet("color: 0;")
            self.gpu_label.setStyleSheet("color: 0;")
            self.cpuName_label.setStyleSheet("color: 0;")
            self.gpuName_label.setStyleSheet("color: 0;")
            self.AMD_logo.setPixmap(self.AMD_Radeon_pixmap)
            if self.rocal_checkBox.isChecked():
                self.MIVisionX_logo.setPixmap(self.rocal_pixmap)
                self.graph_imageLabel.setPixmap(self.graph_image_pixmap)
            else:
                self.MIVisionX_logo.setPixmap(self.MIVisionX_pixmap)
            self.EPYC_logo.setPixmap(self.EPYC_pixmap)
            self.totalCurve.setData(x=self.x, y=self.y, pen=self.pen)
            
    def showVerbose(self):
        if self.verbose_checkBox.isChecked():
            self.dataset_label.show()
            self.fps_label.show()
            self.fps_lcdNumber.show()
            self.legend.show()
            self.cpu_label.show()
            self.gpu_label.show()
            self.cpuName_label.show()
            self.gpuName_label.show()
        else:
            self.dataset_label.hide()
            self.fps_label.hide()
            self.fps_lcdNumber.hide()
            self.legend.hide()
            self.cpu_label.hide()
            self.gpu_label.hide()
            self.gpuName_label.hide()
            self.cpuName_label.hide()
        
    def showRocal(self):
        if self.rocal_checkBox.isChecked():
            if self.dark_checkBox.isChecked():
                self.MIVisionX_logo.setPixmap(self.rocal_white_pixmap)
                self.graph_imageLabel.setPixmap(self.graph_image_white_pixmap)
            else:
                self.MIVisionX_logo.setPixmap(self.rocal_pixmap)
                self.graph_imageLabel.setPixmap(self.graph_image_pixmap)
            self.graph_imageLabel.show()
        else:
            if self.dark_checkBox.isChecked():
                self.MIVisionX_logo.setPixmap(self.MIVisionX_white_pixmap)
            else:
                self.MIVisionX_logo.setPixmap(self.MIVisionX_pixmap)
            self.graph_imageLabel.hide()

    def displayFPS(self):
        self.fps_lcdNumber.display(self.inferenceEngine.getFPS())

    def pauseView(self):
        self.pauseState = not self.pauseState
        if self.pauseState:
            self.totalElapsedTime += self.elapsedTime.elapsed() / 1000.0
            self.pause_pushButton.setText('Resume')
        else:
            self.elapsedTime.restart()
            self.pause_pushButton.setText('Pause')

        self.inferenceEngine.pauseInference()

    def terminate(self):
        self.inferenceEngine.terminate()
        self.receiver_thread.quit()
        for count in range(10):
            QThread.msleep(50)

        self.close()

    def closeEvent(self, event):
        self.terminate()
        exit(0)

    def setIntensity(self):
        augIntensity = (float)(self.level_slider.value()) / 100.0
        self.inferenceEngine.setIntensity(augIntensity)

    def calculateIndex(self, x, y):
        if self.batch_size_int == 64:
            imgWidth = self.aug_label.width() / 16.0
        else:
            imgWidth = self.aug_label.width() / 4.0
        imgHeight = self.aug_label.height() / 4.0
        x -= self.aug_label.x()
        y -= self.aug_label.y()
        column = (int)(x / imgWidth)
        row = (int)(y / imgHeight)
        index = 4 * column + row
        return index