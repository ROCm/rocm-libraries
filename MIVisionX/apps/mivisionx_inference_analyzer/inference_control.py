import os
from PyQt5 import QtWidgets, uic

class inference_control(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(inference_control, self).__init__(parent)
        self.model_format = ''
        self.model_name = ''
        self.model = ''
        self.input_dims = ''
        self.output_dims = ''
        self.label = ''
        self.output = ''
        self.image = ''
        self.val = ''
        self.hier = ''
        self.add = '0,0,0'
        self.multiply = '1,1,1'
        self.fp16 = 'no'
        self.replace = 'no'
        self.verbose = 'no'
        self.runningState = False
        self.initUI()

    def initUI(self):
        uic.loadUi("inference_control.ui", self)
        #self.setStyleSheet("background-color: white")
        self.upload_comboBox.activated.connect(self.fromFile)
        self.file_pushButton.clicked.connect(self.browseFile)
        self.output_pushButton.clicked.connect(self.browseOutput)
        self.label_pushButton.clicked.connect(self.browseLabel)
        self.image_pushButton.clicked.connect(self.browseImage)
        self.val_pushButton.clicked.connect(self.browseVal)
        self.hier_pushButton.clicked.connect(self.browseHier)
        self.run_pushButton.clicked.connect(self.runConfig)
        self.file_lineEdit.textChanged.connect(self.checkInput)
        self.name_lineEdit.textChanged.connect(self.checkInput)
        self.idims_lineEdit.textChanged.connect(self.checkInput)
        self.odims_lineEdit.textChanged.connect(self.checkInput)
        self.output_lineEdit.textChanged.connect(self.checkInput)
        self.label_lineEdit.textChanged.connect(self.checkInput)
        self.image_lineEdit.textChanged.connect(self.checkInput)
        self.image_lineEdit.textChanged.connect(self.checkInput)
        self.close_pushButton.clicked.connect(self.closeEvent)
        self.file_lineEdit.setPlaceholderText("File Directory for NNEF / Model File for CAFFE/ONNX [required]")
        self.name_lineEdit.setPlaceholderText("Model Name [required]")
        self.idims_lineEdit.setPlaceholderText("c,h,w [required]")
        self.odims_lineEdit.setPlaceholderText("c,h,w [required]")
        self.padd_lineEdit.setPlaceholderText("r,g,b [optional]")
        self.pmul_lineEdit.setPlaceholderText("r,g,b [optional]")
        self.output_lineEdit.setPlaceholderText("Output Directory [required]")
        self.label_lineEdit.setPlaceholderText("Label File [required]")
        self.image_lineEdit.setPlaceholderText("Image Folder [required]")
        self.val_lineEdit.setPlaceholderText("[optional]")
        self.hier_lineEdit.setPlaceholderText("[optional]")
        self.close_pushButton.setStyleSheet("color: white; background-color: darkRed")
        self.readSetupFile()
        self.show()

    def browseFile(self):
        if self.format_comboBox.currentText() == 'nnef':
            self.file_lineEdit.setText(QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))
        else:
            filename, _filter = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', './')
            self.file_lineEdit.setText(filename)

    def browseOutput(self):
        self.output_lineEdit.setText(QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))

    def browseLabel(self):
        filename, _filter = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', './', filter="Text files (*.txt)")
        self.label_lineEdit.setText(filename)

    def browseImage(self):
        self.image_lineEdit.setText(QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))

    def browseVal(self):
        filename, _filter = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', './', filter="Text files (*.txt)")
        self.val_lineEdit.setText(filename)

    def browseHier(self):
        filename, _filter = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', './', '*.csv')
        self.hier_lineEdit.setText(filename)

    def readSetupFile(self):
        setupDir = '~/.mivisionx-inference-analyzer'
        analyzerDir = os.path.expanduser(setupDir)
        if os.path.isfile(analyzerDir + "/setupFile.txt"):
            for line in open(analyzerDir + "/setupFile.txt", "r"):
                token = line.split(';')
                if len(token) > 1:
                    modelName = token[1]
                    self.upload_comboBox.addItem(modelName)
            
    def fromFile(self):
        if self.upload_comboBox.currentIndex() == 0:
            self.name_lineEdit.setEnabled(True)
            self.file_lineEdit.setEnabled(True)
            self.idims_lineEdit.setEnabled(True)
            self.odims_lineEdit.setEnabled(True)
            self.label_lineEdit.setEnabled(True)
            self.output_lineEdit.setEnabled(True)
            self.image_lineEdit.setEnabled(True)
            self.val_lineEdit.setEnabled(True)
            self.hier_lineEdit.setEnabled(True)
            self.padd_lineEdit.setEnabled(True)
            self.pmul_lineEdit.setEnabled(True)
            self.fp16_checkBox.setEnabled(True)
            self.replace_checkBox.setEnabled(True)
            self.verbose_checkBox.setEnabled(True)
            self.file_pushButton.setEnabled(True)
            self.format_comboBox.setEnabled(True)
            self.output_pushButton.setEnabled(True)
            self.label_pushButton.setEnabled(True)
            self.image_pushButton.setEnabled(True)
            self.val_pushButton.setEnabled(True)
            self.hier_pushButton.setEnabled(True)
            self.format_comboBox.setCurrentIndex(0)
            self.name_lineEdit.clear()
            self.file_lineEdit.clear()
            self.idims_lineEdit.clear()
            self.odims_lineEdit.clear()
            self.label_lineEdit.clear()
            self.output_lineEdit.clear()
            self.image_lineEdit.clear()
            self.val_lineEdit.clear()
            self.hier_lineEdit.clear()
            self.padd_lineEdit.clear()
            self.pmul_lineEdit.clear()
            self.fp16_checkBox.setChecked(False)
            self.replace_checkBox.setChecked(False)
            self.verbose_checkBox.setChecked(False)
        else:
            modelName = self.upload_comboBox.currentText()
            setupDir = '~/.mivisionx-inference-analyzer'
            analyzerDir = os.path.expanduser(setupDir)
            for line in open(analyzerDir + "/setupFile.txt", "r"):
                tokens = line.split(';')
                if len(tokens) > 1:
                    name = tokens[1]
                    if modelName == name:
                        if tokens[0] == 'caffe':
                            format = 0
                        elif tokens[0] == 'onnx':
                            format = 1
                        else:
                            format = 2
                        self.format_comboBox.setCurrentIndex(format)
                        self.name_lineEdit.setText(tokens[1])
                        self.file_lineEdit.setText(tokens[2])
                        self.idims_lineEdit.setText(tokens[3])
                        self.odims_lineEdit.setText(tokens[4])
                        self.label_lineEdit.setText(tokens[5])
                        self.output_lineEdit.setText(tokens[6])
                        self.image_lineEdit.setText(tokens[7])
                        self.val_lineEdit.setText(tokens[8])
                        self.hier_lineEdit.setText(tokens[9])
                        self.padd_lineEdit.setText(tokens[10])
                        self.pmul_lineEdit.setText(tokens[11])
                        self.fp16_checkBox.setChecked(True) if tokens[12] == 'yes\n' or tokens[12] == 'yes' else self.fp16_checkBox.setChecked(False)
                        self.replace_checkBox.setChecked(True) if tokens[13] == 'yes\n' or tokens[13] == 'yes' else self.replace_checkBox.setChecked(False)
                        self.verbose_checkBox.setChecked(True) if tokens[14] == 'yes\n' or tokens[14] == 'yes' else self.verbose_checkBox.setChecked(False)
                        self.name_lineEdit.setEnabled(False)
                        self.file_lineEdit.setEnabled(False)
                        self.idims_lineEdit.setEnabled(False)
                        self.odims_lineEdit.setEnabled(False)
                        self.label_lineEdit.setEnabled(False)
                        self.output_lineEdit.setEnabled(False)
                        self.image_lineEdit.setEnabled(False)
                        self.val_lineEdit.setEnabled(False)
                        self.hier_lineEdit.setEnabled(False)
                        self.padd_lineEdit.setEnabled(False)
                        self.pmul_lineEdit.setEnabled(False)
                        self.fp16_checkBox.setEnabled(False)
                        self.output_pushButton.setEnabled(False)
                        self.label_pushButton.setEnabled(False)
                        self.image_pushButton.setEnabled(False)
                        self.val_pushButton.setEnabled(False)
                        self.hier_pushButton.setEnabled(False)
                        self.verbose_checkBox.setEnabled(False)
                        self.file_pushButton.setEnabled(False)
                        self.format_comboBox.setEnabled(False)

    def checkInput(self):
        if not (self.file_lineEdit.text() == '') and not (self.name_lineEdit.text() == '') \
            and not (self.idims_lineEdit.text() == '')  and not (self.odims_lineEdit.text() == '') \
            and not (self.output_lineEdit.text() == '') and not (self.label_lineEdit.text() == '') \
            and not (self.image_lineEdit.text() == ''):
                self.run_pushButton.setEnabled(True)
                self.run_pushButton.setStyleSheet("background-color: lightgreen")
        else:
            self.run_pushButton.setEnabled(False)
            self.run_pushButton.setStyleSheet("background-color: 0")

    def runConfig(self):
        self.model_format = self.format_comboBox.currentText()
        self.model_name = self.name_lineEdit.text()
        self.model = self.file_lineEdit.text()
        self.input_dims = '%s' % (self.idims_lineEdit.text())
        self.output_dims = '%s' % (self.odims_lineEdit.text())
        self.label = self.label_lineEdit.text()
        self.output = self.output_lineEdit.text()
        self.image = self.image_lineEdit.text()
        self.val = self.val_lineEdit.text()
        self.hier = self.hier_lineEdit.text()
        if len(self.padd_lineEdit.text()) < 1:
            self.add = '[0,0,0]'
        else:
            self.add = '[%s]' % (self.padd_lineEdit.text())
        if len(self.pmul_lineEdit.text()) < 1:
            self.multiply = '[1,1,1]'
        else:
            self.multiply = '[%s]' % (self.pmul_lineEdit.text())
        self.fp16 = 'yes' if self.fp16_checkBox.isChecked() else 'no'
        self.replace = 'yes' if self.replace_checkBox.isChecked() else 'no'
        self.verbose = 'yes' if self.verbose_checkBox.isChecked() else 'no'
        self.runningState = True
        self.close()

    def closeEvent(self, event):
        if not self.runningState:
            exit(0)
