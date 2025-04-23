/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "inference_control.h"
#include "inference_panel.h"
#include "inference_compiler.h"
#include "tcpconnection.h"
#include "assets.h"
#include <QGridLayout>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QFileDialog>
#include <QLabel>
#include <QStandardPaths>
#include <QFile>
#include <QTextStream>
#include <QIntValidator>
#include <QDoubleValidator>
#include <QMessageBox>
#include <QFileInfo>
#include <QFrame>
#include <QTimer>
#include <QCheckBox>
#include <QStyle>
#include <QDesktopServices>

#define CONFIGURATION_CACHE_FILENAME ".annInferenceApp.txt"
#define BUILD_VERSION "alpha2"

inference_control::inference_control(int operationMode_, QWidget *parent)
    : QWidget(parent), connectionSuccessful{ false }, modelType{ 0 }, numModelTypes{ 0 }, dataLabels{ nullptr }, dataHierarchy{ nullptr },
      lastPreprocessMpy{ "1", "1", "1" }, lastPreprocessAdd{ "0", "0", "0" }
{
    setWindowTitle("AMD MIVisionX Inference Client Application");
    setMinimumWidth(800);

    maxGPUs = 1;
    enableSF = 0;
    sendFileName = 0;
    enableTopK = 0;
    topKValue = 0;
    compiler_status.completed = false;
    compiler_status.dimOutput[0] = 0;
    compiler_status.dimOutput[1] = 0;
    compiler_status.dimOutput[2] = 0;
    compiler_status.errorCode = 0;
    operationMode = operationMode_;
    dataLabels = new QVector<QString>();
    dataHierarchy = new QVector<QString>();

    // default configuration
    QGridLayout * controlLayout = new QGridLayout;
    int editSpan = 3;
    int row = 0;

    //////////////
    /// \brief labelIntro
    ///
    QLabel * labelIntro = new QLabel("OBJECT RECOGNITION CONTROL PANEL");
    labelIntro->setStyleSheet("font-weight: bold; color: green; font-size: 21pt;");
    labelIntro->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    QPushButton * buttonLogo1 = new QPushButton();
    QPushButton * buttonLogo2 = new QPushButton();
    QPixmap pixmap1;
    QPixmap pixmap2;
    QByteArray arr1(assets::getLogoPng1Buf(), assets::getLogoPng1Len());
    QByteArray arr2(assets::getLogoPng2Buf(), assets::getLogoPng2Len());
    pixmap1.loadFromData(arr1);
    pixmap2.loadFromData(arr2);
    buttonLogo1->setIcon(pixmap1);
    buttonLogo1->setIconSize(QSize(96,64));
    buttonLogo1->setFixedSize(QSize(96,64));
    buttonLogo1->setFlat(true);
    buttonLogo2->setIcon(pixmap2);
    buttonLogo2->setIconSize(QSize(64,64));
    buttonLogo2->setFixedSize(QSize(64,64));
    controlLayout->addWidget(buttonLogo1, row, 0, 1, 1, Qt::AlignCenter);
    controlLayout->addWidget(labelIntro, row, 1, 1, editSpan);
    controlLayout->addWidget(buttonLogo2, row, 1 + editSpan, 1, 1, Qt::AlignCenter);
    connect(buttonLogo1, SIGNAL(released()), this, SLOT(onLogo1Click()));
    connect(buttonLogo2, SIGNAL(released()), this, SLOT(onLogo2Click()));
    row++;

    QFrame * sepHLine1 = new QFrame();
    sepHLine1->setFrameShape(QFrame::HLine);
    sepHLine1->setFrameShadow(QFrame::Sunken);
    controlLayout->addWidget(sepHLine1, row, 0, 1, editSpan + 2);
    row++;

    //////////////
    /// \brief labelServer
    ///
    QLabel * labelServer = new QLabel("Inference Server");
    labelServer->setStyleSheet("font-weight: bold; color: red; font-size: 18pt;");
    controlLayout->addWidget(labelServer, row, 0, 1, 5);
    row++;

    QLabel * labelServerHost = new QLabel("Hostname:");
    editServerHost = new QLineEdit("localhost");
    editServerPort = new QLineEdit("28282");
    buttonConnect = new QPushButton("Connect");
    labelServerHost->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelServerHost->setAlignment(Qt::AlignLeft);
    buttonConnect->setStyleSheet("font-weight: bold;");
    connect(buttonConnect, SIGNAL(released()), this, SLOT(runConnection()));
    controlLayout->addWidget(labelServerHost, row, 0, 1, 1);
    controlLayout->addWidget(editServerHost, row, 1, 1, 2);
    controlLayout->addWidget(editServerPort, row, 3, 1, 1);
    controlLayout->addWidget(buttonConnect, row, 1 + editSpan, 1, 1);
    row++;
    labelServerStatus = new QLabel("");
    labelServerStatus->setStyleSheet("font-style: italic;");
    labelServerStatus->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelServerStatus, row, 1, 1, editSpan);
    QPushButton * exitButton = new QPushButton("Exit");
    controlLayout->addWidget(exitButton, row, 1 + editSpan, 1, 1);
    connect(exitButton, SIGNAL(released()), this, SLOT(exitControl()));
    row++;

    QFrame * sepHLine2 = new QFrame();
    sepHLine2->setFrameShape(QFrame::HLine);
    sepHLine2->setFrameShadow(QFrame::Sunken);
    controlLayout->addWidget(sepHLine2, row, 0, 1, editSpan + 2);
    row++;

    //////////////
    /// \brief labelCompiler
    ///
    typeModelFile1Label.push_back("Prototxt:");
    typeModelFile2Label.push_back("CaffeModel:");
    typeModelFile1Desc.push_back("Prototxt (*.prototxt)");
    typeModelFile2Desc.push_back("CaffeModel (*.caffemodel)");
    numModelTypes++;
    QLabel * labelCompiler = new QLabel("Inference Compiler");
    labelCompiler->setStyleSheet("font-weight: bold; color: red; font-size: 18pt;");
    controlLayout->addWidget(labelCompiler, row, 0, 1, 5);
    row++;
    QLabel * labelModel = new QLabel("CNN Model:");
    comboModelSelect = new QComboBox();
    buttonCompile = new QPushButton(tr("Upload && Compile"), this);
    comboModelSelect->addItem("Upload a pre-trained Caffe model (i.e., .prototxt and .caffemodel)");
    labelModel->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelModel->setAlignment(Qt::AlignLeft);
    connect(comboModelSelect, SIGNAL(activated(int)), this, SLOT(modelSelect(int)));
    connect(buttonCompile, SIGNAL(released()), this, SLOT(runCompiler()));
    controlLayout->addWidget(labelModel, row, 0, 1, 1);
    controlLayout->addWidget(comboModelSelect, row, 1, 1, editSpan);
    controlLayout->addWidget(buttonCompile, row, 1 + editSpan, 1, 1);
    row++;
    QLabel * labelInputDim = new QLabel("CxHxW(inp):");
    QLineEdit * editDimC = new QLineEdit("3");
    editDimH = new QLineEdit("224");
    editDimW = new QLineEdit("224");
    editDimC->setValidator(new QIntValidator(3,3));
    editDimH->setValidator(new QIntValidator(1,16384));
    editDimW->setValidator(new QIntValidator(1,16384));
    editDimC->setEnabled(false);
    labelInputDim->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelInputDim->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelInputDim, row, 0, 1, 1);
    controlLayout->addWidget(editDimC, row, 1, 1, 1);
    controlLayout->addWidget(editDimH, row, 2, 1, 1);
    controlLayout->addWidget(editDimW, row, 3, 1, 1);
    row++;
    QLabel * labelOutputDim = new QLabel("CxHxW(out):");
    editOutDimC = new QLineEdit("");
    editOutDimH = new QLineEdit("");
    editOutDimW = new QLineEdit("");
    editOutDimC->setEnabled(false);
    editOutDimH->setEnabled(false);
    editOutDimW->setEnabled(false);
    labelOutputDim->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelOutputDim->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelOutputDim, row, 0, 1, 1);
    controlLayout->addWidget(editOutDimC, row, 1, 1, 1);
    controlLayout->addWidget(editOutDimH, row, 2, 1, 1);
    controlLayout->addWidget(editOutDimW, row, 3, 1, 1);
    row++;
    labelModelFile1 = new QLabel("--");
    editModelFile1 = new QLineEdit("");
    buttonModelFile1 = new QPushButton(tr("Browse..."), this);
    connect(buttonModelFile1, &QAbstractButton::clicked, this, &inference_control::browseModelFile1);
    labelModelFile1->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelModelFile1->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelModelFile1, row, 0, 1, 1);
    controlLayout->addWidget(editModelFile1, row, 1, 1, editSpan);
    controlLayout->addWidget(buttonModelFile1, row, 1 + editSpan, 1, 1);
    row++;
    labelModelFile2 = new QLabel("--");
    editModelFile2 = new QLineEdit("");
    buttonModelFile2 = new QPushButton(tr("Browse..."), this);
    connect(buttonModelFile2, &QAbstractButton::clicked, this, &inference_control::browseModelFile2);
    labelModelFile2->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelModelFile2->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelModelFile2, row, 0, 1, 1);
    controlLayout->addWidget(editModelFile2, row, 1, 1, editSpan);
    controlLayout->addWidget(buttonModelFile2, row, 1 + editSpan, 1, 1);
    row++;
    labelPreprocessMpy = new QLabel("Preprocess(mpy):");
    labelPreprocessMpy->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelPreprocessMpy->setAlignment(Qt::AlignLeft);
    editPreprocessMpyC0 = new QLineEdit("");
    editPreprocessMpyC1 = new QLineEdit("");
    editPreprocessMpyC2 = new QLineEdit("");
    editPreprocessMpyC0->setValidator(new QDoubleValidator());
    editPreprocessMpyC1->setValidator(new QDoubleValidator());
    editPreprocessMpyC2->setValidator(new QDoubleValidator());
    controlLayout->addWidget(labelPreprocessMpy, row, 0, 1, 1);
    controlLayout->addWidget(editPreprocessMpyC0, row, 1, 1, 1);
    controlLayout->addWidget(editPreprocessMpyC1, row, 2, 1, 1);
    controlLayout->addWidget(editPreprocessMpyC2, row, 3, 1, 1);
    row++;
    labelPreprocessAdd = new QLabel("Preprocess(add):");
    labelPreprocessAdd->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelPreprocessAdd->setAlignment(Qt::AlignLeft);
    editPreprocessAddC0 = new QLineEdit("");
    editPreprocessAddC1 = new QLineEdit("");
    editPreprocessAddC2 = new QLineEdit("");
    editPreprocessAddC0->setValidator(new QDoubleValidator());
    editPreprocessAddC1->setValidator(new QDoubleValidator());
    editPreprocessAddC2->setValidator(new QDoubleValidator());
    controlLayout->addWidget(labelPreprocessAdd, row, 0, 1, 1);
    controlLayout->addWidget(editPreprocessAddC0, row, 1, 1, 1);
    controlLayout->addWidget(editPreprocessAddC1, row, 2, 1, 1);
    controlLayout->addWidget(editPreprocessAddC2, row, 3, 1, 1);
    row++;
    labelCompilerOptions = new QLabel("Options:");
    labelCompilerOptions->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelCompilerOptions->setAlignment(Qt::AlignLeft);
    comboInvertInputChannels = new QComboBox();
    comboInvertInputChannels->addItem("RGB");
    comboInvertInputChannels->addItem("BGR");
    comboPublishOptions = new QComboBox();
    comboPublishOptions->addItem("");
    comboPublishOptions->addItem("PublishAs");
    comboPublishOptions->addItem("PublishAs (override)");
    editModelName = new QLineEdit("");
    editServerPassword = new QLineEdit("");
    editServerPort->setValidator(new QIntValidator(1,65535));
    editServerPassword->setEchoMode(QLineEdit::Password);
    controlLayout->addWidget(labelCompilerOptions, row, 0, 1, 1);
    controlLayout->addWidget(comboInvertInputChannels, row, 1, 1, 1);
    controlLayout->addWidget(comboPublishOptions, row, 2, 1, 1);
    controlLayout->addWidget(editModelName, row, 3, 1, 1);
    controlLayout->addWidget(editServerPassword, row, 4, 1, 1);
    row++;
    checkShadowFolder = new QCheckBox("Send FileName");
    checkShadowFolder->setChecked(false);
    checkShadowFolder->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    editShadowFolderAddr = new QLineEdit("");
    editShadowFolderAddr->setVisible(false);
    buttonShadowFolder = new QPushButton(tr("Browse..."), this);
    buttonShadowFolder->setVisible(false);
    controlLayout->addWidget(checkShadowFolder, row, 0, 1, 1);
    controlLayout->addWidget(editShadowFolderAddr, row, 1, 1, editSpan);
    controlLayout->addWidget(buttonShadowFolder, row, 1 + editSpan, 1, 1);
    connect(checkShadowFolder, SIGNAL(clicked(bool)), this, SLOT(shadowFolderEnable(bool)));   
    connect(buttonShadowFolder, &QAbstractButton::clicked, this, &inference_control::browseShadowFolder);


    row++;
    connect(editDimH, SIGNAL(textChanged(const QString &)), this, SLOT(onChangeDimH(const QString &)));
    connect(editDimW, SIGNAL(textChanged(const QString &)), this, SLOT(onChangeDimW(const QString &)));
    connect(editModelFile1, SIGNAL(textChanged(const QString &)), this, SLOT(onChangeModelFile1(const QString &)));
    connect(editModelFile2, SIGNAL(textChanged(const QString &)), this, SLOT(onChangeModelFile2(const QString &)));
    connect(comboInvertInputChannels, SIGNAL(activated(int)), this, SLOT(onChangeInputInverserOrder(int)));
    connect(editPreprocessMpyC0, SIGNAL(textChanged(const QString &)), this, SLOT(onChangePreprocessMpyC0(const QString &)));
    connect(editPreprocessMpyC1, SIGNAL(textChanged(const QString &)), this, SLOT(onChangePreprocessMpyC1(const QString &)));
    connect(editPreprocessMpyC2, SIGNAL(textChanged(const QString &)), this, SLOT(onChangePreprocessMpyC2(const QString &)));
    connect(editPreprocessAddC0, SIGNAL(textChanged(const QString &)), this, SLOT(onChangePreprocessAddC0(const QString &)));
    connect(editPreprocessAddC1, SIGNAL(textChanged(const QString &)), this, SLOT(onChangePreprocessAddC1(const QString &)));
    connect(editPreprocessAddC2, SIGNAL(textChanged(const QString &)), this, SLOT(onChangePreprocessAddC2(const QString &)));
    connect(comboPublishOptions, SIGNAL(activated(int)), this, SLOT(onChangePublishMode(int)));
    connect(editModelName, SIGNAL(textChanged(const QString &)), this, SLOT(onChangeModelName(const QString &)));
    labelCompilerStatus = new QLabel("");
    labelCompilerStatus->setStyleSheet("font-style: italic; color: gray;");
    labelCompilerStatus->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelCompilerStatus, row, 1, 1, editSpan + 1);
    row++;

    QFrame * sepHLine3 = new QFrame();
    sepHLine3->setFrameShape(QFrame::HLine);
    sepHLine3->setFrameShadow(QFrame::Sunken);
    controlLayout->addWidget(sepHLine3, row, 0, 1, editSpan + 2);
    row++;

    //////////////
    /// \brief labelRuntime
    ///
    QLabel * labelRuntime = new QLabel("Inference Run-time");
    labelRuntime->setStyleSheet("font-weight: bold; color: red; font-size: 18pt;");
    controlLayout->addWidget(labelRuntime, row, 0, 1, 5);
    row++;
    checkTopKResult = new QCheckBox("Top K Results");
    checkTopKResult->setChecked(false);
    checkTopKResult->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    controlLayout->addWidget(checkTopKResult, row, 0, 1, 1);
    comboTopKResult = new QComboBox();
    comboTopKResult->addItems({ "1", "2", "3", "4", "5" });
    comboTopKResult->setEnabled(false);
    controlLayout->addWidget(comboTopKResult, row, 1, 1, 1);
    connect(checkTopKResult, SIGNAL(clicked(bool)), this, SLOT(topKResultsEnable(bool)));
    buttonInference = new QPushButton("Run");
    controlLayout->addWidget(buttonInference, row, 1 + editSpan, 1, 1);
    connect(buttonInference, SIGNAL(released()), this, SLOT(runInference()));
    row++;
    QLabel * labelGPUs = new QLabel("GPUs:");
    editGPUs = new QLineEdit("1");
    labelMaxGPUs = new QLabel("");
    editGPUs->setValidator(new QIntValidator(1,maxGPUs));
    editGPUs->setEnabled(false);
    labelGPUs->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelGPUs->setAlignment(Qt::AlignLeft);
    QLabel * labelDecodeMode = new QLabel("Decode Mode:");
    labelDecodeMode->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    comboDecodeMode = new QComboBox();
    comboDecodeMode->addItems({ "OpenCV", "rocAL" });
    controlLayout->addWidget(labelGPUs, row, 0, 1, 1);
    controlLayout->addWidget(editGPUs, row, 1, 1, 1);
    controlLayout->addWidget(labelMaxGPUs, row, 2, 1, 1);
    controlLayout->addWidget(labelDecodeMode, row, 3, 1, 1);
    controlLayout->addWidget(comboDecodeMode, row, 4, 1, 1);
    row++;
    QLabel * labelImageLabelsFile = new QLabel("Labels:");
    editImageLabelsFile = new QLineEdit("");
    QPushButton * buttonDataLabels = new QPushButton(tr("Browse..."), this);
    connect(buttonDataLabels, &QAbstractButton::clicked, this, &inference_control::browseDataLabels);
    labelImageLabelsFile->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelImageLabelsFile->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelImageLabelsFile, row, 0, 1, 1);
    controlLayout->addWidget(editImageLabelsFile, row, 1, 1, editSpan);
    controlLayout->addWidget(buttonDataLabels, row, 1 + editSpan, 1, 1);
    row++;
    QLabel * labelImagehierarchyFile = new QLabel("Label Hierarchy:");
    labelImagehierarchyFile->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelImagehierarchyFile->setAlignment(Qt::AlignLeft);
    editImageHierarchyFile = new QLineEdit("");
    QPushButton * buttonDataHierarchy = new QPushButton(tr("Browse..."), this);
    connect(buttonDataHierarchy, &QAbstractButton::clicked, this, &inference_control::browseDataHierarchy);
    controlLayout->addWidget(labelImagehierarchyFile, row, 0, 1, 1);
    controlLayout->addWidget(editImageHierarchyFile, row, 1, 1, editSpan);
    controlLayout->addWidget(buttonDataHierarchy, row, 1 + editSpan, 1, 1);
    row++;
    QLabel * labelImageFolder = new QLabel("Image Folder:");
    editImageFolder = new QLineEdit("");
    QPushButton * buttonDataFolder = new QPushButton(tr("Browse..."), this);
    connect(buttonDataFolder, &QAbstractButton::clicked, this, &inference_control::browseDataFolder);
    labelImageFolder->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelImageFolder->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelImageFolder, row, 0, 1, 1);
    controlLayout->addWidget(editImageFolder, row, 1, 1, editSpan);
    controlLayout->addWidget(buttonDataFolder, row, 1 + editSpan, 1, 1);
    row++;
    QLabel * labelImageList = new QLabel("Image List:");
    editImageListFile = new QLineEdit("");
    QPushButton * buttonDataFilename = new QPushButton(tr("Browse..."), this);
    connect(buttonDataFilename, &QAbstractButton::clicked, this, &inference_control::browseDataFilename);
    labelImageList->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelImageList->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelImageList, row, 0, 1, 1);
    controlLayout->addWidget(editImageListFile, row, 1, 1, editSpan);
    controlLayout->addWidget(buttonDataFilename, row, 1 + editSpan, 1, 1);
    row++;
    QLabel * labelMaxDataSize = new QLabel("Image Count:");
    editMaxDataSize = new QLineEdit("");
    editMaxDataSize->setValidator(new QIntValidator());
    labelMaxDataSize->setStyleSheet("font-weight: bold; font-style: italic; font-size: 15pt;");
    labelMaxDataSize->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelMaxDataSize, row, 0, 1, 1);
    controlLayout->addWidget(editMaxDataSize, row, 1, 1, 1);
    checkScaledImages = new QCheckBox("Send Resized Images");
    checkScaledImages->setChecked(true);
    controlLayout->addWidget(checkScaledImages, row, 2, 1, 1);
    checkRepeatImages = nullptr;
    if(operationMode) {
        checkRepeatImages = new QCheckBox("Repeat Until Abort");
        checkRepeatImages->setChecked(true);
        controlLayout->addWidget(checkRepeatImages, row, 3, 1, 1);
    }
    row++;

    setLayout(controlLayout);

    // activate based on configuration
    loadConfig();
    modelSelect(comboModelSelect->currentIndex());

    // start timer for update
    QTimer *timer = new QTimer();
    connect(timer, SIGNAL(timeout()), this, SLOT(tick()));
    timer->start(1000);
}

void inference_control::saveConfig()
{
    bool repeat_images = false;
    if(checkRepeatImages && checkRepeatImages->checkState())
        repeat_images = true;
    int maxDataSize = editMaxDataSize->text().toInt();
    if(maxDataSize < 0) {
        repeat_images = true;
        maxDataSize = abs(maxDataSize);
    }
    bool sendScaledImages = false;
    if(checkScaledImages && checkScaledImages->checkState())
        sendScaledImages = true;
    // save configuration
    QString homeFolder = QStandardPaths::standardLocations(QStandardPaths::HomeLocation)[0];
    QFile fileObj(homeFolder + "/" + CONFIGURATION_CACHE_FILENAME);
    if(fileObj.open(QIODevice::WriteOnly)) {
        QTextStream fileOutput(&fileObj);
        fileOutput << BUILD_VERSION << endl;
        fileOutput << editServerHost->text() << endl;
        fileOutput << editServerPort->text() << endl;
        fileOutput << lastModelFile1 << endl;
        fileOutput << lastModelFile2 << endl;
        fileOutput << lastDimH << endl;
        fileOutput << lastDimW << endl;
        fileOutput << lastInverseInputChannelOrder << endl;
        fileOutput << lastPublishMode << endl;
        fileOutput << lastModelName << endl;
        fileOutput << editGPUs->text() << endl;
        fileOutput << editImageLabelsFile->text() << endl;
        fileOutput << editImageHierarchyFile->text() << endl;
        fileOutput << editImageFolder->text() << endl;
        fileOutput << editImageListFile->text() << endl;
        QString text;
        fileOutput << ((maxDataSize > 0) ? text.sprintf("%d", maxDataSize) : "") << endl;
        fileOutput << (repeat_images ? 1 : 0) << endl;
        fileOutput << (sendScaledImages ? 1 : 0) << endl;
    }
    fileObj.close();
}

void inference_control::loadConfig()
{
    // load default configuration
    QString homeFolder = QStandardPaths::standardLocations(QStandardPaths::HomeLocation)[0];
    QFile fileObj(homeFolder + "/" + CONFIGURATION_CACHE_FILENAME);
    if(fileObj.open(QIODevice::ReadOnly)) {
        QTextStream fileInput(&fileObj);
        QString version = fileInput.readLine();
        if(version == BUILD_VERSION) {
            editServerHost->setText(fileInput.readLine());
            editServerPort->setText(fileInput.readLine());
            editModelFile1->setText(fileInput.readLine());
            editModelFile2->setText(fileInput.readLine());
            editDimH->setText(fileInput.readLine());
            editDimW->setText(fileInput.readLine());
            comboInvertInputChannels->setCurrentIndex(fileInput.readLine().toInt());
            comboPublishOptions->setCurrentIndex(fileInput.readLine().toInt());
            editModelName->setText(fileInput.readLine());
            editGPUs->setText(fileInput.readLine());
            editImageLabelsFile->setText(fileInput.readLine());
            editImageHierarchyFile->setText(fileInput.readLine());
            editImageFolder->setText(fileInput.readLine());
            editImageListFile->setText(fileInput.readLine());
            editMaxDataSize->setText(fileInput.readLine());
            bool repeat_images = false;
            if(fileInput.readLine() == "1")
                repeat_images = true;
            if(checkRepeatImages) {
                checkRepeatImages->setChecked(repeat_images);
            }
            else if(repeat_images && editMaxDataSize->text().length() > 0 && editMaxDataSize->text()[0] != '-') {
                editMaxDataSize->setText("-" + editMaxDataSize->text());
            }
            bool sendScaledImages = true;
            if(fileInput.readLine() == "0")
                sendScaledImages = false;
            if(checkScaledImages) {
                checkScaledImages->setChecked(sendScaledImages);
            }
        }
    }
    fileObj.close();
    // save last options
    lastDimW = editDimW->text();
    lastDimH = editDimH->text();
    lastModelFile1 = editModelFile1->text();
    lastModelFile2 = editModelFile2->text();
    lastInverseInputChannelOrder = comboInvertInputChannels->currentIndex();
    lastPublishMode = comboPublishOptions->currentIndex();
    lastModelName = editModelName->text();
}

bool inference_control::isConfigValid(QPushButton * button, QString& err)
{
    if(editServerHost->text().length() <= 0) { err = "Server invalid server host"; editServerHost->setFocus(); return false; }
    if(editServerPort->text().toInt() <= 0) { err = "Server: invalid server port"; editServerPort->setFocus(); return false; }
    if(button == buttonCompile) {
        if(comboModelSelect->currentIndex() < numModelTypes) {
            if(!QFileInfo(editModelFile1->text()).isFile()) {
                err = typeModelFile1Label[comboModelSelect->currentIndex()] + editModelFile1->text() + " file doesn't exist.";
                editModelFile1->setFocus();
                return false;
            }
            if(!QFileInfo(editModelFile2->text()).isFile()) {
                err = typeModelFile2Label[comboModelSelect->currentIndex()] + editModelFile2->text() + " file doesn't exist.";
                editModelFile2->setFocus();
                return false;
            }
            if(editDimW->text().toInt() <= 0) { err = "Dimensions: width must be positive."; editDimW->setFocus(); return false; }
            if(editDimH->text().toInt() <= 0) { err = "Dimensions: height must be positive."; editDimH->setFocus(); return false; }
            if((comboPublishOptions->currentIndex() >= 1) && (editModelName->text().length() < 4)) {
                editModelName->setFocus();
                err = "modelName is invalid or too small.";
                return false;
            }
            if((comboPublishOptions->currentIndex() >= 1) && (editServerPassword->text().length() == 0)) {
                editServerPassword->setFocus();
                err = "Need to enter password to publish. Note that the default server password is 'radeon'.";
                return false;
            }
        }
    }
    if(button == buttonInference) {
        if(editGPUs->text().toInt() <= 0) { err = "GPUs: must be positive."; editGPUs->setFocus(); return false; }
    }
    return true;
}

void inference_control::modelSelect(int model)
{
    QString text;
    bool compilationCompleted = (compiler_status.errorCode > 0) && compiler_status.completed;
    int dimOutput[3] = { compiler_status.dimOutput[0], compiler_status.dimOutput[1], compiler_status.dimOutput[2] };
    QString modelName;
    if(model < numModelTypes) {
        // input dimensions
        editDimW->setDisabled(false);
        editDimH->setDisabled(false);
        // model file selection
        if(connectionSuccessful && editModelFile1->text().length() > 0 && editModelFile2->text().length() > 0) {
            buttonCompile->setEnabled(true);
            buttonCompile->setStyleSheet("font-weight: bold; color: darkblue; background-color: lightblue;");
        }
        else {
            buttonCompile->setEnabled(false);
            buttonCompile->setStyleSheet("font-weight: normal; color: gray;");
        }
        labelModelFile1->setText(typeModelFile1Label[model]);
        if(editModelFile1->text() != lastModelFile1)
            editModelFile1->setText(lastModelFile1);
        editModelFile1->setEnabled(true);
        buttonModelFile1->setEnabled(true);
        labelModelFile2->setText(typeModelFile2Label[model]);
        if(editModelFile2->text() != lastModelFile2)
            editModelFile2->setText(lastModelFile2);
        editModelFile2->setEnabled(true);
        buttonModelFile2->setEnabled(true);
        comboInvertInputChannels->setDisabled(false);
        if(comboInvertInputChannels->currentIndex() != lastInverseInputChannelOrder)
            comboInvertInputChannels->setCurrentIndex(lastInverseInputChannelOrder);
        comboPublishOptions->setDisabled(false);
        if(comboPublishOptions->currentIndex() != lastPublishMode)
            comboPublishOptions->setCurrentIndex(lastPublishMode);
        if(comboPublishOptions->currentIndex() >= 1) {
            editModelName->setDisabled(false);
            if(editModelName->text() != lastModelName)
                editModelName->setText(lastModelName);
            editServerPassword->setDisabled(false);
        }
        else {
            editModelName->setDisabled(true);
            editModelName->setText("");
            editServerPassword->setDisabled(true);
            editServerPassword->setText("");
        }
        if(compiler_status.completed && compiler_status.errorCode > 0) {
            modelName = compiler_status.message;
        }
        editPreprocessMpyC0->setDisabled(false);
        editPreprocessMpyC1->setDisabled(false);
        editPreprocessMpyC2->setDisabled(false);
        editPreprocessAddC0->setDisabled(false);
        editPreprocessAddC1->setDisabled(false);
        editPreprocessAddC2->setDisabled(false);
        if(editPreprocessMpyC0->text() != lastPreprocessMpy[0])
            editPreprocessMpyC0->setText(lastPreprocessMpy[0]);
        if(editPreprocessMpyC1->text() != lastPreprocessMpy[1])
            editPreprocessMpyC1->setText(lastPreprocessMpy[1]);
        if(editPreprocessMpyC2->text() != lastPreprocessMpy[2])
            editPreprocessMpyC2->setText(lastPreprocessMpy[2]);
        if(editPreprocessAddC0->text() != lastPreprocessAdd[0])
            editPreprocessAddC0->setText(lastPreprocessAdd[0]);
        if(editPreprocessAddC1->text() != lastPreprocessAdd[1])
            editPreprocessAddC1->setText(lastPreprocessAdd[1]);
        if(editPreprocessAddC2->text() != lastPreprocessAdd[2])
            editPreprocessAddC2->setText(lastPreprocessAdd[2]);
    }
    else {
        model -= numModelTypes;
        // already compiled
        compilationCompleted = true;
        dimOutput[0] = modelList[model].outputDim[0];
        dimOutput[1] = modelList[model].outputDim[1];
        dimOutput[2] = modelList[model].outputDim[2];
        // input & output dimensions
        editDimW->setDisabled(true);
        editDimH->setDisabled(true);
        editDimW->setText(text.sprintf("%d", modelList[model].inputDim[0]));
        editDimH->setText(text.sprintf("%d", modelList[model].inputDim[1]));
        // model file selection
        labelModelFile1->setText("--");
        editModelFile1->setEnabled(false);
        editModelFile1->setText("");
        buttonModelFile1->setEnabled(false);
        labelModelFile2->setText("--");
        editModelFile2->setEnabled(false);
        editModelFile2->setText("");
        buttonModelFile2->setEnabled(false);
        buttonCompile->setEnabled(false);
        buttonCompile->setStyleSheet("font-weight: normal; color: gray;");
        comboInvertInputChannels->setDisabled(true);
        comboInvertInputChannels->setCurrentIndex(modelList[model].reverseInputChannelOrder);
        comboPublishOptions->setDisabled(true);
        comboPublishOptions->setCurrentIndex(0);
        editModelName->setDisabled(true);
        editModelName->setText("");
        editServerPassword->setDisabled(true);
        editServerPassword->setText("");
        editPreprocessMpyC0->setDisabled(true);
        editPreprocessMpyC1->setDisabled(true);
        editPreprocessMpyC2->setDisabled(true);
        editPreprocessAddC0->setDisabled(true);
        editPreprocessAddC1->setDisabled(true);
        editPreprocessAddC2->setDisabled(true);
        editPreprocessMpyC0->setText(text.sprintf("%g", modelList[model].preprocessMpy[0]));
        editPreprocessMpyC1->setText(text.sprintf("%g", modelList[model].preprocessMpy[1]));
        editPreprocessMpyC2->setText(text.sprintf("%g", modelList[model].preprocessMpy[2]));
        editPreprocessAddC0->setText(text.sprintf("%g", modelList[model].preprocessAdd[0]));
        editPreprocessAddC1->setText(text.sprintf("%g", modelList[model].preprocessAdd[1]));
        editPreprocessAddC2->setText(text.sprintf("%g", modelList[model].preprocessAdd[2]));
    }
    if(modelName.length() > 0) {
        labelCompilerStatus->setText("[" + modelName + "]*");
    }
    else {
        labelCompilerStatus->setText("");
    }
    // output dimensions
    for(int i = 0; i < 3; i++) {
        text = "";
        if(dimOutput[i] != 0)
            text.sprintf("%d", dimOutput[i]);
        if(i == 0 && editOutDimW->text() != text)
            editOutDimW->setText(text);
        if(i == 1 && editOutDimH->text() != text)
            editOutDimH->setText(text);
        if(i == 2 && editOutDimC->text() != text)
            editOutDimC->setText(text);
    }
    // enable GPUs
    editGPUs->setEnabled(compilationCompleted);
    // enable run button
    if(compilationCompleted && dimOutput[0] > 0 && dimOutput[1] > 0 && dimOutput[2] > 0 &&
       editImageLabelsFile->text().length() > 0 && editImageFolder->text().length() > 0)
    {
        buttonInference->setEnabled(true);
        buttonInference->setStyleSheet("font-weight: bold; color: darkgreen; background-color: lightgreen");
    }
    else {
        buttonInference->setEnabled(false);
        buttonInference->setStyleSheet("font-weight: normal; color: gray;");
    }
}

void inference_control::tick()
{
    modelSelect(comboModelSelect->currentIndex());
}

void inference_control::onChangeDimH(const QString & text)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastDimH =  text;
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangeDimW(const QString & text)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastDimW =  text;
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangeModelFile1(const QString & text)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastModelFile1 =  text;
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangeModelFile2(const QString & text)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastModelFile2 =  text;
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangeInputInverserOrder(int order)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastInverseInputChannelOrder = order;
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangePreprocessMpyC0(const QString& text)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastPreprocessMpy[0] = text;
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangePreprocessMpyC1(const QString& text)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastPreprocessMpy[1] = text;
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangePreprocessMpyC2(const QString& text)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastPreprocessMpy[2] = text;
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangePreprocessAddC0(const QString& text)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastPreprocessAdd[0] = text;
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangePreprocessAddC1(const QString& text)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastPreprocessAdd[1] = text;
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangePreprocessAddC2(const QString& text)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastPreprocessAdd[2] = text;
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangePublishMode(int mode)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastPublishMode = mode;
        comboPublishOptions->setCurrentIndex(mode);
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::onChangeModelName(const QString & text)
{
    if(comboModelSelect->currentIndex() < numModelTypes) {
        lastModelName =  text;
    }
}

void inference_control::browseModelFile1()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), nullptr, typeModelFile1Desc[modelType]);
    if(fileName.size() > 0) {
        editModelFile1->setText(fileName);
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::browseModelFile2()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), nullptr, typeModelFile2Desc[modelType]);
    if(fileName.size() > 0) {
        editModelFile2->setText(fileName);
        modelSelect(comboModelSelect->currentIndex());
    }
}

void inference_control::browseShadowFolder()
{

    QString dir = QFileDialog::getExistingDirectory(this, tr("Select Shadow Folder Root on Client"), nullptr,
                        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if(dir.size() > 0)
        editShadowFolderAddr->setText(dir);
}

void inference_control::browseDataLabels()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Labels File"), nullptr, tr("Labels Text (*.txt)"));
    if(fileName.size() > 0)
        editImageLabelsFile->setText(fileName);
}

void inference_control::browseDataHierarchy()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Label Hierarchy File"), nullptr, tr("Label Hierarchy Text (*.csv)"));
    if(fileName.size() > 0)
        editImageHierarchyFile->setText(fileName);
}

void inference_control::browseDataFolder()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Select Image Folder"), nullptr,
                        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if(dir.size() > 0)
        editImageFolder->setText(dir);
}

void inference_control::browseDataFilename()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image List File"), nullptr, tr("Image List Text (*.txt);;Image List Text (*.csv)"));
    if(fileName.size() > 0)
        editImageListFile->setText(fileName);
}

void inference_control::exitControl()
{
    close();
}

void inference_control::runConnection()
{
    // check configuration
    QString err;
    if(!isConfigValid(buttonConnect, err)) {
        QMessageBox::critical(this, windowTitle(), err, QMessageBox::Ok);
        return;
    }
    // save configuration
    saveConfig();

    // start server connection
    TcpConnection * connection = new TcpConnection(editServerHost->text(), editServerPort->text().toInt(), 3000, this);
    // initialize default values
    connectionSuccessful = false;
    QString status = "ERROR: connect to " + editServerHost->text() + ":" + editServerPort->text() + " is not successful";
    int pendingModelCount = 0;
    // loop to process commands from server
    InfComCommand cmd;
    while(connection->recvCmd(cmd)) {
        if(cmd.magic != INFCOM_MAGIC) {
            status.sprintf("ERROR: got invalid magic 0x%08x", cmd.magic);
            break;
        }
        if(cmd.command == INFCOM_CMD_DONE) {
            connection->sendCmd(cmd);
            break;
        }
        else if(cmd.command == INFCOM_CMD_SEND_MODE) {
            InfComCommand reply = {
                INFCOM_MAGIC, INFCOM_CMD_SEND_MODE,
                { INFCOM_MODE_CONFIGURE },
                { 0 }, { 0 }
            };
            connection->sendCmd(reply);
        }
        else if(cmd.command == INFCOM_CMD_CONFIG_INFO) {
            connection->sendCmd(cmd);
            pendingModelCount = cmd.data[0];
            maxGPUs = cmd.data[1];
            enableSF = cmd.data[2];
            QString text;
            editGPUs->setText(text.sprintf("%d", maxGPUs));
            editGPUs->setValidator(new QIntValidator(1,maxGPUs));
            labelMaxGPUs->setText(text.sprintf("(upto %d)", maxGPUs));
            if(!enableSF) {checkShadowFolder->setEnabled(false); editShadowFolderAddr->setEnabled(false); buttonShadowFolder->setEnabled(false);}
            while(comboModelSelect->count() > 1)
                comboModelSelect->removeItem(1);
            modelList.clear();
            connectionSuccessful = true;
            status = "OK: Connected to " + editServerHost->text() + ":" + editServerPort->text();
            if(pendingModelCount <= 0) {
                break;
            }
        }
        else if(cmd.command == INFCOM_CMD_MODEL_INFO) {
            connection->sendCmd(cmd);
            ModelInfo info = {
                cmd.message,
                { cmd.data[0], cmd.data[1], cmd.data[2] },
                { cmd.data[3], cmd.data[4], cmd.data[5] },
                cmd.data[6],
                { *(float *)&cmd.data[7], *(float *)&cmd.data[8], *(float *)&cmd.data[9] },
                { *(float *)&cmd.data[10], *(float *)&cmd.data[11], *(float *)&cmd.data[12] }
            };
            modelList.push_back(info);
            comboModelSelect->addItem(info.name);
            pendingModelCount--;
            if(pendingModelCount <= 0) {
                break;
            }
        }
        else {
            status.sprintf("ERROR: got invalid command received 0x%08x", cmd.command);
            break;
        }
    }
    connection->close();
    delete connection;
    labelServerStatus->setText(status);

    // update status
    if(comboModelSelect->currentIndex() > modelList.length()) {
        comboModelSelect->setCurrentIndex(modelList.length() - 1);
    }
    modelSelect(comboModelSelect->currentIndex());
}

void inference_control::runCompiler()
{
    // check configuration
    QString err;
    if(!isConfigValid(buttonCompile, err)) {
        QMessageBox::critical(this, windowTitle(), err, QMessageBox::Ok);
        return;
    }
    buttonCompile->setEnabled(false);

    // save configuration
    saveConfig();

    // start compiler
    QString options;
    if(comboPublishOptions->currentIndex() >= 1) {
        options = "save=" + editModelName->text();
        if(comboPublishOptions->currentIndex() == 2) {
            options += ",override";
        }
        if(editServerPassword->text().length() > 0) {
            options += ",passwd=" + editServerPassword->text();
        }
    }
    float preprocessMpy[3] = {
        editPreprocessMpyC0->text().toFloat(),
        editPreprocessMpyC1->text().toFloat(),
        editPreprocessMpyC2->text().toFloat()
    };
    float preprocessAdd[3] = {
        editPreprocessAddC0->text().toFloat(),
        editPreprocessAddC1->text().toFloat(),
        editPreprocessAddC2->text().toFloat()
    };
    inference_compiler * compiler = new inference_compiler(
                true,
                editServerHost->text(), editServerPort->text().toInt(),
                3,
                editDimH->text().toInt(),
                editDimW->text().toInt(),
                editModelFile1->text(), editModelFile2->text(),
                comboInvertInputChannels->currentIndex(),
                preprocessMpy, preprocessAdd,
                options,
                &compiler_status);
    compiler->show();
}

void inference_control::runInference()
{
    // check configuration
    QString err;
    if(isConfigValid(buttonInference, err)) {
        if(!QFileInfo(editImageFolder->text()).isDir())
            err = "Image Folder: doesn't exist: " + editImageFolder->text();
        else {
            QFile fileObj(editImageLabelsFile->text());
            if(fileObj.open(QIODevice::ReadOnly)) {
                QTextStream fileInput(&fileObj);
                dataLabels->clear();
                while (!fileInput.atEnd()) {
                    QString line = fileInput.readLine();
                    line = line.trimmed();
                    if(line.size() > 0)
                        dataLabels->push_back(line);
                }
                if(dataLabels->size() != editOutDimC->text().toInt()) {
                    err.sprintf("Labels: need %d labels in %s: found %d", editOutDimC->text().toInt(),
                            editImageLabelsFile->text().toStdString().c_str(), dataLabels->size());
                }
            }
            else {
                err = "Labels: unable to open: " + editImageLabelsFile->text();
            }
        }
        if(!editImageHierarchyFile->text().isEmpty()){
            QFile fileObj(editImageHierarchyFile->text());
            if(fileObj.open(QIODevice::ReadOnly)) {
                QTextStream fileInput(&fileObj);
                dataHierarchy->clear();
                while (!fileInput.atEnd()) {
                    QString line = fileInput.readLine();
                    line = line.trimmed();
                    if(line.size() > 0)
                        dataHierarchy->push_back(line);
                }
                if(dataHierarchy->size() != editOutDimC->text().toInt()) {
                    err.sprintf("Labels Hierarchy: need %d label Hierarchy in %s: found %d", editOutDimC->text().toInt(),
                                editImageHierarchyFile->text().toStdString().c_str(), dataHierarchy->size());
                }
            }
            else {
                err = "Label Hierarchy: unable to open: " + editImageHierarchyFile->text();
            }
        }
    }
    if(err.length() > 0) {
        QMessageBox::critical(this, windowTitle(), err, QMessageBox::Ok);
        return;
    }

    // save configuration
    saveConfig();

    // start viewer
    QString modelName = comboModelSelect->currentText();
    if(comboModelSelect->currentIndex() < numModelTypes) {
        modelName = compiler_status.message;
    }
    int dimInput[3] = { editDimW->text().toInt(), editDimH->text().toInt(), 3 };
    int dimOutput[3] = { editOutDimW->text().toInt(), editOutDimH->text().toInt(), editOutDimC->text().toInt() };
    bool repeat_images = false;
    if(checkRepeatImages && checkRepeatImages->checkState())
        repeat_images = true;
    int maxDataSize = editMaxDataSize->text().toInt();
    if(maxDataSize < 0) {
        repeat_images = true;
        if(maxDataSize == -1)
            maxDataSize = 0;
        else
            maxDataSize = abs(maxDataSize);
    }
    bool sendScaledImages = false;
    if(checkScaledImages && checkScaledImages->checkState())
        sendScaledImages = true;
    if(enableTopK)
        topKValue = ( comboTopKResult->currentIndex() + 1 );

    decodeMode = comboDecodeMode->currentIndex();
    inference_panel *display_panel = new inference_panel;
    display_panel->setWindowIcon(QIcon(":/images/vega_icon_150.png"));
    //display_panel->show();

    sendFileName = (decodeMode == 1); // rocAL decode only suports sendFileName mode
    
    inference_viewer * viewer = new inference_viewer(
                editServerHost->text(), editServerPort->text().toInt(), modelName,
                dataLabels, dataHierarchy, editImageListFile->text(), editImageFolder->text(),
                dimInput, editGPUs->text().toInt(), dimOutput, maxDataSize, repeat_images, sendScaledImages, sendFileName, topKValue, decodeMode);
    viewer->setWindowIcon(QIcon(":/images/vega_icon_150.png"));
    viewer->show();
    close();
}

void inference_control::onLogo1Click()
{
    QDesktopServices::openUrl(QUrl("http://www.amd.com/en"));
}

void inference_control::onLogo2Click()
{
    QDesktopServices::openUrl(QUrl("https://instinct.radeon.com/en/"));
}

void inference_control::topKResultsEnable(bool topKEnable)
{
    if(topKEnable){
        comboTopKResult->setEnabled(true);
        comboTopKResult->setCurrentIndex(0);
        enableTopK = 1;
    }
    else
    {
        comboTopKResult->setEnabled(false);
        comboTopKResult->setCurrentIndex(0);
        enableTopK = 0;
    }
}

void inference_control::shadowFolderEnable(bool shadowEnable)
{
    if(shadowEnable){
        editShadowFolderAddr->setVisible(false);
        buttonShadowFolder->setVisible(false);
        sendFileName = 1;
    }
    else
    {
        editShadowFolderAddr->setVisible(false);
        buttonShadowFolder->setVisible(false);
        sendFileName = 0;
    }
}
