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

#include "inference_compiler.h"
#include "inference_control.h"
#include "tcpconnection.h"
#include <QGridLayout>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QFileDialog>
#include <QLabel>
#include <QStandardPaths>
#include <QFile>
#include <QTextStream>
#include <QIntValidator>
#include <QMessageBox>
#include <QFileInfo>
#include <QFrame>
#include <QTimer>
#include <QThread>

bool inference_model_uploader::abortRequested = false;

void inference_model_uploader::abort()
{
    abortRequested = true;
}

inference_model_uploader::inference_model_uploader(
        bool enableServer_,
        QString serverHost_, int serverPort_,
        int c, int h, int w,
        QString modelFile1_, QString modelFile2_,
        int reverseInputChannelOrder_,
        float preprocessMpy_[3],
        float preprocessAdd_[3],
        QString compilerOptions_,
        inference_compiler_status * progress_,
        QObject *parent) : QObject(parent)
{
    enableServer = enableServer_;
    serverHost = serverHost_;
    serverPort = serverPort_;
    dimC = c;
    dimH = h;
    dimW = w;
    modelFile1 = modelFile1_;
    modelFile2 = modelFile2_;
    reverseInputChannelOrder = reverseInputChannelOrder_;
    preprocessMpy[0] = preprocessMpy_[0];
    preprocessMpy[1] = preprocessMpy_[1];
    preprocessMpy[2] = preprocessMpy_[2];
    preprocessAdd[0] = preprocessAdd_[0];
    preprocessAdd[1] = preprocessAdd_[1];
    preprocessAdd[2] = preprocessAdd_[2];
    compilerOptions = compilerOptions_;
    progress = progress_;
    abortRequested = false;
}

inference_model_uploader::~inference_model_uploader()
{

}

void inference_model_uploader::run()
{
    // connect to the server for inference compiler mode
    //    - configure the connection in inference compiler mode
    //    - upload File#1, File#2, and configuration parameters
    //    - update status of remote compilation process

    if(enableServer)
    {
        // start server connection
        TcpConnection * connection = new TcpConnection(serverHost, serverPort, 3000, this);
        if(connection->connected()) {
            InfComCommand cmd;
            while(connection->recvCmd(cmd)) {
                if(cmd.magic != INFCOM_MAGIC) {
                    progress->errorCode = -1;
                    progress->message.sprintf("ERROR: got invalid magic 0x%08x", cmd.magic);
                    break;
                }
                else if(cmd.command == INFCOM_CMD_DONE) {
                    connection->sendCmd(cmd);
                    progress->errorCode = cmd.data[0];
                    progress->message.sprintf("%s", cmd.message);
                    break;
                }
                else if(cmd.command == INFCOM_CMD_SEND_MODE) {
                    InfComCommand reply = {
                        INFCOM_MAGIC, INFCOM_CMD_SEND_MODE,
                        { INFCOM_MODE_COMPILER, dimW, dimH, dimC, 0,
                          reverseInputChannelOrder,
                          *(int *)&preprocessMpy[0], *(int *)&preprocessMpy[1], *(int *)&preprocessMpy[2],
                          *(int *)&preprocessAdd[0], *(int *)&preprocessAdd[1], *(int *)&preprocessAdd[2]
                        },
                        { 0 }
                    };
                    strncpy(reply.message, compilerOptions.toStdString().c_str(), sizeof(reply.message));
                    connection->sendCmd(reply);
                }
                else if(cmd.command == INFCOM_CMD_SEND_MODELFILE1) {
                    if(!connection->sendFile(INFCOM_CMD_SEND_MODELFILE1, modelFile1, progress->modelFile1UploadProgress, progress->message, abortRequested)) {
                        progress->errorCode = -1;
                        break;
                    }
                }
                else if(cmd.command == INFCOM_CMD_SEND_MODELFILE2) {
                    if(!connection->sendFile(INFCOM_CMD_SEND_MODELFILE2, modelFile2, progress->modelFile2UploadProgress, progress->message, abortRequested)) {
                        progress->errorCode = -2;
                        break;
                    }
                }
                else if(cmd.command == INFCOM_CMD_COMPILER_STATUS) {
                    connection->sendCmd(cmd);
                    progress->completed = (cmd.data[0] != 0) ? true : false;
                    progress->errorCode = cmd.data[0];
                    progress->compilationProgress = cmd.data[1];
                    progress->dimOutput[0] = cmd.data[2];
                    progress->dimOutput[1] = cmd.data[3];
                    progress->dimOutput[2] = cmd.data[4];
                    progress->message = cmd.message;
                    if(progress->completed)
                        break;
                }
                else {
                    progress->errorCode = -1;
                    progress->message.sprintf("ERROR: got invalid command 0x%08x", cmd.command);
                    break;
                }
            }
        }
        else {
            progress->errorCode = -1;
            progress->message.sprintf("ERROR: Unable to connect to %s:%d", serverHost.toStdString().c_str(), serverPort);
        }
        if(abortRequested)
            progress->message += " [aborted]";
        connection->close();
        delete connection;
        progress->completed = true;
    }
    else {
        int counter = 0;
        while(!abortRequested) {
           counter++;
           progress->modelFile1UploadProgress = 1.0/counter;
           progress->modelFile2UploadProgress = counter/16.0;
           progress->compilationProgress = counter;
           progress->message.sprintf("Message 101 with tick at %d", counter);
           if(counter == 600) {
               progress->dimOutput[0] = 1;
               progress->dimOutput[1] = 1;
               progress->dimOutput[2] = 1000;
           }
           if(counter == 1000) progress->completed = true;
           QThread::msleep(2);
        }
    }

    emit finished();
}

void inference_compiler::startModelUploader()
{
    // start receiver thread
    QThread * thread = new QThread;
    worker = new inference_model_uploader(enableServer, serverHost, serverPort,
                        dimC, dimH, dimW, modelFile1, modelFile2,
                        reverseInputChannelOrder, preprocessMpy, preprocessAdd,
                        compilerOptions,
                        progress);
    worker->moveToThread(thread);
    connect(worker, SIGNAL (error(QString)), this, SLOT (errorString(QString)));
    connect(thread, SIGNAL (started()), worker, SLOT (run()));
    connect(worker, SIGNAL (finished()), thread, SLOT (quit()));
    connect(worker, SIGNAL (finished()), worker, SLOT (deleteLater()));
    connect(thread, SIGNAL (finished()), thread, SLOT (deleteLater()));
    thread->start();
    thread->terminate();
}

inference_compiler::inference_compiler(
        bool enableServer_,
        QString serverHost_, int serverPort_,
        int c, int h, int w,
        QString modelFile1_, QString modelFile2_,
        int reverseInputChannelOrder_,
        float preprocessMpy_[3],
        float preprocessAdd_[3],
        QString compilerOptions_,
        inference_compiler_status * progress_,
        QWidget *parent) : QWidget(parent)
{
    setWindowTitle("Inference Compiler");
    setMinimumWidth(800);

    enableServer = enableServer_;
    serverHost = serverHost_;
    serverPort = serverPort_;
    dimC = c;
    dimH = h;
    dimW = w;
    modelFile1 = modelFile1_;
    modelFile2 = modelFile2_;
    reverseInputChannelOrder = reverseInputChannelOrder_;
    preprocessMpy[0] = preprocessMpy_[0];
    preprocessMpy[1] = preprocessMpy_[1];
    preprocessMpy[2] = preprocessMpy_[2];
    preprocessAdd[0] = preprocessAdd_[0];
    preprocessAdd[1] = preprocessAdd_[1];
    preprocessAdd[2] = preprocessAdd_[2];
    compilerOptions = compilerOptions_;
    progress = progress_;

    // status
    worker = nullptr;
    progress->completed = false;
    progress->errorCode = 0;
    progress->modelFile1UploadProgress = 0;
    progress->modelFile2UploadProgress = 0;
    progress->compilationProgress = 0;
    progress->message = "";
    progress->dimOutput[0] = 0;
    progress->dimOutput[1] = 0;
    progress->dimOutput[2] = 0;

    // GUI

    QGridLayout * controlLayout = new QGridLayout;
    int editSpan = 5;
    int row = 0;

    //////////////
    /// \brief labelIntro
    ///
    QString text;
    QLabel * labelIntro = new QLabel("Inference Compiler: remote connection to " + serverHost + ":" + text.sprintf("%d", serverPort));
    labelIntro->setStyleSheet("font-weight: bold; color: green");
    labelIntro->setAlignment(Qt::AlignCenter);
    controlLayout->addWidget(labelIntro, row, 0, 1, editSpan + 2);
    row++;

    QFrame * sepHLine1 = new QFrame();
    sepHLine1->setFrameShape(QFrame::HLine);
    sepHLine1->setFrameShadow(QFrame::Sunken);
    controlLayout->addWidget(sepHLine1, row, 0, 1, editSpan + 2);
    row++;

    //////////////
    /// \brief labelStatus
    ///
    QLabel * labelH1 = new QLabel("File#1 upload");
    QLabel * labelH2 = new QLabel("File#2 upload");
    QLabel * labelH3 = new QLabel("compilation");
    labelH1->setStyleSheet("font-weight: bold; font-style: italic");
    labelH2->setStyleSheet("font-weight: bold; font-style: italic");
    labelH3->setStyleSheet("font-weight: bold; font-style: italic");
    labelH1->setAlignment(Qt::AlignCenter);
    labelH2->setAlignment(Qt::AlignCenter);
    labelH3->setAlignment(Qt::AlignCenter);
    controlLayout->addWidget(labelH1, row, 1, 1, 1);
    controlLayout->addWidget(labelH2, row, 2, 1, 1);
    controlLayout->addWidget(labelH3, row, 3, 1, 1);
    row++;

    labelStatus = new QLabel("Progress:");
    editModelFile1UploadProgress = new QLineEdit("");
    editModelFile2UploadProgress = new QLineEdit("");
    editCompilerProgress = new QLineEdit("");
    editModelFile1UploadProgress->setEnabled(false);
    editModelFile2UploadProgress->setEnabled(false);
    editCompilerProgress->setEnabled(false);
    labelStatus->setStyleSheet("font-weight: bold; font-style: italic");
    labelStatus->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelStatus, row, 0, 1, 1);
    controlLayout->addWidget(editModelFile1UploadProgress, row, 1, 1, 1);
    controlLayout->addWidget(editModelFile2UploadProgress, row, 2, 1, 1);
    controlLayout->addWidget(editCompilerProgress, row, 3, 1, 1);
    row++;

    QLabel * labelInputDim = new QLabel("CxHxW(inp):");
    editDimC = new QLineEdit(text.sprintf("%d", dimC));
    editDimH = new QLineEdit(text.sprintf("%d", dimH));
    editDimW = new QLineEdit(text.sprintf("%d", dimW));
    editDimC->setEnabled(false);
    editDimH->setEnabled(false);
    editDimW->setEnabled(false);
    labelInputDim->setStyleSheet("font-weight: bold; font-style: italic");
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
    labelOutputDim->setStyleSheet("font-weight: bold; font-style: italic");
    labelOutputDim->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelOutputDim, row, 0, 1, 1);
    controlLayout->addWidget(editOutDimC, row, 1, 1, 1);
    controlLayout->addWidget(editOutDimH, row, 2, 1, 1);
    controlLayout->addWidget(editOutDimW, row, 3, 1, 1);
    row++;

    QLabel * labelMessage = new QLabel("Message:");
    editCompilerMessage = new QLineEdit("");
    editCompilerMessage->setEnabled(false);
    labelMessage->setStyleSheet("font-weight: bold; font-style: italic");
    labelMessage->setAlignment(Qt::AlignLeft);
    controlLayout->addWidget(labelMessage, row, 0, 1, 1);
    controlLayout->addWidget(editCompilerMessage, row, 1, 1, editSpan);
    row++;

    //////////////
    /// \brief compilerButtonBox
    ///
    QDialogButtonBox * compilerButtonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(compilerButtonBox, &QDialogButtonBox::accepted, this, &inference_compiler::Ok);
    connect(compilerButtonBox, &QDialogButtonBox::rejected, this, &inference_compiler::Cancel);
    okCompilerButton = compilerButtonBox->button(QDialogButtonBox::Ok);
    cancelCompilerButton = compilerButtonBox->button(QDialogButtonBox::Cancel);
    okCompilerButton->setText("Running...");
    okCompilerButton->setEnabled(false);
    cancelCompilerButton->setText("Abort");
    controlLayout->addWidget(compilerButtonBox, row, (1 + editSpan)/2, 1, 1);
    row++;

    setLayout(controlLayout);

    // start timer for update
    QTimer *timer = new QTimer();
    connect(timer, SIGNAL(timeout()), this, SLOT(tick()));
    timer->start(200);

    // start model uploader
    startModelUploader();
}

void inference_compiler::errorString(QString err)
{
    qDebug("inference_compiler: %s", err.toStdString().c_str());
}

void inference_compiler::tick()
{
    // update Window
    if(progress->completed) {
        if(progress->errorCode >= 0)
            okCompilerButton->setEnabled(true);
        okCompilerButton->setText("Close");
        cancelCompilerButton->setText("Exit");
        progress->completed = true;
    }
    QString text;
    editModelFile1UploadProgress->setText(text.sprintf("%d%%", progress->modelFile1UploadProgress));
    editModelFile2UploadProgress->setText(text.sprintf("%d%%", progress->modelFile2UploadProgress));
    editCompilerProgress->setText(text.sprintf("%d%%", progress->compilationProgress));
    editOutDimW->setText(text.sprintf("%d", progress->dimOutput[0]));
    editOutDimH->setText(text.sprintf("%d", progress->dimOutput[1]));
    editOutDimC->setText(text.sprintf("%d", progress->dimOutput[2]));
    if(progress->errorCode < 0) {
        editCompilerMessage->setText(text.sprintf("[E%d] %s", progress->errorCode, progress->message.toStdString().c_str()));
    }
    else {
        editCompilerMessage->setText(text.sprintf("%s%s",
                      progress->completed ? "[completed] " : "", progress->message.toStdString().c_str()));
        if(progress->modelFile1UploadProgress == 100 && progress->modelFile2UploadProgress == 100 &&
           progress->compilationProgress == 100 && progress->completed)
        {
            close();
        }
    }
}

void inference_compiler::Ok()
{
    inference_model_uploader::abort();
    close();
}

void inference_compiler::Cancel()
{
    inference_model_uploader::abort();
    exit(1);
}
