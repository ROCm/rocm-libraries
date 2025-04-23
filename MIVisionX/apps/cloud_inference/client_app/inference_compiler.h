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

#ifndef INFERENCE_COMPILER_H
#define INFERENCE_COMPILER_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <mutex>

struct inference_compiler_status {
    bool completed;
    int errorCode;
    int modelFile1UploadProgress;
    int modelFile2UploadProgress;
    int compilationProgress;
    int dimOutput[3];
    QString message;
};

class inference_model_uploader : public QObject
{
    Q_OBJECT
public:
    explicit inference_model_uploader(
            bool enableServer,
            QString serverHost, int serverPort,
            int c, int h, int w,
            QString modelFile1, QString modelFile2,
            int reverseInputChannelOrder,
            float preprocessMpy[3],
            float preprocessAdd[3],
            QString compilerOptions,
            inference_compiler_status * progress,
            QObject *parent = nullptr);
    ~inference_model_uploader();

    static void abort();

signals:
    void finished();
    void error(QString err);

public slots:
    void run();

private:
    static bool abortRequested;

private:
    std::mutex mutex;
    // config
    bool enableServer;
    QString serverHost;
    int serverPort;
    int dimC;
    int dimH;
    int dimW;
    QString modelFile1;
    QString modelFile2;
    int reverseInputChannelOrder;
    float preprocessMpy[3];
    float preprocessAdd[3];
    QString compilerOptions;
    inference_compiler_status * progress;
};

class inference_compiler : public QWidget
{
    Q_OBJECT
public:
    explicit inference_compiler(
            bool enableServer,
            QString serverHost, int serverPort,
            int c, int h, int w,
            QString modelFile1, QString modelFile2,
            int reverseInputChannelOrder,
            float preprocessMpy[3],
            float preprocessAdd[3],
            QString compilerOptions,
            inference_compiler_status * progress,
            QWidget *parent = nullptr);

protected:
    void Ok();
    void Cancel();
    void startModelUploader();

signals:

public slots:
    void tick();
    void errorString(QString err);

private:
    // config
    bool enableServer;
    QString serverHost;
    int serverPort;
    int dimC;
    int dimH;
    int dimW;
    QString modelFile1;
    QString modelFile2;
    int reverseInputChannelOrder;
    float preprocessMpy[3];
    float preprocessAdd[3];
    QString compilerOptions;
    // status
    QLabel * labelStatus;
    QLineEdit * editModelFile1UploadProgress;
    QLineEdit * editModelFile2UploadProgress;
    QLineEdit * editCompilerProgress;
    QLineEdit * editDimC;
    QLineEdit * editDimH;
    QLineEdit * editDimW;
    QLineEdit * editOutDimC;
    QLineEdit * editOutDimH;
    QLineEdit * editOutDimW;
    QLineEdit * editCompilerMessage;
    QPushButton * okCompilerButton;
    QPushButton * cancelCompilerButton;
    inference_compiler_status * progress;
    inference_model_uploader * worker;
};

#endif // INFERENCE_COMPILER_H
