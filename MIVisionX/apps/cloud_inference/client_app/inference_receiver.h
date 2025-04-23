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

#ifndef INFERENCE_RECEIVER_H
#define INFERENCE_RECEIVER_H

#include <QObject>
#include <QVector>
#include <QQueue>
#include <QMutex>
#include <QElapsedTimer>
#include <QMouseEvent>
#include <mutex>

struct runtime_receiver_status {
    bool completed;
    int errorCode;
    QString message;
    bool repeat_images;
    bool completed_send;
    bool completed_decode;
    bool completed_load;
    int images_loaded;
    int images_decoded;
    int images_sent;
    int images_received;
};

class inference_receiver : public QObject
{
    Q_OBJECT
public:
    explicit inference_receiver(
            QString serverHost, int serverPort, QString modelName,
            int GPUs, int * inputDim, int * outputDim, const char * runtimeOptions,
            QVector<QByteArray> * imageBuffer,
            runtime_receiver_status * progress, int sendFileName, int topKValue,
            QVector<QString> * shadowFileBuffer, int decodeMode, QString dataFolder,
            QObject *parent = nullptr);
    ~inference_receiver();

    static void abort();
    void setImageCount(int imageCount, int labelCount, QVector<QString> * dataLabels);
    void getReceivedList(QVector<int>& indexQ, QVector<int>& labelQ, QVector<QString>& summaryQ,
                         QVector<QVector<int> >& labelTopK, QVector<QVector<float> >& probTopK);
    float getPerfImagesPerSecond();
    
signals:
    void finished();
    void error(QString err);

public slots:
    void run();

private:
    static bool abortRequested;

private:
    std::mutex mutex;
    int imageCount;
    int labelCount;
    QQueue<int> imageIndex;
    QQueue<int> imageLabel;
    QQueue<QString> imageSummary;
    QQueue<QVector<int>> imageTopkLabels;
    QQueue<QVector<float>> imageTopkConfidence;
    QElapsedTimer perfTimer;
    int perfImageCount;
    float perfRate;
    QVector<QByteArray> * imageBuffer;
    QVector<QString> * dataLabels;
    QVector<QString> * shadowFileBuffer;
    QString serverHost;
    int serverPort;
    QString modelName;
    int GPUs;
    int * inputDim;
    int * outputDim;
    const char * runtimeOptions;
    runtime_receiver_status * progress;
    int sendFileName;
    int topKValue;
    int decodeMode;
    QString dataFolder;
};

#endif // INFERENCE_RECEIVER_H
