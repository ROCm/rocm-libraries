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

#include "inference_receiver.h"
#include "tcpconnection.h"
#include <iostream>
#include <QThread>

bool inference_receiver::abortRequested = false;

void inference_receiver::abort()
{
    abortRequested = true;
}

inference_receiver::inference_receiver(
        QString serverHost_, int serverPort_, QString modelName_,
        int GPUs_, int * inputDim_, int * outputDim_, const char * runtimeOptions_,
        QVector<QByteArray> * imageBuffer_,
        runtime_receiver_status * progress_, int sendFileName_, int topKValue_,
        QVector<QString> * shadowFileBuffer_, int decodeMode_,  QString dataFolder_,
        QObject *parent) : QObject(parent)
{
    perfRate = 0;
    perfImageCount = 0;
    perfTimer.start();
    imageCount = 0;
    labelCount = 0;
    dataLabels = nullptr;
    imageBuffer = imageBuffer_;
    serverHost = serverHost_;
    serverPort = serverPort_;
    modelName = modelName_;
    GPUs = GPUs_;
    inputDim = inputDim_;
    outputDim = outputDim_;
    runtimeOptions = runtimeOptions_;
    progress = progress_;
    sendFileName = sendFileName_;
    topKValue = topKValue_;
    decodeMode = decodeMode_;
    dataFolder = dataFolder_;
    shadowFileBuffer = shadowFileBuffer_;
}

inference_receiver::~inference_receiver()
{

}

void inference_receiver::getReceivedList(QVector<int>& indexQ, QVector<int>& labelQ, QVector<QString>& summaryQ,
                                         QVector<QVector<int> >& labelTopK, QVector<QVector<float> >& probTopK)
{
    std::lock_guard<std::mutex> guard(mutex);
    while(imageIndex.length() > 0) {
        indexQ.push_back(imageIndex.front());
        labelQ.push_back(imageLabel.front());
        summaryQ.push_back(imageSummary.front());
        imageIndex.pop_front();
        imageLabel.pop_front();
        imageSummary.pop_front();
    }
    while (imageTopkLabels.length() > 0)
    {
        labelTopK.push_back(imageTopkLabels.front());
        probTopK.push_back(imageTopkConfidence.front());
        imageTopkLabels.pop_front();
        imageTopkConfidence.pop_front();
    }
}

void inference_receiver::run()
{
    // connect to the server for inference run-time mode
    //    - configure the connection in inference run-time mode
    //    - keep sending images and tag if server can accept more work
    //    - when results are received add the results to imageIndex, imageLabel, imageSummary queues

    progress->images_sent = 0;
    progress->images_received = 0;
    progress->completed_send = false;
    progress->completed = false;

    TcpConnection * connection = new TcpConnection(serverHost, serverPort, 3000, this);
    if(connection->connected()) {
        int nextImageToSend = 0;
        InfComCommand cmd;
        while(!abortRequested && connection->recvCmd(cmd)) {
            if(abortRequested)
                break;
            if(cmd.magic != INFCOM_MAGIC) {
                progress->errorCode = -1;
                progress->message.sprintf("ERROR: got invalid magic 0x%08x", cmd.magic);
                break;
            }
            else if(cmd.command == INFCOM_CMD_DONE) {
                connection->sendCmd(cmd);
                break;
            }
            else if(cmd.command == INFCOM_CMD_INFERENCE_INITIALIZATION) {
                connection->sendCmd(cmd);
                progress->message.sprintf("[%s]", cmd.message);
            }
            else if(cmd.command == INFCOM_CMD_SEND_MODE) {
                InfComCommand reply = {
                    INFCOM_MAGIC, INFCOM_CMD_SEND_MODE,
                    { INFCOM_MODE_INFERENCE, GPUs,
                      inputDim[0], inputDim[1], inputDim[2], outputDim[0], outputDim[1], outputDim[2], sendFileName, topKValue, 0, decodeMode, progress->repeat_images },
                    { 0 }, { 0 }
                };
                QString text = modelName;
                if(runtimeOptions || *runtimeOptions) {
                    text += " ";
                    text += runtimeOptions;
                }
                strncpy(reply.message, text.toStdString().c_str(), sizeof(reply.message));

                QString path = dataFolder;
                strlcpy(reply.path, path.toStdString().c_str(), sizeof(reply.path));
                connection->sendCmd(reply);
            }
            else if(cmd.command == INFCOM_CMD_SEND_IMAGES) {
                progress->message = "";
                int count_requested = cmd.data[0];
                int count = progress->completed_send ? -1 :
                                std::min(imageCount - nextImageToSend, count_requested);
                InfComCommand reply = {
                    INFCOM_MAGIC, INFCOM_CMD_SEND_IMAGES,
                    { count },
                    { 0 }, { 0 }
                };
                if(!connection->sendCmd(reply))
                    break;
                bool failed = false;
                for(int i = 0; i < count; i++) {
                    // send the image at nextImageToSend
                    if (sendFileName) {
                        QByteArray fileNameBuffer;
                        fileNameBuffer.append((*shadowFileBuffer)[nextImageToSend]);
                        if(!connection->sendImage(nextImageToSend, fileNameBuffer, progress->errorCode, progress->message, abortRequested)) {
                            failed = true;
                            break;
                        }
                    }
                    else if(!connection->sendImage(nextImageToSend, (*imageBuffer)[nextImageToSend], progress->errorCode, progress->message, abortRequested)) {
                        failed = true;
                        break;
                    }
                    // update nextImageToSend
                    nextImageToSend++;
                    progress->images_sent++;
                }
                if(failed)
                    break;
                if(nextImageToSend >= imageCount) {
                    if(progress->repeat_images) {
                        nextImageToSend = 0;
                    }
                    else if(progress->completed_load && progress->images_loaded == progress->images_sent) {
                        progress->completed_send = true;
                    }
                }
            }
            else if(cmd.command == INFCOM_CMD_INFERENCE_RESULT) {
                connection->sendCmd(cmd);
                int count = cmd.data[0];
                int status = cmd.data[1];
                if(status == 0 && count > 0 && count < (int)((sizeof(cmd)-16)/(2*sizeof(int)))) {
                    std::lock_guard<std::mutex> guard(mutex);
                    for(int i = 0; i < count; i++) {
                        int tag = cmd.data[2 + 2*i + 0];
                        int label = cmd.data[2 + 2*i + 1];
                        imageIndex.push_back(tag);
                        imageLabel.push_back(label);
                        if(dataLabels && label >= 0 && label < dataLabels->size()) {
                            imageSummary.push_back((*dataLabels)[label]);
                        }
                        else {
                            imageSummary.push_back("Unknown");
                        }
                        perfImageCount++;
                        progress->images_received++;
                    }
                    if(!progress->repeat_images && progress->completed_load &&
                        progress->images_loaded == progress->images_received)
                    {
                        abort();
                    }
                }
            }
            else if(cmd.command == INFCOM_CMD_TOPK_INFERENCE_RESULT) {
                connection->sendCmd(cmd);
                int count = cmd.data[0];
                int top_k = cmd.data[1];
                int item_size = top_k + 1;
                if(top_k > 0 && count > 0 && count < (int)((sizeof(cmd)-16)/(sizeof(int)*item_size))) {
                    std::lock_guard<std::mutex> guard(mutex);
                    QVector<int> labelVec;
                    QVector<float> probVec;
                    for(int i = 0; i < count; i++) {
                        int tag = cmd.data[2 + item_size*i + 0];
                        imageIndex.push_back(tag);
                        for (int j=1; j<=top_k; j++){
                            int label = cmd.data[2 + i*item_size + j];      // label has both label and prob
                            float prob =  (label>>16)*(1.0f/(float)32768.0f);
                            prob = std::min(prob, 1.0f);
                            labelVec.push_back(label & 0xFFFF);
                            probVec.push_back(prob);
                        }
                        imageTopkLabels.push_back(labelVec);
                        imageTopkConfidence.push_back(probVec);
                        // get the top label
                        int topLabel = labelVec[0];
                        imageLabel.push_back(topLabel);
                        if(dataLabels && topLabel >= 0 && topLabel < dataLabels->size()) {
                            imageSummary.push_back((*dataLabels)[topLabel]);
                        }
                        else {
                            imageSummary.push_back("Unknown");
                        }
                        perfImageCount++;
                        progress->images_received++;
                        labelVec.clear();
                        probVec.clear();
                    }
                    if(!progress->repeat_images && progress->completed_load &&
                        progress->images_loaded == progress->images_received)
                    {
                        abort();
                    }
                }
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
        progress->message += "[stopped]";
    connection->close();
    delete connection;
    progress->completed = true;

    if(progress->errorCode) {
        qDebug("inference_receiver::run() terminated: errorCode=%d", progress->errorCode);
    }
}

float inference_receiver::getPerfImagesPerSecond()
{
    std::lock_guard<std::mutex> guard(mutex);
    qint64 msec = perfTimer.elapsed();
    if(!progress->completed && msec > 2000) {
        perfRate = (float)perfImageCount * 1000.0 / (float)msec;
        perfImageCount = 0;
        perfTimer.start();
    }
    return perfRate;
}

void inference_receiver::setImageCount(int imageCount_, int labelCount_, QVector<QString> * dataLabels_)
{
    imageCount = imageCount_;
    labelCount = labelCount_;
    dataLabels = dataLabels_;
}
