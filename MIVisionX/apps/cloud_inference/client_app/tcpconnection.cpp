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

#include "tcpconnection.h"
#include <QThread>
#include <QTimer>
#include <QFile>

#define DEBUG_CMD_TRACE  0    // debug trace of messages

TcpConnection::TcpConnection(QString hostname, int port, int timeout_, QObject *parent)
    : QTcpSocket(parent), error{ 0 }, timeout{ timeout_ }
{
    connectToHost(hostname, port);
    if(!waitForConnected(timeout)) {
        error = -1;
    }
}

bool TcpConnection::connected()
{
    return !error ? true : false;
}

int TcpConnection::recv(void * pkt, size_t len)
{
    while(!error && (state() == QAbstractSocket::ConnectedState)) {
        if(waitForReadyRead() && (bytesAvailable() >= (qint64)len) && (read((char *)pkt, (qint64)len) == (qint64)len)) {
            return 0;
        }
        QThread::msleep(1000);
    }
    return -1;
}

int TcpConnection::send(const void * pkt, size_t len)
{
    if(!error && (write((const char *)pkt, (qint64)len) == (qint64)len) && (waitForBytesWritten(timeout))) {
        return 0;
    }
    else {
        return -1;
    }
}

bool TcpConnection::recvCmd(InfComCommand& cmd)
{
    int err = recv((char *)&cmd, sizeof(cmd));
    if(!err) {
#if DEBUG_CMD_TRACE
        qDebug("recvCmd: 0x%08x %3d { %d %d %d %d - %d %d %d %d } %s",
               cmd.magic, cmd.command,
               cmd.data[0], cmd.data[1], cmd.data[2], cmd.data[3], cmd.data[4], cmd.data[5], cmd.data[6], cmd.data[7],
               cmd.message);
#endif
        return true;
    }
    else {
        return false;
    }
}

bool TcpConnection::sendCmd(const InfComCommand& cmd)
{
#if DEBUG_CMD_TRACE
    qDebug("sendCmd: 0x%08x %3d { %d %d %d %d - %d %d %d %d } %s",
           cmd.magic, cmd.command,
           cmd.data[0], cmd.data[1], cmd.data[2], cmd.data[3], cmd.data[4], cmd.data[5], cmd.data[6], cmd.data[7],
           cmd.message);
#endif
    int err = send((char *)&cmd, sizeof(cmd));
    if(!err) {
        return true;
    }
    else {
        return false;
    }
}

bool TcpConnection::sendFile(int command, const QString fileName, volatile int& progress, QString& mesg, volatile bool& abortRequested)
{
    progress = -1;
    mesg.sprintf("Uploading %s ...", fileName.toStdString().c_str());

    QFile fileObj(fileName);
    if(!fileObj.open(QIODevice::ReadOnly)) {
        mesg.sprintf("ERROR: unable to open: %s", fileName.toStdString().c_str());
        return false;
    }
    QByteArray byteArray = fileObj.readAll();
    InfComCommand reply = {
        INFCOM_MAGIC, command,
        { byteArray.size(), 0 },
        { 0 }, { 0 }
    };
    QStringList text = fileName.split("/");
    strncpy(reply.message, text[text.size()-1].toStdString().c_str(), sizeof(reply.message));
    sendCmd(reply);

    progress = 0;
    const char * buf = byteArray.constData();
    int len = byteArray.size();
    int pos = 0;
    while(!abortRequested && (pos < len)) {
        int pktSize = std::min(INFCOM_MAX_PACKET_SIZE, len-pos);
        if(send(&buf[pos], pktSize) < 0) {
            progress = -1;
            mesg.sprintf("ERROR: sendFile: write(data:%d) failed after %d/%d bytes - %s", pktSize, pos, len, fileName.toStdString().c_str());
            return false;
        }
        pos += pktSize;
        progress = (int)((float)pos * 100.0 / len + 0.5);
    }
    int eofMarked = INFCOM_EOF_MARKER;
    if(!abortRequested && send(&eofMarked, sizeof(eofMarked)) < 0) {
        progress = -1;
        mesg.sprintf("ERROR: sendFile: write(eofMarked:%ld) - %s", sizeof(eofMarked), fileName.toStdString().c_str());
        return false;
    }
    if(abortRequested) {
        progress = -1;
        mesg.sprintf("ERROR: sendFile: aborted - %s", fileName.toStdString().c_str());
    }
    else {
        progress = 100;
    }
    return true;
}

bool TcpConnection::sendImage(int tag, QByteArray& byteArray, int& errorCode, QString& message, volatile bool& abortRequested)
{
    const char * buf = byteArray.constData();
    int len = byteArray.size();
    int header[2] = { tag, len };
    if(send((const char *)&header[0], sizeof(header)) < 0) {
        errorCode = -1;
        message.sprintf("ERROR: sendImage: write(header:%ld) - tag:%d", sizeof(header), tag);
        return false;
    }
    int pos = 0;
    while(!abortRequested && (pos < len)) {
        int pktSize = std::min(INFCOM_MAX_PACKET_SIZE, len-pos);
        if(send(&buf[pos], pktSize) < 0) {
            errorCode = -1;
            message.sprintf("ERROR: sendImage: write(pkt:%d) failed after %d/%d bytes - tag:%d", pktSize, pos, len, tag);
            return false;
        }
        pos += pktSize;
    }
    int eofMarked = INFCOM_EOF_MARKER;
    if(!abortRequested && send((const char *)&eofMarked, sizeof(eofMarked)) < 0) {
        errorCode = -1;
        message.sprintf("ERROR: sendImage: write(eofMarked:%ld) - tag:%d", sizeof(eofMarked), tag);
        return false;
    }
    if(abortRequested) {
        errorCode = -1;
        message.sprintf("ERROR: SendImage: aborted");
    }
    return true;
}
