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

#ifndef TCPCONNECTION_H
#define TCPCONNECTION_H

#include "infcom.h"
#include <QTcpSocket>

class TcpConnection : public QTcpSocket
{
public:
    TcpConnection(QString hostname, int port, int timeout, QObject *parent = Q_NULLPTR);
    bool connected();

    bool recvCmd(InfComCommand& cmd);
    bool sendCmd(const InfComCommand& cmd);
    bool sendFile(int command, const QString fileName, volatile int& progress, QString& mesg, volatile bool& abortRequested);
    bool sendImage(int tag, QByteArray& byteArray, int& errorCode, QString& message, volatile bool& abortRequested);

protected:
    int recv(void * pkt, size_t len);
    int send(const void * pkt, size_t len);

private:
    int error;
    int timeout;
};

#endif // TCPCONNECTION_H
