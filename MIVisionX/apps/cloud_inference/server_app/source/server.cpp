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

#include "server.h"
#include "configure.h"
#include "compiler.h"
#include "inference.h"
#include "netutil.h"
#include "shadow.h"
#include <thread>

int connection(int sock, Arguments * args, std::string clientName)
{
    info("== CONNECTED to %s ================", clientName.c_str());

    // ask connection mode by sending InfComCommand:INFCOM_CMD_SEND_MODE
    InfComCommand cmd = {
        INFCOM_MAGIC, INFCOM_CMD_SEND_MODE, { 0 }, { 0 }
    };
    ERRCHK(sendCommand(sock, cmd, clientName));
    ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_SEND_MODE));
    int mode = cmd.data[0];
    if(mode != INFCOM_MODE_CONFIGURE && mode != INFCOM_MODE_COMPILER && mode != INFCOM_MODE_INFERENCE && mode != INFCOM_MODE_SHADOW) {
        dumpCommand("reply", cmd);
        close(sock);
        return error("received incorrect response to INFCOM_CMD_SEND_MODE from %s", clientName.c_str());
    }
    // run proper module
    int status = 0;
    if(mode == INFCOM_MODE_CONFIGURE) {
        status = runConfigure(sock, args, clientName, &cmd);
    }
    else if(mode == INFCOM_MODE_COMPILER) {
        status = runCompiler(sock, args, clientName, &cmd);
    }
    else if(mode == INFCOM_MODE_INFERENCE) {
#if ENABLE_OPENCL      
        InferenceEngine * ie = new InferenceEngine(sock, args, clientName, &cmd);
#else
        InferenceEngineHip * ie;
        int decodeMode = cmd.data[11];
        std::string dataFolder(cmd.path);
        if(decodeMode == 0) 
            ie = new InferenceEngineHip(sock, args, clientName, &cmd);
        else if(decodeMode == 1)    
            ie = new InferenceEngineRocalHip(sock, args, clientName, &cmd, dataFolder);
#endif        
        if(ie) {
            status = ie->run();
            delete ie;
        }
    }
    else if(mode == INFCOM_MODE_SHADOW) {
        status = runShadow(sock, args, clientName, &cmd);
    }
    if(status == 0) {
        close(sock);
        info("== disconnected %s ================", clientName.c_str());
    }

    return status;
}

int server(Arguments * args)
{
    // setup socket address structure and create socket
    struct sockaddr_in server_addr;
    memset(&server_addr,0,sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(args->getPort());
    server_addr.sin_addr.s_addr = INADDR_ANY;
    int sockServer = socket(PF_INET,SOCK_STREAM, 0);
    if (!sockServer) {
        return error("socket() failed");
    }

    // set socket to immediately reuse port when the application closes
    int reuse = 1;
    if (setsockopt(sockServer, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        return error("setsockopt() failed");
    }

    // call bind to associate the socket with our local address and port
    if (bind(sockServer, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        return error_close(sockServer, "bind(port:%d) failed", args->getPort());
    }

    // convert the socket to listen for incoming connections
    if (listen(sockServer, SOMAXCONN) < 0) {
        return error_close(sockServer, "listen() failed");
    }
    info("listening on port %d for annInferenceApp connections ...", args->getPort());

    // accept clients
    struct sockaddr_in client_addr;
    socklen_t clientlen = sizeof(client_addr);
    int sockClient = -1;
    while ((sockClient = accept(sockServer, (struct sockaddr *)&client_addr, &clientlen)) > 0) {
        // client info
        char clientName[256] = "Unknown";
        inet_ntop(AF_INET, &client_addr.sin_addr, clientName, sizeof(clientName));

        // run client connection in a separate thread
        std::thread work(connection, sockClient, args, clientName);
        work.detach();
    }

    // close server
    close(sockServer);

    return 0;
}
