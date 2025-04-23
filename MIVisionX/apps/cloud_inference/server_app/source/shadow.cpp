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

#include <sys/stat.h>
#include "shadow.h"
#include "netutil.h"
#include "common.h"
#include <errno.h>
#include <iostream>
#include <algorithm>

int createPath( mode_t mode, const std::string& rootPath, std::string path )
{
    struct stat st;
    for( std::string::iterator iter = path.begin() ; iter != path.end(); )
    {
         std::string::iterator newIter = std::find( iter, path.end(), '/' );
         std::string newPath = rootPath + "/" + std::string( path.begin(), newIter);

         if( stat( newPath.c_str(), &st) != 0)
         {
             if( mkdir( newPath.c_str(), mode) != 0 && errno != EEXIST )
             {
                std::cout << "cannot create folder [" << newPath << "] : " << strerror(errno) << std::endl;
                return -1;
             }
         }
         else
            if( !S_ISDIR(st.st_mode) )
             {
                 errno = ENOTDIR;
                 std:: cout << "path [" << newPath << "] not a dir " << std::endl;
                 return -1;
             }
             else
                 std::cout << "path [" << newPath << "] already exists " << std::endl;


         iter = newIter;
         if( newIter != path.end() )
             ++ iter;
    }
    return 0;
}

int runShadow(int sock, Arguments * args, std::string& clientName, InfComCommand * cmd)
{
    std::string shadowFolder = args->getlocalShadowRootDir();
    if (shadowFolder.empty())
    {
        return error_close(sock, "runShadow: Server is not running in shadow mode");
        return -1;
    }
    for(bool endOfSequence = false; !endOfSequence; ) {
        InfComCommand cmd = {
            INFCOM_MAGIC, INFCOM_CMD_SHADOW_SEND_FOLDERNAMES, { 6 }, { 0 }      // limit to 6 because max 6 result can fit into data
        };
        ERRCHK(sendCommand(sock, cmd, clientName));
        ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_SHADOW_SEND_FOLDERNAMES));
        int folderCountReceived = cmd.data[0];
        if (folderCountReceived <= 0) endOfSequence = true;
        if (folderCountReceived > 0) {
            int resultCount = folderCountReceived;
            InfComCommand result_cmd = {
                INFCOM_MAGIC, INFCOM_CMD_SHADOW_RESULT, { resultCount, INFCOM_CMD_SHADOW_SEND_FOLDERNAMES }, { 0 }
            };
            for(int i=0; i < resultCount; i++) {
                int header[2] = { 0, 0 };
                ERRCHK(recvBuffer(sock, &header, sizeof(header), clientName));
                int folder_tag = header[0];
                int size = header[1];
                result_cmd.data[2+i*2+0] = folder_tag;
                result_cmd.data[2+i*2+1] = -1;           // not present
                if (folder_tag < 0)
                {
                    result_cmd.data[2+i*2+1] = 0;
                    result_cmd.data[2+i*2+0] = folder_tag;
                    endOfSequence = true;
                    resultCount = i;
                    break;
                }
                // do sanity check with unreasonable parameters
                if(size <= 0 || size > 1024) {
                    return error_close(sock, "invalid shadowfolder(tag:%d,size:%d) from %s", folder_tag, size, clientName.c_str());
                }
                std::string folderNameDir = args->getlocalShadowRootDir() + "/";
                char * buff = new char [size];
                ERRCHK(recvBuffer(sock, buff, size, clientName));
                folderNameDir.append(std::string(buff, size));
                struct stat sb;
                if (stat(folderNameDir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
                {
                    result_cmd.data[2+i*2+1] = 0;
                }
                int eofMarker = 0;
                ERRCHK(recvBuffer(sock, &eofMarker, sizeof(eofMarker), clientName));
                if(eofMarker != INFCOM_EOF_MARKER) {
                    return error_close(sock, "shadowfolder: eofMarker 0x%08x (incorrect)", eofMarker);
                }
            }
            if(resultCount > 0) {
                result_cmd.data[0] = resultCount;
                ERRCHK(sendCommand(sock, result_cmd, clientName));
                ERRCHK(recvCommand(sock, result_cmd, clientName, INFCOM_CMD_SHADOW_RESULT));
            }
        }
    }
    // create Folder
    for(bool endOfSequence = false; !endOfSequence; ) {
        InfComCommand cmd = {
            INFCOM_MAGIC, INFCOM_CMD_SHADOW_CREATE_FOLDER, { 6 }, { 0 }     // limit to 6 because max 6 <tag,present> fits into data
        };
        ERRCHK(sendCommand(sock, cmd, clientName));
        ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_SHADOW_CREATE_FOLDER));
        int folderCountReceived = cmd.data[0];
        if (folderCountReceived <= 0) endOfSequence = true;
        if (folderCountReceived > 0)
        {
            int resultCount = folderCountReceived;
            InfComCommand result_cmd = {
                INFCOM_MAGIC, INFCOM_CMD_SHADOW_RESULT, { resultCount, INFCOM_CMD_SHADOW_CREATE_FOLDER }, { 0 }
            };
            for(int i=0; i < resultCount; i++) {
                int header[2] = { 0, 0 };
                ERRCHK(recvBuffer(sock, &header, sizeof(header), clientName));
                int folder_tag = header[0];
                int size = header[1];
                result_cmd.data[2+i*2+0] = folder_tag;
                result_cmd.data[2+i*2+1] = -1;           // not present
                if (folder_tag < 0)
                {
                    result_cmd.data[2+i*2+1] = 0;
                    result_cmd.data[2+i*2+0] = folder_tag;
                    endOfSequence = true;
                    resultCount = i;
                    break;
                }
                // do sanity check with unreasonable parameters
                if(size <= 0 || size > 1024) {
                    return error_close(sock, "invalid shadowfolder(tag:%d,size:%d) from %s", folder_tag, size, clientName.c_str());
                }
                char * buff = new char [size];
                ERRCHK(recvBuffer(sock, buff, size, clientName));
                result_cmd.data[2+i*2+0] = folder_tag;
                int status = createPath(0777, args->getlocalShadowRootDir(), std::string(buff, size));  // equivalent of 0700
                result_cmd.data[2+i*2+1] = status;
                int eofMarker = 0;
                ERRCHK(recvBuffer(sock, &eofMarker, sizeof(eofMarker), clientName));
                if(eofMarker != INFCOM_EOF_MARKER) {
                    return error_close(sock, "shadowfolder: eofMarker 0x%08x (incorrect)", eofMarker);
                }
            }
            if(resultCount > 0) {
                result_cmd.data[0] = resultCount;
                ERRCHK(sendCommand(sock, result_cmd, clientName));
                ERRCHK(recvCommand(sock, result_cmd, clientName, INFCOM_CMD_SHADOW_RESULT));
            }
        }
    }
    // sendFiles protocol
    for(bool endOfSequence = false; !endOfSequence; ) {
        InfComCommand cmd = {
            INFCOM_MAGIC, INFCOM_CMD_SHADOW_SEND_FILES, { 6 }, { 0 }        // limit to 6 because max 6 <tag,present> fits into data
        };
        ERRCHK(sendCommand(sock, cmd, clientName));
        ERRCHK(recvCommand(sock, cmd, clientName, INFCOM_CMD_SHADOW_SEND_FILES));
        int fileCountReceived = cmd.data[0];
        if (fileCountReceived <= 0) endOfSequence = true;
        int resultCount = fileCountReceived;         // max 6 <tag,present> fits into data
        InfComCommand result_cmd = {
            INFCOM_MAGIC, INFCOM_CMD_SHADOW_RESULT, { resultCount, INFCOM_CMD_SHADOW_SEND_FILES }, { 0 }
        };
        for(int i=0; i < resultCount; i++) {
            int header[2] = { 0 };
            ERRCHK(recvBuffer(sock, &header, sizeof(header), clientName));
            int tag = header[0];
            int size_filename = (header[1]&0xFFFF0000)>>16;
            int size_data     = header[1]&0xFFFF;
            int size          = size_filename + size_data;
            result_cmd.data[2+i*2+0] = tag;
            result_cmd.data[2+i*2+1] = -1;           // not present
            // do sanity check with unreasonable parameters
            if(tag < 0 || size <= 0 || size > 50000000) {
                return error_close(sock, "invalid filename(tag:%d,size:%d) from %s", tag, size, clientName.c_str());
            }
            // get filename from folder_tag and file_tag
            std::string fileName = args->getlocalShadowRootDir() + "/";
            char * buff = new char [size];
            ERRCHK(recvBuffer(sock, buff, size, clientName));
            fileName.append(std::string(buff, size_filename));
            struct stat sb;
            result_cmd.data[2+i*2+1] = 0;       // new file created
            if (stat(fileName.c_str(), &sb) == 0 && S_ISREG(sb.st_mode))
            {
                result_cmd.data[2+i*2+1] = 1;       // file already present
            }else
            {
                // open file and write data
                FILE *fp = fopen(fileName.c_str(), "wb");
                if(!fp) {
                    return error_close(sock, "shadowmode: couldn't open file %s errno:%d", fileName.c_str(), errno);
                }
                fwrite(&buff[size_filename], 1, size_data, fp);
                fclose(fp);
            }

            int eofMarker = 0;
            ERRCHK(recvBuffer(sock, &eofMarker, sizeof(eofMarker), clientName));
            if(eofMarker != INFCOM_EOF_MARKER) {
                return error_close(sock, "shadowfolder: eofMarker 0x%08x (incorrect)", eofMarker);
            }
        }
        if(resultCount > 0) {

            result_cmd.data[0] = resultCount;
            ERRCHK(sendCommand(sock, result_cmd, clientName));
            ERRCHK(recvCommand(sock, result_cmd, clientName, INFCOM_CMD_SHADOW_RESULT));
        }
    }
    // send and wait for INFCOM_CMD_DONE message
    InfComCommand reply = {
        INFCOM_MAGIC, INFCOM_CMD_DONE, { 0 }, { 0 }
    };
    ERRCHK(sendCommand(sock, reply, clientName));
    ERRCHK(recvCommand(sock, reply, clientName, INFCOM_CMD_DONE));
    return 0;
}

