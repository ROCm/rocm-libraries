import os
import sys
import socket
import struct

# InfComCommand
INFCOM_MAGIC                         = 0x02388e50
INFCOM_CMD_DONE                      = 0
INFCOM_CMD_SEND_MODE                 = 1
INFCOM_CMD_CONFIG_INFO               = 101
INFCOM_CMD_MODEL_INFO                = 102
INFCOM_CMD_SEND_MODELFILE1           = 201
INFCOM_CMD_SEND_MODELFILE2           = 202
INFCOM_CMD_COMPILER_STATUS           = 203
INFCOM_CMD_INFERENCE_INITIALIZATION  = 301
INFCOM_CMD_SEND_IMAGES               = 302
INFCOM_CMD_INFERENCE_RESULT          = 303
INFCOM_CMD_TOPK_INFERENCE_RESULT     = 304
INFCOM_CMD_BB_INFERENCE_RESULT       = 305
INFCOM_CMD_SHADOW_SEND_FOLDERNAMES   = 401
INFCOM_CMD_SHADOW_CREATE_FOLDER      = 402
INFCOM_CMD_SHADOW_SEND_FILES         = 403
INFCOM_CMD_SHADOW_RESULT             = 404
INFCOM_MODE_CONFIGURE                = 1
INFCOM_MODE_COMPILER                 = 2
INFCOM_MODE_INFERENCE                = 3
INFCOM_MODE_SHADOW                   = 4
INFCOM_EOF_MARKER                    = 0x12344321
INFCOM_MAX_IMAGES_PER_PACKET         = 6
INFCOM_MAX_PACKET_SIZE               = 8192

# process command-lines
if len(sys.argv) < 2:
    print('Usage: python annInferenceApp.py [-v] [-host:<hostname>] [-port:<port>] -model:<modelName> [-upload:deploy.prototxt,weights.caffemodel,iw,ih,ic,mode,order,m0,m1,m2,a0,a1,a2[,save=modelName[,override][,passwd=...]]] [-synset:<synset.txt>] [-output:<output.csv>] [-shadow] [-detect] [-topK:top_k] [-shadowSetup:<checkFolderList>,<CreateFolderList>,<files>]<folder>|<file(s)>')
    sys.exit(1)
host = 'localhost'
port = 28282
modelName = ''
imageDirPath = ''
imageFileList = []
outputFileName = None
synsetFileName = None
uploadParams = ''
shadowParams = ''
verbose = False
sendFileName = 0
topkValue = 0
detectBB = 0
arg = 1
while arg < len(sys.argv):
    if sys.argv[arg][:6] == '-host:':
        host = sys.argv[arg][6:]
        arg = arg + 1
    elif sys.argv[arg][:6] == '-port:':
        port = int(sys.argv[arg][6:])
        arg = arg + 1
    elif sys.argv[arg][:7] == '-model:':
        modelName = sys.argv[arg][7:]
        arg = arg + 1
    elif sys.argv[arg][:8] == '-output:':
        outputFileName = sys.argv[arg][8:]
        arg = arg + 1
    elif sys.argv[arg][:8] == '-synset:':
        synsetFileName = sys.argv[arg][8:]
        arg = arg + 1
    elif sys.argv[arg][:8] == '-upload:':
        uploadParams = sys.argv[arg][8:]
        arg = arg + 1
    elif sys.argv[arg] == '-shadow':
        sendFileName = 1    
        arg = arg + 1
    elif sys.argv[arg] == '-v':
        verbose = True
        arg = arg + 1
    elif sys.argv[arg][:6] == '-topK:':
        topkValue = int(sys.argv[arg][6:])
        if topkValue > 5:
            print('ERROR: topK value more than 5 not supported')
            sys.exit(1)
        arg = arg + 1
    elif sys.argv[arg] == '-detect':
        detectBB = 1
        arg = arg + 1
    elif sys.argv[arg][:13] == '-shadowSetup:':
        shadowParams = sys.argv[arg][13:]    
        arg = arg + 1
    elif sys.argv[arg][:1] == '-':
        print('ERROR: invalid option: ' + sys.argv[arg])
        sys.exit(1)
    else:
        break
if arg == len(sys.argv)-1:
    if os.path.isdir(sys.argv[arg]):
        imageDirPath = sys.argv[arg]
        imageFileList = os.listdir(imageDirPath)
        if imageDirPath[-1] != '/':
            imageDirPath = imageDirPath + '/'
    else:
        imageFileList = [sys.argv[arg]]
elif arg < len(sys.argv)-1:
    imageFileList = sys.argv[arg:]

def recvpkt(sock):
    data = sock.recv(128)
    if len(data) != 128:
        return (0,0,(0,),'')
    msg = ''
    for c in data[64:]:
        if c == chr(0):
            break
        msg = msg + c
    info = (struct.unpack('i', data[:4])[0],struct.unpack('i', data[4:8])[0],struct.unpack('i'*14, data[8:64]),msg)
    #print('RECV' + str(info))
    return info

def sendpkt(sock,pkt):
    #print('SEND' + str(pkt))
    data =        struct.pack('i',pkt[0])
    data = data + struct.pack('i',pkt[1])
    vl = pkt[2]
    for i in range(14):
        v = 0
        if i < len(vl):
            v = vl[i]
        data = data + struct.pack('i',v)
    vl = pkt[3]
    if len(vl) < 64:
        vl = vl + chr(0) * (64-len(vl))
    data = data + vl
    sock.send(data)

def sendFile(sock,cmd,fileName):
    fp = open(fileName,'r')
    buf = fp.read()
    fp.close()
    sendpkt(sock,(INFCOM_MAGIC,cmd,(len(buf),),''))
    sock.send(buf + struct.pack('i',INFCOM_EOF_MARKER))

def sendImageFile(sock,tag,fileName):
    fp = open(fileName,'r')
    buf = fp.read()
    fp.close()
    sock.send(struct.pack('ii',tag,len(buf)) + buf + struct.pack('i',INFCOM_EOF_MARKER))

def sendImageFileName(sock,tag,fileName):
    buf = bytearray(fileName)
    sock.send(struct.pack('ii',tag,len(buf)) + buf + struct.pack('i',INFCOM_EOF_MARKER))

def sendFileNameAndFile(sock,tag,imageDirPath,fileName):
    fp = open(imageDirPath+fileName,'r')
    buf = bytearray(fileName) + fp.read()
    fp.close()
    hdr = (len(fileName)<<16)| (len(buf)-len(fileName))
    sock.send(struct.pack('ii',tag,hdr) + buf + struct.pack('i',INFCOM_EOF_MARKER))

def getConfig(host,port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except Exception:
        print('ERROR: unable to connect to %s:%d' % (host,port))
        sys.exit(1)
    maxGPUs = 0
    numModels = 0
    modelList = []
    while True:
        info = recvpkt(sock)
        if info[0] != INFCOM_MAGIC:
            print('RECV',info)
            print('ERROR: missing INFCOM_MAGIC')
            break
        if info[1] == INFCOM_CMD_DONE:
            sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_DONE,(0,),''))
            break
        elif info[1] == INFCOM_CMD_SEND_MODE:
            sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SEND_MODE,(INFCOM_MODE_CONFIGURE,),''))
        elif info[1] == INFCOM_CMD_CONFIG_INFO:
            sendpkt(sock,info)
            numModels = info[2][0]
            maxGPUs = info[2][1]
            modelCount = 0
            if numModels == 0:
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_DONE,(0,),''))
                break
        elif info[1] == INFCOM_CMD_MODEL_INFO:
            sendpkt(sock,info)
            model = [info[3], info[2][:3], info[2][3:6], info[2][6], struct.unpack('f'*6,struct.pack('i'*6,*info[2][7:13]))]
            modelList.append(model)
            modelCount = modelCount + 1
            if modelCount >= numModels:
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_DONE,(0,),''))
                break
        else:
            sendpkt(sock,info)
            print(info)
            print('ERROR: unsupported command')
            break
    sock.close()
    return [maxGPUs, modelList]

def uploadModel(host,port,uploadParams):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except Exception:
        print('ERROR: unable to connect to %s:%d' % (host,port))
        sys.exit(1)
    par = uploadParams.split(',')
    if len(par) < 13:
        print('ERROR: missing upload parameters in -upload:%s' % (uploadParams))
        sys.exit(1)
    ow = 0
    oh = 0
    oc = 0
    modelName = ''
    while True:
        info = recvpkt(sock)
        if info[0] != INFCOM_MAGIC:
            print('RECV',info)
            print('ERROR: missing INFCOM_MAGIC')
            break
        if info[1] == INFCOM_CMD_DONE:
            sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_DONE,(0,),''))
            break
        elif info[1] == INFCOM_CMD_SEND_MODE:
            compilerOptions = ''
            if len(par) >= 13:
                compilerOptions = ','.join(par[13:])
            sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SEND_MODE,(INFCOM_MODE_COMPILER,int(par[2]),int(par[3]),int(par[4]),int(par[5]),int(par[6]),struct.unpack('<I',struct.pack('<f',float(par[7])))[0],struct.unpack('<I',struct.pack('<f',float(par[8])))[0],struct.unpack('<I',struct.pack('<f',float(par[9])))[0],struct.unpack('<I',struct.pack('<f',float(par[10])))[0],struct.unpack('<I',struct.pack('<f',float(par[11])))[0],struct.unpack('<I',struct.pack('<f',float(par[12])))[0]),compilerOptions))
        elif info[1] == INFCOM_CMD_SEND_MODELFILE1:
            sendFile(sock,INFCOM_CMD_SEND_MODELFILE1,par[0])
        elif info[1] == INFCOM_CMD_SEND_MODELFILE2:
            sendFile(sock,INFCOM_CMD_SEND_MODELFILE2,par[1])
        elif info[1] == INFCOM_CMD_COMPILER_STATUS:
            sendpkt(sock,info)
            status = info[2][0]
            progress = info[2][1]
            ow = info[2][2]
            oh = info[2][3]
            oc = info[2][4]
            if info[3] != '':
                print('%3d%% [%d] %s' % (progress, status, info[3]))
            if status != 0:
                if status < 0:
                    print('ERROR: ' + str(info))
                    sys.exit(1)
                modelName = info[3]
                break
        else:
            sendpkt(sock,info)
            print(info)
            print('ERROR: unsupported command')
            sys.exit(1)
            break
    sock.close()
    model = [modelName, [int(par[2]),int(par[3]),int(par[4])], [ow,oh,oc], int(par[6]), (float(par[7]),float(par[8]),float(par[9]),float(par[10]),float(par[11]),float(par[12]))]
    return model

def runInference(host,port,GPUs,model,imageDirPath,imageFileList,synsetFileName,outputFileName,sendFileName,topkValue,detectBB,verbose):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except Exception:
        print('ERROR: unable to connect to %s:%d' % (host,port))
        sys.exit(1)
    sendCount = 0
    resultCount = 0
    resultList = [-1] * len(imageFileList)
    boundingBoxlList = [[] for i in  range(len(imageFileList))]
    if synsetFileName:
        fp = open(synsetFileName, 'r')
        synsetList = fp.readlines()
        synsetList = [x.strip() for x in synsetList]
        fp.close()
    fp = None
    if outputFileName:
        fp = open(outputFileName, 'w')
    while True:
        info = recvpkt(sock)
        if info[0] != INFCOM_MAGIC:
            print('RECV',info)
            print('ERROR: missing INFCOM_MAGIC')
            break
        if info[1] == INFCOM_CMD_DONE:
            sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_DONE,(0,),''))
            break
        elif info[1] == INFCOM_CMD_SEND_MODE:
            sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SEND_MODE,(INFCOM_MODE_INFERENCE,GPUs,model[1][0],model[1][1],model[1][2],model[2][0],model[2][1],model[2][2],sendFileName,topkValue,detectBB),model[0]))
        elif info[1] == INFCOM_CMD_INFERENCE_INITIALIZATION:
            sendpkt(sock,info)
            print('OK: ' + info[3])
        elif info[1] == INFCOM_CMD_SEND_IMAGES:
            count = min(info[2], len(imageFileList)-sendCount)
            if count < 1:
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SEND_IMAGES,(-1,),''))
            else:
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SEND_IMAGES,(count,),''))
                for i in range(count):
                    tag = sendCount
                    if sendFileName == 0:
                        sendImageFile(sock,tag,imageDirPath + imageFileList[tag])
                    else:
                        sendImageFileName(sock,tag,imageDirPath + imageFileList[tag])
                    sendCount = sendCount + 1
        elif info[1] == INFCOM_CMD_INFERENCE_RESULT:
            sendpkt(sock,info)
            count = info[2][0]
            status = info[2][1]
            if status != 0:
                print('ERROR: INFERENCE ' + str(info))
                break
            for i in range(count):
                tag = info[2][2 + i * 2 + 0]
                label = info[2][2 + i * 2 + 1]
                if tag >= 0:
                    resultList[tag] = label
                    resultCount = resultCount + 1
                    line = '%s,%d' % (imageFileList[tag], label)
                    if synsetFileName:
                        line = line + ',' + synsetList[label]
                    if fp:
                        fp.write(line + '\n')
                    if verbose or not fp:
                        print(line)
                else:
                    print('RECEIVED INCORRECT RESULT: ', tag, label)
            if resultCount >= len(imageFileList):
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_DONE,(0,),''))
                break
        elif info[1] == INFCOM_CMD_TOPK_INFERENCE_RESULT:
            sendpkt(sock,info)
            count = info[2][0]
            top_k = info[2][1]
            itemSize = top_k+1
            #print('INFO:' + str(info))
            if top_k <= 0:
                print('ERROR: INFERENCE ' + str(info))
                break
            for i in range(count):
                tag = info[2][2 + i * itemSize + 0]
                if tag >= 0:
                    line = '%s' % (imageFileList[tag])
                    resultCount = resultCount + 1
                    for j in range(top_k):
                        label = info[2][3 + i * itemSize + j]
                        topLabel = label&0xFFFF
                        line = line + ',%d' %(topLabel)
                        if synsetFileName:
                            line = line + ',%s(%f)' %(synsetList[topLabel], (label>>16)*(1.0/32767.0))
                        if j==0:
                            resultList[tag] = topLabel
                    if fp:
                        fp.write(line + '\n')
                    if verbose or not fp:
                        print(line)
                else:
                    print('RECEIVED INCORRECT RESULT: ', tag, topLabel)
            if resultCount >= len(imageFileList):
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_DONE,(0,),''))
                break
        elif info[1] == INFCOM_CMD_BB_INFERENCE_RESULT:
            sendpkt(sock,info)
            tag   = info[2][0]
            count = (info[2][1] & 0xFFFF)
            totalcount = (info[2][1]>>16)
            #print('INFO:' + str(info))
            if count < 0 or tag < 0 or totalcount < 0:
                print('ERROR: INFCOM_CMD_BB_INFERENCE_RESULT ' + str(info))
                break
            for i in range(count):
                box = [info[2][4*i+2], info[2][4*i+3], info[2][4*i+4], info[2][4*i+5]]
                boundingBoxlList[tag].append(box)
            if (len(boundingBoxlList[tag]) >= totalcount):
                resultCount = resultCount + 1
                line = '%s' % (imageFileList[tag])
                for i in range(totalcount):
                    box = boundingBoxlList[tag][i]
                    line = line + ' Box: %d (%f %f %f %f ),%s(%f)' %(i, (box[0]&0xFFFF)*(1.0/32767.0), (box[0]>>16)*(1.0/32767.0), (box[1]&0xFFFF)*(1.0/32767.0), (box[1]>>16)*(1.0/32767.0), synsetList[box[3]], box[2]*(1.0/0x3FFFFFFF))
                if fp:
                    fp.write(line + '\n')
                if verbose or not fp:
                    print(line)
            if resultCount >= len(imageFileList):
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_DONE,(0,),''))
                break
        else:
            sendpkt(sock,info)
            print(info)
            print('ERROR: unsupported command')
            sys.exit(1)
            break
    sock.close()
    if fp:
        fp.close()
    return resultList

def RunShadow(host,port,imageDirPath,shadowParams):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except Exception:
        print('ERROR: unable to connect to %s:%d' % (host,port))
        sys.exit(1)
    checkFolderCount = 0
    createFolderCount = 0
    sendFilesCount = 0     
    checkFolder,CreateFolder,Sendfiles = shadowParams.split(',')
    if checkFolder != '':
        fp = open(checkFolder, 'r')
        checkFolderList = fp.readlines()
        fp.close()
    if CreateFolder != '':
        fp = open(CreateFolder, 'r')
        CreateFolderList = fp.readlines()
        fp.close()
    if Sendfiles != '':
        fp = open(Sendfiles, 'r')
        SendfilesList = fp.readlines()
        fp.close()

    while True:
        info = recvpkt(sock)
        if info[0] != INFCOM_MAGIC:
            print('RECV',info)
            print('ERROR: missing INFCOM_MAGIC')
            break
        if info[1] == INFCOM_CMD_DONE:
            sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_DONE,(0,),''))
            break
        elif info[1] == INFCOM_CMD_SEND_MODE:
            sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SEND_MODE,(INFCOM_MODE_SHADOW,),''))            
        elif info[1] == INFCOM_CMD_SHADOW_RESULT:
            sendpkt(sock,info)
            count = info[2][0]
            mode = info[2][1]
            if not INFCOM_CMD_SHADOW_SEND_FOLDERNAMES <= mode <= INFCOM_CMD_SHADOW_SEND_FILES:
                print('ERROR: Shadow:RESULT invalid mode')
                break
            for i in range(count):
                tag = info[2][2 + i * 2 + 0]
                status = info[2][2 + i * 2 + 1]
                if tag >= 0:
                    if status < 0:
                        print('Shadow Check failed for mode %d and tag %d' % (mode, tag))
                else:
                    print('ERROR:SHADOW RECEIVED INCORRECT RESULT tag:%d index: %d' % (tag, i))

        elif info[1] == INFCOM_CMD_SHADOW_SEND_FOLDERNAMES:
            count = min(info[2][0], len(checkFolderList)-checkFolderCount)
            first = True
            if count < 1 and first:
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SHADOW_SEND_FOLDERNAMES,(-1,),''))
                first = False
            else:
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SHADOW_SEND_FOLDERNAMES,(count,),''))
                for i in range(count):
                    tag = checkFolderCount
                    sendImageFileName(sock,tag,checkFolderList[tag][:-1])
                    checkFolderCount = checkFolderCount + 1
        elif info[1] == INFCOM_CMD_SHADOW_CREATE_FOLDER:
            count = min(info[2][0], len(CreateFolderList)-createFolderCount)
            if count < 1:
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SHADOW_CREATE_FOLDER,(-1,),''))
            else:
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SHADOW_CREATE_FOLDER,(count,),''))
                for i in range(count):
                    tag = createFolderCount
                    sendImageFileName(sock,tag,CreateFolderList[tag][:-1])
                    createFolderCount = createFolderCount + 1
        elif info[1] == INFCOM_CMD_SHADOW_SEND_FILES:
            count = min(info[2][0], len(SendfilesList)-sendFilesCount)
            if count < 1:
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SHADOW_SEND_FILES,(-1,),''))
            else:
                sendpkt(sock,(INFCOM_MAGIC,INFCOM_CMD_SHADOW_SEND_FILES,(count,),''))
                for i in range(count):
                    tag = sendFilesCount
                    sendFileNameAndFile(sock,tag,imageDirPath,SendfilesList[tag][:-1])
                    sendFilesCount = sendFilesCount + 1

# get configuration from server
config = getConfig(host,port)
GPUs = config[0]
# run shadow
if shadowParams != '':
    RunShadow(host,port,imageDirPath,shadowParams)
    print('OK: shadow configured')

if modelName == '':
    print('OK: server has ' + str(GPUs) + ' GPUs')
    for v in config[1]:
        print('OK: server has model ' + str(v))
    if uploadParams == '':
        sys.exit(0)

# upload and pick the model
model = []
if uploadParams != '':
    model = uploadModel(host,port,uploadParams)
    modelName = model[0]
else:
    for v in config[1]:
        if v[0] == modelName:
            model = v
            break
if len(model) == 0:
    print(config)
    print('ERROR: unable to find model: ' + modelName)
    sys.exit(1)
print('OK: found model ' + str(model))

# run inference
if len(imageFileList) > 0:
    if modelName == '':
        print('ERROR: no model available to run inference')
        sys.exit(1)
    runInference(host,port,GPUs,model,imageDirPath,imageFileList,synsetFileName,outputFileName,sendFileName,topkValue,detectBB,verbose)
    if outputFileName:
        print('OK: saved inference results in ' + outputFileName)
