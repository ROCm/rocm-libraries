/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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


#include "ago_internal.h"
#include <mutex>

#if _WIN32
static DWORD WINAPI agoGraphThreadFunction(LPVOID graph_)
#else
static void agoGraphThreadFunction(LPVOID graph_)
#endif
{
    AgoGraph * graph = (AgoGraph *)graph_;
    while (WaitForSingleObject(graph->hSemToThread, INFINITE) == WAIT_OBJECT_0) {
        graph->threadThreadWaitState = 2;
        if (graph->threadThreadTerminationState)
            break;

        // execute graph
        graph->status = agoProcessGraph(graph);

        // inform caller
        graph->threadExecuteCount++;
        ReleaseSemaphore(graph->hSemFromThread, 1, nullptr);
    }
    // inform caller about termination
    graph->threadThreadTerminationState = 2;
    ReleaseSemaphore(graph->hSemFromThread, 1, nullptr);
#if _WIN32
    return 0;
#endif
}

AgoContext * agoCreateContextFromPlatform(struct _vx_platform * platform)
{
    CAgoLockGlobalContext lock;

    // check if CPU hardware supports
    bool isHardwareSupported = agoIsCpuHardwareSupported();
    if (!isHardwareSupported) {
        agoAddLogEntry(NULL, VX_FAILURE, "ERROR: Unsupported CPU (requires SSE 4.2)\n");
        return NULL;
    }

    // create context and initialize
    AgoContext * acontext = new AgoContext;
    if (acontext) {
        acontext->ref.platform = platform;
        agoResetReference(&acontext->ref, VX_TYPE_CONTEXT, acontext, NULL);
        acontext->ref.external_count++;
        // initialize image formats
        if (agoInitializeImageComponentsAndPlanes(acontext)) {
            delete acontext;
            return NULL;
        }
        // initialize kernels
        if (agoPublishKernels(acontext)) {
            delete acontext;
            return NULL;
        }
        // initialize thread config
        char textBuffer[1024];
        if (agoGetEnvironmentVariable("AGO_THREAD_CONFIG", textBuffer, sizeof(textBuffer))) {
            acontext->thread_config = atoi(textBuffer);
        }
    }
    return (AgoContext *)acontext;
}

int agoReleaseContext(AgoContext * acontext)
{
    CAgoLockGlobalContext lock;

    if (!agoIsValidContext(acontext))
        return -1;

    vx_reference ref = (vx_reference)acontext;
    ref->external_count = ref->external_count- 1;
    if(ref->external_count >= 0)
        acontext->num_active_references--;

    if(ref->external_count == 0)
    {
        EnterCriticalSection(&acontext->cs);
        // release all the resources
        LeaveCriticalSection(&acontext->cs);
        delete acontext;
    }
    return 0;
}

AgoGraph * agoCreateGraph(AgoContext * acontext)
{
    AgoGraph * agraph = new AgoGraph;
    if (!agraph || !acontext) {
        return nullptr;
    }

    // initialize
    agoResetReference(&agraph->ref, VX_TYPE_GRAPH, acontext, NULL);
    agraph->attr_affinity = acontext->attr_affinity;
    char textBuffer[256];
    if (agoGetEnvironmentVariable("VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS", textBuffer, sizeof(textBuffer))) {
        if (sscanf(textBuffer, "%i", &agraph->optimizer_flags) == 1) {
            agoAddLogEntry(&agraph->ref, VX_SUCCESS, "DEBUG: VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS = 0x%08x\n", agraph->optimizer_flags);
        }
    }

    { // link graph to the context
        CAgoLock lock(acontext->cs);
        agoAddGraph(&acontext->graphList, agraph);
        agraph->ref.external_count++;
        acontext->num_active_references++;
    }
    if (acontext->thread_config & 1) {
        // create semaphore and thread for graph scheduling: limit 1000 pending requests
        agraph->hSemToThread = CreateSemaphore(nullptr, 0, 1000, nullptr);
        agraph->hSemFromThread = CreateSemaphore(nullptr, 0, 1000, nullptr);
        if (agraph->hSemToThread == NULL || agraph->hSemFromThread == NULL) {
            agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: CreateSemaphore() failed\n");
            agoReleaseGraph(agraph);
            return nullptr;
        }
        agraph->hThread = CreateThread(NULL, 0, agoGraphThreadFunction, agraph, 0, NULL);
#if _WIN32 // TBD: need to enable this check for non-windows platforms
        if (agraph->hThread == NULL) {
            agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: CreateThread() failed\n");
            agoReleaseGraph(agraph);
            return nullptr;
        }
#if _DEBUG
        agoAddLogEntry(&agraph->ref, VX_SUCCESS, "OK: enabled graph scheduling in separate threads\n");
#endif
#endif
    }
    agraph->reverify = agraph->verified;
    agraph->verified = vx_false_e;
    agraph->state = VX_GRAPH_STATE_UNVERIFIED;
    return (AgoGraph *)agraph;
}

int agoReleaseGraph(AgoGraph * agraph)
{
    CAgoLock lock(agraph->ref.context->cs);

    int status = 0;
    agraph->ref.external_count--;
    if(agraph->ref.external_count >= 0)
        agraph->ref.context->num_active_references--;
    if (agraph->ref.external_count == 0) {
        EnterCriticalSection(&agraph->cs);
        // stop graph thread
        if (agraph->hThread) {
            agraph->threadThreadTerminationState = 1;
            ReleaseSemaphore(agraph->hSemToThread, 1, nullptr);
            while (agraph->threadThreadTerminationState == 1) {
                // give a chance for the thread to run in case it is waititng
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                ReleaseSemaphore(agraph->hSemToThread, 1, nullptr);
            }
            if (agraph->hSemToThread) {
                CloseHandle(agraph->hSemToThread);
            }
            if (agraph->hSemFromThread) {
                CloseHandle(agraph->hSemFromThread);
            }
            CloseHandle(agraph->hThread);
        }
        // deinitialize the graph
        for (AgoNode * node = agraph->nodeList.head; node; node = node->next)
        {
            status = agoShutdownNode(node);
            if (status) {
                break;
            }
        }
        if (!status) {
            // remove graph from context
            if (agoRemoveGraph(&agraph->ref.context->graphList, agraph) != agraph) {
                status = -1;
                LeaveCriticalSection(&agraph->cs);
            }
            else {
        #if ENABLE_OPENCL
                // Releasing the command queue for the graph because it is not needed
                agoGpuOclReleaseGraph(agraph);
        #elif ENABLE_HIP
                agoGpuHipReleaseGraph(agraph);
        #endif
                LeaveCriticalSection(&agraph->cs);
                // move graph to garbage list
                agraph->next = agraph->ref.context->graph_garbage_list;
                agraph->ref.context->graph_garbage_list = agraph;
            }
        }
        else {
            LeaveCriticalSection(&agraph->cs);
        }
    }
    return status;
}

int agoOptimizeGraph(AgoGraph * agraph)
{
    if (!agraph->status) {
        CAgoLock lock(agraph->cs);
        CAgoLock lock2(agraph->ref.context->cs);

        // run DRAMA graph optimizer
        agraph->status = agoOptimizeDrama(agraph);
    }
    return agraph->status;
}

int agoWriteGraph(AgoGraph * agraph, AgoReference * * ref, int num_ref, FILE * fp, const char * comment)
{
    CAgoLock lock(agraph->cs);
    CAgoLock lock2(agraph->ref.context->cs);

#if ENABLE_DEBUG_MESSAGES
    agoOptimizeDramaMarkDataUsage(agraph);
#endif

    bool * imported = new bool[agraph->ref.context->num_active_modules + 1];
    for (vx_uint32 i = 0; i < agraph->ref.context->num_active_modules; i++)
        imported[i] = false;
    fprintf(fp, "# ago graph dump BEGIN [%s]\n", comment ? comment : "");
    for (auto aus = agraph->ref.context->userStructList.begin(); aus != agraph->ref.context->userStructList.end(); aus++) {
        if (aus->importing_module_index_plus1) {
            if (!imported[aus->importing_module_index_plus1 - 1]) {
                fprintf(fp, "import %s\n", agraph->ref.context->modules[aus->importing_module_index_plus1 - 1].module_name);
                imported[aus->importing_module_index_plus1 - 1] = true;
            }
        }
        else {
            if (!aus->name.length()) {
                vx_char name[64];
                snprintf(name, sizeof(name), "AUTO-USER-STRUCT!%03d!", aus->id - VX_TYPE_USER_STRUCT_START + 1);
                aus->name = name;
            }
            fprintf(fp, "type %s userstruct:" VX_FMT_SIZE "\n", aus->name.c_str(), aus->size);
        }
    }
    for (AgoKernel * akernel = agraph->ref.context->kernelList.head; akernel; akernel = akernel->next) {
        if (akernel->flags & AGO_KERNEL_FLAG_GROUP_USER) {
            if (akernel->importing_module_index_plus1) {
                if (!imported[akernel->importing_module_index_plus1 - 1]) {
                    fprintf(fp, "import %s\n", agraph->ref.context->modules[akernel->importing_module_index_plus1 - 1].module_name);
                    imported[akernel->importing_module_index_plus1 - 1] = true;
                }
            }
        }
    }
    for (AgoData * adata = agraph->ref.context->dataList.head; adata; adata = adata->next) {
        // check if data is part of specified ref[] arguments
        int index = -1;
        for (int i = 0; i < num_ref; i++) {
            if (adata == (AgoData *)ref[i]) {
                index = i;
                break;
            }
        }
        // output data statements for non ref[] and non internal generated data objects
        if (index < 0 && adata->name.length() > 0 && adata->name[0] != '!' && !adata->parent) {
            char desc[MAX_DESCRIPTION_DATA_SIZE] = "*ERROR*";
            agoGetDescriptionFromData(agraph->ref.context, desc, adata);
            fprintf(fp, "data %s = %s", adata->name.length() ? adata->name.c_str() : "*UNKNOWN*", desc);
#if ENABLE_DEBUG_MESSAGES
            if (adata->inputUsageCount | adata->outputUsageCount | adata->inoutUsageCount)
                fprintf(fp, " #usageCount[%d,%d,%d]", adata->inputUsageCount, adata->outputUsageCount, adata->inoutUsageCount);
#endif
            fprintf(fp, "\n");
        }
    }
    for (AgoData * adata = agraph->dataList.head; adata; adata = adata->next) {
        // check if data is part of specified ref[] arguments
        int index = -1;
        for (int i = 0; i < num_ref; i++) {
            if (adata == (AgoData *)ref[i]) {
                index = i;
                break;
            }
        }
        // output data statements for non ref[] and non internal generated data objects
        if (index < 0 && adata->name.length() > 0 && adata->name[0] != '!' && !adata->parent) {
            char desc[MAX_DESCRIPTION_DATA_SIZE] = "*ERROR*";
            agoGetDescriptionFromData(agraph->ref.context, desc, adata);
            fprintf(fp, "data %s = %s", adata->name.length() ? adata->name.c_str() : "*UNKNOWN*", desc);
#if ENABLE_DEBUG_MESSAGES
            if (adata->inputUsageCount | adata->outputUsageCount | adata->inoutUsageCount)
                fprintf(fp, " #usageCount[%d,%d,%d]", adata->inputUsageCount, adata->outputUsageCount, adata->inoutUsageCount);
            fprintf(fp, " #(virtual)");
#endif
            fprintf(fp, "\n");
        }
    }
    for (AgoNode * anode = agraph->nodeList.head; anode; anode = anode->next) {
        fprintf(fp, "node %s", anode->akernel->name);
        vx_uint32 paramCount = anode->paramCount;
        while (paramCount > 0 && !anode->paramList[paramCount - 1])
            paramCount--;
        for (vx_uint32 i = 0; i < paramCount; i++) {
            AgoData * data = anode->paramList[i];
            if (!data) {
                fprintf(fp, " null");
            }
            else {
                // check if data is part of specified ref[] arguments, if so use $1..$N in output
                int index = -1;
                for (int i = 0; i < num_ref; i++) {
                    if (data == (AgoData *)ref[i]) {
                        index = i;
                        break;
                    }
                }
                if (index >= 0) {
                    fprintf(fp, " $%d", index + 1);
                }
                else {
                    char name[1024];
                    agoGetDataName(name, data);
                    if (name[0]) {
                        fprintf(fp, " %s", name);
                    }
                    else {
                        char desc[MAX_DESCRIPTION_DATA_SIZE];
                        agoGetDescriptionFromData(agraph->ref.context, desc, data);
                        fprintf(fp, " %s", desc);
                    }
                }
            }
        }
        if (anode->attr_border_mode.mode == VX_BORDER_MODE_REPLICATE) fprintf(fp, " attr:BORDER_MODE:REPLICATE");
        else if (anode->attr_border_mode.mode == VX_BORDER_MODE_CONSTANT) fprintf(fp, " attr:BORDER_MODE:CONSTANT,0x%08x", anode->attr_border_mode.constant_value.U32);
        if (anode->attr_affinity.device_type) {
            fprintf(fp, " attr:AFFINITY:%s", (anode->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU) ? "GPU" : "CPU");
            if (anode->attr_affinity.device_info)
                fprintf(fp, "%d", anode->attr_affinity.device_info);
            if (anode->attr_affinity.group)
                fprintf(fp, ",%d", anode->attr_affinity.group);
        }
#if _DEBUG || ENABLE_DEBUG_MESSAGES
        fprintf(fp, " #L%d", anode->hierarchical_level);
#endif
        fprintf(fp, "\n");
    }
    fprintf(fp, "# ago graph dump END [%s]\n", comment ? comment : "");
    fflush(fp);
    delete[] imported;

    return 0;
}

static const char * agoReadLine(char * line, int size, const char * str)
{
    if (!str || !*str)
        return NULL;
    line[0] = 0; size -= 2;
    for (int i = 0; i < size; i++) {
        char c = line[i] = *str++;
        if (c == 0) {
            str--;
            break;
        }
        else if (c == '\n') {
            line[i + 1] = 0;
            break;
        }
    }
    return str;
}

static void agoUpdateLine(char * line, std::vector< std::pair< std::string, std::string > >& vars, std::string localPrefix)
{
    char lineOriginal[2048]; strcpy(lineOriginal, line);
    int ki = 0;
    for (int i = 0; lineOriginal[i]; i++, ki++) {
        line[ki] = lineOriginal[i];
        if (lineOriginal[i] == '$' && lineOriginal[i + 1] >= 'A' && lineOriginal[i + 1] <= 'Z') {
            // get variable name
            char * s = &lineOriginal[i + 1];
            int k = 1;
            for (; (s[k] >= 'A' && s[k] <= 'Z') || (s[k] >= 'a' && s[k] <= 'z') || (s[k] >= '0' && s[k] <= '9') || s[k] == '_'; k++)
                ;
            // search variable name
            for (std::vector< std::pair< std::string, std::string > >::iterator it = vars.begin(); it != vars.end(); ++it) {
                if (!strncmp(it->first.c_str(), s, k)) {
                    strcpy(&line[ki], it->second.c_str());
                    ki = (int)strlen(line) - 1;
                    i += k;
                    break;
                }
            }
        }
        else if (lineOriginal[i] == '$' && lineOriginal[i + 1] == '!') {
            strcpy(&line[ki], localPrefix.c_str());
            ki = (int)strlen(line) - 1;
            line[++ki] = '!';
            i += 1;
        }
    }
    line[ki] = 0;
}

static void agoUpdateN(char * output, char * input, int N, int Nchar)
{
    int ki = 0;
    for (int i = 0; input[i]; i++, ki++) {
        output[ki] = input[i];
        if (input[i] == '{') {
            // get variable name
            char * s = &input[i + 1];
            int k = 0;
            int index = 0, v = 0, op = '+';
            for (; (s[k] >= '0' && s[k] <= '9') || (Nchar && s[k] == Nchar) || (s[k] == '+') || (s[k] == '-'); k++) {
                if (s[k] == Nchar) v = N;
                else if (s[k] == '+' || s[k] == '-') {
                    index += (op == '+') ? v : -v;
                    op = s[k];
                    v = 0;
                }
                else v = v * 10 + s[k] - '0';
            }
            index += (op == '+') ? v : -v;
            if (s[k] == '}') {
                // replace $[expr] with index
                snprintf(&output[ki], sizeof(&output[ki]), "%d", index);
                ki = (int)strlen(output) - 1;
                i += k + 1;
            }
        }
    }
    output[ki] = 0;
}

static void agoReadGraphFromStringInternal(AgoGraph * agraph, AgoReference * * ref, int num_ref, ago_data_registry_callback_f callback_f, void * callback_obj, const char * str, vx_int32 dumpToConsole, std::vector< std::pair< std::string, std::string > >& vars, std::string localPrefix)
{
    vx_context context = agraph->ref.context;
    std::vector< std::pair< std::string, std::string > > aliases;
    // set default values to for/if constructs
    vx_int32 Nbegin = 0, Nend = 0, Nstep = 1, Nchar = '\0', forConstruct = 0;
    vx_uint32 ifdepth = 0, ifcur = 0, ifall = 0;
    // process one line at a time
    char line[2048];
    for (int lineno = 1; (str = agoReadLine(line, sizeof(line) - 16, str)) != NULL; lineno++)
    {
        int N = (int)strlen(line);
        while (N > 0 && (line[N - 1] == '\r' || line[N - 1] == '\n'))
            line[--N] = 0;
        if (dumpToConsole) agoAddLogEntry(NULL, VX_SUCCESS, "%s\n", line);
        while (N > 0 && line[N - 1] == '\\') {
            int pos = N - 1;
            if (!(str = agoReadLine(line + pos, sizeof(line) - 16 - pos, str))) break;
            N = (int)strlen(line);
            while (N > 0 && (line[N - 1] == '\r' || line[N - 1] == '\n'))
                line[--N] = 0;
            if (dumpToConsole) agoAddLogEntry(NULL, VX_SUCCESS, "%s\n", line+pos);
            lineno++;
        }
        agoUpdateLine(line, vars, localPrefix);
        char lineCopy[sizeof(line)]; strcpy(lineCopy, line);
        char * s = strstr(line, "#");
        if (s) { *s = 0; N = (int)strlen(line); }
        char * argv[64] = { 0 };
        int narg = 0;
        for (s = line; narg < 64;)
        {
            while (*s && (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n')) s++;
            if (!*s) break;
            argv[narg++] = s;
            while (*s && !(*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n')) s++;
            if (*s) *s++ = 0;
            else break;
        }
        // process for construct
        if (!forConstruct) {
            // reset for-loop parameters to single iteration
            Nbegin = 0, Nend = 0, Nstep = 1, Nchar = '\0';
            if (narg == 4 && !strcmp(argv[0], "for") && !strcmp(argv[2], "in") && strlen(argv[1]) == 1 && (argv[1][0] >= 'a' && argv[1][0] <= 'z')) {
                // set for-loop parameters
                Nchar = argv[1][0];
                char range[128]; agoUpdateN(range, argv[3], 0, '\0');
                if (sscanf(range, "%d:%d,%d", &Nbegin, &Nend, &Nstep) < 2 || Nstep <= 0) {
                    agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: invalid for syntax: should be 'for i in <begin>:<end>[,<step>]'\n>>>> %s\n", lineno, lineCopy);
                    agraph->status = -1;
                    break;
                }
                forConstruct = 1;
                continue;
            }
        }
        else if (narg == 1 && !strcmp(argv[0], "endfor")) {
            // reset for-loop parameters to single iteration
            Nbegin = 0, Nend = 0, Nstep = 1, Nchar = '\0';
            forConstruct = 0;
            continue;
        }
        if (narg == 4 && (!strcmp(argv[0], "if") || !strcmp(argv[0], "elseif"))) {
            char expr1[128]; agoUpdateN(expr1, argv[1], 0, '\0');
            char expr2[128]; agoUpdateN(expr2, argv[3], 0, '\0');
            int value1 = atoi(expr1);
            int value2 = atoi(expr2);
            bool result = false;
            if (!strcmp(argv[2], "==")) result = (value1 == value2);
            else if (!strcmp(argv[2], "!=")) result = (value1 != value2);
            else if (!strcmp(argv[2], "<=")) result = (value1 <= value2);
            else if (!strcmp(argv[2], ">=")) result = (value1 >= value2);
            else if (!strcmp(argv[2], "<")) result = (value1 < value2);
            else if (!strcmp(argv[2], ">")) result = (value1 > value2);
            else {
                agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: invalid if-command syntax: should be '[else]if <value1> ==|!=|<|>|<=|>= <value2>'\n>>>> %s\n", lineno, lineCopy);
                agraph->status = -1;
                break;
            }
            if (!strcmp(argv[0], "if")) {
                // increase the depth and mark result in lowest bit (0:true, 1:false)
                ifdepth++;
                ifcur <<= 1;
                ifall <<= 1;
                if (!result) {
                    ifcur += 1;
                    ifall += 1;
                }
            }
            else {
                // if previously if/elseif resulted in true, mark result as false
                if (!(ifall & 1))
                    result = false;
                // set lowest bit of both ifcur and ifall
                if (result) {
                    ifcur &= ~1;
                    ifall &= ~1;
                }
                else {
                    ifcur |= 1;
                }
            }
            continue;
        }
        else if (narg == 1 && !strcmp(argv[0], "else")) {
            if (ifdepth == 0) {
                agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: found else without matching if statement'\n>>>> %s\n", lineno, lineCopy);
                agraph->status = -1;
                break;
            }
            ifcur = (ifcur & ~1) | !(ifall & 1);
            continue;
        }
        else if (narg == 1 && !strcmp(argv[0], "endif")) {
            if (ifdepth == 0) {
                agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: found endif without matching if statement'\n>>>> %s\n", lineno, lineCopy);
                agraph->status = -1;
                break;
            }
            ifdepth--;
            ifcur >>= 1;
            ifall >>= 1;
            continue;
        }
        // check skip if earlier conditional statements required to do so
        if (ifcur)
            continue;
        // process command with optional for-command-prefix support
        for (int N = Nbegin; N <= Nend; N += Nstep) {
            // create arguments with {N} expression substitution
            char argBuf[2048] = { 0 }, *arg[64] = { 0 };
            for (int i = 0, j = 0; i < narg; i++) {
                arg[i] = argBuf + j;
                agoUpdateN(arg[i], argv[i], N, Nchar);
                j += (int)strlen(arg[i]) + 1;
            }
            // process the actual commands
            if (narg == 4 && !strcmp(arg[0], "data") && !strcmp(arg[2], "=")) {
                // create new AgoData and add it to the dataList
                AgoData * data = agoCreateDataFromDescription(context, agraph, arg[3], false);
                if (!data) {
                    agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: data type not supported\n>>>> %s\n", lineno, lineCopy);
                    agraph->status = -1;
                    break;
                }
                data->name = arg[1];
                agoAddData(data->isVirtual ? &agraph->dataList : &context->dataList, data);
                // if data has children (e.g., pyramid, delay, image), add them too
                if (data->children) {
                    for (vx_uint32 i = 0; i < data->numChildren; i++) {
                        if (data->children[i]) {
                            for (vx_uint32 j = 0; j < data->children[i]->numChildren; j++) {
                                if (data->children[i]->children[j]) {
                                    agoAddData(data->isVirtual ? &agraph->dataList : &context->dataList, data->children[i]->children[j]);
                                }
                            }
                            agoAddData(data->isVirtual ? &agraph->dataList : &context->dataList, data->children[i]);
                        }
                    }
                }
                // inform application about data -- ignore this for scalar strings
                if (callback_f && !(data->ref.type == VX_TYPE_SCALAR && data->u.scalar.type == VX_TYPE_STRING_AMD)) {
                    // skip till ':'
                    const char * param = arg[3];
                    for (; *param && *param != ':'; param++)
                        ;
                    if (*param == ':') {
                        // still till another ':'
                        for (param++; *param && *param != ':'; param++)
                            ;
                        if (*param == ':') {
                            param++;
                            // invoke the application callback with object name and parameter strings
                            data->ref.external_count++;
                            callback_f(callback_obj, &data->ref, data->name.c_str(), param);
                        }
                    }
                }
            }
            else if ((narg >= 3 && !strcmp(arg[0], "node")) || (narg >= 3 && !strcmp(arg[0], "macro")) || (narg >= 2 && !strcmp(arg[0], "file"))) {
                std::string localSuffix = "!";
                AgoKernel * akernel = NULL;
                AgoNode * node = NULL;
                char * str_subgraph = NULL;
                bool str_subgraph_allocated = false;
                AgoReference * ref_subgraph[AGO_MAX_PARAMS] = { 0 };
                if (!strcmp(arg[0], "node")) {
                    if (!(akernel = agoFindKernelByName(context, arg[1]))) {
                        agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: kernel not supported\n>>>> %s\n", lineno, lineCopy);
                        agraph->status = -1;
                        break;
                    }
                    // create a new AgoNode and add it to the nodeList
                    node = agoCreateNode(agraph, akernel);
                }
                else if (!strcmp(arg[0], "macro")) {
                    for (auto it = context->macros.begin(); it != context->macros.end(); ++it) {
                        if (!strcmp(it->name, arg[1])) {
                            localSuffix += it->name;
                            str_subgraph = it->text;
                            break;
                        }
                    }
                    if (!str_subgraph) {
                        agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: unable to find macro '%s'\n>>>> %s\n", lineno, arg[1], lineCopy);
                        agraph->status = -1;
                        break;
                    }
                }
                else {
                    FILE * fp = fopen(arg[1], "rb");
                    if (!fp) {
                        agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: unable to open '%s'\n>>>> %s\n", lineno, arg[1], lineCopy);
                        agraph->status = -1;
                        break;
                    }
                    fseek(fp, 0L, SEEK_END); long size = ftell(fp); fseek(fp, 0L, SEEK_SET);
                    if (!(str_subgraph = new char[size + 1]()) || (fread(str_subgraph, sizeof(char), size, fp) != size)) {
                        agoAddLogEntry(&agraph->ref, VX_FAILURE, "FATAL: calloc/fread(%d) failed\n", (int)size + 1);
                        agraph->status = -1;
                        break;
                    }
                    str_subgraph_allocated = true;
                    fclose(fp);
                    // update suffix
                    const char * name = arg[1];
                    for (char *p = arg[1]; *p; p++) {
                        if (*p == '/' || *p == '\\' || *p == ':')
                            name = p + 1;
                        else if (*p == '.') *p = '\0';
                    }
                    localSuffix += name;
                }
                // look through all parameters
                for (int p = 0; p < narg - 2; p++)
                {
                    if (node && strncmp(arg[2 + p], "attr:", 5) == 0) {
                        if (!strncmp(&arg[2 + p][5], "BORDER_MODE:", 12)) {
                            if (!strcmp(&arg[2 + p][17], "UNDEFINED")) {
                                node->attr_border_mode.mode = VX_BORDER_MODE_UNDEFINED;
                                memset(&node->attr_border_mode.constant_value, 0, sizeof(node->attr_border_mode.constant_value));
                            }
                            else if (!strcmp(&arg[2 + p][17], "REPLICATE")) {
                                node->attr_border_mode.mode = VX_BORDER_MODE_REPLICATE;
                                memset(&node->attr_border_mode.constant_value, 0, sizeof(node->attr_border_mode.constant_value));
                            }
                            else if (!strncmp(&arg[2 + p][17], "CONSTANT,", 9)) {
                                node->attr_border_mode.mode = VX_BORDER_MODE_CONSTANT;
                                memset(&node->attr_border_mode.constant_value, 0, sizeof(node->attr_border_mode.constant_value));
                                (void)sscanf(&arg[2 + p][17 + 9], "%i", &node->attr_border_mode.constant_value.U32);
                            }
                            else {
                                agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: invalid/unsupported border mode attribute -- arg#%d\n>>>> %s\n", lineno, p, lineCopy);
                                agraph->status = -1;
                                break;
                            }
                        }
                        else if (!strncmp(&arg[2 + p][5], "AFFINITY:", 9)) {
                            vx_uint32 group = 0;
                            char device[64] = "CPU";
                            const char * szGroup = strstr(&arg[2 + p][14], ",");
                            if (szGroup) {
                                group = atoi(&szGroup[1]);
                            }
                            node->attr_affinity.group = group;
                            (void)sscanf(&arg[2 + p][14], "%s", device);
                            if (!strncmp(device, "CPU", 3)) {
                                node->attr_affinity.device_type = AGO_KERNEL_FLAG_DEVICE_CPU;
                                node->attr_affinity.device_info = atoi(&device[3]);
                            }
                            else if (!strncmp(device, "GPU", 3)) {
                                node->attr_affinity.device_type = AGO_KERNEL_FLAG_DEVICE_GPU;
                                node->attr_affinity.device_info = atoi(&device[3]);
                            }
                            else {
                                agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: invalid/unsupported affinity attribute -- arg#%d\n>>>> %s\n", lineno, p, lineCopy);
                                agraph->status = -1;
                                break;
                            }
                        }
                    }
                    else if (!strcmp(arg[0], "file") && strncmp(arg[2 + p], "/def-var:", 9) == 0) {
                        char command[256]; snprintf(command, sizeof(command), "def-var %s\n", &arg[2 + p][9]);
                        char * equal = strstr(command, "=");
                        if (!equal) {
                            agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: invalid def-var syntax: expected /def-var:<NAME>=<VALUE>\n>>>> %s\n", lineno, lineCopy);
                            agraph->status = -1;
                            break;
                        }
                        *equal = ' ';
                        agoReadGraphFromStringInternal(agraph, ref, num_ref, callback_f, callback_obj, command, 0, vars, localPrefix);
                        if (agraph->status)
                            break;
                    }
                    else {
                        AgoData * data = NULL;
                        if (arg[2 + p][0] == '$') {
                            int index = atoi(&arg[2 + p][1]) - 1;
                            if (index >= 0 && index < num_ref) {
                                data = (AgoData *)ref[index];
                            }
                        }
                        else if (strcmp(arg[2 + p], "null") != 0) {
                            char name[128]; strcpy(name, arg[2 + p]);
                            // check if there is an name alias
                            for (std::vector< std::pair< std::string, std::string > >::iterator it = aliases.begin(); it != aliases.end(); ++it) {
                                if (!strcmp(it->first.c_str(), name)) {
                                    strcpy(name, it->second.c_str());
                                    if (name[0] == '$') {
                                        int index = atoi(&name[1]) - 1;
                                        if (index >= 0 && index < num_ref) {
                                            data = (AgoData *)ref[index];
                                        }
                                    }
                                    break;
                                }
                            }
                            // get data object
                            if (!data) {
                                data = agoFindDataByName(context, agraph, name);
                            }
                            if (!data) {
                                // create new AgoData and add it to the dataList
                                data = agoCreateDataFromDescription(context, agraph, name, false);
                                if (!data) {
                                    agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: data type not supported -- arg#%d\n>>>> %s\n", lineno, p, lineCopy);
                                    agraph->status = -1;
                                    break;
                                }
                                agoAddData(&agraph->dataList, data);
                                // if data has children (e.g., pyramid), add them too
                                if (data->children) {
                                    for (vx_uint32 i = 0; i < data->numChildren; i++) {
                                        if (data->children[i]) {
                                            char childname[256];
                                            snprintf(childname, sizeof(childname), "%s[%d]", data->name.c_str(), i);
                                            data->children[i]->name = childname;
                                            agoAddData(&agraph->dataList, data->children[i]);
                                        }
                                    }
                                }
                            }
                        }
                        if (data) {
                            if (node) {
                                node->paramList[p] = data;
                                // check if specified data type is correct
                                // NOTE: kernel can specify to ignore this checking by setting argType[] to ZERO
                                if (akernel->argType[p] && (akernel->argType[p] != data->ref.type)) {
                                    char type_expected_buf[64], type_specified_buf[64];
                                    const char * type_expected = agoEnum2Name(akernel->argType[p]);
                                    const char * type_specified = agoEnum2Name(data->ref.type);
                                    if (!type_expected) { snprintf(type_expected_buf, sizeof(type_expected_buf), "0x%08x", akernel->argType[p]); type_expected = type_expected_buf; }
                                    if (!type_specified) { snprintf(type_specified_buf, sizeof(type_expected_buf), "0x%08x", data->ref.type); type_specified = type_specified_buf; }
                                    agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: data type %s expected -- arg#%d has %s\n>>>> %s\n", lineno, type_expected, p, type_specified, lineCopy);
                                    agraph->status = -1;
                                    break;
                                }
                            }
                            else if (p < AGO_MAX_PARAMS)
                                ref_subgraph[p] = &data->ref;
                            else {
                                agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: number of arguments exceeded MAX(%d)\n>>>> %s\n", lineno, AGO_MAX_PARAMS, lineCopy);
                                agraph->status = -1;
                                break;
                            }
                        }
                    }
                }
                if (str_subgraph && !agraph->status) {
                    agoReadGraphFromStringInternal(agraph, ref_subgraph, narg - 2, callback_f, callback_obj, str_subgraph, (dumpToConsole > 0) ? dumpToConsole - 1 : vx_false_e, vars, localPrefix + localSuffix);
                }
                if (str_subgraph_allocated)
                    delete[] str_subgraph;
                if (agraph->status)
                    break;
            }
            else if (narg == 2 && !strcmp(arg[0], "import")) {
                char * module_name = arg[1];
                if (agoLoadModule(context, module_name)) {
                    agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: import: unable to load module: %s\n", module_name);
                    agraph->status = -1;
                    break;
                }
            }
            else if (narg == 3 && !strcmp(arg[0], "type") && !strncmp(arg[2], "userstruct:", 11)) {
                vx_enum user_struct_id = 0;
                char * name = arg[1];
                if (agoGetUserStructSize(context, name) > 0) {
                    agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: type: name already in-use: %s\n", name);
                    agraph->status = -1;
                    break;
                }
                vx_size size = atoi(&arg[2][11]);
                if (agoAddUserStruct(context, size, name) == VX_TYPE_INVALID) {
                    agraph->status = -1;
                }
            }
            else if (narg == 2 && !strcmp(arg[0], "def-macro")) {
                char macro_name[256]; strncpy(macro_name, arg[1], sizeof(macro_name));
                const char * str_begin = str;
                const char * str_end = str;
                for (; (str = agoReadLine(line, sizeof(line)-16, str)) != NULL; lineno++) {
                    if (dumpToConsole) agoAddLogEntry(NULL, VX_SUCCESS, "%s", line);
                    agoUpdateLine(line, vars, localPrefix);
                    char word[256];
                    if (sscanf(line, "%s", word) == 1 && !strcmp(word, "endmacro"))
                        break;
                    str_end = str;
                }
                if (!str) {
                    agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: incomplete macro definition: %s\n>>>> %s\n", lineno, macro_name, lineCopy);
                    agraph->status = -1;
                    break;
                }
                else {
                    for (auto it = context->macros.begin(); it != context->macros.end(); ++it) {
                        if (!strcmp(it->name, macro_name)) {
                            agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: macro already exists: %s\n>>>> %s\n", lineno, macro_name, lineCopy);
                            agraph->status = -1;
                            break;
                        }
                    }
                    if (agraph->status)
                        break;
                    else {
                        MacroData macro;
                        macro.text = macro.text_allocated = (char *)calloc(1, str_end - str_begin + 1);
                        strncpy(macro.name, macro_name, sizeof(macro.name) - 1);
                        strncpy(macro.text, str_begin, str_end - str_begin);
                        context->macros.push_back(macro);
                    }
                }
            }
            else if ((narg == 2 || narg == 3) && (!strcmp(arg[0], "def-var") || !strcmp(arg[0], "def-var-default"))) {
                for (vx_uint32 i = 0; arg[1][i]; i++) {
                    char c = arg[1][i];
                    if (!(i == 0 && c >= 'A' && c <= 'Z') && !(i > 0 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'))) {
                        agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: variable names can only have [A-Z][A-Za-z0-9_]* format: %s\n>>>> %s\n", lineno, arg[1], lineCopy);
                        agraph->status = -1;
                        break;
                    }
                }
                if (agraph->status)
                    break;
                bool found = false;
                for (std::vector< std::pair< std::string, std::string > >::iterator it = vars.begin(); it != vars.end(); ++it) {
                    if (!strcmp(it->first.c_str(), arg[1])) {
                        found = true;
                        if (!strcmp(arg[0], "def-var")) {
                            agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: variable already exists: %s\n>>>> %s\n", lineno, arg[1], lineCopy);
                            agraph->status = -1;
                        }
                        break;
                    }
                }
                if (agraph->status)
                    break;
                else if (!found) {
                    char value[2048];
                    if (narg == 2) {
                        value[0] = 0;
                    }
                    else {
                        strcpy(value, arg[2]);
                        agoEvaluateIntegerExpression(value);
                    }
                    if ((!strncmp(value, "WIDTH(", 6) || !strncmp(value, "HEIGHT(", 7) || !strncmp(value, "FORMAT(", 7)) && value[strlen(value) - 1] == ')') {
                        char * name = strstr(value, "(") + 1; value[strlen(value) - 1] = 0;
                        AgoData * pdata = agoFindDataByName(context, agraph, name);
                        if (!pdata && name[0] == '$' && name[1] >= '1' && name[1] <= '9') {
                            int v = atoi(&name[1]) - 1;
                            if (v < num_ref)
                                pdata = (AgoData *)ref[v];
                            if (!pdata) {
                                agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: specified argument is not available: %s\n>>>> %s\n", lineno, arg[2], lineCopy);
                                agraph->status = -1;
                                break;
                            }
                        }
                        if (!pdata || (pdata->ref.type != VX_TYPE_IMAGE && pdata->ref.type != VX_TYPE_PYRAMID)) {
                            agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: invalid data name specified: %s\n>>>> %s\n", lineno, name, lineCopy);
                            agraph->status = -1;
                            break;
                        }
                        if (!strncmp(value, "WIDTH(", 6)) {
                            int v = 0;
                            if (pdata->ref.type == VX_TYPE_IMAGE) v = pdata->u.img.width;
                            else if (pdata->ref.type == VX_TYPE_PYRAMID) v = pdata->u.pyr.width;
                            snprintf(value, sizeof(value), "%d", v);
                        }
                        else if (!strncmp(value, "HEIGHT(", 7)) {
                            int v = 0;
                            if (pdata->ref.type == VX_TYPE_IMAGE) v = pdata->u.img.height;
                            else if (pdata->ref.type == VX_TYPE_PYRAMID) v = pdata->u.pyr.height;
                            snprintf(value, sizeof(value), "%d", v);
                        }
                        else if (!strncmp(value, "FORMAT(", 7)) {
                            vx_df_image v = VX_DF_IMAGE_U8;
                            if (pdata->ref.type == VX_TYPE_IMAGE) v = pdata->u.img.format;
                            else if (pdata->ref.type == VX_TYPE_PYRAMID) v = pdata->u.pyr.format;
                            snprintf(value, sizeof(value), "%4.4s", FORMAT_STR(v));
                        }
                    }
                    vars.push_back(std::pair< std::string, std::string >(arg[1], value));
                    // special AGO flags
                    if (!strcmp(arg[1], "AgoOptimizerFlags") && value[0] >= '0' && value[0] <= '9') {
                        agraph->optimizer_flags = atoi(value);
                        break;
                    }
                }
            }
            else if (narg == 2 && !strcmp(arg[0], "affinity")) {
                if (!strcmp(arg[1], "GPU")) {
                    agraph->attr_affinity.device_type = AGO_KERNEL_FLAG_DEVICE_GPU;
                }
                else if (!strcmp(arg[1], "CPU")) {
                    agraph->attr_affinity.device_type = AGO_KERNEL_FLAG_DEVICE_CPU;
                }
                else {
                    agraph->attr_affinity.device_type = 0;
                }
            }
            else if (narg == 3 && !strcmp(arg[0], "alias")) {
                for (vx_uint32 i = 0; arg[1][i]; i++) {
                    char c = arg[1][i];
                    if (!(i == 0 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) && !(i > 0 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'))) {
                        agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: alias names can only have [A-Z][A-Za-z0-9_]* format: %s\n>>>> %s\n", lineno, arg[1], lineCopy);
                        agraph->status = -1;
                        break;
                    }
                }
                if (agraph->status)
                    break;
                char name1[128]; agoUpdateN(name1, arg[1], 0, '\0');
                char name2[128]; agoUpdateN(name2, arg[2], 0, '\0');
                for (std::vector< std::pair< std::string, std::string > >::iterator it = aliases.begin(); it != aliases.end(); ++it) {
                    if (!strcmp(it->first.c_str(), name1)) {
                        agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: alias already exists: %s\n>>>> %s\n", lineno, name1, lineCopy);
                        agraph->status = -1;
                        break;
                    }
                }
                if (agraph->status)
                    break;
                aliases.push_back(std::pair< std::string, std::string >(name1, name2));
            }
            else if (narg > 0 && !strcmp(arg[0], "set-args")) {
                if (narg - 1 > num_ref) {
                    agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: set-args: number of argument (%d) exceeded internal buffer (%d)\n>>>> %s\n", lineno, narg - 1, num_ref, lineCopy);
                    agraph->status = -1;
                    break;
                }
                // clear all previous arguments
                for (int i = 0; i < num_ref; i++) {
                    // TBD handle memory leaks
                    ref[i] = NULL;
                }
                for (int j = 0; j < narg - 1; j++) {
                    // create new AgoData and add it to the dataList
                    AgoData * data = agoCreateDataFromDescription(context, agraph, arg[j + 1], false);
                    if (!data) {
                        agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: set-args: invalid object description: %s\n>>>> %s\n", lineno, arg[j + 1], lineCopy);
                        agraph->status = -1;
                        break;
                    }
                    agoAddData(data->isVirtual ? &agraph->dataList : &agraph->ref.context->dataList, data);
                    // if data has children (e.g., pyramid), add them too
                    if (data->children) {
                        for (vx_uint32 i = 0; i < data->numChildren; i++) {
                            if (data->children[i]) {
                                agoAddData(data->isVirtual ? &agraph->dataList : &agraph->ref.context->dataList, data->children[i]);
                            }
                        }
                    }
                    ref[j] = &data->ref;
                }
                if (agraph->status)
                    break;
            }
            else if (narg == 3 && !strcmp(arg[0], "directive")) {
                agraph->status = -1;
                char name1[128]; agoUpdateN(name1, arg[1], 0, '\0');
                char name2[128]; agoUpdateN(name2, arg[2], 0, '\0');
                AgoData * data = agoFindDataByName(context, agraph, name1);
                if (data) {
                    vx_enum directive = agoName2Enum(name2);
                    if (!directive) {
                        (void)sscanf(name2, "%i", &directive);
                    }
                    if (directive) {
                        agraph->status = agoDirective((vx_reference)data, directive);
                    }
                }
                if (agraph->status) {
                    agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: invalid object or directive: directive %s %s\n>>>> %s\n", lineno, name1, name2, lineCopy);
                    break;
                }
            }
            else if (narg >= 1 && !strcmp(arg[0], "exit")) {
                break;
            }
            else if (narg > 0) {
                agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: line %d: syntax error\n>>>> %s\n", lineno, lineCopy);
                agraph->status = -1;
                break;
            }
        }
        if (agraph->status)
            break;
    }
}

int agoReadGraph(AgoGraph * agraph, AgoReference * * ref, int num_ref, ago_data_registry_callback_f callback_f, void * callback_obj, FILE * fp, vx_int32 dumpToConsole)
{
    if (!agraph) return -1;
    vx_context context = agraph->ref.context;
    CAgoLock lock(agraph->cs);
    CAgoLock lock2(context->cs);

    // read the whole file into a local buffer
    long cur = ftell(fp); fseek(fp,  0L, SEEK_END);
    long end = ftell(fp); fseek(fp, cur, SEEK_SET);
    long size = end - cur; if (size < 1) return agraph->status;
    char * str = new char [size + 1]();
    if (!str || (fread(str, sizeof(char), size, fp) != size))
        return -1;

    // read the graph from file
    std::vector< std::pair< std::string, std::string > > vars;
    agoReadGraphFromStringInternal(agraph, ref, num_ref, callback_f, callback_obj, str, dumpToConsole, vars, "L");
    delete[] str;

    // mark the scope of all virtual data to graph
    for (AgoData * data = agraph->dataList.head; data; data = data->next) {
        data->ref.scope = &agraph->ref;
    }
    // check if graph is a valid graph
    if (agraph->status == VX_SUCCESS) {
        agraph->status = agoVerifyGraph(agraph);
        if (agraph->status) {
            agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: agoVerifyGraph() => %d (failed)\n", agraph->status);
        }
    }
    return agraph->status;
}

int agoReadGraphFromString(AgoGraph * agraph, AgoReference * * ref, int num_ref, ago_data_registry_callback_f callback_f, void * callback_obj, char * str, vx_int32 dumpToConsole)
{
    if (!agraph) return -1;
    vx_context context = agraph->ref.context;
    CAgoLock lock(agraph->cs);
    CAgoLock lock2(context->cs);

    // read the graph from string
    std::vector< std::pair< std::string, std::string > > vars;
    agoReadGraphFromStringInternal(agraph, ref, num_ref, callback_f, callback_obj, str, dumpToConsole, vars, "L");

    // mark the scope of all virtual data to graph
    for (AgoData * data = agraph->dataList.head; data; data = data->next) {
        data->ref.scope = &agraph->ref;
    }
    // check if graph is a valid graph
    if (agraph->status == VX_SUCCESS) {
        agraph->status = agoVerifyGraph(agraph);
        if (agraph->status) {
            agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoReadGraph: agoVerifyGraph() => %d (failed)\n", agraph->status);
        }
    }
    return agraph->status;
}

int agoLoadModule(AgoContext * context, const char * module)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context)) {
        CAgoLock lock(context->cs);
        status = VX_ERROR_INVALID_PARAMETERS;
        char filePath[1024]; snprintf(filePath, sizeof(filePath), SHARED_LIBRARY_PREFIX "%s" SHARED_LIBRARY_EXTENSION, module);
        for (vx_uint32 index = 0; index < context->num_active_modules; index++) {
            if (strcmp(filePath, context->modules[index].module_path) == 0) {
                agoAddLogEntry(&context->ref, VX_SUCCESS, "WARNING: kernels already loaded from %s\n", filePath);
                return VX_SUCCESS;
            }
        }
        ago_module hmodule = agoOpenModule(filePath);
        if (hmodule == NULL) {
            status = VX_ERROR_INVALID_MODULE;
            agoAddLogEntry(&context->ref, status, "ERROR: Unable to load module %s\n", filePath);
        }
        else {
            vx_publish_kernels_f publish_kernels_f = (vx_publish_kernels_f)agoGetFunctionAddress(hmodule, "vxPublishKernels");
            if (!publish_kernels_f) {
                status = VX_ERROR_INVALID_MODULE;
                agoAddLogEntry(&context->ref, status, "ERROR: vxPublishKernels symbol missing in %s\n", filePath);
            }
            else {
                // add module entry into context
                ModuleData data;
                strncpy(data.module_name, module, sizeof(data.module_name) - 1);
                strncpy(data.module_path, filePath, sizeof(data.module_path) - 1);
                data.hmodule = hmodule;
                data.module_internal_data_ptr = nullptr;
                data.module_internal_data_size = 0;
                context->modules.push_back(data);
                context->num_active_modules++;
                // invoke vxPublishKernels
                vx_uint32 count = context->kernelList.count;
                context->importing_module_index_plus1 = context->num_active_modules;
                status = publish_kernels_f(context);
                context->importing_module_index_plus1 = 0;
                if (status == VX_SUCCESS) {
                    agoAddLogEntry(&context->ref, VX_SUCCESS, "OK: loaded %d kernels from %s\n", context->kernelList.count - count, filePath);
                }
                else {
                    agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: vxPublishKernels(%s) failed (%d:%s)\n", module, status, agoEnum2Name(status));
                }
            }
        }
    }
    return status;
}

int agoUnloadModule(AgoContext * context, const char * module)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidContext(context)) {
        CAgoLock lock(context->cs);
        status = VX_ERROR_INVALID_MODULE;
        char filePath[1024]; snprintf(filePath, sizeof(filePath), SHARED_LIBRARY_PREFIX "%s" SHARED_LIBRARY_EXTENSION, module);
        for (vx_uint32 index = 0; index < context->num_active_modules; index++) {
            if (strcmp(filePath, context->modules[index].module_path) == 0) {
                vx_unpublish_kernels_f unpublish_kernels_f = (vx_unpublish_kernels_f)agoGetFunctionAddress(context->modules[index].hmodule, "vxUnpublishKernels");
                if (!unpublish_kernels_f) {
                    status = VX_ERROR_NOT_SUPPORTED;
                    agoAddLogEntry(&context->ref, status, "ERROR: vxUnpublishKernels symbol missing in %s\n", filePath);
                }
                else {
                    status = unpublish_kernels_f(context);
                    if (status == VX_SUCCESS) {
                        agoCloseModule(context->modules[index].hmodule);
                        context->modules.erase (context->modules.begin()+index);
                        context->num_active_modules--;
                    }
                }
                break;
            }
        }
    }
    return status;
}

vx_status agoVerifyNode(AgoNode * node)
{
    AgoGraph * graph = (AgoGraph *)node->ref.scope;
    AgoKernel * kernel = node->akernel;
    vx_status status = VX_SUCCESS;

    // check if node has required arguments and initialize data required for further graph processing
    node->hierarchical_level = 0;
    for (vx_uint32 arg = 0; arg < AGO_MAX_PARAMS; arg++) {
        AgoData * data = node->paramList[arg];
        if (!data || (arg >= node->paramCount)) {
            if (((kernel->argConfig[arg] & AGO_KERNEL_ARG_OPTIONAL_FLAG) == 0) && ((kernel->argConfig[arg] & (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) != 0)) {
                agoAddLogEntry(&kernel->ref, VX_ERROR_NOT_SUFFICIENT, "ERROR: agoVerifyGraph: kernel %s: missing argument#%d\n", kernel->name, arg);
                return VX_ERROR_NOT_SUFFICIENT;
            }
        }
        else if ((kernel->argConfig[arg] & (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) == 0) {
            agoAddLogEntry(&kernel->ref, VX_ERROR_NOT_SUFFICIENT, "ERROR: agoVerifyGraph: kernel %s: unexpected argument#%d\n", kernel->name, arg);
            return VX_ERROR_NOT_SUFFICIENT;
        }
        if (data) {
            data->hierarchical_level = 0;
            // reset meta data of the node for output argument processing
            if ((kernel->argConfig[arg] & (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) == AGO_KERNEL_ARG_OUTPUT_FLAG) {
                vx_meta_format meta = &node->metaList[arg];
                meta->data.ref.type = data->ref.type;
                meta->data.ref.external_count = 1;
                meta->set_valid_rectangle_callback = nullptr;
                if (data->ref.type == VX_TYPE_IMAGE) {
                    meta->data.u.img.rect_valid.start_x = 0;
                    meta->data.u.img.rect_valid.start_y = 0;
                    meta->data.u.img.rect_valid.end_x = INT_MAX;
                    meta->data.u.img.rect_valid.end_y = INT_MAX;
                }
                else if (data->ref.type == VX_TYPE_PYRAMID) {
                    meta->data.u.pyr.rect_valid.start_x = 0;
                    meta->data.u.pyr.rect_valid.start_y = 0;
                    meta->data.u.pyr.rect_valid.end_x = INT_MAX;
                    meta->data.u.pyr.rect_valid.end_y = INT_MAX;
                }
            }
        }
    }

    // mark the kernels with VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_UPDATE_CALLBACK
    // with enableUserBufferGPU
    if (kernel->gpu_buffer_update_callback_f) {
        AgoData * data = node->paramList[kernel->gpu_buffer_update_param_index];
        if (!data || !data->isVirtual || data->ref.type != VX_TYPE_IMAGE || data->u.img.planes != 1 || data->ownerOfUserBufferGPU || data->u.img.enableUserBufferGPU) {
            status = VX_ERROR_INVALID_PARAMETERS;
            agoAddLogEntry(&kernel->ref, status, "ERROR: agoVerifyGraph: kernel %s: unexpected/unsupported argument#%d -- needs virtual image with single-plane\n", kernel->name, kernel->gpu_buffer_update_param_index);
            return status;
        }
        // mark that the buffer gets initialized a node
        data->u.img.enableUserBufferGPU = vx_true_e;
        data->ownerOfUserBufferGPU = node;
    }
    // check if node arguments are valid
    if (kernel->func) {
        // validate arguments for built-in kernel functions
        vx_status status = kernel->func(node, ago_kernel_cmd_validate);
        if (status) {
            agoAddLogEntry(&kernel->ref, status, "ERROR: agoVerifyGraph: kernel %s: ago_kernel_cmd_validate failed (%d)\n", kernel->name, status);
            return status;
        }
    }
    else if (kernel->validate_f) {
        // validate arguments for user-kernels functions
        vx_meta_format metaList[AGO_MAX_PARAMS];
        for (int i = 0; i < AGO_MAX_PARAMS; i++)
            metaList[i] = &node->metaList[i];
        status = kernel->validate_f(node, (vx_reference *)node->paramList, node->paramCount, metaList);
        if (status) {
            agoAddLogEntry(&kernel->ref, status, "ERROR: agoVerifyGraph: kernel %s: kernel_validate failed (%d)\n", kernel->name, status);
            return status;
        }
    }
    else {
        // check if node input arguments are valid
        for (vx_uint32 arg = 0; arg < node->paramCount; arg++) {
            if (node->paramList[arg]) {
                if (kernel->argConfig[arg] & AGO_KERNEL_ARG_INPUT_FLAG) {
                    if (kernel->input_validate_f) {
                        vx_status status = kernel->input_validate_f(node, arg);
                        if (status) {
                            agoAddLogEntry(&kernel->ref, status, "ERROR: agoVerifyGraph: kernel %s: input_validate failed (%d) for argument#%d\n", kernel->name, status, arg);
                            return status;
                        }
                    }
                }
            }
        }
        // check if node output arguments are valid
        for (vx_uint32 arg = 0; arg < node->paramCount; arg++) {
            AgoData * data = node->paramList[arg];
            if (data) {
                if ((kernel->argConfig[arg] & (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) == AGO_KERNEL_ARG_OUTPUT_FLAG) {
                    if (kernel->output_validate_f) {
                        vx_meta_format meta = &node->metaList[arg];
                        vx_status status = kernel->output_validate_f(node, arg, meta);
                        if (status) {
                            agoAddLogEntry(&kernel->ref, status, "ERROR: agoVerifyGraph: kernel %s: output_validate failed (%d) for argument#%d\n", kernel->name, status, arg);
                            return status;
                        }
                    }
                }
            }
        }
    }
    // check if node output arguments are valid
    for (vx_uint32 arg = 0; arg < node->paramCount; arg++) {
        AgoData * data = node->paramList[arg];
        if (data) {
            if ((kernel->argConfig[arg] & (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) == AGO_KERNEL_ARG_OUTPUT_FLAG) {
                vx_meta_format meta = &node->metaList[arg];
                if (kernel->argType[arg] && kernel->argType[arg] != VX_TYPE_REFERENCE && (meta->data.ref.type != kernel->argType[arg])) {
                    agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_TYPE, "ERROR: agoVerifyGraph: kernel %s: output argument type mismatch for argument#%d\n", kernel->name, arg);
                    return VX_ERROR_INVALID_TYPE;
                }
                else if (meta->data.ref.type == VX_TYPE_IMAGE) {
                    bool updated = false;
                    if (data->isVirtual) {
                        // update format/width/height if not specified
                        if (data->u.img.format == VX_DF_IMAGE_VIRT) {
                            data->u.img.format = meta->data.u.img.format;
                            updated = true;
                        }
                        if (data->u.img.width == 0) {
                            data->u.img.width = meta->data.u.img.width;
                            updated = true;
                        }
                        if (data->u.img.height == 0) {
                            data->u.img.height = meta->data.u.img.height;
                            updated = true;
                        }
                    }
                    // make sure that the data come from output validator matches with object
                    if (data->u.img.format != meta->data.u.img.format) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_FORMAT, "ERROR: agoVerifyGraph: kernel %s: invalid format for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_FORMAT;
                    }
                    else if (data->u.img.width != meta->data.u.img.width || data->u.img.height != meta->data.u.img.height) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_DIMENSION, "ERROR: agoVerifyGraph: kernel %s: invalid dimension for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_DIMENSION;
                    }
                    // re-initialize, if updated
                    if (updated) {
                        char desc[64]; snprintf(desc, sizeof(desc), "image-virtual:%4.4s,%d,%d", FORMAT_STR(data->u.img.format), data->u.img.width, data->u.img.height);
                        data->isNotFullyConfigured = vx_true_e;
                        if (agoGetDataFromDescription(graph->ref.context, graph, data, desc)) {
                            agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoVerifyGraph: agoVerifyGraph update failed for virtual-image: %4.4s %dx%d\n", FORMAT_STR(data->u.img.format), data->u.img.width, data->u.img.height);
                            return -1;
                        }
                        if (data->children) {
                            for (vx_uint32 i = 0; i < data->numChildren; i++) {
                                agoAddData(&graph->dataList, data->children[i]);
                            }
                        }
                        data->isNotFullyConfigured = vx_false_e;
                    }
                    // update valid rectangle
                    data->u.img.rect_valid = meta->data.u.img.rect_valid;
                    if (data->u.img.rect_valid.end_x == INT_MAX)
                        data->u.img.rect_valid.end_x = data->u.img.width;
                    if (data->u.img.rect_valid.end_y == INT_MAX)
                        data->u.img.rect_valid.end_y = data->u.img.height;
                    // check for VX_IMAGE_ATTRIBUTE_AMD_ENABLE_USER_BUFFER_GPU attribute
                    if (meta->data.u.img.enableUserBufferGPU) {
                        // supports only virtual images without ROI (planes commented out for accepting NV12 user buffer for amd_media)
                        if (!data->isVirtual /*|| data->u.img.planes != 1 */|| data->u.img.isROI || data->ownerOfUserBufferGPU) {
                            agoAddLogEntry(&kernel->ref, VX_ERROR_NOT_SUPPORTED, "ERROR: agoVerifyGraph: kernel %s: VX_IMAGE_ATTRIBUTE_AMD_ENABLE_USER_BUFFER_GPU is not supported for argument#%d\n", kernel->name, arg);
                            return VX_ERROR_NOT_SUPPORTED;
                        }
                        data->u.img.enableUserBufferGPU = vx_true_e;
                    }
                }
                else if (meta->data.ref.type == VX_TYPE_PYRAMID) {
                    bool updated = false;
                    if (data->isVirtual) {
                        // update format/width/height if not specified
                        if (data->u.pyr.format == VX_DF_IMAGE_VIRT) {
                            data->u.pyr.format = meta->data.u.pyr.format;
                            updated = true;
                        }
                        if (data->u.pyr.width == 0) {
                            data->u.pyr.width = meta->data.u.pyr.width;
                            updated = true;
                        }
                        if (data->u.pyr.height == 0) {
                            data->u.pyr.height = meta->data.u.pyr.height;
                            updated = true;
                        }
                    }
                    // make sure that the data come from output validator matches with object
                    if (data->u.pyr.levels != meta->data.u.pyr.levels || data->u.pyr.scale != meta->data.u.pyr.scale) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_VALUE, "ERROR: agoVerifyGraph: kernel %s: invalid value for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_VALUE;
                    }
                    else if (data->u.pyr.format != meta->data.u.pyr.format) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_FORMAT, "ERROR: agoVerifyGraph: kernel %s: invalid format for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_FORMAT;
                    }
                    else if (data->u.pyr.width != meta->data.u.pyr.width || data->u.pyr.height != meta->data.u.pyr.height) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_DIMENSION, "ERROR: agoVerifyGraph: kernel %s: invalid dimension for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_DIMENSION;
                    }
                    // re-initialize, if updated
                    if (updated) {
                        char scale[256], desc[512];
                        if (data->u.pyr.scale == VX_SCALE_PYRAMID_HALF) snprintf(scale, sizeof(scale), "HALF");
                        else if (data->u.pyr.scale == VX_SCALE_PYRAMID_ORB) snprintf(scale, sizeof(scale), "ORB");
                        else snprintf(scale, sizeof(scale), "%g", data->u.pyr.scale);
                        snprintf(desc, sizeof(desc), "pyramid-virtual:%4.4s,%d,%d," VX_FMT_SIZE ",%s", FORMAT_STR(data->u.pyr.format), data->u.pyr.width, data->u.pyr.height, data->u.pyr.levels, scale);
                        data->isNotFullyConfigured = vx_true_e;
                        if (agoGetDataFromDescription(graph->ref.context, graph, data, desc)) {
                            agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoVerifyGraph: agoVerifyGraph update failed for %s\n", desc);
                            return -1;
                        }
                        if (data) {
                            agoAddData(&graph->dataList, data);
                            // add the children too
                            for (vx_uint32 i = 0; i < data->numChildren; i++) {
                                agoAddData(&graph->dataList, data->children[i]);
                                for (vx_uint32 j = 0; j < data->children[i]->numChildren; j++) {
                                    if (data->children[i]->children[j]) {
                                        agoAddData(&graph->dataList, data->children[i]->children[j]);
                                    }
                                }
                            }
                        }
                        data->isNotFullyConfigured = vx_false_e;
                    }
                    // update valid rectangle
                    data->u.pyr.rect_valid = meta->data.u.pyr.rect_valid;
                    if (data->u.pyr.rect_valid.end_x == INT_MAX)
                        data->u.pyr.rect_valid.end_x = data->u.pyr.width;
                    if (data->u.pyr.rect_valid.end_y == INT_MAX)
                        data->u.pyr.rect_valid.end_y = data->u.pyr.height;
                    // propagate valid rectangle to all images inside the pyramid
                    for (vx_uint32 i = 0; i < data->numChildren; i++) {
                        AgoData * img = data->children[i];
                        if (img) {
                            vx_float32 xscale = (vx_float32)img->u.img.width / (vx_float32)data->u.pyr.width;
                            vx_float32 yscale = (vx_float32)img->u.img.height / (vx_float32)data->u.pyr.height;
                            img->u.img.rect_valid.start_x = (vx_uint32)ceilf(data->u.pyr.rect_valid.start_x * xscale);
                            img->u.img.rect_valid.start_y = (vx_uint32)ceilf(data->u.pyr.rect_valid.start_y * yscale);
                            img->u.img.rect_valid.end_x = (vx_uint32)floorf(data->u.pyr.rect_valid.end_x * xscale);
                            img->u.img.rect_valid.end_y = (vx_uint32)floorf(data->u.pyr.rect_valid.end_y * yscale);
                        }
                    }
                }
                else if (meta->data.ref.type == VX_TYPE_ARRAY) {
                    bool updated = false;
                    if (data->isVirtual) {
                        // update itemtype/capacity if not specified
                        if (data->u.arr.itemtype == VX_TYPE_INVALID) {
                            data->u.arr.itemtype = meta->data.u.arr.itemtype;
                            updated = true;
                        }
                        if (data->u.arr.capacity == 0) {
                            data->u.arr.capacity = meta->data.u.arr.capacity;
                            updated = true;
                        }
                    }
                    // make sure that the data come from output validator matches with object
                    if (data->u.arr.itemtype != meta->data.u.arr.itemtype) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_TYPE, "ERROR: agoVerifyGraph: kernel %s: invalid array type for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_TYPE;
                    }
                    else if (!data->u.arr.capacity || (meta->data.u.arr.capacity && meta->data.u.arr.capacity > data->u.arr.capacity)) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_DIMENSION, "ERROR: agoVerifyGraph: kernel %s: invalid dimension for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_DIMENSION;
                    }
                    if (updated) {
                        data->isNotFullyConfigured = vx_true_e;
                        char desc[64]; snprintf(desc, sizeof(desc), "array-virtual:%u,%u", data->u.arr.itemtype, (vx_uint32)data->u.arr.capacity);
                        if (agoGetDataFromDescription(graph->ref.context, graph, data, desc)) {
                            agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: agoVerifyGraph: agoVerifyGraph update failed for %s\n", desc);
                            return -1;
                        }
                        data->isNotFullyConfigured = vx_false_e;
                    }
                }
                else if (meta->data.ref.type == VX_TYPE_OBJECT_ARRAY) {
                    bool updated = false;
                    if (data->isVirtual) {
                        // update itemtype if not specified
                        if (data->u.objarr.itemtype == VX_TYPE_INVALID) {
                            data->u.objarr.itemtype = meta->data.u.objarr.itemtype;
                            updated = true;
                        }
                        if (data->u.objarr.numitems == 0) {
                            data->u.objarr.numitems = meta->data.u.objarr.numitems;
                            updated = true;
                        }
                    }
                    // make sure that the data come from output validator matches with object
                    if (data->u.objarr.itemtype != meta->data.u.objarr.itemtype) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_TYPE, "ERROR: agoVerifyGraph: kernel %s: invalid object-array type for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_TYPE;
                    }
                    else if (!data->u.objarr.numitems || (meta->data.u.objarr.numitems && meta->data.u.objarr.numitems > data->u.objarr.numitems)) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_DIMENSION, "ERROR: agoVerifyGraph: kernel %s: invalid dimension for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_DIMENSION;
                    }
                    if (updated) {
                        data->isNotFullyConfigured = vx_true_e;
                        char desc[64]; snprintf(desc, sizeof(desc), "objectarray-virtual:%u", data->u.objarr.itemtype);
                        if (agoGetDataFromDescription(graph->ref.context, graph, data, desc)) {
                            agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: agoVerifyGraph: agoVerifyGraph update failed for %s\n", desc);
                            return -1;
                        }
                        data->isNotFullyConfigured = vx_false_e;
                    }
                }
                else if (meta->data.ref.type == VX_TYPE_SCALAR) {
                    // make sure that the data come from output validator matches with object
                    if (data->u.scalar.type != meta->data.u.scalar.type) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_TYPE, "ERROR: agoVerifyGraph: kernel %s: invalid type for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_TYPE;
                    }
                }
                else if (meta->data.ref.type == VX_TYPE_MATRIX) {
                    // make sure that the data come from output validator matches with object
                    if ((data->u.mat.type != meta->data.u.mat.type) || (data->u.mat.columns != meta->data.u.mat.columns) || (data->u.mat.rows != meta->data.u.mat.rows)) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_TYPE, "ERROR: agoVerifyGraph: kernel %s: invalid matrix meta for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_TYPE;
                    }
                }
                else if (meta->data.ref.type == VX_TYPE_DISTRIBUTION) {
                    // nothing to do
                }
                else if (meta->data.ref.type == VX_TYPE_THRESHOLD) {
                    if ((data->u.thr.thresh_type != meta->data.u.thr.thresh_type)) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_TYPE, "ERROR: agoVerifyGraph: kernel %s: invalid threshold meta for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_TYPE;
                    }
                }
                else if (meta->data.ref.type == VX_TYPE_LUT) {
                    // nothing to do
                }
                else if (meta->data.ref.type == VX_TYPE_REMAP) {
                    // nothing to do
                }
                else if (meta->data.ref.type == VX_TYPE_TENSOR) {
                    // make sure that the data come from output validator matches with object
                    bool mismatched = false;
                    if ((data->u.tensor.num_dims != meta->data.u.tensor.num_dims) || (data->u.tensor.data_type != meta->data.u.tensor.data_type) || (data->u.tensor.fixed_point_pos != meta->data.u.tensor.fixed_point_pos)) {
                        mismatched = true;
                    }
                    for (vx_size i = 0; i < data->u.tensor.num_dims; i++) {
                        if (data->u.tensor.dims[i] != meta->data.u.tensor.dims[i])
                            mismatched = true;
                    }
                    if (mismatched) {
                        agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_TYPE, "ERROR: agoVerifyGraph: kernel %s: invalid tensor meta for argument#%d\n", kernel->name, arg);
                        return VX_ERROR_INVALID_TYPE;
                    }
                }
                else if (meta->data.ref.type == AGO_TYPE_CANNY_STACK) {
                    // nothing to do
                }
                else if (meta->data.ref.type == AGO_TYPE_MINMAXLOC_DATA) {
                    // nothing to do
                }
                else if (meta->data.ref.type == AGO_TYPE_MEANSTDDEV_DATA) {
                    // nothing to do
                }
                else if (kernel->argType[arg]) {
                    agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_TYPE, "ERROR: agoVerifyGraph: kernel %s: invalid type for argument#%d\n", kernel->name, arg);
                    return VX_ERROR_INVALID_TYPE;
                }
            }
            else if ((kernel->argConfig[arg] & (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) == (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) {
#if 0 // TBD: disabled temporarily as a quick workaround for bidirectional buffer issue
                // virtual objects can not be used as bidirectional arguments
                if (data->isVirtual) {
                    agoAddLogEntry(&kernel->ref, VX_ERROR_INVALID_PARAMETERS, "ERROR: agoVerifyGraph: kernel %s: bidirectional argument shouldn't be virtual for argument#%d (%s)\n", kernel->name, arg, data->name.c_str());
                    return VX_ERROR_INVALID_PARAMETERS;
                }
#endif
            }
        }
    }
    return status;
}

int agoVerifyGraph(AgoGraph * graph)
{
    // compute node hierarchy in the graph: this takes care of
    //    - single writers
    //    - no loops
    vx_status status = agoOptimizeDramaComputeGraphHierarchy(graph);
    if (status) {
        return status;
    }
    agoOptimizeDramaSortGraphHierarchy(graph);

    // initialize valid region every input image/pyramid to its full region
    // and reset the user virtul buffer owner
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        for (vx_uint32 i = 0; i < node->paramCount; i++) {
            AgoData * data = node->paramList[i];
            if (data) {
                if (data->ref.type == VX_TYPE_IMAGE) {
                    if (data->isVirtual) {
                        data->ownerOfUserBufferGPU = nullptr;
                    }
                }
            }
        }
    }

    ////////////////////////////////////////////////
    // validate node arguments
    ////////////////////////////////////////////////
    graph->detectedInvalidNode = false;
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        status = agoVerifyNode(node);
        if (status) {
            return status;
        }
    }

    // compute node hierarchy in the graph: this takes care of
    //    - single writers
    //    - no loops
    status = agoOptimizeDramaComputeGraphHierarchy(graph);
    if (status) {
        return status;
    }

#if (ENABLE_OPENCL || ENABLE_HIP)
    // if all nodes run on GPU, clear enable_node_level_gpu_flush
    bool cpuTargetBufferNodesExists = false;
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_CPU && !node->akernel->opencl_buffer_access_enable)
            cpuTargetBufferNodesExists = true;
    }
    if(!cpuTargetBufferNodesExists) {
        graph->enable_node_level_gpu_flush = false;
    }
#endif

    return status;
}

vx_status agoPrepareImageValidRectangleBuffers(AgoGraph * graph)
{
    vx_status status = VX_SUCCESS;

    ////////////////////////////////////////////////
    // prepare for image valid rectangle computation
    ////////////////////////////////////////////////
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        vx_uint32 valid_rect_num_inputs = 0, valid_rect_num_outputs = 0;
        for (vx_uint32 i = 0; i < node->paramCount; i++) {
            AgoParameter * param = &node->parameters[i];
            AgoData * data = node->paramList[i];
            if (data) {
                if (data->ref.type == VX_TYPE_IMAGE) {
                    if (param->direction == VX_INPUT)
                        valid_rect_num_inputs++;
                    if (param->direction == VX_OUTPUT)
                        valid_rect_num_outputs++;
                }
                else if (data->ref.type == VX_TYPE_PYRAMID) {
                    if (param->direction == VX_INPUT)
                        valid_rect_num_inputs += (vx_uint32)data->u.pyr.levels;
                    if (param->direction == VX_OUTPUT)
                        valid_rect_num_outputs += (vx_uint32)data->u.pyr.levels;
                }
            }
        }
        node->valid_rect_num_inputs = valid_rect_num_inputs;
        node->valid_rect_num_outputs = valid_rect_num_outputs;
        if (node->akernel->func && ((node->akernel->flags & AGO_KERNEL_FLAG_GROUP_MASK) == AGO_KERNEL_FLAG_GROUP_AMDLL ||
            (node->akernel->flags & AGO_KERNEL_FLAG_GROUP_MASK) == AGO_KERNEL_FLAG_GROUP_OVX10))
        {
            // nothing to do for built-in kernels
        }
        else
        {
            if (node->valid_rect_inputs)
                delete[] node->valid_rect_inputs;
            if (node->valid_rect_outputs)
                delete[] node->valid_rect_outputs;
            node->valid_rect_inputs = nullptr;
            node->valid_rect_outputs = nullptr;
            if (valid_rect_num_outputs > 0) {
                // allocate valid_rect_outputs[] and valid_rect_inputs[]
                node->valid_rect_outputs = new vx_rectangle_t *[valid_rect_num_outputs]();
                if (!node->valid_rect_outputs) {
                    status = VX_ERROR_NO_MEMORY;
                    break;
                }
                if (valid_rect_num_inputs > 0) {
                    node->valid_rect_inputs = new vx_rectangle_t *[valid_rect_num_inputs]();
                    if (!node->valid_rect_inputs) {
                        status = VX_ERROR_NO_MEMORY;
                        break;
                    }
                    // prepare valid_rect_inputs[] with valid pointers
                    vx_uint32 index = 0;
                    for (vx_uint32 i = 0; i < node->paramCount; i++) {
                        AgoParameter * param = &node->parameters[i];
                        AgoData * data = node->paramList[i];
                        if (data && param->direction == VX_INPUT) {
                            if (data->ref.type == VX_TYPE_IMAGE) {
                                node->valid_rect_inputs[index++] = &data->u.img.rect_valid;
                            }
                            else if (data->ref.type == VX_TYPE_PYRAMID) {
                                for (vx_size level = 0; level < data->u.pyr.levels; level++) {
                                    node->valid_rect_inputs[index++] = &data->children[level]->u.img.rect_valid;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return status;
}

vx_status agoComputeImageValidRectangleOutputs(AgoGraph * graph)
{
    vx_status status = VX_SUCCESS;

    // compute output image valid rectangles
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (node->akernel->func && ((node->akernel->flags & AGO_KERNEL_FLAG_GROUP_MASK) == AGO_KERNEL_FLAG_GROUP_AMDLL ||
            (node->akernel->flags & AGO_KERNEL_FLAG_GROUP_MASK) == AGO_KERNEL_FLAG_GROUP_OVX10))
        {
            status = node->akernel->func(node, ago_kernel_cmd_valid_rect_callback);
            if (status == AGO_ERROR_KERNEL_NOT_IMPLEMENTED) {
                // consider unimplemented cases as success (nothing needed)
                status = VX_SUCCESS;
            }
        }
        else if (node->valid_rect_outputs) {
            // use node->valid_rect_outputs[] to propagate valid rectangles
            for (vx_uint32 i = 0; i < node->paramCount; i++) {
                AgoParameter * param = &node->parameters[i];
                AgoData * data = node->paramList[i];
                if (data && param->direction == VX_OUTPUT) {
                    if (data->ref.type == VX_TYPE_IMAGE) {
                        if (node->metaList[i].set_valid_rectangle_callback) {
                            node->valid_rect_outputs[0] = &data->u.img.rect_valid;
                            status = node->metaList[i].set_valid_rectangle_callback(node, i, node->valid_rect_inputs, node->valid_rect_outputs);
                        }
                        else if (node->valid_rect_reset) {
                            data->u.img.rect_valid.start_x = 0;
                            data->u.img.rect_valid.start_y = 0;
                            data->u.img.rect_valid.end_x = data->u.img.width;
                            data->u.img.rect_valid.end_y = data->u.img.height;
                        }
                    }
                    else if (data->ref.type == VX_TYPE_PYRAMID) {
                        if (node->metaList[i].set_valid_rectangle_callback) {
                            for (vx_size level = 0; level < data->u.pyr.levels; level++) {
                                node->valid_rect_outputs[level] = &data->children[level]->u.img.rect_valid;
                            }
                            status = node->metaList[i].set_valid_rectangle_callback(node, i, node->valid_rect_inputs, node->valid_rect_outputs);
                        }
                        else if (node->valid_rect_reset) {
                            for (vx_size level = 0; level < data->u.pyr.levels; level++) {
                                data->children[level]->u.img.rect_valid.start_x = 0;
                                data->children[level]->u.img.rect_valid.start_y = 0;
                                data->children[level]->u.img.rect_valid.end_x = data->children[level]->u.img.width;
                                data->children[level]->u.img.rect_valid.end_y = data->children[level]->u.img.height;
                            }
                        }
                    }
                }
            }
        }
#if 0 // TBD remove -- dump valid rectangle info
        for (vx_uint32 i = 0; i < node->paramCount; i++) {
            AgoParameter * param = &node->parameters[i];
            AgoData * data = node->paramList[i];
            if (data && param->direction == VX_OUTPUT) {
                if (data->ref.type == VX_TYPE_IMAGE) {
                    printf("valid_rect [ %5d %5d %5d %5d ] image %s\n", data->u.img.rect_valid.start_x, data->u.img.rect_valid.start_y, data->u.img.rect_valid.end_x, data->u.img.rect_valid.end_y, data->name.c_str());
                }
                else if (data->ref.type == VX_TYPE_PYRAMID) {
                    for (vx_size level = 0; level < data->u.pyr.levels; level++) {
                        printf("valid_rect [ %5d %5d %5d %5d ] pyrL%d %s\n", data->children[level]->u.img.rect_valid.start_x, data->children[level]->u.img.rect_valid.start_y, data->children[level]->u.img.rect_valid.end_x, data->children[level]->u.img.rect_valid.end_y, (int)level, data->name.c_str());
                    }
                }
            }
        }
#endif
    }

    return status;
}

int agoInitializeGraph(AgoGraph * graph)
{
    for (AgoNode * node = graph->nodeList.head; node; node = node->next)
    {
        AgoKernel * kernel = node->akernel;
        vx_status status = VX_SUCCESS;
        // handle reverification path
        vx_bool first_time_verify = ((graph->verified == vx_false_e) && (graph->reverify == vx_false_e)) ? vx_true_e : vx_false_e;
        if (kernel->user_kernel) {
            if (!first_time_verify) { //re-verify
                if(kernel->deinitialize_f) {
                    if(node->local_data_set_by_implementation == vx_false_e) {
                        node->local_data_change_is_enabled = vx_true_e;
                    }
                    status = kernel->deinitialize_f(node, (vx_reference *)node->paramList, node->paramCount);
                    node->local_data_change_is_enabled = vx_false_e;
                    if (status != VX_SUCCESS) {
                        graph->reverify = vx_false_e;
                        graph->verified = vx_true_e;
                        graph->state = VX_GRAPH_STATE_VERIFIED;
                    }
                    else {
                        graph->reverify = vx_true_e;
                        graph->verified = vx_true_e;
                        graph->state = VX_GRAPH_STATE_VERIFIED;
                    }
                }
                if (node->localDataSize == 0) {
                    if(node->localDataPtr) {
                        if(!first_time_verify && node->localDataPtr)
                            free(node->localDataPtr);
                        node->localDataSize = 0;
                        node->localDataPtr = nullptr;
                    }
                }
                node->local_data_set_by_implementation = vx_false_e;
            }
        }
        if (kernel->func) {
            status = kernel->func(node, ago_kernel_cmd_initialize);
        }
        else if (kernel->initialize_f) {
            if((kernel->user_kernel == vx_true_e) && (node->localDataSize == 0)) {
                node->local_data_change_is_enabled = vx_true_e;
            }
            status = kernel->initialize_f(node, (vx_reference *)node->paramList, node->paramCount);
            node->local_data_change_is_enabled = vx_false_e;
        }
        if (status) {
            return status;
        }
        else {
            if (node->localDataSize > 0 && node->localDataPtr == nullptr) {
                if (node->localDataPtr_allocated)
                    delete[] node->localDataPtr_allocated;
                node->localDataPtr = node->localDataPtr_allocated = (vx_uint8 *)agoAllocMemory(node->localDataSize);
                if (!node->localDataPtr) {
                    return VX_ERROR_NO_MEMORY;
                }
                memset(node->localDataPtr, 0, node->localDataSize);
                if(kernel->user_kernel == vx_true_e)
                    node->local_data_set_by_implementation = vx_true_e;
            }
            node->initialized = true;
            // keep a copy of paramList into paramListForAgeDelay
            memcpy(node->paramListForAgeDelay, node->paramList, sizeof(node->paramListForAgeDelay));
        }
    }
    return VX_SUCCESS;
}

#if ENABLE_OPENCL
static int agoWaitForNodesCompletion(AgoGraph * graph)
{
    int status = VX_SUCCESS;
    if (!graph->gpu_nodeListQueued.empty()) {
        for (vx_size i = 0; i < graph->gpu_nodeListQueued.size(); i++) {
            AgoNode * node = graph->gpu_nodeListQueued[i];
            if (node->supernode) {
                if (!node->supernode->launched || agoGpuOclSuperNodeWait(graph, node->supernode) < 0) {
                    agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: agoWaitForNodesCompletion: launched=%d supernode wait failed\n", node->supernode->launched);
                    return VX_FAILURE;
                }
                agoPerfCaptureStop(&node->perf);
                for (size_t index = 0; index < node->supernode->nodeList.size(); index++) {
                    AgoNode * anode = node->supernode->nodeList[index];
                    // node callback
                    if (anode->callback) {
                        vx_action action = anode->callback(anode);
                        if (action == VX_ACTION_ABANDON) {
                            graph->state = VX_GRAPH_STATE_ABANDONED;
                            status = VX_ERROR_GRAPH_ABANDONED;
                            break;
                        }
                    }
                }
            }
            else {
                if (agoGpuOclSingleNodeWait(graph, node) < 0) {
                    agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: agoWaitForNodesCompletion: single node wait failed\n");
                    return VX_FAILURE;
                }
                agoPerfCaptureStop(&node->perf);
                // node callback
                if (node->callback) {
                    vx_action action = node->callback(node);
                    if (action == VX_ACTION_ABANDON) {
                        graph->state = VX_GRAPH_STATE_ABANDONED;
                        status = VX_ERROR_GRAPH_ABANDONED;
                        break;
                    }
                }
            }
        }
        graph->gpu_nodeListQueued.clear();
    }
    return status;
}

static int agoDataSyncFromGpuToCpu(AgoGraph * graph, AgoNode * node, AgoData * dataToSync)
{
    cl_command_queue opencl_cmdq = graph->opencl_cmdq ? graph->opencl_cmdq : graph->ref.context->opencl_cmdq;

    if (dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
        if (node->flags & AGO_KERNEL_FLAG_DEVICE_GPU) {
            if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
                int64_t stime = agoGetClockCounter();
                vx_size size = dataToSync->size;
                if (dataToSync->ref.type == VX_TYPE_LUT) {
                    cl_int err = clEnqueueWriteBuffer(opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, size, dataToSync->buffer, 0, NULL, NULL);
                    if (err) {
                        agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: clEnqueueWriteBuffer(lut) => %d\n", err);
                        return -1;
                    }
                }
                else {
                    vx_size size = dataToSync->size;
                    if (dataToSync->ref.type == VX_TYPE_ARRAY) {
                        // transfer only valid data
                        size = dataToSync->u.arr.itemsize * dataToSync->u.arr.numitems;
                    }
                    if (size > 0) {
                        cl_int err = clEnqueueWriteBuffer(opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, size, dataToSync->buffer, 0, NULL, NULL);
                        if (err) {
                            agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: clEnqueueWriteBuffer() => %d\n", err);
                            return -1;
                        }
                    }
                }
                dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                int64_t etime = agoGetClockCounter();
                graph->gpu_perf.buffer_write += etime - stime;
            }
        }
        else {
            if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                if (dataToSync->ref.type == VX_TYPE_LUT) {
                    int64_t stime = agoGetClockCounter();
                    vx_size size = dataToSync->size;
                    cl_int err = clEnqueueReadBuffer(opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, size, dataToSync->buffer, 0, NULL, NULL);
                    if (err) {
                        agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: clEnqueueReadBuffer(lut) => %d\n", err);
                        return -1;
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    int64_t etime = agoGetClockCounter();
                    graph->gpu_perf.buffer_read += etime - stime;
                }
                else {
                    vx_size size = dataToSync->size;
                    if (dataToSync->ref.type == VX_TYPE_ARRAY) {
                        // transfer only region that has valid data
                        size = dataToSync->u.arr.numitems * dataToSync->u.arr.itemsize;
                    }
                    else if (node->akernel->opencl_buffer_access_enable) {
                        // no need to transfer to CPU for this node
                        size = 0;
                    }
                    if (size > 0) {
                        int64_t stime = agoGetClockCounter();
                        if (size > 0) {
                            cl_int err = clEnqueueReadBuffer(opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, size, dataToSync->buffer, 0, NULL, NULL);
                            if (err) {
                                agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: clEnqueueReadBuffer(0x%x,%s,%ld,%ld) => %d\n", dataToSync->ref.type, dataToSync->name.c_str(), dataToSync->gpu_buffer_offset, size, err);
                                return -1;
                            }
                        }
                        dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        int64_t etime = agoGetClockCounter();
                        graph->gpu_perf.buffer_read += etime - stime;
                    }
                }
            }
        }
    }
    return 0;
}
#elif ENABLE_HIP
static int agoWaitForNodesCompletion(AgoGraph * graph)
{
    int status = VX_SUCCESS;
    if (!graph->gpu_nodeListQueued.empty()) {
        for (vx_size i = 0; i < graph->gpu_nodeListQueued.size(); i++) {
            AgoNode * node = graph->gpu_nodeListQueued[i];
            if (node->supernode) {
                if (!node->supernode->launched || agoGpuHipSuperNodeWait(graph, node->supernode) < 0) {
                    agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: agoWaitForNodesCompletion: launched=%d supernode wait failed\n", node->supernode->launched);
                    return VX_FAILURE;
                }
                agoPerfCaptureStop(&node->perf);
                for (size_t index = 0; index < node->supernode->nodeList.size(); index++) {
                    AgoNode * anode = node->supernode->nodeList[index];
                    // node callback
                    if (anode->callback) {
                        vx_action action = anode->callback(anode);
                        if (action == VX_ACTION_ABANDON) {
                            status = VX_ERROR_GRAPH_ABANDONED;
                            break;
                        }
                    }
                }
            }
            else {
                if (agoGpuHipSingleNodeWait(graph, node) < 0) {
                    agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: agoWaitForNodesCompletion: single node wait failed\n");
                    return VX_FAILURE;
                }
                agoPerfCaptureStop(&node->perf);
                // node callback
                if (node->callback) {
                    vx_action action = node->callback(node);
                    if (action == VX_ACTION_ABANDON) {
                        graph->state = VX_GRAPH_STATE_ABANDONED;
                        status = VX_ERROR_GRAPH_ABANDONED;
                        break;
                    }
                }
            }
        }
        graph->gpu_nodeListQueued.clear();
    }
    return status;
}

static int agoDataSyncFromGpuToCpu(AgoGraph * graph, AgoNode * node, AgoData * dataToSync)
{
    if (dataToSync->hip_memory && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
        if (node->flags & AGO_KERNEL_FLAG_DEVICE_GPU) {
            if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
                int64_t stime = agoGetClockCounter();
                vx_size size = dataToSync->size;
                if (dataToSync->ref.type == VX_TYPE_LUT) {
                    hipError_t err = hipMemcpyHtoD(dataToSync->hip_memory+dataToSync->gpu_buffer_offset, dataToSync->buffer, size);
                    if (err) {
                        agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: hipMemcpyHtoD(lut) => %d\n", err);
                        return -1;
                    }
                }
                else {
                    if (size > 0) {
                        hipError_t err = hipMemcpyHtoD(dataToSync->hip_memory+dataToSync->gpu_buffer_offset, dataToSync->buffer, size);
                        if (err) {
                            agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: hipMemcpyHtoD() => %d\n", err);
                            return -1;
                        }
                    }
                }
                dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                int64_t etime = agoGetClockCounter();
                graph->gpu_perf.buffer_write += etime - stime;
            }
        }
        else {
            if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL)) {
                if (dataToSync->ref.type == VX_TYPE_LUT) {
                    int64_t stime = agoGetClockCounter();
                    vx_size size = dataToSync->size;
                    hipError_t err = hipMemcpyDtoH(dataToSync->buffer, dataToSync->hip_memory+dataToSync->gpu_buffer_offset, size);
                    if (err) {
                        agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: hipMemcpyDtoH(DtoH_lut) => %d\n", err);
                        return -1;
                    }
                    dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                    int64_t etime = agoGetClockCounter();
                    graph->gpu_perf.buffer_read += etime - stime;
                }
                else {
                    vx_size size = dataToSync->size;
                    if (dataToSync->ref.type == VX_TYPE_ARRAY) {
                        // transfer only region that has valid data
                        size = dataToSync->u.arr.numitems * dataToSync->u.arr.itemsize;
                    }
                    else if (node->akernel->opencl_buffer_access_enable) {
                        // no need to transfer to CPU for this node
                        size = 0;
                    }
                    if (size > 0) {
                        int64_t stime = agoGetClockCounter();
                        if (size > 0) {
                            hipError_t err = hipMemcpyDtoH(dataToSync->buffer, dataToSync->hip_memory+dataToSync->gpu_buffer_offset, size);
                            if (err) {
                                agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: hipMemcpyDtoH(0x%x,%s,%ld,%ld) => %d\n", dataToSync->ref.type, dataToSync->name.c_str(), dataToSync->gpu_buffer_offset, size, err);
                                return -1;
                            }
                        }
                        dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                        int64_t etime = agoGetClockCounter();
                        graph->gpu_perf.buffer_read += etime - stime;
                    }
                }
            }
        }
    }
    return 0;
}
#endif

int agoUpdateDelaySlots(AgoNode * node)
{
    vx_graph graph = (vx_graph)node->ref.scope;
    for (vx_uint32 arg = 0; arg < node->paramCount; arg++) {
        AgoData * data = node->paramList[arg];
        if (data && agoIsPartOfDelay(data)) {
            // get the trace to delay object from original node parameter without vxAgeDelay changes
            int siblingTrace[AGO_MAX_DEPTH_FROM_DELAY_OBJECT], siblingTraceCount = 0;
            AgoData * delay = agoGetSiblingTraceToDelayForUpdate(node->paramListForAgeDelay[arg], siblingTrace, siblingTraceCount);
            if (delay) {
                // get the data
                data = agoGetDataFromTrace(delay, siblingTrace, siblingTraceCount);
                if (data) {
                    // update the node parameter
                    node->paramList[arg] = data;
                }
                else {
                    agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: SiblingTrace#1 missing\n");
                    return VX_FAILURE;
                }
            }
            else {
                agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: SiblingTrace#2 missing\n");
                return VX_FAILURE;
            }
        }
    }
    return 0;
}

int agoExecuteGraph(AgoGraph * graph)
{
    if (graph->detectedInvalidNode) {
        agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: agoExecuteGraph: detected invalid node\n");
        return VX_FAILURE;
    }
    else if (!graph->nodeList.head)
        return VX_SUCCESS;
    int status = VX_SUCCESS;
    graph->state = VX_GRAPH_STATE_RUNNING;
    agoPerfProfileEntry(graph, ago_profile_type_exec_begin, &graph->ref);
    agoPerfCaptureStart(&graph->perf);

    // update delay slots
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        status = agoUpdateDelaySlots(node);
        if (status != VX_SUCCESS)
            return status;
    }
#if (ENABLE_OPENCL||ENABLE_HIP)
    for (AgoSuperNode * supernode = graph->supernodeList; supernode; supernode = supernode->next) {
        for (size_t arg = 0; arg < supernode->dataList.size(); arg++) {
            AgoData * data = supernode->dataList[arg];
            if (data && agoIsPartOfDelay(data)) {
                // get the trace to delay object from original node parameter without vxAgeDelay changes
                int siblingTrace[AGO_MAX_DEPTH_FROM_DELAY_OBJECT], siblingTraceCount = 0;
                AgoData * delay = agoGetSiblingTraceToDelayForUpdate(supernode->dataListForAgeDelay[arg], siblingTrace, siblingTraceCount);
                if (delay) {
                    // get the data
                    data = agoGetDataFromTrace(delay, siblingTrace, siblingTraceCount);
                    if (data) {
                        // update the supernode parameter
                        supernode->dataList[arg] = data;
                    }
                    else {
                        agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: SiblingTrace#3 missing\n");
                        return VX_FAILURE;
                    }
                }
                else {
                    agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: SiblingTrace#4 missing\n");
                    return VX_FAILURE;
                }
            }
        }
    }
#endif

#if ENABLE_OPENCL
    // clear opencl_buffer for all virtual images with enableUserBufferGPU == true
    for (AgoData * data = graph->dataList.head; data; data = data->next) {
        if (data->ref.type == VX_TYPE_IMAGE && data->u.img.enableUserBufferGPU) {
            data->opencl_buffer = nullptr;
        }
    }
#elif ENABLE_HIP
    for (AgoData * data = graph->dataList.head; data; data = data->next) {
        if (data->ref.type == VX_TYPE_IMAGE && data->u.img.enableUserBufferGPU) {
            data->hip_memory = nullptr;
        }
    }
#endif
    // mark that none of the supernode has been launched
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (node->supernode) {
            node->supernode->launched = false;
        }
    }
#if (ENABLE_OPENCL||ENABLE_HIP)
    graph->gpu_nodeListQueued.clear();
    vx_uint32 nodeLaunchHierarchicalLevel = 0;
    memset(&graph->gpu_perf, 0, sizeof(graph->gpu_perf));
#endif
    // execute one nodes in one hierarchical level at a time
    bool opencl_buffer_access_enable = false;
    for (auto enode = graph->nodeList.head; enode;) {
        // get snode..enode with next hierarchical_level
        auto hierarchical_level = enode->hierarchical_level;
        auto snode = enode; enode = enode->next;
        while (enode && enode->hierarchical_level == hierarchical_level)
            enode = enode->next;
#if ENABLE_OPENCL
        // process GPU nodes at current hierarchical level
        for (auto node = snode; node != enode; node = node->next) {
            if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU) {
                bool launched = true;
                agoPerfProfileEntry(graph, ago_profile_type_launch_begin, &node->ref);
                agoPerfCaptureStart(&node->perf);
                // make sure that all input buffers are synched
                for (vx_uint32 i = 0; i < node->paramCount; i++) {
                    AgoData * data = node->paramList[i];
                    if (data &&
                        (node->parameters[i].direction == VX_INPUT || node->parameters[i].direction == (vx_direction_e)VX_BIDIRECTIONAL))
                    {
                        auto dataToSync = (data->ref.type == VX_TYPE_IMAGE && data->u.img.isROI) ? data->u.img.roiMasterImage : data;
                        if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT) &&
                            dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED))
                        {
                            status = agoDirective((vx_reference)dataToSync, VX_DIRECTIVE_AMD_COPY_TO_OPENCL);
                            if(status != VX_SUCCESS) {
                                agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: agoDirective(*,VX_DIRECTIVE_AMD_COPY_TO_OPENCL) failed (%d:%s)\n", status, agoEnum2Name(status));
                                return status;
                            }
                        }
                    }
                }
                if (!node->supernode) {
                    // launch the single node
                    if (agoGpuOclSingleNodeLaunch(graph, node) < 0) {
                        return -1;
                    }
                }
                else if (!node->supernode->launched) {
                    // launch the super node
                    if (agoGpuOclSuperNodeLaunch(graph, node->supernode) < 0) {
                        return -1;
                    }
                    node->supernode->launched = true;
                }
                else {
                    launched = false;
                }
                if (launched) {
                    graph->gpu_nodeListQueued.push_back(node);
                    if (nodeLaunchHierarchicalLevel == 0) {
                        nodeLaunchHierarchicalLevel = node->hierarchical_level;
                    }
                }
                agoPerfProfileEntry(graph, ago_profile_type_launch_end, &node->ref);
            }
        }
#elif ENABLE_HIP
        // process GPU nodes at current hierarchical level
        for (auto node = snode; node != enode; node = node->next) {
            if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU) {
                bool launched = true;
                node->hip_stream0 = graph->hip_stream0;
                agoPerfProfileEntry(graph, ago_profile_type_launch_begin, &node->ref);
                agoPerfCaptureStart(&node->perf);
                // make sure that all input buffers are synched
                for (vx_uint32 i = 0; i < node->paramCount; i++) {
                    AgoData * data = node->paramList[i];
                    if (data &&
                        (node->parameters[i].direction == VX_INPUT || node->parameters[i].direction == (vx_direction_e)VX_BIDIRECTIONAL))
                    {
                        auto dataToSync = (data->ref.type == VX_TYPE_IMAGE && data->u.img.isROI) ? data->u.img.roiMasterImage : data;
                        if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT) &&
                            dataToSync->hip_memory && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED))
                        {
                            status = agoDirective((vx_reference)dataToSync, VX_DIRECTIVE_AMD_COPY_TO_HIPMEM);
                            if(status != VX_SUCCESS) {
                                agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: agoDirective(*,VX_DIRECTIVE_AMD_COPY_TO_HIPMEM) failed (%d:%s)\n", status, agoEnum2Name(status));
                                return status;
                            }
                        }
                    }
                }
                if (!node->supernode) {
                    // launch the single node
                    if (agoGpuHipSingleNodeLaunch(graph, node) < 0) {
                        return -1;
                    }
                }
                else if (!node->supernode->launched) {
                    // launch the super node
                    if (agoGpuHipSuperNodeLaunch(graph, node->supernode) < 0) {
                        return -1;
                    }
                    node->supernode->launched = true;
                }
                else {
                    launched = false;
                }
                if (launched) {
                    graph->gpu_nodeListQueued.push_back(node);
                    if (nodeLaunchHierarchicalLevel == 0) {
                        nodeLaunchHierarchicalLevel = node->hierarchical_level;
                    }
                }
                agoPerfProfileEntry(graph, ago_profile_type_launch_end, &node->ref);
            }
        }
#endif
        // process CPU nodes at current hierarchical level
        for (auto node = snode; node != enode; node = node->next) {
            if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_CPU) {
#if (ENABLE_OPENCL||ENABLE_HIP)
                opencl_buffer_access_enable |= (node->akernel->opencl_buffer_access_enable ? true : false);
                if (!node->akernel->opencl_buffer_access_enable) {
                    agoPerfProfileEntry(graph, ago_profile_type_wait_begin, &node->ref);
                    if (nodeLaunchHierarchicalLevel > 0 && nodeLaunchHierarchicalLevel < node->hierarchical_level) {
                        status = agoWaitForNodesCompletion(graph);
                        if (status != VX_SUCCESS) {
                            agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: agoWaitForNodesCompletion failed (%d:%s)\n", status, agoEnum2Name(status));
                            return status;
                        }
                        nodeLaunchHierarchicalLevel = 0;
                    }
                    if(opencl_buffer_access_enable) {
#if ENABLE_OPENCL
                        cl_int err = clFinish(graph->opencl_cmdq);
                        if (err) {
                            agoAddLogEntry(NULL, VX_FAILURE, "ERROR: clFinish(graph) => %d\n", err);
                            return VX_FAILURE;
                        }
#else
                        hipError_t err = hipStreamSynchronize(graph->hip_stream0);
                        if (err) {
                            agoAddLogEntry(NULL, VX_FAILURE, "ERROR: hipStreamSynchronize(graph) => %d\n", err);
                            return VX_FAILURE;
                        }
#endif
                        opencl_buffer_access_enable = false;
                    }
                    agoPerfProfileEntry(graph, ago_profile_type_wait_end, &node->ref);
                }
                agoPerfProfileEntry(graph, ago_profile_type_copy_begin, &node->ref);
                // make sure that all input buffers are synched
                if (node->akernel->opencl_buffer_access_enable) {
                    for (vx_uint32 i = 0; i < node->paramCount; i++) {
                        AgoData * data = node->paramList[i];
                        if (data &&
                            (node->parameters[i].direction == VX_INPUT || node->parameters[i].direction == (vx_direction_e)VX_BIDIRECTIONAL))
                        {
                            auto dataToSync = (data->ref.type == VX_TYPE_IMAGE && data->u.img.isROI) ? data->u.img.roiMasterImage : data;
                            if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT) &&
#if ENABLE_OPENCL
                            dataToSync->opencl_buffer && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED))
                            {
                                status = agoDirective((vx_reference)dataToSync, VX_DIRECTIVE_AMD_COPY_TO_OPENCL);
                                if(status != VX_SUCCESS) {
                                    agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: agoDirective(*,VX_DIRECTIVE_AMD_COPY_TO_OPENCL) failed (%d:%s)\n", status, agoEnum2Name(status));
                                    return status;
                                }
                            }
#else
                            dataToSync->hip_memory && !(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED))
                            {
                                status = agoDirective((vx_reference)dataToSync, VX_DIRECTIVE_AMD_COPY_TO_HIPMEM);
                                if(status != VX_SUCCESS) {
                                    agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: agoDirective(*,VX_DIRECTIVE_AMD_COPY_TO_HIPMEM) failed (%d:%s)\n", status, agoEnum2Name(status));
                                    return status;
                                }
                            }
#endif
                        }
                    }
                }
                else {
                    for (vx_uint32 i = 0; i < node->paramCount; i++) {
                        AgoData * data = node->paramList[i];
                        if (data && (node->parameters[i].direction == VX_INPUT || node->parameters[i].direction == (vx_direction_e)VX_BIDIRECTIONAL)) {
                            auto dataToSync = (data->ref.type == VX_TYPE_IMAGE && data->u.img.isROI) ? data->u.img.roiMasterImage : data;
                            status = agoDataSyncFromGpuToCpu(graph, node, dataToSync);
                            for (vx_uint32 j = 0; !status && j < dataToSync->numChildren; j++) {
                                AgoData * jdata = dataToSync->children[j];
                                if (jdata)
                                    status = agoDataSyncFromGpuToCpu(graph, node, jdata);
                            }
                            if (status) {
                                agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: agoDataSyncFromGpuToCpu failed (%d:%s) for node(%s) arg#%d data(%s)\n", status, agoEnum2Name(status), node->akernel->name, i, data->name.c_str());
                                return status;
                            }
                        }
                    }
                }
                agoPerfProfileEntry(graph, ago_profile_type_copy_end, &node->ref);
#endif
                // execute node
                agoPerfProfileEntry(graph, ago_profile_type_exec_begin, &node->ref);
                agoPerfCaptureStart(&node->perf);
                AgoKernel * kernel = node->akernel;
                status = VX_SUCCESS;
                if (kernel->func) {
                    status = kernel->func(node, ago_kernel_cmd_execute);
                    if (status == AGO_ERROR_KERNEL_NOT_IMPLEMENTED)
                        status = VX_ERROR_NOT_IMPLEMENTED;
                }
                else if (kernel->kernel_f) {
                    status = kernel->kernel_f(node, (vx_reference *)node->paramList, node->paramCount);
                }
                if (status) {
                    if (status == VX_ERROR_GRAPH_ABANDONED)
                        agoAddLogEntry((vx_reference)graph, VX_FAILURE, "INFO: kernel %s exec returned graph_stopped status: (this could mean EOS for amd_media extension (%d))\n", kernel->name, status);
                    else
                        agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: kernel %s exec failed (%d:%s)\n", kernel->name, status, agoEnum2Name(status));
                    return status;
                }
                agoPerfCaptureStop(&node->perf);
                agoPerfProfileEntry(graph, ago_profile_type_exec_end, &node->ref);
                // mark that node outputs are dirty
                for (vx_uint32 i = 0; i < node->paramCount; i++) {
#if ENABLE_OPENCL
                    AgoData * data = node->paramList[i];
                    if (data && data->opencl_buffer &&
                        (node->parameters[i].direction == VX_OUTPUT || node->parameters[i].direction == (vx_direction_e)VX_BIDIRECTIONAL))
                    {
                        auto dataToSync = (data->ref.type == VX_TYPE_IMAGE && data->u.img.isROI) ? data->u.img.roiMasterImage : data;
                        dataToSync->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                        dataToSync->buffer_sync_flags |=
                            ((node->akernel->opencl_buffer_access_enable || data->u.img.enableUserBufferGPU)
                                ? AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL
                                : AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE);
                    }
#elif ENABLE_HIP
                    AgoData * data = node->paramList[i];
                    if (data && data->hip_memory &&
                            (node->parameters[i].direction == VX_OUTPUT || node->parameters[i].direction == (vx_direction_e)VX_BIDIRECTIONAL))
                    {
                        auto dataToSync = (data->ref.type == VX_TYPE_IMAGE && data->u.img.isROI) ? data->u.img.roiMasterImage : data;
                        dataToSync->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
                        dataToSync->buffer_sync_flags |=
                            ((node->akernel->opencl_buffer_access_enable || data->u.img.enableUserBufferGPU)
                                ? AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL
                                : AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE);
                    }
#endif
                }
                // node callback
                if (node->callback) {
                    vx_action action = node->callback(node);
                    if (action == VX_ACTION_ABANDON) {
                        graph->state = VX_GRAPH_STATE_ABANDONED;
                        return VX_ERROR_GRAPH_ABANDONED;
                    }
                }
            }
        }
    }
#if (ENABLE_OPENCL||ENABLE_HIP)
    agoPerfProfileEntry(graph, ago_profile_type_wait_begin, &graph->ref);
    if (nodeLaunchHierarchicalLevel > 0) {
        status = agoWaitForNodesCompletion(graph);
        if (status != VX_SUCCESS) {
            agoAddLogEntry((vx_reference)graph, VX_FAILURE, "ERROR: agoWaitForNodesCompletion failed (%d:%s)\n", status, agoEnum2Name(status));
            return status;
        }
    }
    if(opencl_buffer_access_enable) {
#if ENABLE_OPENCL
        cl_int err = clFinish(graph->opencl_cmdq);
        if (err) {
            agoAddLogEntry(NULL, VX_FAILURE, "ERROR: clFinish(graph) => %d\n", err);
            return VX_FAILURE;
        }
#else
        hipError_t err = hipStreamSynchronize(graph->hip_stream0);
        if (err) {
            agoAddLogEntry(NULL, VX_FAILURE, "ERROR: hipStreamSynchronize(graph) => %d\n", err);
            return VX_FAILURE;
        }
#endif
    }
    agoPerfProfileEntry(graph, ago_profile_type_wait_end, &graph->ref);
    graph->gpu_perf_total.kernel_enqueue += graph->gpu_perf.kernel_enqueue;
    graph->gpu_perf_total.kernel_wait += graph->gpu_perf.kernel_wait;
    graph->gpu_perf_total.buffer_read += graph->gpu_perf.buffer_read;
    graph->gpu_perf_total.buffer_write += graph->gpu_perf.buffer_write;
#endif

    // auto age delays
    for (auto it = graph->autoAgeDelayList.begin(); it != graph->autoAgeDelayList.end(); it++) {
        if (agoIsValidData(*it, VX_TYPE_DELAY)) {
            agoAgeDelay(*it);
        }
    }

    agoPerfCaptureStop(&graph->perf);
    agoPerfProfileEntry(graph, ago_profile_type_exec_end, &graph->ref);
    graph->execFrameCount++;

    if (status == VX_SUCCESS)
        graph->state = VX_GRAPH_STATE_COMPLETED;

    return status;
}

int agoAgeDelay(AgoData * delay)
{
    // cycle through all the pointers by swapping
    AgoData * childLast = delay->children[delay->u.delay.count - 1];
    for (vx_int32 i = (vx_int32)delay->u.delay.count - 1; i > 0; i--) {
        delay->children[i] = delay->children[i - 1];
    }
    delay->children[0] = childLast;
    delay->u.delay.age++;
    return VX_SUCCESS;
}

vx_status agoDirective(vx_reference reference, vx_enum directive)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidReference(reference)) {
        vx_context context = reference->context;
        if (agoIsValidContext(context)) {
            CAgoLock lock(context->cs);
            status = VX_SUCCESS;
            switch (directive)
            {
            case VX_DIRECTIVE_ENABLE_LOGGING:
                reference->enable_logging = true;
                break;
            case VX_DIRECTIVE_DISABLE_LOGGING:
                reference->enable_logging = false;
                break;
            case VX_DIRECTIVE_ENABLE_PERFORMANCE:
                if (context) reference->enable_perf = true;
                else
                    status = VX_ERROR_NOT_SUPPORTED;
                break;
            case VX_DIRECTIVE_DISABLE_PERFORMANCE:
                if (context) reference->enable_perf = false;
                else
                    status = VX_ERROR_NOT_SUPPORTED;
                break;
            case VX_DIRECTIVE_AMD_READ_ONLY:
                if (reference->type == VX_TYPE_CONVOLUTION || reference->type == VX_TYPE_MATRIX) {
                    if (((AgoData *)reference)->buffer) {
                        reference->read_only = true;
                    }
                    else {
                        status = VX_ERROR_NOT_SUPPORTED;
                    }
                }
                else {
                    status = VX_ERROR_NOT_SUPPORTED;
                }
                break;
#if ENABLE_OPENCL
            case VX_DIRECTIVE_AMD_COPY_TO_OPENCL:
                status = VX_ERROR_NOT_SUPPORTED;
                if (reference->context->opencl_cmdq) {
                    auto data = (AgoData *)reference;
                    auto dataToSync = (data->ref.type == VX_TYPE_IMAGE && data->u.img.isROI) ? data->u.img.roiMasterImage : data;
                    if (dataToSync->ref.type == VX_TYPE_LUT) {
                        if (dataToSync->opencl_buffer) {
                            cl_int err = clEnqueueWriteBuffer(dataToSync->ref.context->opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, dataToSync->size, dataToSync->buffer, 0, NULL, NULL);
                            if (err) {
                                agoAddLogEntry(NULL, VX_FAILURE, "ERROR: clEnqueueWriteImage(lut) => %d\n", err);
                                return VX_FAILURE;
                            }
                            dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                            status = VX_SUCCESS;
                        }
                        else {
                            status = VX_ERROR_NOT_ALLOCATED;
                        }
                    }
                    else if (dataToSync->ref.type == VX_TYPE_IMAGE && dataToSync->numChildren > 0) {
                        status = VX_ERROR_NOT_ALLOCATED;
                        for (vx_uint32 plane = 0; plane < dataToSync->numChildren; plane++) {
                            AgoData * img = dataToSync->children[plane];
                            if (img) {
                                if (img->opencl_buffer) {
                                    cl_int err = clEnqueueWriteBuffer(img->ref.context->opencl_cmdq, img->opencl_buffer, CL_TRUE, img->gpu_buffer_offset, img->size, img->buffer, 0, NULL, NULL);
                                    if (err) {
                                        agoAddLogEntry(NULL, VX_FAILURE, "ERROR: clEnqueueWriteBuffer() => %d\n", err);
                                        return VX_FAILURE;
                                    }
                                    img->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                                    status = VX_SUCCESS;
                                }
                            }
                        }
                    }
                    else {
                        if (dataToSync->opencl_buffer) {
                            vx_size size = dataToSync->size;
                            if (dataToSync->ref.type == VX_TYPE_ARRAY) {
                                // transfer only valid data
                                size = dataToSync->u.arr.itemsize * dataToSync->u.arr.numitems;
                            }
                            if (size > 0) {
                                cl_int err = clEnqueueWriteBuffer(dataToSync->ref.context->opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->gpu_buffer_offset, size, dataToSync->buffer, 0, NULL, NULL);
                                if (err) {
                                    agoAddLogEntry(NULL, VX_FAILURE, "ERROR: clEnqueueWriteBuffer() => %d\n", err);
                                    return VX_FAILURE;
                                }
                            }
                            dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                            status = VX_SUCCESS;
                        }
                        else {
                            status = VX_ERROR_NOT_ALLOCATED;
                        }
                    }
                }
                break;
#elif ENABLE_HIP
            case VX_DIRECTIVE_AMD_COPY_TO_HIPMEM:
                {
                  status = VX_ERROR_NOT_SUPPORTED;
                  auto data = (AgoData *)reference;
                  auto dataToSync = (data->ref.type == VX_TYPE_IMAGE && data->u.img.isROI) ? data->u.img.roiMasterImage : data;
                  if (dataToSync->ref.type == VX_TYPE_LUT) {
                      if (dataToSync->hip_memory) {
                          hipError_t err = hipMemcpyHtoD(dataToSync->hip_memory + dataToSync->gpu_buffer_offset, dataToSync->buffer, dataToSync->size);
                          if (err) {
                              agoAddLogEntry(NULL, VX_FAILURE, "ERROR: hipMemcpyHtoD failed => %d\n", err);
                              return VX_FAILURE;
                          }
                          dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                          status = VX_SUCCESS;
                      }
                      else {
                          status = VX_ERROR_NOT_ALLOCATED;
                      }
                  } else if (dataToSync->ref.type == VX_TYPE_IMAGE && dataToSync->numChildren > 0) {
                      status = VX_ERROR_NOT_ALLOCATED;
                      for (vx_uint32 plane = 0; plane < dataToSync->numChildren; plane++) {
                          AgoData * img = dataToSync->children[plane];
                          if (img) {
                              if (img->hip_memory) {
                                  hipError_t err = hipMemcpyHtoD(img->hip_memory + img->gpu_buffer_offset, img->buffer, img->size);
                                  if (err) {
                                      agoAddLogEntry(NULL, VX_FAILURE, "ERROR: hipMemcpyHtoD failed => %d\n", err);
                                      return VX_FAILURE;
                                  }
                                  img->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                                  status = VX_SUCCESS;
                              }
                          }
                      }
                  } else if (dataToSync->ref.type == VX_TYPE_CONVOLUTION) {
                      if (dataToSync->hip_memory && data->size >0) {
                          hipError_t err = hipMemcpyHtoD(dataToSync->hip_memory + dataToSync->gpu_buffer_offset, dataToSync->reserved, data->size << 1);
                          if (err) {
                              agoAddLogEntry(NULL, VX_FAILURE, "ERROR: hipMemcpyHtoD failed => %d\n", err);
                              return VX_FAILURE;
                          }
                          dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                          status = VX_SUCCESS;
                      }
                  } else {
                      if (dataToSync->hip_memory) {
                          vx_size size = dataToSync->size;
                          if (dataToSync->ref.type == VX_TYPE_ARRAY) {
                              // transfer only valid data
                              size = dataToSync->u.arr.itemsize * dataToSync->u.arr.numitems;
                          }
                          if (size > 0) {
                              hipError_t err = hipMemcpyHtoD(dataToSync->hip_memory + dataToSync->gpu_buffer_offset, dataToSync->buffer, dataToSync->size);
                              if (err) {
                                  agoAddLogEntry(NULL, VX_FAILURE, "ERROR: hipMemcpyHtoD failed => %d\n", err);
                                  return VX_FAILURE;
                              }
                          }
                          dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
                          status = VX_SUCCESS;
                      }
                      else {
                          status = VX_ERROR_NOT_ALLOCATED;
                      }
                  }
                }
            break;
#endif
            case VX_DIRECTIVE_AMD_ENABLE_PROFILE_CAPTURE:
            case VX_DIRECTIVE_AMD_DISABLE_PROFILE_CAPTURE:
                if (reference->type == VX_TYPE_GRAPH) {
                    ((AgoGraph *)reference)->enable_performance_profiling =
                        (directive == VX_DIRECTIVE_AMD_ENABLE_PROFILE_CAPTURE) ? true : false;
                }
                else {
                    status = VX_ERROR_NOT_SUPPORTED;
                }
                break;
#if ENABLE_OPENCL
            case VX_DIRECTIVE_AMD_DISABLE_GPU_FLUSH:
                if (reference->type == VX_TYPE_GRAPH) {
                    ((AgoGraph *)reference)->enable_node_level_gpu_flush = false;
                }
                else {
                    status = VX_ERROR_NOT_SUPPORTED;
                }
                break;
#endif
            default:
                status = VX_ERROR_NOT_SUPPORTED;
                break;
            }
        }
    }
    return status;
}

vx_status agoGraphDumpPerformanceProfile(AgoGraph * graph, const char * fileName)
{
    bool use_stdout = true;
    FILE * fp = stdout;
    if (fileName && strcmp(fileName, "stdout") != 0) {
        use_stdout = false;
        fp = fopen(fileName, "w");
        if (!fp) {
            agoAddLogEntry(NULL, VX_FAILURE, "ERROR: unable to create: %s\n", fileName);
            return VX_FAILURE;
        }
    }
    fprintf(fp, " COUNT,tmp(ms),avg(ms),min(ms),max(ms),DEV,KERNEL\n");
    int64_t freq = agoGetClockFrequency();
    float factor = 1000.0f / (float)freq; // to convert clock counter to ms
    if (graph->perf.num > 0) {
        fprintf(fp, "%6d,%7.3f,%7.3f,%7.3f,%7.3f,%s,%s\n",
            (int)graph->perf.num, (float)graph->perf.tmp * factor,
            (float)graph->perf.sum * factor / (float)graph->perf.num,
            (float)graph->perf.min * factor, (float)graph->perf.max * factor,
            graph->attr_affinity.device_type == AGO_TARGET_AFFINITY_GPU ? "GPU" : "CPU",
            "GRAPH");
    }
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (node->perf.num > 0) {
            fprintf(fp, "%6d,%7.3f,%7.3f,%7.3f,%7.3f,%s,%s\n",
                (int)node->perf.num, (float)node->perf.tmp * factor,
                (float)node->perf.sum * factor / (float)node->perf.num,
                (float)node->perf.min * factor, (float)node->perf.max * factor,
                node->attr_affinity.device_type == AGO_TARGET_AFFINITY_GPU ? "GPU" : "CPU",
                node->akernel->name);
        }
    }
    if (graph->enable_performance_profiling && graph->performance_profile.size() > 0) {
        fprintf(fp, "***PROFILER-OUTPUT***\n");
        fprintf(fp, " frame,type,timestamp(ms),object-name\n");
        int64_t stime = graph->performance_profile[0].time;
        for (auto entry : graph->performance_profile) {
            char name[1024];
            if (entry.ref->type == VX_TYPE_GRAPH) strcpy(name, "GRAPH");
            else if (entry.ref->type == VX_TYPE_NODE) strncpy(name, ((AgoNode *)entry.ref)->akernel->name, sizeof(name) - 1);
            else agoGetDataName(name, (AgoData *)entry.ref);
            static const char * type_str[] = {
                "launch(s)", "launch(e)", "wait(s)", "wait(e)", "copy(s)", "copy(e)", "exec(s)", "exec(e)",
                "8", "9", "10", "11", "12", "13", "14", "15"
            };
            fprintf(fp, "%6d,%-9.9s,%13.3f,%s\n", entry.id, type_str[entry.type], (float)(entry.time - stime) * factor, name);
        }
        // clear the profiling data
        graph->performance_profile.clear();
    }
    fflush(fp);
    if (!use_stdout) {
        fclose(fp);
    }
    return VX_SUCCESS;
}

int agoProcessGraph(AgoGraph * graph)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        CAgoLock lock(graph->cs);
        // make sure that graph is verified
        status = VX_SUCCESS;
        if (!graph->verified) {
            status = vxVerifyGraph(graph);
        }
        // execute graph if possible
        if (status == VX_SUCCESS) {
            if (graph->verified && graph->isReadyToExecute) {
                status = agoExecuteGraph(graph);
            }
            else {
                agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: agoProcessGraph: not verified (%d) or not ready to execute (%d)\n", graph->verified, graph->isReadyToExecute);
                status = VX_FAILURE;
            }
        }
    }
    return status;
}

int agoScheduleGraph(AgoGraph * graph)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        status = VX_SUCCESS;
        graph->threadScheduleCount++;
        if (graph->hThread) {
            if (!graph->verified) {
                // make sure to verify the graph in master thread
                CAgoLock lock(graph->cs);
                status = vxVerifyGraph(graph);
            }
            if (status == VX_SUCCESS) {
                // inform graph thread to execute
                if (!ReleaseSemaphore(graph->hSemToThread, 1, nullptr)) {
                    status = VX_ERROR_NO_RESOURCES;
                }
            }
        }
        else {
            status = agoProcessGraph(graph);
        }
    }
    return status;
}

int agoWaitGraph(AgoGraph * graph)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        status = VX_SUCCESS;
        graph->threadWaitCount++;
        if (graph->threadScheduleCount <= 0) // the graph was never scheduled so return VX_FAILURE
            return VX_FAILURE;
        if (graph->hThread) {
            graph->threadThreadWaitState = 1;
            while (graph->threadThreadWaitState == 1) {
                // wait for the agoGraphThreadFunction to be done
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                // release the semaphore in case the agoScheduleGraph was called before the agoGraphThreadFunction
                ReleaseSemaphore(graph->hSemToThread, 1, nullptr);
            }
            while (graph->threadExecuteCount < graph->threadScheduleCount) {
                if (WaitForSingleObject(graph->hSemFromThread, INFINITE) != WAIT_OBJECT_0) {
                    agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: agoWaitGraph: WaitForSingleObject failed\n");
                    status = VX_FAILURE;
                    break;
                }
            }
        }
        if(status == VX_SUCCESS)
            status = graph->status;
    }
    return status;
}