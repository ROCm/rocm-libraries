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


#ifndef __vxEngineUtil_h__
#define __vxEngineUtil_h__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class CFileBuffer {
public:
	CFileBuffer(const char * fileName) {
		size_in_bytes = 0; buffer_allocated = buffer_aligned = 0;
		FILE * fp = fopen(fileName, "rb");
		if (!fp) {
			printf("ERROR: unable to open '%s'\n", fileName);
		}
		else {
			fseek(fp, 0L, SEEK_END); size_in_bytes = ftell(fp); fseek(fp, 0L, SEEK_SET);
			buffer_allocated = new unsigned char[size_in_bytes + 32];
			buffer_aligned = (unsigned char *)((((size_t)buffer_allocated) + 31) & ~31);
			size_t n = fread(buffer_aligned, 1, size_in_bytes, fp);
			if (n < size_in_bytes)
				memset(&buffer_aligned[n], 0, size_in_bytes - n);
			buffer_aligned[size_in_bytes] = 0;
			//printf("OK: read %d bytes from %s\n", size_in_bytes, fileName);
			fclose(fp);
		}
	}
	CFileBuffer(size_t _size_in_bytes, size_t _prefix_bytes = 0, size_t _postfix_bytes = 0) {
		size_in_bytes = _size_in_bytes;
		prefix_bytes = _prefix_bytes;
		postfix_bytes = _postfix_bytes;
		buffer_allocated = new unsigned char[size_in_bytes + prefix_bytes + postfix_bytes + 32];
		buffer_aligned = (unsigned char *)((((size_t)(buffer_allocated + prefix_bytes)) + 31) & ~31);
		memset(buffer_aligned, 0, size_in_bytes);
	}
	~CFileBuffer() { if (buffer_allocated) delete[] buffer_allocated; }
	void * GetBuffer() { return buffer_aligned; }
	size_t GetSizeInBytes() { return size_in_bytes; }
	int WriteFile(const char * fileName) {
		if (!buffer_aligned) return -1;
		FILE * fp = fopen(fileName, "wb"); if (!fp) { printf("ERROR: unable to open '%s'\n", fileName); return -1; }
		fwrite(buffer_aligned, 1, size_in_bytes, fp); fclose(fp);
		printf("OK: wrote %d bytes into %s\n", (int)size_in_bytes, fileName);
		return 0;
	}
private:
	unsigned char * buffer_allocated, *buffer_aligned;
	size_t size_in_bytes, prefix_bytes, postfix_bytes;
};

#endif /* __vxEngineUtil_h__ */