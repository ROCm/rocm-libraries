"""Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

#---------------------------------File Note------------------------------------
#Date: 27 January 2012
#This file defines all the error code and error handler mechanism
#--------------------------------Global Variables------------------------------

UINS_CAT = 100
WIN_REG_SEARCH_FAIL = 101
UNIMPL_APP = 200
SYS_ERR = 300
TIME_OUT = 400
DIM_INCO_FILE_FMT = 500 #incorrect file format for dimension
DIM_FILE_VAL_INCO = 501 #Value coming from dimension file is incorrect

#__errorTable : Defines all the errors in the system. Add a new error code and
#               error message here
"""Error table is defined as private to this module"""
errorTable = {
              UINS_CAT: 'Application is not able to find the installed catalyst',
              WIN_REG_SEARCH_FAIL: 'Windows Registry search for catalysts version is unsuccessful',
              UNIMPL_APP: 'Unimplemented Application requirement',
              SYS_ERR:    'System error occurred - Please check the source code',
              TIME_OUT: 'Operation is timed out',
              DIM_INCO_FILE_FMT: 'incorrect file format for dimension - Not able to find dimension',
              DIM_FILE_VAL_INCO: 'Value coming from dimension file is incorrect'
              }

#--------------------------------Class Definitions-----------------------------
class TimeoutException(Exception):
    pass

"""Base class for handling all the application generated exception"""
class ApplicationException(Exception):

    def __init__(self, fileName, errno, msg = ""):
        self.fileName = fileName
        self.errno = errno
        self.mess = errorTable[errno] + msg
        self.message = 'Application ERROR:'+repr(self.fileName+'-'+str(self.errno)+'-'+self.mess)

    def __str__(self):
        return repr(self.fileName+'-'+str(self.errno)+'-'+self.mess)


#--------------------------------Global Function-------------------------------
if __name__ == '__main__':
    #print errorTable
    try:
        raise ApplicationException('errorHandler', SYS_ERR)

    except:
        print 'Generic exception'
