!! Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
!!
!! Permission is hereby granted, free of charge, to any person obtaining a copy
!! of this software and associated documentation files (the "Software"), to deal
!! in the Software without restriction, including without limitation the rights
!! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
!! copies of the Software, and to permit persons to whom the Software is
!! furnished to do so, subject to the following conditions:
!!
!! The above copyright notice and this permission notice shall be included in
!! all copies or substantial portions of the Software.
!!
!! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
!! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
!! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
!! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
!! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
!! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
!! THE SOFTWARE.

module hipfor
    use iso_c_binding
    implicit none

    integer, public :: hipSuccess = 0

    integer, public :: hipMemcpyHostToHost = 0
    integer, public :: hipMemcpyHostToDevice = 1
    integer, public :: hipMemcpyDeviceToHost = 2
    integer, public :: hipMemcpyDeviceToDevice = 3
    integer, public :: hipMemcpyDefault = 4

    type, bind(C) :: dim3
        integer(c_int) :: x, y, z
    end type dim3

    interface
        function hipMalloc(ptr, length) bind(C, name = "hipMalloc")
            use iso_c_binding
            implicit none
            type(c_ptr) :: ptr
            integer(c_size_t), value :: length
            integer(c_int) :: hipMalloc
        end function

        function hipMemcpy(dst, src, length, dir) bind(C, name = "hipMemcpy")
            use iso_c_binding
            implicit none
            type(c_ptr), value :: dst, src
            integer(c_size_t), value :: length
            integer(c_int), value :: dir
            integer(c_int) :: hipMemcpy
            end function

        function hipFree(ptr) bind(C, name = "hipFree")
            use iso_c_binding
            implicit none
            type(c_ptr), value :: ptr
            integer(c_int) :: hipFree
        end function

        function hipDeviceSynchronize() bind(C, name = "hipDeviceSynchronize")
            use iso_c_binding
            implicit none
            integer(c_int) :: hipDeviceSynchronize
        end function
    end interface

end module
