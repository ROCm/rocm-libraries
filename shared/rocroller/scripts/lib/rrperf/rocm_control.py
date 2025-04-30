################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import shutil
import subprocess
from time import sleep


def pin_clocks(ROCmSMIPath):
    print("Attempting to pin clocks...")
    rocm_smi_found = shutil.which(ROCmSMIPath) is not None
    if rocm_smi_found:
        print("{} found, pinning clocks...".format(ROCmSMIPath))
        pinresult = subprocess.run(
            [
                ROCmSMIPath,
                "-d",
                "0",
                "--setfan",
                "255",
                "--setsclk",
                "7",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(
            "Pinning clocks finished...\n{}\n{}".format(
                pinresult.stdout.decode("ascii"),
                pinresult.stderr.decode("ascii"),
            )
        )
        sleep(1)
        checkresult = subprocess.run(
            [ROCmSMIPath, "-d", "0", "-a"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Clocks status:\n{}".format(checkresult.stdout.decode("ascii")))
        print("Setting up clock restore...")
        setupRestoreClocks(ROCmSMIPath)
        print("Pinning clocks finished.")
    else:
        print("{} not found, unable to pin clocks.".format(ROCmSMIPath))


def setupRestoreClocks(ROCmSMIPath):
    import atexit

    def restoreClocks():
        print("Resetting clocks...")
        subprocess.call([ROCmSMIPath, "-d", "0", "--resetclocks"])
        print("Resetting fans...")
        subprocess.call([ROCmSMIPath, "-d", "0", "--resetfans"])

    atexit.register(restoreClocks)
