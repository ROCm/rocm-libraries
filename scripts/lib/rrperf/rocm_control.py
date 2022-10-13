import subprocess
import shutil
from time import sleep

def pin_clocks(ROCmSMIPath):
    print("Attempting to pin clocks...")
    rocm_smi_found = shutil.which(ROCmSMIPath) != None
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
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("Pinning clocks finished...\n{}\n{}".format(pinresult.stdout.decode("ascii"), pinresult.stderr.decode("ascii")))
        sleep(1)
        checkresult = subprocess.run(
            [
                ROCmSMIPath,
                "-d",
                "0",
                "-a"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE
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
