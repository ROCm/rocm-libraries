#!/usr/bin/env python3
#####################################################################################
#  The MIT License (MIT)
#
#  Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#####################################################################################
#################################### GUIDE ##########################################
#####################################################################################
# This Check_Stamped script is triggered by the Github workflows when a pull request is created.
# It works by generating a list of files that have been modified/created between the current up-to-date Develop Branch
# from MIGraphx and the Pull Request Branch via GIT DIFF. The script then checks that each file has the current year
# in the license stamp, with the assumption being that any modifications/creations will need to be stamped to the year that the
# modification/creation was made.
#####################################################################################
import subprocess, sys, datetime, argparse, os

current_year = datetime.date.today().year

# The filetypes we want to check for that are stamped
# LICENSE is included here as it SHOULD have a license in it otherwise flag it as unstamped
supported_file_types = (".cpp", ".hpp", ".h", ".ipynb", ".py", ".txt", ".sh",
                        ".bsh", "LICENSE", ".cmake")

# add general stuff we shouldn't stamp and any exceptions here
unsupported_file_types = [
    ".onnx", ".pb", ".rst", ".jpg", ".jpeg", ".proto", ".md", ".clang",
    ".weight", ".ini", ".json", ".docker", ".git", ".rules", ".yml"
]

specificIgnores = ("digits.txt", "Dockerfile", "Jenkinsfile",
                   'imagenet_classes.txt')

unsupportedFiles = []
unstampedFiles = []
stampedFilesWithBadYear = []
unknownFiles = []


def hasKeySequence(inputfile: str, key_message: str) -> bool:
    if key_message in inputfile:
        return True
    return False


# Simple just open and write stuff to each file with the license stamp
def needStampCheck(filename: str, debug: bool) -> bool:
    # open save old contents and append things here
    if debug: print("Open", filename, end=' ')
    #Empty name isn't a filename
    if filename == '' or filename == "": return False

    try:
        file = open(filename, 'r')
    except OSError as e:
        if debug: print(str(e) + "....Open Error: Skipping  file ")
        file.close()
        return False
    else:
        with file as contents:
            try:
                save = contents.read()

                # Check if we have a license stamp already
                if hasKeySequence(
                        save,
                        "Advanced Micro Devices, Inc. All rights reserved"):
                    if not hasKeySequence(
                            save,
                            f"2015-{current_year} Advanced Micro Devices"):
                        if debug: print("....Already Stamped but wrong year")
                        stampedFilesWithBadYear.append(filename)

                    elif debug:
                        print("....Already Stamped: Skipping  file ")

                    contents.close()
                    return False

            except UnicodeDecodeError as eu:
                if debug: print(f"{str(eu)}...Skipping binary file ")
                contents.close()
                return False

    return True


# Check if any element in fileTuple is in filename
def check_filename(filename: str, fileTuple: tuple or list) -> bool:
    if any([x in filename for x in fileTuple]):
        return True
    return False


def eval(cmd, **kwargs):
    return subprocess.run(cmd,
                          capture_output=True,
                          shell=isinstance(cmd, str),
                          check=True,
                          **kwargs).stdout.decode('utf-8').strip()


def is_excluded(f):
    base = os.path.basename(f)
    return base in unsupported_file_types


def get_top():
    return eval("git rev-parse --show-toplevel")


def get_head():
    return eval("git rev-parse --abbrev-ref HEAD")


def get_merge_base(branch):
    head = get_head()
    return eval(f"git merge-base {branch} {head}")


def get_files_changed(against):
    files = eval(
        f"git diff-index --cached --name-only --diff-filter=d {against}",
        cwd=get_top()).splitlines()
    return (f for f in files
            if f.endswith(supported_file_types) and not is_excluded(f))


def main(branch, debug) -> None:
    unsupported_file_types.extend(specificIgnores)

    fileList = list(get_files_changed(branch))

    if debug:
        print(f"Branch: {branch}, Target file list {len(fileList)}:\n" +
              str(fileList))

    for file in fileList:
        if check_filename(file, supported_file_types):
            if needStampCheck(file, debug) and not check_filename(
                    file, unsupported_file_types):
                unstampedFiles.append(file)
            else:
                unsupportedFiles.append(file)

        elif check_filename(file, unsupported_file_types):
            unsupportedFiles.append(file)
        else:
            if file != '':
                unknownFiles.append(file)

    # Do a bunch of checks based on our file lists
    if len(unstampedFiles) > 0:
        print("\nError: The following " + str(len(unstampedFiles)) +
              " files are currently without a license:")
        print(str(unstampedFiles))
        sys.exit(1)

    elif len(stampedFilesWithBadYear) > 0:
        print(
            f"\nError: The licenses for the following {str(len(stampedFilesWithBadYear))} file(s) either... do not match the year of commit, have a different copyright format or have not been synced from the latest {branch} branch:\n{str(stampedFilesWithBadYear)}\nThere is a license_stamper script (./tools/license_stamper.py), which you can run to automatically update and add any needed license stamps"
        )
        sys.exit(1)

    elif len(unknownFiles) > 0:
        print("\nError: The following " + str(len(unknownFiles)) +
              " files not handled:")
        print(str(unknownFiles))
        sys.exit(2)

    else:
        print("Success: All files stamped and up to date")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("branch")
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()

    main(args.branch, args.debug)
