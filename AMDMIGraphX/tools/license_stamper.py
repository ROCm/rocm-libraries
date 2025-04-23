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
# This license_stamper script is to be triggered manually via users to update the license stamps for all the
# files in the current local branch. It works by generating a list of all the files in the current active
# local branch via GIT LS-FILES and then compares each files license stamp year to the latest
# commit year (which is found via GIT LOG). It then updates the year if needed. If a license stamp is not found, then
# it will add a stamp at the begenning of the file with the year set to the current year.
#####################################################################################
import subprocess, os, datetime, re, argparse

current_year = datetime.date.today().year

__repo_dir__ = os.path.normpath(
    os.path.join(os.path.realpath(__file__), '..', '..'))

delimiters = {
    ".cpp": "*",
    ".hpp": "*",
    ".h": "*",
    ".ipynb": "//",
    ".py": "#",
    ".txt": "#",
    ".bsh": "#",
    ".sh": "#",
    ".cmake": "#"
}
extensions = delimiters.keys()


# Get the delimiter from the file type based on what we care about to tag with our licence
# file. Otherwise return None for the delimiter and skip the file
def getDelimiter(filename):
    delimiter = None
    for extension in extensions:
        if filename.endswith(extension):
            delimiter = delimiters[extension]
            break
    return delimiter


def eval(cmd, **kwargs):
    return subprocess.run(cmd,
                          capture_output=True,
                          shell=True,
                          check=True,
                          **kwargs).stdout.decode('utf-8').strip()


def get_head():
    return eval("git rev-parse --abbrev-ref HEAD")


def get_merge_base(branch):
    head = get_head()
    return eval(f"git merge-base {branch} {head}")


def get_files_changed(against):
    files = eval(f"git diff-index --cached --name-only {against}",
                 cwd=__repo_dir__).splitlines()
    return [f for f in files if getDelimiter(f)]


def get_all_files():
    files = eval("git ls-files --exclude-standard",
                 cwd=__repo_dir__).splitlines()
    return [f for f in files if getDelimiter(f)]


# Markdown code blob we should use to insert into notebook files
def getipynb_markdownBlockAsList():
    markdownBlock = [
        '\t{\n'
        '\t\t"cell_type": "code",\n', '\t\t"execution_count": null,\n',
        '\t\t"metadata": {},\n', '\t\t"outputs": [],\n', '\t\t"source": [\n',
        '\t\t\t\"#  The MIT License (MIT)\\n\",\n', '\t\t\t\"#\\n\",\n',
        f'\t\t\t\"#  Copyright (c) 2015-{current_year} Advanced Micro Devices, Inc. All rights reserved.\\n\",\n',
        '\t\t\t\"#\\n\",\n',
        '\t\t\t\"#  Permission is hereby granted, free of charge, to any person obtaining a copy\\n\",\n',
        '\t\t\t\"#  of this software and associated documentation files (the \'Software\'), to deal\\n\",\n',
        '\t\t\t\"#  in the Software without restriction, including without limitation the rights\\n\",\n',
        '\t\t\t\"#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\\n\",\n',
        '\t\t\t\"#  copies of the Software, and to permit persons to whom the Software is\\n\",\n',
        '\t\t\t\"#  furnished to do so, subject to the following conditions:\\n\",\n',
        '\t\t\t\"#\\n\",\n',
        '\t\t\t\"#  The above copyright notice and this permission notice shall be included in\\n\",\n',
        '\t\t\t\"#  all copies or substantial portions of the Software.\\n\",\n',
        '\t\t\t\"#\\n\",\n',
        '\t\t\t\"#  THE SOFTWARE IS PROVIDED \'AS IS\', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\\n\",\n',
        '\t\t\t\"#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\\n\",\n',
        '\t\t\t\"#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE\\n\",\n',
        '\t\t\t\"#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\\n\",\n',
        '\t\t\t\"#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\\n\",\n',
        '\t\t\t\"#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\\n\",\n',
        '\t\t\t\"#  THE SOFTWARE.\\n\"\n', '\t\t]\n', '\t},'
    ]
    return markdownBlock


def hasKeySequence(inputfile, key_message):
    result = False
    if key_message in inputfile:
        result = True

    return result


def getYearOfLatestCommit(rfile: str) -> int:
    proc2 = subprocess.run(f"git log -1 --format=%cd --date=short {rfile}",
                           shell=True,
                           stdout=subprocess.PIPE,
                           cwd=__repo_dir__)
    year = datetime.datetime.strptime(proc2.stdout.decode().strip(),
                                      '%Y-%m-%d').year
    return year


def currentYear() -> int:
    return datetime.datetime.now().date().year


def updateYear(filename: str, lastCommitYear: int) -> None:
    with open(filename, 'r') as f:
        newfileContent = re.sub(
            "2015-\d+ Advanced Micro Devices",
            f'2015-{lastCommitYear} Advanced Micro Devices', f.read())

    with open(filename, 'w') as f:
        f.write(newfileContent)


# Header and footer of the comment block
# modify these if we want some different style
def topHeader(commentChar):
    delim = None

    #Early return
    if "//" in commentChar:
        delim = getipynb_markdownBlockAsList()
        delim.append("\n")
        return ''.join(str(x) for x in delim)

    if "*" in commentChar:
        delim = "/*\n"
    if "#" in commentChar:
        delim = "#####################################################################################\n"
    return delim


def bottomFooter(commentChar):
    delim = None
    #Early return - no footer handled by
    if "//" in commentChar:
        return delim

    if "*" in commentChar:
        delim = "*/\n"
    if "#" in commentChar:
        delim = "#####################################################################################\n"
    return delim


# Simple just open and write stuff to each file with the license stamp
def openAndWriteFile(rfile,
                     message,
                     commentChar,
                     useLastCommitYear=False,
                     debug=False):

    filename = os.path.join(__repo_dir__, rfile)
    add_shebang = False
    #markdown file stamping for .ipynb
    save_markdown_lines = []
    modify_markdown = False

    #open save old contents and append things here
    if debug is True:
        print("Open", filename, end='')

    try:
        file = open(filename, 'r')
    except OSError as e:
        if debug is True:
            print(str(e) + "....Open Error: Skipping file ")
        file.close()
        return
    else:
        with file as contents:
            try:
                if commentChar != "//":
                    saved_shebang = contents.readline()
                    add_shebang = hasKeySequence(saved_shebang, "#!")

                    # No shebang so start at beginning line
                    if add_shebang is False:
                        contents.seek(0)

                # Get the first tags in notebook before we insert license into a cell as a comment block
                if commentChar == "//":
                    save_markdown_lines.extend(contents.readline())  # { tag
                    save_markdown_lines.extend(
                        contents.readline())  # "cells": [ tag
                    modify_markdown = True

                #read remaining lines in the original file
                save = contents.read()

                hasAmdLic = hasKeySequence(
                    save, "Advanced Micro Devices, Inc. All rights reserved")
                hasOtherLic = hasKeySequence(save, "Software License")

                # Check if we have a licence stamp already and the latest
                # commit year matches license year (unless commit year is
                # pre-2023, which case its ignored)
                if hasAmdLic or hasOtherLic is True:
                    contents.close()

                    year = getYearOfLatestCommit(
                        rfile) if useLastCommitYear else currentYear()

                    if not hasKeySequence(
                            save, f'2015-{year} Advanced Micro Devices'
                    ) and year > 2022:
                        if debug:
                            print(
                                f"....Already stamped but wrong year: Updating the year to {year}"
                            )
                        return updateYear(filename, year)

                    if debug:
                        print("....Already Stamped: Skipping file ")
                    return

            except UnicodeDecodeError as eu:
                if debug is True:
                    print(str(eu) + "...Skipping binary file ")
                contents.close()
                return

    if debug is True:
        print("...Writing header", end='')

    with open(filename, 'w') as contents:
        # append the licence to the top of the file

        # Append shebang before license
        if add_shebang is True:
            contents.write(saved_shebang + "\n")

        #Append markdown hooks before license
        if modify_markdown is True:
            contents.write(''.join(str(x) for x in save_markdown_lines))

        delim = topHeader(commentChar)
        if delim is not None:
            contents.write(delim)
            #print(delim)

        if modify_markdown is False:
            for line in message:
                if line != '':
                    contents.write(commentChar + " " + line + "\n")
                else:
                    contents.write(commentChar + "\n")

        delim = bottomFooter(commentChar)
        if delim is not None:
            contents.write(delim)

        # write remaining contents
        contents.write(save)
    if debug is True:
        print("...done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('against', default='develop', nargs='?')
    parser.add_argument('-a',
                        '--all',
                        action='store_true',
                        help='Update all files')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable debug output')
    args = parser.parse_args()

    message = open(os.path.join(__repo_dir__, 'LICENSE')).read().split('\n')

    # Get a list of all the files in our git repo
    fileList = None
    if args.all:
        fileList = get_all_files()
    else:
        fileList = get_files_changed(get_merge_base(args.against))

    if args.debug is True:
        print("Target file list:\n" + '\n'.join(fileList))
        print("Output Message:\n" + '\n'.join(message))

    for rfile in fileList:
        commentDelim = getDelimiter(rfile)
        if commentDelim is not None:
            print(f"Updating file: {rfile}")
            openAndWriteFile(rfile,
                             message,
                             commentDelim,
                             useLastCommitYear=args.all,
                             debug=args.debug)
        elif args.debug:
            print(f"No valid delimeter for file: {rfile}")


if __name__ == "__main__":
    main()
