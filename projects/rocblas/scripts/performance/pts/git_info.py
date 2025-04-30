"""Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

import git
from pathlib import Path
import sys



def get_merge_id(git_log : list):

    """
    This method is used to get merge id and pull id from a git log.

    Args
    ------
    git_log : string consisting of all the information about commit id.

    Returns
    -------
    merge_id, pull_id : Merge id , Pull id for the latest commit.

    """

    merge_id = ''
    pull_id = ''

    try :
        merge_id = git_log[0][7:]
        for i in git_log:
            if "merge pull request #" in i.lower():
                pull_id = i.lower().split()[3][1:]

    except Exception as e:
        print("No Merge requests found.")

    return merge_id, pull_id


def create_github_file(filename: str) -> str:
    """
    This method creates a github information file in the specified directory.

    This function creates four different parameters of a
    git repo { Branch Name , Merge Id ,  Pull Id ,  Git Hash }

    To run this function you must be in a project/package or in any git repo folder
    (or) you need to give the location of a git repo when prompted.

    Args
    -----
    filename : Absolute filename in which the path is to be created

    Returns
    -------
    filename : Absolute filename of created file

    Raises
    ------
    InvalidGitRepositoryError : If the package is running outside of a git repo
    IOError (Is a directory error) : If the specified filename ( input ) should be a file
                                     and not a directory

    Usage
    -----
    CLI syntax
    ----------
    linux : python3 -m pts_amd create_git_info < filename >
    windows : python -m pts_amd create_git_info < filename >

    """

    # Declaring the git info parameters as empty strings
    branch_name = ''
    merge_id = ''
    pull_id = ''
    git_hash = ''

    try:
        # Retrieving the values, when package/project is running inside any git repo
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha
        git_logs = repo.git.log("--grep=Merge","--max-count=1")
        if len(git_logs) > 0 :
            merge_id, pull_id = get_merge_id(git_logs.split("\n"))
        try:
            branch_name = repo.active_branch.name
        except TypeError:
            branch_name = "None"



    except (git.InvalidGitRepositoryError, git.exc.NoSuchPathError):
        print('')

    # Asking the user to specify location, if the above code doesn't retrieve the parameter values.
    if git_hash == '':
        print("Git information file is missing in the specified location.")
        github_loc = Path(input("Please specify the location of local repository or name of the file containing git info: "))

        if github_loc.exists():
            try:
                repo = git.Repo(github_loc, search_parent_directories=True)
                git_hash = repo.head.object.hexsha
                git_logs = repo.git.log("--grep=Merge","--max-count=1")
                if len(git_logs) > 0 :
                    merge_id, pull_id = get_merge_id(git_logs.split("\n"))
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = "None"

            except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
                try :
                    with open(github_loc, 'r') as git_file:
                        all_lines = git_file.read().splitlines()
                        lines = [name for name in all_lines if name]
                        for i in lines:
                            if 'branch' in i.lower():
                                branch_name = i.split(':')[1]
                            elif 'merge' in i.lower():
                                merge_id = i.split(':')[1]
                            elif 'pull' in i.lower():
                                pull_id = i.split(':')[1]
                            elif 'hash' in i.lower():
                                git_hash = i.split(':')[1]
                        if (branch_name == merge_id == pull_id == git_hash == '') and len(lines) >= 4 :
                            branch_name = lines[0]
                            merge_id = lines[1]
                            pull_id = lines[2]
                            git_hash = lines[3]

                except IOError:
                    print('Could not find any git files in the specified location.')
                    sys.exit()
        else:
            print('The specified location is not found.')
            sys.exit()

    if not branch_name:
        print('')
        branch_name = input("Branch Name is not defined. please provide branch name :")
    if not merge_id:
        print('')
        merge_id = 'None'
    if not pull_id:
        print('')
        pull_id = 'None'
    if not git_hash:
        print('')
        git_hash = input("Git Hash is not defined. please provide git hash :")

    # If any parameter still remains as empty string then terminating the process.
    if (not branch_name) or (not merge_id) or (not pull_id) or (not git_hash):
        print('')
        print('Git Info is not specified correctly.')
        sys.exit()

    # If all parameters are captured successfully then those parameters are written in a file.
    try :
        with open(filename, 'w') as file:
            file.write(f"Branch Name : {branch_name}\n")
            file.write(f"Merge Id : {merge_id}\n")
            file.write(f"Pull Id : {pull_id}\n")
            file.write(f"Git Hash : {git_hash}\n")

    except Exception as error:
        print(error)
        sys.exit()

    return filename
