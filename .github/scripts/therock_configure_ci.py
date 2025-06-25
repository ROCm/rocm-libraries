"""
This script determines which build flag and tests to run based on SUBTREES

Required environment variables:
  - SUBTREES
"""

import json
from therock_matrix import subtree_to_project_map, project_map
from typing import Mapping
import os

SUBTREES = os.getenv("SUBTREES", "")


def set_github_output(d: Mapping[str, str]):
    """Sets GITHUB_OUTPUT values.
    See https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/passing-information-between-jobs
    """
    print(f"Setting github output:\n{d}")
    step_output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not step_output_file:
        print("Warning: GITHUB_OUTPUT env var not set, can't set github outputs")
        return
    with open(step_output_file, "a") as f:
        f.writelines(f"{k}={v}" + "\n" for k, v in d.items())


def run():
    subtrees = SUBTREES.split("\n")
    projects = set()
    # collect the associated subtree to project
    for subtree in subtrees:
        if subtree in subtree_to_project_map:
            projects.add(subtree_to_project_map.get(subtree))
    
    # retrieve the subtrees to checkout, cmake options to build, and projects to test 
    project_to_run = []
    for project in projects:
        if project in project_map:
            project_to_run.append(project_map.get(project))
        

    set_github_output({"projects": json.dumps(project_to_run)})


if __name__ == "__main__":
    run()
