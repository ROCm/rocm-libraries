#!/usr/bin/env python3

import os
import argparse


def create_directory(directory_path):
    try:
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")


def create_file(directory_path, file_name_with_ext, file_contents=""):
    # Create the complete file path
    file_path = os.path.join(directory_path, file_name_with_ext)

    try:
        # Open the file in write mode
        with open(file_path, "w") as file:
            # Write the contents to the file
            file.write(file_contents)
        print(f"File '{file_path}' created/overwritten.")
    except IOError:
        print(f"IOError on '{file_path}'.")


def main(args):
    name = args.name
    directory_path = os.path.join(os.getcwd(), args.path, name)
    create_directory(directory_path)
    create_file(directory_path, name + "Selected.cpp")
    create_file(directory_path, name + "NotSelected.cpp")
    create_file(directory_path, name + ".expected")
    create_file(directory_path, name + ".qlref", name + ".ql")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates and loads a directory with files to test a CodeQl query"
    )
    parser.add_argument("name", help="file name (no ext) of the CodeQL query")
    parser.add_argument(
        "-p", "--path", help="path for dir to be created in", required=True
    )
    args = parser.parse_args()

    main(args)
