# Using VSCode with Docker

## Caveats

Git and the Docker container have been set up to work with the host's git name and email.  However, git credentials have not been passed. To perform git pull and push commands, please
run **Remote-Containers: Reopen folder in SSH** in VSCode to exit the container.  Alternatively, you can follow these [additional instructions](https://code.visualstudio.com/docs/remote/containers#_using-ssh-keys).

## Using VSCode's Remote Container

**WARNING**: Delete the build directory in this repository if you have been using root in a previous Docker container. Do this from the previous Docker container. Failing to do so can lead to files that are difficult to remove and can cause CMake to fail.

1. Install the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

2. Run the following to create the file on the machine with the source code from the root of the workspace directory:

    `mkdir -p .devcontainer && touch .devcontainer/devcontainer.json`

3. Edit .devcontainer/devcontainer.json to contain the following information. Replace items in angled brackets with your information.

```json
// For format details, see https://aka.ms/devcontainer.json. For config options, see the README at:
// https://github.com/microsoft/vscode-dev-containers/tree/v0.234.0/containers/cpp
{
    "name": "gcc",
    "build": {
        "dockerfile": "../docker/dockerfile-ubuntu-gcc",
        "args": {
            "VARIANT": "focal"
        },
        "context": ".."
    },
    "runArgs": [
        "--cap-add=SYS_PTRACE",
        "--security-opt",
        "seccomp=unconfined",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--net",
        "host",
        "-u=vscode",
        "--group-add",
        "video",
        "-v",
        "<ABSOLUTE_PATH_TO_LOCAL_REPOSITORY>/rocroller:/data",
        "-e",
        "GIT_NAME",
        "-e",
        "GIT_EMAIL"
    ],
    // Set *default* container specific settings.json values on container create.
    "settings": {},
    // Add the IDs of extensions you want installed when the container is created.
    "extensions": [
        "ms-vscode.cpptools",
        "cschlosser.doxdocgen"
    ],
    // Use 'forwardPorts' to make a list of ports inside the container available locally.
    // "forwardPorts": [],
    // Use 'postCreateCommand' to run commands after the container is created.
    // "postCreateCommand": "gcc -v",
    // Comment out to connect as root instead. More info: https://aka.ms/vscode-remote/containers/non-root.
    "remoteUser": "vscode",
    "features": {
        "git": "os-provided"
    }
}

```

4. **NOTE**: To set up a clang configuration, update the following entry:

    `"dockerfile": "../docker/dockerfile-ubuntu-clang"`

5. Run **Remote-Containers: Reopen in Container** in VSCode to use the container. You may be asked to rebuild the container if there have been alterations to the Dockerfiles.

## Debugging with VSCode

For gdb debugging, add the following to .vscode/launch.json:

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "rocRoller Tests (gdb)",
            "type": "cppdbg",
            "request": "launch",
            "program": "/data/build/test/rocroller-tests",
            "args": ["--gtest_catch_exceptions=0", "--gtest_break_on_failure"],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/build",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                },
                {
                    "description":  "Set Disassembly Flavor to Intel",
                    "text": "-gdb-set disassembly-flavor intel",
                    "ignoreFailures": true
                }
            ]
        }

    ]
}
```

Note the following:

- Multiple configurations can be added by adding to the `configuration` list. Change the `name`, `program`, and any other fields as required.  This can also be done by the VSCode wizard.
- The current working directory should be the build directory.
- Refer to [Using C++ on Linux](https://code.visualstudio.com/docs/cpp/config-linux) and [Debug C++](https://code.visualstudio.com/docs/cpp/cpp-debug) from the VSCode website for more information.
