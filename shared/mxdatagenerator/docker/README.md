# Using Project Dockerfiles

## Building/Running User Image

[user-image/start_user_container](user-image/start_user_container) can be used to automatically build and run the base and user dockerfiles. It must be run from the root of the project repo. It does the following for you:

- Builds both of the base dockerfiles([amdclang](dockerfile-ubuntu-amdclang) and/or [gcc](dockerfile-ubuntu-gcc)).
- Builds the [user dockerfile](user-image/Dockerfile), based on both of the newly built base images.
- Starts a container from the newly built user images with the following properties:
  - It's named with your host username so it's easy to manage later.
  - It has a user whose name and ID match your host user.
  - That user has git configured with the same git username and email as the host user.
  - The host git repo is mounted in the `/data` folder.
  - The correct arguments are passed in to give the container access to the host GPUs.

The defaults are good for most cases. For example, the following will checkout the repo and build/run the gcc and clang dockerfiles.

```shell
$ git checkout git@github.com:ROCm/mxDataGenerator mxDataGenerator
$ cd mxDataGenerator
$ ./docker/user-image/start_user_container
...
Enter these containers with the following commands:
docker exec -ti -u ${USER} ${USER}_dev_clang bash
docker exec -ti -u ${USER} ${USER}_dev_gcc bash
```

After running the script, there will be containers running that you can enter and exit:

```shell
$ docker ps -a
CONTAINER ID   IMAGE                            COMMAND                  CREATED             STATUS            PORTS     NAMES
b6d83ba92ff3   mxdatagen-gcc-user:latest        "/opt/user-entrypoint"   20 minutes ago      Up <1 minutes               USER_dev_gcc
5e502b7b5cab   mxdatagen-amdclang-user:latest   "/opt/user-entrypoint"   21 minutes ago      Up <1 minutes               USER_dev_amdclang
$ docker exec -ti -u ${USER} ${USER}_dev_amdclang bash
$ ls /data
CMakeLists.txt  README.md  build_amdclang  build_gcc  docker  lib test
$ exit
$ docker exec -ti -u ${USER} ${USER}_dev_gcc bash
$ ls /data
CMakeLists.txt  README.md  build_amdclang  build_gcc  docker  lib test
$ exit
```

> Note: The containers started by the script will exist until they are manually removed. To manually remove a container you can use `docker rm -f <container name>`.

The following is a more complicated example that uses the following options:

- `--privileged`: runs the container in prvileged mode.
- `--delete_existing_container`: will remove any already launched containers and replace them with newly launched ones.
- `--versions clang`: instructs the script to only build and run the clang version of the dockerfile.
- `--user_build_extra_args="--no-cache"`: passes the extra argument `--no-cache` to `docker build` when building the user image.
- `--volumes="/home/USER:/host_home"`: adds the extra volume mapping to the `docker run` command.

```shell
$ ./docker/user-image/start_user_container --privileged \
                                           --delete_existing_container \
                                           --versions amdclang \
                                           --user_build_extra_args="--no-cache" \
                                           --volumes="/home/USER:/host_home"
```
For a full list of options see `./docker/user-image/start_user_container -h`.
