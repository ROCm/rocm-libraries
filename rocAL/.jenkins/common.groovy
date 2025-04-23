// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false) {
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String backend = 'HIP'
    String enableSCL = 'echo rocAL Build'
    String libLocation = ''

    if (platform.jenkinsLabel.contains('centos')) {
        backend = 'CPU'
        if (platform.jenkinsLabel.contains('centos7')) {
            enableSCL = 'source scl_source enable llvm-toolset-7'
        }
    }
    else if (platform.jenkinsLabel.contains('rhel')) {
        libLocation = ':/usr/local/lib'
    }
    else if (platform.jenkinsLabel.contains('ubuntu20')) {
        backend = 'OCL'
    }

    def command = """#!/usr/bin/env bash
                set -x
                ${enableSCL}
                echo Build rocAL - ${buildTypeDir}
                cd ${project.paths.project_build_prefix}
                sudo python rocAL-setup.py --backend ${backend}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                cmake -DBACKEND=${backend} -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" ../..
                make -j\$(nproc)
                sudo cmake --build . --target PyPackageInstall
                sudo make install
                ldd -v /opt/rocm/lib/librocal.so
                """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project) {

    String libLocation = ''
    String packageManager = 'apt -y'
    String toolsPackage = 'llvm-amdgpu-dev'
    String llvmLocation = '/opt/amdgpu/lib/x86_64-linux-gnu/llvm-20.1/bin'
    
    if (platform.jenkinsLabel.contains('rhel')) {
        libLocation = ':/usr/local/lib:/usr/local/lib/x86_64-linux-gnu'
        packageManager = 'yum -y'
        toolsPackage = 'llvm-amdgpu-devel'
        llvmLocation = '/opt/amdgpu/lib64/llvm-20.1/bin'
    }
    else if (platform.jenkinsLabel.contains('sles')) {
        libLocation = ':/usr/local/lib:/usr/local/lib/x86_64-linux-gnu'
        packageManager = 'zypper -n'
        toolsPackage = 'llvm-amdgpu-devel'
        llvmLocation = '/opt/amdgpu/lib64/llvm-20.1/bin'
    }

    String commitSha
    String repoUrl
    (commitSha, repoUrl) = util.getGitHubCommitInformation(project.paths.project_src_prefix)

    withCredentials([string(credentialsId: "mathlibs-codecov-token-rocal", variable: 'CODECOV_TOKEN')])
    {
        def command = """#!/usr/bin/env bash
                    export HOME=/home/jenkins
                    set -x
                    cd ${project.paths.project_build_prefix}/build
                    export LLVM_PROFILE_FILE=\"\$(pwd)/rawdata/rocal-%p.profraw\"
                    echo \$LLVM_PROFILE_FILE
                    cd release
                    mkdir -p test && cd test
                    cmake /opt/rocm/share/rocal/test/
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ctest -VV --rerun-failed --output-on-failure
                    cd ../
                    wget http://math-ci.amd.com/userContent/computer-vision/MIVisionX-data/MIVisionX-data-main.zip
                    unzip MIVisionX-data-main.zip
                    export ROCAL_DATA_PATH=\$(pwd)/MIVisionX-data-main/
                    mkdir -p rocal-unit-tests && cd rocal-unit-tests
                    python3 -m pip install Pillow
                    cp -r /opt/rocm/share/rocal/test/unit_tests/ .
                    cd unit_tests/
                    chmod +x -R testAllScripts.sh
                    ./testAllScripts.sh
                    cd ../../ && mkdir -p external-source-reader-test && cd external-source-reader-test
                    cmake /opt/rocm/share/rocal/test/external_source/
                    make -j
                    ./external_source ../MIVisionX-data-main/rocal_data/coco/coco_10_img/images/
                    ./external_source ../MIVisionX-data-main/rocal_data/coco/coco_10_img/images/ 1
                    cd ../ && mkdir -p audio-tests && cd audio-tests
                    python3 /opt/rocm/share/rocal/test/audio_tests/audio_tests.py
                    cd ../ && mkdir -p python-api-tests && cd python-api-tests
                    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib/
                    export PATH=\$PATH:/opt/rocm/bin
                    export PYTHONPATH=/opt/rocm/lib:\$PYTHONPATH
                    python3 -m pip install opencv-python
                    python3 /opt/rocm/share/rocal/test/python_api/prefetch_queue_depth/prefetch_queue_depth.py /opt/rocm/share/rocal/test/data/images/AMD-tinyDataSet cpu 128 8
                    python3 /opt/rocm/share/rocal/test/python_api/prefetch_queue_depth/prefetch_queue_depth.py /opt/rocm/share/rocal/test/data/images/AMD-tinyDataSet gpu 128 8
                    python3 /opt/rocm/share/rocal/test/python_api/external_source_reader.py cpu 128
                    python3 /opt/rocm/share/rocal/test/python_api/external_source_reader.py gpu 128
                    python3 /opt/rocm/share/rocal/test/python_api/numpy_reader.py --image-dataset-path ../MIVisionX-data-main/rocal_data/numpy/ --no-rocal-gpu
                    python3 /opt/rocm/share/rocal/test/python_api/numpy_reader.py --image-dataset-path ../MIVisionX-data-main/rocal_data/numpy/ --rocal-gpu
                    cd ../../
                    sudo ${packageManager} install lcov ${toolsPackage}
                    ${llvmLocation}/llvm-profdata merge -sparse rawdata/*.profraw -o rocal.profdata
                    ${llvmLocation}/llvm-cov export -object release/lib/librocal.so --instr-profile=rocal.profdata --format=lcov > coverage.info
                    lcov --remove coverage.info '/opt/*' --output-file coverage.info
                    lcov --list coverage.info
                    lcov --summary  coverage.info
                    curl -Os https://uploader.codecov.io/latest/linux/codecov
                    chmod +x codecov
                    ./codecov -v -U \$http_proxy -t ${CODECOV_TOKEN} --file coverage.info --name rocAL --sha ${commitSha}
                    """

        platform.runCommand(this, command)
    }
// Unit tests - TBD
}

def runPackageCommand(platform, project) {

    def packageHelper = platform.makePackage(platform.jenkinsLabel, "${project.paths.project_build_prefix}/build/release")
    
    String packageType = ''
    String packageInfo = ''
    String packageDetail = ''
    String osType = ''
    String packageRunTime = ''

    if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('rhel') || platform.jenkinsLabel.contains('sles')) {
        packageType = 'rpm'
        packageInfo = 'rpm -qlp'
        packageDetail = 'rpm -qi'
        packageRunTime = 'rocal-*'

        if (platform.jenkinsLabel.contains('sles')) {
            osType = 'sles'
        }
        else if (platform.jenkinsLabel.contains('centos7')) {
            osType = 'centos7'
        }
        else if (platform.jenkinsLabel.contains('rhel8')) {
            osType = 'rhel8'
        }
        else if (platform.jenkinsLabel.contains('rhel9')) {
            osType = 'rhel9'
        }
    }
    else
    {
        packageType = 'deb'
        packageInfo = 'dpkg -c'
        packageDetail = 'dpkg -I'
        packageRunTime = 'rocal_*'

        if (platform.jenkinsLabel.contains('ubuntu20')) {
            osType = 'ubuntu20'
        }
        else if (platform.jenkinsLabel.contains('ubuntu22')) {
            osType = 'ubuntu22'
        }
    }

    def command = """#!/usr/bin/env bash
                set -x
                export HOME=/home/jenkins
                echo Make rocal Package
                cd ${project.paths.project_build_prefix}/build/release
                sudo make package
                mkdir -p package
                mv rocal-test*.${packageType} package/${osType}-rocal-test.${packageType}
                mv rocal-dev*.${packageType} package/${osType}-rocal-dev.${packageType}
                mv ${packageRunTime}.${packageType} package/${osType}-rocal.${packageType}
                mv Testing/Temporary/LastTest.log ${osType}-LastTest.log
                mv Testing/Temporary/LastTestsFailed.log ${osType}-LastTestsFailed.log
                ${packageDetail} package/${osType}-rocal-test.${packageType}
                ${packageDetail} package/${osType}-rocal-dev.${packageType}
                ${packageDetail} package/${osType}-rocal.${packageType}
                ${packageInfo} package/${osType}-rocal-test.${packageType}
                ${packageInfo} package/${osType}-rocal-dev.${packageType}
                ${packageInfo} package/${osType}-rocal.${packageType}
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
