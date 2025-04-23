// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false) {
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    
    def command = """#!/usr/bin/env bash
                set -ex
                echo Build rocDecode - ${buildTypeDir}
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" ../..
                make -j\$(nproc)
                sudo make install
                sudo make package
                objdump -x /opt/rocm/lib/librocdecode.so | grep NEEDED
                ldd -v /opt/rocm/lib/librocdecode.so
                """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project) {

    String libLocation = ''
    String libvaDriverPath = ""
    String packageManager = 'apt -y'
    String toolsPackage = 'llvm-amdgpu-dev'
    String llvmLocation = '/opt/amdgpu/lib/x86_64-linux-gnu/llvm-20.1/bin'

    if (platform.jenkinsLabel.contains('rhel')) {
        libLocation = ':/usr/local/lib'
        packageManager = 'yum -y'
        toolsPackage = 'llvm-amdgpu-devel'
        llvmLocation = '/opt/amdgpu/lib64/llvm-20.1/bin'
    }
    else if (platform.jenkinsLabel.contains('sles')) {
        libLocation = ':/usr/local/lib'
        libvaDriverPath = "export LIBVA_DRIVERS_PATH=/opt/amdgpu/lib64/dri"
        packageManager = 'zypper -n'
        toolsPackage = 'llvm-amdgpu-devel'
        llvmLocation = '/opt/amdgpu/lib64/llvm-20.1/bin'
    }
    
    String commitSha
    String repoUrl
    (commitSha, repoUrl) = util.getGitHubCommitInformation(project.paths.project_src_prefix)

    withCredentials([string(credentialsId: "mathlibs-codecov-token-rocdecode", variable: 'CODECOV_TOKEN')])
    {
        def prereq = """
                    if [ -d "\${JENKINS_HOME_DIR}/rocDecode" ]; then
                        # Count the number of files in the folder
                        FILE_COUNT=\$(find \${JENKINS_HOME_DIR}/rocDecode/AvcConformance -type f | wc -l)
                        # Check if there are 254 files
                        if [ "\$FILE_COUNT" -ne 254 ]; then
                            echo "wrong file count"
                            ls
                            cd \${JENKINS_HOME_DIR}/rocDecode
                            wget http://math-ci.amd.com/userContent/computer-vision/rocDecodeConformance/AvcConformance.zip
                            unzip AvcConformance.zip
                        fi
                        FILE_COUNT=\$(find \${JENKINS_HOME_DIR}/rocDecode/Av1Conformance_v1.0 -type f | wc -l)
                        # Check if there are 326 files
                        if [ "\$FILE_COUNT" -ne 326 ]; then
                            echo "wrong file count"
                            ls
                            cd \${JENKINS_HOME_DIR}/rocDecode
                            wget http://math-ci.amd.com/userContent/computer-vision/rocDecodeConformance/Av1Conformance_v1.0.zip
                            unzip Av1Conformance_v1.0.zip
                        fi
                        FILE_COUNT=\$(find \${JENKINS_HOME_DIR}/rocDecode/Vp9Conformance -type f | wc -l)
                        # Check if there are 216 files
                        if [ "\$FILE_COUNT" -ne 216 ]; then
                            echo "wrong file count"
                            ls
                            cd \${JENKINS_HOME_DIR}/rocDecode
                            wget http://math-ci.amd.com/userContent/computer-vision/rocDecodeConformance/Vp9Conformance.zip
                            unzip Vp9Conformance.zip
                        fi
                        FILE_COUNT=\$(find \${JENKINS_HOME_DIR}/rocDecode/HevcConformance -type f | wc -l)
                        # Check if there are 270 files
                        if [ "\$FILE_COUNT" -ne 270 ]; then
                            echo "wrong file count"
                            ls                            
                            cd \${JENKINS_HOME_DIR}/rocDecode
                            wget http://math-ci.amd.com/userContent/computer-vision/HevcConformance/*zip*/HevcConformance.zip
                            unzip HevcConformance.zip
                        fi
                        if [ ! -f \${JENKINS_HOME_DIR}/rocDecode/data1.img ]; then
                            echo "File does not exist."
                            cd \${JENKINS_HOME_DIR}/rocDecode
                            wget http://math-ci.amd.com/userContent/computer-vision/data1.img
                        fi
                    else
                        echo "The folder path does not exist."
                        mkdir -p \${JENKINS_HOME_DIR}/rocDecode
                        cd \${JENKINS_HOME_DIR}/rocDecode
                        wget http://math-ci.amd.com/userContent/computer-vision/data1.img
                        wget http://math-ci.amd.com/userContent/computer-vision/HevcConformance/*zip*/HevcConformance.zip
                        wget http://math-ci.amd.com/userContent/computer-vision/rocDecodeConformance/Vp9Conformance.zip
                        wget http://math-ci.amd.com/userContent/computer-vision/rocDecodeConformance/Av1Conformance_v1.0.zip
                        wget http://math-ci.amd.com/userContent/computer-vision/rocDecodeConformance/AvcConformance.zip
                        unzip HevcConformance.zip
                        unzip Vp9Conformance.zip
                        unzip Av1Conformance_v1.0.zip
                        unzip AvcConformance.zip
                    fi
        """
        def command = """#!/usr/bin/env bash
                    set -ex
                    export HOME=/home/jenkins
                    ${libvaDriverPath}
                    echo make test
                    cd ${project.paths.project_build_prefix}/build
                    export LLVM_PROFILE_FILE=\"\$(pwd)/rawdata/rocdecode-%p.profraw\"
                    echo \$LLVM_PROFILE_FILE
                    cd release
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} make test ARGS="-VV --rerun-failed --output-on-failure"
                    echo rocdecode-sample - videoDecode
                    mkdir -p rocdecode-sample && cd rocdecode-sample
                    cmake /opt/rocm/share/rocdecode/samples/videoDecode/
                    make -j8
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./videodecode -i /opt/rocm/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4
                    echo rocdecode-test package verification
                    cd ../ && mkdir -p rocdecode-test && cd rocdecode-test
                    cmake /opt/rocm/share/rocdecode/test/
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ctest -VV --rerun-failed --output-on-failure
                    echo rocdecode conformance tests
                    cd ../ && mkdir -p conformance && cd conformance
                    pip3 install pandas
                    mkdir hevc-conformance && cd hevc-conformance
                    python3 /opt/rocm/share/rocdecode/test/testScripts/run_rocDecode_Conformance.py --videodecode_exe ./../../rocdecode-sample/videodecode --files_directory \${JENKINS_HOME_DIR}/rocDecode/HevcConformance --results_directory .
                    cd ../
                    mkdir avc-conformance && cd avc-conformance
                    python3 /opt/rocm/share/rocdecode/test/testScripts/run_rocDecode_Conformance.py --videodecode_exe ./../../rocdecode-sample/videodecode --files_directory \${JENKINS_HOME_DIR}/rocDecode/AvcConformance --results_directory .
                    cd ../
                    mkdir vp9-conformance && cd vp9-conformance
                    python3 /opt/rocm/share/rocdecode/test/testScripts/run_rocDecode_Conformance.py --videodecode_exe ./../../rocdecode-sample/videodecode --files_directory \${JENKINS_HOME_DIR}/rocDecode/Vp9Conformance --results_directory .
                    cd ../
                    mkdir av1-conformance && cd av1-conformance
                    python3 /opt/rocm/share/rocdecode/test/testScripts/run_rocDecode_Conformance.py --videodecode_exe ./../../rocdecode-sample/videodecode --files_directory \${JENKINS_HOME_DIR}/rocDecode/Av1Conformance_v1.0 --results_directory .
                    cd ../../
                    echo rocdecode-sample - videoDecode with data1 video test
                    cd rocdecode-sample
                    cp \${JENKINS_HOME_DIR}/rocDecode/data1.img \$PWD
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./videodecode -i ./data1.img
                    echo rocdecode-sample - videoDecodePerf with data1 video test
                    mkdir -p rocdecode-perf && cd rocdecode-perf
                    cmake /opt/rocm/share/rocdecode/samples/videoDecodePerf/
                    make -j8
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./videodecodeperf -i ./../data1.img
                    echo \$(pwd)
                    cd  ../../../
                    echo \$(pwd)
                    sudo ${packageManager} install lcov ${toolsPackage}
                    ${llvmLocation}/llvm-profdata merge -sparse rawdata/*.profraw -o rocdecode.profdata
                    ${llvmLocation}/llvm-cov export -object release/lib/librocdecode.so --instr-profile=rocdecode.profdata --format=lcov > coverage.info
                    lcov --remove coverage.info '/opt/*' --output-file coverage.info
                    lcov --list coverage.info
                    lcov --summary  coverage.info
                    curl -Os https://uploader.codecov.io/latest/linux/codecov
                    chmod +x codecov
                    ./codecov -v -U \$http_proxy -t ${CODECOV_TOKEN} --file coverage.info --name rocDecode --sha ${commitSha}
                    """
        platform.runCommand(this, prereq)
        platform.runCommand(this, command)
    }
}

def runPackageCommand(platform, project) {

    def packageHelper = platform.makePackage(platform.jenkinsLabel, "${project.paths.project_build_prefix}/build/release")

    String packageType = ''
    String packageInfo = ''
    String packageDetail = ''
    String packageInstall = ''
    String osType = ''
    String packageRunTime = ''

    if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('rhel') || platform.jenkinsLabel.contains('sles')) {
        packageType = 'rpm'
        packageInfo = 'rpm -qlp'
        packageDetail = 'rpm -qi'
        packageInstall = 'rpm -i'
        packageRunTime = 'rocdecode-*'

        if (platform.jenkinsLabel.contains('sles')) {
            osType = 'sles'
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
        packageInstall = 'dpkg -i'
        packageRunTime = 'rocdecode_*'

        if (platform.jenkinsLabel.contains('ubuntu20')) {
            osType = 'ubuntu20'
        }
        else if (platform.jenkinsLabel.contains('ubuntu22')) {
            osType = 'ubuntu22'
        }
    }

    def command = """#!/usr/bin/env bash
                set -ex
                export HOME=/home/jenkins
                echo Make rocDecode Package
                cd ${project.paths.project_build_prefix}/build/release
                sudo make package
                mkdir -p package
                mv rocdecode-dev*.${packageType} package/${osType}-rocdecode-dev.${packageType}
                mv rocdecode-test*.${packageType} package/${osType}-rocdecode-test.${packageType}
                mv ${packageRunTime}.${packageType} package/${osType}-rocdecode.${packageType}
                ${packageDetail} package/${osType}-rocdecode-dev.${packageType}
                ${packageDetail} package/${osType}-rocdecode-test.${packageType}
                ${packageDetail} package/${osType}-rocdecode.${packageType}
                ${packageInfo} package/${osType}-rocdecode-dev.${packageType}
                ${packageInfo} package/${osType}-rocdecode-test.${packageType}
                ${packageInfo} package/${osType}-rocdecode.${packageType}
                sudo ${packageInstall} package/${osType}-rocdecode.${packageType}
                sudo ${packageInstall} package/${osType}-rocdecode-dev.${packageType}
                sudo ${packageInstall} package/${osType}-rocdecode-test.${packageType}
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
