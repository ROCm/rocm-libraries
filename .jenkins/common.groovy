// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false) {
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    
    def command = """#!/usr/bin/env bash
                set -ex
                echo Build rocJPEG - ${buildTypeDir}
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" ../..
                make -j\$(nproc)
                sudo make install
                sudo make package
                objdump -x /opt/rocm/lib/librocjpeg.so | grep NEEDED
                ldd -v /opt/rocm/lib/librocjpeg.so
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

    withCredentials([string(credentialsId: "mathlibs-codecov-token-rocjpeg", variable: 'CODECOV_TOKEN')])
    {
        def command = """#!/usr/bin/env bash
                    set -ex
                    export HOME=/home/jenkins
                    ${libvaDriverPath}
                    echo make test
                    cd ${project.paths.project_build_prefix}/build
                    export LLVM_PROFILE_FILE=\"\$(pwd)/rawdata/rocjpeg-%p.profraw\"
                    echo \$LLVM_PROFILE_FILE
                    cd release
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} make test ARGS="-VV --rerun-failed --output-on-failure"
                    echo rocjpeg-sample - jpegDecode
                    mkdir -p rocjpeg-sample && cd rocjpeg-sample
                    cmake /opt/rocm/share/rocjpeg/samples/jpegDecode/
                    make -j8
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./jpegdecode -i /opt/rocm/share/rocjpeg/images/
                    echo rocjpeg additional tests
                    wget http://math-ci.amd.com/userContent/computer-vision/rocJPEG/jpeg_samples.zip
                    unzip jpeg_samples.zip
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./jpegdecode -i jpeg_samples
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./jpegdecode -i jpeg_samples -fmt yuv_planar
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./jpegdecode -i jpeg_samples -fmt y
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./jpegdecode -i jpeg_samples -fmt rgb
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./jpegdecode -i jpeg_samples -fmt rgb_planar
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./jpegdecode -i jpeg_samples -crop 0,0,100,100
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./jpegdecode -i jpeg_samples -fmt yuv_planar -crop 0,0,100,100
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./jpegdecode -i jpeg_samples -fmt y -crop 0,0,100,100
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./jpegdecode -i jpeg_samples -fmt rgb -crop 0,0,100,100
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ./jpegdecode -i jpeg_samples -fmt rgb_planar -crop 0,0,100,100
                    echo rocjpeg-test package verification
                    cd ../ && mkdir -p rocjpeg-test && cd rocjpeg-test
                    cmake /opt/rocm/share/rocjpeg/test/
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ctest -VV --rerun-failed --output-on-failure
                    cd  ../../
                    echo \$(pwd)
                    sudo ${packageManager} install lcov ${toolsPackage}
                    ${llvmLocation}/llvm-profdata merge -sparse rawdata/*.profraw -o rocjpeg.profdata
                    ${llvmLocation}/llvm-cov export -object release/lib/librocjpeg.so --instr-profile=rocjpeg.profdata --format=lcov > coverage.info
                    lcov --remove coverage.info '/opt/*' --output-file coverage.info
                    lcov --list coverage.info
                    lcov --summary  coverage.info
                    curl -Os https://uploader.codecov.io/latest/linux/codecov
                    chmod +x codecov
                    ./codecov -v -U \$http_proxy -t ${CODECOV_TOKEN} --file coverage.info --name rocJPEG --sha ${commitSha}
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
        packageRunTime = 'rocjpeg-*'

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
        packageRunTime = 'rocjpeg_*'

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
                echo Make rocJPEG Package
                cd ${project.paths.project_build_prefix}/build/release
                sudo make package
                mkdir -p package
                mv rocjpeg-dev*.${packageType} package/${osType}-rocjpeg-dev.${packageType}
                mv rocjpeg-test*.${packageType} package/${osType}-rocjpeg-test.${packageType}
                mv ${packageRunTime}.${packageType} package/${osType}-rocjpeg.${packageType}
                ${packageDetail} package/${osType}-rocjpeg-dev.${packageType}
                ${packageDetail} package/${osType}-rocjpeg-test.${packageType}
                ${packageDetail} package/${osType}-rocjpeg.${packageType}
                ${packageInfo} package/${osType}-rocjpeg-dev.${packageType}
                ${packageInfo} package/${osType}-rocjpeg-test.${packageType}
                ${packageInfo} package/${osType}-rocjpeg.${packageType}
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
