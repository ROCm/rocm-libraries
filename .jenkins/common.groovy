// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false) {
    project.paths.construct_build_prefix()

    String libLocation = ''
    if (platform.jenkinsLabel.contains('rhel')) {
        libLocation = ':/usr/local/lib:/usr/local/lib/x86_64-linux-gnu'
    }
    else if (platform.jenkinsLabel.contains('sles')) {
        libLocation = ':/usr/local/lib:/usr/local/lib/x86_64-linux-gnu'
    }

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'

    def command = """#!/usr/bin/env bash
                set -ex
                export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation}
                echo Build rocPyDecode - ${buildTypeDir}
                cd ${project.paths.project_build_prefix}
                sudo python3 rocPyDecode-requirements.py
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" ../..
                make -j\$(nproc)
                sudo make install
                ldd -v /opt/rocm/lib/rocpydecode.*.so
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

    withCredentials([string(credentialsId: "mathlibs-codecov-token-rocpydecode", variable: 'CODECOV_TOKEN')])
    {
        def command = """#!/usr/bin/env bash
                    export HOME=/home/jenkins
                    set -x
                    cd ${project.paths.project_build_prefix}/build
                    mkdir -p test && cd test
                    export LLVM_PROFILE_FILE=\"\$(pwd)/rawdata/rocpydecode-%p.profraw\"
                    echo \$LLVM_PROFILE_FILE
                    cmake /opt/rocm/share/rocpydecode/tests/
                    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib${libLocation} ctest -VV --rerun-failed --output-on-failure
                    sudo ${packageManager} install lcov ${toolsPackage}
                    ${llvmLocation}/llvm-profdata merge -sparse rawdata/*.profraw -o rocpydecode.profdata
                    ${llvmLocation}/llvm-cov export -object ../release/rocpydecode/lib/*.so --instr-profile=rocpydecode.profdata --format=lcov > coverage.info
                    lcov --remove coverage.info '/opt/*' --output-file coverage.info
                    lcov --list coverage.info
                    lcov --summary  coverage.info
                    curl -Os https://uploader.codecov.io/latest/linux/codecov
                    chmod +x codecov
                    ./codecov -v -U \$http_proxy -t ${CODECOV_TOKEN} --file coverage.info --name rocpydecode --sha ${commitSha}
                    """

        platform.runCommand(this, command)
    }
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
        packageRunTime = 'rocpydecode_*'

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
        packageRunTime = 'rocpydecode_*'

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
                echo Make rocPyDecode Package
                cd ${project.paths.project_build_prefix}/build/release
                sudo make package
                mkdir -p package
                mv rocpydecode-test*.${packageType} package/${osType}-rocpydecode-test.${packageType}
                mv ${packageRunTime}.${packageType} package/${osType}-rocpydecode.${packageType}
                mv Testing/Temporary/LastTest.log ${osType}-LastTest.log
                mv Testing/Temporary/LastTestsFailed.log ${osType}-LastTestsFailed.log
                ${packageDetail} package/${osType}-rocpydecode-test.${packageType}
                ${packageDetail} package/${osType}-rocpydecode.${packageType}
                ${packageInfo} package/${osType}-rocpydecode-test.${packageType}
                ${packageInfo} package/${osType}-rocpydecode.${packageType}
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, packageHelper[1])
}


return this
