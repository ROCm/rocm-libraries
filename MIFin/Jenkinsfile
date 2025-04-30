

def rocmnode(name) {
    def node_name = 'tunatest'
    if(name == 'fiji') {
        node_name = 'tunatest && fiji';
    } else if(name == 'vega') {
        node_name = 'tunatest && vega';
    } else if(name == 'vega10') {
        node_name = 'tunatest && vega10';
    } else if(name == 'vega20') {
        node_name = 'tunatest && vega20';
    } else if(name == 'gfx908') {
        node_name = 'gfx908';
    } else {
        node_name = name
    }
    return node_name
}



def cmake_build(compiler, flags, prefixpath="/opt/rocm"){
    def workspace_dir = pwd()
    def archive = (flags == '-DCMAKE_BUILD_TYPE=release')
    def config_targets = "all" 
    def test_flags = "--disable-verification-cache"
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined"
    def compilerpath = ""
    def configargs = ""
    if (prefixpath == "")
        compilerpath = compiler;
    else
    {
        compilerpath = prefixpath + "/bin/" + compiler
        configargs = "-DCMAKE_PREFIX_PATH=${prefixpath}"
    }

    if (archive == true) {
        config_targets = "package"
    }
    def cmd = """
        echo \$HSA_ENABLE_SDMA
        ulimit -c unlimited
        rm -rf build
        mkdir build
        cd build
        CXX=${compilerpath} cmake ${configargs}  ${flags} ..
        dumb-init make -j\$(nproc) ${config_targets}
    """
    echo cmd
    sh cmd
    // Only archive from master or develop
    if (archive == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master")) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
}

def buildJob(compiler, flags, image, prefixpath="/opt/rocm", cmd = ""){

        env.HSA_ENABLE_SDMA=0
        checkout scm
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --user=root --privileged "
        def dockerArgs = "--build-arg PREFIX=${prefixpath} "
        if(prefixpath == "")
        {
            dockerArgs = ""
        }
        def date = sh(script: 'date +"%m%d%H"', returnStdout: true).trim()
        dockerArgs += " --build-arg CACHE_DATE=${date} "
        def retimage
        try {
            echo "build docker"
            retimage = docker.build("${image}", dockerArgs + ' .')
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                }
            }
        } catch(Exception ex) {
            echo "exception ocurred"
            retimage = docker.build("${image}", dockerArgs + " --no-cache .")
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 5, unit: 'MINUTES')
                {
                    sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64/:$PATH" clinfo'
                }
            }
        }

        withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
            timeout(time: 5, unit: 'HOURS')
            {
                if(cmd == ""){
                    cmake_build(compiler, flags, prefixpath)
                }else{
                    echo "run shell command"
                    sh cmd
                }
            }
        }
        return retimage
}



pipeline {
    agent none
    options {
        parallelsAlwaysFailFast()
    }
    environment{
        image = "fin"
    }
    stages{
        // Run all static analysis tests
        stage("Static checks"){
            parallel{
                stage('Clang Format') {
                    agent{ label rocmnode("tunatest") }
                    environment{
                        cmd = "cd src; find . -iname \'*.h\' \
                                -o -iname \'*.hpp\' \
                                -o -iname \'*.cpp\' \
                                -o -iname \'*.h.in\' \
                                -o -iname \'*.hpp.in\' \
                                -o -iname \'*.cpp.in\' \
                                -o -iname \'*.cl\' \
                                | grep -v 'build/' \
                                | grep -v 'base64' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-12 -style=file {} | diff - {}\'"
                    }
                    steps{
                        buildJob('clang++', '-DCMAKE_BUILD_TYPE=release', image, "", cmd)
                    }
                }

                stage('Hip Tidy') {
                    agent{ label rocmnode("tunatest") }
                    environment{
                        cmd = "rm -rf build; \
                                mkdir build; \
                                cd build; \
                                CXX=/opt/rocm/llvm/bin/clang++ cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DBUILD_DEV=On ..; \
                                make -j\$(nproc) -k analyze;"
                    }
                    steps{
                        buildJob('clang++', '-DCMAKE_BUILD_TYPE=release', image, "", cmd)
                    }
                }
                stage('Build Fin') {
                    agent{ label rocmnode("tunatest") }
                    environment{
                        cmd = "rm -rf build; \
                                mkdir build; \
                                cd build; \
                                CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_PREFIX_PATH=/opt/rocm ..; \
                                make -j\$(nproc) all;"
                    }
                    steps{
                        buildJob('clang++', '-DCMAKE_BUILD_TYPE=release', image, "", cmd)
                    }
                }
                stage('Fin Tests') {
                    agent{ label rocmnode("tunatest") }
                    environment{
                        cmd = "rm -rf build; \
                                mkdir build; \
                                cd build; \
                                CXX=/opt/rocm/llvm/bin/clang++ cmake -DBUILD_DEV=On -DCMAKE_PREFIX_PATH=/opt/rocm ..; \
                                make -j\$(nproc) fin_check;"
                    }
                    steps{
                        buildJob('clang++', '-DCMAKE_BUILD_TYPE=release', image, "", cmd)
                    }
                }
            }
        }



    }
}

