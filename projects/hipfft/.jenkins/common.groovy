import static groovy.io.FileType.FILES

def runCompileCommand(platform, project, jobName, boolean sameOrg = false)
{
    project.paths.construct_build_prefix()

    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, null, sameOrg)
        }
    }

    String cmake = platform.jenkinsLabel.contains('centos') ? "cmake3" : "cmake" 
    String warningArgs = platform.jenkinsLabel.contains('cuda') ? '':'-DWERROR=ON'
    String hipClang = platform.jenkinsLabel.contains('hipClang') ? "HIP_COMPILER=clang" : ""
    String path = platform.jenkinsLabel.contains('centos7') ? "export PATH=/opt/rh/devtoolset-7/root/usr/bin:$PATH" : ":"
    String dir = jobName.contains('Debug') ? "debug" : "release"

    // hipcc with CUDA backend needs HIP_PLATFORM set accordingly in the environment
    String hipPlatformCommand = platform.jenkinsLabel.contains("cuda") ? "export HIP_PLATFORM=nvidia" : ""

    def command = """#!/usr/bin/env bash
                set -x
                
                ls /fftw/lib
                export FFTW_ROOT=/fftw
                export FFTW_INCLUDE_PATH=\${FFTW_ROOT}/include
                export FFTW_LIB_PATH=\${FFTW_ROOT}/lib
                export LD_LIBRARY_PATH=\${FFTW_LIB_PATH}:/opt/rocm/lib:/opt/rocm/hip/lib
                export CPLUS_INCLUDE_PATH=\${FFTW_INCLUDE_PATH}:\${CPLUS_INCLUDE_PATH}
                export CMAKE_PREFIX_PATH=\${FFTW_LIB_PATH}/cmake/fftw3:\${CMAKE_PREFIX_PATH}
                export CMAKE_PREFIX_PATH=\${FFTW_LIB_PATH}/cmake/fftw3f:\${CMAKE_PREFIX_PATH}
		# default container flags cause problems for CUDA backend, and aren't useful for ROCm
		unset HIPCC_COMPILE_FLAGS_APPEND
		unset HIPCC_LINK_FLAGS_APPEND
		${hipPlatformCommand}
                
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${dir} && cd build/${dir}
                ${getDependenciesCommand}
                ${path}
                ${hipClang} ${cmake} ${warningArgs} ${project.paths.build_command}
                make -j\$(nproc)
            """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter)
{
    String cudaArgs = platform.jenkinsLabel.contains('cuda') ? '--double_epsilon=5e-11'
                        : '--precompile=rocfft-test-precompile.db'

    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release/clients/staging
                    GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipfft-test ${cudaArgs} --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}
                """

    platform.runCommand(this, command)
    //junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
}

def runPackageCommand(platform, project, jobName, label='')
{
    def command

    label = label != '' ? '-' + label.toLowerCase() : ''
    String ext = platform.jenkinsLabel.contains('ubuntu') ? "deb" : "rpm"
    String dir = jobName.contains('Debug') ? "debug" : "release"

    command = """
            set -x
            cd ${project.paths.project_build_prefix}/build/${dir}
            make package
            mkdir -p package
            for f in hipfft*.$ext
            do 
                mv "\$f" "hipfft${label}-\${f#*-}"
            done
            mv *.${ext} package/
        """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/${dir}/package/*.${ext}""")
}


return this
