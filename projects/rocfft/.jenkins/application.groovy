#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI = 
{
    nodeDetails, jobName->

    def prj  = new rocProject('rocFFT-internal', 'application')

    prj.defaults.ccache = true
    prj.timeout.compile = 600
    prj.timeout.test = 600
    prj.libraryDependencies = ['rocFFT', 'hipFFT']

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    def commonGroovy

    def compileCommand =
    {
        platform, project->
        def getDependenciesCommand = ""
        if (project.installLibraryDependenciesFromCI)
        {
            project.libraryDependencies.each
            { libraryName ->
                getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, null, false)
            }
        } 

        def command = """#!/usr/bin/env bash
                         set -ex
                         cd ${project.paths.project_build_prefix}
                         ${getDependenciesCommand}
                         git clone -b develop-2021 https://github.com/ROCmSoftwarePlatform/Gromacs.git
                         cd Gromacs
                         
                         mkdir build_tmpi
                         cd build_tmpi
                         cmake -DCMAKE_HIP_ARCHITECTURES=gfx90a -DBUILD_SHARED_LIBS=ON -DGMX_BUILD_FOR_COVERAGE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DGMX_MPI=OFF -DGMX_GPU=hip -DGMX_OPENMP=ON -DGMX_SIMD=AVX2_256 -DREGRESSIONTEST_DOWNLOAD=OFF -DGMX_GPU_USE_VKFFT=OFF -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=../gromacs-install ..
                         make
                         make install
                         cd ..

                         mkdir build_mpi
                         cd build_mpi
                         cmake -DCMAKE_HIP_ARCHITECTURES=gfx908 -DBUILD_SHARED_LIBS=ON -DGMX_BUILD_FOR_COVERAGE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpic++ -DGMX_MPI=ON -DGMX_GPU=hip -DGMX_OPENMP=ON -DGMX_SIMD=AVX2_256 -DREGRESSIONTEST_DOWNLOAD=OFF -DGMX_GPU_USE_VKFFT=OFF -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=../gromacs-install ..
                         make
                         make install
                         cd ..
                      """
        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->
        
        def command = """#!/usr/bin/env bash
                         set -ex
                         cd ${project.paths.project_build_prefix}
                         cd Gromacs

                         source gromacs-install/bin/GMXRC
                         gmx --version

                         export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib
                         echo \$LD_LIBRARY_PATH

                         git clone https://github.com/jychang48/benchmark-gromacs.git
                         cd benchmark-gromacs

                         export GMX_MAXBACKUP=-1

                         echo "* Threaded MPI ******************************************************************************************************"

                         #ADH_DODEC
                         cd adh_dodec
                         tar zxf adh_dodec.tar.gz
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 1 -ntomp 64 -noconfout -nb gpu -bonded gpu -pme gpu -v -gpu_id 0 -s topol.tpr -nstlist 100               # 1 GPU
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 4 -ntomp 16 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01 -s topol.tpr -nstlist 200      # 2 GPUs   
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 4 -ntomp 16 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 0123 -s topol.tpr -nstlist 200    # 4 GPUs
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 8 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01234567 -s topol.tpr -nstlist 150 # 8 GPUs
                         
                         # STMV
                         cd ..
                         cd stmv/
                         tar zxf stmv.tar.gz
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 1 -ntomp 64 -noconfout -nb gpu -bonded gpu -pme gpu -v -gpu_id 0 -s topol.tpr -nstlist 200               # 1 GPU
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 4 -ntomp 16 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01 -s topol.tpr -nstlist 200      # 2 GPUs
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 8 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 0123 -s topol.tpr -nstlist 400     # 4 GPUs
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 8 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01234567 -s topol.tpr -nstlist 400 # 8 GPUs
 
                         # CELLULOSE_NVE
                         cd ..
                         cd cellulose_nve/
                         tar zxf cellulose_nve.tar.gz
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 1 -ntomp 64 -noconfout -nb gpu -bonded gpu -pme gpu -v -gpu_id 0 -s topol.tpr -nstlist 100               # 1 GPU
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 4 -ntomp 16 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01 -s topol.tpr -nstlist 200      # 2 GPUs
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 8 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 0123 -s topol.tpr -nstlist 200     # 4 GPUs
                         gmx --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 8 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01234567 -s topol.tpr -nstlist 200 # 8 GPUs

                         echo "* MPI ***************************************************************************************************************" 
 
                         # ADH_DODEC
                         cd ..
                         cd adh_dodec/
                         tar zxf adh_dodec.tar.gz
                         mpirun -np 1 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 64 -noconfout -nb gpu -bonded gpu -pme gpu -v -gpu_id 0 -s topol.tpr                # 1 GPU
                         mpirun -np 4 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01 -s topol.tpr        # 2 GPUs
                         mpirun -np 8 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 6 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 0123 -s topol.tpr      # 4 GPUs
                         mpirun -np 8 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 6 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01234567 -s topol.tpr  # 8 GPUs
 
                         # STMV
                         cd ..
                         cd stmv/
                         tar zxf stmv.tar.gz
                         mpirun -np 1 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 64 -noconfout -nb gpu -bonded gpu -pme gpu -v -nstlist 400 -gpu_id 0 -s topol.tpr   # 1 GPU
                         mpirun -np 4 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01 -s topol.tpr        # 2 GPUs
                         mpirun -np 8 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 0123 -s topol.tpr      # 4 GPUs
                         mpirun -np 8 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01234567 -s topol.tpr  # 8 GPUs
 
                         # CELLULOSE_NVE
                         cd ..
                         cd cellulose_nve/
                         tar zxf cellulose_nve.tar.gz
                         mpirun -np 1 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 64 -noconfout -nb gpu -bonded gpu -pme gpu -v -gpu_id 0 -s topol.tpr                # 1 GPU
                         mpirun -np 4 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01 -s topol.tpr        # 2 GPUs
                         mpirun -np 8 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 6 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 0123 -s topol.tpr      # 4 GPUs
                         mpirun -np 8 gmx_mpi --quiet mdrun -pin on -nsteps 100000 -resetstep 90000 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -gpu_id 01234567 -s topol.tpr  # 8 GPUs
                      """
        platform.runCommand(this, command)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)
}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 5')])]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi-hipclang":([ubuntu20:['8gfx90a']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each 
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    jobNameList.each 
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(jobName) {
                runCI(nodeDetails, jobName)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(urlJobName) {
            runCI([ubuntu18:['8gfx90a']], urlJobName)
        }
    }
}
