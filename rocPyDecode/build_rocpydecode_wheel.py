import subprocess
import os
import shutil

# Clean any existing build/dist files
if os.path.exists('build'):
    shutil.rmtree('build')
if os.path.exists('dist'):
    shutil.rmtree('dist')

# Build the source distribution and wheel
subprocess.run(['sudo', 'python3', 'setup.py', 'bdist_wheel'], check=True) # build wheel & install it

# Verify the content of the created wheel
wheel_dir = os.path.join('dist')
wheel_files = os.listdir(wheel_dir)

print(f"rocPyDecode wheel files created: {wheel_files}\n")