---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Install '...' version '...'
2. Run '...' with data '...'
3. See error on logfile '...'

**Expected behavior**
A clear and concise description of what you expected to happen.

**Log-files**
Add *full* logfiles to help explain your problem.

**Environment**
Make sure that ROCm is correctly installed and run the following command:
```
printf '=== environment\n' > environment.txt && 
printf '\n\n=== date\n' >> environment.txt && date >> environment.txt && 
printf '\n\n=== Linux Kernel\n' >> environment.txt && uname -a  >> environment.txt && 
printf '\n\n=== rocm-smi' >> environment.txt && rocm-smi  >> environment.txt && 
printf '\n\n' >> environment.txt && hipconfig  >> environment.txt && 
printf '\n\n=== rocminfo\n' >> environment.txt && rocminfo  >> environment.txt && 
printf '\n\n=== lspci VGA\n' >> environment.txt && lspci | grep -i vga >> environment.txt
```

Attach `environment.txt`

**Additional context**
Add any other context about the problem here.
