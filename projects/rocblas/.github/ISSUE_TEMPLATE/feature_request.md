---
name: Feature request
about: Suggest an idea for this project
title: "[Feature]: "
labels: ["feature", "triage"]

---

### Is your feature request related to a problem? Please describe.
A clear and concise description of what the problem is.

### Describe the solution you'd like
A clear and concise description of what you want to happen.

### Describe alternatives you've considered
A clear and concise description of any alternative solutions or features you've considered.

### Additional context
Add any other context or screenshots about the feature request here.

### Library context
| Software | version |
|-----|-----|
| rocblas | v0.0 |

The above Table information can be queried with:
```
Ubuntu/Debian:
dpkg -s rocblas | grep Version
Centos/RHEL:
rpm -qa | grep rocblas
SLES:
zypper se -s | grep rocblas
```
