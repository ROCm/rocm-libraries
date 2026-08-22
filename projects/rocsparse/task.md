* implement os-agnostic interface to read environment variables
* for windows use WinAPI
* for linux usw Posix API
* ignore other operating systems then windows and linux
* used created api to replace getenv and std::getenv in the rocsparse project
