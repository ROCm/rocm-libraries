# Normalize paths
s:^/\(.*\)/rocroller/:/data/:
# Remove timestamp
/^Created: .*$/d
# Delete line numbers, prepend template instantiations with |||
s/^\s*\(\(|\)\s*\)\?[0-9]*|/\2\2\2/
# Non-covered lines
s/^\(|||\)\?\s\+0|/\1|-ZERO-|/
# <10 hit lines
s/^\(|||\)\?\s*\([0-9]\)|/\1|     \2|/
# >=10 hit lines
s/^\(|||\)\?\s*\([0-9\.]\+\)|/\1| >=10 |/
# >=1k hit lines
s/^\(|||\)\?\s*\([0-9\.]\+\)\([kMGE]\)|/\1| >=1\3 |/
