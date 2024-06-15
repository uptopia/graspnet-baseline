#!/bin/bash

# Get dependent parameters
# source "$(dirname "$(readlink -f "${0}")")/get_param.sh"
# echo ${WS_PATH}

WS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "WS directory: $WS_DIR"

# Compile and install [pointnet2]
cd ${WS_DIR}/pointnet2
pip3 install .

# Compile and install [knn]
cd ${WS_DIR}/knn
pip3 install .

# Change directory to [graspnetAPI] and install the package
cd ${WS_DIR}/graspnetAPI
sudo pip3 install .
