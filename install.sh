#!/bin/bash

# Get dependent parameters
source "$(dirname "$(readlink -f "${0}")")/get_param.sh"

# Change directory to [graspnetAPI] and install the package
cd ../${WS_PATH}/graspnetAPI
sudo pip3 install .

# Compile and install [pointnet2]
cd ../${WS_PATH}/graspnet-baseline/pointnet2
sudo pip3 install .

# Compile and install [knn]
cd ../${WS_PATH}/graspnet-baseline/knn
sudo pip3 install .
