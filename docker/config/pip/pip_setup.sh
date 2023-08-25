#!/usr/bin/env bash

file_dir=$(dirname "$(readlink -f "${0}")")
python3 -m pip install --upgrade --force-reinstall pip \
&& pip3 install --ignore-installed -r "${file_dir}"/requirements.txt