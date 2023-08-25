
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

############################## SYSTEM PARAMETERS ##############################
# * Arguments
ARG USER=initial
ARG GROUP=initial
ARG UID=1000
ARG GID="${UID}"
ARG SHELL=/bin/bash
ARG HARDWARE=x86_64
ARG ENTRYPOINT_FILE=entrypint.sh

# # * Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
# ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# * Setup users and groups
RUN groupadd --gid "${GID}" "${GROUP}" \
    && useradd --gid "${GID}" --uid "${UID}" -ms "${SHELL}" "${USER}" \
    && mkdir -p /etc/sudoers.d \
    && echo "${USER}:x:${UID}:${UID}:${USER},,,:/home/${USER}:${SHELL}" >> /etc/passwd \
    && echo "${USER}:x:${UID}:" >> /etc/group \
    && echo "${USER} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${USER}" \
    && chmod 0440 "/etc/sudoers.d/${USER}"

# * Replace apt urls
# ? Change to tku
RUN sed -i 's@archive.ubuntu.com@ftp.tku.edu.tw@g' /etc/apt/sources.list
# ? Change to Taiwan
# RUN sed -i 's@archive.ubuntu.com@tw.archive.ubuntu.com@g' /etc/apt/sources.list

# * Time zone
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/"${TZ}" /etc/localtime && echo "${TZ}" > /etc/timezone

# * Copy custom configuration
# ? Requires docker version >= 17.09
COPY --chmod=0775 ./${ENTRYPOINT_FILE} /entrypoint.sh
COPY --chown="${USER}":"${GROUP}" --chmod=0775 config config
# ? docker version < 17.09
# COPY ./${ENTRYPOINT_FILE} /entrypoint.sh
# COPY config config
# RUN sudo chmod 0775 /entrypoint.sh && \
    # sudo chown -R "${USER}":"${GROUP}" config \
    # && sudo chmod -R 0775 config

############################### INSTALL #######################################

# * Install packages
RUN apt update \
    && apt install -y --no-install-recommends \
        sudo \
        vim \
        git \
        htop \
        wget \
        curl \
        psmisc \
        # * Shell
        tmux \
        terminator \
        # * base tools
        python3-pip \
        python3-dev \
        python3-setuptools \
        # * tools
        libgl1 \
        cmake \
        build-essential \
        software-properties-common \
        apt-transport-https \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt install -y --no-install-recommends \
        python3.9 \
        python3.9-dev \
        python3.9-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 3 \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

    #https://bobbyhadz.com/blog/python-importerror-cannot-import-name-html5lib-from-pip-vendor
    # # && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    # && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 \
    # && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 3 \
    # # && apt update \
    # # && apt install -y --no-install-recommends \
    # #     # * base tools
    # #     python3-pip \
    # #     python3-dev \
    # #     python3-setuptools \
    # #     # python3.10-distutils \
    # #     build-essential \
    # && apt clean \
    # && rm -rf /var/lib/apt/lists/*

RUN ./config/pip/pip_setup.sh

# PyTorch v1.10.1 CUDA 11.1
# https://pytorch.org/get-started/previous-versions/
RUN pip3 install --upgrade --force-reinstall pip && \
    pip3 install \
    torch==1.10.1+cu111 \
    torchvision==0.11.2+cu111 \
    torchaudio==0.10.1 \
    -f  https://download.pytorch.org/whl/cu111/torch_stable.html


# * Install ROS noetic
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > \
    /etc/apt/sources.list.d/ros-latest.list \
    && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' \
    --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && curl -sSL \
    'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' \
    | sudo apt-key add - \
    && apt update && apt install -y --no-install-recommends ros-noetic-desktop-full \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO noetic

# RUN apt install python3-apt \ 
#     && apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE \
#     && add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
RUN mkdir -p /etc/apt/keyrings \
    && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null \
    && echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
        tee /etc/apt/sources.list.d/librealsense.list \
    && apt-get update

RUN apt update \
    && apt install -y --no-install-recommends \
        # * ROS depend tools
        ros-$ROS_DISTRO-ddynamic-reconfigure \
        ros-$ROS_DISTRO-rgbd-launch \
        ros-$ROS_DISTRO-catkin \
        python3-catkin-tools \ 
        python3-rospkg \
        # # * Install librealsense
        librealsense2-utils \
        librealsense2-dkms \
        # * Install realsense sdk
        ros-$ROS_DISTRO-realsense2-camera \
        python3-netifaces \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >>/home/$USER/.bashrc

WORKDIR /

############################## USER CONFIG ####################################
# * Switch user to ${USER}
USER ${USER}

RUN ./config/shell/bash_setup.sh "${USER}" "${GROUP}" \
    && ./config/shell/terminator/terminator_setup.sh "${USER}" "${GROUP}" \
    && ./config/shell/tmux/tmux_setup.sh "${USER}" "${GROUP}" \
    && sudo rm -rf /config

# * Install realsense-ros 
# https://blog.csdn.net/z1872385/article/details/130578113
# * Create work folder
RUN sudo mkdir -p /home/"${USER}"/realsense_ros/src
WORKDIR /home/"${USER}"/realsense_ros/src
RUN sudo git clone https://github.com/IntelRealSense/realsense-ros.git
WORKDIR /home/"${USER}"/realsense_ros/src/realsense-ros/
#RUN sudo git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1`
RUN sudo git checkout ros1-legacy
WORKDIR /home/"${USER}"/realsense_ros/src/
# RUN sudo /opt/ros/noetic/bin/catkin_init_workspace
RUN sudo /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin_init_workspace'

WORKDIR /home/"${USER}"/realsense_ros/
#RUN sudo /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin_make clean'
RUN sudo /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin_make clean; catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release; catkin_make install'
RUN sudo echo "source /home/"${USER}"/realsense_ros/devel/setup.bash" >> ~/.bashrc

# # * Install graspnet-baseline
# WORKDIR /home/"${USER}"
# RUN sudo git clone https://github.com/uptopia/graspnet-baseline.git
# # WORKDIR /home/"${USER}"/graspnet-baseline/pointnet2
# # RUN sudo python3 setup.py install

# # WORKDIR /home/"${USER}"/graspnet-baseline/knn
# # RUN sudo python3 setup.py install

# # * Install graspnetAPI
# WORKDIR /home/"${USER}"
# RUN sudo git clone https://github.com/graspnet/graspnetAPI.git
# # WORKDIR /home/"${USER}"/graspnetAPI
# # RUN sudo pip3 install .

# * Create work folder
RUN sudo mkdir -p /home/"${USER}"/work
WORKDIR /home/"${USER}"/work

# # Change python3 version back, else roscore will report error
# # ModuleNotFoundError: No module named 'netifaces'
# #https://blog.csdn.net/watt1208/article/details/127391701
# sudo rm /usr/bin/python3
# sudo ln -sf /usr/bin/python3.8 /usr/bin/python3


# * Make SSH available
EXPOSE 22

#ENTRYPOINT [ "/entrypoint.sh", "terminator" ]
# ENTRYPOINT [ "/entrypoint.sh", "tmux" ]
ENTRYPOINT [ "/entrypoint.sh", "bash" ]
# ENTRYPOINT [ "/entrypoint.sh" ]
